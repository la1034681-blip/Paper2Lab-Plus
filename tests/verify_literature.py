"""「搜」外部文献检索验证（阶段四 · 4A）—— 离线为主，联网为可选加测。

关注点:
1. **六源解析器能吃真实响应格式**（用真实响应裁剪的 fixture，不是想象出来的字段）；
2. **去重规则**：DOI 优先、同标题的「有 DOI 版」与「无 DOI 版」要合并成一条；
3. **不硬凑**：相关度不达标的一律丢，宁可返回空；
4. **不编文献 / 不冒充核验**：`verified` 恒为 False，顶层 `unverified` 恒为 True；
5. **一个源挂了不拖垮全局**：失败原因逐源记录，全挂也如实说"都没连上"；
6. **缓存与开关**：查过的不再打人家的 API；关掉就如实说未启用。

联网段（第 7 节）在无外网时自动跳过并说明，不判红 —— 测试不该因为没网就失败，
但**也不能假装跑过**（跳过要打印出来）。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 本文件要真测"开着"的行为；全套体检默认把外部检索关掉（离线可跑），
# 所以这里**显式打开**，而不是 setdefault。
os.environ["PAPER2LAB_LIT"] = "on"

from paper2lab import literature as L  # noqa: E402

CHECKS = 0
FAILURES = []


def check(ok, label):
    global CHECKS
    CHECKS += 1
    assert isinstance(ok, bool), "[%s] ok 不是布尔值: %r" % (label, ok)
    assert isinstance(label, str)
    if not ok:
        FAILURES.append(label)
        print("  *x* " + label)
    else:
        print("  [ok] " + label)


def sec(n):
    print("\n" + "─" * 62)
    print(n)
    print("─" * 62)


# ═══════════════════════════════════════ 1. 工具函数

sec("1. 工具函数 —— 分词 / DOI 归一 / 标题指纹 / 摘要还原")

t = L._terms("Why Warmup the Learning Rate?")
check("warmup" in t and "learning" in t and "rate" in t, "英文实词被切出来")
check("the" not in t and "why" not in t, "停用词被剔除（命中它们不代表相关）")
t2 = L._terms("深度学习学习率预热")
check("学习" in t2 and "预热" in t2, "中文按 2 字滑窗切分（不引分词依赖）")
check(L._terms("") == set(), "空串 -> 空集合（不是全集）")
check(L._hit_ratio(set(), "任意文本") == 0.0,
      "查询词为空时命中率为 0（不能因为'没词可比'就算全中）")

check(L._norm_doi("https://doi.org/10.1000/ABC") == "10.1000/abc", "DOI 去前缀转小写")
check(L._norm_doi("doi: 10.1000/abc") == "10.1000/abc", "去掉 doi: 前缀")
check(L._norm_doi("10.1051/itmconf/20268403033/pdf") == "10.1051/itmconf/20268403033",
      "剥掉 /pdf 后缀（实测 OpenAlex 会把全文链接当 DOI 给）")
check(L._norm_doi("10.1000/abc.pdf") == "10.1000/abc", "剥掉 .pdf 后缀")
check(L._norm_doi(None) == "", "None -> 空串，不抛异常")

check(L._norm_title("A  Closer Look: Deep-Learning!") == "acloserlooldeeplearning"
      or L._norm_title("A  Closer Look: Deep-Learning!") == "acloserlookdeeplearning",
      "标题指纹抹平标点与大小写")
check(L._norm_title("A B") == L._norm_title("a  b!"), "同一标题的不同写法指纹相同")

check(L._clean("<jats:p>Hello &amp; world</jats:p>") == "Hello & world",
      "摘要里的标签被去掉、实体被反转义")
ab = L._openalex_abstract({"world": [1], "hello": [0], "again": [2]})
check(ab == "hello world again", "OpenAlex 倒排索引按位置还原成句子（乱序输入也对）")
check(L._openalex_abstract(None) == "", "没有摘要时给空串，不造摘要")


# ═══════════════════════════════════════ 2. 六源解析（真实响应裁剪）

sec("2. 六个源的解析器 —— 吃真实响应格式，字段不漏")

OPENALEX = {"results": [{
    "display_name": "Why Warmup the Learning Rate? Underlying Mechanisms",
    "doi": "https://doi.org/10.52202/079017-3549",
    "publication_year": 2024, "cited_by_count": 23, "type": "proceedings-article",
    "authorships": [{"author": {"display_name": "Jane Doe"}}],
    "primary_location": {"landing_page_url": "https://example.org/a",
                         "source": {"display_name": "NeurIPS"}},
    "open_access": {"oa_url": "https://example.org/a.pdf"},
    "abstract_inverted_index": {"warmup": [0], "helps": [1], "training": [2]},
}]}

CROSSREF = {"message": {"items": [{
    "title": ["Why Warmup the Learning Rate? Underlying Mechanisms"],
    "DOI": "10.52202/079017-3549",
    "issued": {"date-parts": [[2024]]},
    "container-title": ["NeurIPS"],
    "is-referenced-by-count": 23,
    "type": "proceedings-article",
    "URL": "https://example.org/a",
    "author": [{"given": "Jane", "family": "Doe"}],
    "abstract": "<jats:p>Warmup helps training</jats:p>",
}]}}

EPMC = {"resultList": {"result": [{
    "title": "A Study of Protein Folding Pathways",
    "doi": "10.1000/epmc.1", "pubYear": "2022", "journalTitle": "Nature",
    "authorString": "Smith J, Lee K", "citedByCount": 100, "source": "MED",
    "id": "12345", "pubType": "research-article",
    "abstractText": "We study protein folding kinetics.",
}]}}

ARXIV = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-02T00:00:00Z</published>
    <title>Why Warmup the Learning Rate? Underlying Mechanisms</title>
    <summary>Warmup helps training a lot.</summary>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>"""

DATACITE = {"data": [{
    "id": "10.5281/zenodo.1",
    "attributes": {"doi": "10.5281/zenodo.1",
                   "titles": [{"title": "Dataset for learning rate warmup study"}],
                   "creators": [{"name": "Doe, Jane"}],
                   "publicationYear": 2023, "publisher": "Zenodo",
                   "types": {"resourceTypeGeneral": "Dataset"},
                   "url": "https://zenodo.org/record/1",
                   "descriptions": [{"description": "Warmup ablation data."}]},
}]}

DOAJ = {"results": []}


def fake_get(url, timeout):
    """按域名分发 fixture —— 顺带验证"每个源真的只看自己那段"。"""
    if "openalex" in url:
        return json.dumps(OPENALEX).encode()
    if "crossref" in url:
        return json.dumps(CROSSREF).encode()
    if "europepmc" in url:
        return json.dumps(EPMC).encode()
    if "arxiv" in url:
        return ARXIV.encode()
    if "datacite" in url:
        return json.dumps(DATACITE).encode()
    if "doaj" in url:
        return json.dumps(DOAJ).encode()
    raise AssertionError("出现未预期的 URL: " + url)


with mock.patch.object(L, "_get", fake_get):
    oa = L._fetch_openalex("q", 5, 3)
    cr = L._fetch_crossref("q", 5, 3)
    ep = L._fetch_europepmc("q", 5, 3)
    ax = L._fetch_arxiv("q", 5, 3)
    dc = L._fetch_datacite("q", 5, 3)
    dj = L._fetch_doaj("q", 5, 3)

check(len(oa) == 1 and oa[0]["title"].startswith("Why Warmup"), "OpenAlex 解析出标题")
check(oa[0]["doi"] == "10.52202/079017-3549", "OpenAlex DOI 被归一")
check(oa[0]["venue"] == "NeurIPS" and oa[0]["year"] == 2024, "OpenAlex 期刊与年份对")
check("warmup helps training" in oa[0]["abstract"], "OpenAlex 摘要从倒排索引还原")

check(len(cr) == 1 and cr[0]["authors"] == ["Jane Doe"], "Crossref 作者名拼装正确")
check(cr[0]["year"] == 2024 and cr[0]["citations"] == 23, "Crossref 年份与被引数对")
check("<" not in cr[0]["abstract"], "Crossref 摘要里的 JATS 标签被清掉")

check(len(ep) == 1 and ep[0]["venue"] == "Nature", "Europe PMC 期刊名对")
check(ep[0]["url"] == "https://doi.org/10.1000/epmc.1", "Europe PMC 优先给 DOI 链接")

check(len(ax) == 1 and ax[0]["year"] == 2024, "arXiv Atom XML 解析出年份")
check(ax[0]["citations"] is None, "arXiv 不给被引数 -> 留 None（不猜成 0）")
check(ax[0]["doi"] == "", "arXiv 没给 DOI -> 留空（不拼一个假的）")

check(len(dc) == 1 and dc[0]["kind"] == "Dataset", "DataCite 解析出资源类型")
check(len(dj) == 0, "DOAJ 空结果 -> 空列表，不炸")


# ═══════════════════════════════════════ 3. 合并与重排

sec("3. 合并与重排 —— DOI 去重、无 DOI 并入有 DOI、相关度门槛")

with mock.patch.object(L, "_get", fake_get):
    r = L.search("warmup learning rate", limit=8, use_cache=False)

check(r["reached"] == 6, "6 个源都返回了（reached=6）")
titles = [it["title"] for it in r["items"]]
why = [it for it in r["items"] if it["title"].startswith("Why Warmup")]
check(len(why) == 1, "同一篇的 OpenAlex/Crossref/arXiv 三条被合成一条")
check(sorted(why[0]["found_in"]) == ["arxiv", "crossref", "openalex"],
      "合并后 found_in 记下了三个源（跨源互证）")
check(why[0]["doi"] == "10.52202/079017-3549",
      "无 DOI 的 arXiv 版没有顶掉有 DOI 的那条（保住可点的 DOI）")
check(any(it["title"].startswith("A Study of Protein Folding") for it in r["items"]) is False,
      "与查询无关的文献被门槛丢掉（不拿不相关的充数）")
check(all(it["verified"] is False for it in r["items"]),
      "**每条 verified 都是 False**（文献真实存在 ≠ 支持结论）")
check(all(it["exists"] is True for it in r["items"]), "每条都标 exists=True（可核）")
check(all(bool(it["why"]) for it in r["items"]), "每条都给出打分理由（可解释）")
check(r["items"] == sorted(r["items"], key=lambda x: -x["score"]), "按分数降序")
check("非向量" in r["mode"], "自述写明不是向量/语义检索（不吹成 RAG）")

m = L._merge(
    [{"title": "Same Title", "doi": "", "source": "arxiv", "abstract": ""},
     {"title": "same title!", "doi": "", "source": "doaj", "abstract": ""}],
    L._terms("same title"), None)
check(len(m) == 1, "两条无 DOI 的同标题记录也合并（指纹抹平标点）")

m2 = L._merge(
    [{"title": "Warmup study", "doi": "10.1/a", "source": "x", "abstract": "", "year": 2010}],
    L._terms("warmup"), 2020)
check(m2 == [], "year_from 生效：早于下限的丢掉")

noise = L._merge(
    [{"title": "Table 4: Proposed model performance tuning with different warmup",
      "doi": "10.1/t4", "source": "crossref", "abstract": ""},
     {"title": "Supplemental Information 1: Combinations of warmup steps",
      "doi": "10.1/s1", "source": "crossref", "abstract": ""},
     {"title": "A real paper about warmup", "doi": "10.1/real",
      "source": "crossref", "abstract": ""}],
    L._terms("warmup"), None)
check(len(noise) == 1 and noise[0].title == "A real paper about warmup",
      "「Table N / Supplemental」这类正文零件被当假文献滤掉（实测 Crossref 真有这种 DOI）")

check(L._hit_ratio({"rate"}, "generate normal rate") == 1.0,
      "英文按词边界比：rate 不会因为 generate 里含 rate 就算命中")
check(L._hit_ratio({"rate"}, "generated accuracy") == 0.0,
      "只含 rate 子串的词不再虚报命中")


# ═══════════════════════════════════════ 4. 失败与降级

sec("4. 失败与降级 —— 一个源挂了不拖垮全局，全挂也如实说")

def boom(url, timeout):
    raise OSError("network down")


with mock.patch.object(L, "_get", boom):
    dead = L.search("warmup learning rate", limit=5, use_cache=False)
check(dead["ok"] is True, "全源失败时 ok=True（流程没崩），不是抛异常")
check(dead["reached"] == 0 and dead["items"] == [], "全源失败 -> 0 条结果")
check("都没连上" in dead["note"], "如实写'都没连上'（不假装搜过）")
check(all("network down" in (v.get("error") or "") for v in dead["providers"].values()),
      "每个源都记了失败原因（可排障）")

called = {"n": 0}


def flaky(url, timeout):
    called["n"] += 1
    if "openalex" in url:
        raise OSError("only this one is down")
    return fake_get(url, timeout)


with mock.patch.object(L, "_get", flaky):
    half = L.search("warmup learning rate", limit=5, use_cache=False)
check(half["reached"] == 5, "只挂一个源 -> 其余 5 个照常返回")
check(half["providers"]["openalex"]["ok"] is False, "挂掉的那个源被如实标 ok=False")
check(len(half["items"]) > 0, "部分失败不影响结果（有就拿）")


def slow(url, timeout):
    time.sleep(6)
    return fake_get(url, timeout)


t0 = time.time()
with mock.patch.object(L, "_get", slow):
    timed = L.search("warmup learning rate", limit=3, use_cache=False,
                     timeout=0.3, budget=0.8)
spent = time.time() - t0
check(spent < 5, f"整体预算生效：卡住的源没有拖死调用（{spent:.1f}s < 5s）")
check(timed["reached"] == 0 and "超时" in json.dumps(timed["providers"], ensure_ascii=False),
      "卡住的源被标为超时（不是当成 0 条结果）")


# ═══════════════════════════════════════ 5. 开关、缓存、空查询

sec("5. 开关 / 缓存 / 空查询")

empty = L.search("   ")
check(empty["ok"] is False and "query 为空" in empty.get("error", ""),
      "空 query -> ok=False 且不发请求")

os.environ["PAPER2LAB_LIT"] = "off"
check(L.enabled() is False, "PAPER2LAB_LIT=off -> enabled() False")
off = L.search("warmup")
check(off["items"] == [] and "关闭" in off["note"], "关掉后如实说未启用，不假装搜过")
os.environ["PAPER2LAB_LIT"] = "on"
check(L.enabled() is True, "改回 on 即恢复")

with tempfile.TemporaryDirectory() as td:
    with mock.patch.object(L, "CACHE_DIR", Path(td)):
        with mock.patch.object(L, "_get", mock.MagicMock(side_effect=fake_get)) as g1:
            c1 = L.search("warmup learning rate", limit=5)
            n1 = g1.call_count
        with mock.patch.object(L, "_get", mock.MagicMock(side_effect=fake_get)) as g2:
            c2 = L.search("warmup learning rate", limit=5)
            n2 = g2.call_count
    check(n1 == 6, "第一次查真的打了 6 个源")
    check(n2 == 0 and c2["cached"] is True,
          "第二次命中缓存、一个请求都没发（不重复打人家的 API）")
    check([x["title"] for x in c1["items"]] == [x["title"] for x in c2["items"]],
          "缓存内容与首次结果一致")


# ═══════════════════════════════════════ 6. 接进审计链

sec("6. 接进审计链 —— for_findings / verify_citations / MCP 工具")

from types import SimpleNamespace as NS  # noqa: E402

from paper2lab.retrieval import literature_search, verify_citations  # noqa: E402


def _f(key, disp, st):
    return NS(param_key=key, display_name=disp, status=NS(value=st),
              paper=None, code=None, confidence=NS(value="explicit"))


FINDINGS = [_f("learning_rate", "Learning Rate", "inconsistent"),
            _f("batch_size", "Batch Size", "consistent"),
            _f("warmup_steps", "Warmup Steps", "missing_in_code")]
PAPER = NS(title="Why Warmup the Learning Rate", page_texts=[(1, "learning rate 1e-4")])

with mock.patch.object(L, "_get", fake_get):
    ff = L.for_findings(FINDINGS, PAPER.title, limit=3)
check(len(ff["queries"]) >= 1, "由结论自动生成了检索词")
check(ff["queries"][0].startswith("Learning Rate"),
      "第一个查询词以**可疑项的参数名**开头（对不上的最需要依据）")
check(all(q[0].isupper() or q[0].isdigit() for q in ff["queries"]),
      "每个查询词都从参数名起头（不是拿内部措辞去搜学术库）")
check(bool(ff["by_param"]), "带回参数 -> 查询词的对照表")
check(all(it["exists"] is True and it["verified"] is False for it in ff["items"]),
      "审计侧拿到的文献同样 exists=True / verified=False")

with mock.patch.object(L, "_get", fake_get):
    ls = literature_search(FINDINGS, paper_title=PAPER.title, limit=3)
check(ls["enabled"] is True, "literature_search 如实报告已启用")
check(ls["unverified"] is True, "顶层 unverified=True（下游只看一个键也不会漏）")
check(sorted(ls["sources"]) == sorted(L.PROVIDERS), "带回本次查过的源清单")

with mock.patch.object(L, "_get", fake_get):
    rep = verify_citations(NS(findings=FINDINGS, paper=PAPER))
md = rep.to_markdown()
check("外部文献候选" in md, "报告里有外部文献段")
check("没有一条被机器判为" in md, "报告里写明'没有一条被判为支持结论'")
check(any("未核验" in n for n in rep.notes), "notes 里同样标未核验")

from paper2lab import mcp_server as M  # noqa: E402

check("literature_search" in M.TOOL_NAMES, "对外工具清单里有 literature_search")
check(M.TOOL_STATIONS["literature_search"] == M.S_VERIFY, "它挂在③验证站")
check(len(M.TOOL_NAMES) == len(set(M.TOOL_NAMES)), "工具名无重复")
check(M.SERVER_VERSION.startswith("1."), f"服务版本号已更新（{M.SERVER_VERSION}）")

with mock.patch.object(L, "_get", fake_get):
    tr = M.tool_literature_search("warmup learning rate", limit=3)
check(tr["ok"] is True and tr["n"] >= 1, "MCP 工具能直接查（不需要论文/代码）")
check(all(it["verified"] is False for it in tr["items"]), "MCP 返回同样 verified=False")
check("需人读确认" in tr["boundary"], "MCP 返回自带边界声明")
check(M.tool_literature_search("  ")["ok"] is False, "MCP 空 query -> 结构化失败")

with mock.patch.object(L, "_get", fake_get):
    sub = M.tool_literature_search("warmup", limit=2, providers="arxiv,doaj")
check(sub["ok"] is True and sub["of"] == 2, "providers 参数只查指定源")


# ═══════════════════════════════════════ 7. 联网加测（无网关自动跳过）

sec("7. 联网加测 —— 真打一次 6 个公开索引（无外网则跳过，不判红）")

net_ok = os.environ.get("PAPER2LAB_LIT_TEST_NET", "1") != "0"
if not net_ok:
    print("  [--] PAPER2LAB_LIT_TEST_NET=0，联网段按要求跳过")
else:
    t0 = time.time()
    # 先清一遍这台机器上的检索缓存，否则"第二次走缓存"可能命中的是以前留下的
    L.clear_cache()
    live = L.search("learning rate warmup transformer training", limit=5, use_cache=True)
    dt = time.time() - t0
    print(f"  实测：{live.get('reached')}/6 源返回，{len(live.get('items') or [])} 条结果，{dt:.1f}s")
    if live.get("reached", 0) == 0:
        print("  [--] 本机当前连不上外网 —— 联网段跳过（不计入通过/失败）")
    else:
        check(live["reached"] >= 3, f"真网下至少 3 个源可用（实际 {live['reached']}）")
        check(len(live["items"]) > 0, "真网下查得到相关文献")
        check(all(bool(it["title"]) for it in live["items"]), "真网结果每条都有标题")
        check(all(bool(it["doi"] or it["url"]) for it in live["items"]),
              "真网结果每条都有 DOI 或落地页（可点开核）")
        check(live["elapsed_s"] < 12, f"耗时在预算内（{live['elapsed_s']}s）")
        again = L.search("learning rate warmup transformer training", limit=5)
        check(again.get("cached") is True, "同一查询第二次走缓存")


# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段四「4A 免费多源文献检索」验证: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_literature.py)")
print("=" * 62)
