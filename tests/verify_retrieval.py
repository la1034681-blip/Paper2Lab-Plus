"""「搜」文献依据检索与引证核验验证（阶段五 · M2）——离线、零成本。

关注点:
1. 检索是**字面检索**（同义词扩展 + 取值命中），不对外说成语义/RAG；
2. 检索不到就说检索不到，**不拿不相关段落充数**；
3. 出处核验与 `selfcheck` 的「证据自证」是**两个维度**，本模块不重复它的工作；
4. 外部检索两条路：`external_search()`（模型钩子，默认关）与 `literature_search()`
   （真实多源查询，4A）。本项只看它们**都不把外部线索当结论**，
   所以显式关掉真实检索、避免本项结果受外网影响 —— 真实检索由
   `verify_literature.py` 专测（含离线桩与真网加测）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ["PAPER2LAB_LIT"] = "off"

from paper2lab.models import (Category, Confidence, Evidence, Finding, PaperInfo,
                              Risk, Status)
from paper2lab.retrieval import (SEARCH_MODE, external_search, search_paper,
                                 verify_citations)

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


def make_paper():
    p = PaperInfo(title="Demo")
    p.page_texts = [
        (1, "Abstract. We train a ResNet on CIFAR-10. The learning rate is 1e-4 "
            "and we use cosine annealing."),
        (3, "4. Training Setup. We train for 100 epochs with batch size 256. "
            "The learning rate is 1e-4 with a warmup of 1000 steps."),
        (5, "Results show 94.3% accuracy on the test set."),
    ]
    p.page_count = 5
    return p


def make_finding(key="learning_rate", disp="Learning Rate",
                 pv="1e-4", cv="0.001", status=Status.INCONSISTENT,
                 conf=Confidence.CONFIRMED, snippet=None):
    return Finding(
        param_key=key, display_name=disp, category=Category.PARAM, status=status,
        confidence=conf, risk=Risk.HIGH,
        paper=Evidence(source="paper", location="Page 3 · 4. Training Setup",
                       snippet=snippet if snippet is not None else f"learning rate is {pv}",
                       raw_value=pv),
        code=Evidence(source="code", location="train.py:11",
                      snippet=f"lr = {cv}", raw_value=cv),
    )


class _Audit:
    def __init__(self, paper, findings):
        self.paper, self.findings = paper, findings


# ═══════════════════════════════════════ 1. 论文内部检索

sec("1. search_paper —— 字面检索（同义词扩展 + 取值命中）")

paper = make_paper()
hits = search_paper(paper, "learning_rate", top_k=3, value="1e-4")
check(len(hits) == 2, "命中 2 页（第 5 页与参数无关，不返回）")
check(hits[0]["score"] >= hits[1]["score"], "按相关度降序")
check(all(h["page"] in (1, 3) for h in hits), "命中页码正确")
check(all(h["snippet"] for h in hits), "每条都带原文片段（不是空壳）")
check(all(isinstance(h["score"], int) and h["score"] > 0 for h in hits),
      "score 都是正整数（score=0 的不返回）")

check(search_paper(paper, "learning_rate", top_k=1) ==
      search_paper(paper, "learning_rate", top_k=1), "同输入同结果（可复现）")
check(len(search_paper(paper, "learning_rate", top_k=1)) == 1, "top_k 生效")

# 同义词扩展: 用别名检索也能命中
check(len(search_paper(paper, "lr", top_k=3)) > 0, "用同义词 lr 也能命中（复用 synonyms 词表）")

# 检索不到 -> 空，不充数
check(search_paper(paper, "num_clusters", top_k=3) == [],
      "论文里没有的参数 -> 返回空（不拿不相关段落充数）")
check(search_paper(None, "learning_rate") == [], "paper 为 None -> 空")
p_empty = PaperInfo(title="空")
check(search_paper(p_empty, "learning_rate") == [], "page_texts 为空 -> 空列表")
p_irrelevant = PaperInfo(title="无关")
p_irrelevant.page_texts = [(1, "This paper studies protein folding kinetics.")]
check(search_paper(p_irrelevant, "learning_rate") == [],
      "正文完全不提该参数 -> 空（不硬凑一段）")

# 取值命中权重更高
h_val = search_paper(paper, ["banana"], top_k=3, value="1e-4")
check(len(h_val) == 2 and all(h["score"] >= 5 for h in h_val),
      "只给取值也能命中，且权重 ≥5（取值命中比词命中更有用）")

# ═══════════════════════════════════════ 2. 引证核验

sec("2. verify_citations —— 出处核验与覆盖统计")

f1 = make_finding()
f2 = make_finding(key="optimizer", disp="Optimizer", pv="", cv="Adam",
                  status=Status.MISSING_IN_PAPER, conf=Confidence.INFERRED)
f2.paper = None      # 没有论文侧证据 -> 只能靠代码出处
rep = verify_citations(_Audit(paper, [f1, f2]))

cov = rep.coverage
check(cov["total"] == 2, "结论总数 2")
check(cov["cited"] == 2, "两条都有可定位出处（论文页码 / 代码 file:line）")
check(abs(cov["cited_ratio"] - 1.0) < 1e-9, "覆盖率 100%")
check(cov["mismatched"] == 0, "没有「引文与原文不符」的情况")
check(rep.items[0]["primary_source"].startswith("Page 3"),
      "一级出处优先取论文页码")
check(rep.items[1]["primary_source"] == "train.py:11",
      "没有论文证据时退回代码 file:line")
check(rep.items[1]["primary_kind"] == "code", "标明出处类型")
check(rep.items[0]["n_citations"] >= 1, "第 1 条检索到了相关段落")

# 无出处的结论 -> 如实记录，不编一个出处
f3 = make_finding(key="momentum", disp="Momentum", status=Status.NOT_FIXED)
f3.paper = None
f3.code = None
rep3 = verify_citations(_Audit(paper, [f3]))
check(rep3.coverage["cited"] == 0, "两侧都没有出处 -> cited=0")
check(rep3.coverage["cited_ratio"] == 0.0, "覆盖率 0")
check(any("没有可定位出处" in n or "0" in n for n in rep3.notes),
      "notes 里如实说明有多少条没有出处")
check(rep3.items[0]["primary_source"] == "", "不替它编一个出处")

# 引文与原文不符 -> mismatched（复用 selfcheck 的口径）
f4 = make_finding(snippet="we set the learning rate to 1e-5")   # 引文里没有 raw_value 1e-4
rep4 = verify_citations(_Audit(paper, [f4]))
check(rep4.coverage["mismatched"] == 1, "引文里找不到该取值 -> 计入 mismatched")
check("找不到该取值" in rep4.items[0]["mismatch_note"], "并写明原因（可核验）")

# 仅 AI 推断且无原文取值 -> inferred_only
f5 = make_finding(key="threshold", disp="Threshold", pv="", status=Status.MISSING_IN_PAPER,
                  conf=Confidence.INFERRED)
rep5 = verify_citations(_Audit(paper, [f5]))
check(rep5.coverage["inferred_only"] == 1, "仅 AI 推断且无原文取值 -> 计入 inferred_only")

# 空 audit 不炸
rep0 = verify_citations(_Audit(paper, []))
check(rep0.coverage["total"] == 0 and rep0.items == [], "没有结论 -> 空结果，不炸")

# ═══════════════════════════════════════ 3. 外部检索（默认关闭）

sec("3. external_search —— 模型钩子默认关闭，开启也要标未核验")

off = external_search("任意意图")
check(off["enabled"] is False, "默认关闭")
check(off["items"] == [], "关闭时不给任何线索")
check("未启用" in off["note"], "如实说明未启用（不假装搜过）")

on_nokey = external_search("意图", enabled=True)
check(on_nokey["enabled"] is True and on_nokey["items"] == [],
      "开启但无模型 API -> 空结果")
check("未配置" in on_nokey["note"], "如实说明未配置（不假装搜过）")

fake = {"items": [{"title": "Deep Residual Learning", "source": "CVPR 2016",
                   "why": "同一残差结构", "confidence": "high"},
                  {"title": "B", "source": "S2", "why": "", "confidence": "low"},
                  {"title": "C", "source": "S3", "why": "", "confidence": "low"},
                  {"title": "D", "source": "S4", "why": "", "confidence": "low"},
                  {"title": "E", "source": "S5", "why": "", "confidence": "low"}]}


def fake_chat(prompt):
    assert isinstance(prompt, str) and "检索" in prompt, "prompt 必须包含检索意图"
    return fake, "AI（search）· 测试桩"


got = external_search("意图", enabled=True, chat=fake_chat)
check(len(got["items"]) == 3, "最多返回 3 条（截断生效）")
check(all(it["unverified"] is True for it in got["items"]),
      "**每条都标 unverified=True**（外部线索不得当结论用）")
check(got["items"][0]["title"] == "Deep Residual Learning", "线索字段透传正确")
check("未核验" in got["note"], "note 里说明这些线索未核验")


def bad_chat(prompt):
    raise RuntimeError("网络炸了")


bad = external_search("意图", enabled=True, chat=bad_chat)
check(bad["items"] == [] and "失败" in bad["note"], "调用失败 -> 空结果 + 如实记原因（不炸主流程）")


def none_chat(prompt):
    return None, "AI 未返回"


non = external_search("意图", enabled=True, chat=none_chat)
check(non["items"] == [], "模型没返回 -> 空结果")
check(bool(non["note"]), "仍给出说明")

# 脏数据：缺 title 的项被丢掉
dirty = {"items": [{"source": "X"}, {"title": "OK", "source": "Y"}]}
dirt = external_search("意图", enabled=True, chat=lambda p: (dirty, ""))
check(len(dirt["items"]) == 1 and dirt["items"][0]["title"] == "OK",
      "缺 title 的脏项被丢掉（不给下游塞坏数据）")

# ═══════════════════════════════════════ 4. 契约与自述

sec("4. 契约与自述")

check("字面检索" in SEARCH_MODE, "SEARCH_MODE 自述为字面检索")
check("非向量" in SEARCH_MODE, "明确声明不是向量/语义检索（不吹成 RAG）")
md = rep.to_markdown()
check("「搜」" in md, "markdown 有标题")
check(SEARCH_MODE in md, "markdown 里写明检索方式")
check(isinstance(rep.to_dict(), dict) and "coverage" in rep.to_dict(), "可序列化")
check(bool(rep.notes), "结论里带说明（不是空 notes）")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段五「搜」验证: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_retrieval.py)")
print("=" * 62)
