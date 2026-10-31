"""「搜」文献依据检索与引证核验（阶段五 · M2 闭环扩展）。

**为什么需要这一步**：审计报告里每条结论都写着一个出处（`Page 3 · Training Setup`
或 `train.py:11`），但用户真正要确认的时候还得自己翻回去找。这一步把它变成
**机器先找好、人只看结论**：为每条结论从论文原文里检索出最相关的段落贴在旁边，
并统计"这条结论有没有出处、出处够不够具体"。

**三件事**：

1. **论文内部检索（本模块主力，全程本地、零成本）**
   按参数同义词 + 实际取值在论文正文里检索，返回最相关的段落与页码。
   词表复用 `synonyms.PARAM_SYNONYMS`，不另立一份，否则检索口径会与对齐引擎漂移。
2. **引证核验（清单式）**
   逐条给出：出处类型（论文页 / 代码 file:line）、出处是否可定位、是否属于
   "只有 AI 推断、没有原文出处"。**与 `selfcheck` 的「证据自证」是两个维度**：
   selfcheck 查的是"证据自己是否自洽"（snippet 里真的含这个值），
   这里查的是"这条结论的出处够不够、能不能被独立复查"，两者互补不重复。
3. **外部文献检索（默认走真实多源检索，见 `literature.py`）**
   原来这里是个"让模型凭记忆写几篇文献"的钩子 —— 那本质是让模型背课文，
   写出来的东西无法证实存在，是最该防的幻觉源。现在默认走 `literature.search()`：
   并发查 6 个公开学术索引（OpenAlex / Crossref / Europe PMC / arXiv / DataCite /
   DOAJ），DOI 去重后按相关度返回，**每条都能点开核**。
   模型钩子 `external_search()` 保留为可选补全（`PAPER2LAB_LIT=off` 时的老路径），
   两者的返回项一律 `unverified=True`。
   **边界**：文献真实存在 ≠ 它支持这条结论。前者机器能证，后者只能人读——
   所以外部线索永远不混进确定性结论，也不参与任何判定。

**两条边界**：
1. 检索是**字面检索**（同义词扩展 + 取值命中），不是语义检索。
   项目决策里"向量检索"要等评测数据再定（总纲 §14），所以这里如实标为
   字面检索，不吹成 RAG。
2. 检索不到就报"检索不到"，**不拿不相关的段落充数**（score=0 的段落一律不返回）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .synonyms import PARAM_SYNONYMS, canonical_key

__all__ = [
    "search_paper", "cite_findings", "verify_citations",
    "external_search", "literature_search", "CitationReport", "SEARCH_MODE",
]

# 检索方式的自述（写进结果里，避免对外被说成"语义检索"）
SEARCH_MODE = "字面检索（同义词扩展 + 取值命中，非向量/语义检索）"

_SENT_SPLIT = re.compile(r"(?<=[.。!?！？;；\n])")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).lower()


def _terms_for(key: str, value: Optional[str] = None) -> list:
    """某个参数的检索词：规范键 + 同义词表 + 实际取值。"""
    out: list = []
    if key:
        out.append(str(key))
        entry = PARAM_SYNONYMS.get(key)
        if entry:
            # PARAM_SYNONYMS 的结构是 (展示名, [同义词...])，两处都不能漏
            for part in entry:
                if isinstance(part, str):
                    out.append(part)
                elif isinstance(part, (list, tuple, set)):
                    out.extend([str(p) for p in part])
    if value:
        out.append(str(value))
    seen: set = set()
    uniq: list = []
    for t in out:
        t = _norm(t).strip()
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _snippet_around(text: str, needle: str, width: int = 180) -> str:
    """从原文里截出命中词附近的句子（尽量给完整句，便于人直接读）。"""
    low = _norm(text)
    i = low.find(needle)
    if i < 0:
        return text[:width].strip()
    # 往前找句首
    start = 0
    for m in _SENT_SPLIT.finditer(text):
        if m.end() <= i:
            start = m.end()
        else:
            break
    end = min(len(text), start + width)
    frag = text[start:end].strip()
    if start > 0:
        frag = "…" + frag
    if end < len(text):
        frag = frag + "…"
    return frag


def search_paper(paper: Any, query: Any, *, top_k: int = 3,
                 value: Optional[str] = None) -> list:
    """在论文正文里检索与某个参数最相关的段落。

    `query` 可以是参数键（canonical key 或任意写法），也可以直接给一组检索词。
    返回 `[{page, snippet, score, terms}]`，按相关度降序；**score=0 的不返回**。
    """
    if paper is None:
        return []
    pages = list(getattr(paper, "page_texts", None) or [])
    if not pages:
        return []
    if isinstance(query, (list, tuple, set)):
        terms = [str(t) for t in query]
    else:
        key = canonical_key(str(query)) or str(query)
        terms = _terms_for(key, value)
    hits: list = []
    for page_no, text in pages:
        low = _norm(text)
        score = 0
        first_term = ""
        for t in terms:
            c = low.count(t)
            if c:
                score += min(c, 3)
                first_term = first_term or t
        # 实际取值命中权重更高（"Page 3 出现了 1e-4" 比"出现了 learning rate"更有用）
        if value and _norm(value) in low:
            score += 5
            first_term = _norm(value)
        if score <= 0:
            continue
        hits.append({"page": page_no, "score": score,
                     "terms": [t for t in terms if t in low][:6],
                     "snippet": _snippet_around(text, first_term)})
    hits.sort(key=lambda h: (-h["score"], h["page"]))
    return hits[:max(0, top_k)]


# ---------------------------------------------------------------- 引证核验

@dataclass
class CitationReport:
    """引证核验结果（逐条 + 汇总）。"""
    items: list = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    mode: str = SEARCH_MODE
    external: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"items": self.items, "coverage": self.coverage,
                "notes": self.notes, "mode": self.mode, "external": self.external}

    def to_markdown(self) -> str:
        c = self.coverage or {}
        md = ["## 「搜」文献依据检索与引证核验", "",
              f"> 检索方式：{self.mode}。检索不到的如实记为「未检索到」，"
              "不拿不相关段落充数。", "",
              f"- 结论总数：**{c.get('total', 0)}**",
              f"- 有可定位出处（论文页码 或 代码 file:line）：**{c.get('cited', 0)}**"
              f" / {c.get('total', 0)}（{c.get('cited_ratio', 0):.0%}）",
              f"- 仅 AI 推断、无原文出处：**{c.get('inferred_only', 0)}**",
              f"- 给了引文但与原文不符：**{c.get('mismatched', 0)}**", ""]
        if self.items:
            md += ["| 参数 | 一级出处 | 检索到的相关段落 |", "| --- | --- | --- |"]
            for it in self.items:
                sup = it.get("citations") or []
                first = (f"p.{sup[0]['page']}：{sup[0]['snippet'][:70]}…"
                         if sup else "—")
                md.append(f"| {it.get('display') or it.get('param_key')} | "
                          f"{it.get('primary_source') or '—'} | {first} |")
            md.append("")
        ext = self.external or {}
        if ext.get("items"):
            md += ["", "### 外部文献候选（真实存在，但是否支持该结论需人读确认）", "",
                   f"> 检索方式：{ext.get('mode') or '外部检索'}。"
                   f"下面 {len(ext['items'])} 条都能点开核，但**没有一条被机器判为"
                   "「支持某条结论」**，请当作候选依据使用。", "",
                   "| 年份 | 标题 | 收录库 | 链接 |", "| --- | --- | --- | --- |"]
            for it in ext["items"]:
                title = str(it.get("title") or "").replace("|", "／")[:80]
                link = it.get("doi") or it.get("url") or "—"
                if it.get("doi"):
                    link = f"[{it['doi']}](https://doi.org/{it['doi']})"
                md.append(f"| {it.get('year') or '—'} | {title} | "
                          f"{', '.join(it.get('found_in') or []) or '—'} | {link} |")
            md.append("")
        elif ext.get("enabled"):
            md += ["", f"> 外部文献：{ext.get('note') or '本次没有检索到相关文献'}"]
        if self.notes:
            md += [f"- {n}" for n in self.notes]
        return "\n".join(md)


def _primary_source(f: Any) -> tuple:
    """一条结论的"一级出处"：优先论文页，其次代码 file:line。"""
    def _loc(ev):
        return getattr(ev, "location", "") if ev is not None else ""

    paper_ev = getattr(f, "paper", None)
    code_ev = getattr(f, "code", None)
    if paper_ev is not None and _loc(paper_ev):
        return _loc(paper_ev), "paper"
    if code_ev is not None and _loc(code_ev):
        return _loc(code_ev), "code"
    return "", ""


def cite_findings(audit: Any, *, top_k: int = 2) -> CitationReport:
    """为每条结论检索论文原文里的相关段落，并核验其出处。

    `audit` 需要 `findings` 与 `paper` 两个属性（`AuditResult` 即是）。
    """
    rep = CitationReport()
    findings = list(getattr(audit, "findings", None) or [])
    paper = getattr(audit, "paper", None)
    total = cited = inferred_only = mismatched = 0
    for f in findings:
        total += 1
        src, kind = _primary_source(f)
        if src:
            cited += 1

        # --- 检索相关段落（论文侧：按参数键 + 实际取值）
        pv = getattr(getattr(f, "paper", None), "raw_value", None)
        hits = search_paper(paper, getattr(f, "param_key", ""), top_k=top_k, value=pv)

        # --- 引文是否与原文对得上（复用 selfcheck 的口径，不另写一份比较逻辑）
        mismatch_note = ""
        if getattr(f, "paper", None) is not None:
            try:
                from .selfcheck import _snippet_contains
                ev = f.paper
                if not _snippet_contains(getattr(ev, "snippet", ""),
                                         getattr(ev, "raw_value", "")):
                    mismatched += 1
                    mismatch_note = "给出的引文里找不到该取值"
            except Exception:      # noqa: BLE001  —— 核验本身失败不该拖垮主流程
                mismatch_note = ""

        # --- 是否"只有 AI 推断、无原文出处"
        conf = getattr(getattr(f, "confidence", None), "value", "")
        only_inferred = (conf == "inferred") and not pv
        if only_inferred:
            inferred_only += 1

        rep.items.append({
            "param_key": getattr(f, "param_key", ""),
            "display": getattr(f, "display_name", ""),
            "status": getattr(getattr(f, "status", None), "value", ""),
            "primary_source": src, "primary_kind": kind,
            "citations": hits,
            "n_citations": len(hits),
            "mismatch_note": mismatch_note,
            "inferred_only": only_inferred,
        })

    rep.coverage = {
        "total": total, "cited": cited,
        "cited_ratio": (cited / total) if total else 0.0,
        "with_retrieved": sum(1 for it in rep.items if it["n_citations"]),
        "inferred_only": inferred_only, "mismatched": mismatched,
    }
    rep.notes.append(
        "「搜」分两条腿：论文内部字面检索（证明哪一页写了什么，本地零成本）"
        "＋ 外部文献检索（真实多源查询，结果只当候选依据，不参与任何判定）。")
    if total and cited < total:
        rep.notes.append(
            f"有 {total - cited} 条结论没有可定位出处 —— 已列在 items 里，"
            "建议人工复核（系统不替它编一个出处）。")
    return rep


# ---------------------------------------------------------------- 外部检索（可选钩子）

_EXTERNAL_PROMPT = """你在帮一份科研复现审计报告做**外部文献线索检索**。

【只在有把握时给线索】不确定的一律不要写。宁可返回空列表，也不要编造文献。

【检索意图】
{intent}

【约束】
1. 只给你能确认存在的公开文献/官方文档；
2. 每条给：title / source（期刊、会议或官方文档名）/ why（一句话说明它与本问题
   的关系）/ confidence（high 或 low）；
3. 最多 3 条；一条都没有就返回空列表；
4. 只输出 JSON：{{"items": [{{"title": "...", "source": "...", "why": "...",
   "confidence": "high"}}]}}，不要任何解释文字。
"""


def external_search(intent: str, *, enabled: bool = False, chat: Optional[Callable] = None,
                    api_key: str = "") -> dict:
    """外部文献检索 —— **默认关闭**的可选钩子（要接 MindSearch / 端砚时从这里进）。

    为什么默认关：项目线上是"无口令门 + 无 API key"的零成本配置，外部检索是要花钱的；
    且检索回来的文献**无法自动核验**，只能当线索。

    返回 `{enabled, items, note}`；所有返回项都带 `unverified=True`，
    调用方不得把它们当作已核验的引证。
    """
    if not enabled:
        return {"enabled": False, "items": [],
                "note": "外部文献检索未启用（默认关闭，零成本配置）；本次只有论文内部检索。"}
    if chat is None and not api_key:
        return {"enabled": True, "items": [],
                "note": "外部文献检索已启用，但未配置模型 API —— 如实记为未检索，不假装搜过。"}
    if chat is None:
        from .llm import ask_json
        chat = lambda p: ask_json(p, api_key, mode="search", max_tokens=600)  # noqa: E731
    try:
        obj, note = chat(_EXTERNAL_PROMPT.format(intent=intent[:600]))
    except Exception as ex:      # noqa: BLE001
        return {"enabled": True, "items": [],
                "note": f"外部检索调用失败: {type(ex).__name__}: {ex}"}
    if not obj:
        return {"enabled": True, "items": [], "note": note or "外部检索未返回结果"}
    items = []
    for it in (obj.get("items") or [])[:3]:
        if not isinstance(it, dict) or not it.get("title"):
            continue
        items.append({"title": str(it.get("title")),
                      "source": str(it.get("source", "")),
                      "why": str(it.get("why", "")),
                      "confidence": str(it.get("confidence", "low")),
                      # 关键: 一律标未核验，避免外部线索被当成审计结论
                      "unverified": True})
    # 「未核验」这句**必须由本模块强制补上**，不能依赖模型返回的说明 —— 否则换个
    # 模型（或模型今天心情不同）返回别的话，这条安全提醒就静默消失了。
    parts = [p for p in (note, f"共 {len(items)} 条**未核验**外部线索，仅供人工参考"
                         if items else "") if p]
    if items:
        parts.append("外部检索结果无法自动核验，不得当作出自论文的引证使用")
    return {"enabled": True, "items": items, "note": "；".join(parts)}


def literature_search(findings: Any = None, *, query: str = "",
                      paper_title: str = "", max_params: int = 3,
                      limit: int = 4, **kw) -> dict:
    """外部文献检索 —— **真实多源查询**（`literature.py` 的审计侧入口）。

    两种用法：
      · 给了 `query`   → 拿它直接查（对外 MCP 工具 / 用户手动搜都走这条）；
      · 只给 `findings` → 由审计结论自动生成 2~3 个查询词（参数名 ＋ 论文标题实词）。

    返回结构对外统一成 `{enabled, mode, items, sources, reached, note, unverified}`，
    与旧的 `external_search()` **同名字段兼容**（`enabled` / `items` / `note`），
    这样报告层不用改就能两种都渲染。
    """
    from . import literature

    if not literature.enabled():
        return {"enabled": False, "mode": literature.MODE, "items": [], "sources": [],
                "reached": 0,
                "note": ("外部文献检索已关闭（PAPER2LAB_LIT=off）—— 本次只做论文内部"
                         "检索，不假装搜过。"),
                "unverified": True}

    if str(query or "").strip():
        res = literature.search(str(query), limit=limit, **kw)
    else:
        res = literature.for_findings(findings or [], paper_title,
                                      max_params=max_params, limit=limit, **kw)

    items = list(res.get("items") or [])
    if items:
        note = res.get("note") or ""
        tail = f"共 {len(items)} 条**未核验**候选文献，仅供人工参考"
        note = "；".join(p for p in (note, tail) if p)
    else:
        note = res.get("note") or res.get("error") or "本次没有检索到相关文献"

    out = {
        "enabled": True,
        "mode": res.get("mode") or literature.MODE,
        "items": items,
        "sources": sorted(res.get("providers") or {}),
        "providers": res.get("providers") or {},
        "reached": int(res.get("reached") or 0),
        "of": int(res.get("of") or len(literature.PROVIDERS)),
        "note": note,
        # 顶层也标一次：外部线索不得当结论用（下游只看一个键时也不会漏）
        "unverified": True,
    }
    for k in ("query", "queries", "per_query", "by_param", "elapsed_s", "cached"):
        if k in res:
            out[k] = res[k]
    return out


def verify_citations(audit: Any, *, top_k: int = 2,
                     external: Optional[dict] = None,
                     use_literature: bool = True) -> CitationReport:
    """引证核验总入口：内部检索 + 出处核验 + 外部文献候选。

    `external` 显式给了就用它（测试桩 / 上游已查过，避免重复打网络）；
    否则默认走真实多源检索（`use_literature=False` 可强制关掉）。
    """
    rep = cite_findings(audit, top_k=top_k)
    if external is not None:
        rep.external = external
    elif use_literature:
        rep.external = literature_search(
            getattr(audit, "findings", None),
            paper_title=getattr(getattr(audit, "paper", None), "title", "") or "")
    else:
        rep.external = external_search("", enabled=False)
    if rep.external.get("enabled") and rep.external.get("note"):
        rep.notes.append(rep.external["note"])
    if rep.external.get("items"):
        rep.notes.append(
            f"外部文献线索 {len(rep.external['items'])} 条，**均未核验**，仅供人工参考。")
    return rep
