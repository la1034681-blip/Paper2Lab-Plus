"""规划 / 调度（规则兜底版）—— 阶段一 1C 起：**主导权在模型，规则只作兜底**。

职责: 排一份"本次要跑哪些工具、跳过哪些"的调用序列，并给**每条决策留下理由**。
      现在这份工作的**主导者是大模型**（`use_llm=True` 即走模型路径）；
      本文件的规则版是它的**兜底路径** —— 模型不可用或输出不合法时回落，并如实
      标 `planner="rule"`，不许悄悄当成模型结果。

与旧版（AI 提议 + 程序盖章）的唯一区别，就一条:
  **不设程序闸门。** 模型说 run 就 run、说 skip 就 skip，程序不再否决。
  程序只做两件**不参与决策**的事:
    ① 把工具清单（菜单）交给模型 —— 见 science_tools.menu()；清单外的工具
       根本不存在，所以"乱调"没有对象，边界靠"没有那个东西"而不是靠"门"；
    ② 每次决策原样留痕（谁定的 / 什么理由 / 有没有给理由）—— 事后可逐条倒查。
  唯一保留的机械动作: 模型不可用 → 回落规则版并标 planner="rule"（防卡死）。

⚠️ 别把"规则兜底"当成 bug 回滚: 文件名叫「调度员（规则版）」是**历史叫法**
  （那时确实是规则程序在排计划）。1C 之后对外统一讲「规划与调度由模型主导」。
  顺带一个代价，必须主动讲清: 交权之后**取证路线可能每次不同** ——
  可复现收窄为**只在判定环节**（判定内核 / 核验三通道 / 复算仍是确定性的）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .toolbox import TOOLS, Job

# 论文规模阈值: 决定 Agent 补漏的耗时预算(页数少则收紧, 页数多则放宽)
_SHORT_PAPER_PAGES = 8
_LONG_PAPER_PAGES = 40


@dataclass
class PlanDecision:
    """一条调度决策。"""
    tool: str          # 工具 key
    title: str
    action: str        # run | skip
    reason: str
    # 这条决策**谁定的**:
    #   rule —— 规则兜底路径定的（材料事实，或"模型没表态 → 按默认执行"）
    #   llm  —— 模型定的（1C 起直接生效，程序不再否决）
    # 注: 1C 之前还有第三个取值 `veto`（模型想这么定、被程序按硬规则否决）。
    #     **程序闸门已按用户口径删除**，所以 `veto` 不再产生；旧留痕仍能读
    #     （字段就是普通字符串，向后兼容）。
    source: str = "rule"


@dataclass
class Plan:
    """一次执行的计划。"""
    mode: str = "planned"                 # 只剩一种：按材料排出的计划（2026-10-25 删掉 classic）
    order: tuple = ()                     # 要执行的工具 key(按顺序)
    decisions: list = field(default_factory=list)   # [PlanDecision]
    notes: list = field(default_factory=list)       # 面向用户的范围说明
    agent_budget: dict = field(default_factory=dict)  # 调度员可调整 Agent 预算
    planner: str = "rule"                 # rule | llm（谁排的这份计划）

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "planner": self.planner,
            "order": list(self.order),
            "decisions": [{"tool": d.tool, "title": d.title,
                           "action": d.action, "reason": d.reason,
                           "source": d.source}
                          for d in self.decisions],
            "notes": list(self.notes),
            "agent_budget": dict(self.agent_budget),
            "ran": sum(1 for d in self.decisions if d.action == "run"),
            "skipped": sum(1 for d in self.decisions if d.action == "skip"),
        }


def _page_count(pdf_path: str) -> int:
    """只读页数(不解析正文), 用于规模判断。"""
    try:
        import pymupdf
        with pymupdf.open(pdf_path) as doc:
            return doc.page_count
    except Exception:      # noqa: BLE001
        return 0


def _chem_survey(pdf_path: str, max_pages: int = 6) -> int:
    """排计划前扫一眼正文, 数一数有没有化学标记（分子式 / CAS / SMILES / 分子量声明）。

    为什么必须在 plan 阶段做: 模型排计划时论文**还没解析**（`parse_paper` 排在 `plan`
    之后），它手上只有页数和材料清单。没有事实就只能瞎猜 —— 而"本文有 12 处化学标记"
    这句话，才是它判断「化学身份核验该不该跑」的依据。**交权要连知情权一起交**。

    只取前 max_pages 页: 化学标记几乎必出现在摘要 / 方法 / 结果段，扫全篇纯属浪费。
    扫描失败一律当 0 —— 这里只用来"给建议"，绝不能因为探针出错就拦住审计。
    """
    if not pdf_path:
        return 0
    try:
        import pymupdf

        from . import chem as _chem
        n = 0
        with pymupdf.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                if i >= max_pages:
                    break
                n += len(_chem.measure_text(page.get_text() or "", i + 1))
        return n
    except Exception:      # noqa: BLE001
        return 0


def _chem_markers_of(job) -> int:
    """取本次材料的化学标记数（同一次规划里只扫一遍, 结果缓存在 job 上）。"""
    cached = getattr(job, "_chem_markers", None)
    if cached is None:
        cached = _chem_survey(getattr(job, "pdf_path", "") or "")
        try:
            job._chem_markers = cached
        except Exception:      # noqa: BLE001
            pass
    return int(cached)


def make_plan(job: Job, *, use_llm: bool = False,
              proposal: Optional[dict] = None) -> Plan:
    """排计划 —— **只有一条路: 按材料排, 能交给模型就交给模型**。

    · `use_llm=True`  → 模型全权排计划（1C 起的主导路径，见 `_llm_plan`）。
      `proposal` 可直接注入一份模型的输出，用于**不调 API 的确定性测试**；
    · `use_llm=False` → 规则版计划（**兜底路径**：可解释、零成本、可复现）。
      模型不可用 / 调用失败 / 输出不合法时也自动落到这里，并标 `planner="rule"`。

    2026-10-25 按用户口径**删除「经典模式」**（不看材料、固定全序全跑）:
    计划永远按材料事实排 —— 没有代码包就不排代码工序，正文没有化学内容就不排化学核验。
    "固定全跑"那套只剩一个用途（消融对照），已由 `skip_tools` 的反向口径承担。
    """
    base = _rule_plan(job)
    if not use_llm:
        return base
    return _llm_plan(job, base, proposal=proposal)


def _rule_plan(job: Job) -> Plan:
    """规则版计划（**兜底路径**）—— 只看材料事实, 不问模型。"""
    # ---- 材料勘察（只看"有没有", 不解析内容）
    has_code = bool(job.code_path) and Path(job.code_path).exists()
    # 2026-10-30：**「路径填错」和「没给代码包」是两件事**，不能共用一句话。
    # 入口（pipeline.run_audit / graph.run_audit_*）已前置校验并会抛错；
    # 这一层只负责措辞 —— 万一有人直接构造 Job 跑，也别把「填错了」说成「没给」，
    # 否则用户会以为「代码里确实没有这些参数」，把一次输入事故当成事实。
    _code_path_bad = bool(job.code_path) and not has_code
    _NO_CODE_REASON = (
        f"代码包路径不存在：{job.code_path} —— 这是路径填错，不是「没有代码」；"
        "请改正路径或把这一栏留空后重跑"
        if _code_path_bad else
        "未提供代码包 —— 本次只能做论文侧自查，代码侧比对整体缺席")
    has_results = bool(job.user_results)
    # 读图通道是否可用, 必须与图像通道**同一口径**: 页面传了 key, 或者 .env/环境变量里配了。
    # (早期版本只看传入的 key, 导致 CLI 入口明明在 .env 配了 key 却被判"未配置"而白白跳过读图)
    try:
        from .tools import vision_ready
        has_vision = vision_ready(job.vision_kwargs)
    except Exception:      # noqa: BLE001
        has_vision = bool(job.vision_kwargs.get("api_key"))
    has_llm = bool(job.api_key)
    pages = _page_count(job.pdf_path)
    # 化学标记数：规则版据此决定「化学身份核验」跑不跑；模型版把它当**事实**递过去。
    # 只取前几页、纯正则，几十毫秒 —— 换模型一次"知情决策"，很划算。
    chem_markers = _chem_markers_of(job)

    decisions: list = []
    order: list = []
    notes: list = []

    for t in TOOLS:
        if t.core:
            decisions.append(PlanDecision(t.key, t.title, "run", "核心步骤，不可跳过"))
            order.append(t.key)
            continue

        if t.key == "analyze_code" and not has_code:
            decisions.append(PlanDecision(
                t.key, t.title, "skip", _NO_CODE_REASON))
            continue

        # 根因追查 / 补丁草案也要读代码: 没有代码包时它们无事可做（不是"跑了没结果", 是"无从追起"）。
        # 注意补丁还依赖根因, 所以这一条一起盖住两站, 避免出现"根因被跳过、补丁却跑了"的错序。
        if t.key in ("rootcause", "patch") and not has_code:
            decisions.append(PlanDecision(
                t.key, t.title, "skip",
                _NO_CODE_REASON if _code_path_bad else
                "未提供代码包 —— 追根因要读代码，没有代码就无从追起（补丁草案随之跳过）"))
            continue

        if t.key == "analyze_images":
            if not has_vision:
                decisions.append(PlanDecision(
                    t.key, t.title, "skip",
                    "未配置读图通道 —— 图片/扫描页里的参数不参与本次审计"))
                continue
            if pages and pages <= 2:
                decisions.append(PlanDecision(
                    t.key, t.title, "skip", f"论文仅 {pages} 页，无图区预期"))
                continue

        # 化学身份核验（2026-10-25 从「图表解析」里独立出来）：正文里**有没有化学标记**，
        # 是"该不该核"最直接的依据 —— 没有标记的论文核了也是 0 条，白占一次执行记录。
        if t.key == "chem_check":
            if chem_markers <= 0:
                decisions.append(PlanDecision(
                    t.key, t.title, "skip",
                    "正文前几页未检测到分子式 / CAS / SMILES 等化学标记 —— "
                    "这份材料不是化学类，核了也是 0 条"))
                continue
            decisions.append(PlanDecision(
                t.key, t.title, "run",
                f"正文检测到 {chem_markers} 处化学标记（分子式 / CAS / SMILES），"
                "可逐条复算对账"))
            order.append(t.key)
            continue

        if t.key == "fill_missing" and not has_llm:
            decisions.append(PlanDecision(
                t.key, t.title, "run",
                "未配置 LLM —— 补漏阶梯仅剩本地三级（同句/近邻/否定式），LLM 级自动跳过"))
            order.append(t.key)
            continue

        if t.key == "llm_analyze" and not has_llm:
            decisions.append(PlanDecision(
                t.key, t.title, "run",
                "未配置 LLM —— 归因改走内置模板（只影响解释文字，不影响结论）"))
            order.append(t.key)
            continue

        decisions.append(PlanDecision(t.key, t.title, "run", "材料齐备，按计划执行"))
        order.append(t.key)

    # ---- Agent 预算按论文规模微调（用户显式给定的值优先, 不覆盖）
    budget = dict(job.agent_budget or {})
    if pages:
        if pages <= _SHORT_PAPER_PAGES and "max_seconds" not in budget:
            budget["max_seconds"] = 30.0
            notes.append(f"论文 {pages} 页（短），Agent 补漏耗时上限收紧到 30s")
        elif pages >= _LONG_PAPER_PAGES and "max_seconds" not in budget:
            budget["max_seconds"] = 90.0
            notes.append(f"论文 {pages} 页（长），Agent 补漏耗时上限放宽到 90s")

    if not has_results:
        notes.append("未提供实测结果 —— 结果一致性维度不在本次评估范围内")
    if not has_code:
        notes.append("未提供代码包 —— 报告中「代码未找到」类结论源于缺材料，不代表论文有问题")
    if not notes:
        notes.append("材料齐备，未触发任何跳过规则")

    # 上下文装配器(阶段三第2步): 会话记忆按"论文侧 vs 代码侧"对齐后的对照屏
    #   —— 比一段人话更利于模型一眼对齐。
    screen = getattr(job, "context_screen", None)
    if screen is not None:
        summary = _screen_summary(screen)
        if summary:
            notes.insert(0, summary)

    # 跨任务记忆命中的线索(只提示, 不影响上面的任何决策)
    for ln in (job.memory_lines or []):
        notes.insert(0, ln)

    return Plan(mode="planned", planner="rule", order=tuple(order), decisions=decisions,
                notes=notes, agent_budget=budget)


# ---------------------------------------------------------------- 模型排计划（1C 起的主导路径）

# 「材料在就必须做」的五件 —— 1C 之后**不再是程序闸门**，只作为**事实**告诉模型:
# 跳过这五件不是"值不值得"，而是"还算不算一次审计"。但决定权在模型手上:
# 它硬要跳，程序不再拦（只把"它跳了核心步骤"如实写进 notes，事后可倒查）。
# 保留这个集合的用途只剩一个 —— 生成提示，不作否决。
_MUST_RUN = {"parse_paper", "analyze_code", "align", "selfcheck", "report"}

# 「跳了它这一轮就没有任何结论」的步骤 → 底线条目（不是闸门，是防空转的底线）。
# 现状只有 align 一个：它是唯一产出 job.audit 的地方（见 _run_align）。
# 新增条目前必须确认"没有它就真的一个结论都出不来"，否则就是在夺回模型决定权。
_NO_RESULT_WITHOUT = {"align": "「参数对齐与结果核验」"}

# 四站主干里**① 假设 / ② 实验**两道工序 → 底线条目（2026-10-31 用户定版）。
# 背景：模型排计划会"砍步骤" —— 实测给 demo 只排 5 件，把假设生成和实验设计都跳了，
#   于是界面上四站里的前两站恒为空，用户看到的"提→做→判→改"只剩下"判→改"。
#   用户拍板：**保留模型排计划（决定权仍归它），但①②不许空**。
# 与 `_NO_RESULT_WITHOUT` 的区别：
#   · align 那条是"没有它一个结论都出不来"（防空转）；
#   · 这两条是"没有它四站主干就断了两站"（防缺站）。
#   两者都只兜底**规则版本要跑、被模型跳过**的那一件（材料本身没有的不补，那不是取舍）。
_BASELINE_STATIONS = {
    "hypothesize": "「假设生成」（四站主干①：原始材料 → 假设卡）",
    "design_experiment": "「实验设计与真跑」（四站主干②：假设卡 → 实验卡）",
}

# 合并后的底线表：key → 说明（为什么跳了也会被执行）。
_BASELINE = {
    **_NO_RESULT_WITHOUT,
    **{k: f"{v}是这条流水线的主干步骤" for k, v in _BASELINE_STATIONS.items()},
}


def _screen_summary(screen) -> Optional[str]:
    """把装配好的上下文屏摘要成一行，给 planner 的 notes 用。

    只说"有哪些参数在历史里存在对照"，不给值（值在模型侧的 prompt 里）。
    """
    try:
        rows = screen.param_table or []
    except AttributeError:
        return None
    if not rows:
        return None
    present = [
        r.param_key for r in rows
        if getattr(r, "paper", None) is not None and getattr(r, "code", None) is not None
    ]
    if not present:
        return None
    return f"【记忆对照】{len(present)} 个参数在历史记忆里同时有论文侧与代码侧线索: " + "、".join(present[:6])


def _llm_prompt(job: Job, base: Plan, entries: list) -> str:
    """拼给模型的规划 prompt —— 核心是那张**按本次材料裁剪过的菜单**。

    为什么菜单要按材料裁剪: 这就是 1C 之后取代"程序闸门"的那件事 ——
    **没有代码包时，「代码扫描」根本不在菜单上**（没有可作用的对象）。
    模型不是"被否决"，是"清单里没有这个东西"，所以它无从乱调、也无从误选。
    边界靠"没有那个东西"，而不是靠"门"。

    菜单条目与执行体（toolbox.TOOLS）同源派生 —— 3A 登记了科学工具，
    它会自动出现在这里，本文件一行都不用改。
    """
    facts = {
        "有代码包": bool(job.code_path),
        "有实测结果": bool(job.user_results),
        "论文页数": _page_count(job.pdf_path),
        "可读图": any(d.tool == "analyze_images" and d.action == "run" for d in base.decisions),
        # 程序先扫前几页数出来的 —— 不是命令，是让你**知情**的事实：
        # 化学身份核验该不该跑，看这一项比猜标题靠谱得多。
        "正文化学标记数": _chem_markers_of(job),
    }
    shown = [{"key": c.key, "名称": c.title, "成本": c.cost,
              "需要产物": list(c.needs), "产出": list(c.gives),
              "核心步骤": c.core, "归属站": c.station}
             for c in entries]
    return (
        "你在给一个『论文↔代码一致性审计』流水线**全权规划**本次要调哪些工具。\n"
        "决定权在你手上: 说 run 就跑，说 skip 就跳，**不会再有人否决你**。\n"
        "**唯一的例外（提前说清，免得你白写理由）**: 这条流水线是四站主干\n"
        "  ① 假设生成（原始材料 → 假设卡）② 实验设计与真跑（假设卡 → 实验卡）\n"
        "  ③ 参数对齐与结果核验 ④ 报告与修正建议。①② 你即使判 skip 也会被执行\n"
        "  （理由照记，事后可倒查）—— 因为①②空了，剩下的就不叫这条流水线了。\n"
        "  其余工具你判 skip 就真的不跑，没人会改回来。\n"
        "下面是**你能调用的全部工具**（本次材料的清单）; 清单之外的工具**本次不存在**，\n"
        "不要臆造（例如没有代码包时，扫代码/追根因这些根本不在清单上）。\n"
        "请对**清单上的每一个工具**给出 run 或 skip，并**逐个写明理由**。\n\n"
        "几条事实（供你判断，不是命令）:\n"
        f"  · 材料情况: {json.dumps(facts, ensure_ascii=False)}\n"
        "  · 标着「核心步骤」的五件，跳过它们意味着这次审计本身不成立 ——\n"
        "    但**最终由你决定**，你跳了就如实记着。\n"
        "  · 某工具需要上游产物（见「需要产物」）: 上游没跑，本步执行时会被\n"
        "    前置检查拦下并记为「执行不了」—— 这不是程序否决你，是它真的跑不动。\n"
        "  · 拿不准就选 run。\n"
        "  · 「正文化学标记数」是程序扫正文前几页数出来的（分子式 / CAS / SMILES / 分子量声明）：\n"
        "    大于 0 说明这份材料里有可**逐条复算对账**的化学身份；为 0 说明不是化学类，核了也是 0 条。\n"
        "    这一项不是命令，是给你判断用的事实。\n\n"
        f"工具菜单: {json.dumps(shown, ensure_ascii=False)}\n"
        f"规则兜底版给出的计划（可参考、可推翻）: "
        f"{json.dumps([{'tool': d.tool, 'action': d.action, 'reason': d.reason}
                       for d in base.decisions if d.tool in {e.key for e in entries}],
                      ensure_ascii=False)}\n\n"
        "只输出 JSON，格式："
        '{"decisions":[{"tool":"工具key","action":"run|skip","reason":"一句话理由"}],'
        '"notes":["一句总说明"]}'
    )


def _llm_plan(job: Job, base: Plan, *, proposal: Optional[dict] = None) -> Plan:
    """模型全权排计划（1C 起的主导路径）—— **程序不再否决**。

    程序在这里只做两件不参与决策的事:
      ① 递菜单（按本次材料裁剪过的 `science_tools` 条目）；
      ② 原样留痕（每条决策谁定的、什么理由、有没有给理由）。
    唯一保留的机械动作: 模型不可用 / 输出不合法 → 回落规则版并标 `planner="rule"`。

    菜单内: 模型的 run / skip **直接生效**（这是"交权"）。
    菜单外: 那些工具**本次不存在**（材料里没有它们的对象），保持规则版的跳过口径，
            并如实标 `rule`（这条不是模型定的，别记成它定的）。

    三条如实记录（不是否决，是"把话说全"）:
      · 模型说了 skip 却没给理由 → **照跳**（决定权在它），但理由栏如实写
        "未给理由" —— 既不代它编理由，也不假装它没提；
      · 模型跳了核心步骤 → 照跳，另在 notes 里点名（事后可倒查）；
      · 模型提到清单外工具 → 不采纳（不存在），原样记进 notes。
    依赖关系也降级成 notes 里的**提示**: 真正拦住"上游没跑、下游照跑"的是
    执行期的前置检查，那属于"执行不了"，不属于"决策被否决"。
    """
    from .llm import ask_json
    from .science_tools import CATALOG

    # 菜单 = 本次材料下**存在**的工具（规则版因材料事实跳过的，压根不进菜单）
    _possible = {d.tool for d in base.decisions if d.action == "run"}
    entries = [c for c in CATALOG if c.key in _possible]
    allowed = {c.key for c in entries}

    note = "已注入提案（测试模式）"
    if proposal is None:
        proposal, note = ask_json(_llm_prompt(job, base, entries), job.api_key,
                                  mode="plan", max_tokens=900,
                                  force_mock=job.force_mock,
                                  base_url=getattr(job, "llm_base_url", "") or "",
                                  model=getattr(job, "llm_model", "") or "")
    if not isinstance(proposal, dict) or not isinstance(proposal.get("decisions"), list):
        base.notes.insert(0, f"🧠 模型排计划未生效（{note}）—— 已回退规则版计划")
        return base

    said: dict = {}
    unknown: list = []
    for d in (proposal.get("decisions") or []):
        if not isinstance(d, dict) or not d.get("tool"):
            continue
        k = str(d["tool"])
        if k not in allowed:
            unknown.append(k)          # 清单外 = 本次不存在，如实记下来（不采纳也不报错）
            continue
        said[k] = d

    by_key = {t.key: t for t in TOOLS}
    decisions: list = []
    no_reason: list = []
    jumped_core: list = []
    baseline_forced: list = []      # ①② 被底线兜底执行的（模型原本判 skip）
    dep_forced: list = []           # 为让底线跑得动而恢复执行的（上游产物缺）
    for d in base.decisions:
        tool = by_key[d.tool]
        if d.action == "skip":
            # 材料事实上不存在的工具 —— 保持规则口径（它不是"被否决"，是不在清单上）
            decisions.append(PlanDecision(d.tool, d.title, "skip", d.reason, "rule"))
            continue
        want = said.get(d.tool)
        if want is None:
            # 模型没对这一步表态 —— 按默认执行，但**如实标 rule**（不是它定的）
            decisions.append(PlanDecision(
                d.tool, d.title, "run",
                "模型未对这一步表态 —— 按默认执行", "rule"))
            continue
        action = str(want.get("action") or "run").strip().lower()
        if action not in ("run", "skip"):
            action = "run"
        reason = str(want.get("reason") or "").strip()
        if action == "skip" and not reason:
            no_reason.append(d.title)
            reason = "模型判定跳过，但未给理由 —— 如实记录，不代它补理由"
        elif reason:
            reason = f"模型判断：{reason}"
        else:
            reason = "模型判定执行（未给理由）"
        if action == "skip" and (tool.core or d.tool in _MUST_RUN):
            jumped_core.append(d.title)
        # ⚠ 一条**底线**, 不是闸门（2026-10-15 修的真实空转）:
        #   `align` 是**唯一产出审计结论**的步骤（它在 _run_align 里跑 aligner.align,
        #   job.audit 就是那儿来的）。模型若把它跳了, 整轮结束时 job.audit 仍是 None,
        #   run_audit 就返回 (None, None) —— 界面上表现为「点开始一直转、结果不出来」,
        #   实测就是这个组合（只传论文 + 模型排计划）必现。
        #
        #   为什么只保它一个、而不是把 _MUST_RUN 全保下来:
        #     1C 的口径是"调不调由模型定", 其余四件跳掉只是少一步, 仍有结论;
        #     而跳过结论产出这一步, 这一轮**根本不算审计** —— 那是空转, 不是取舍。
        #   所以这里只对"没有它就没有结论"的那一件兜底, 且**如实留痕**
        #   (source 仍记 llm、理由栏写清是底线兜底), 不静默改写模型的决策。
        #
        # 2026-10-31 用户加码：四站主干①②（假设 / 实验）也是底线 —— 模型的 skip
        #   在这两件上不生效（证据：实测 demo 只排 5 件、①② 恒空，四站变两站）。
        #   注意**只兜底规则版本来要跑的那一件**：材料本身没论文时规则版就是 skip
        #   （source="rule"），那种情况在循环开头已经 continue 放过，不在这里补。
        if action == "skip" and d.tool in _BASELINE:
            if d.tool in _NO_RESULT_WITHOUT:
                reason = (f"模型判定跳过，但{_NO_RESULT_WITHOUT[d.tool]}是本轮唯一的结论产出步骤"
                          f" —— 跳过它这一轮就没有任何结论（空转）。已按底线兜底执行；"
                          f"模型原话保留在「模型判定跳过」备注里")
            else:
                reason = (f"模型判定跳过，但{_BASELINE_STATIONS[d.tool]}"
                          f"是四站主干，跳过它这一站就空 —— 已按底线兜底执行；"
                          f"模型原话保留在「模型判定跳过」备注里")
                baseline_forced.append(d.title)
            action = "run"
        decisions.append(PlanDecision(d.tool, d.title, action, reason, "llm"))

    # ---- 底线②：底线工序的上游产物必须有人产出 --------------------------------
    # 为什么需要这一步（2026-10-31 加底线时想到、并实测复现过）：
    #   把「假设生成」强制成 run 之后，它 needs 的 `paper` 仍可能没人产出 ——
    #   模型完全可以连「论文解析」一起跳。那种情况下底线工序会在 gate() 处被判
    #   「缺少前置产物」直接 blocked，界面上还是 0 张卡，等于白兜底一次。
    # 只补**被模型跳过、而规则版本要跑**的那一件；材料事实导致 skip 的（source=rule）不动。
    _producers: dict = {}
    for _t in TOOLS:
        for _g in _t.gives:
            _producers.setdefault(_g, set()).add(_t.key)
    _idx = {d.tool: i for i, d in enumerate(decisions)}
    _guard = 0
    _changed = True
    while _changed and _guard < 8:
        _changed = False
        _guard += 1
        _ran = {d.tool for d in decisions if d.action == "run"}
        for _d in decisions:
            # ⚠ 只给**四站主干①②**补上游，不给 align 补（2026-10-31 实测踩到）：
            #   align 的 needs 里有 `code`，若把它也算进来，模型跳「代码扫描」就会被
            #   强行拉回来 —— 那是替用户多要了一样他并不要的东西。① 只要 paper、
            #   ② 只要 hypotheses，闭包很浅，不会顺带推翻模型的其他决策。
            if _d.action != "run" or _d.tool not in _BASELINE_STATIONS:
                continue
            for _need in by_key[_d.tool].needs:
                if _producers.get(_need, set()) & _ran:
                    continue                      # 有人产出，不用管
                _cand = [k for k in _producers.get(_need, set())
                         if k in _idx and decisions[_idx[k]].action == "skip"
                         and decisions[_idx[k]].source == "llm"]
                if not _cand:
                    continue                      # 材料上就没有 → 无从补，交给 gate 如实拦
                _cand.sort(key=lambda k: _idx[k])
                _k = _cand[0]
                _old = decisions[_idx[_k]]
                decisions[_idx[_k]] = PlanDecision(
                    _old.tool, _old.title, "run",
                    f"底线依赖补齐：{by_key[_d.tool].title}必须有「{_need}」，"
                    f"而本次执行集合里没人产出它 —— 已按底线恢复执行；"
                    f"模型原判定：跳过",
                    "llm")
                dep_forced.append(_old.title)
                _changed = True

    order = tuple(d.tool for d in decisions if d.action == "run")

    # ---- 依赖提示（**只提示，不改决策**）: 模型选的执行集合里，有谁的必需产物没人产出
    producers: dict = {}
    for t in TOOLS:
        for g in t.gives:
            producers.setdefault(g, set()).add(t.key)
    ran = {d.tool for d in decisions if d.action == "run"}
    missing = {g for g, ps in producers.items() if not (ps & ran)}
    starved = [by_key[d.tool].title for d in decisions
               if d.action == "run" and (set(by_key[d.tool].needs) & missing)]

    notes = list(base.notes)
    n_skip = sum(1 for d in decisions if d.source == "llm" and d.action == "skip")
    if n_skip:
        notes.insert(0, f"🧠 模型排计划：模型判定 {n_skip} 个工具本次不值得跑（理由见决策表）")
    else:
        notes.insert(0, "🧠 模型排计划：模型未提出额外跳过 —— 执行集合与规则兜底版相同")
    _advisories: list = []
    if jumped_core:
        _advisories.append("模型跳过了核心步骤：" + "、".join(jumped_core)
                           + "（决定权在模型，已如实记录）")
    if baseline_forced:
        # 用户定的口径：①假设 / ②实验**不许空**。模型的 skip 在这两件上不生效，
        #   但要写清楚"这是底线，不是模型改主意了"，否则事后倒查看不出来。
        _advisories.append("模型判定跳过四站主干，已按底线执行：" + "、".join(baseline_forced)
                           + "（①② 为空等于断了两站，故兜底；模型原话仍留在理由栏）")
    if dep_forced:
        _advisories.append("为让底线工序跑得动，恢复了被模型跳过的上游：" + "、".join(dep_forced)
                           + "（否则会在前置检查处被拦，底线也白兜）")
    if no_reason:
        _advisories.append("模型跳过但未给理由：" + "、".join(no_reason)
                           + "（照跳，理由栏如实写「未给理由」）")
    if starved:
        _advisories.append("下列工具的必要产物本次没人产出，执行时会被前置检查拦下："
                           + "、".join(starved))
    if unknown:
        _advisories.append("模型提到了清单外工具（本次不存在，未采纳）："
                           + "、".join(sorted(set(unknown))))
    if _advisories:
        notes.insert(1, "📌 " + "；".join(_advisories))
    notes.append(f"排计划方式：{note}；**决定权在模型**（程序只兜四站主干①②与结论产出，"
                 f"其余一律照模型的判定，全程留痕可倒查）")
    return Plan(mode="planned", planner="llm", order=order, decisions=decisions,
                notes=notes, agent_budget=dict(base.agent_budget or {}))
