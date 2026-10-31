"""LangGraph 骨架（阶段三第3步）——把审计流水线画成状态机，每个动作过闸门并留痕。

为什么是状态机而不是"一个 if 串到底":
  现有 pipeline 是硬编码顺序（parse -> analyze -> align -> ...），
  改一个步骤的先后就要改一整段代码，而且"跳过"与"失败"的处理散落在各处。
  LangGraph 把这件事变成:
    状态（state）= 当前走到哪、手里有什么、哪些过了闸门
    节点（node）= 一个工具的执行（每个工具一个节点）
    边（edge）= 闸门 —— 规则判断"能不能去下一个节点"，不过就拒绝并留痕

  这就是"可验证智能体"的骨架: 模型可以自由探索，但每个动作都要过"机器能查的闸门"。

设计目标（本文件只做骨架，不改动既有代码的行为）:
  1. 现有 10 个工具 -> 10 个节点，每个节点调用 toolbox.TOOLS 里的同一个 run 函数，
     因此"骨架版"与"经典版"跑的是同一份代码，不会漂移。
  2. 每个节点前有一个闸门（gate hook）: 规则检查该工具的输入产物是否齐备，
     不齐备 -> 拒绝（标 skipped + 记录理由），不执行也不静默跳过。
  3. 装配上下文是一个独立节点（consume assembler.assembler 的输出），
     在排计划之前跑，这样 planner 的 notes 里就能带上历史对照。
  4. 失败不吞: 节点抛异常 -> 记入 state.errors -> 走到终止节点（不静默吞掉）。
  5. 留痕: 每个节点的执行结果（done / skipped / failed + 理由 + 耗时）都记在
     state.records 里，与 runner.ExecutionLog 同源，最后交给 report/界面。

与经典版的关系:
  本文件新增，不修改 pipeline/runner 的任何路径。
  经典版（run_audit 全序执行）仍原样保留，CLI 与测试脚本不变。
  骨架版通过 `run_audit_graph()` 暴露，app.py 加一个开关即可切换对比
  （做消融实验时正好用: 同数据跑两版，结论必须一致）。

状态说明（AuditGraphState）:
  输入: pdf_path, code_path, user_results, api_key, ...
  中间: paper, code, audit, report, context_screen, decisions, records
  控制: current_step, next_step, errors
  产出: findings, readiness, execution_log


══════ 多智能体扩展（阶段四）══════

总纲 §6 锁定的改进版:
  1 并行独立抽取: 解析员只看论文，代码验证员只看代码，彼此看不见
  2 三方对撞: 论文值 vs 代码值 + 数值合理性
  3 仲裁员: 冲突时基于证据裁决 + 标置信度（不是简单投票）
  4 红队: 专挑结论漏洞（对抗式验证，降幻觉）

生死线（上下文隔离）:
  每个角色只看该看的，防污染、保独立性。
  若能互相看见或串行传递，就退化成"一个模型自审自"。

为什么串行执行同样满足隔离:
  隔离的关键是"上下文里摆什么"，而不是"同时跑几个"。
  角色层（agent_roles.py）的每个 Agent __call__ 只读自己该看的字段:
    解析员 · 论文视野  只读 job.paper
    解析员 · 代码视野  只读 job.code
    计算员             只看两侧已经抽好的值
  即使按顺序执行，每个角色也物理上看不见另外两侧的材料。
  （阶段一 1B: 三角色转正为七位置编制，此处用到的是解析员与计算员；
    解析员内部仍是两只互不通气的眼睛 —— 隔离从角色边界下移到内部，未松。）

并行区的工程取舍（LangGraph 1.2 的限制）:
  1.2 不支持多个并行任务同时写同一个状态字段（要 2.x 的 Annotated key），
  fan-in 会抛 InvalidUpdateError。因此并行区写成"各写各的槽、仲裁节点收拢"，
  同样满足隔离，且可复现、可做消融对照。

与阶段三的关系:
  - 复用阶段三的节点（start / assemble_context / plan / 10工具 / finalize），
    只是在 plan 之后、align 之前插入一段"角色对撞区"。
  - 复用阶段二的装配器: 对撞区只消费 ContextScreen 里的参数名表，
    不消费全文，上下文不会随角色数翻倍。
  - 对撞口（collider）是纯规则、不调模型，因此:
      同数据跑 classic / graph / multagent 三版，结论必须一致
    —— 消融实验的唯一判据。

checkpoint（断点/恢复）:
  项目 .venv 装了 langgraph-checkpoint，本文件在 build_multagent_graph() 里
  把 MemorySaver 传进去，调用方给一个 thread_id 就能断点续跑。
  注意: 1.2 版只有内存版（MemorySaver），不是 SqliteSaver，
  所以断点只在当前进程有效。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import hashlib
import json
import os
import sys
import time
from time import monotonic as _mono

from .assembler import assemble
from .planner import make_plan
from .runner import _fill_empty, _store_session_memory
from .toolbox import (STEPS, TOOLS, Job, attach_chemistry, attach_experiments,
                      attach_figures, attach_hypotheses, ensure_chemistry)
from .models import AuditResult, Confidence, Evidence

try:
    from langgraph.graph import END, StateGraph
    from typing import TypedDict
    _HAS_LANGGRAPH = True
except Exception:      # noqa: BLE001
    _HAS_LANGGRAPH = False


# ---------------------------------------------------------------- 状态

class AuditGraphState(TypedDict):
    """LangGraph 的状态 schema（每个节点读/写它）。"""
    # 输入
    pdf_path: str
    code_path: str
    user_results: Optional[dict]
    api_key: str
    llm_base_url: str
    llm_model: str
    force_mock: bool
    vision_kwargs: dict
    vision_max_regions: Optional[int]      # 读图区域上限（与 pipeline.run_audit 同源）
    enable_agent: bool
    agent_budget: dict
    llm_planner: bool
    llm_patch: bool
    needs_index: Optional[bool]
    snapshot: Optional[tuple]
    session_fp: str
    # 中间产物
    paper: Optional[Any]
    code: Optional[Any]
    audit: Optional[Any]
    report: str
    context_screen: Optional[Any]          # assembler.ContextScreen
    decisions: list                         # [PlanDecision]
    plan_dict: dict                         # 计划的完整留痕（plan.to_dict()，与 runner 同口径）
    records: list                           # [NodeRecord]
    errors: list                            # 错误留痕
    # 多智能体（阶段四）: 角色只写自己那一个槽，仲裁员收拢
    paper_view: dict                        # analyst 的抽取视图
    code_view: dict                         # verifier 的抽取视图
    numeric_view: dict                      # numeric 的合理性检查
    collisions: list                        # [collider.Collision]
    arbitrations: list                       # [collider.Arbitration]
    red_team: list                          # [collider.RedTeamNote]
    crosscheck: list                        # 双路核对表（基线 vs 对撞，见 node_report_findings）
    # 阶段二 2A：① 假设站的产物（hypothesis.Hypothesis，已按可验证性排序）
    hypotheses: list
    all_hypotheses: list                    # 全部卡（② 站用；hypotheses 是界面展示的截断版）
    hypothesis_total: int                   # 共提出多少条（界面如实报总数, 不只报展示的那几条）
    hypothesis_stats: dict                  # 阶段二 2B：推演统计（空 dict = 本次没推演）
    # 阶段二 2D：② 实验站的产物（experiment.ExperimentCard，已排序）
    experiments: list
    experiment_total: int                   # 共设计了多少张（界面如实报总数, 不只报展示的那几条）
    experiment_stats: dict                  # 实验统计（空 dict = 本次没做实验）
    # 阶段二 2E：① 假设站「读」这一步的图表线索（multimodal.FigureClue，已排序）
    figure_clues: list
    figure_total: int                       # 共读到多少条（界面如实报总数）
    figure_stats: dict                      # 图表统计（空 dict = 本次没读到图注/图）
    # 阶段二 2E-化学: 独立工具 chem_check 的产物（chem.ChemItem，已判定、已排序）
    chem_items: list
    chem_total: int                         # 共抽到多少条（界面如实报总数）
    chem_stats: dict                        # 化学统计（空 dict = 本次没抽到化学信息）
    chem_fallback: bool                     # 结果是「模型跳过后程序补算」的（要如实标注）
    role_llm: bool                          # 对撞区角色是否用模型做规范化（**默认 False**）
    role_llm_note: str                      # 模型这次到底有没有参与（如实留痕，不静默）
    # 2026-10-26 自主模式：② 实验站是否让模型挑计算工具（默认 False，只挑键不造数）
    experiment_llm: bool
    # 2026-10-28 修：`judge_llm` 必须在这里声明。
    # 踩坑经过：`_job_from_state` 会从 state 取 `judge_llm` 传给 Job，看上去接线齐全，
    # 但这个 TypedDict 里漏了它 —— LangGraph 只透传**声明过的键**，于是传进来的 True
    # 在图入口就被丢掉，Job 恒拿到 False，假设卡/参数配对的「模型先判」在
    # 多智能体档（网页默认档）**一次都没跑过**，界面显示「未启用」倒是实话。
    # 经典入口 run_audit 不走这张图，所以那儿一直是好的 —— 这类"一条路通、另一条路空转"
    # 只能靠对两条路分别断言发现（tests/verify_graph_flags.py 已固化）。
    judge_llm: bool
    # 2026-10-28 同类修复：`fingerprint` 也在初始 state 里给了值，但没声明。
    # 它用于「同论文跨任务记忆」的索引键（`_store_session_memory` 读 job.fingerprint），
    # 声明缺失会让图路径上的记忆写入拿到空键（经典路径不受影响）。
    fingerprint: str
    # 2026-10-26 修：归因这一步的运行事实必须跨节点带着走。
    # 原来只在 llm_analyze 节点内部的 Job 上设了这几个值，下一个节点
    # `_job_from_state` 重建 Job 时全丢，于是界面明明在调模型，
    # 「归因引擎」那一栏却显示「内置模板（未配置 API key）」—— 结论对、标注假。
    llm_mode: str                           # deepseek / fallback / 空
    llm_real_calls: int                     # 真实调用次数（不含缓存）
    llm_cache_hits: int                     # 缓存命中次数
    llm_error: str                          # 降级原因（llm_mode=fallback 时）
    llm_shard: dict                         # 归因分片并发记录
    llm_model_used: str                     # 归因实际用的模型名
    # 阶段五（M2 闭环扩展）: 「搜 / 算 / 做」+ 迭代回边
    loop_enabled: bool                      # 是否启用闭环扩展链（默认 False，不动既有路径）
    loop_rounds: int                        # 迭代回边轮数上限（默认 3）
    compute: dict                           # 「算」小规模复算与静态推演
    citations: dict                         # 「搜」引证检索与核验
    repro_plan: dict                        # 「做」复现方案 + 可复现性评级
    loop_result: dict                       # 迭代回边留痕（逐轮推演记录）
    # 2026-10-25 自主回头：模型决定「要不要再走一遍、换什么值再算」，程序七道闸门验
    auto_loop: bool                         # 是否允许模型自主回头（默认 False）
    auto_round: int                         # 已经回头几次（硬上限见 autonomy.MAX_AUTO_ROUNDS）
    auto_decisions: list                    # 每次决策 + 过闸门全过程 + 事后回填（报告读它）
    auto_assume: dict                       # 本次回头的假设值 {key, side, value}（用完即弃）
    # 控制
    current_step: str                       # 当前节点 key
    next_step: str                          # 下一节点 key（由边的条件函数写）


@dataclass
class NodeRecord:
    """一个节点的执行留痕（与 runner.StepRecord 同源，加 gate 字段）。"""
    key: str
    status: str            # done | skipped | failed
    reason: str = ""
    seconds: float = 0.0
    gate: str = ""         # 闸门结果: "pass" | "blocked"
    checkpoints: tuple = ()

    def to_dict(self) -> dict:
        return {
            "key": self.key, "status": self.status, "reason": self.reason,
            "seconds": round(self.seconds, 2), "gate": self.gate,
            "checkpoints": list(self.checkpoints),
        }


# ---------------------------------------------------------------- 闸门

# 每个工具的"输入产物必须齐备"——不过闸就不执行，而不是静默跳过。
# 口径与 toolbox.TOOLS 的 needs 一致，但这里显式写出来供审计（可核查，不是隐式约定）。
_PRECONDITIONS = {
    "parse_paper": {},
    "analyze_images": {"paper"},
    # 阶段二 2E：图表解析只要求论文（图注来自正文文本层; 无代码包、无视觉 key 都能跑）。
    "read_figures": {"paper"},
    # 阶段二 2E-化学：化学身份核验（2026-10-25 从「图表解析」拆成独立工具）同样只要有论文。
    # 不登记的话 gate() 会当成"无前置要求"直接放行 —— 论文没解析时跑了也是空转。
    "chem_check": {"paper"},
    "fill_missing": {"paper"},
    "analyze_code": {},
    # 阶段二 2A：只要论文就能提假设（代码 ZIP 是可选的；没有代码包时提"论文独有"的卡）。
    # 不登记的话 gate() 会当成"无前置要求"直接放行，等于没这道检查。
    "hypothesize": {"paper"},
    # 阶段二 2D：② 实验站要有假设卡才谈得上"设计实验" —— 没有卡不是被拦, 是真的无从下手。
    "design_experiment": {"hypotheses"},
    "align": {"paper", "code"},
    "llm_analyze": {"findings"},
    "selfcheck": {"findings"},
    "rootcause": {"code", "findings"},
    "patch": {"rootcause"},
    "report": {"findings"},
    # 多智能体对撞区（阶段四）——不登记的话 gate() 会当成"无前置要求"直接放行，
    # 闸门就形同虚设了。登记后：没有两侧产物就不许开对撞，没有冲突就不许开仲裁。
    "roles": {"paper", "code"},
    "collider": {"views"},
    "arbiter": {"collisions"},
    "redteam": {"collisions"},
    # report_findings 只要求"两侧视图在"，不要求"有冲突" ——
    # 没有冲突时它仍要把（空的）结论写进 audit.findings，保持与经典版口径一致。
    "report_findings": {"views"},
    # 阶段五（M2 闭环扩展）—— 同样必须登记，否则闸门形同虚设。
    # 「算」「搜」都要有结论才谈得上复核；「做」要结论齐；迭代回边还要能读到代码。
    "compute": {"findings"},
    "search": {"findings"},
    "repro_plan": {"findings"},
    "loop": {"findings", "code"},
}


def gate(tool_key: str, state: AuditGraphState) -> tuple[str, str]:
    """检查某个工具的输入产物是否齐备。

    返回 (gate, reason): gate="pass" 才执行；gate="blocked" 则拒绝并记录理由。
    """
    needs = _PRECONDITIONS.get(tool_key, ())
    if not needs:
        return "pass", "无需前置产物"
    missing = [n for n in needs if not _have(state, n)]
    if missing:
        return "blocked", f"缺少前置产物: {'、'.join(missing)}"
    return "pass", "前置产物齐备"


def _have(state: AuditGraphState, name: str) -> bool:
    """判断状态里是否已有某个产物。"""
    if name == "paper":
        return bool(state.get("paper"))
    if name == "code":
        return bool(state.get("code"))
    if name == "hypotheses":
        # 阶段二 2A/2D：① 站提出来的假设卡 —— ② 站据此设计实验。
        # 空列表 = 这次没提出假设（材料两侧没有值得追问的分歧）→ ② 站如实 blocked。
        return bool(state.get("hypotheses"))
    if name == "findings":
        audit = state.get("audit")
        return bool(audit and getattr(audit, "findings", None))
    if name == "rootcause":
        audit = state.get("audit")
        return bool(audit and getattr(audit, "rootcause", None))
    # 对撞区产物（阶段四）
    if name == "views":
        # 两侧视图都非空才叫"能对撞"；只有一侧等于没对手
        return bool(state.get("paper_view")) and bool(state.get("code_view"))
    if name == "collisions":
        return bool(state.get("collisions"))
    if name == "arbitrations":
        return bool(state.get("arbitrations"))
    # 阶段五（M2）
    if name == "compute":
        return bool(state.get("compute"))
    if name == "citations":
        return bool(state.get("citations"))
    if name == "repro_plan":
        return bool(state.get("repro_plan"))
    return False


# ---------------------------------------------------------------- 节点

def _record(state: AuditGraphState, key: str, status: str, reason: str = "",
            seconds: float = 0.0, gate: str = "", checkpoints: tuple = ()) -> None:
    state.setdefault("records", []).append(
        NodeRecord(key=key, status=status, reason=reason, seconds=seconds,
                   gate=gate, checkpoints=checkpoints).to_dict()
    )


def node_start(state: AuditGraphState) -> dict:
    """起点: 初始化状态，把 Job 的字段填进去（兼容既有调用方）。"""
    return dict(state, records=[], errors=[], current_step="start",
                next_step="assemble_context", decisions=state.get("decisions") or [])


def node_assemble_context(state: AuditGraphState) -> dict:
    """上下文装配（阶段三第2步）——独立节点，在排计划之前跑。"""
    t0 = _mono()
    try:
        from .session import get as _get_session
        store = _get_session()
        recall = []
        if store is not None:
            fp = state.get("session_fp") or state.get("fingerprint") or ""
            if fp:
                recall = store.recall(fp)
        job = _job_from_state(state)
        screen = assemble(job, recall=recall if recall else None)
        state["context_screen"] = screen
        _record(state, "assemble_context", "done", seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"assemble_context: {type(ex).__name__}: {ex}")
        _record(state, "assemble_context", "failed", str(ex),
                seconds=_mono() - t0)
    state["next_step"] = "plan"
    return dict(state, next_step="plan")


def node_plan(state: AuditGraphState) -> dict:
    """排计划: 由 planner.make_plan 排（1C 起**模型主导**，规则版兜底）。"""
    t0 = _mono()
    try:
        job = _job_from_state(state)
        plan = make_plan(job, use_llm=job.llm_planner)
        state["decisions"] = [
            {"tool": d.tool, "title": d.title, "action": d.action,
             "reason": d.reason, "source": getattr(d, "source", "rule")}
            for d in plan.decisions
        ]
        # 计划的完整留痕要跟着走 —— 多智能体路径原先把它丢了（finalize 里写死 notes=[]、
        # plan={}），于是"这次是谁排的、跳了什么、为什么"在执行记录里看不到。
        # 1C 交权之后，留痕就是唯一的可倒查手段，不能再丢；顺带把口径对齐 runner。
        state["plan_dict"] = plan.to_dict()
        _record(state, "plan", "done", reason=f"排计划方式={plan.planner}",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"plan: {type(ex).__name__}: {ex}")
        _record(state, "plan", "failed", str(ex),
                seconds=_mono() - t0)
    state["next_step"] = "parse_paper"
    return dict(state, next_step="parse_paper")


def _job_from_state(state: AuditGraphState) -> Job:
    """从图状态还原一个 Job（与 pipeline.run_audit 同源，保证口径一致）。"""
    return Job(
        fingerprint=state.get("fingerprint", ""),
        pdf_path=state.get("pdf_path", ""),
        code_path=state.get("code_path", ""),
        user_results=state.get("user_results"),
        api_key=state.get("api_key", ""),
        llm_base_url=state.get("llm_base_url", ""),
        llm_model=state.get("llm_model", ""),
        force_mock=state.get("force_mock", False),
        vision_kwargs=state.get("vision_kwargs", {}),
        vision_max_regions=state.get("vision_max_regions"),
        enable_agent=state.get("enable_agent", True),
        agent_budget=state.get("agent_budget", {}),
        llm_planner=state.get("llm_planner", False),
        llm_patch=state.get("llm_patch", False),
        judge_llm=state.get("judge_llm", False),
        experiment_llm=state.get("experiment_llm", False),
        needs_index=state.get("needs_index"),
        snapshot=state.get("snapshot"),
        paper=state.get("paper"),
        code=state.get("code"),
        audit=state.get("audit"),
        # 2026-10-26 修：② 实验站必须拿到 ① 站的假设卡。
        # 原来这里没传 —— 于是每次 `_job_from_state` 都是空工单，
        # `design_experiments(cards=[])` 返回 0 张，而节点状态仍是 done（静默空转）。
        # 这类"跑了但什么都没算"最难发现：进度条走完、报告有章节、内容全是零。
        # `all_hypotheses` 专门给 ② 站用（不截断），`hypotheses` 是界面展示那份。
        hypotheses=state.get("hypotheses") or [],
        all_hypotheses=state.get("all_hypotheses") or state.get("hypotheses") or [],
        # 2026-10-26 修：归因事实跨节点带着走（否则界面「归因引擎」一栏永远显示未配置）
        llm_mode=state.get("llm_mode", ""),
        llm_real_calls=state.get("llm_real_calls", 0) or 0,
        llm_cache_hits=state.get("llm_cache_hits", 0) or 0,
        llm_error=state.get("llm_error", "") or "",
        llm_shard=dict(state.get("llm_shard") or {}),
        llm_model_used=state.get("llm_model_used", "") or "",
    )


# 多智能体图的工具节点顺序覆盖（analyze_code 之后不是 align，而是 roles）。
# 定义成模块级常量（而不是闭包参数），因为 _make_tool_node 在 build_graph()
# 里先被调用，闭包会捕获到 None。
_MULTAGENT_AFTER_MAP = {
    # 阶段二 2A：多智能体路径里 analyze_code 之后**先提假设**再进对撞区。
    # 为什么必须显式串起来: hypothesize 按 TOOLS 顺序排在 analyze_code 之后，
    # 而本表把 analyze_code 的后继改成了 roles —— 不补这一条，假设节点会被整段跳过，
    # 于是"单路径链有假设卡、多智能体模式没有" = 两条链口径漂移，消融对照直接废掉。
    "analyze_code": "hypothesize",
    # 阶段二 2D：提完假设先**设计实验并真跑**（② 站）, 再进 ③ 对撞区。
    # 与上面同一条理由 —— 不补这两条, 实验节点在多智能体档会被整段跳过,
    # 于是"单路径链有实验卡、多智能体模式没有" = 两条链口径漂移, 消融对照直接废掉。
    "hypothesize": "design_experiment",
    "design_experiment": "roles",
    "align": "report_findings",
}


def _ensure_chem(job, state) -> None:
    """化学核验的**最后一道兜底**（纯规则、零成本、不调模型）。

    2026-10-25 起化学核验是**独立工具** `chem_check`（不再寄生在图表解析里，
    模型也不会因为跳过图表解析而连带丢掉它）。这里保底只为一件事：
    **允许模型跳过，但不许结果静默消失** ——
      · 模型跳过了该步 → 生成报告前补算一遍，并标 `fallback`；
      · report 节点会**重建** Job、不携带 chem_items → 这里把结果重新写回 state/audit。

    已跑过就直接复用（毫秒级，不重复计算），跳过这件事本身照样留痕（见 fallback）。
    """
    from .chem import CHEM_MAX, judge, measure

    audit = getattr(job, "audit", None)
    # 先交给公共兜底（与经典路径同一份实现，不各写一遍）
    ensure_chemistry(job)
    items = list(getattr(job, "chem_items", None) or [])
    total = int(getattr(job, "chem_total", 0) or 0)
    stats = getattr(job, "chem_note", None)
    # job 上没有 paper（state 里有）时自己测一次，别静默放过
    if not items and total == 0 and state.get("paper") is not None:
        raw, stats = measure(state["paper"])
        items = judge(raw)[:CHEM_MAX]
        total = len(raw)
        job.chem_items, job.chem_total, job.chem_note = items, total, stats
        job.chem_fallback = bool(raw)     # 是这里补出来的 → 标注
    fallback = bool(getattr(job, "chem_fallback", False)
                    or state.get("chem_fallback"))
    if items or total:
        state["chem_items"] = items
        state["chem_total"] = total
        state["chem_stats"] = stats
        state["chem_fallback"] = fallback
    if audit is not None:
        attach_chemistry(audit, items, total, stats, fallback=fallback)


def _vision_kwargs_default(vision_kwargs: Optional[dict]) -> dict:
    """把读图参数补齐成与 pipeline.run_audit 同一形状（五键齐备，值可为空）。

    ⚠ 2026-10-29 修: 图版两个入口原来默认给 `{}`，而 `tools.vision_ready()` 对空 dict
    直接判「未配置读图通道」；经典版给的是五键 dict（值为空时回落到 .env / 环境变量）。
    结果同一份材料：经典版读图、图版跳过 —— 三版消融对照失真（verify_graph 实测
    经典 13 执行/1 跳过，图版 12/2）。
    """
    d = dict(vision_kwargs or {})
    d.setdefault("api_key", "")
    d.setdefault("base_url", "")
    d.setdefault("model", "")
    d.setdefault("source_hint", "")
    d.setdefault("budget_seconds", None)
    return d


def _decision_of(state: AuditGraphState, tool_key: str) -> Optional[dict]:
    """取计划里对某个工具的决策（没有计划 / 没表态 → None，按默认执行）。"""
    for d in state.get("decisions") or []:
        if isinstance(d, dict) and d.get("tool") == tool_key:
            return d
    return None


def _apply_skip(state: AuditGraphState, tool_key: str) -> None:
    """计划跳过: 补一份语义正确的空产物（与 runner._fill_empty 同一份实现）。

    ⚠ 少了这一笔, 下游会把"计划跳过"读成"产物不在": 无代码包时 code=None,
    align 被前置检查拦下 → 整轮一条结论都出不来; 而经典版跳过 analyze_code
    仍会补一份空 CodeInfo 照常对表 —— 两条链的跳过语义必须一致。
    """
    job = _job_from_state(state)
    _fill_empty(job, tool_key)
    if tool_key == "analyze_code":
        # _fill_empty 只改绑 job 的属性, 这里要显式写回状态
        state["code"] = job.code
    elif tool_key == "hypothesize":
        state["hypotheses"] = []
        state["all_hypotheses"] = []
        state["hypothesis_total"] = 0
        state["hypothesis_stats"] = {}
    elif tool_key == "design_experiment":
        state["experiments"] = []
        state["experiment_total"] = 0
        state["experiment_stats"] = {}
    elif tool_key == "read_figures":
        state["figure_clues"] = []
        state["figure_total"] = 0
        state["figure_stats"] = {}
    # analyze_images / selfcheck / rootcause / patch 的空产物写在 paper / audit
    # 对象上 —— 与 state 里是同一个引用, 无需再写回。


def _make_tool_node(tool_key: str, *, multagent: bool = False):
    """给一个工具生成"过闸门 -> 执行 -> 留痕"的节点函数。

    multagent=True 时，用 _MULTAGENT_AFTER_MAP 覆盖默认顺序
    （多智能体图里 analyze_code 之后接 roles）。
    """
    tool = next((t for t in TOOLS if t.key == tool_key), None)
    if tool is None:
        raise KeyError(f"未知工具: {tool_key}")

    def _node(state: AuditGraphState) -> dict:
        t0 = _mono()
        # 计划决策优先（与 runner.run_job 同一口径）: 计划说 skip 就不执行。
        # 2026-10-29 修: 原来这里只查"前置产物在不在", 计划写了 skip 也照跑 ——
        # 无代码包时 analyze_code 仍记 done、llm_analyze 整段被绕过,
        # 界面读到"执行 + 跳过 ≠ 全部工具"（实测 A 13/0、B 12/1）。
        _dec = _decision_of(state, tool_key)
        if _dec is not None and _dec.get("action") == "skip":
            _apply_skip(state, tool_key)
            _record(state, tool.key, "skipped", str(_dec.get("reason") or ""),
                    seconds=_mono() - t0, gate="blocked",
                    checkpoints=tool.checkpoints)
            state["next_step"] = _next_after(
                tool_key, state, _MULTAGENT_AFTER_MAP if multagent else None)
            return state
        g, reason = gate(tool_key, state)
        if g != "pass":
            _record(state, tool.key, "skipped", reason,
                    seconds=_mono() - t0, gate=g,
                    checkpoints=tool.checkpoints)
            state["next_step"] = _next_after(tool_key, state,
                                             _MULTAGENT_AFTER_MAP if multagent else None)
            return state
        job = _job_from_state(state)
        try:
            # 复用工箱的执行体（与经典版同一份代码，不漂移）
            # 化学核验：生成报告**之前**必须已跑过（模型跳过 read_figures 时兜底，
            # 否则报告里的「化学身份核验」章节会空 — 详见 _ensure_chem 说明）
            if tool.key == "report":
                _ensure_chem(job, state)
            tool.run(job, lambda n, suffix="": None)
            # 写回状态
            if tool.key == "parse_paper":
                state["paper"] = job.paper
            elif tool.key == "analyze_code":
                state["code"] = job.code
            elif tool.key == "hypothesize":
                # 阶段二 2A：① 假设站的产物写回状态（finalize 再挂到 audit 上）
                state["hypotheses"] = job.hypotheses
                # 2026-10-26：② 站要对着**全部**卡设计实验（job.hypotheses 是截断展示版）。
                # 单独存一个键而不是改上面那条 —— 改已有键的语义 = 动契约（见 L8），
                # 会让"界面显示几条"与"报告写几条"再次分裂。
                state["all_hypotheses"] = list(getattr(job, "all_hypotheses", None)
                                               or job.hypotheses or [])
                state["hypothesis_total"] = job.hypothesis_total
                # 阶段二 2B：推演统计一并带走 —— 图版与经典版必须给出同一套数字，
                # 否则"界面说 2 条已排除、报告说没推演"这种分裂迟早出现。
                state["hypothesis_stats"] = dict(getattr(job, "derivation_note", None) or {})
            elif tool.key == "design_experiment":
                # 阶段二 2D：② 实验站的产物写回状态（finalize 再挂到 audit 上）
                state["experiments"] = job.experiments
                state["experiment_total"] = job.experiment_total
                state["experiment_stats"] = dict(getattr(job, "experiment_note", None) or {})
            elif tool.key == "read_figures":
                # 阶段二 2E：① 假设站「读」这一步的图表线索写回状态（finalize 再挂到 audit）
                state["figure_clues"] = job.figure_clues
                state["figure_total"] = job.figure_total
                state["figure_stats"] = dict(getattr(job, "figure_note", None) or {})
            elif tool.key == "llm_analyze":
                # 2026-10-26 修：归因这一步的"运行事实"必须写回状态。
                # 少了这一段，下面 selfcheck 节点重建 Job 时拿到的 llm_mode 恒为空，
                # 界面「归因引擎」就显示「内置模板（未配置 API key）」——
                # 明明在真调模型，标注却是假的。结论不受影响，但标注必须真。
                state["llm_mode"] = job.llm_mode
                state["llm_real_calls"] = job.llm_real_calls
                state["llm_cache_hits"] = job.llm_cache_hits
                state["llm_error"] = job.llm_error
                state["llm_shard"] = dict(getattr(job, "llm_shard", None) or {})
                state["llm_model_used"] = job.llm_model_used
            elif tool.key == "chem_check":
                # 阶段二 2E-化学：独立工具（2026-10-25 从 read_figures 拆出来）的产物写回状态。
                # 拆开的好处正是"读图"与"核化学"两个决策解耦：模型可以只跑这一个。
                state["chem_items"] = getattr(job, "chem_items", [])
                state["chem_total"] = getattr(job, "chem_total", 0)
                state["chem_stats"] = dict(getattr(job, "chem_note", None) or {})
                state["chem_fallback"] = bool(getattr(job, "chem_fallback", False))
            elif tool.key == "align":
                state["audit"] = job.audit
            elif tool.key == "report":
                state["report"] = job.report
            # 每次执行后把证据写成线索（与 runner 同源）
            if job.fingerprint and getattr(job, "audit", None) is not None:
                _store_session_memory(job)
            _record(state, tool.key, "done",
                    seconds=_mono() - t0, gate=g,
                    checkpoints=tool.checkpoints)
        except Exception as ex:      # noqa: BLE001
            state["errors"].append(f"{tool.key}: {type(ex).__name__}: {ex}")
            _record(state, tool.key, "failed", str(ex),
                    seconds=_mono() - t0, gate=g,
                    checkpoints=tool.checkpoints)
            return dict(state, next_step="finalize")
        nxt = _next_after(tool_key, state,
                          _MULTAGENT_AFTER_MAP if multagent else None)
        return dict(state, next_step=nxt)

    _node.__name__ = f"node_{tool_key}"
    return _node


def _next_after(tool_key: str, state: AuditGraphState,
                after_map: Optional[dict] = None) -> str:
    """一个工具执行完后，下一个该谁（按 TOOLS 顺序，跳过的也算过）。

    after_map: 多智能体图用，覆盖默认顺序（如 analyze_code 之后接 roles）。
    """
    if after_map and tool_key in after_map:
        return after_map[tool_key]
    keys = [t.key for t in TOOLS]
    try:
        i = keys.index(tool_key)
    except ValueError:
        return "finalize"
    if i + 1 < len(keys):
        return keys[i + 1]
    return "finalize"


def node_finalize(state: AuditGraphState) -> dict:
    """收尾: 把状态拼回 (AuditResult, report) 元组，与 run_audit 同源。

    execution_log 的口径必须与 runner.ExecutionLog.to_dict() 一致（键名、含义、
    统计项都对齐），否则界面（app.py 走 execution_log["steps"]）会什么都看不到 ——
    这就是"图跑对了但界面一片空白"的成因。
    额外多给两个键:
      records —— 全量节点留痕（含 roles/collider/... 这些没有 STEPS 对应项的节点）
      errors  —— 节点异常，别静默吞
    """
    audit = state.get("audit")
    records = state.get("records", [])
    if audit is not None:
        # 阶段二 2A：① 假设站的产物挂到 audit 上（与经典路径同一个挂载函数，
        # 界面与报告只读 audit.hypotheses 一处，不分模式）
        attach_hypotheses(audit, state.get("hypotheses"),
                          state.get("hypothesis_total") or 0,
                          state.get("hypothesis_stats") or None)
        # 阶段二 2D：② 实验站的产物同款挂法（与经典路径、与报告生成前同一个函数）
        attach_experiments(audit, state.get("experiments"),
                           state.get("experiment_total") or 0,
                           state.get("experiment_stats") or None)
        # 阶段二 2E：① 假设站「读」这一步的图表线索同款挂法
        attach_figures(audit, state.get("figure_clues"),
                       state.get("figure_total") or 0,
                       state.get("figure_stats") or None)
        # 阶段二 2E-化学：化学身份核验。
        # ⚠ 这里**必须兜底重跑一次**（2026-10-24 实测踩到）：化学核验挂在
        #   `read_figures` 这件工序里，而「模型排计划」模式下模型可以**跳过**
        #   read_figures（demo 化学样例实测：跳过 8 个工具，图表解析/图像解析都没跑），
        #   于是化学核验跟着一起消失 —— 界面显示「核验 0 条」，而同一份 PDF 在
        #   命令行能核出 10 条。这不是"没抽到"，是**根本没执行**。
        #   2026-10-25 起它是独立工具 chem_check（模型可单独跳过），兜底口径不变:
        #   **允许跳过，但不许结果静默消失** —— 补出来的结果标 fallback，报告如实写明。
        _chem_items = list(state.get("chem_items") or [])
        _chem_total = int(state.get("chem_total") or 0)
        _chem_stats = state.get("chem_stats") or None
        _chem_fb = bool(state.get("chem_fallback"))
        if not _chem_items and _chem_total == 0 and state.get("paper") is not None:
            from . import chem as _chem
            _raw, _cst = _chem.measure(state["paper"])
            _chem_items = _chem.judge(_raw)[:_chem.CHEM_MAX]
            _chem_total = len(_raw)
            _chem_stats = _cst
            _chem_fb = bool(_raw)
        # 排查开关：只在需要看"为什么化学核验是 0 条"时打开（默认静默）
        if os.getenv("PAPER2LAB_DEBUG_CHEM"):
            _p = state.get("paper")
            print(f"[chem] paper={'有' if _p is not None else '无'} "
                  f"pages={len(getattr(_p, 'pages', None) or [])} "
                  f"items={len(_chem_items)} total={_chem_total} "
                  f"fallback={_chem_fb} stats={bool(_chem_stats)}", file=sys.stderr)
        attach_chemistry(audit, _chem_items, _chem_total, _chem_stats,
                         fallback=_chem_fb)
        # 工具节点 -> steps（界面进度条只认这个）；title 从 TOOLS 取，不写死
        _title = {t.key: t.title for t in TOOLS}
        steps = [{
            "key": r["key"],
            "title": _title.get(r["key"], r["key"]),
            "status": r["status"],
            "reason": r.get("reason", ""),
            "seconds": r.get("seconds", 0.0),
            "checkpoints": list(r.get("checkpoints", ())),
        } for r in records if r["key"] in _title]
        audit.execution_log = {
            "mode": "graph",
            "steps": steps,
            # 排计划说明与计划留痕（含"这次是模型排的还是回退规则版"）——
            # 1C 交权后留痕是唯一的可倒查手段；与 runner 口径一致（那边也是 plan.to_dict()）。
            "notes": list((state.get("plan_dict") or {}).get("notes") or []),
            "plan": dict(state.get("plan_dict") or {}),
            "seconds": round(sum(r.get("seconds", 0.0) for r in records), 2),
            "ran": sum(1 for s in steps if s["status"] == "done"),
            "skipped": sum(1 for s in steps if s["status"] == "skipped"),
            "failed": sum(1 for s in steps if s["status"] == "failed"),
            # nodes_done 是阶段三就有的旧键名，保留它（别改口径）——
            # 既有验证脚本（verify_graph.py）按它断言，换名等于悄悄改契约。
            # 注意口径: nodes_done 数的是**全部 records**（含 assemble_context/plan
            # 这些控制节点与阶段四的对撞区节点），与 ran（只数工具）不同。
            "nodes_done": sum(1 for r in records if r["status"] == "done"),
            "nodes_total": len(STEPS),
            "nodes_skipped": sum(len(s["checkpoints"]) for s in steps
                                 if s["status"] == "skipped"),
            # 以下四个是图版特有，界面可选消费
            "records": records,
            "errors": state.get("errors", []),
            "crosscheck": state.get("crosscheck", []),
            "arbitrations": state.get("arbitrations", []),
            "red_team": state.get("red_team", []),
            # 对撞区角色有没有用模型 + 模型到底参没参与（默认 False，"没参与"也留痕）
            "role_llm": bool(state.get("role_llm")),
            "role_llm_note": state.get("role_llm_note", ""),
            # 阶段五（M2 闭环）: 搜 / 算 / 做 + 迭代回边
            "loop_enabled": bool(state.get("loop_enabled")),
            "compute": state.get("compute", {}),
            "citations": state.get("citations", {}),
            "repro_plan": state.get("repro_plan", {}),
            "loop": state.get("loop_result", {}),
            # 2026-10-25 自主回头：模型每次「要不要再走一遍」的决定 + 过闸门全过程
            # （只加新键，不动上面任何一个旧键 —— 见 MEMORY 的 L8 契约）
            "auto_loop": bool(state.get("auto_loop")),
            "auto_round": int(state.get("auto_round") or 0),
            "auto_decisions": list(state.get("auto_decisions") or []),
        }
    state["current_step"] = "finalize"
    return dict(state, next_step=END)


# ══════ 多智能体角色节点（阶段四）══════

def node_roles(state: AuditGraphState) -> dict:
    """角色对撞区第一步: 各 Agent 只看自己该看的材料，独立抽取。

    阶段一 1B: 三个角色转正为七位置编制（解析·检索·主持·实验·计算·审查·撰写），
    本节点只用到其中两个 —— **解析员**（内部论文视野 / 代码视野）与**计算员**。
    隔离仍是代码层面强制的（agent_roles.py 的 __call__ 只读自己那一个字段），
    不是靠提示词求它别看。
    """
    t0 = _mono()
    g, reason = gate("roles", state)
    if g != "pass":
        _record(state, "roles", "skipped", reason, seconds=_mono() - t0,
                gate=g)
        state["next_step"] = "collider"
        return dict(state, next_step="collider")
    try:
        from .agent_roles import Calculator, Parser, RoleLLM
        job = _job_from_state(state)
        # 模型版（**默认关**）: 角色只多做"规范化"这一步，抽取/比对全不动。
        # 不启用时 RoleLLM 为 None，解析员走的仍是规则版 ——
        # 与阶段四原版逐字节一致（这是"换编制不改结论"的硬保证）。
        _llm = None
        if state.get("role_llm"):
            _llm = RoleLLM(api_key=state.get("api_key", ""),
                           base_url=state.get("llm_base_url", ""),
                           model=state.get("llm_model", ""),
                           force_mock=bool(state.get("force_mock")))
        # 解析员: 两只眼睛各读自己那一侧（论文视野 / 代码视野），互相看不见。
        # 它交出的就是下面两张视图，并顺手把 job.paper_view / job.code_view 挂好。
        _parser = Parser(_llm)
        _views = _parser(job)
        paper_view, code_view = _views["paper"], _views["code"]
        # 计算员: 只看两侧已经抽好的值，自己不做任何抽取
        numeric_view = Calculator()(job)
        state["paper_view"] = paper_view
        state["code_view"] = code_view
        state["numeric_view"] = numeric_view
        # 模型这次到底参没参与 —— 如实记进留痕，不静默（"没参与"本身也要说）
        _note = _parser.last_note
        state["role_llm_note"] = _note
        _record(state, "roles", "done", reason=_note, seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"roles: {type(ex).__name__}: {ex}")
        _record(state, "roles", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "collider"
    return dict(state, next_step="collider")


def node_collider(state: AuditGraphState) -> dict:
    """对撞: 把三方的结构化结论并排比对，产出冲突清单（纯规则，不调模型）。"""
    t0 = _mono()
    try:
        from .collider import collide
        collisions = collide(state.get("paper_view", {}),
                             state.get("code_view", {}),
                             state.get("numeric_view"))
        state["collisions"] = [c.to_dict() for c in collisions]
        _record(state, "collider", "done", seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"collider: {type(ex).__name__}: {ex}")
        _record(state, "collider", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "arbiter"
    return dict(state, next_step="arbiter")


def node_arbiter(state: AuditGraphState) -> dict:
    """仲裁: 不是简单投票，而是按证据强度裁决（规则版兜底）。"""
    t0 = _mono()
    try:
        from .collider import Arbitration, Collision, arbitrate_all
        collisions = [Collision.from_dict(d) for d in state.get("collisions", [])]
        arbitrations = arbitrate_all(collisions)
        state["arbitrations"] = [a.to_dict() for a in arbitrations]
        _record(state, "arbiter", "done", seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"arbiter: {type(ex).__name__}: {ex}")
        _record(state, "arbiter", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "redteam"
    return dict(state, next_step="redteam")


def node_redteam(state: AuditGraphState) -> dict:
    """红队: 专挑结论漏洞（对抗式验证，降幻觉）。"""
    t0 = _mono()
    try:
        from .collider import Collision, RedTeamNote, red_team
        collisions = [Collision.from_dict(d) for d in state.get("collisions", [])]
        notes = red_team(collisions, state.get("paper_view", {}), state.get("code_view", {}))
        state["red_team"] = [n.to_dict() for n in notes]
        _record(state, "redteam", "done", seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"redteam: {type(ex).__name__}: {ex}")
        _record(state, "redteam", "failed", str(ex), seconds=_mono() - t0)
    # 顺序: 对撞区先跑完，再跑 align 出"基线结论"，最后由 report_findings 做双路核对。
    # 为什么不让 report_findings 直接去 selfcheck（跳过 align）:
    #   align 还负责"参数归一化 / 交叉匹配 / 实验结果核验"这些非 findings 产物，
    #   跳过它会丢掉这些，且与经典版的产物口径不再对齐、消融没法比。
    state["next_step"] = "align"
    return dict(state, next_step="align")


def node_report_findings(state: AuditGraphState) -> dict:
    """双路核对: align 的基线结论 vs 对撞区的独立结论，取并集并记录分歧。

    为什么要这一步（而不是让对撞直接产出 findings）:
      对撞区（Analyst/CodeVerifier/collider）和 align（既有的归一化+交叉匹配）
      是**两条互相独立的推理路径**。两者都对 -> 结论被互证，可信度自然更高；
      两者不一致 -> 这正是最该暴露给审稿人的地方（分歧本身就是发现）。
      直接让对撞覆盖 align，等于把一条路径的白干藏起来，也丢掉了互证这个证据。

    产出口径:
      findings  = align 基线 ∪ 对撞新增（基线优先，保证与经典版消融可比）
      crosscheck = 逐参数核对表（互证 / 分歧 / 仅基线 / 仅对撞）
      两者都挂在 audit.execution_log 上（finalize 时统一写），不污染数据模型。
    """
    t0 = _mono()
    try:
        from .collider import Collision
        from .models import AuditResult, Finding
        collisions = [Collision.from_dict(d) for d in state.get("collisions", [])]
        arbitrations = {a["param_key"]: a for a in state.get("arbitrations", [])}

        # ---- ① 对撞结论 -> Finding（只保留没被仲裁撤销的）
        collided: dict = {}
        for c in collisions:
            a = arbitrations.get(c.param_key)
            if a is not None and a.get("verdict") == "撤销冲突":
                continue
            # 仲裁降级为"疑似"时，置信度跟着降
            conf = Confidence.INFERRED if (a and a["confidence"] == "inferred") else c.confidence
            collided[c.param_key] = Finding(
                param_key=c.param_key,
                display_name=c.display_name,
                category=c.category,
                status=c.status,
                confidence=conf,
                risk=c.risk,
                paper=_ev(c.paper, c.paper_source, c.paper_doc, "paper") if c.paper else None,
                code=_ev(c.code, c.code_source, c.code_doc, "code") if c.code else None,
                note=f"{c.note}；仲裁: {a['verdict']}" if a else c.note,
                risk_reason=c.risk_reason,
            )

        # ---- ② 基线（align 已经跑过，产物在 audit.findings 里）
        audit = state.get("audit")
        if audit is None:
            audit = AuditResult(paper=state.get("paper"), code=state.get("code"))
            state["audit"] = audit
        baseline = {f.param_key: f for f in (getattr(audit, "findings", None) or [])}

        # ---- ③ 逐参数核对（口径: param_key + status）
        crosscheck: list = []
        for key in sorted(set(baseline) | set(collided)):
            b, c = baseline.get(key), collided.get(key)
            if b is None:
                # 对撞发现了基线漏掉的参数 —— 这是多智能体的增量价值
                crosscheck.append({"param_key": key, "verdict": "仅对撞",
                                   "baseline": "", "collision": c.status.value})
            elif c is None:
                crosscheck.append({"param_key": key, "verdict": "仅基线",
                                   "baseline": b.status.value, "collision": ""})
            elif b.status.value == c.status.value:
                crosscheck.append({"param_key": key, "verdict": "互证",
                                   "baseline": b.status.value, "collision": c.status.value})
            else:
                crosscheck.append({"param_key": key, "verdict": "分歧",
                                   "baseline": b.status.value, "collision": c.status.value})

        # ---- ④ 取并集: 基线优先，对撞补上基线没有的
        merged: list = list(baseline.values())
        for key, f in collided.items():
            if key not in baseline:
                merged.append(f)
        audit.findings = merged

        state["crosscheck"] = crosscheck
        n_agree = sum(1 for x in crosscheck if x["verdict"] == "互证")
        n_diff = sum(1 for x in crosscheck if x["verdict"] == "分歧")
        _record(state, "report_findings", "done",
                reason=f"互证 {n_agree} / 分歧 {n_diff} / 共 {len(crosscheck)}",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"report_findings: {type(ex).__name__}: {ex}")
        state.setdefault("crosscheck", [])
        _record(state, "report_findings", "failed", str(ex), seconds=_mono() - t0)
    # 对撞区与 align 都跑完了，交给下游。
    # 阶段五（闭环扩展）开着时先走「算 → 搜 → 做 → 迭代回边」，否则直接进归因。
    # ⚠ 2026-10-29 修: 原来这里写死 selfcheck，把夹在中间的 `llm_analyze` 整段跳过了
    #   （实测 findings 11 条、ai_analysis 全空，执行记录里干脆没有这一步）。
    #   归因要吃 findings，正好排在这里；它的默认后继就是 selfcheck。
    _nxt = "compute" if state.get("loop_enabled") else "llm_analyze"
    state["next_step"] = _nxt
    return dict(state, next_step=_nxt)


def _ev(value: str, source: str, doc: str, channel: str) -> Optional[Evidence]:
    """把一个 Agent 视图格转成 Evidence（供 Finding 使用）。"""
    if not value:
        return None
    return Evidence(source=channel, location=source, snippet=doc[:120], raw_value=value)


# ══════ 阶段五 · M2 闭环扩展（搜 / 算 / 做 + 迭代回边）══════
#
# 四个节点**全部是纯规则、零成本、离线可跑**（唯一可选的外部检索默认关闭），
# 因此它们与对撞区一样：可复现、可做消融、不开 key 也能用。
#
# 为什么单独成一条链、而不是塞进 toolbox.TOOLS:
#   TOOLS 决定进度条节点数（verify_graph 断言"工具数 >= 10"，界面进度条格数也从它派生）。
#   把新能力塞进 TOOLS 等于**改进度契约**，会让既有验证脚本与界面同时失真。
#   所以它们与阶段四的 roles/collider 同等待遇: 新增节点、不占进度槽、只走条件边。

def node_compute(state: AuditGraphState) -> dict:
    """「算」: 小规模复算 + 静态推演（纯规则，不调模型、不执行用户代码）。

    自主回头（`auto_assume`）时：把模型假设的那一侧、那个参数换成它给的值再算一遍 ——
    这是回边产生**新结果**的唯一来源（确定性系统里换输入才换来不同输出）。
    替换只在本节点的局部副本上做，**不回写 paper/code 对象**。
    """
    t0 = _mono()
    try:
        from .compute import review_numbers
        audit = state.get("audit")
        paper = state.get("paper")
        code = state.get("code")
        pp = (getattr(paper, "params", None) if paper is not None else None)
        cp = (getattr(code, "params", None) if code is not None else None)
        auto = dict(state.get("auto_assume") or {})
        if auto.get("key") and auto.get("value") not in (None, ""):
            _pp, _cp = dict(pp or {}), dict(cp or {})
            if auto.get("side") == "paper":
                _pp[auto["key"]] = str(auto["value"])
            else:
                _cp[auto["key"]] = str(auto["value"])
            pp, cp = _pp, _cp
            state["auto_assume"] = {}     # 用完即弃：不许影响下一次决策之外的任何节点
        d = review_numbers(
            findings=(getattr(audit, "findings", None) if audit is not None else None),
            paper_params=pp,
            code_params=cp,
        )
        state["compute"] = d
        if audit is not None:
            audit.compute = d
        _record(state, "compute", "done",
                reason=f"复算 {d.get('checked', 0)} 条 / 提示 {d.get('flagged', 0)} 条 / "
                       f"未算 {d.get('skipped', 0)} 条",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"compute: {type(ex).__name__}: {ex}")
        state.setdefault("compute", {})
        _record(state, "compute", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "search"
    return dict(state, next_step="search")


def node_search(state: AuditGraphState) -> dict:
    """「搜」: 论文内部字面检索 + 引证核验 + 外部文献候选（4A：真实多源查询）。"""
    t0 = _mono()
    try:
        from .retrieval import verify_citations
        audit = state.get("audit")
        rep = verify_citations(audit)
        d = rep.to_dict()
        state["citations"] = d
        if audit is not None:
            audit.citations = d
        c = d.get("coverage") or {}
        _record(state, "search", "done",
                reason=f"引证覆盖 {c.get('cited', 0)}/{c.get('total', 0)}，"
                       f"检索到段落 {c.get('with_retrieved', 0)} 条",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"search: {type(ex).__name__}: {ex}")
        state.setdefault("citations", {})
        _record(state, "search", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "repro_plan"
    return dict(state, next_step="repro_plan")


def node_repro_plan(state: AuditGraphState) -> dict:
    """「做」: 复现方案 + 可复现性评级（规则算分，逐条给出扣分理由）。"""
    t0 = _mono()
    try:
        from .repro_plan import build_repro_plan
        audit = state.get("audit")
        plan = build_repro_plan(audit, compute=state.get("compute"),
                               citations=state.get("citations"),
                               crosscheck=state.get("crosscheck"))
        d = plan.to_dict()
        d["markdown"] = plan.to_markdown()      # 供界面直接下载，无需二次渲染
        state["repro_plan"] = d
        if audit is not None:
            audit.repro_plan = d
            # 评级进 readiness（附加键，不动既有键，报告/界面都能取）
            try:
                audit.readiness["repro_grade"] = plan.grade
            except Exception:      # noqa: BLE001
                pass
        g = d.get("grade") or {}
        _record(state, "repro_plan", "done",
                reason=f"可复现性评级 {g.get('level', '—')}（{g.get('score', '—')}/100），"
                       f"扣分项 {len(g.get('deductions') or [])} 个",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"repro_plan: {type(ex).__name__}: {ex}")
        state.setdefault("repro_plan", {})
        _record(state, "repro_plan", "failed", str(ex), seconds=_mono() - t0)
    state["next_step"] = "loop"
    return dict(state, next_step="loop")


def node_loop(state: AuditGraphState) -> dict:
    """迭代回边: 对不上 → 定位根因 → 修正 → 再验证，**有界 ≤3 轮**。

    硬约束: 「修正」是**内存里的反事实推演**，绝不写文件、绝不改用户代码；
    补丁草案沿用 patch.py 的"只生成不落盘"。
    """
    t0 = _mono()
    g, reason = gate("loop", state)
    if g != "pass":
        _record(state, "loop", "skipped", reason, seconds=_mono() - t0, gate=g)
        state["next_step"] = "llm_analyze"
        return dict(state, next_step="llm_analyze")
    try:
        from .loop import run_loop
        audit = state.get("audit")
        rep = run_loop(audit, code_path=state.get("code_path", ""),
                       max_rounds=int(state.get("loop_rounds") or 3))
        d = rep.to_dict()
        d["markdown"] = rep.to_markdown()
        state["loop_result"] = d
        if audit is not None:
            audit.loop = d
        _record(state, "loop", "done",
                reason=f"{len(d.get('rounds') or [])} 轮内收敛 {d.get('resolved', 0)} 条，"
                       f"升级人工确认 {len(d.get('unresolved') or [])} 条",
                seconds=_mono() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"loop: {type(ex).__name__}: {ex}")
        state.setdefault("loop_result", {})
        _record(state, "loop", "failed", str(ex), seconds=_mono() - t0)
    # 2026-10-25 自主回头：原来这里写死 selfcheck（一条单向边）。
    # 现在「要不要再走一遍、换什么值再算」交给模型定，七道闸门验证（见 autonomy.py）；
    # 结论仍由 loop.run_loop 重新算 —— 模型说回头，改不了任何一条判定。
    return dict(state, next_step=_autonomy_next(state))


def _autonomy_next(state: AuditGraphState) -> str:
    """`loop` 之后往哪走：收尾（selfcheck）还是自主回头（compute）。

    ⚠ 只有 `auto_loop=True` 才问模型；关着时行为与改造前**逐字一致**。
    ⚠ 回填放在问模型之前：上一次回头到底换没换来新证据，要先算清楚再决定要不要再来一次。
    """
    decs = state.get("auto_decisions")
    if not isinstance(decs, list):
        decs = []
        state["auto_decisions"] = decs
    lr = dict(state.get("loop_result") or {})
    if decs and lr:
        try:
            from .autonomy import record_outcome
            record_outcome(decs, int(lr.get("resolved") or 0))
        except Exception as ex:            # noqa: BLE001
            state.setdefault("errors", []).append(
                f"autonomy.outcome: {type(ex).__name__}: {ex}")
    if not state.get("auto_loop"):
        state["next_step"] = "llm_analyze"
        return "llm_analyze"
    try:
        from .autonomy import decide
        dec = decide(lr, round_no=int(state.get("auto_round") or 0) + 1,
                     api_key=state.get("api_key", ""),
                     force_mock=bool(state.get("force_mock")),
                     base_url=state.get("llm_base_url", ""),
                     model=state.get("llm_model", ""))
    except Exception as ex:                # noqa: BLE001
        state.setdefault("errors", []).append(
            f"autonomy: {type(ex).__name__}: {ex}")
        state["next_step"] = "llm_analyze"
        return "llm_analyze"
    decs.append(dec.to_dict())
    # 同步挂到 loop_result 上 —— 它和 audit.loop 是同一个 dict 对象，
    # 于是报告与界面无需新增字段就能读到这几次决策（L8：只加键、不改旧键）。
    try:
        state["loop_result"]["auto_decisions"] = decs
    except Exception:                      # noqa: BLE001
        pass
    if dec.action == "retry":
        state["auto_round"] = int(state.get("auto_round") or 0) + 1
        state["auto_assume"] = dict(dec.assume or {})
        nxt = dec.target or "compute"
    else:
        state["auto_assume"] = {}
        nxt = "llm_analyze"
    _append_auto_md(state, decs)
    state["next_step"] = nxt
    return nxt


def _append_auto_md(state: AuditGraphState, decs: list) -> None:
    """把决策留痕追加进闭环那份 markdown —— 界面下载件与正文口径同源。"""
    try:
        from .autonomy import to_markdown
        md = to_markdown(decs)
        lr = state.get("loop_result")
        if not md or not isinstance(lr, dict):
            return
        # 重画而不是叠加：每回头一次就会调到这里，直接追加会把同一段写两遍
        base = str(lr.get("markdown") or "")
        lr["markdown"] = base.split("\n## 自主回头")[0].rstrip() + "\n" + md
    except Exception:                      # noqa: BLE001
        pass


# ---------------------------------------------------------------- 图

def build_graph(*, multagent: bool = False) -> Any:
    """构建 LangGraph 状态机。

    需要 langgraph 包（项目 .venv 已装）。

    multagent=True 时，analyze_code 之后接多智能体对撞区（roles/collider/...），
    而不是经典的 align。build_multagent_graph() 就是 multagent=True 的包装。
    """
    if not _HAS_LANGGRAPH:
        raise RuntimeError("未安装 langgraph（pip install langgraph）")

    graph = StateGraph(AuditGraphState)

    graph.add_node("start", node_start)
    graph.add_node("assemble_context", node_assemble_context)
    graph.add_node("plan", node_plan)

    # 10 个工具 -> 10 个节点
    for tool in TOOLS:
        graph.add_node(tool.key, _make_tool_node(tool.key))

    graph.add_node("finalize", node_finalize)

    # 边: 线性主链（下一步由 state["next_step"] 决定）
    graph.add_edge("start", "assemble_context")
    graph.add_edge("assemble_context", "plan")
    graph.add_edge("plan", "parse_paper")

    # 条件边: 每个节点执行完后，由 next_step 字段指向下一个
    def _route(state: AuditGraphState) -> str:
        nxt = state.get("next_step")
        if nxt == END:
            return END
        return nxt

    for tool in TOOLS:
        graph.add_conditional_edges(tool.key, _route)
    graph.add_conditional_edges("finalize", _route)

    graph.set_entry_point("start")
    return graph.compile()


# ---------------------------------------------------------------- 多智能体图（阶段四）

def build_multagent_graph(*, checkpointer: Optional[Any] = None,
                          interrupt_before: Optional[list] = None,
                          multagent: bool = True,
                          loop: bool = False) -> Any:
    """构建多智能体状态机（阶段四），可选挂上阶段五的闭环扩展链。

    结构（multagent=True, loop=False）:
      start -> assemble_context -> plan -> parse_paper -> fill_missing ->
      analyze_images -> analyze_code -> roles -> collider -> arbiter ->
      redteam -> report_findings -> selfcheck -> llm_analyze -> rootcause ->
      patch -> report -> finalize

    loop=True 时在 report_findings 与 selfcheck 之间插入阶段五的四个节点:
      report_findings -> compute -> search -> repro_plan -> loop -> selfcheck
    这四个节点是纯规则、零成本、离线可跑，**且不占进度槽**（与 roles/collider 同等
    待遇），因此开关它不会改变界面进度口径，也不影响既有验证脚本。

    说明:
      - roles 是三 Agent 的入口: analyst / verifier / numeric 各写各的槽
        （各只看自己该看的材料，隔离是代码层面强制的）。
      - collider / arbiter / redteam 全部是纯规则、不调模型，因此可做消融对照。
      - report_findings 把仲裁后的冲突转成 Finding，交给下游 selfcheck /
        rootcause / patch / report 消费（它们只认 audit.findings）。
      - 每个节点仍过闸门（_PRECONDITIONS），产物不齐就 blocked 并记录理由。
      - 失败不吞: 节点抛异常 -> 记入 state.errors -> 走到 finalize。

    checkpoint:
      把 MemorySaver 当 checkpointer 传进去，调用方给 thread_id 就能断点续跑。
      interrupt_before 可指定在哪些节点执行前中断（比如停在 "collider" 看对撞结果）。
    """
    if not _HAS_LANGGRAPH:
        raise RuntimeError("未安装 langgraph（pip install langgraph）")

    graph = StateGraph(AuditGraphState)

    graph.add_node("start", node_start)
    graph.add_node("assemble_context", node_assemble_context)
    graph.add_node("plan", node_plan)

    # 10 个工具 -> 10 个节点
    for tool in TOOLS:
        graph.add_node(tool.key, _make_tool_node(tool_key=tool.key, multagent=multagent))

    if multagent:
        # 多智能体图把 analyze_code 之后的流向改成对撞区（roles/collider/...），
        # 而不是经典的 align。align 节点仍注册着（被 report_findings 绕过）。
        graph.add_node("roles", node_roles)
        graph.add_node("collider", node_collider)
        graph.add_node("arbiter", node_arbiter)
        graph.add_node("redteam", node_redteam)
        graph.add_node("report_findings", node_report_findings)

    if loop:
        # 阶段五: 搜 / 算 / 做 + 迭代回边（纯规则，零成本）
        graph.add_node("compute", node_compute)
        graph.add_node("search", node_search)
        graph.add_node("repro_plan", node_repro_plan)
        graph.add_node("loop", node_loop)

    graph.add_node("finalize", node_finalize)

    # 边的规则（注意: 同一个节点绝不能同时加硬边和条件边，否则 LangGraph 当成两条
    # 并行路径，fan-in 时报 "Can receive only one value per step"）:
    #   硬边   —— 只在链头用: start -> assemble_context -> plan -> parse_paper
    #            这三个不会跳过，硬边就够了。
    #   条件边 —— 所有可能跳过的节点用 next_step 决定下一个。
    graph.add_edge("start", "assemble_context")
    graph.add_edge("assemble_context", "plan")
    graph.add_edge("plan", "parse_paper")

    # 条件边: 工具节点与角色节点都由 next_step 决定下一个。
    def _route(state: AuditGraphState):
        nxt = state.get("next_step")
        if nxt == END:
            return [END]
        return [nxt]

    for tool in TOOLS:
        graph.add_conditional_edges(tool.key, _route)
    if multagent:
        graph.add_conditional_edges("roles", _route)
        graph.add_conditional_edges("collider", _route)
        graph.add_conditional_edges("arbiter", _route)
        graph.add_conditional_edges("redteam", _route)
        graph.add_conditional_edges("report_findings", _route)
    if loop:
        for _k in ("compute", "search", "repro_plan", "loop"):
            graph.add_conditional_edges(_k, _route)
    graph.add_conditional_edges("finalize", _route)

    graph.set_entry_point("start")
    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def run_audit_multagent(
    pdf_path: str,
    code_path: str,
    user_results: Optional[dict] = None,
    api_key: Optional[str] = None,
    *,
    llm_base_url: str = "",
    llm_model: str = "",
    force_mock: bool = False,
    progress: Optional[Callable[[int, str], None]] = None,
    vision_kwargs: Optional[dict] = None,
    vision_max_regions: Optional[int] = None,
    enable_agent: bool = True,
    agent_budget: Optional[dict] = None,
    llm_planner: bool = False,
    llm_patch: bool = False,
    needs_index: Optional[bool] = None,
    snapshot: Optional[tuple] = None,
    session_fp: str = "",
    role_llm: bool = False,
    loop: bool = False,
    loop_rounds: int = 3,
    auto_loop: bool = False,
    thread_id: Optional[str] = None,
    interrupt_before: Optional[list] = None,
    judge_llm: bool = False,
    experiment_llm: bool = False,
) -> tuple:
    """多智能体版审计（与 pipeline.run_audit / run_audit_graph 同源签名）。

    返回 (AuditResult, Markdown 报告)。与经典版、骨架版同源，因此三版可消融对比:
      同数据跑 classic / graph / multagent，结论必须一致，只改调度方式。

    参数:
      role_llm: 对撞区角色是否用模型做「规范化」（默认 False = 全规则版）。
                开启后角色只多做一件事: 把自己那一侧的值规范化成可比对的标记
                （论文写"余弦退火"、代码写 CosineAnnealingLR -> 都归到 cosine）。
                抽取/对撞/仲裁/红队全部不动，隔离也不破（只喂本侧材料）。
                没 key / 调用失败一律回退规则版，并如实记在 role_llm_note 里。
      loop: 是否启用阶段五（M2）的闭环扩展链（默认 False）:
            「算」小规模复算 → 「搜」引证检索 → 「做」复现方案 + 可复现性评级
            → 迭代回边（≤ loop_rounds 轮，内存反事实推演，绝不写文件）。
            四个节点全是纯规则、零成本、离线可跑，不开 key 也能用。
      loop_rounds: 迭代回边轮数上限（默认 3，硬上限 3 —— 有界自主防死循环）。
      thread_id: 传给 checkpoint（MemorySaver），缺省为本次输入的 hash，
                 这样同一次输入走同一个检查点，可续跑。
      interrupt_before: 在哪些节点执行前中断（如 ["collider"]），
                        返回 None 作为"暂停"标记。
    """
    from .envfile import get as _env_get
    from .memory import fingerprint as _mem_fp
    from .preflight import check_inputs as _check_inputs

    # 2026-10-30：路径填错在**进图之前**就报，不许让它变成「某一步 failed」。
    # 图里每个节点都有 except Exception 兜底（单个工具出错不拖垮全流程），
    # 于是坏路径会被记成一次失败、继续往下跑、最后照样出结论 —— 用户看到
    # 「论文有、代码没有」，会当成事实。校验必须放在最外侧。
    _check_inputs(pdf_path, code_path)

    state = {
        "pdf_path": pdf_path, "code_path": code_path,
        "user_results": user_results,
        "api_key": (api_key or _env_get("LLM_API_KEY")
                    or _env_get("DEEPSEEK_API_KEY") or ""),
        "llm_base_url": llm_base_url, "llm_model": llm_model,
        "force_mock": force_mock,
        "vision_kwargs": _vision_kwargs_default(vision_kwargs),
        "vision_max_regions": vision_max_regions,
        "enable_agent": enable_agent,
        "agent_budget": dict(agent_budget or {}),
        "llm_planner": llm_planner, "llm_patch": llm_patch,
        # 归因运行事实的初值（由 llm_analyze 节点回写，selfcheck 读它写进 readiness）
        "llm_mode": "", "llm_real_calls": 0, "llm_cache_hits": 0,
        "llm_error": "", "llm_shard": {}, "llm_model_used": "",
        "needs_index": needs_index, "snapshot": snapshot,
        "session_fp": session_fp,
        "fingerprint": _mem_fp(pdf_path, code_path),
        "paper": None, "code": None, "audit": None, "report": "",
        "context_screen": None, "decisions": [], "plan_dict": {}, "records": [], "errors": [],
        "current_step": "", "next_step": "",
        "paper_view": {}, "code_view": {}, "numeric_view": {},
        "collisions": [], "arbitrations": [], "red_team": [], "crosscheck": [],
        "role_llm": bool(role_llm), "role_llm_note": "",
        # 2026-10-25: 模型先判、程序后验（默认 False = 图上行为完全不变）
        "judge_llm": bool(judge_llm),
        # 2026-10-26 自主模式：② 实验站模型挑工具（默认 False = 行为完全不变）
        "experiment_llm": bool(experiment_llm),
        # 阶段二 2A/2B/2D: ① 站的假设卡（含推演统计）与 ② 站的实验卡 —— finalize 挂到 audit 上
        "hypotheses": [], "all_hypotheses": [], "hypothesis_total": 0, "hypothesis_stats": {},
        "experiments": [], "experiment_total": 0, "experiment_stats": {},
        "figure_clues": [], "figure_total": 0, "figure_stats": {},
        "chem_items": [], "chem_total": 0, "chem_stats": {},
        "chem_fallback": False,
        # 阶段五（M2 闭环扩展）: 默认关 —— 不动既有路径与界面进度口径
        "loop_enabled": bool(loop), "loop_rounds": int(loop_rounds),
        "compute": {}, "citations": {}, "repro_plan": {}, "loop_result": {},
        # 2026-10-25 自主回头（默认 False = 图的形状与行为完全不变）
        "auto_loop": bool(auto_loop), "auto_round": 0,
        "auto_decisions": [], "auto_assume": {},
    }
    graph = build_multagent_graph(
        checkpointer=_make_checkpointer(),
        interrupt_before=interrupt_before,
        loop=bool(loop),
    )
    config = {"configurable": {"thread_id": thread_id or _hash_state(state)}}
    final = state
    # 进度口径与 pipeline.run_audit 一致: (已完成工具序号, 环节名)。
    # 注意: 只有工具节点占进度槽 —— 角色节点（roles/collider/...）没有对应的
    # STEPS 项，把它们算进去会让进度条跑到 100% 以上。
    _title = {t.key: t.title for t in TOOLS}
    _last_key = ""
    for event in graph.stream(state, config, stream_mode="values"):
        final = event
        if progress:
            _done = [r for r in (event.get("records") or [])
                     if r.get("key") in _title and r.get("status") != "skipped"]
            if _done and _done[-1]["key"] != _last_key:
                _last_key = _done[-1]["key"]
                progress(len(_done) - 1, _title[_last_key])
    audit = final.get("audit")
    return audit, final.get("report", "")


def _make_checkpointer() -> Any:
    """创建 checkpoint（只有内存版，进程内有效）。

    为什么要显式给 serde:
      checkpoint 会把 state 序列化存下来，而 state 里带着本项目的自定义类型
      （Evidence / PaperInfo / CodeInfo / ContextScreen）。
      langgraph 默认对"未登记类型"只是警告，但**未来版本会直接拒绝反序列化**
      —— 到那时断点/续跑会突然坏掉，且报错信息跟本项目的代码毫无关联。
      所以这里提前把这些类型登记进 allowlist，把"未来的坑"变成"现在就合规"。
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        # (模块名, 类名) —— 必须与 models.py / assembler.py 里的真实位置一致
        allowed = [
            ("paper2lab.models", "Evidence"),
            ("paper2lab.models", "PaperInfo"),
            ("paper2lab.models", "CodeInfo"),
            ("paper2lab.models", "Finding"),
            ("paper2lab.models", "AuditResult"),
            # 下面四个是 Finding / AuditResult 的字段枚举（Status / Confidence /
            # Risk / Category）。它们会随 state 一起进 checkpoint；漏登记时
            # langgraph 目前只告警并把枚举降级成普通值 —— 续跑会把它们还原成
            # 字符串，且未来版本会直接拒绝反序列化。故一并登记，把坑提前填掉。
            ("paper2lab.models", "Status"),
            ("paper2lab.models", "Confidence"),
            ("paper2lab.models", "Risk"),
            ("paper2lab.models", "Category"),
            ("paper2lab.assembler", "ContextScreen"),
            ("paper2lab.assembler", "AlignedParam"),
            ("paper2lab.assembler", "MemorySide"),
            ("paper2lab.collider", "Collision"),
            ("paper2lab.collider", "Arbitration"),
            ("paper2lab.graph", "NodeRecord"),
            # 阶段二 2A: state["hypotheses"]（以及 AuditResult.hypotheses）里存的是
            # Hypothesis 对象, 它会随 state 一起进 checkpoint。漏登记时 langgraph
            # 目前只告警并把对象降级, 未来版本会直接拒绝反序列化 —— 与上面
            # Evidence/PaperInfo 同一条理由, 一并把坑填掉。
            ("paper2lab.hypothesis", "Hypothesis"),
            # 阶段二 2D: state["experiments"] 里存的是 ExperimentCard —— 同上, 一并登记。
            ("paper2lab.experiment", "ExperimentCard"),
            # 阶段二 2E: state["figure_clues"] 里存的是 FigureClue —— 同上, 一并登记。
            ("paper2lab.multimodal", "FigureClue"),
        ]
        serde = JsonPlusSerializer(allowed_msgpack_modules=allowed)
        return MemorySaver(serde=serde)
    except Exception:      # noqa: BLE001
        try:
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        except Exception:      # noqa: BLE001
            return None


def _hash_state(state: dict) -> str:
    """给一次审计生成一个稳定的 thread_id（同输入同 id，可续跑）。"""
    import hashlib
    blob = json.dumps({
        "pdf": state.get("pdf_path"),
        "code": state.get("code_path"),
        "results": state.get("user_results"),
        "mock": state.get("force_mock"),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]

def run_audit_graph(
    pdf_path: str,
    code_path: str,
    user_results: Optional[dict] = None,
    api_key: Optional[str] = None,
    *,
    llm_base_url: str = "",
    llm_model: str = "",
    force_mock: bool = False,
    progress: Optional[Callable[[int, str], None]] = None,
    vision_kwargs: Optional[dict] = None,
    vision_max_regions: Optional[int] = None,
    enable_agent: bool = True,
    agent_budget: Optional[dict] = None,
    llm_planner: bool = False,
    llm_patch: bool = False,
    needs_index: Optional[bool] = None,
    snapshot: Optional[tuple] = None,
    session_fp: str = "",
) -> tuple:
    """骨架版审计（与 pipeline.run_audit 同源签名）。

    返回 (AuditResult, Markdown 报告)，与经典版可互换，因此能做消融对比:
      同数据跑 classic / graph 两版，结论必须一致（只改调度方式）。
    """
    from .envfile import get as _env_get
    from .memory import fingerprint as _mem_fp
    from .preflight import check_inputs as _check_inputs

    # 2026-10-30：同 run_audit_multagent —— 坏路径必须在进图之前报。
    _check_inputs(pdf_path, code_path)

    state = {
        "pdf_path": pdf_path, "code_path": code_path,
        "user_results": user_results,
        "api_key": (api_key or _env_get("LLM_API_KEY")
                    or _env_get("DEEPSEEK_API_KEY") or ""),
        "llm_base_url": llm_base_url, "llm_model": llm_model,
        "force_mock": force_mock,
        "vision_kwargs": _vision_kwargs_default(vision_kwargs),
        "vision_max_regions": vision_max_regions,
        "enable_agent": enable_agent,
        "agent_budget": dict(agent_budget or {}),
        "llm_planner": llm_planner, "llm_patch": llm_patch,
        # 归因运行事实的初值（由 llm_analyze 节点回写，selfcheck 读它写进 readiness）
        "llm_mode": "", "llm_real_calls": 0, "llm_cache_hits": 0,
        "llm_error": "", "llm_shard": {}, "llm_model_used": "",
        "needs_index": needs_index, "snapshot": snapshot,
        "session_fp": session_fp,
        "fingerprint": _mem_fp(pdf_path, code_path),
        "paper": None, "code": None, "audit": None, "report": "",
        "context_screen": None, "decisions": [], "plan_dict": {}, "records": [], "errors": [],
        "current_step": "", "next_step": "",
        # 阶段二 2A/2B/2D: ① 站的假设卡（含推演统计）与 ② 站的实验卡 —— finalize 挂到 audit 上
        "hypotheses": [], "all_hypotheses": [], "hypothesis_total": 0, "hypothesis_stats": {},
        "experiments": [], "experiment_total": 0, "experiment_stats": {},
        "figure_clues": [], "figure_total": 0, "figure_stats": {},
        "chem_items": [], "chem_total": 0, "chem_stats": {},
        "chem_fallback": False,
    }
    graph = build_graph()
    final = state
    for event in graph.stream(state, stream_mode="values"):
        final = event
        if progress:
            # 每个 update 对应一步完成
            pass
    audit = final.get("audit")
    return audit, final.get("report", "")