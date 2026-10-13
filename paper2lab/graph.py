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
    Analyst     只读 job.paper
    Verifier    只读 job.code
    Numeric     只看两侧已经抽好的值
  即使按顺序执行，每个角色也物理上看不见另外两侧的材料。

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
import time

from .assembler import assemble
from .planner import make_plan
from .runner import _store_session_memory
from .toolbox import STEPS, TOOLS, Job
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
    role_llm: bool                          # 对撞区角色是否用模型做规范化（**默认 False**）
    role_llm_note: str                      # 模型这次到底有没有参与（如实留痕，不静默）
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
    "fill_missing": {"paper"},
    "analyze_code": {},
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
    t0 = time.time()
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
        _record(state, "assemble_context", "done", seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"assemble_context: {type(ex).__name__}: {ex}")
        _record(state, "assemble_context", "failed", str(ex),
                seconds=time.time() - t0)
    state["next_step"] = "plan"
    return dict(state, next_step="plan")


def node_plan(state: AuditGraphState) -> dict:
    """排计划: 由 planner.make_plan 排（规则版兜底，AI 可选）。"""
    t0 = time.time()
    try:
        job = _job_from_state(state)
        plan = make_plan(job, plan_mode=True, use_llm=job.llm_planner)
        state["decisions"] = [
            {"tool": d.tool, "title": d.title, "action": d.action,
             "reason": d.reason, "source": getattr(d, "source", "rule")}
            for d in plan.decisions
        ]
        _record(state, "plan", "done", seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"plan: {type(ex).__name__}: {ex}")
        _record(state, "plan", "failed", str(ex),
                seconds=time.time() - t0)
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
        needs_index=state.get("needs_index"),
        snapshot=state.get("snapshot"),
        paper=state.get("paper"),
        code=state.get("code"),
        audit=state.get("audit"),
    )


# 多智能体图的工具节点顺序覆盖（analyze_code 之后不是 align，而是 roles）。
# 定义成模块级常量（而不是闭包参数），因为 _make_tool_node 在 build_graph()
# 里先被调用，闭包会捕获到 None。
_MULTAGENT_AFTER_MAP = {"analyze_code": "roles", "align": "report_findings"}


def _make_tool_node(tool_key: str, *, multagent: bool = False):
    """给一个工具生成"过闸门 -> 执行 -> 留痕"的节点函数。

    multagent=True 时，用 _MULTAGENT_AFTER_MAP 覆盖默认顺序
    （多智能体图里 analyze_code 之后接 roles）。
    """
    tool = next((t for t in TOOLS if t.key == tool_key), None)
    if tool is None:
        raise KeyError(f"未知工具: {tool_key}")

    def _node(state: AuditGraphState) -> dict:
        t0 = time.time()
        g, reason = gate(tool_key, state)
        if g != "pass":
            _record(state, tool.key, "skipped", reason,
                    seconds=time.time() - t0, gate=g,
                    checkpoints=tool.checkpoints)
            state["next_step"] = _next_after(tool_key, state,
                                             _MULTAGENT_AFTER_MAP if multagent else None)
            return state
        job = _job_from_state(state)
        try:
            # 复用工箱的执行体（与经典版同一份代码，不漂移）
            tool.run(job, lambda n, suffix="": None)
            # 写回状态
            if tool.key == "parse_paper":
                state["paper"] = job.paper
            elif tool.key == "analyze_code":
                state["code"] = job.code
            elif tool.key == "align":
                state["audit"] = job.audit
            elif tool.key == "report":
                state["report"] = job.report
            # 每次执行后把证据写成线索（与 runner 同源）
            if job.fingerprint and getattr(job, "audit", None) is not None:
                _store_session_memory(job)
            _record(state, tool.key, "done",
                    seconds=time.time() - t0, gate=g,
                    checkpoints=tool.checkpoints)
        except Exception as ex:      # noqa: BLE001
            state["errors"].append(f"{tool.key}: {type(ex).__name__}: {ex}")
            _record(state, tool.key, "failed", str(ex),
                    seconds=time.time() - t0, gate=g,
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
            "notes": [],
            "plan": {},
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
        }
    state["current_step"] = "finalize"
    return dict(state, next_step=END)


# ══════ 多智能体角色节点（阶段四）══════

def node_roles(state: AuditGraphState) -> dict:
    """角色对撞区第一步: 三个 Agent 各只看自己该看的材料，独立抽取。

    隔离是代码层面强制的（agent_roles.py 的 __call__ 只读自己那一个字段），
    不是靠提示词求它别看。
    """
    t0 = time.time()
    g, reason = gate("roles", state)
    if g != "pass":
        _record(state, "roles", "skipped", reason, seconds=time.time() - t0,
                gate=g)
        state["next_step"] = "collider"
        return dict(state, next_step="collider")
    try:
        from .agent_roles import Analyst, CodeVerifier, NumericVerifier, RoleLLM
        job = _job_from_state(state)
        # 模型版（**默认关**）: 角色只多做"规范化"这一步，抽取/比对全不动。
        # 不启用时 RoleLLM 为 None，Analyst/CodeVerifier 走的仍是规则版 ——
        # 与阶段四原版逐字节一致（这是"换调度不改结论"的硬保证）。
        _llm = None
        if state.get("role_llm"):
            _llm = RoleLLM(api_key=state.get("api_key", ""),
                           base_url=state.get("llm_base_url", ""),
                           model=state.get("llm_model", ""),
                           force_mock=bool(state.get("force_mock")))
        # 每个 Agent 只拿到自己该看的视图（job 上挂载的局部视图，不是全文）
        _analyst, _verifier = Analyst(_llm), CodeVerifier(_llm)
        paper_view = _analyst(job)
        code_view = _verifier(job)
        # 数值验证员只看两侧已经抽好的值，不做任何抽取
        job.paper_view = paper_view
        job.code_view = code_view
        numeric_view = NumericVerifier()(job)
        state["paper_view"] = paper_view
        state["code_view"] = code_view
        state["numeric_view"] = numeric_view
        # 模型这次到底参没参与 —— 如实记进留痕，不静默（"没参与"本身也要说）
        _note = "；".join(n for n in (_analyst.last_note, _verifier.last_note) if n)
        state["role_llm_note"] = _note
        _record(state, "roles", "done", reason=_note, seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"roles: {type(ex).__name__}: {ex}")
        _record(state, "roles", "failed", str(ex), seconds=time.time() - t0)
    state["next_step"] = "collider"
    return dict(state, next_step="collider")


def node_collider(state: AuditGraphState) -> dict:
    """对撞: 把三方的结构化结论并排比对，产出冲突清单（纯规则，不调模型）。"""
    t0 = time.time()
    try:
        from .collider import collide
        collisions = collide(state.get("paper_view", {}),
                             state.get("code_view", {}),
                             state.get("numeric_view"))
        state["collisions"] = [c.to_dict() for c in collisions]
        _record(state, "collider", "done", seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"collider: {type(ex).__name__}: {ex}")
        _record(state, "collider", "failed", str(ex), seconds=time.time() - t0)
    state["next_step"] = "arbiter"
    return dict(state, next_step="arbiter")


def node_arbiter(state: AuditGraphState) -> dict:
    """仲裁: 不是简单投票，而是按证据强度裁决（规则版兜底）。"""
    t0 = time.time()
    try:
        from .collider import Arbitration, Collision, arbitrate_all
        collisions = [Collision.from_dict(d) for d in state.get("collisions", [])]
        arbitrations = arbitrate_all(collisions)
        state["arbitrations"] = [a.to_dict() for a in arbitrations]
        _record(state, "arbiter", "done", seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"arbiter: {type(ex).__name__}: {ex}")
        _record(state, "arbiter", "failed", str(ex), seconds=time.time() - t0)
    state["next_step"] = "redteam"
    return dict(state, next_step="redteam")


def node_redteam(state: AuditGraphState) -> dict:
    """红队: 专挑结论漏洞（对抗式验证，降幻觉）。"""
    t0 = time.time()
    try:
        from .collider import Collision, RedTeamNote, red_team
        collisions = [Collision.from_dict(d) for d in state.get("collisions", [])]
        notes = red_team(collisions, state.get("paper_view", {}), state.get("code_view", {}))
        state["red_team"] = [n.to_dict() for n in notes]
        _record(state, "redteam", "done", seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"redteam: {type(ex).__name__}: {ex}")
        _record(state, "redteam", "failed", str(ex), seconds=time.time() - t0)
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
    t0 = time.time()
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
                seconds=time.time() - t0)
    except Exception as ex:      # noqa: BLE001
        state["errors"].append(f"report_findings: {type(ex).__name__}: {ex}")
        state.setdefault("crosscheck", [])
        _record(state, "report_findings", "failed", str(ex), seconds=time.time() - t0)
    # 对撞区与 align 都跑完了，交给下游自检/归因
    state["next_step"] = "selfcheck"
    return dict(state, next_step="selfcheck")


def _ev(value: str, source: str, doc: str, channel: str) -> Optional[Evidence]:
    """把一个 Agent 视图格转成 Evidence（供 Finding 使用）。"""
    if not value:
        return None
    return Evidence(source=channel, location=source, snippet=doc[:120], raw_value=value)


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
                          multagent: bool = True) -> Any:
    """构建多智能体状态机（阶段四）。

    结构:
      start -> assemble_context -> plan -> parse_paper -> fill_missing ->
      analyze_images -> analyze_code -> roles -> collider -> arbiter ->
      redteam -> report_findings -> selfcheck -> llm_analyze -> rootcause ->
      patch -> report -> finalize

    说明:
      - roles 是三 Agent 的入口: analyst / verifier / numeric 各写各的槽
        （各只看自己该看的材料，隔离是代码层面强制的）。
      - collider / arbiter / redteam 全部是纯规则、不调模型，因此可做消融对照。
      - report_findings 把仲裁后的冲突转成 Finding，交给下游 selfcheck /
        rootcause / patch / report 消费（它们只认 audit.findings）。
      - align 被跳过了: 对撞口已经做了比对，再跑 align 会覆盖对撞结果。
        所以 report_findings 直接指向 selfcheck。
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
    thread_id: Optional[str] = None,
    interrupt_before: Optional[list] = None,
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
      thread_id: 传给 checkpoint（MemorySaver），缺省为本次输入的 hash，
                 这样同一次输入走同一个检查点，可续跑。
      interrupt_before: 在哪些节点执行前中断（如 ["collider"]），
                        返回 None 作为"暂停"标记。
    """
    from .envfile import get as _env_get
    from .memory import fingerprint as _mem_fp

    state = {
        "pdf_path": pdf_path, "code_path": code_path,
        "user_results": user_results,
        "api_key": (api_key or _env_get("LLM_API_KEY")
                    or _env_get("DEEPSEEK_API_KEY") or ""),
        "llm_base_url": llm_base_url, "llm_model": llm_model,
        "force_mock": force_mock,
        "vision_kwargs": vision_kwargs or {},
        "vision_max_regions": vision_max_regions,
        "enable_agent": enable_agent,
        "agent_budget": dict(agent_budget or {}),
        "llm_planner": llm_planner, "llm_patch": llm_patch,
        "needs_index": needs_index, "snapshot": snapshot,
        "session_fp": session_fp,
        "fingerprint": _mem_fp(pdf_path, code_path),
        "paper": None, "code": None, "audit": None, "report": "",
        "context_screen": None, "decisions": [], "records": [], "errors": [],
        "current_step": "", "next_step": "",
        "paper_view": {}, "code_view": {}, "numeric_view": {},
        "collisions": [], "arbitrations": [], "red_team": [], "crosscheck": [],
        "role_llm": bool(role_llm), "role_llm_note": "",
    }
    graph = build_multagent_graph(
        checkpointer=_make_checkpointer(),
        interrupt_before=interrupt_before,
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
            ("paper2lab.assembler", "ContextScreen"),
            ("paper2lab.assembler", "AlignedParam"),
            ("paper2lab.assembler", "MemorySide"),
            ("paper2lab.collider", "Collision"),
            ("paper2lab.collider", "Arbitration"),
            ("paper2lab.graph", "NodeRecord"),
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

    state = {
        "pdf_path": pdf_path, "code_path": code_path,
        "user_results": user_results,
        "api_key": (api_key or _env_get("LLM_API_KEY")
                    or _env_get("DEEPSEEK_API_KEY") or ""),
        "llm_base_url": llm_base_url, "llm_model": llm_model,
        "force_mock": force_mock,
        "vision_kwargs": vision_kwargs or {},
        "vision_max_regions": vision_max_regions,
        "enable_agent": enable_agent,
        "agent_budget": dict(agent_budget or {}),
        "llm_planner": llm_planner, "llm_patch": llm_patch,
        "needs_index": needs_index, "snapshot": snapshot,
        "session_fp": session_fp,
        "fingerprint": _mem_fp(pdf_path, code_path),
        "paper": None, "code": None, "audit": None, "report": "",
        "context_screen": None, "decisions": [], "records": [], "errors": [],
        "current_step": "", "next_step": "",
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