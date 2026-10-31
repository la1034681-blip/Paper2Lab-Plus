"""执行器（第二期第一批）——按计划调用工具, 并如实记录每一步的去向。

记录三件事: 执行了什么、跳过了什么、为什么跳过。这是界面「执行计划」面板的数据源。

两个硬约束:
1. **跳过不等于缺产物**: 工具被跳过时, 由 `_fill_empty` 补一份语义正确的空产物,
   保证后面的工具不会因为"东西不在"而崩, 也不会凭空多出一条误导性结论;
2. **失败即中止**: 工具抛异常时如实记录后向上抛出, 与改造前 run_audit 的行为一致,
   不做静默吞掉。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .models import CodeInfo
from .planner import make_plan
from .toolbox import (ABLATION_LOCKED, STEPS, TOOL_BY_KEY, TOOLS, Job,
                      attach_chemistry, attach_experiments, attach_figures,
                      attach_hypotheses, ensure_chemistry)

#: 消融档位跳过的统一理由 —— 执行记录里要一眼看出「不是调度员判断不值得跑」，
#: 而是「评测时人为少开了一项能力」。措辞同步进报告与评测报告，别各处另写一句。
ABLATION_SKIP_REASON = "消融档位：本次评测**有意**未开这项能力（3C）"


@dataclass
class StepRecord:
    key: str
    title: str
    status: str            # done | skipped | failed
    reason: str = ""
    seconds: float = 0.0
    checkpoints: tuple = ()


@dataclass
class ExecutionLog:
    """一次执行的完整记录。"""
    mode: str = "classic"
    steps: list = field(default_factory=list)     # [StepRecord]
    notes: list = field(default_factory=list)     # 面向用户的范围说明
    plan: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "steps": [{"key": s.key, "title": s.title, "status": s.status,
                       "reason": s.reason, "seconds": round(s.seconds, 2),
                       "checkpoints": list(s.checkpoints)} for s in self.steps],
            "notes": list(self.notes),
            "plan": dict(self.plan),
            "seconds": round(self.seconds, 2),
            "ran": sum(1 for s in self.steps if s.status == "done"),
            "skipped": sum(1 for s in self.steps if s.status == "skipped"),
            "failed": sum(1 for s in self.steps if s.status == "failed"),
            "nodes_total": len(STEPS),
            "nodes_skipped": sum(len(s.checkpoints) for s in self.steps
                                 if s.status == "skipped"),
        }


def _fill_empty(job: Job, tool_key: str) -> None:
    """被跳过的工具留下的空产物 —— 语义要与"跑了但没结果"一致。"""
    if tool_key == "hypothesize":
        # 被跳过 = 没有论文（或材料太残缺到不值得提假设）。空列表的语义是
        # "这次没提出假设"，**不是**"提了但没提出来" —— 这两者在界面上必须说得清。
        job.hypotheses = []
        job.hypothesis_total = 0
        if job.audit is not None:
            job.audit.hypotheses = []
    elif tool_key == "design_experiment":
        # 被跳过 = 没有假设卡（或材料太残缺到不值得设计实验）。空列表的语义是
        # "这次没做实验"，**不是**"做了但一张卡都没出"。
        job.experiments = []
        job.experiment_total = 0
        if job.audit is not None:
            job.audit.experiments = []
    elif tool_key == "analyze_code":
        if job.code is None:
            job.code = CodeInfo()
    elif tool_key == "read_figures":
        # 被跳过 = 没有论文（或材料太残缺到不值得读图）。空列表的语义是
        # "本次没读到图表线索"，**不是**"读了但一条都没读到"。
        job.figure_clues = []
        job.figure_total = 0
        if job.audit is not None:
            job.audit.figure_clues = []
    elif tool_key == "rootcause":
        # 被跳过 = 没有代码包。空列表的语义正是"没有可追的项", 不是"追了没追到"。
        job.rootcause = []
        if job.audit is not None:
            job.audit.rootcause = []
    elif tool_key == "patch":
        job.patches = []
        if job.audit is not None:
            job.audit.patches = []
    elif tool_key == "selfcheck":
        # 阶段三 3C：自检被**档位**跳过时，必须留下"这次没判"的明确信号。
        # 为什么不能就这样放着: `Finding.verified` 的默认值是 True，
        # 不写这一笔的话，一份"根本没做自证"的答卷会被读成"全部自证通过"——
        # 那是把没测的东西报成满分, 比缺指标更坏。
        # 只加新键（skipped=True）＋ 计数归零，不改既有键语义（L8）。
        if job.audit is not None:
            job.audit.selfcheck = {"passed": 0, "checked": 0, "failed": 0,
                                   "skipped": True,
                                   "note": ABLATION_SKIP_REASON}
            job.audit.coverage = dict(getattr(job.audit, "coverage", {}) or {})
            job.audit.coverage["skipped"] = True
    elif tool_key == "analyze_images":
        if job.paper is not None:
            job.paper.image_params = {}
            job.paper.image_meta = {
                "regions": 0, "enabled": False, "mode": "skipped", "model": "",
                "kept": 0, "dropped": 0, "region_list": [],
                "notes": ["计划跳过：未配置读图通道，图片/扫描页未读取"],
            }


def run_job(job: Job, *, progress: Optional[Callable[[int, str], None]] = None) -> ExecutionLog:
    """按计划执行全部工具, 返回执行记录。

    2026-10-25 起**只有一条路**: 先排计划（模型能排就模型排, 否则规则版兜底）再执行。
    原先的"经典模式"（不看材料、固定全序全跑）已按用户口径删除 ——
    它只是把"没代码也扫一遍代码"这种无效动作固化下来，留着只会让结论口径分叉。
    """
    # 跨任务记忆(第二期第三批): 只读"线索"用于提示, 不改变任何执行决策
    if job.fingerprint:
        try:
            from .memory import hint_lines
            job.memory_lines = hint_lines(job.fingerprint)
        except Exception:      # noqa: BLE001
            job.memory_lines = []

    # 上下文装配器(阶段三第2步): 把会话级记忆按"论文侧 vs 代码侧"的对照结构
    #   装进上下文, 让模型一眼对齐"论文说 1e-4, 代码写 1e-3",
    #   而不是从两段混在一起的话里自己翻。纯函数, 在 make_plan 之前跑,
    #   因此 planner 的 notes 里就能用上。
    try:
        from .session import get as _get_session
        from .assembler import assemble
        _store = _get_session()
        if _store is not None:
            recall = _store.recall(job.fingerprint)
            job.context_screen = assemble(job, recall=recall)
    except Exception:      # noqa: BLE001
        job.context_screen = None

    # 快路径(阶段三第1步): 同会话内同一套输入第二次跑, 解析产物已缓存,
    #   直接复用, 不重复解析 PDF 也不重扫代码 AST。
    #   解析是纯函数, 结论与重跑完全一致, 只省本地 CPU 时间。
    if job.needs_index is False and job.snapshot is not None:
        job.paper, job.code = job.snapshot
        job.snapshot_reused = True

    plan = make_plan(job, use_llm=job.llm_planner)
    log = ExecutionLog(mode=plan.mode, notes=list(plan.notes),
                       plan=plan.to_dict())

    if plan.agent_budget:
        job.agent_budget = dict(plan.agent_budget)

    def _tick(name: str, suffix: str = "") -> None:
        if not progress:
            return
        try:
            idx = STEPS.index(name)
        except ValueError:
            return
        progress(idx, f"{name}{suffix}")

    action_of = {d.tool: d for d in plan.decisions}
    # 消融档位（3C）：人为少开的能力。放在循环之前校验 —— 档位写错要当场炸，
    # 不能等跑完半套才发现「跳的是不存在的工具」或「把底座跳了」。
    forced = tuple(job.skip_tools or ())
    for _k in forced:
        assert _k in TOOL_BY_KEY, f"消融档位里出现了不存在的工具键: {_k}"
        assert _k not in ABLATION_LOCKED, f"工具 {_k} 是消融受保护的底座，不许跳"
    forced_set = set(forced)
    if forced:
        log.notes.append(
            "🧪 消融档位：" + "、".join(TOOL_BY_KEY[k].title for k in forced)
            + " 本档**未开启**（如实记，不是没跑出来）")
    t_all = time.monotonic()

    for tool in TOOLS:
        if tool.key in forced_set:
            _fill_empty(job, tool.key)
            log.steps.append(StepRecord(
                key=tool.key, title=tool.title, status="skipped",
                reason=ABLATION_SKIP_REASON, checkpoints=tool.checkpoints))
            for cp in tool.checkpoints:
                _tick(cp, "（消融档位跳过）")
            continue

        decision = action_of.get(tool.key)

        if decision is not None and decision.action == "skip":
            _fill_empty(job, tool.key)
            log.steps.append(StepRecord(
                key=tool.key, title=tool.title, status="skipped",
                reason=decision.reason, checkpoints=tool.checkpoints))
            for cp in tool.checkpoints:
                _tick(cp, "（计划跳过）")
            continue

        t0 = time.time()
        try:
            tool.run(job, _tick)
        except Exception as ex:      # noqa: BLE001
            log.steps.append(StepRecord(
                key=tool.key, title=tool.title, status="failed",
                reason=f"{type(ex).__name__}: {ex}",
                seconds=time.time() - t0, checkpoints=tool.checkpoints))
            log.seconds = time.monotonic() - t_all
            job.execution_log = log.to_dict()
            if job.audit is not None:
                job.audit.execution_log = job.execution_log
            raise
        log.steps.append(StepRecord(
            key=tool.key, title=tool.title, status="done",
            seconds=time.time() - t0, checkpoints=tool.checkpoints))

    log.seconds = time.monotonic() - t_all
    job.execution_log = log.to_dict()
    if job.audit is not None:
        job.audit.execution_log = job.execution_log
        # 阶段二 2A：① 假设站的产物挂到 audit —— 与图版 node_finalize、与报告生成前
        # 走的是**同一个函数**（toolbox.attach_hypotheses），界面 / 报告只读
        # audit.hypotheses 一处，不分经典版 / 图版（分了就一定漂移）。
        attach_hypotheses(job.audit, job.hypotheses, job.hypothesis_total,
                          getattr(job, "derivation_note", None))
        # 阶段二 2D：② 实验站的产物同款挂法 —— 与图版 finalize、与报告生成前走同一个函数，
        # 界面 / 报告只读 audit.experiments 一处，不分经典版 / 图版。
        attach_experiments(job.audit, job.experiments, job.experiment_total,
                           getattr(job, "experiment_note", None))
        # 阶段二 2E：① 假设站「读」这一步的图表线索同款挂法。
        attach_figures(job.audit, job.figure_clues, job.figure_total,
                       getattr(job, "figure_note", None))
        # 阶段二 2E-化学：独立工具 chem_check 的产物同款挂法（同一函数, 不分经典/图版）。
        # ⚠ 挂之前必须 ensure_chemistry：模型**可以单独跳过**这一步（决策权在它），
        #   但不许结果静默消失 —— 跳过就补算，并标 fallback 如实写明「该步被跳过」。
        ensure_chemistry(job)
        attach_chemistry(job.audit, getattr(job, "chem_items", None),
                         getattr(job, "chem_total", 0),
                         getattr(job, "chem_note", None),
                         fallback=bool(getattr(job, "chem_fallback", False)))

    # 写入记忆: **只写线索**(哪些参数出现在哪些页/哪条通道/哪些已确认无解), 不写任何取值
    if job.fingerprint and job.audit is not None:
        try:
            from .memory import remember
            remember(job.audit, job.fingerprint)
        except Exception:      # noqa: BLE001
            pass
    # 会话级记忆(阶段三第2步): 把本次审计找到的证据写成线索, 供同一会话内
    #   下次跑时按"论文侧 vs 代码侧"对照读出。
    #   与 memory.remember 分工: 后者是跨会话的 JSON 线索摘要, 供提示用;
    #   这里是会话内的结构化线索, 供装配器用。
    if job.fingerprint and job.audit is not None:
        _store_session_memory(job)
    return log


def _store_session_memory(job: Job) -> None:
    """把审计发现里的证据线索写入会话记忆（只写线索，不写结论）。"""
    from .session import get as _get_session
    store = _get_session()
    if store is None:
        return
    fp = job.fingerprint
    for f in job.audit.findings:
        for channel, ev in (("paper", f.paper), ("code", f.code)):
            if ev is None or getattr(ev, "context_excluded", False):
                continue
            value = getattr(ev, "raw_value", None) or getattr(ev, "snippet", "")
            try:
                store.store_memory(fp, f.param_key, str(value), channel,
                                   getattr(ev, "location", ""),
                                   getattr(ev, "snippet", ""))
            except Exception:      # noqa: BLE001
                pass
