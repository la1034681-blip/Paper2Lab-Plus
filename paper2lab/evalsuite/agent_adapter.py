"""Paper2Lab → 司南 agent 协议的适配器。

`Paper2LabAgent` 满足 SciEvalKit `AgentBase` 的三要件（name / model_version / run），
于是可以被 `agent_runner.py` 或本地 `scieval_compat.run_agent_eval` 批量评测。

**本模块只做"取路径 → 调审计 → 定型答案"，不含任何审计逻辑** ——
业务逻辑仍只在 pipeline / graph 里，保证"网页、MCP、评测"三条路口径同源。

三种模式（用于消融对照）：
    classic     经典流水线（pipeline.run_audit）
    multagent   多智能体对撞（graph.run_audit_multagent）
    closeloop   多智能体 + 搜/算/做闭环与迭代回边
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from .scieval_compat import AgentBase, EvalResult, EvalSample
from .. import pipeline as _pipeline

#: 作答结构版本（判分器按此解析；改动要同步 scoring 与测试）
ANSWER_SCHEMA = "p2l-audit-answer-v1"

#: agent 版本（写进结果目录，便于横向比较不同版本）
AGENT_VERSION = "v1"

_MODE_ALIAS = {
    "classic": "classic",
    "multagent": "multagent",
    "multi": "multagent",
    "closeloop": "closeloop",
    "loop": "closeloop",
}

#: 单值最大长度（作答会落盘，别把正文整段带过去）
_MAX_VAL = 160


def _cut(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s if len(s) <= _MAX_VAL else s[: _MAX_VAL - 1] + "…"


def _serialize(audit: Any, sid: str, elapsed: float) -> Dict[str, Any]:
    """把 AuditResult 定型成判分器认识的作答结构。"""
    from ..models import Status

    p_keys = sorted(audit.paper.params) if getattr(audit, "paper", None) else []
    c_keys = sorted(audit.code.params) if getattr(audit, "code", None) else []

    inc = []
    for f in getattr(audit, "findings", []) or []:
        if f.status == Status.CONSISTENT:
            continue
        inc.append({
            "param_key": f.param_key,
            "display": f.display_name,
            "status": f.status.value,
            "risk": f.risk.value,
            "verified": bool(f.verified),
            "paper": _cut(f.paper.raw_value if f.paper else None),
            "code": _cut(f.code.raw_value if f.code else None),
            "extra": len(f.extra or []),
        })

    cov = getattr(audit, "coverage", {}) or {}
    return {
        "schema": ANSWER_SCHEMA,
        "sid": sid,
        "ok": True,
        "found_params": {"paper": p_keys, "code": c_keys},
        "inconsistencies": inc,
        "selfcheck": {
            "passed": int(getattr(audit, "selfcheck", {}).get("passed", 0) or 0),
            "checked": int(getattr(audit, "selfcheck", {}).get("checked", 0) or 0),
            "failed": int(getattr(audit, "selfcheck", {}).get("failed", 0) or 0),
        },
        "readiness": (getattr(audit, "readiness", {}) or {}).get("total"),
        "coverage": {
            "confidence_level": cov.get("confidence_level"),
            "confidence_score": cov.get("confidence_score"),
            "rejected": len(cov.get("rejected", []) or []),
        },
        "elapsed_s": round(elapsed, 3),
    }


class Paper2LabAgent(AgentBase):
    """把一次审计包成"一次作答"。"""

    name = "paper2lab"

    def __init__(self, mode: str = "classic", force_mock: bool = True,
                 model_version: Optional[str] = None, **kwargs):
        m = _MODE_ALIAS.get(str(mode).lower())
        if m is None:
            raise ValueError(f"未知模式: {mode}（可选 classic / multagent / closeloop）")
        super().__init__(model_version=model_version or f"{AGENT_VERSION}-{m}")
        self.mode = m
        self.force_mock = bool(force_mock)

    def run(self, sample: EvalSample) -> EvalResult:
        pdf = sample.files.get("pdf") or ""
        code = sample.files.get("code") or ""
        res = sample.files.get("results") or None
        sid = sample.metadata.get("sid", "")

        if not pdf or not code:
            return EvalResult(False, json.dumps(
                {"schema": ANSWER_SCHEMA, "sid": sid, "ok": False,
                 "error": "题目缺少 pdf / code 路径"}, ensure_ascii=False))

        t0 = time.time()
        try:
            if self.mode == "classic":
                audit, _report = _pipeline.run_audit(
                    pdf, code, res, force_mock=self.force_mock)
            else:  # multagent / closeloop
                from .. import graph as _graph
                audit, _elapsed = _graph.run_audit_multagent(
                    pdf, code, res, force_mock=self.force_mock,
                    loop=(self.mode == "closeloop"), loop_rounds=3)
        except Exception as exc:  # 失败一律结构化返回，不向评测器抛异常
            return EvalResult(False, json.dumps(
                {"schema": ANSWER_SCHEMA, "sid": sid, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))

        ans = _serialize(audit, sid, time.time() - t0)
        ans["mode"] = self.mode
        return EvalResult(True, json.dumps(ans, ensure_ascii=False))
