"""Paper2Lab → 司南 agent 协议的适配器。

`Paper2LabAgent` 满足 SciEvalKit `AgentBase` 的三要件（name / model_version / run），
于是可以被 `agent_runner.py` 或本地 `scieval_compat.run_agent_eval` 批量评测。

**本模块只做"取路径 → 调审计 → 定型答案"，不含任何审计逻辑** ——
业务逻辑仍只在 pipeline / graph 里，保证"网页、MCP、评测"三条路口径同源。

两条**互相正交**的对照轴（都只为量"加了多少能力、结论动了没有"）：

  一、模式轴（`mode`）—— 换**谁来调度**
      classic     经典流水线（pipeline.run_audit）
      multagent   多智能体对撞（graph.run_audit_multagent）
      closeloop   多智能体 ＋ 搜/算/做闭环与迭代回边

  二、档位轴（`tier`，阶段三 3C）—— 换**开了几项能力**（只跑经典链）
      parser      只开解析员：读材料 ＋ 直接把两侧摆一起看差异（**不算数、不下判**）
      calc        加计算员：另开 ② 实验站（提假设卡 ＋ 设计实验 ＋ 真算一遍，只出数）
      review      全开（加审查员）：另开 ③ 验证站（证据自证 ＋ 判定）

为什么档位的分界是"算"和"判"，而不是"读"和"比"：
    没有**比对**就没有任何审计结论（那是底座，不是一项可开可关的能力），
    所以三档都跑「读 ＋ 比」。真正能一级一级加上去的只有两件:
    ② 能不能**把数算出来**、③ 能不能**对结论下判定**。
    这样对比才说明问题 —— 召回与误报在三档之间**应当不变**（加能力不改结论），
    变的是「算出多少条数」和「判了多少条」。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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


# ---------------------------------------------------------------- 档位（3C）
@dataclass(frozen=True)
class Tier:
    """一档能力配置。

    `skip` 是**真的不跑**（不是把输出藏起来）—— 走 runner 的消融通道，
    执行记录里会留下 `status=skipped` 与统一理由，可查可核。

    `metrics` 是**本档测得出来**的指标。测不出来的绝不给 0：
    没开审查员就没有「证据站不站得住」这回事，报 0 等于把"没测"说成"很差"。
    """
    key: str
    title: str
    stations: Tuple[str, ...]
    skip: Tuple[str, ...]
    metrics: Tuple[str, ...]
    desc: str


#: 三档（顺序即能力递增；报告按这个顺序并列，别按字典序）
TIERS: Tuple[Tier, ...] = (
    Tier("parser", "只开解析员", ("① 假设",),
         ("hypothesize", "design_experiment", "selfcheck"),
         ("recall", "decoy_fp", "n_findings"),
         "读材料 ＋ 直接把两侧摆一起看差异；不算数、不下判"),
    Tier("calc", "加计算员", ("① 假设", "② 实验"),
         ("selfcheck",),
         ("recall", "decoy_fp", "n_findings", "calc_coverage"),
         "在档一之上开 ② 实验站：提假设卡、设计实验并真算一遍（只出数，仍不下判）"),
    Tier("review", "全开（加审查员）", ("① 假设", "② 实验", "③ 验证", "④ 迭代"),
         (),
         ("recall", "precision", "f1", "selfcheck_rate", "decoy_fp",
          "n_findings", "calc_coverage", "judged"),
         "再加 ③ 验证站：证据自证与判定，于是「准 / 自证率」这两项才有数"),
)

TIER_BY_KEY: Dict[str, Tier] = {t.key: t for t in TIERS}

assert len(TIERS) == 3, "三档消融 —— 增删档位必须同步改报告与文档口径"
assert len(TIER_BY_KEY) == 3, "档位键必须唯一"
# 防呆: 能力必须**单调递增**（后一档的指标必须是前一档的超集、跳过的工具必须更少），
# 否则"加了一档反而少了东西"，消融表就是错的。
for _a, _b in zip(TIERS, TIERS[1:]):
    assert set(_a.metrics) <= set(_b.metrics), f"档位 {_b.key} 的指标应覆盖 {_a.key}"
    assert set(_b.skip) <= set(_a.skip), f"档位 {_b.key} 跳过的工具应不多于 {_a.key}"


def tier_keys() -> Tuple[str, ...]:
    """档位键全集（CLI 校验用，避免手抄）。"""
    return tuple(t.key for t in TIERS)


def _cut(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v)
    return s if len(s) <= _MAX_VAL else s[: _MAX_VAL - 1] + "…"


def _calc_keys(audit: Any) -> set:
    """② 实验站**真算出了数**的参数键。

    只认 `actual is not None` —— 卡片存在但没算出值（缺输入 / 推不了）不算，
    这正是"不造数"的体现：没算出来就是没算出来。
    """
    out = set()
    for c in getattr(audit, "experiments", None) or []:
        if getattr(c, "actual", None) is not None:
            out.add(getattr(c, "key", ""))
    return out


def _serialize(audit: Any, sid: str, elapsed: float,
               tier: Optional[Tier] = None) -> Dict[str, Any]:
    """把 AuditResult 定型成判分器认识的作答结构。"""
    from ..models import Status

    p_keys = sorted(audit.paper.params) if getattr(audit, "paper", None) else []
    c_keys = sorted(audit.code.params) if getattr(audit, "code", None) else []

    # 没开审查员时 `verified` 一律写 None（未判），**绝不沿用默认为 True 的初值** ——
    # 否则"根本没做自证"的答卷会被读成"全部自证通过"。
    judged = bool(tier is None or "selfcheck_rate" in tier.metrics)
    calc_on = bool(tier is not None and "calc_coverage" in tier.metrics)
    calc_keys = _calc_keys(audit) if calc_on else set()

    inc = []
    for f in getattr(audit, "findings", []) or []:
        if f.status == Status.CONSISTENT:
            continue
        row = {
            "param_key": f.param_key,
            "display": f.display_name,
            "status": f.status.value,
            "risk": f.risk.value,
            "verified": bool(f.verified) if judged else None,
            "paper": _cut(f.paper.raw_value if f.paper else None),
            "code": _cut(f.code.raw_value if f.code else None),
            "extra": len(f.extra or []),
        }
        if calc_on:
            row["calc"] = f.param_key in calc_keys
        inc.append(row)

    cov = getattr(audit, "coverage", {}) or {}
    sc = getattr(audit, "selfcheck", {}) or {}
    ans: Dict[str, Any] = {
        "schema": ANSWER_SCHEMA,
        "sid": sid,
        "ok": True,
        "found_params": {"paper": p_keys, "code": c_keys},
        "inconsistencies": inc,
        "selfcheck": {
            "passed": int(sc.get("passed", 0) or 0),
            "checked": int(sc.get("checked", 0) or 0),
            "failed": int(sc.get("failed", 0) or 0),
            # 自检被档位跳过：如实标，判分器据此不产出「自证率」
            "skipped": bool(sc.get("skipped", False)),
        },
        "readiness": (getattr(audit, "readiness", {}) or {}).get("total"),
        "coverage": {
            "confidence_level": cov.get("confidence_level"),
            "confidence_score": cov.get("confidence_score"),
            "rejected": len(cov.get("rejected", []) or []),
        },
        # 各档的规模（不进均值，用来看"多开一档多做了什么"）
        "stats": {
            "n_params_paper": len(p_keys),
            "n_params_code": len(c_keys),
            "n_hypotheses": int(getattr(audit, "hypothesis_total", 0) or 0),
            "n_experiments": int(getattr(audit, "experiment_total", 0) or 0),
            "n_calc": sum(1 for i in inc if i.get("calc")),
            "n_judged": sum(1 for i in inc if i.get("verified") is not None),
        },
        "elapsed_s": round(elapsed, 3),
    }
    if tier is not None:
        ans["tier"] = tier.key
        ans["tier_title"] = tier.title
        ans["stations_open"] = list(tier.stations)
        ans["metrics_available"] = list(tier.metrics) + ["parse_ok"]
    return ans


class Paper2LabAgent(AgentBase):
    """把一次审计包成"一次作答"。"""

    name = "paper2lab"

    def __init__(self, mode: str = "classic", force_mock: bool = True,
                 model_version: Optional[str] = None, tier: Optional[str] = None,
                 **kwargs):
        m = _MODE_ALIAS.get(str(mode).lower())
        if m is None:
            raise ValueError(f"未知模式: {mode}（可选 classic / multagent / closeloop）")
        if tier is not None and str(tier).lower() not in (None, "", "full", "all"):
            tk = str(tier).lower()
            if tk not in TIER_BY_KEY:
                raise ValueError(
                    f"未知档位: {tier}（可选 {' / '.join(tier_keys())}）")
            if m != "classic":
                # 档位只在经典链上做: 图版是硬边串起来的固定流程，逐件跳过会破图。
                raise ValueError("档位消融只跑 classic 模式（图版按固定流程串边，不可逐件跳）")
            t = TIER_BY_KEY[tk]
        else:
            t = None
        ver = f"{AGENT_VERSION}-{t.key}-{m}" if t is not None else f"{AGENT_VERSION}-{m}"
        super().__init__(model_version=model_version or ver)
        self.mode = m
        self.tier = t
        self.force_mock = bool(force_mock)

    @property
    def skip_tools(self) -> Tuple[str, ...]:
        return tuple(self.tier.skip) if self.tier is not None else ()

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
                    pdf, code, res, force_mock=self.force_mock,
                    skip_tools=self.skip_tools)
            else:  # multagent / closeloop
                from .. import graph as _graph
                audit, _elapsed = _graph.run_audit_multagent(
                    pdf, code, res, force_mock=self.force_mock,
                    loop=(self.mode == "closeloop"), loop_rounds=3)
        except Exception as exc:  # 失败一律结构化返回，不向评测器抛异常
            return EvalResult(False, json.dumps(
                {"schema": ANSWER_SCHEMA, "sid": sid, "ok": False,
                 "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))

        ans = _serialize(audit, sid, time.time() - t0, self.tier)
        ans["mode"] = self.mode
        return EvalResult(True, json.dumps(ans, ensure_ascii=False))
