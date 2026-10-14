"""「准 / 全 / F1 / 稳」四维判分 —— 审计评测的度量核心。

六个问题，六个指标（都是 0~1 的比值，便于均值聚合）：

  全（recall）      金标准里"该抓到的参数"抓到了几成
  准（precision）   报出来的问题里，证据站得住（自证通过）的有几成
  F1                准与全的调和平均
  自证率            全部结论的证据自证通过率（审计的审计）
  诱饵误报          被误升级为不一致的"合法差异"条数（**越低越好**，单独看）
  解析成功          作答能被解析成结构化结果（0/1）

要求达标的只是"作答可解析"；其余一律**如实报数**，不做通过/不通过判定 ——
评测的产出是一个可横向比较的刻度，不是开关。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def evaluate_answer(
    *,
    expect: Iterable[str],
    decoy: Iterable[str],
    answer: Dict[str, Any],
) -> Dict[str, float]:
    """对单份作答打分。

    参数
    ----
    expect : 金标准"该抓到的参数键"
    decoy  : "不该被报为不一致"的诱饵参数键
    answer : agent 作答（已在 agent_adapter 里定型的结构化 dict）
    """
    expect = list(expect)
    decoy = list(decoy)

    found = set(answer.get("found_params", {}).get("paper", []))
    found |= set(answer.get("found_params", {}).get("code", []))

    hit = [k for k in expect if k in found]
    miss = [k for k in expect if k not in found]
    recall = (len(hit) / len(expect)) if expect else 0.0

    inc: List[Dict[str, Any]] = list(answer.get("inconsistencies", []))
    n_inc = len(inc)
    n_verified = sum(1 for i in inc if i.get("verified"))
    # 一条不一致都没报时，precision 记为 1.0（没冤枉人），但由 recall/自证率另作刻画
    precision = (n_verified / n_inc) if n_inc else 1.0

    decoy_fp = [k for k in decoy if any(i.get("param_key") == k for i in inc)]

    sc = answer.get("selfcheck", {}) or {}
    checked = int(sc.get("checked", 0) or 0)
    passed = int(sc.get("passed", 0) or 0)
    selfcheck_rate = (passed / checked) if checked else 0.0
    # 错位值防线：被规则拒收（疑似错位）的抽取数 —— 拒得越准，精度越稳
    rejected = float((answer.get("coverage", {}) or {}).get("rejected", 0) or 0)

    out: Dict[str, float] = {
        "selfcheck_rate": round(selfcheck_rate, 4),
        "decoy_fp": float(len(decoy_fp)),
        "n_findings": float(n_inc),
        "rejected": rejected,
        "parse_ok": 1.0,
    }
    # 诱饵样本没有"期望项"，只考「别乱报」—— 不产出召回/精度，避免污染整体均值
    if expect:
        out.update({
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(_f1(precision, recall), 4),
            "n_miss": float(len(miss)),
        })
    return out


#: 明细里额外记录（不进均值）的键
DETAIL_KEYS = ("missing", "decoy_hit", "sid", "kind")


def detail_for(expect: Iterable[str], decoy: Iterable[str],
               answer: Dict[str, Any]) -> Dict[str, Any]:
    """给报告用的逐样本明细（不进聚合均值）。"""
    expect = list(expect)
    decoy = list(decoy)
    found = set(answer.get("found_params", {}).get("paper", []))
    found |= set(answer.get("found_params", {}).get("code", []))
    inc = list(answer.get("inconsistencies", []))
    return {
        "missing": [k for k in expect if k not in found],
        "decoy_hit": [k for k in decoy if any(i.get("param_key") == k for i in inc)],
        "hit": [k for k in expect if k in found],
    }
