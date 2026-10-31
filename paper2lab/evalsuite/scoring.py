"""「准 / 全 / F1 / 稳」四维判分 —— 审计评测的度量核心。

六个问题，六个指标（都是 0~1 的比值，便于均值聚合）：

  全（recall）      金标准里"该抓到的参数"抓到了几成
  准（precision）   报出来的问题里，证据站得住（自证通过）的有几成
  F1                准与全的调和平均
  自证率            全部结论的证据自证通过率（审计的审计）
  诱饵误报          被误升级为不一致的"合法差异"条数（**越低越好**，单独看）
  解析成功          作答能被解析成结构化结果（0/1）

  阶段三 3C 另加两项，用来量"多开一档能力多做了什么"（不开放能力时**不产出**）：
  算出比例          有 ② 站实际值的结论占几成（没开计算员 = 未测，不是 0）
  已判条数          被 ③ 站真正下过判定的结论数（没开审查员 = 未测）

**测不出来的指标一律不产出、绝不填 0** —— 0 会被聚合成"表现极差"，
而真相是"这一档根本没开这项能力"。作答里的 `metrics_available` 是这份白名单；
缺这个字段时按**全量**处理（兼容旧作答与既有测试）。
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

#: 全部指标（作答未标 metrics_available 时按这个全集算，兼容旧作答）
ALL_METRICS = frozenset({"recall", "precision", "f1", "selfcheck_rate",
                         "decoy_fp", "n_findings", "calc_coverage", "judged"})


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

    # 白名单：作答自己声明这一档能测什么；没声明 = 全量（旧作答 / 既有测试照旧）
    avail = set(answer.get("metrics_available") or ALL_METRICS)

    found = set(answer.get("found_params", {}).get("paper", []))
    found |= set(answer.get("found_params", {}).get("code", []))

    hit = [k for k in expect if k in found]
    miss = [k for k in expect if k not in found]
    recall = (len(hit) / len(expect)) if expect else 0.0

    inc: List[Dict[str, Any]] = list(answer.get("inconsistencies", []))
    n_inc = len(inc)
    # `verified is True` 才算数 —— None 表示"本档没判"，不能被当成通过
    n_verified = sum(1 for i in inc if i.get("verified") is True)
    # 一条不一致都没报时，precision 记为 1.0（没冤枉人），但由 recall/自证率另作刻画
    precision = (n_verified / n_inc) if n_inc else 1.0

    decoy_fp = [k for k in decoy if any(i.get("param_key") == k for i in inc)]

    sc = answer.get("selfcheck", {}) or {}
    checked = int(sc.get("checked", 0) or 0)
    passed = int(sc.get("passed", 0) or 0)
    selfcheck_rate = (passed / checked) if checked else 0.0
    # 错位值防线：被规则拒收（疑似错位）的抽取数 —— 拒得越准，精度越稳
    rejected = float((answer.get("coverage", {}) or {}).get("rejected", 0) or 0)

    out: Dict[str, float] = {"parse_ok": 1.0}
    if "decoy_fp" in avail:
        out["decoy_fp"] = float(len(decoy_fp))
    if "n_findings" in avail:
        out["n_findings"] = float(n_inc)
    out["rejected"] = rejected
    # 诱饵样本没有"期望项"，只考「别乱报」—— 不产出召回/精度，避免污染整体均值
    if expect and "recall" in avail:
        out["recall"] = round(recall, 4)
        out["n_miss"] = float(len(miss))
    if expect and "precision" in avail:
        out["precision"] = round(precision, 4)
        out["f1"] = round(_f1(precision, recall), 4)
    if "selfcheck_rate" in avail:
        out["selfcheck_rate"] = round(selfcheck_rate, 4)
    # 3C 两项：只有本档真开了对应能力才产出
    if "calc_coverage" in avail:
        n_calc = sum(1 for i in inc if i.get("calc"))
        out["calc_coverage"] = round((n_calc / n_inc) if n_inc else 0.0, 4)
    if "judged" in avail:
        out["judged"] = float(sum(1 for i in inc if i.get("verified") is not None))
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
