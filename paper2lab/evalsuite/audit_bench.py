"""金标准集 → 司南 dataset 协议。

`AuditBenchDataset` 满足 SciEvalKit 的 dataset 四要件：
    dataset_name / __len__ / build_agent_sample(idx) / score_agent_sample(idx, answer)

于是 `scieval_compat.run_agent_eval(Paper2LabAgent(...), AuditBenchDataset())`
就能一刻不停地跑完一整套考卷并给出分数。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import goldset
from .scieval_compat import EvalSample
from .scoring import evaluate_answer, detail_for

#: 考卷名（同时是结果目录名）
DATASET_NAME = "Paper2Lab-AuditBench"

#: 发给 agent 的题目模板
_PROMPT = (
    "审计题 {sid}：请对下面的论文与代码做一致性审计，"
    "识别论文声明与代码实现不一致的关键参数，并为每条结论给出证据出处。\n"
    "论文: {pdf}\n代码: {code}"
)


class AuditBenchDataset:
    """自建审计考卷（5 套真实论文 + 演示 + 诱饵）。"""

    dataset_name = DATASET_NAME

    def __init__(self, samples: Optional[List[goldset.ResolvedSample]] = None):
        self.samples: List[goldset.ResolvedSample] = (
            list(samples) if samples is not None else goldset.list_resolved()
        )
        #: 最近一次打分的明细（供报告使用，不进聚合）
        self.details: Dict[int, dict] = {}

    # ---------------------------------------------------------- 协议实现
    def __len__(self) -> int:
        return len(self.samples)

    def build_agent_sample(self, idx: int) -> EvalSample:
        s = self.samples[idx]
        return EvalSample(
            prompt=_PROMPT.format(sid=s.sid, pdf=s.pdf, code=s.code),
            files={"pdf": s.pdf, "code": s.code, "results": s.results or ""},
            metadata=s.to_meta(),
        )

    def score_agent_sample(self, idx: int, final_answer: str, **judge_kwargs) -> Dict[str, float]:
        s = self.samples[idx]
        try:
            ans = json.loads(final_answer) if isinstance(final_answer, str) else final_answer
        except Exception:
            self.details[idx] = {"sid": s.sid, "kind": s.kind, "error": "答案不是合法 JSON"}
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                    "selfcheck_rate": 0.0, "decoy_fp": 0.0,
                    "n_findings": 0.0, "n_miss": float(len(s.expect)), "parse_ok": 0.0}

        if not isinstance(ans, dict):
            self.details[idx] = {"sid": s.sid, "kind": s.kind, "error": "答案不是对象"}
            return {"recall": 0.0, "precision": 0.0, "f1": 0.0,
                    "selfcheck_rate": 0.0, "decoy_fp": 0.0,
                    "n_findings": 0.0, "n_miss": float(len(s.expect)), "parse_ok": 0.0}

        score = evaluate_answer(expect=s.expect, decoy=s.decoy, answer=ans)
        det = detail_for(s.expect, s.decoy, ans)
        det.update({"sid": s.sid, "kind": s.kind, "note": s.note,
                    "found_paper": ans.get("found_params", {}).get("paper", []),
                    "found_code": ans.get("found_params", {}).get("code", []),
                    "n_findings": score["n_findings"],
                    # 3C：档位与规模随明细一起留底 —— 报告要能说清"这一档多做了什么"
                    "tier": ans.get("tier", ""),
                    "stats": dict(ans.get("stats", {}) or {}),
                    "score": dict(score)})
        self.details[idx] = det
        return score

    # ---------------------------------------------------------- 辅助
    def describe(self) -> dict:
        return goldset.describe()
