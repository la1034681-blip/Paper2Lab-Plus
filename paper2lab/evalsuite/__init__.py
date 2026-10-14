"""evalsuite —— 阶段七·评测打磨。

把 Paper2Lab 的审计能力接到书生·司南（SciEvalKit）的评测口径上：
  - scieval_compat.py  与司南完全一致的评测契约（零重依赖）
  - goldset.py         自建审计金标准集（真实论文+代码 → 该抓/不该抓）
  - audit_bench.py     金标准集 → 司南 dataset 协议
  - agent_adapter.py   Paper2Lab → 司南 agent 协议
  - scoring.py         「准 / 全 / F1 / 稳」四维判分

设计边界（诚实声明）：
  司南评的是「通用科学能力」（会不会做科研题），其榜单**不含审计类任务**。
  因此本套评测是**自建考卷、契约对齐**：接口按司南规范实现，将来可直接挂进
  它的 agent_runner 跑；但"审计抓错准不准"的分，来自我们自己的金标准集。
"""
from __future__ import annotations

__all__ = ["scieval_compat", "goldset", "audit_bench", "agent_adapter", "scoring"]
