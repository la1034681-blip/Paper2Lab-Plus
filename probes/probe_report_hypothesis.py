"""探针：报告里的「① 假设站」长什么样（人眼确认排版，不参与体检）。

跑：.venv/Scripts/python.exe -X utf8 probes/probe_report_hypothesis.py
产物：.trash_tmp/report_with_hypotheses_demo.md（不进工作区根目录）

为什么要有这个探针：报告这一节是**给评委看的那一份**。断言只能保证"章节在、
字段在", 保证不了"读起来像人话、不像调试输出"——排版这件事得用眼睛过一遍。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.pipeline import run_audit  # noqa: E402

DEMO = ROOT / "demo"
OUT = ROOT / ".trash_tmp" / "report_with_hypotheses_demo.md"
OUT.parent.mkdir(exist_ok=True)

audit, report_md = run_audit(str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
                             force_mock=True)
OUT.write_text(report_md, encoding="utf-8")

cards = list(getattr(audit, "hypotheses", None) or [])
print("假设卡 %d 条 / 共 %d 条" % (len(cards), audit.hypothesis_total))
for c in cards:
    print("   #%d %-18s %-6s %s" % (c.rank, c.key, c.kind,
                                    "需实机" if c.needs_rerun else "静态可判"))

_start = report_md.index("## ① 假设站")
_end = report_md.index("## 审计概览")
print("\n" + "=" * 70)
print(report_md[_start:_end].rstrip())
print("=" * 70)
print("\n完整报告已落盘: %s" % OUT)
