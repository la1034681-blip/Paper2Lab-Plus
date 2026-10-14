"""审计评测入口 —— 对齐书生·司南（SciEvalKit）agent_runner 的跑法。

用法示例
--------
    # 单模式跑全套考卷
    .venv\\Scripts\\python.exe tools/run_eval.py --agent-mode classic

    # 三模式同跑并对比（消融：只加能力、不改结论）
    .venv\\Scripts\\python.exe tools/run_eval.py --agent-mode classic,multagent,closeloop

    # 只跑推理不判分 / 复用上次结果 / 另存报告
    ... --mode infer
    ... --reuse
    ... --report C:\\Users\\lijiahui\\Desktop\\Paper2Lab测试集\\eval_report.md

口径说明见 docs/eval-spec.md。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 脚本在 tools/ 下，需把项目根加回 sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from paper2lab.evalsuite import goldset                       # noqa: E402
from paper2lab.evalsuite.audit_bench import AuditBenchDataset, DATASET_NAME  # noqa: E402
from paper2lab.evalsuite.agent_adapter import Paper2LabAgent  # noqa: E402
from paper2lab.evalsuite.scieval_compat import SCHEMA_SOURCE, run_agent_eval  # noqa: E402

_METRIC_ORDER = [("recall", "召回"), ("precision", "精度"), ("f1", "F1"),
                 ("selfcheck_rate", "自证率"), ("decoy_fp", "诱饵误报"),
                 ("n_findings", "结论数")]


def _fmt(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"


def run_one(agent_mode: str, work_dir: str, mode: str, reuse: bool,
            only: list = None) -> dict:
    if only:
        samples = [s for s in goldset.list_resolved() if s.sid in only]
    else:
        samples = None
    ds = AuditBenchDataset(samples)
    if len(ds) == 0:
        raise RuntimeError("没有可跑的考卷：请确认测试集目录存在（见 goldset.REAL_ROOT）")
    agent = Paper2LabAgent(mode=agent_mode)
    t0 = time.time()
    summary = run_agent_eval(agent, ds, work_dir=work_dir, mode=mode, reuse=reuse)
    summary["_elapsed_s"] = round(time.time() - t0, 2)
    summary["_details"] = {idx: ds.details.get(idx, {}) for idx in range(len(ds))}
    summary["_samples"] = [s.sid for s in ds.samples]
    return summary


def print_block(agent_mode: str, summary: dict) -> list:
    lines = [f"===== 评测汇总 · 模式 {agent_mode} ====="]
    sids = summary["_samples"]
    lines.append(f"{'样本':<14}{'召回':>7}{'精度':>7}{'F1':>7}{'自证率':>8}{'诱饵误报':>9}{'结论数':>7}")
    for i, sid in enumerate(sids):
        det = summary["_details"].get(i, {})
        sc = det.get("score", {})
        cells = [_fmt(sc[k]) if k in sc else "—"
                 for k in ("recall", "precision", "f1", "selfcheck_rate", "decoy_fp")]
        n = float(sc.get("n_findings", 0.0))
        lines.append(f"{sid:<14}{cells[0]:>7}{cells[1]:>7}{cells[2]:>7}"
                     f"{cells[3]:>8}{cells[4]:>9}{_fmt(n):>7}")
    lines.append("")
    overall = [f"{label} {_fmt(summary.get(k, 0.0))}" for k, label in _METRIC_ORDER]
    lines.append("--- 整体均值 ---")
    lines.append("  " + " ｜ ".join(overall))
    if summary["_details"]:
        miss = {d.get("sid", "?"): d.get("missing", []) for d in summary["_details"].values() if d.get("missing")}
        if miss:
            lines.append("--- 未抓到的期望项 ---")
            for sid, m in miss.items():
                lines.append(f"  {sid}: {', '.join(m)}")
    print("\n".join(lines))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper2Lab 审计评测（对齐司南 SciEvalKit）")
    ap.add_argument("--agent-mode", default="classic",
                    help="classic / multagent / closeloop，可逗号分隔多值对比")
    ap.add_argument("--mode", default="all", choices=["all", "infer", "eval"])
    ap.add_argument("--work-dir", default="eval_output")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--report", default="", help="另存 markdown 报告路径")
    ap.add_argument("--only", default="", help="只跑指定样本 id（逗号分隔），便于快速回归")
    args = ap.parse_args()
    _only = [x.strip() for x in args.only.split(",") if x.strip()]

    modes = [m.strip() for m in args.agent_mode.split(",") if m.strip()]
    print(f"评测契约来源: {SCHEMA_SOURCE}")
    print(f"考卷: {DATASET_NAME} | 可用 {len(goldset.list_resolved())}/{len(goldset.GOLD_SAMPLES)} 套")
    missing = goldset.missing()
    if missing:
        print(f"  本机缺失（自动跳过）: {', '.join(missing)}")
    print()

    md = [f"# Paper2Lab 审计评测报告", "",
          f"- 契约来源: {SCHEMA_SOURCE}", f"- 考卷: {DATASET_NAME}",
          f"- 金标准依据: {goldset.EVIDENCE_NOTE}", ""]
    all_summaries = {}
    for m in modes:
        s = run_one(m, args.work_dir, args.mode, args.reuse, _only)
        all_summaries[m] = s
        block = print_block(m, s)
        print()
        md += ["```text"] + block + ["```", ""]

    # 多模式对比：同一批考察项是否"结论不变"
    if len(modes) > 1:
        md += ["## 多模式对比", "", "| 指标 | " + " | ".join(modes) + " |",
               "| --- | " + " | ".join("---" for _ in modes) + " |"]
        for k, label in _METRIC_ORDER:
            md.append(f"| {label} | " + " | ".join(_fmt(all_summaries[m].get(k, 0.0)) for m in modes) + " |")
        md.append("")

    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(md), encoding="utf-8")
        print(f"报告已写入: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
