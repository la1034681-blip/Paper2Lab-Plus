"""审计评测入口 —— 对齐书生·司南（SciEvalKit）agent_runner 的跑法。

用法示例
--------
    # 单模式跑全套考卷
    .venv\\Scripts\\python.exe tools/run_eval.py --agent-mode classic

    # 三模式同跑并对比（消融：只换调度方式、不改结论）
    .venv\\Scripts\\python.exe tools/run_eval.py --agent-mode classic,multagent,closeloop

    # 三档同跑并对比（3C 消融：只加能力、不改结论）
    .venv\\Scripts\\python.exe tools/run_eval.py --tiers parser,calc,review

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
from paper2lab.evalsuite.agent_adapter import (Paper2LabAgent, TIER_BY_KEY,  # noqa: E402
                                               tier_keys)
from paper2lab.evalsuite.audit_bench import AuditBenchDataset, DATASET_NAME  # noqa: E402
from paper2lab.evalsuite.scieval_compat import SCHEMA_SOURCE, run_agent_eval  # noqa: E402

_METRIC_ORDER = [("recall", "召回"), ("precision", "精度"), ("f1", "F1"),
                 ("selfcheck_rate", "自证率"), ("decoy_fp", "诱饵误报"),
                 ("n_findings", "结论数"), ("calc_coverage", "算出比例"),
                 ("judged", "已判条数")]

#: 档位表额外两列（来自作答的 stats，不进均值而是逐样本平均）
_STAT_ORDER = [("n_hypotheses", "假设卡"), ("n_experiments", "实验卡")]


def _fmt(v: float) -> str:
    return f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"


def _cell(summary: dict, key: str) -> str:
    """这一档有没有这个指标 —— 没有就报「—」，**不填 0**（没测 ≠ 很差）。"""
    return _fmt(summary[key]) if key in summary else "—"


def _w(s: str) -> int:
    """显示宽度：中文按两格算（否则表格在记事本里会歪）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in str(s))


def _pad(s, width: int, right: bool = False) -> str:
    s = str(s)
    gap = max(0, width - _w(s))
    return (" " * gap + s) if right else (s + " " * gap)


def run_one(agent_mode: str, work_dir: str, mode: str, reuse: bool,
            only: list = None, tier: str = "") -> dict:
    if only:
        samples = [s for s in goldset.list_resolved() if s.sid in only]
    else:
        samples = None
    ds = AuditBenchDataset(samples)
    if len(ds) == 0:
        raise RuntimeError("没有可跑的考卷：请确认测试集目录存在（见 goldset.REAL_ROOT）")
    agent = Paper2LabAgent(mode=agent_mode, tier=tier or None)
    t0 = time.time()
    summary = run_agent_eval(agent, ds, work_dir=work_dir, mode=mode, reuse=reuse)
    summary["_elapsed_s"] = round(time.time() - t0, 2)
    summary["_details"] = {idx: ds.details.get(idx, {}) for idx in range(len(ds))}
    summary["_samples"] = [s.sid for s in ds.samples]
    summary["_version"] = agent.model_version
    return summary


def _stat_mean(summary: dict, key: str) -> str:
    vals = [float((d.get("stats") or {}).get(key, 0) or 0)
            for d in summary["_details"].values() if d.get("stats")]
    return _fmt(sum(vals) / len(vals)) if vals else "—"


def print_block(agent_mode: str, summary: dict) -> list:
    lines = [f"===== 评测汇总 · {agent_mode} ====="]
    sids = summary["_samples"]
    head = (_pad("样本", 14) + _pad("召回", 7, True) + _pad("精度", 7, True)
            + _pad("F1", 7, True) + _pad("自证率", 8, True)
            + _pad("诱饵误报", 9, True) + _pad("结论数", 7, True))
    lines.append(head.rstrip())
    for i, sid in enumerate(sids):
        det = summary["_details"].get(i, {})
        sc = det.get("score", {})
        cells = [_fmt(sc[k]) if k in sc else "—"
                 for k in ("recall", "precision", "f1", "selfcheck_rate", "decoy_fp")]
        n = float(sc.get("n_findings", 0.0))
        lines.append(_pad(sid, 14)
                     + _pad(cells[0], 7, True) + _pad(cells[1], 7, True)
                     + _pad(cells[2], 7, True) + _pad(cells[3], 8, True)
                     + _pad(cells[4], 9, True) + _pad(_fmt(n), 7, True))
    lines.append("")
    overall = [f"{label} {_cell(summary, k)}" for k, label in _METRIC_ORDER]
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


def print_tiers(rows: list) -> list:
    """三档消融表 —— 每行一档，看"多开一档多做了什么"。"""
    cols = [("档位", 18), ("开到的站", 28)] + [(lbl, 9) for _, lbl in _METRIC_ORDER] \
        + [(lbl, 9) for _, lbl in _STAT_ORDER]
    head = "".join(_pad(name, w) for name, w in cols)
    lines = ["===== 三档消融 · 能力递增（只加能力、不改结论）=====", head, "-" * _w(head)]
    for key, s in rows:
        t = TIER_BY_KEY[key]
        vals = [t.title, "/".join(t.stations)] + [_cell(s, k) for k, _ in _METRIC_ORDER] \
            + [_stat_mean(s, k) for k, _ in _STAT_ORDER]
        line = ""
        for i, (v, (_n, w)) in enumerate(zip(vals, cols)):
            line += _pad(v, w, right=(i >= 2))       # 前两列是文字，其余是数
        lines.append(line.rstrip())
    lines.append("-" * _w(head))
    lines.append("说明：「—」= 本档**没开**这项能力所以没测，不是 0 分；"
                 "召回与诱饵误报三档应当一致（加能力不该改结论）。")
    print("\n".join(lines))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Paper2Lab 审计评测（对齐司南 SciEvalKit）")
    ap.add_argument("--agent-mode", default="classic",
                    help="classic / multagent / closeloop，可逗号分隔多值对比")
    ap.add_argument("--tiers", default="",
                    help="档位消融: parser,calc,review（逗号分隔，只跑 classic）")
    ap.add_argument("--mode", default="all", choices=["all", "infer", "eval"])
    ap.add_argument("--work-dir", default="eval_output")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--report", default="", help="另存 markdown 报告路径")
    ap.add_argument("--only", default="", help="只跑指定样本 id（逗号分隔），便于快速回归")
    args = ap.parse_args()
    _only = [x.strip() for x in args.only.split(",") if x.strip()]

    modes = [m.strip() for m in args.agent_mode.split(",") if m.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    if tiers:
        bad = [t for t in tiers if t not in TIER_BY_KEY]
        if bad:
            ap.error(f"未知档位 {bad}；可选 {' / '.join(tier_keys())}")
        if len(modes) > 1 or modes[0] != "classic":
            # 档位只在经典链上做（图版是硬边串起来的固定流程，逐件跳过会破图）
            ap.error("--tiers 只能配 --agent-mode classic（档位消融只跑经典链）")

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

    # ---- 档位轴：一条能力曲线（3C）
    if tiers:
        rows = []
        for t in tiers:
            label = f"classic · 档位 {TIER_BY_KEY[t].title}"
            s = run_one("classic", args.work_dir, args.mode, args.reuse, _only, tier=t)
            all_summaries[label] = s
            rows.append((t, s))
            block = print_block(f"{label}（{s['_version']}）", s)
            print()
            md += ["```text"] + block + ["```", ""]
        table = print_tiers(rows)
        print()
        md += ["## 三档消融（能力递增）", "", "```text"] + table + ["```", ""]
        md += ["| 档位 | 开到哪些站 | 本档新增什么 |", "| --- | --- | --- |"]
        for key, _s in rows:
            t = TIER_BY_KEY[key]
            md.append(f"| {t.title} | {'/'.join(t.stations)} | {t.desc} |")
        md.append("")

    # ---- 模式轴：换调度方式
    if modes and not tiers:
        for m in modes:
            s = run_one(m, args.work_dir, args.mode, args.reuse, _only)
            all_summaries[m] = s
            block = print_block(f"{m}（{s['_version']}）", s)
            print()
            md += ["```text"] + block + ["```", ""]

    # 多模式对比：同一批考察项是否"结论不变"
    if not tiers and len(modes) > 1:
        md += ["## 多模式对比", "", "| 指标 | " + " | ".join(modes) + " |",
               "| --- | " + " | ".join("---" for _ in modes) + " |"]
        for k, label in _METRIC_ORDER:
            md.append(f"| {label} | " + " | ".join(_cell(all_summaries[m], k) for m in modes) + " |")
        md.append("")

    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(md), encoding="utf-8")
        print(f"报告已写入: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
