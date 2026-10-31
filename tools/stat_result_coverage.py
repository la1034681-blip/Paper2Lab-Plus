"""结果指标抽取的覆盖率盘点（51 份样例 / 任意目录）。

为什么要有这个（2026-10-30）:
    「抽到了几条」这种数字最容易说了不算 —— 换个目录、换一批文件、开关一个通道，
    结果就变。所以要有一个**固定口径、可一键复现、能把改前改后摆在一起看**的脚本。

用法:
    # 当前代码（三条通道全开）
    .venv/Scripts/python.exe tools/stat_result_coverage.py

    # 关掉表格通道 = 改前的行为，用来做对照
    .venv/Scripts/python.exe tools/stat_result_coverage.py --no-table

    # 只看某批
    .venv/Scripts/python.exe tools/stat_result_coverage.py --dir "C:/path/to/样例" --only S07,S13

输出:
    逐份的「抽到几条 / 各通道各几条」，末尾一行汇总，明细 JSON 落到桌面的测试集目录。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_DIR = Path(r"C:/Users/lijiahui/Desktop/Paper2Lab测试集/50份样例")
DEFAULT_OUT = Path(r"C:/Users/lijiahui/Desktop/Paper2Lab测试集/result_extract_stats.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="样例根目录（每份一个子目录，内含 paper.pdf）")
    ap.add_argument("--only", default="", help="只跑名字里含这些片段的（逗号分隔）")
    ap.add_argument("--no-table", action="store_true", help="关掉表格结果通道（= 改前行为，做对照用）")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    from paper2lab import paper_parser as pp

    if args.no_table:
        # 只关通道、不关坐标重组 —— 这样耗时对比也干净（两次都做了坐标计算）
        pp._rows_to_results = lambda rows: []

    root = Path(args.dir)
    if not root.is_dir():
        print(f"目录不存在: {root}")
        return 2
    wanted = [s.strip() for s in args.only.split(",") if s.strip()]

    rows = []
    t0 = time.time()
    for i, sub in enumerate(sorted(root.iterdir()), 1):
        pdf = sub / "paper.pdf"
        if not pdf.exists() or not sub.is_dir():
            continue
        if wanted and not any(w in sub.name for w in wanted):
            continue
        try:
            info = pp.parse_paper(str(pdf))
        except Exception as e:          # noqa: BLE001
            rows.append({"name": sub.name, "n": -1, "err": f"{type(e).__name__}: {e}"})
            print(f"{i:3d} {sub.name[:40]:42s} 解析失败 {e}", flush=True)
            continue
        src: dict[str, int] = {}
        for ev in info.results.values():
            src[ev.parser] = src.get(ev.parser, 0) + 1
        rows.append({"name": sub.name, "n": len(info.results), "src": src,
                     "items": {k: v.raw_value for k, v in info.results.items()}})
        print(f"{i:3d} {sub.name[:40]:42s} {len(info.results)} 条 {src}"
              + ("" if info.results else "  <-- 零结果"), flush=True)

    ok = [r for r in rows if r.get("n", 0) > 0]
    zero = [r for r in rows if r.get("n", 0) == 0]
    total = sum(r.get("n", 0) for r in rows)
    tag = "改前(关表格)" if args.no_table else "当前"
    print(f"\n[{tag}] 合计 {len(rows)} 份，有结果 {len(ok)} 份，零结果 {len(zero)} 份，"
          f"总条数 {total}，用时 {time.time() - t0:.0f}s")
    if zero:
        print("零结果：" + ", ".join(r["name"] for r in zero))

    out = Path(args.out)
    payload = {"mode": tag, "total_papers": len(rows), "with_result": len(ok),
               "total_items": total, "elapsed_s": round(time.time() - t0, 1),
               "rows": rows}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("明细:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
