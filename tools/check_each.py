"""逐项单跑全套检查（2026-10-29 新建）。

为什么要它
----------
`run_all_checks.py` 是把所有检查**串起来一把跑**，用来发现"协同运行时才暴露"的问题。
但它看不到另一件事：**每一项单独拿出来跑，还过不过**。两者是一对——
全套过、单项也过，才能说"每个功能点都被独立执行过"。

这个脚本就是逐项单跑那一半：每次只起一个子进程、只跑一项、单独记退出码与耗时，
结果追加到 JSONL（中断了可以续跑，不重复跑已经跑完的项）。

跑法：
  .venv/Scripts/python.exe tools/check_each.py --record <路径>.jsonl            # 全跑
  .venv/Scripts/python.exe tools/check_each.py --record <路径>.jsonl --from 1 --to 10
  .venv/Scripts/python.exe tools/check_each.py --record <路径>.jsonl --summary  # 只看汇总
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_suite():
    """按文件路径加载套件模块 —— tests/ 不是包，直接 import 会失败。"""
    spec = importlib.util.spec_from_file_location(
        "rac_suite", ROOT / "tests" / "run_all_checks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 模块级只有常量与函数定义，无副作用
    return mod


def _read(rec: Path) -> list[dict]:
    if not rec.exists():
        return []
    out = []
    for ln in rec.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue                       # 中断时最后一行可能是半截
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="JSONL 记录文件（追加写）")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="已跑过的也重跑")
    ap.add_argument("--summary", action="store_true", help="只读记录并打印汇总")
    a = ap.parse_args()

    suite = _load_suite()
    rec = Path(a.record)
    rows = _read(rec)

    if a.summary:
        total = len(suite.CHECKS)
        by_i = {r["i"]: r for r in rows}
        print(f"{'#':>3}  {'状态':<9} {'耗时':>8}  检查项")
        print("-" * 74)
        bad = []
        for i, (name, _argv, _to) in enumerate(suite.CHECKS, 1):
            r = by_i.get(i)
            if not r:
                print(f"{i:>3}  {'未跑':<9} {'':>8}  {name}")
                continue
            st = "TIMEOUT" if r["timeout"] else ("通过" if r["code"] == 0 else f"退出码{r['code']}")
            flag = "" if (r["code"] == 0 and not r["timeout"]) else "  <<"
            if flag:
                bad.append(i)
            print(f"{i:>3}  {st:<9} {r['elapsed']:>7.1f}s  {name}{flag}")
        print("-" * 74)
        print(f"共 {total} 项，已跑 {len(by_i)} 项，未过 {len(bad)} 项"
              + (f"：第 {', '.join(map(str, bad))} 项" if bad else ""))
        warn = [r["i"] for r in rows if r.get("hits")]
        if warn:
            print(f"含警示关键词（不一定算失败，需人看）：第 {', '.join(map(str, warn))} 项")
        return 1 if bad else 0

    done = {r["i"] for r in rows}
    total = len(suite.CHECKS)
    nd = 0
    for i, (name, argv, to) in enumerate(suite.CHECKS, 1):
        if i < a.start or (a.end and i > a.end):
            continue
        if i in done and not a.force:
            print(f"[{i}/{total}] 已跑过，跳过：{name}", flush=True)
            continue
        print(f"[{i}/{total}] {name} …", flush=True)
        r = suite.run_one(name, argv, to)
        r["i"] = i
        with open(rec, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        st = "TIMEOUT" if r["timeout"] else ("通过" if r["code"] == 0 else f"退出码{r['code']}")
        print(f"    -> {st}  {r['elapsed']:.1f}s"
              + (f"  警示 {len(r['hits'])} 条" if r["hits"] else ""), flush=True)
        nd += 1
    print(f"\n本轮跑了 {nd} 项，记录：{rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
