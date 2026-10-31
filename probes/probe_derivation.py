"""探针: 看 2B 的推演把假设卡分成了几堆（纯离线试跑, 不接流水线）。

用法:
    .venv\\Scripts\\python.exe probes\\probe_derivation.py [样本目录]

默认用 demo/paper.pdf + demo/project.zip；给目录则用该目录下的
paper.pdf + code.zip（如 C:\\Users\\lijiahui\\Desktop\\Paper2Lab测试集\\01_DDPM）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 脚本下沉到 probes/ 后 sys.path[0] 是 probes/，必须自己补工程根（老坑）
sys.path.insert(0, str(ROOT))

from paper2lab.code_analyzer import analyze_code            # noqa: E402
from paper2lab.derivation import derive, summarize          # noqa: E402
from paper2lab.hypothesis import build_hypotheses           # noqa: E402
from paper2lab.paper_parser import parse_paper              # noqa: E402

DEMO = ROOT / "demo"


def _resolve(arg):
    if arg:
        d = Path(arg)
        return d / "paper.pdf", d / "code.zip", d.name
    return DEMO / "paper.pdf", DEMO / "project.zip", "demo"


def main() -> None:
    pdf, zip_, name = _resolve(sys.argv[1] if len(sys.argv) > 1 else None)
    for p in (pdf, zip_):
        if not p.exists():
            print(f"⚠️ 样本缺失: {p}")
            return

    t0 = time.time()
    paper = parse_paper(str(pdf))
    code = analyze_code(str(zip_))
    cards = build_hypotheses(paper, code)
    t1 = time.time()
    cards = derive(cards, paper, code)
    t2 = time.time()

    st = summarize(cards)
    print(f"\n===== 样本: {name} =====")
    print(f"解析+提假设 {t1 - t0:.1f}s，推演 {t2 - t1:.3f}s")
    print(f"引擎: {st['engine']}")
    print(f"共 {st['total']} 条 —— 已定位 {st['flagged']} / 待验 {st['pending']} / 已排除 {st['excluded']}\n")

    for c in cards:
        mark = {"已定位": "★", "已排除": "×", "待验": "·"}.get(c.verdict, "?")
        print(f"{mark} [{c.rank}] {c.verdict}｜{c.kind}｜{c.key}   (原分 {c.score}, 需实机 {'是' if c.needs_rerun else '否'})")
        print(f"    陈述: {c.claim}")
        if c.derivation:
            print(f"    推演({c.derivation_kind}): {c.derivation}")
        if c.paper_value:
            print(f"    论文: {c.paper_value}   ← {c.paper_where}")
        if c.code_value:
            print(f"    代码: {c.code_value}   ← {c.code_where}")
        print()

    by_kind: dict = {}
    for c in cards:
        by_kind[c.derivation_kind or "(无推演)"] = by_kind.get(c.derivation_kind or "(无推演)", 0) + 1
    print("推演类型计数:", by_kind)
    by_verdict: dict = {}
    for c in cards:
        by_verdict[c.verdict] = by_verdict.get(c.verdict, 0) + 1
    print("判决计数:", by_verdict)


if __name__ == "__main__":
    main()
