"""探针: 看 2A 产出的假设卡长什么样、质量如何（纯离线试跑, 不接进流水线）。

用法:
    .venv\\Scripts\\python.exe probes\\probe_hypothesis.py [样本目录]

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

from paper2lab.code_analyzer import analyze_code          # noqa: E402
from paper2lab.hypothesis import build_hypotheses          # noqa: E402
from paper2lab.paper_parser import parse_paper             # noqa: E402

DEMO = ROOT / "demo"


def _resolve(arg: str | None) -> tuple[Path, Path, str]:
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
    t1 = time.time()
    cards = build_hypotheses(paper, code)
    t2 = time.time()

    print(f"\n===== 样本: {name} =====")
    print(f"论文参数 {len(paper.params)} 项 / 代码参数 {len(code.params)} 项")
    print(f"解析 {t1 - t0:.1f}s，提假设 {t2 - t1:.2f}s（纯规则, 零成本）")
    print(f"共提出 {len(cards)} 张假设卡\n")

    for c in cards[:12]:
        print(f"[{c.rank}] {c.kind} · {c.key}   分数 {c.score} · {c.why}")
        print(f"    陈述: {c.claim}")
        print(f"    判据: {c.falsifiable_by}")
        if c.paper_value:
            print(f"    论文: {c.paper_value}   ← {c.paper_where}")
        if c.code_value:
            print(f"    代码: {c.code_value}   ← {c.code_where}")
        print(f"    需实机验证: {'是' if c.needs_rerun else '否（读代码即可定论）'}")
        print()

    by_kind: dict = {}
    for c in cards:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    print("分类计数:", by_kind)
    print("前 8 条的分数:", [c.score for c in cards[:8]])

    # 可证伪性自检: 每张卡都必须有判据（2A 的第二条硬规矩）
    bad = [c.key for c in cards if not c.falsifiable_by.strip()]
    print(f"缺判据的卡: {len(bad)} 张 {bad}")


if __name__ == "__main__":
    main()
