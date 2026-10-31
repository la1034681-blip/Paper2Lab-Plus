"""探针: 看 2D 的实验站对假设卡设计了什么实验、真跑出了什么（纯离线试跑, 不接流水线）。

用法:
    .venv\\Scripts\\python.exe probes\\probe_experiment.py [样本目录]

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
from paper2lab.derivation import derive                     # noqa: E402
from paper2lab.experiment import design_experiments, summarize  # noqa: E402
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
    cards = derive(build_hypotheses(paper, code), paper, code)
    t1 = time.time()
    exps = design_experiments(cards, paper, code)
    t2 = time.time()

    st = summarize(exps)
    print(f"\n===== 样本: {name} =====")
    print(f"解析+提假设+推演 {t1 - t0:.1f}s，实验设计与真跑 {t2 - t1:.3f}s")
    print(f"引擎: {st['engine']}")
    print(f"共 {st['total']} 张实验卡 —— 对不上 {st['disagree']} / 需实机 {st['machine']} / "
          f"一致 {st['agree']} / 算不出来 {st['undecided']} / 无需实验 {st['no_need']}")
    print(f"工具调用 {st['calls']} 次，用到: {st['tools']}\n")

    for e in exps:
        print(f"[{e.rank}] {e.status}｜{e.key}")
        print(f"    设计: {e.design}")
        if e.tool_path:
            print(f"    工具路径: {' → '.join(e.tool_path)}")
        for r in e.records:
            flag = "✓" if r["ok"] else "×"
            print(f"      {flag} {r['seq']}. {r['name']}（{r['ms']}ms）: {r['output']}")
        if e.actual is not None or e.expected is not None:
            print(f"    预期: {e.expected}  ← {e.expected_from}")
            print(f"    实际: {e.actual}  ← {e.actual_from}")
        if e.rerun_plan:
            print(f"    实机方案: {e.rerun_plan}")
        print(f"    谁挑的: {e.plan_reason}")
        print()

    by: dict = {}
    for e in exps:
        by[e.status] = by.get(e.status, 0) + 1
    print("观察态计数:", by)


if __name__ == "__main__":
    main()
