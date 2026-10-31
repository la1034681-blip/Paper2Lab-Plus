"""离线试跑探针（阶段二 2E）—— 单跑「图表解析」这一层, 打印每条线索与出处。

用途: 不开网页、不跑整条流水线, 直接看这一层读到了什么、哪些被守卫挡掉了。
默认跑带图的演示数据 `demo/figure_demo.pdf`; 也可以传入任意 PDF 路径。

读图通道需要视觉 key（或离线 mock）:
  set PAPER2LAB_FIGURE_MOCK_FILE=<一个 {"clues":[...]} 的 JSON>
无 key 无 mock 时只有图注通道的产物 —— 这是设计如此, 不是失效。

用法:
  .venv\\Scripts\\python.exe probes\\probe_multimodal.py
  .venv\\Scripts\\python.exe probes\\probe_multimodal.py 某篇论文.pdf
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from paper2lab.multimodal import engine_name, read_figures      # noqa: E402
from paper2lab.paper_parser import parse_paper                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "demo" / "figure_demo.pdf"


def main() -> int:
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not target.exists():
        print(f"找不到 {target} —— 先跑 tools/make_figure_demo.py 生成演示数据")
        return 2
    paper = parse_paper(str(target))
    clues, stats = read_figures(paper, pdf_path=str(target), vision_kwargs={})
    print("=" * 74)
    print(f"目标: {target.name}　页数: {paper.page_count}")
    print(f"引擎: {engine_name()}")
    print(f"候选页: {stats['pages']}　读图启用: {stats['enabled']}"
          f"（模式 {stats['mode']}）　实际读图 {stats['regions_read']} 页")
    print(f"读到 {stats['total']} 条　图注 {stats['captions']}　读图 {stats['vision']}"
          f"　丢弃 {stats['dropped']}　耗时 {stats['seconds']}s")
    print("-" * 74)
    for c in clues:
        print(f"#{c.rank}　[{c.clue_type}] {c.figure}（{c.kind}）　来源 {c.source}")
        print(f"     {c.statement}")
        print(f"     图上依据: {c.anchor}")
        print(f"     出处: {c.page_ref}")
    print("-" * 74)
    for n in stats.get("notes") or []:
        print(f"· {n}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
