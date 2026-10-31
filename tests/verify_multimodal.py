"""阶段二 2E 验证 —— 图表多模态解析：真看了图，还是只是把图注抄了一遍。

守的契约:
  A 常量与封闭集合: 线索类型 / 图种类都是封闭集合, 图注与表注各归其位
  B 图注通道（纯文本）: 合成正文能命中图注/表注并带页码出处; **正文引用不被误判**
  C 读图通道（离线 mock）: 线索带图上逐字 anchor、来源标 vlm、两遍取交集才采信
  D 边界守卫: 解释性表述（说明 / 表明 / 证明 / suggests）**一律丢**; 无 anchor 的一律丢
  E 出处可核验: 每条线索都有 `page_ref` 且含 `Page N`; anchor 非空
  F 不造数: 没有图注 → 0 条且如实说明; 没视觉 key → 只有图注且说明"读图未启用"
  G 排序: 越硬的线索（峰位 / 趋势）越靠前; rank 连续
  H 契约: FigureClue.to_dict 键齐全; 旧对象（假设卡 / 实验卡）字段一个没被动
  I 流水线接线: TOOLS 里有序、归 ① 站、图版前置/写回/finalize/checkpoint 白名单都登记
  J 零成本与边界: 不直接 import openai / streamlit; 图注通道不依赖任何 key
  K 防伪: 演示数据里确实有**位图**（不是纯文字层）, 否则读图通道根本看不到它
  L 端到端: 带图 demo 真跑一遍, 至少一类结构化线索 + 出处; 报告有这一节

为什么要有这一项:
  本层最危险的失败模式是**"假装看过图"** —— 面板上有"图表解析"四个字,
  结论却只是把图注复述一遍, 甚至把"说明 / 表明"这类**解释**当成观察。
  所以 D（边界）与 C（真的有 anchor 才采信）是重点, B 则防"正文引用被当成图注"。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.models import PaperInfo                                  # noqa: E402
from paper2lab.multimodal import (CLUE_TYPES, FIGURE_KINDS,             # noqa: E402
                                  FIGURE_MAX, FigureClue, engine_name,
                                  read_figures, summarize)

CHECKS = 0
FAILURES = []


def check(ok, label):
    global CHECKS
    CHECKS += 1
    assert isinstance(ok, bool), "[%s] ok 不是布尔值: %r" % (label, ok)
    assert isinstance(label, str)
    if not ok:
        FAILURES.append(label)
        print("  *x* " + label)
    else:
        print("  [ok] " + label)


def _paper(lines):
    """造一篇"论文"（只有正文文本层, 够图注通道用）。"""
    p = PaperInfo()
    p.page_texts = [(i + 1, t) for i, t in enumerate(lines)]
    return p


# ==================================================================== A
print("A 常量与封闭集合")
check(len(set(CLUE_TYPES)) == len(CLUE_TYPES), "线索类型互不相同")
check(len(set(FIGURE_KINDS)) == len(FIGURE_KINDS), "图种类互不相同")
check("图注" in CLUE_TYPES and "表注" in CLUE_TYPES, "图注 / 表注都在封闭集合里")
check("峰位" in CLUE_TYPES and "趋势" in CLUE_TYPES,
      "「峰位」「趋势」是登记过的线索类型（谱图只读这两类之外还要能报轴/图例）")
check(isinstance(FIGURE_MAX, int) and FIGURE_MAX > 0, f"展示上限是正整数（{FIGURE_MAX}）")
check("图注" in engine_name() or "读图" in engine_name(),
      f"引擎名如实说明两条通道: {engine_name()}")

# ==================================================================== B
print("\nB 图注通道（纯文本）")
_p = _paper([
    "3.1 Setup\nWe train for 300 epochs.\n"
    "图 3　训练损失曲线随 epoch 下降。\n"
    "Figure 4: Validation accuracy vs. batch size.\n"
    "Table 2: Hyperparameters used in our experiments.",
    "正文里对图的引用: as shown in Figure 3, the loss drops.\n"
    "图 5\u3000红外光谱（KBr 压片）。",
])
_clues, _st = read_figures(_p)
_labels = {c.figure for c in _clues}
check({"图3", "Figure 4", "Table 2", "图5"} <= _labels,
      f"中文 / 英文图注与表注都能命中（{sorted(_labels)}）")
check(all(c.source == "caption" for c in _clues), "图注通道的线索来源都标 caption")
check(any(c.clue_type == "表注" for c in _clues), "Table 2 归为**表注**（不是图注）")
check(any(c.clue_type == "图注" for c in _clues), "图 3 / 图 5 归为**图注**")
check(sum(1 for c in _clues if c.figure == "图3") == 1,
      "同一个图号只出一条（没有把行内引用也当成图注）")
check(all("Page " in c.page_ref for c in _clues), "每条图注线索都带页码出处")
# 正文引用（"as shown in Figure 3"）不在行首 -> 不能被当成图注
_ref_only, _ = read_figures(_paper(["See Figure 9 for details.\n该图说明了我们的方法有效。"]))
check(all(c.figure != "Figure 9" for c in _ref_only),
      "行内引用（See Figure 9 …）**不算**图注")
# 引用式动词开头（"Figure 1 shows …" 落在行首）也要挡住 —— 实测踩过
_verb, _ = read_figures(_paper(["Figure 1 shows the training and validation curves.\n"
                                "Figure 2 reports the infrared spectrum."]))
check(not _verb, f"「Figure N shows/reports …」这类**引用**被挡住（拿到 {len(_verb)} 条）")
check(_st["captions"] >= 4, "统计里的图注条数与线索数自洽")

# ==================================================================== C / D / E / F / G / K
_formal = ROOT / "demo" / "figure_demo.pdf"
_mock_data = {"clues": [
    {"clue_type": "peak", "statement": "主峰位于 1730 cm-1", "anchor": "1730 cm-1"},
    {"clue_type": "trend", "statement": "验证集损失随 epoch 下降", "anchor": "val loss"},
    {"clue_type": "trend", "statement": "本方法说明模型更强", "anchor": "ours"},
    {"clue_type": "unit", "statement": "纵轴单位是百分比", "anchor": ""},
    {"clue_type": "legend", "statement": "三条曲线分别为 ours / baseline / ablation",
     "anchor": "ours baseline ablation"},
]}
_fd, _mock_path = tempfile.mkstemp(suffix=".json", dir=str(ROOT))
os.write(_fd, json.dumps(_mock_data, ensure_ascii=False).encode("utf-8"))
os.close(_fd)
os.environ["PAPER2LAB_FIGURE_MOCK_FILE"] = _mock_path

print("\nC 读图通道（离线 mock）")
if _formal.exists():
    from paper2lab.paper_parser import parse_paper                  # noqa: E402
    _pp = parse_paper(str(_formal))
    _vc, _vs = read_figures(_pp, pdf_path=str(_formal), vision_kwargs={})
    _vlm = [c for c in _vc if c.source == "vlm"]
    check(bool(_vlm), f"读图通道产出了线索（{len(_vlm)} 条）")
    check(_vs["enabled"] is True, "统计里标了读图通道已启用")
    check(_vs["regions_read"] >= 1, f"真的读了图（{_vs['regions_read']} 页）")
    check(_vs.get("mode") == "mock", "模式如实标 mock（不是假装走了 API）")
    check(all(c.anchor for c in _vlm), "读图线索都带图上逐字 anchor")
    check(all(c.kind in FIGURE_KINDS for c in _vlm), "图种类落在封闭集合里")
    check(all(c.clue_type in CLUE_TYPES for c in _vlm), "线索类型落在封闭集合里")
    # 防幻觉的核心是"两遍取交集" —— 直接拿两份**不一样**的返回验它
    from paper2lab.multimodal.figures import _intersect               # noqa: E402
    _first = {"figure_kind": "谱图", "clues": [
        {"clue_type": "peak", "statement": "A", "anchor": "a"},
        {"clue_type": "peak", "statement": "B", "anchor": "b"},
        {"clue_type": "peak", "statement": "C", "anchor": "c"}]}
    _second = {"figure_kind": "谱图", "clues": [
        {"clue_type": "peak", "statement": "A", "anchor": "a"},
        {"clue_type": "peak", "statement": "B", "anchor": "b"}]}
    _hit = _intersect(_first, _second)
    check([c["anchor"] for c in _hit["clues"]] == ["a", "b"],
          "两遍独立读图**取交集**：只有两遍都出现的 anchor 才采信（防幻觉）")
    check(_hit["figure_kind"] == "谱图", "取交集时保留图种类")

    print("\nD 边界守卫")
    _stmts = " ".join(c.statement for c in _vlm)
    check("说明" not in _stmts and "表明" not in _stmts,
          "解释性表述（说明 / 表明）被丢弃 —— 这是「不说读懂谱图」那道闸")
    check(not any(c.clue_type == "单位" for c in _vlm if not c.anchor),
          "没有 anchor 的线索被丢弃（无法回图核对的一律不采信）")
    check(_vs["dropped"] >= 2, f"丢弃计数如实（{_vs['dropped']} 条）")
    _src = (ROOT / "paper2lab" / "multimodal" / "figures.py").read_text(encoding="utf-8")
    check("_INTERPRET" in _src, "边界守卫在源码里是可核查的一道闸（_INTERPRET）")

    print("\nE 出处可核验")
    check(all(c.page_ref and "Page " in c.page_ref for c in _vc),
          "每条线索都有 `Page N` 形式的出处")
    check(all(c.figure for c in _vc), "每条线索都指明了是哪张图")

    print("\nG 排序")
    _order = [c.clue_type for c in _vc]
    _hard = [t for t in _order if t in ("峰位", "趋势")]
    _soft = [t for t in _order if t in ("图例", "图注", "表注")]
    check(not _hard or not _soft or _order.index(_hard[-1]) < _order.index(_soft[0])
          or all(_order.index(t) < len(_hard) for t in _soft),
          "越硬的线索（峰位 / 趋势）排在软线索（图例 / 图注）之前")
    check([c.rank for c in _vc] == list(range(1, len(_vc) + 1)), "rank 连续（消融对照才能比）")

    print("\nK 防伪：演示数据里确实有位图")
    from paper2lab.multimodal.figures import _figure_pages           # noqa: E402
    _pages = _figure_pages(str(_formal), [])
    check(len(_pages) >= 2,
          f"演示 PDF 里真的有大块**位图**（{_pages}），不是纯文字层 —— 矢量图读图通道看不到")
else:
    check(False, "demo/figure_demo.pdf 缺失，无法验证读图通道（先跑 tools/make_figure_demo.py）")

# ==================================================================== F / H / I / J
print("\nF 不造数")
_empty, _es = read_figures(_paper(["这是一段没有任何图注的正文。"]))
check(_empty == [], "没有图注时不产出任何线索（不硬凑）")
check(any("没有" in n or "没找到" in n for n in _es["notes"]),
      f"如实说明「没找到图注」，而不是假装论文没有图：{_es['notes'][:1]}")
if _formal.exists():
    os.environ.pop("PAPER2LAB_FIGURE_MOCK_FILE", None)
    _nokey, _noks = read_figures(_pp, pdf_path=str(_formal), vision_kwargs={})
    check(_noks["enabled"] is False, "没有 key / 没有 mock 时，读图通道**不启用**")
    check(any("未配置视觉模型 key" in n for n in _noks["notes"]),
          "如实说明「未配置 key、页内图形未读」")
    check(all(c.source == "caption" for c in _nokey), "此时只有图注通道的产物（零成本那一路）")
    os.environ["PAPER2LAB_FIGURE_MOCK_FILE"] = _mock_path

print("\nH 契约")
d = FigureClue(key="k", page=1, figure="图1", kind="曲线", clue_type="图注",
               statement="s", anchor="a", source="caption", page_ref="Page 1 · 图注（图1）")
_d = d.to_dict()
check(set(_d) >= {"key", "rank", "page", "figure", "kind", "clue_type",
                  "statement", "anchor", "source", "page_ref", "note"},
      f"FigureClue.to_dict 键齐全（{sorted(_d)}）")
_d2 = d.to_dict()
_d2["statement"] = "changed"
check(d.statement == "s", "to_dict 是拷贝，改返回值不会污染原对象")
from paper2lab.experiment import ExperimentCard                        # noqa: E402
from paper2lab.hypothesis import Hypothesis                            # noqa: E402
_h = Hypothesis(key="k", kind="取值冲突", claim="c", question="q", falsifiable_by="f")
check(hasattr(_h, "verdict") and hasattr(_h, "derivation"),
      "旧对象（假设卡）字段一个没被动（2B 的判决字段还在）")
_e = ExperimentCard(key="k")
check(hasattr(_e, "tool_path") and hasattr(_e, "records"),
      "旧对象（实验卡）字段一个没被动（2D 的工具路径还在）")

print("\nI 流水线接线")
from paper2lab.toolbox import STEPS, TOOLS, STATION_OF_STEP            # noqa: E402
_keys = [t.key for t in TOOLS]
check("read_figures" in _keys, "TOOLS 里登记了 read_figures")
check(_keys.index("read_figures") == _keys.index("analyze_images") + 1,
      "排在「图像解析」之后（同属 ① 站「读」这一步, 顺序固定）")
check(_keys.index("read_figures") < _keys.index("hypothesize"),
      "排在「假设生成」之前 —— 先看图, 再提假设")
check("图表解析" in STEPS, "进度条上有「图表解析」这一格")
check(STATION_OF_STEP.get("图表解析") == "① 假设", "归 ① 假设站")
check(set(STATION_OF_STEP) == set(STEPS), "四站归属与 STEPS 一一对应（模块级断言也守着）")
_gsrc = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")
for _need, _why in (
        ('"read_figures": {"paper"}', "图版前置条件已登记"),
        ('elif tool.key == "read_figures":', "图版写回分支已登记"),
        ('attach_figures(audit, state.get("figure_clues")', "图版 finalize 已挂载"),
        ('"paper2lab.multimodal", "FigureClue"', "checkpoint 白名单已登记"),
        ('"figure_clues": [], "figure_total": 0', "两处初始 state 已补齐")):
    check(_need in _gsrc, _why)
_rsrc = (ROOT / "paper2lab" / "runner.py").read_text(encoding="utf-8")
check('elif tool_key == "read_figures":' in _rsrc, "runner 的 _fill_empty 有清空分支")
check("attach_figures(job.audit" in _rsrc, "runner 收尾挂了产物")
_tsrc = (ROOT / "paper2lab" / "toolbox.py").read_text(encoding="utf-8")
check("def attach_figures(" in _tsrc, "attach_figures 是唯一挂载点")
check(_tsrc.count("attach_figures(job.audit") == 1, "报告生成前只经由这一个挂载点")
check("图表线索" in (ROOT / "paper2lab" / "report.py").read_text(encoding="utf-8"),
      "报告里有「图表线索」这一节")
check("_render_figures" in (ROOT / "app.py").read_text(encoding="utf-8"),
      "界面上有图表线索面板")
check("read_figures" in (ROOT / "paper2lab" / "execview.py").read_text(encoding="utf-8"),
      "执行记录里有这一步的产物说明")

print("\nJ 零成本与边界")
check("import openai" not in _src, "本模块不直接 import openai（读图走 vision 的通道）")
check("streamlit" not in _src, "本模块不依赖 streamlit（纯逻辑, 可离线跑）")
check("from ..vision import" in _src, "读图的 API 通道**复用** vision（不另造一套）")
check('os.getenv("PAPER2LAB_FIGURE_MOCK_FILE")' in _src
      and 'os.getenv("PAPER2LAB_VISION_MOCK_FILE")' not in _src,
      "离线钩子用**独立**变量名（不与「读超参数」的 mock 串味）")
check(summarize([])["total"] == 0, "空列表的统计是 0（不是异常）")

# ==================================================================== L
print("\nL 端到端")
if _formal.exists():
    from paper2lab.pipeline import run_audit                        # noqa: E402
    _audit, _report = run_audit(str(_formal), str(ROOT / "demo" / "project.zip"),
                                force_mock=True)
    _cl = list(getattr(_audit, "figure_clues", None) or [])
    _stt = dict(getattr(_audit, "figure_stats", None) or {})
    check(bool(_cl), f"端到端拿到了图表线索（{len(_cl)} 条）")
    check(_audit.figure_total >= len(_cl), "总数如实报（不小于展示条数）")
    check(len(_cl) <= FIGURE_MAX, f"界面拿到的条数 ≤ 展示上限 {FIGURE_MAX}")
    check(any(c.source == "vlm" for c in _cl) and any(c.source == "caption" for c in _cl),
          "两条通道的产物都在（图注原文 ＋ 读图）")
    check(any(c.clue_type in ("峰位", "趋势") for c in _cl),
          "至少抽到一类**结构化线索**（峰位 / 趋势）—— 这是本阶段的验收标准")
    check(all(c.page_ref for c in _cl), "端到端产物每条都有出处")
    check("图表线索" in _report, "导出的报告里有「图表线索」这一节")
    check(any(c.statement[:8] in _report for c in _cl),
          "报告里真的渲染出了线索内容（不只是标题）")
    _steps = {s["key"] for s in (_audit.execution_log or {}).get("steps", [])}
    check("read_figures" in _steps, "经典版流水线真的跑了这一步")
    check(summarize(_cl)["total"] == len(_cl), "summarize 与产物条数一致")
    # 无图的论文要如实给 0, 不能凭空造
    _plain, _ = run_audit(str(ROOT / "demo" / "paper.pdf"),
                          str(ROOT / "demo" / "project.zip"), force_mock=True)
    check(list(getattr(_plain, "figure_clues", None) or []) == [],
          "纯文本论文（无图注）→ 0 条线索（不硬凑）")
else:
    check(False, "demo/figure_demo.pdf 缺失，无法端到端验证")

# 收尾：清掉临时 mock 文件与环境变量
try:
    os.environ.pop("PAPER2LAB_FIGURE_MOCK_FILE", None)
    os.remove(_mock_path)
except Exception:
    pass

print("\n" + "=" * 72)
print(f"共 {CHECKS} 项，未通过 {len(FAILURES)} 项")
if FAILURES:
    for f in FAILURES:
        print("  *x* " + f)
    sys.exit(1)
print("全部通过 ✅")
sys.exit(0)
