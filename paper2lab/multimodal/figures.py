"""科学图表多模态解析（阶段二 2E）—— 从谱图 / 曲线 / 图表 / 结构图里读**结构化线索**。

为什么要单起一层（与图像通道 `vision.py` 的分工）:
- `vision.py` 读的是**实验配置值**（学习率、批大小…）—— 目标是"把论文写的超参补回来"。
- 本模块读的是**图本身**（坐标轴、趋势、峰位、图例）—— 目标是"论文里那张图到底画了什么",
  给 ① 假设站提供线索, 而不是去猜配置。
两者共用同一套视觉 API 通道（`vision._call_vision`）, 但**提示词与产物结构不同**, 不合并 ——
合并会让一条提示词同时背两个目标, 迟早互相干扰。

三条边界（对外必须讲清）:
1. **只读图上看得见的东西** —— 坐标轴标签、趋势方向、峰位、图例、图上标注的数值与单位。
2. **不说"读懂了谱图"** —— 不做化合物鉴定、不推断机理、不给结论。出现"说明 / 表明 / 证明 /
   暗示 / 因此"这类**解释性**表述的线索一律丢弃（见 `_INTERPRET` 守卫）。
3. **每条线索都要有出处**（`Page N · 图 k`, 读图线索还带图中逐字文字）; 读不出就说读不出, 不补数。

两条通道（与抽取工具阶梯同款: 便宜的在前、贵的在后）:
- **图注通道**（纯文本, 零成本, **无 key 也能跑**）: 摘录论文自己的图注原文 —— 出处是页码。
- **读图通道**（需视觉模型 key, 可选）: 对**有图注的那几页**读图, 两遍独立读取取交集防幻觉。

离线测试钩子: 设 `PAPER2LAB_FIGURE_MOCK_FILE` 指向一份 `{"clues": [...]}` JSON,
无 key 也能跑通读图链路。**刻意用独立的变量名**（不复用 `PAPER2LAB_VISION_MOCK_FILE`）——
那边伪造的是"超参数"结构, 一个是 `params`、一个是 `clues`, 共用一个变量迟早串味。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------- 常量

# 展示上限（与 HYPOTHESIS_MAX / EXPERIMENT_MAX 同口径: 总数如实报, 界面只列前 N 条）
FIGURE_MAX = 8
# 最多读几张图（控成本; 每张两遍 → 最多 2×N 次视觉调用）
FIGURE_REGION_MAX = 8
# 页面渲染 DPI —— 与 `vision._RENDER_DPI` 保持同一个数（同一套看图口径, 不另定）
from ..vision import _RENDER_DPI, _MIN_IMG_AREA_RATIO

# 线索类型（**封闭集合**: 只有这几种, 不接自由文本, 免得模型拿它写结论）
CLUE_TYPES: tuple = ("峰位", "趋势", "标注值", "坐标轴", "图例", "单位", "图注", "表注")

# 图种类（封闭集合, 供界面分组与统计）
FIGURE_KINDS: tuple = ("谱图", "曲线", "柱状图", "散点图", "热力图",
                      "结构图", "示意图", "表格", "图表")

# 排序: 越"硬"的线索（能当数用的）越靠前 —— 峰位 / 趋势 先于 图例 / 图注
_CLUE_ORDER: dict = {t: i for i, t in enumerate(CLUE_TYPES)}
# 通道权重: 读图（真看了图）与图注（论文原文）—— 同类型时读图在前
_SOURCE_ORDER: dict = {"vlm": 0, "caption": 1}

# ---- 图注识别: 标签必须**在行首**（图注通常独立成行; 正文里的 "as shown in Figure 3"
#      是引用不是图注, 行首判据能把绝大部分引用挡在外面 —— 宁可漏, 不要错认）。
#      ⚠ 只靠行首还不够: PDF 抽出来的正文是**按排版换行**的, 于是
#      "Figure 1 shows the training …" 这种**引用**也会落在行首（实测踩过）。
#      所以再加一道"图注式分隔"判据（见 `_caption_body`）: 编号后面要么紧跟
#      冒号/句点/破折号, 要么是**两个以上空格或是全角空格**, 要么正文直接以中文起头。
_CAP_RE = re.compile(
    r"(?m)^[ \t　]*"
    r"(图|表|Figure|Fig\.?|Table)[ \t]*"
    r"([0-9]+[A-Za-z]?(?:[.\-–—][0-9]+)?)"
    r"([^\n]{0,260})")
# 引用式表述常见的动词开头（"Figure 1 shows/reports/presents…"）—— 见到就丢
_REF_VERB = re.compile(
    r"^(shows?|reports?|presents?|illustrates?|depicts?|displays?|gives?|lists?|"
    r"compares?|summari[sz]es?|describes?)\b", re.IGNORECASE)
# 正文里"另一个图号"出现处 -> 图注文本到此为止（避免把下一句吞进来）
_NEXT_LABEL = re.compile(r"\s(?:图|表|Figure|Fig\.?|Table)\s*[0-9]")
# 图注里出现这些词 -> 判断图是什么种类（第一个命中的算）
_KIND_HINTS: tuple = (
    ("谱图", re.compile(r"谱|spectr|nmr|红外|拉曼|raman|\bir\b|\bxrd\b|\bxps\b|质谱|mass\s*spec",
                        re.IGNORECASE)),
    ("热力图", re.compile(r"热力|heat\s*map|heatmap|混淆矩阵|confusion", re.IGNORECASE)),
    ("柱状图", re.compile(r"柱|bar\s*chart|histogram|直方", re.IGNORECASE)),
    ("散点图", re.compile(r"散点|scatter", re.IGNORECASE)),
    ("结构图", re.compile(r"结构|structure|molecul|分子|晶体|crystal|骨架|skeleton|晶格",
                          re.IGNORECASE)),
    ("示意图", re.compile(r"示意|schematic|概览|overview|framework|架构|architecture|"
                          r"流程|pipeline|workflow|框图", re.IGNORECASE)),
    ("曲线", re.compile(r"曲线|curve|loss\b|accuracy|收敛|convergence|training|validation|"
                        r"epoch|训练|验证|准确率", re.IGNORECASE)),
)

# ---- 边界守卫: 解释性 / 结论性表述。子串命中即丢（这是"不说读懂谱图"的那道闸）。
_INTERPRET = re.compile(
    r"说明|表明|证明|意味着|暗示|证实|揭示了|揭示了|因为|所以|因此|导致|归因于|归结为|"
    r"indicates|implies|suggests|reveals|demonstrates|proves|confirms|"
    r"because|therefore|due\s+to|evidence\s+of",
    re.IGNORECASE)

# 读图提示词: 只准报"看得见的", 每条必须带逐字 anchor
_FIGURE_PROMPT = """你是科研图表读取助手。这是论文里的一幅图（或表）。
只报告**图上直接看得见的东西**, 不要解释、不要下结论、不要推断物理或化学意义。
可以报告的线索类型（每条都要给 anchor: 图中出现的**逐字**文字或数值, 用于人工核对）:
- 谱图里的峰位: 给出横坐标数值（例如 1730 cm-1）
- 曲线趋势: 上升 / 下降 / 持平 / 先升后降（不要解释为什么）
- 坐标轴: 轴标签与范围（例如 x: epoch 0-300）
- 图例: 图例条目（逐字抄）
- 图上标注的数值: 直接印在图上的数字
- 单位
严禁: 化合物鉴定、机理推断、"说明 / 表明 / 证明"这类结论、把看不清的值补出来。
看不清就少报几条, 不要凑数。只输出 JSON:
{"figure_kind":"曲线|谱图|柱状图|散点图|热力图|结构图|示意图|表格|图表",
 "clues":[{"clue_type":"trend","statement":"验证集损失随 epoch 下降","anchor":"y: val loss"}]}
"""

# 模型给的英文类型名 -> 本模块的封闭集合
_CLUE_TYPE_MAP: dict = {
    "peak": "峰位", "peaks": "峰位", "峰": "峰位", "峰位": "峰位",
    "trend": "趋势", "趋势": "趋势",
    "coordinate_axis": "坐标轴", "axis": "坐标轴", "axes": "坐标轴", "坐标轴": "坐标轴",
    "legend": "图例", "图例": "图例",
    "labeled_value": "标注值", "value": "标注值", "labeled": "标注值", "标注值": "标注值",
    "unit": "单位", "units": "单位", "单位": "单位",
    "caption": "图注", "图注": "图注",
}


# ---------------------------------------------------------------- 数据结构

@dataclass
class FigureClue:
    """一条**图表线索** —— 本模块的产物（界面 / 报告 / 留痕都只读这里）。

    字段刻意都取"可核验"的: `anchor` 是图上逐字出现的文字/数值, `page_ref` 是出处。
    没有 `conclusion` 这类字段 —— 本层不允许写结论（那是 ③ 站的事）。
    """
    key: str                       # 稳定标识（去重用）
    rank: int = 0                  # 排序后回填
    page: int = 0
    figure: str = ""               # "图 3" / "Figure 2" / "Page 5 内嵌图"
    kind: str = "图表"             # 图种类（CLUE_TYPES 之外的封闭集合 FIGURE_KINDS）
    clue_type: str = "图注"        # CLUE_TYPES 之一
    statement: str = ""            # 一句**看得见**的描述（不含解释）
    anchor: str = ""               # 图上逐字文字/数值（可核验的锚点）
    source: str = "caption"        # caption（图注原文）| vlm（读图）
    page_ref: str = ""             # "Page 5 · 图 3"
    note: str = ""                 # 补充说明（如"两次读图一致才采信"）

    def to_dict(self) -> dict:
        return {
            "key": self.key, "rank": self.rank, "page": self.page,
            "figure": self.figure, "kind": self.kind, "clue_type": self.clue_type,
            "statement": self.statement, "anchor": self.anchor,
            "source": self.source, "page_ref": self.page_ref, "note": self.note,
        }


# ---------------------------------------------------------------- 通道一: 图注（纯文本）

def _infer_kind(label: str, caption: str) -> str:
    if label.startswith(("表", "Table")):
        return "表格"
    for kind, pat in _KIND_HINTS:
        if pat.search(caption):
            return kind
    return "图表"


def _caption_body(tail: str) -> str:
    """从"编号后面那一串"里判断它是不是**图注**, 并切出图注正文。

    判据（任一满足才算图注）: 紧跟分隔符（: ： . ． - – —）／两个以上空格或全角空格／
    正文直接以中文起头。三条都不满足的（如 `Figure 1 shows the …`）视为**正文引用**。
    """
    tail = tail or ""
    m = re.match(r"^[ \t　]*[:：.．\-–—][ \t　]*", tail)
    if m:
        body = tail[m.end():]
    elif re.match(r"^([ \t　]{2,}|[^\x00-\x7f])", tail):
        body = tail
    else:
        return ""
    body = _NEXT_LABEL.split(body)[0]
    body = re.sub(r"\s+", " ", body).strip(" .。;；")
    if len(body) < 4 or _REF_VERB.match(body):
        return ""
    return body


def _caption_clues(paper) -> list:
    """摘录论文自己的图注原文 —— 零成本, 无视觉 key 也能跑。出处 = 页码 + 图号。"""
    out: list = []
    for page_no, text in (getattr(paper, "page_texts", None) or []):
        for m in _CAP_RE.finditer(text or ""):
            label = (m.group(1) or "").strip()
            num = (m.group(2) or "").strip()
            body = _caption_body(m.group(3))
            if not body:
                continue
            # "图 3" / "Figure 2" —— 中文标签不带空格, 英文标签带空格
            fig = f"{label}{num}" if re.match(r"^[图表]", label) else f"{label} {num}"
            is_table = label.startswith(("表", "Table"))
            clue_type = "表注" if is_table else "图注"
            key = f"cap:{page_no}:{fig}"
            out.append(FigureClue(
                key=key, page=page_no, figure=fig,
                kind=_infer_kind(label, body), clue_type=clue_type,
                statement=body, anchor=body, source="caption",
                page_ref=f"Page {page_no} · {clue_type}（{fig}）",
                note="论文图注原文摘录（逐字），本系统未作解释"))
    return out


# ---------------------------------------------------------------- 通道二: 读图（可选）

def _figure_mock() -> list:
    """离线测试钩子: 读一份伪造的模型返回, 用于无 key 环境验证读图链路。"""
    f = os.getenv("PAPER2LAB_FIGURE_MOCK_FILE")
    if not f:
        return []
    try:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(data.get("clues") or []) if isinstance(data, dict) else []


def _vision_ready(vision_kwargs: Optional[dict]) -> bool:
    """视觉通道是否可用（与 `tools.vision_ready` 同一口径: 页面 key 或 .env 配置）。"""
    if not vision_kwargs:
        return False
    from ..vision import _provider
    return _provider(vision_kwargs.get("api_key", ""),
                     vision_kwargs.get("base_url", ""),
                     vision_kwargs.get("model", "")) is not None


def _figure_pages(pdf_path: str, caption_pages: list) -> list:
    """候选页 = 有图注的页 ∪ **含有大块位图的页**。

    两条来源都要: 有些论文的图注没独立成行（正则抓不到）, 但图是真的（大幅位图）;
    反过来也有些页图注在、图却被拆成矢量小块（这时读整页也能看到）。
    判"大块位图"复用 `vision._MIN_IMG_AREA_RATIO`, 与图像通道同一口径, 不另定阈值。
    """
    import fitz
    out = set(int(p) for p in caption_pages or [])
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc, 1):
                area = abs(page.rect.width * page.rect.height) or 1.0
                try:
                    images = page.get_images(full=True)
                except Exception:
                    continue
                for img in images:
                    try:
                        rects = page.get_image_rects(img[0])
                    except Exception:
                        continue
                    if any(abs(r.width * r.height) / area >= _MIN_IMG_AREA_RATIO
                           for r in rects):
                        out.add(i)
                        break
    except Exception:
        pass
    return sorted(out)


def _page_png(page) -> bytes:
    """把一页渲染成 PNG; 该页有大块位图时**只渲染那块图**（更聚焦、更省 token）。"""
    import fitz  # noqa: F401  （page 已是 fitz 对象, 这里只为显式依赖）
    best = None
    page_area = abs(page.rect.width * page.rect.height) or 1.0
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    for img in images:
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            continue
        for r in rects:
            if best is None or abs(r.width * r.height) > abs(best.width * best.height):
                best = r
    clip = None
    if best is not None and abs(best.width * best.height) / page_area >= _MIN_IMG_AREA_RATIO:
        clip = best
    return page.get_pixmap(dpi=_RENDER_DPI, clip=clip).tobytes("png")


def _vision_clues(pdf_path: str, pages: list, vision_kwargs: dict,
                  stats: dict) -> list:
    """对"有图注的那几页"读图 —— 两遍独立读取, 逐字 anchor 取交集（防幻觉）。"""
    import fitz
    from ..vision import _call_vision, _load_cache, _provider, _save_cache

    prov = _provider(vision_kwargs.get("api_key", ""),
                     vision_kwargs.get("base_url", ""),
                     vision_kwargs.get("model", ""))
    mock = _figure_mock()
    if not prov and not mock:
        stats["notes"].append(
            "发现图注但**未配置视觉模型 key** —— 读图通道未启用, "
            "本节只有论文图注原文（页内图形未读; 配置 key 或勾选「模型支持读图」后重跑即可）")
        return []
    if mock:
        model, call_key, call_base = "figure-mock", "", ""
        stats["mode"] = "mock"
    else:
        call_key, call_base, model, _src = prov
        stats["mode"] = "api"
        model = model or "vision"
    stats["model"] = model

    cache = _load_cache()
    out: list = []
    pages = pages[:FIGURE_REGION_MAX]
    try:
        doc = fitz.open(pdf_path)
    except Exception:      # noqa: BLE001
        # 与图像通道同款：论文打不开就没有图可读，返回空列表并留一句说明。
        stats["notes"].append("论文文件打不开，未读图")
        return out
    with doc:
        for page_no in pages:
            if page_no < 1 or page_no > doc.page_count:
                continue
            try:
                png = _page_png(doc[page_no - 1])
            except Exception as ex:      # noqa: BLE001
                stats["notes"].append(f"第 {page_no} 页渲染失败({type(ex).__name__}), 该页未读图")
                continue
            key_hash = hashlib.sha1(png + model.encode() + _FIGURE_PROMPT.encode()
                                    ).hexdigest()[:16]
            hit = cache.get(key_hash)
            if hit is not None:
                payload = hit
            elif mock:
                payload = {"figure_kind": "图表", "clues": mock}
            else:
                try:
                    first = _call_vision(png, model, call_base, call_key, _FIGURE_PROMPT)
                    second = _call_vision(png, model, call_base, call_key, _FIGURE_PROMPT)
                except Exception as ex:      # noqa: BLE001
                    stats["notes"].append(
                        f"第 {page_no} 页读图失败: {type(ex).__name__}: {str(ex)[:120]}")
                    continue
                payload = _intersect(first, second)
                cache[key_hash] = payload
                _save_cache(cache)
            stats["regions_read"] += 1
            kind = str(payload.get("figure_kind") or "图表").strip()
            if kind not in FIGURE_KINDS:
                kind = "图表"
            fig = f"Page {page_no} 图形"
            for it in (payload.get("clues") or []):
                ct_raw = str(it.get("clue_type", "")).strip()
                ct = _CLUE_TYPE_MAP.get(ct_raw.lower()) or _CLUE_TYPE_MAP.get(ct_raw)
                statement = re.sub(r"\s+", " ", str(it.get("statement", ""))).strip()
                anchor = re.sub(r"\s+", " ", str(it.get("anchor", ""))).strip()
                if ct is None or not statement:
                    stats["dropped"] += 1
                    continue
                # ---- 边界守卫: 解释性 / 结论性表述一律丢（"不说读懂谱图"）
                if _INTERPRET.search(statement):
                    stats["dropped"] += 1
                    continue
                # ---- 可核验守卫: 没有逐字 anchor 的线索不采信（无法回图核对）
                if not anchor:
                    stats["dropped"] += 1
                    continue
                out.append(FigureClue(
                    key=f"vlm:{page_no}:{ct}:{anchor[:24]}", page=page_no,
                    figure=fig, kind=kind, clue_type=ct,
                    statement=statement, anchor=anchor, source="vlm",
                    page_ref=f"Page {page_no} · 读图（{kind}）",
                    note="两次独立读图一致才采信"))
    return out


def _intersect(first: dict, second: dict) -> dict:
    """两次读图取交集: 只有**逐字 anchor** 在两遍里都出现才保留。

    为什么按 anchor 而不是按整句: 整句是自由文本, 两次措辞必然有差;
    anchor 是"图上逐字出现的字", 两遍一致才说明它确实在图上（防幻觉）。
    """
    def _anchors(payload: dict) -> dict:
        out = {}
        for it in (payload or {}).get("clues") or []:
            a = re.sub(r"\s+", " ", str(it.get("anchor", ""))).strip().lower()
            if a:
                out[a] = it
        return out

    a, b = _anchors(first), _anchors(second)
    kept = [it for a_key, it in a.items() if a_key in b]
    return {"figure_kind": (first or {}).get("figure_kind")
                          or (second or {}).get("figure_kind"),
            "clues": kept}


# ---------------------------------------------------------------- 主入口

def read_figures(paper, *, pdf_path: str = "",
                 vision_kwargs: Optional[dict] = None) -> tuple:
    """读论文里的图 —— 返回 (线索列表, 统计)。**列表不截断**, 展示上限由调用方处理。

    先跑图注通道（零成本, 总能跑）; 有视觉 key（或离线 mock）时再跑读图通道。
    两条通道的产物**同构**（都是 FigureClue）, 因此下游（报告 / 界面）只认一种结构。
    """
    t0 = time.perf_counter()
    stats: dict = {
        "total": 0, "captions": 0, "vision": 0, "regions_read": 0, "dropped": 0,
        "kinds": {}, "clue_types": {}, "pages": [], "enabled": False,
        "mode": "off", "model": "", "notes": [], "engine": engine_name(),
        "seconds": 0.0,
    }

    clues = _caption_clues(paper)
    cap_pages = sorted({c.page for c in clues})
    stats["captions"] = len(clues)
    if not clues:
        stats["notes"].append(
            "没有在正文里找到可识别的图注（「图 N / Figure N / 表 N」独立成行）—— "
            "就本节而言只说明**没找到图注**, 不等于论文里没有图。")

    # 候选页: 有图注的页 ＋ 有大块位图的页（两条来源都要, 见 `_figure_pages`）
    pages = _figure_pages(pdf_path, cap_pages) if pdf_path else list(cap_pages)
    stats["pages"] = pages

    # 读图通道: 有视觉 key（或离线 mock）时才跑; 没有就如实说明"只有图注原文"。
    has_vision = _vision_ready(vision_kwargs) or bool(_figure_mock())
    if pdf_path and pages and has_vision:
        stats["enabled"] = True
        clues += _vision_clues(pdf_path, pages, dict(vision_kwargs or {}), stats)
    elif pages and not has_vision:
        stats["notes"].append(
            f"发现 {len(pages)} 页含图注 / 大幅图形, 但**未配置视觉模型 key** —— "
            "读图通道未启用, 本节只有论文图注原文"
            "（配置 key 或勾选「模型支持读图」后重跑即可补上趋势与峰位）")

    # 去重（同页同图号同类型同内容）+ 排序
    seen: set = set()
    uniq: list = []
    for c in clues:
        sig = (c.page, c.clue_type, c.statement.strip())
        if sig in seen:
            stats["dropped"] += 1
            continue
        seen.add(sig)
        uniq.append(c)
    _rank(uniq)
    stats["total"] = len(uniq)
    stats["vision"] = sum(1 for c in uniq if c.source == "vlm")
    stats["captions"] = sum(1 for c in uniq if c.source == "caption")
    for c in uniq:
        stats["kinds"][c.kind] = stats["kinds"].get(c.kind, 0) + 1
        stats["clue_types"][c.clue_type] = stats["clue_types"].get(c.clue_type, 0) + 1
    stats["seconds"] = round(time.perf_counter() - t0, 4)
    if stats["dropped"]:
        stats["notes"].append(
            f"{stats['dropped']} 条线索被丢弃（含解释性表述 / 无图上原文可核验 / 重复）")
    return uniq, stats


def _rank(clues: list) -> None:
    """排序: 越硬的线索（峰位 / 趋势）越靠前; 同类型时读图在前、按页码升序。"""
    clues.sort(key=lambda c: (_CLUE_ORDER.get(c.clue_type, 99),
                              _SOURCE_ORDER.get(c.source, 9),
                              c.page, c.figure))
    for i, c in enumerate(clues, 1):
        c.rank = i


def summarize(clues: list) -> dict:
    """一句话说清这批线索分成了几堆（界面 / 报告共用, 不各写一套）。"""
    clues = list(clues or [])
    kinds: dict = {}
    types: dict = {}
    for c in clues:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
        types[c.clue_type] = types.get(c.clue_type, 0) + 1
    return {
        "total": len(clues),
        "captions": sum(1 for c in clues if c.source == "caption"),
        "vision": sum(1 for c in clues if c.source == "vlm"),
        "kinds": kinds, "clue_types": types,
        "pages": sorted({c.page for c in clues if c.page}),
        "engine": engine_name(),
    }


def engine_name() -> str:
    """本层的引擎名 —— 两条通道各是什么, 如实写出来（不吹成"多模态大模型理解"）。"""
    return ("图注原文抽取（正则, 本地）＋ 读图（视觉模型两遍取交集, 可选）")
