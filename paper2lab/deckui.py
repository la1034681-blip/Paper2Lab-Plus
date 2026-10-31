"""指挥台·可点击版（Streamlit 原生渲染）。

为什么不用一整块 `st.html` —— **`st.html` 里点任何东西都回不到 Python**。
所以把「一行」拆成两件东西：

  ① 一段 HTML（视觉，完全复用 `DECK_CSS` 里那套 `.dk-*` 样式）
  ② 一个透明的 `st.button`，用 CSS 绝对定位盖在它上面（交互）

于是点击区域 = 整行、视觉一行没改，而点击**真能**回到 Python，
再把「你要看哪一段」写进 `session_state["dk_focus"]`，由 `render_detail` 摊开详情。

2026-10-27 第二版（这次的四件事）
--------------------------------
1. **点得开还要看得见**：详情原来画在大屏下方 —— 三栏一长，点第 9 行时
   面板落在屏幕外，看起来就是「点了没反应」。现在改成**弹窗**，点了立刻在
   眼前打开，关掉才回到大屏。
2. **点过的要留痕**：被点中的那一行/那一格加 `sel`（左侧亮条 ＋ 更亮的字），
   一眼知道「我现在看的是哪一块」。
3. **每个区域说自己的话**：六块区域各有各的视图（见 deckviews.py），
   位置看流程、四站看边界、事件看时间、判定看分布 —— 不互相抄。
4. **实时事件也能点**：标题点开是整轮时间线，每一条点开是它在时间上的位置。

口径铁律（与 deck.py 一致，没变）：
  · 格数取 `len(STEPS)`、工序取 `len(TOOLS)`，永不写死；
  · 没做的事标「未执行 / 计划跳过」并给理由，不填 0 糊过去；
  · 详情**只展示已有数据**，不新增判定、不造数、不算分。
"""
from __future__ import annotations

import html as _html

import streamlit as st

from .agent_roles import POSITIONS
from .deck import (DECK_CSS, _pos_state, _ready_block, _station_cells,
                   _verdict_counts, _yield_of)
from .deckviews import (VIEW_CSS, event_view, mat_view, pos_view, ready_view,
                        station_view, timeline_view, tool_view, verdict_view)
from .models import Status
from .toolbox import STATION_OF_STEP, STATIONS, STEPS, TOOLS

FOCUS_KEY = "dk_focus"

#: 站 → 产物优先级（详情面板里说「这一站交出了什么」用）
_ST_ORDER = {
    "① 假设": ("hypothesize", "read_figures", "chem_check", "parse_paper"),
    "② 实验": ("design_experiment",),
    "③ 验证": ("align", "selfcheck", "collider"),
    "④ 迭代": ("rootcause", "patch", "report"),
}


def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


# ---------------------------------------------------------------- 样式
def shell_css() -> str:
    """整套 `.dk-*` 样式换成挂到 Streamlit 容器上。

    做法：直接把 `DECK_CSS` 里的 `.p2l-deck` 前缀替换掉 —— 一套样式两种用法，
    不会出现「大屏版」和「可点击版」两套 CSS 各自漂移。
    """
    base = DECK_CSS.replace(".p2l-deck", ".st-key-dk_shell")
    return base + """<style>
/* ---- 容器：把 Streamlit 自己的间距清干净，才拼得回一整块大屏 ---- */
.st-key-dk_shell{overflow:visible !important;padding:0 !important;
  background:#080d17;border:1px solid #1b2540;border-radius:14px;}
.st-key-dk_shell [data-testid="stVerticalBlock"]{gap:0 !important;}
.st-key-dk_shell [data-testid="stHorizontalBlock"]{gap:8px !important;}
.st-key-dk_shell [data-testid="stElementContainer"]{margin:0 !important;}
.st-key-dk_shell [data-testid="stHtml"]{margin:0 !important;}
.st-key-dk_shell [data-testid="stMarkdownContainer"] p{margin:0;}
.st-key-dk_shell .dk-top{border-radius:13px 13px 0 0;}
.st-key-dk_shell .dk-stg{min-height:62px;}

/* ---- 可点区域：默认就能看出「这里能点」 ---- */
[class*="st-key-dkhot_"]{position:relative;margin-bottom:6px;border-radius:9px;
  transition:background .12s ease;}
[class*="st-key-dkhot_"]:hover{background:rgba(91,140,255,.10);}
[class*="st-key-dkhot_"]:hover .dk-fnm{color:#f2f7ff;}
/* ⚠ 关键：Streamlit 的 block 常常是 position:relative，一旦某一层 relative 了，
   透明按钮的 inset:0 就贴着那一层（而那一层高度=0）→ 按钮高度 0 点不到。
   所以把热区里的中间层全部按回 static，让按钮直接贴住「整行」这个框。 */
.st-key-dk_shell [class*="st-key-dkhot_"] > div,
.st-key-dk_shell [class*="st-key-dkhot_"] [data-testid="stElementContainer"],
.st-key-dk_shell [class*="st-key-dkhot_"] [data-testid="stVerticalBlock"],
.st-key-dk_shell [class*="st-key-dkhot_"] [data-testid="stHtml"]{position:static !important;}
[class*="st-key-dkhot_"] [data-testid="stButton"]{position:absolute !important;
  inset:0;z-index:6;height:100%;}
[class*="st-key-dkhot_"] [data-testid="stButton"] button{
  width:100% !important;height:100% !important;min-height:0 !important;
  opacity:0 !important;border:0 !important;padding:0 !important;
  background:transparent !important;box-shadow:none !important;cursor:pointer;}
.st-key-dk_shell .dk-hot{cursor:pointer;}
.st-key-dk_shell .dk-hot-arrow{float:right;font-size:10px;color:#4c5d80;
  margin-left:6px;opacity:.55;transition:opacity .12s;}
[class*="st-key-dkhot_"]:hover .dk-hot-arrow{opacity:1;color:#8ab0ff;}
/* ---- 选中态：比默认再亮一档，并在左边压一条亮条 ---- */
.st-key-dk_shell .dk-fl.sel{background:rgba(91,140,255,.17);
  border-color:rgba(91,140,255,.55);box-shadow:inset 3px 0 0 #5b8cff;}
.st-key-dk_shell .dk-fl.sel .dk-fnm{color:#ffffff;font-weight:700;}
.st-key-dk_shell .dk-fl.sel .dk-fmeta{color:#c6d9ff;}
.st-key-dk_shell .dk-fl.sel .dk-n{color:#8ab0ff;}
.st-key-dk_shell .dk-mem.sel{background:rgba(91,140,255,.18);color:#ffffff;
  box-shadow:inset 3px 0 0 #5b8cff;}
.st-key-dk_shell .dk-mem.sel .dk-mn{color:#ffffff;font-weight:800;}
.st-key-dk_shell .dk-mat.sel{background:rgba(91,140,255,.14);
  border-color:rgba(91,140,255,.45);}
.st-key-dk_shell .dk-stg.sel{border-color:rgba(91,140,255,.75);
  background:rgba(91,140,255,.14);box-shadow:inset 0 0 0 1px rgba(91,140,255,.25);}
.st-key-dk_shell .dk-stg.sel .dk-stt{color:#dbe8ff;}
.st-key-dk_shell .dk-evt.sel{background:rgba(91,140,255,.14);border-radius:7px;
  padding-left:7px;border-bottom-color:transparent;}
.st-key-dk_shell .dk-evt.sel b{color:#ffffff;}
.st-key-dk_shell .dk-badge.sel{outline:1px solid rgba(91,140,255,.85);
  outline-offset:1px;filter:brightness(1.35);}
.st-key-dk_shell .dk-rdy.sel{border-color:rgba(91,140,255,.6);
  background:rgba(91,140,255,.10);}
/* 四站与工序之间的底色：让整块看起来是一整屏，不是一堆 Streamlit 块 */
.st-key-dk_shell .dk-colpad{padding:0 2px;}
/* 事件行：block 之间不再各自拉开，跟原来那串 HTML 一个观感 */
.st-key-dk_shell .dk-evts{padding:0;}
</style>"""


DETAIL_CSS = """<style>
.st-key-dk_detail{background:#0b1424;border:1px solid #1b2540;border-radius:14px;
  padding:16px 18px 8px;margin:2px 0 4px;color:#d5deef;
  font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",-apple-system,"Segoe UI",sans-serif;font-size:12px;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;font-variant-numeric:tabular-nums;}
.st-key-dk_detail [data-testid="stVerticalBlock"]{gap:6px;}
.dd-kick{font-size:11px;font-weight:800;letter-spacing:1.4px;color:#5b8cff;}
.dd-ttl{font-size:17px;font-weight:800;color:#fff;margin:2px 0 3px;}
.dd-sub{font-size:12px;color:#8496b8;line-height:1.65;}
.dd-mx{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 4px;}
.dd-m{background:#0e1728;border:1px solid #1a2540;border-radius:9px;
  padding:7px 11px;min-width:104px;}
.dd-m b{display:block;font-size:15px;font-weight:800;color:#e8f0ff;
  font-variant-numeric:tabular-nums;}
.dd-m span{font-size:10px;color:#6b7a9b;}
.dd-m.r b{color:#ff9d9d;} .dd-m.g b{color:#9be3bd;} .dd-m.b b{color:#a9c6ff;}
.dd-m.o b{color:#f0c060;} .dd-m.y b{color:#c9b6ff;} .dd-m.dim b{color:#7b8bac;}
.dd-h{font-size:11px;font-weight:800;letter-spacing:1.2px;color:#6b7c9d;
  margin:14px 0 7px;padding-top:10px;border-top:1px solid #16203a;}
.dd-h:first-of-type{border-top:0;padding-top:0;}
.dd-tbl{width:100%;border-collapse:collapse;font-size:12px;}
.dd-tbl th{text-align:left;font-size:10px;font-weight:800;letter-spacing:.6px;
  color:#6b7c9d;padding:4px 8px 6px;border-bottom:1px solid #1a2540;}
.dd-tbl td{padding:5px 8px;border-bottom:1px dashed #16203a;color:#a9b8d4;
  vertical-align:top;max-width:380px;overflow:hidden;text-overflow:ellipsis;}
.dd-tbl td:first-child{color:#dbe6f8;font-weight:600;}
.dd-more{font-size:10px;color:#6b7c9d;margin-top:6px;}
.dd-empty{font-size:12px;color:#7b8bac;padding:6px 0 10px;line-height:1.65;}
.dd-code{background:#080e1b;border:1px solid #16203a;border-radius:8px;
  padding:9px 11px;font-family:ui-monospace,Consolas,monospace;font-size:11px;
  color:#a9bcd8;white-space:pre-wrap;line-height:1.65;max-height:280px;overflow:auto;}
.dd-split{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.dd-card{background:#0e1728;border:1px solid #1a2540;border-radius:9px;padding:9px 11px;}
.dd-card h5{margin:0 0 5px;font-size:11px;color:#8ab0ff;font-weight:800;}
.dd-card .dd-loc{font-size:10px;color:#7b8bac;margin-bottom:4px;}
/* 论文原文片段用正文字形（2026-10-27 定稿）：原句多是中文＋公式，等宽显得又宽又难读；
   等宽只留给真正的代码与程序日志（上面的 .dd-code）。 */
.dd-card .dd-snip{font-size:12px;color:#c3d1e8;line-height:1.65;
  background:rgba(255,255,255,.03);border-radius:6px;padding:6px 8px;
  word-break:break-word;}
.dd-tip{font-size:11px;color:#f0c060;background:rgba(247,160,32,.08);
  border:1px solid rgba(247,160,32,.22);border-radius:8px;padding:7px 10px;margin-top:8px;}
/* ---- 弹窗本体（挂在 body 下，拿不到页面那套深色，必须单独上色） ---- */
/* ⚠ 2026-10-27 查清了结构：`stDialog` 本身是**全屏 fixed 遮罩**（1680×1200，padding 32px），
   卡片是它的第一个子 div。默认遮罩是 rgba(150,176,202,.25) 的浅蓝灰 ⇒ 一开弹窗
   整页被蒙上一层白，看着像"字体发灰、背景不统一"。所以两件事必须分开做：
     ① 遮罩改成半透明深色（后面的页面还能隐约看见，这才是弹窗该有的观感）；
     ② 卡片照旧涂深 + 细边框。
   只把遮罩涂成不透明深色会变成"整页全黑、只剩一张卡"，别那么写。 */
div[data-testid="stDialog"]{background:rgba(4,9,18,.72) !important;}
div[data-testid="stDialog"] > div{background:#0b1424 !important;
  border:1px solid #1e2b47 !important;border-radius:16px !important;
  box-shadow:0 24px 60px rgba(0,0,0,.6) !important;}
div[data-testid="stDialog"] h2{color:#e8f0ff !important;font-size:16px !important;}
div[data-testid="stDialog"] [data-testid="stCaptionContainer"] p{color:#7b8bac !important;}
div[data-testid="stDialog"] .stButton>button{
  background:#0f1a2d !important;color:#a9bcd8 !important;
  border:1px solid #22304d !important;border-radius:9px !important;}
div[data-testid="stDialog"] .stButton>button:hover{
  border-color:#3b5286 !important;color:#dbe6f8 !important;}
@media (max-width:900px){.dd-split{grid-template-columns:1fr;}}
</style>"""


# ---------------------------------------------------------------- 小工具
def _tbl(headers, rows, empty: str = "本次没有数据", cap: int = 60) -> str:
    """保留旧名字（别处按需引用），实现挪到 deckviews，口径只有一处。"""
    from .deckviews import tbl
    return tbl(headers, rows, empty, cap)


def hotspot(key: str, html: str) -> bool:
    """一段 HTML + 一个盖满它的透明按钮。返回是否被点。"""
    with st.container(key=f"dkhot_{key}"):
        st.html(html)
        # label 用「›」而不是空串：空 label 的按钮在部分 Streamlit 版本里会被判成 0 宽，
        # 而它本来就 opacity:0 看不见，可见与否跟点不点到无关。
        # ⚠ 不把 help 交给 Streamlit 的 help：那是 hover 就弹的大浮层，
        #   鼠标划过每一行都会挡住四站带，实际用起来很烦；点击含义靠行尾「详情 ›」说明。
        return st.button("›", key=f"dkbtn_{key}", use_container_width=True)


def _go(focus: str) -> None:
    st.session_state[FOCUS_KEY] = focus
    st.rerun()


def _sel(focus: str, want: str) -> str:
    """当前详情是不是这一块 —— 是就返回 ' sel'，让 CSS 把它点亮。"""
    return " sel" if (focus or "") == want else ""


# ---------------------------------------------------------------- 三栏大屏
def render_deck(audit=None, *, paper_name: str = "", code_name: str = "",
                results_name: str = "", elapsed: float = 0.0,
                planner: str = "", hosted: bool = False) -> None:
    """渲染可点击指挥台。任何一块被点 -> 写 focus -> rerun 弹出详情。

    hosted=True 表示「已经被外层深色壳托管」（app.py 的主界面就是这么用的）：
    不再重复输出顶栏 —— 顶栏由外层画一次（状态胶囊要在跑的过程中变），
    否则屏幕上会并排两条 Paper2Lab 标题栏。
    """
    st.html(shell_css())
    focus = str(st.session_state.get(FOCUS_KEY) or "")
    ex = (getattr(audit, "execution_log", None) or {}) if audit is not None else {}
    steps = list(ex.get("steps") or [])

    done = {s.get("key") for s in steps if (s.get("status") or "") == "done"}
    done_cp: set = set()
    for s in steps:
        if (s.get("status") or "") == "done":
            done_cp.update(s.get("checkpoints") or ())
    for t in TOOLS:
        if t.key in done:
            done_cp.update(t.checkpoints)
    settled = set(done) | {s.get("key") for s in steps
                           if (s.get("status") or "") == "skipped"}

    n_all = len(STEPS)
    n_done = len(done_cp & set(STEPS))

    # ---- 顶栏
    _bits = [x for x in (paper_name, code_name) if x]
    sess = (" × ".join(_e(b) for b in _bits) + f" · {n_all} 格") if _bits else "会话 · 单次"
    _pill = ("ok", "● 已完成") if audit is not None else ("warn", "● 待运行")
    top = (f"<div class='dk-top'><div class='dk-lg'>Paper<b>2</b>Lab</div>"
           f"<div class='dk-pill {_pill[0]}'>{_pill[1]}</div>"
           f"<div class='dk-sid'>{sess}</div></div>")

    with st.container(key="dk_shell"):
        if hosted:
            st.html("<div style='height:10px'></div>")
        else:
            st.html(top)
        cL, cM, cR = st.columns([1.02, 3.0, 1.52], gap="small")

        # ================================================== 左：材料 + 七位置
        with cL:
            st.html("<div class='dk-h6' style='padding:14px 12px 8px'>材 料</div>")
            mats = []
            if paper_name:
                pc = getattr(getattr(audit, "paper", None), "page_count", 0) or 0
                mats.append(("paper", "PDF", paper_name,
                             f"{pc} 页 · 已解析" if pc else "已解析", "b"))
            if code_name:
                nf = getattr(getattr(audit, "code", None), "python_files", 0) or 0
                mats.append(("code", "ZIP", code_name,
                             f"{nf} 个文件 · 已扫描" if nf else "已扫描", "g"))
            if results_name:
                mats.append(("result", "JSN", results_name, "已载入", "o"))
            if not mats:
                mats.append(("none", "—", "尚未载入", "上传论文后开始", "x"))
            for kind, tag, name, sub, tone in mats:
                card = (f"<div class='dk-mat{_sel(focus, f'mat:{kind}')}'>"
                        f"<div class='dk-mic {tone}'>{tag}</div>"
                        f"<div class='dk-mtx'><div class='dk-mnm'>{_e(name)}</div>"
                        f"<div class='dk-msub'>{_e(sub)}"
                        f"<span class='dk-hot-arrow'>查看 ›</span></div>"
                        f"</div></div>")
                if hotspot(f"mat_{kind}", card):
                    _go(f"mat:{kind}")

            st.html("<div class='dk-h6 dk-mt' style='padding:0 12px'>七 位 置</div>")
            for p in POSITIONS:
                word, tone = _pos_state(p.key, done, "", ex, audit)
                dot = {"run": "#5b8cff", "done": "#3f8f68", "idle": "#2a3550"}[tone]
                cls = " act" if tone == "run" else ""
                row = (f"<div class='dk-mem{cls}{_sel(focus, f'pos:{p.key}')}'>"
                       f"<span class='dk-dot' style='background:{dot}'></span>"
                       f"<span class='dk-mn'>{_e(p.cn)}</span>"
                       f"<span class='dk-rt {tone}'>{_e(word)}</span></div>")
                if hotspot(f"pos_{p.key}", row):
                    _go(f"pos:{p.key}")

        # ================================================== 中：四站 + 工序
        with cM:
            station_tools: dict = {s: [] for s in STATIONS}
            st_fail = {s: False for s in STATIONS}
            for t in TOOLS:
                _s = STATION_OF_STEP.get((t.checkpoints or ("",))[0], "")
                if _s in station_tools:
                    station_tools[_s].append(t.key)
            for s in steps:
                if (s.get("status") or "") != "failed":
                    continue
                for cp in (s.get("checkpoints") or ()):
                    _s = STATION_OF_STEP.get(cp, "")
                    if _s in st_fail:
                        st_fail[_s] = True
            cells = _station_cells(done_cp)

            st.html("<div style='height:14px'></div>")
            stg_cols = st.columns(len(STATIONS), gap="small")
            for i, stg in enumerate(STATIONS):
                walked, total = cells.get(stg, [0, 0])
                keys = station_tools.get(stg, [])
                if st_fail.get(stg):
                    cls = "bad"
                elif keys and all(k in settled for k in keys):
                    cls = "done"
                elif walked:
                    cls = "done"
                else:
                    cls = ""
                cls += _sel(focus, f"station:{stg}")
                ytxt = ""
                if audit is not None:
                    for _k in _ST_ORDER.get(stg, ()):
                        y = _yield_of(_k, audit)
                        if y:
                            ytxt = " · " + y
                            break
                sub = f"{walked} / {total} 格{ytxt}" if total else "本版无环节"
                card = (f"<div class='dk-stg {cls}'><div class='dk-stt'>{_e(stg)}</div>"
                        f"<div class='dk-stm'>{_e(sub)}"
                        f"<span class='dk-hot-arrow'>进与出 ›</span></div></div>")
                with stg_cols[i]:
                    if hotspot(f"stg{i}", card):
                        _go(f"station:{stg}")

            st.html(f"<div class='dk-h6' style='padding:14px 2px 8px'>工 序 执 行"
                    f"<span class='dk-cnt'>{len(TOOLS)} 件 · 每行都能点</span></div>")
            stat_of = {s.get("key"): s for s in steps}
            for idx, t in enumerate(TOOLS, start=1):
                rec = stat_of.get(t.key) or {}
                stt = (rec.get("status") or "") or ("done" if t.key in done else "wait")
                sec = rec.get("seconds") or 0
                y = _yield_of(t.key, audit)
                bits = []
                if y:
                    bits.append(f"<b>{_e(y)}</b>")
                if sec:
                    bits.append(f"{sec:.1f}s")
                if stt == "skipped":
                    bits = ["<span class='dk-warn'>计划跳过</span>"]
                elif stt == "failed":
                    bits = ["<span class='dk-bad'>执行失败</span>"]
                elif stt == "wait":
                    bits = ["<span class='dk-dim'>未执行</span>"]
                meta = " · ".join(bits)
                arrow = "<span class='dk-hot-arrow'>详情 ›</span>"
                row = (f"<div class='dk-fl {stt}{_sel(focus, f'tool:{t.key}')}'>"
                       f"<span class='dk-n'>{idx:02d}</span>"
                       f"<span class='dk-fnm'>{_e(t.title)}</span>"
                       f"<span class='dk-fmeta'>{meta}{arrow}</span></div>")
                if hotspot(f"row_{t.key}", row):
                    _go(f"tool:{t.key}")

        # ================================================== 右：事件 + 判定 + 进度
        with cR:
            # 标题本身就是一个热区：点开是整轮的时间线（谁先跑、谁最慢、四站各占多久）
            head = (f"<div class='dk-h6' style='padding:14px 2px 8px'>实 时 事 件"
                    f"<span class='dk-cnt'>{len(steps)} 条</span>"
                    f"<span class='dk-hot-arrow'>时间线 ›</span></div>")
            if hotspot("evt_all", head):
                _go("event:all")

            numbered = list(enumerate(steps))[-7:]
            if not numbered:
                st.html("<div class='dk-evt dk-dim'>还没有执行记录 —— "
                        "点左上角开始审计后，这里会逐条长出来。"
                        "<span class='dk-hot-arrow'>时间线 ›</span></div>")
            for i, s in numbered:
                stt = s.get("status") or ""
                mark = {"done": "ok", "skipped": "warn", "failed": "bad"}.get(stt, "")
                sec = s.get("seconds") or 0
                tm = f"{sec:.1f}s" if sec >= 0.05 else ""
                txt = s.get("reason") or _yield_of(s.get("key") or "", audit) or ""
                txt = f"· {txt}" if txt else ""
                row = (f"<div class='dk-evt{_sel(focus, f'event:{i}')}'>"
                       f"<span class='dk-etm'>{_e(tm)}</span>"
                       f"<b>{_e(s.get('title') or '')}</b>"
                       f"<span class='{mark}'>{_e(txt[:24])}</span></div>")
                if hotspot(f"evt{i}", row):
                    _go(f"event:{i}")

            st.html("<div class='dk-h6 dk-mt' style='padding:0 2px 8px'>判 定 计 数"
                    "<span class='dk-cnt'>点开看分布</span></div>")
            vc = _verdict_counts(audit)
            if vc:
                # 4 类以上排两列：右栏只有 270 来像素，排三列会把
                # 「论文有·代码无」这种长标签截断（2026-10-27 真机看到过）。
                _ncol = 2 if len(vc) > 3 else min(len(vc), 3)
                bcols = st.columns(_ncol, gap="small")
                for i, (lab, n, tone) in enumerate(vc):
                    with bcols[i % len(bcols)]:
                        b = (f"<div class='dk-badge {tone}{_sel(focus, f'verdict:{tone}')}'"
                             f" style='display:block;text-align:center;padding:5px 4px'>"
                             f"{_e(lab)} <b>{n}</b></div>")
                        if hotspot(f"vd_{tone}", b):
                            _go(f"verdict:{tone}")
            else:
                st.html("<span class='dk-badge dim'>本次无判定项</span>")

            st.html("<div class='dk-h6 dk-mt' style='padding:0 2px 8px'>复 现 就 绪 度"
                    "<span class='dk-cnt'>点开看怎么算</span></div>")
            _rdy = _ready_block(audit).replace("<div class='dk-rdy'",
                                               f"<div class='dk-rdy{_sel(focus, 'ready:score')}'", 1)
            if hotspot("ready", _rdy):
                _go("ready:score")

            bar = ""
            for s in STEPS:
                bar += "<i class='f'></i>" if s in done_cp else "<i></i>"
            st.html(f"<div class='dk-prog'><div class='dk-bar'>{bar}</div>"
                    f"<div class='dk-plab'><span>已走 {n_done} / {n_all} 格</span>"
                    f"<span>{('已用 %.1fs' % elapsed) if elapsed else ''}</span>"
                    f"</div></div><div style='height:14px'></div>")


# ---------------------------------------------------------------- 详情弹窗
_KICK = {"tool": "工 序 详 情", "mat": "材 料 详 情", "pos": "位 置 详 情",
         "station": "站 点 详 情", "verdict": "判 定 明 细",
         "event": "时 间 线", "ready": "就 绪 度"}


@st.dialog("详情", width="large")
def _detail_dialog(kick: str, title: str, sub: str, metrics, blocks) -> None:
    """弹出式详情。

    为什么是弹窗而不是「画在大屏下面」：三栏大屏本身就长，点第 9 行工序时
    详情落在屏幕外，用户会以为「点了没反应」。弹窗不管页面滚到哪都在眼前。
    """
    with st.container(key="dk_detail"):
        st.html(DETAIL_CSS)
        st.html(VIEW_CSS)
        mx = "".join(f"<div class='dd-m {t}'><b>{_e(v)}</b><span>{_e(l)}</span></div>"
                     for l, v, t in metrics)
        body = "".join(f"<div class='dd-h'>{_e(h)}</div>{html}" for h, html in blocks)
        st.html(f"<div class='dd-kick'>{_e(kick)}</div>"
                f"<div class='dd-ttl'>{_e(title)}</div>"
                f"<div class='dd-sub'>{sub}</div><div class='dd-mx'>{mx}</div>{body}")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("收起详情", key="dk_close", use_container_width=True):
                st.session_state.pop(FOCUS_KEY, None)
                st.rerun()
        with c2:
            st.caption("这里只摊开已有数据；任何判定都在别处由规则算好，不在这里改分数。")


def render_detail(audit, focus: str) -> None:
    """按 focus 弹出对应区域的详情。

    focus 形如 tool:align / mat:paper / pos:parser / station:① 假设 /
    verdict:r / event:3 / event:all / ready:score。

    每块区域走**各自**的视图函数（见 deckviews.py）：位置看流程、四站看边界、
    事件看时间、判定看分布 —— 数据源有重叠，讲的事情不重叠。
    """
    kind, _, val = (focus or "").partition(":")
    if not kind:
        return
    try:
        if kind == "tool":
            title, sub, metrics, blocks = tool_view(audit, val)
        elif kind == "mat":
            title, sub, metrics, blocks = mat_view(audit, val)
        elif kind == "station":
            title, sub, metrics, blocks = station_view(audit, val)
        elif kind == "pos":
            title, sub, metrics, blocks = pos_view(audit, val)
        elif kind == "verdict":
            title, sub, metrics, blocks = verdict_view(audit, val)
        elif kind == "event":
            if val in ("", "all"):
                title, sub, metrics, blocks = timeline_view(audit)
            else:
                title, sub, metrics, blocks = event_view(audit, int(val))
        elif kind == "ready":
            title, sub, metrics, blocks = ready_view(audit)
        else:
            return
    except Exception as exc:                      # noqa: BLE001
        # 详情画不出来**绝不允许连累大屏** —— 只把这一块降级成一则说明。
        title, sub, metrics = "这一块暂时打不开", "", []
        blocks = [("原因", f"<div class='dd-empty'>读取这一块的数据时出错："
                           f"{_e(type(exc).__name__)}: {_e(str(exc)[:160])}"
                           f"<br/>大屏和其它区域不受影响。</div>")]
    _detail_dialog(_KICK.get(kind, "详 情"), title, sub, metrics, blocks)
