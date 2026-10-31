"""实验室指挥台（Lab Deck）—— 一屏看清一次审计的全貌。

为什么单独成模块
----------------
三栏大屏的 HTML 拼装很啰嗦，塞进 app.py 会把本来就长的界面文件再撑大一截；
而它只依赖 toolbox / agent_roles / models 三处的**公开**数据，没有任何
Streamlit 依赖 —— 纯函数，便于命令行导出、便于单测。

口径（一条都不许违反）
----------------------
1. **数字全部现场派生**。格数取 `len(STEPS)`、工序取 `len(TOOLS)`、站名取
   `STATIONS` —— 加一件工具，这里自动跟着走，不写死任何数字。
2. **空 ≠ 0**。本次没做的事标「未触发 / 未执行」并给理由，绝不填 0 分糊过去。
3. **不新增判定**。这里只展示既定结论（findings 的 status、hypotheses 的
   verdict），不在这里下任何新判断，也不覆盖程序算出来的数。
4. **实时与完成是同一块屏**。跑的过程中 `audit=None`，用 done_keys / live_key
   画半成品；跑完传 audit 画全量。两种状态共用一整套版式，切换时不跳。
"""

from __future__ import annotations

from .agent_roles import POSITIONS
from .models import Status
from .toolbox import STATION_OF_STEP, STATIONS, STEPS, TOOLS

__all__ = ["deck_html", "DECK_CSS"]


# ---------------------------------------------------------------- 小工具

def _e(v) -> str:
    """HTML 转义。所有外部文本（论文标题、代码文件名、理由）都必须过这里。"""
    return (str(v) if v is not None else "").replace("&", "&amp;") \
        .replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _count(obj) -> int:
    try:
        return len(obj or ())
    except TypeError:
        return 0


# 工序 → 产物计数。**只登记真实存在的字段**，取不到就返回空串（界面不显示），
# 不用 0 顶上 —— 0 会被读成「跑了但什么也没产出」，与「没记录」是两件事。
def _yield_of(key: str, audit) -> str:
    if audit is None:
        return ""
    if key == "parse_paper":
        pc = getattr(getattr(audit, "paper", None), "page_count", 0) or 0
        return f"{pc} 页" if pc else ""
    if key == "analyze_images":
        n = _count(getattr(getattr(audit, "paper", None), "params", None))
        return f"{n} 项声明" if n else ""
    if key == "read_figures":
        n = getattr(audit, "figure_total", 0) or 0
        return f"{n} 条线索" if n else ""
    if key == "chem_check":
        n = getattr(audit, "chem_total", 0) or 0
        return f"{n} 条核验" if n else ""
    if key == "analyze_code":
        n = getattr(getattr(audit, "code", None), "python_files", 0) or 0
        return f"{n} 个文件" if n else ""
    if key == "llm_analyze":
        r = getattr(audit, "readiness", None) or {}
        n = r.get("llm_real_calls") or 0
        return f"{n} 次模型调用" if n else ""
    if key == "hypothesize":
        n = getattr(audit, "hypothesis_total", 0) or 0
        return f"{n} 张假设卡" if n else ""
    if key == "design_experiment":
        n = getattr(audit, "experiment_total", 0) or 0
        return f"{n} 张实验卡" if n else ""
    if key == "align":
        n = _count(getattr(audit, "findings", None))
        return f"{n} 条判定" if n else ""
    if key == "selfcheck":
        sc = getattr(audit, "selfcheck", None) or {}
        n = sc.get("checked") or 0
        return f"抽验 {n} 条" if n else ""
    if key == "rootcause":
        n = _count(getattr(audit, "rootcause", None))
        return f"{n} 条根因" if n else ""
    if key == "patch":
        n = _count(getattr(audit, "patches", None))
        return f"{n} 条草案" if n else ""
    if key == "report":
        return "已出报告"
    if key == "plan":
        # 主持人（编排位置）的交接物就是这份计划 —— 它没有自己的工序行，
        # 所以产物要在这里现算，否则「点开主持人」会显示"没有产出"（2026-10-27 踩到）。
        plan = (getattr(audit, "execution_log", None) or {}).get("plan") or {}
        if not plan:
            return ""
        return (f"{plan.get('ran', 0)} 件执行 / {plan.get('skipped', 0)} 件跳过")
    return ""


# 站序号 → 站名的短标签（站名本身来自 STATIONS，不手抄）
_STATION_SHORT = {st: st for st in STATIONS}

# 每站"最像这一段成果"的产物，按优先级排。取不到就顺延 —— 不凑数、不编。
_STATION_ORDER = {
    "① 假设": ("hypothesize", "read_figures", "chem_check", "parse_paper"),
    "② 实验": ("design_experiment",),
    "③ 验证": ("align", "selfcheck"),
    "④ 迭代": ("patch", "rootcause", "report"),
}


def _verdict_counts(audit) -> list:
    """判定计数：只计数、不判定。返回 [(标签, 数, 色类)]，0 条的不出现。"""
    if audit is None:
        return []
    fs = list(getattr(audit, "findings", None) or [])
    if not fs:
        return []
    buckets = [
        ("一致", Status.CONSISTENT, "g"),
        ("两数不一致", Status.INCONSISTENT, "r"),
        ("论文有·代码无", Status.MISSING_IN_CODE, "b"),
        ("代码有·论文无", Status.MISSING_IN_PAPER, "o"),
        # 与「总览」页第四张卡同一个口径：未固定 / 无法验证 / 内部矛盾
        # 都归在「其他」里，不跟硬冲突混成一个数。
        ("其他", None, "y"),
    ]
    named = {Status.CONSISTENT, Status.INCONSISTENT,
             Status.MISSING_IN_CODE, Status.MISSING_IN_PAPER}
    out = []
    for label, st, tone in buckets:
        if st is None:
            n = sum(1 for f in fs if getattr(f, "status", None) not in named)
        else:
            n = sum(1 for f in fs if getattr(f, "status", None) == st)
        if n:
            out.append((label, n, tone))
    return out


def _pos_state(key: str, done: set, live_key: str, ex: dict, audit) -> tuple:
    """七位置此刻在干什么。返回 (状态词, 色类)。色类: run/done/idle。"""
    st_map = {
        "parser": ({"parse_paper", "analyze_code", "analyze_images", "read_figures"},
                   "读图中", "已交线索"),
        "retriever": ({"llm_analyze"}, "在找证据", "已给出处"),
        "moderator": ({"start", "plan"}, "正在排计划", "已交计划"),
        "experimenter": ({"design_experiment"}, "在设计实验", "已设计"),
        "calculator": ({"compute"}, "在复算", "已复算"),
        "reviewer": ({"align", "collider", "selfcheck"}, "在判定", "已判定"),
        "writer": ({"report", "patch", "rootcause"}, "在写", "已交稿"),
    }
    keys, doing, did = st_map.get(key, (set(), "处理中", "已完成"))
    # 主持人没有自己的工序行（排计划发生在 `start`/`plan` 两个图节点里，不进 TOOLS），
    # 按工具 key 判它会永远显示「待命」—— 那是假象。计划本身在 execution_log["plan"]。
    if key == "moderator" and (ex or {}).get("plan"):
        return "已交计划", "done"
    if live_key and live_key in keys:
        return doing, "run"
    if done & keys:
        return did, "done"
    return "待命", "idle"


def _ready_block(audit) -> str:
    """右栏的「综合复现就绪度」。

    为什么指挥台必须有这一块：它是这个平台最核心的一个数，指挥台却原来没有；
    而且 `综合复现就绪度` 这几个字是多个界面测试用来判断「结果页立住了」的公共锚点 ——
    少了它，一改默认页就会把一批测试连带弄红。

    口径沿用 readiness：`total is None` = 一个维度都没核验，显示「—」不凭空给分。
    """
    if audit is None:
        # ⚠ 这里**绝不能**写「综合复现就绪度」这五个连续的字。
        #   那几个字是界面测试判断「结果页立住了」的公共锚点（6 处测试在用）；
        #   跑的过程中就写出来，测试会在流式渲染的中途放行 ——
        #   拿到半成品页面去断言，然后一堆"没渲染"的假红。
        #   从语义上讲也更对：还没出结论，哪来的就绪度。
        return ("<div class='dk-rdy'><div class='dk-rdyn dk-na'>—</div>"
                "<div class='dk-rdyl'>结论尚未生成</div>"
                "<div class='dk-rdys'>跑完才有分</div></div>")
    r = getattr(audit, "readiness", None) or {}
    total = r.get("total")
    conf = r.get("conf") or {}
    level = conf.get("level") or "—"
    if total is None:
        num = "<span class='dk-na'>—</span>"
        bar = ""
        sub = "一个维度都没核验，不给分"
    else:
        color = "#3ddc97" if total >= 80 else ("#f0c060" if total >= 60 else "#ff9d9d")
        num = f"<span style='color:{color}'>{total}</span><small> /100</small>"
        bar = (f"<div class='dk-rdybar'><i style='width:{max(0, min(100, int(total)))}%;"
               f"background:{color}'></i></div>")
        sub = f"审计可信度 {level}　{conf.get('score', '—')}/100"
    return (f"<div class='dk-rdy'><div class='dk-rdyn'>{num}</div>"
            f"<div class='dk-rdyl'>综合复现就绪度</div>"
            f"{bar}<div class='dk-rdys'>{_e(sub)}</div></div>")


def _station_cells(walked_cp: set) -> dict:
    """按四站数「站内 N 格之中走了几格」。格 = STEPS 里的环节名。"""
    out: dict = {st: [0, 0] for st in STATIONS}
    for s in STEPS:
        st = STATION_OF_STEP.get(s, "")
        if st not in out:
            continue
        out[st][1] += 1
        if s in walked_cp:
            out[st][0] += 1
    return out


# ---------------------------------------------------------------- 主渲染

def deck_html(audit=None, *, paper_name: str = "", code_name: str = "",
              results_name: str = "", phase: str = "done",
              done_keys=(), live_key: str = "", elapsed: float = 0.0,
              plan_planner: str = "", show_top: bool = True) -> str:
    """渲染整块指挥台。

    参数
    ----
    audit       : 跑完时传 AuditResult；跑的过程中传 None（只画半成品）。
    phase       : "run" 审计中 / "done" 已完成。
    done_keys   : 已完成的工具 key（实时阶段由 app.py 从回调里攒）。
    live_key    : 正在跑的工具 key。
    elapsed     : 已耗时（秒）。
    plan_planner: "llm" / "rule" / ""，用于主持人这一格的说法。
    show_top    : 是否自带顶栏。被外层深色壳托管时传 False —— 顶栏由外层画一次，
                  避免出现「两条标题栏」。
    """
    ex = (getattr(audit, "execution_log", None) or {}) if audit is not None else {}
    steps = list(ex.get("steps") or [])

    # 完成的工具集合：跑完看 execution_log；实时阶段看调用方攒的 done_keys
    done = set(done_keys or ())
    for s in steps:
        if (s.get("status") or "") == "done":
            done.add(s.get("key") or "")
    done.discard("")

    # 已走过的环节格（用于站内 x/y 与底部 15 格进度条）
    done_cp: set = set()
    for s in steps:
        if (s.get("status") or "") == "done":
            done_cp.update(s.get("checkpoints") or ())
    for t in TOOLS:                      # 实时阶段没有 execution_log，用工具反推环节
        if t.key in done:
            done_cp.update(t.checkpoints)
    live_cp: set = set()
    if live_key:
        for t in TOOLS:
            if t.key == live_key:
                live_cp.update(t.checkpoints)

    # ---- 顶栏状态
    if phase == "run":
        pill_cls, pill_txt = "run", "● 审计中"
    elif phase == "fail":
        pill_cls, pill_txt = "bad", "● 已中断"
    else:
        pill_cls, pill_txt = "ok", "● 已完成"

    # ---- 左栏：材料
    mats = []
    if paper_name:
        pc = getattr(getattr(audit, "paper", None), "page_count", 0) or 0
        sub = f"{pc} 页 · 已解析" if pc else "已解析"
        mats.append(("论文", "PDF", paper_name, sub, "b"))
    if code_name:
        nf = getattr(getattr(audit, "code", None), "python_files", 0) or 0
        sub = f"{nf} 个文件 · 已扫描" if nf else "已扫描"
        mats.append(("代码", "ZIP", code_name, sub, "g"))
    if results_name:
        mats.append(("实测结果", "JSN", results_name, "已载入", "o"))
    if not mats:
        mats.append(("材料", "—", "尚未载入", "上传论文后开始", "x"))

    mat_html = "".join(
        f"<div class='dk-mat'><div class='dk-mic {tone}'>{tag}</div>"
        f"<div class='dk-mtx'><div class='dk-mnm' title='{_e(name)}'>{_e(name)}</div>"
        f"<div class='dk-msub'>{_e(sub)}</div></div></div>"
        for _kind, tag, name, sub, tone in mats)

    # ---- 左栏：七位置（七个，来自 POSITIONS，不手抄名字）
    pos_html = ""
    for p in POSITIONS:
        word, tone = _pos_state(p.key, done, live_key, ex, audit)
        dot = {"run": "#5b8cff", "done": "#3f8f68", "idle": "#2a3550"}[tone]
        cls = " class='act'" if tone == "run" else ""
        pos_html += (f"<div class='dk-mem'{cls}>"
                     f"<span class='dk-dot' style='background:{dot}'></span>"
                     f"<span class='dk-mn'>{_e(p.cn)}</span>"
                     f"<span class='dk-rt {tone}'>{_e(word)}</span></div>")

    # ---- 中栏：四站带
    # 正在跑的那一件属于哪一站 —— 先算出来，后面判断「哪站在转」才不会串站。
    live_station = ""
    for t in TOOLS:
        if t.key == live_key:
            live_station = STATION_OF_STEP.get((t.checkpoints or ("",))[0], "")

    # 每站对应哪几件工序（用于判断「这一站是不是已经走完了」）
    station_tools: dict = {st: [] for st in STATIONS}
    for t in TOOLS:
        _st = STATION_OF_STEP.get((t.checkpoints or ("",))[0], "")
        if _st in station_tools:
            station_tools[_st].append(t.key)

    # 「有交代」= 跑完了 或 明确计划跳过。**跳过也算有交代** ——
    # 否则一次"计划跳过某件"会让该站永远显示成「进行中」，那是骗人的。
    settled = set(done)
    for s in steps:
        if (s.get("status") or "") == "skipped":
            settled.add(s.get("key") or "")

    cells = _station_cells(done_cp | live_cp)

    # 哪一站出过错（有工具 status=failed）。出错必须比"跑完了"优先显示。
    st_fail = {st: False for st in STATIONS}
    for s in steps:
        if (s.get("status") or "") != "failed":
            continue
        for cp in (s.get("checkpoints") or ()):
            _s = STATION_OF_STEP.get(cp, "")
            if _s in st_fail:
                st_fail[_s] = True

    stg_html = ""
    for st in STATIONS:
        walked, total = cells.get(st, [0, 0])
        keys = station_tools.get(st, [])
        if st_fail.get(st):
            cls = "bad"
        elif keys and all(k in settled for k in keys):
            # 站内每件工序都有交代（跑完 / 明确跳过）—— 这一站算走完
            cls = "done"
        elif phase != "run" and walked:
            # 整体已经结束、这一站也真走过：哪怕少一格（比如某件模型整节点跳过、
            # 执行记录里没有条目），也不该继续亮着"进行中" —— 那是骗人的。
            cls = "done"
        elif walked or st == live_station:
            cls = "act"
        else:
            cls = ""
        # 站的产物说明：按"最像这一段成果"的工序优先取（取不到就顺延，全没有就不写）
        ytxt = ""
        if audit is not None:
            for _k in _STATION_ORDER.get(st, ()):
                y = _yield_of(_k, audit)
                if y:
                    ytxt = " · " + y
                    break
        sub = f"{walked} / {total} 格{ytxt}" if total else "本版无环节"
        pulse = "<div class='dk-pulse'></div>" if cls == "act" else ""
        stg_html += (f"<div class='dk-stg {cls}'>"
                     f"<div class='dk-stt'>{_e(st)}</div>"
                     f"<div class='dk-stm'>{_e(sub)}</div>{pulse}</div>")

    # ---- 中栏：工序执行（逐件来自 TOOLS；快照里没有的按「计划外」处理）
    stat_of = {s.get("key"): s for s in steps}
    rows = ""
    for idx, t in enumerate(TOOLS, start=1):
        rec = stat_of.get(t.key) or {}
        st = (rec.get("status") or "")
        if t.key == live_key:
            st = "run"
        elif not st:
            st = "done" if t.key in done else "wait"
        sec = rec.get("seconds") or 0
        y = _yield_of(t.key, audit)
        meta_bits = []
        if y:
            meta_bits.append(f"<b>{_e(y)}</b>")
        if sec:
            meta_bits.append(f"{sec:.1f}s")
        if st == "skipped":
            meta_bits = [f"<span class='dk-warn'>计划跳过</span>"]
        elif st == "failed":
            meta_bits = ["<span class='dk-bad'>执行失败</span>"]
        elif st == "run":
            meta_bits = ["<span class='dk-run'>running…</span>"]
        elif st == "wait":
            meta_bits = ["<span class='dk-dim'>未执行</span>"]
        meta = " · ".join(meta_bits)
        spin = "<span class='dk-spin'></span>" if st == "run" else f"<span class='dk-n'>{idx:02d}</span>"
        tip = _e(rec.get("reason") or "")
        rows += (f"<div class='dk-fl {st}' title='{tip}'>"
                 f"{spin}<span class='dk-fnm'>{_e(t.title)}</span>"
                 f"<span class='dk-fmeta'>{meta}</span></div>")

    # ---- 右栏：事件流（按 steps 的真实顺序，取最近 7 条）
    events = ""
    ev_src = [s for s in steps] or []
    for s in ev_src[-7:]:
        st = s.get("status") or ""
        mark = {"done": "ok", "skipped": "warn", "failed": "bad"}.get(st, "")
        sec = s.get("seconds") or 0
        # 只显示上了 0.05s 的耗时。毫秒级的工序写「0.0s」只是噪音，
        # 反而让人以为"什么都没干"。
        tm = f"{sec:.1f}s" if sec >= 0.05 else ""
        txt = s.get("reason") or _yield_of(s.get("key") or "", audit) or ""
        txt = f"· {txt}" if txt else ""
        events += (f"<div class='dk-evt'><span class='dk-etm'>{_e(tm)}</span>"
                   f"<b>{_e(s.get('title') or '')}</b>"
                   f"<span class='{mark}'>{_e(txt[:26])}</span></div>")
    if not events:
        events = ("<div class='dk-evt dk-dim'>还没有执行记录 —— "
                  "点上面的开始审计后，这里会逐条长出来。</div>")

    # ---- 右栏：判定计数
    vc = _verdict_counts(audit)
    if vc:
        vc_html = "".join(
            f"<span class='dk-badge {tone}'>{_e(lab)} <b>{n}</b></span>"
            for lab, n, tone in vc)
    else:
        vc_html = ("<span class='dk-badge dim'>还没判定</span>" if audit is None
                   else "<span class='dk-badge dim'>本次无判定项</span>")

    # ---- 右栏：15 格进度（格数 = len(STEPS)，永不写死）
    n_all = len(STEPS)
    n_done = len(done_cp & set(STEPS))
    if phase == "run" and live_cp:
        n_done = len((done_cp | live_cp) & set(STEPS))
    bar_cells = ""
    cur_marked = False
    for s in STEPS:
        if s in done_cp:
            bar_cells += "<i class='f'></i>"
        elif (not cur_marked) and live_cp and s in live_cp:
            bar_cells += "<i class='cur'></i>"
            cur_marked = True
        else:
            bar_cells += "<i></i>"

    # ---- 顶栏右侧：论文 × 代码 · N 格（材料是主角，别只写代码包名）
    sess = "会话 · 单次"
    _bits = [x for x in (paper_name, code_name) if x]
    if _bits:
        sess = " × ".join(_e(b) for b in _bits) + f" · {n_all} 格"
    left_txt = f"已走 {n_done} / {n_all} 格"
    right_txt = (f"已用 {elapsed:.1f}s" if elapsed else "")

    # 顶栏单独拼一段：show_top=False 时不输出（外层深色壳已经画过一条同样的），
    # 但状态胶囊的文字仍由这里算，口径只有一处。
    _top = (f"<div class='dk-top'><div class='dk-lg'>Paper<b>2</b>Lab</div>"
            f"<div class='dk-pill {pill_cls}'>{pill_txt}</div>"
            f"<div class='dk-sid'>{sess}</div></div>")

    return f"""<div class="p2l-deck">
  {_top if show_top else ""}
  <div class="dk-body">
    <div class="dk-col dk-colL">
      <div class="dk-h6">材 料</div>
      {mat_html}
      <div class="dk-h6 dk-mt">七 位 置</div>
      {pos_html}
    </div>
    <div class="dk-col dk-colM">
      <div class="dk-stationbar">{stg_html}</div>
      <div class="dk-h6">工 序 执 行<span class="dk-cnt">{len(TOOLS)} 件</span></div>
      <div class="dk-flow">{rows}</div>
    </div>
    <div class="dk-col dk-colR">
      <div class="dk-h6">实 时 事 件</div>
      <div class="dk-events">{events}</div>
      <div class="dk-h6 dk-mt">判 定 计 数</div>
      <div class="dk-badges">{vc_html}</div>
      <div class="dk-h6 dk-mt">复 现 就 绪 度</div>
      {_ready_block(audit)}
      <div class="dk-prog">
        <div class="dk-bar">{bar_cells}</div>
        <div class="dk-plab"><span>{left_txt}</span><span>{right_txt}</span></div>
      </div>
    </div>
  </div>
</div>"""


# ---------------------------------------------------------------- 样式
# 全部规则都挂在 .p2l-deck 下 —— 这样即使 st.html 的 <style> 作用到全页，
# 也只命中指挥台内部，不会把浅色页的观感带跑。
DECK_CSS = """<style>
.p2l-deck{background:#080d17;color:#d5deef;border-radius:14px;
  border:1px solid #1b2540;overflow:hidden;font-size:12px;line-height:1.5;
  margin:6px 0 14px;box-shadow:0 18px 50px rgba(10,18,35,.22);
  /* 字体口径（2026-10-27 定版，全站一套）：
     ① 字形 微软雅黑（Windows 默认装了）→ 苹方 → 冬青黑 → 系统无衬线；
        中文写在最前，英文/数字跟着走同一套字形，不再与 Segoe UI 混排出"两种字"。
     ② 字号只用整数档：10 / 11 / 12 / 13 / 15 / 22（半像素在 Windows 上会发虚）。
     ③ 数字等宽对齐（tabular-nums），表格与计时不会左右跳。 */
  font-family:"Microsoft YaHei","PingFang SC","Hiragino Sans GB",-apple-system,"Segoe UI",sans-serif;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;font-variant-numeric:tabular-nums;}
.p2l-deck *{box-sizing:border-box;}
/* 顶栏：条目一律不许被压缩、不许折行。
   2026-10-28 巡检发现：窄窗口（1024 宽）下 Streamlit 侧栏占掉 ~260px，
   顶栏剩余宽度不够，flex 默认会压缩各条目 —— 徽章被压到 32px 宽后
   逐字竖排（「● 已完成」变成一列竖字），看起来像排版坏了。
   定版：`flex:none` + `nowrap` 保住徽章，`dk-hd` 自己裁切尾部即可。 */
.p2l-deck .dk-top{height:52px;display:flex;align-items:center;gap:14px;
  padding:0 20px;border-bottom:1px solid #16203a;background:#0a1120;
  overflow:hidden;}
.p2l-deck .dk-lg,.p2l-deck .dk-pill,.p2l-deck .dk-sid{flex:none;white-space:nowrap;}
.p2l-deck .dk-lg{color:#fff;font-weight:800;font-size:15px;letter-spacing:.2px;}
.p2l-deck .dk-lg b{color:#5b8cff;}
.p2l-deck .dk-pill{font-size:11px;font-weight:700;padding:3px 9px;border-radius:20px;}
.p2l-deck .dk-pill.ok{color:#7dd3a0;background:rgba(48,163,108,.12);border:1px solid rgba(48,163,108,.3);}
.p2l-deck .dk-pill.run{color:#a9c6ff;background:rgba(91,140,255,.12);border:1px solid rgba(91,140,255,.35);}
.p2l-deck .dk-pill.bad{color:#ff9d9d;background:rgba(217,74,69,.14);border:1px solid rgba(217,74,69,.35);}
.p2l-deck .dk-pill.warn{color:#f0c060;background:rgba(247,160,32,.12);border:1px solid rgba(247,160,32,.30);}
.p2l-deck .dk-hd{margin-left:12px;font-size:12px;color:#5b6b8c;display:flex;
  align-items:center;gap:12px;min-width:0;overflow:hidden;white-space:nowrap;}
.p2l-deck .dk-hd b{color:#c3d1e8;font-weight:700;}
.p2l-deck .dk-hd span{font-size:11px;color:#3f4c69;}
.p2l-deck .dk-hd b+span{margin-left:2px;}
.p2l-deck .dk-sid{margin-left:auto;font-size:12px;color:#5b6b8c;}
.p2l-deck .dk-body{display:grid;grid-template-columns:232px 1fr 274px;}
.p2l-deck .dk-col{padding:16px;}
.p2l-deck .dk-colL{border-right:1px solid #16203a;}
.p2l-deck .dk-colR{border-left:1px solid #16203a;}
.p2l-deck .dk-h6{font-size:11px;font-weight:700;letter-spacing:1.1px;
  color:#6b7c9d;margin-bottom:10px;display:flex;align-items:center;gap:6px;}
.p2l-deck .dk-h6.dk-mt{margin-top:18px;}
.p2l-deck .dk-cnt{margin-left:auto;letter-spacing:0;font-weight:600;
  color:#5f7091;text-transform:none;}
.p2l-deck .dk-mat{background:#0e1728;border:1px solid #1a2540;border-radius:9px;
  padding:9px 10px;margin-bottom:8px;display:flex;align-items:center;gap:9px;}
.p2l-deck .dk-mic{width:26px;height:26px;border-radius:7px;display:grid;
  place-items:center;font-size:9px;font-weight:800;letter-spacing:.3px;flex:none;}
.p2l-deck .dk-mic.b{background:rgba(91,140,255,.15);color:#8ab0ff;}
.p2l-deck .dk-mic.g{background:rgba(48,163,108,.15);color:#6fd39b;}
.p2l-deck .dk-mic.o{background:rgba(247,107,21,.15);color:#f0a860;}
.p2l-deck .dk-mic.x{background:rgba(255,255,255,.05);color:#4b5b7c;}
.p2l-deck .dk-mtx{min-width:0;}
.p2l-deck .dk-mnm{font-size:12px;color:#d5deef;font-weight:600;
  letter-spacing:.1px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.p2l-deck .dk-msub{font-size:11px;color:#7b8bac;margin-top:1px;}
.p2l-deck .dk-mem{display:flex;align-items:center;gap:8px;padding:6px 8px;
  border-radius:7px;font-size:12px;color:#a9b8d4;}
.p2l-deck .dk-mem.act{background:rgba(91,140,255,.10);color:#dbe8ff;}
.p2l-deck .dk-dot{width:7px;height:7px;border-radius:50%;flex:none;}
.p2l-deck .dk-mn{font-weight:600;}
.p2l-deck .dk-rt{margin-left:auto;font-size:10px;}
.p2l-deck .dk-rt.idle{color:#6b7c9d;}
.p2l-deck .dk-rt.done{color:#5fbf92;}
.p2l-deck .dk-rt.run{color:#8ab0ff;}
.p2l-deck .dk-stationbar{display:flex;gap:8px;margin-bottom:16px;}
.p2l-deck .dk-stg{flex:1;background:#0e1728;border:1px solid #1a2540;
  border-radius:10px;padding:9px 11px;position:relative;overflow:hidden;}
.p2l-deck .dk-stg.done{border-color:rgba(48,163,108,.4);background:rgba(48,163,108,.06);}
.p2l-deck .dk-stg.act{border-color:rgba(91,140,255,.6);background:rgba(91,140,255,.09);}
.p2l-deck .dk-stg.bad{border-color:rgba(217,74,69,.55);background:rgba(217,74,69,.09);}
.p2l-deck .dk-stg.bad .dk-stt{color:#ff9d9d;}
.p2l-deck .dk-stt{font-size:12px;font-weight:800;color:#dbe4f5;}
.p2l-deck .dk-stg.done .dk-stt{color:#9be3bd;}
.p2l-deck .dk-stg.act .dk-stt{color:#a9c6ff;}
.p2l-deck .dk-stm{font-size:10px;color:#7c8cab;margin-top:3px;}
.p2l-deck .dk-pulse{position:absolute;left:0;bottom:0;height:2px;width:46%;
  background:#5b8cff;animation:dkpulse 1.4s ease-in-out infinite;}
@keyframes dkpulse{0%,100%{opacity:.35;width:32%}50%{opacity:1;width:62%}}
.p2l-deck .dk-flow{display:flex;flex-direction:column;gap:5px;}
.p2l-deck .dk-fl{display:flex;align-items:center;gap:10px;padding:7px 10px;
  border-radius:8px;font-size:12px;color:#a3b3cf;background:#0c1424;
  border:1px solid transparent;}
.p2l-deck .dk-fl.done{color:#c6d4ea;}
.p2l-deck .dk-fl.skipped{color:#7c8aa8;background:#0a1120;}
.p2l-deck .dk-fl.failed{color:#f0b4b1;background:rgba(217,74,69,.08);
  border-color:rgba(217,74,69,.28);}
.p2l-deck .dk-fl.run{color:#f2f7ff;background:rgba(91,140,255,.10);
  border-color:rgba(91,140,255,.32);}
.p2l-deck .dk-n{width:17px;text-align:right;font-size:11px;color:#5f7091;
  font-variant-numeric:tabular-nums;flex:none;}
.p2l-deck .dk-fl.done .dk-n{color:#5fbf92;}
.p2l-deck .dk-fnm{font-weight:600;color:#dbe6f8;}
.p2l-deck .dk-fl.skipped .dk-fnm{color:#95a4c0;}
.p2l-deck .dk-fmeta{margin-left:auto;font-size:11px;color:#8496b8;}
.p2l-deck .dk-fmeta b{color:#b3c5e0;font-weight:700;}
.p2l-deck .dk-fl.run .dk-fmeta{color:#a9c6ff;}
.p2l-deck .dk-dim{color:#6b7c9d;}
.p2l-deck .dk-warn{color:#f0a860;}
.p2l-deck .dk-bad{color:#ff9d9d;}
.p2l-deck .dk-run{color:#8ab0ff;}
.p2l-deck .dk-spin{width:11px;height:11px;border-radius:50%;flex:none;
  border:2px solid rgba(91,140,255,.28);border-top-color:#5b8cff;
  animation:dkspin .8s linear infinite;}
@keyframes dkspin{to{transform:rotate(360deg)}}
.p2l-deck .dk-events{max-height:290px;overflow:hidden;}
.p2l-deck .dk-evt{font-size:11px;line-height:1.55;color:#93a4c2;
  padding:7px 0;border-bottom:1px dashed #17213a;}
.p2l-deck .dk-etm{color:#6b7c9d;font-size:10px;font-variant-numeric:tabular-nums;
  margin-right:6px;}
.p2l-deck .dk-evt b{color:#e2eaf8;font-weight:700;}
.p2l-deck .dk-evt .ok{color:#6fd39b;} .p2l-deck .dk-evt .warn{color:#f0a860;}
.p2l-deck .dk-evt .bad{color:#ff9d9d;}
.p2l-deck .dk-badges{display:flex;gap:7px;flex-wrap:wrap;}
.p2l-deck .dk-badge{font-size:10px;font-weight:800;padding:3px 8px;border-radius:5px;
  white-space:nowrap;}
.p2l-deck .dk-badge b{font-size:11px;margin-left:3px;}
.p2l-deck .dk-badge.g{color:#9be3bd;background:rgba(48,163,108,.14);}
.p2l-deck .dk-badge.r{color:#ff9d9d;background:rgba(217,74,69,.14);}
.p2l-deck .dk-badge.b{color:#a9c6ff;background:rgba(91,140,255,.14);}
.p2l-deck .dk-badge.o{color:#f0c060;background:rgba(247,160,32,.14);}
.p2l-deck .dk-badge.y{color:#c9b6ff;background:rgba(124,58,237,.16);}
.p2l-deck .dk-badge.dim{color:#4b5b7c;background:rgba(255,255,255,.04);}
.p2l-deck .dk-rdy{background:#0e1728;border:1px solid #1a2540;border-radius:9px;
  padding:9px 11px;}
.p2l-deck .dk-rdyn{font-size:22px;font-weight:800;letter-spacing:-.5px;
  line-height:1.1;font-variant-numeric:tabular-nums;color:#d5deef;}
.p2l-deck .dk-rdyn small{font-size:11px;font-weight:600;color:#4b5b7c;margin-left:2px;}
.p2l-deck .dk-rdyl{font-size:11px;color:#8496b8;margin-top:2px;}
.p2l-deck .dk-rdybar{height:4px;background:#131e33;border-radius:3px;overflow:hidden;
  margin-top:7px;}
.p2l-deck .dk-rdybar i{display:block;height:100%;border-radius:3px;}
.p2l-deck .dk-rdys{font-size:10px;color:#7b8bac;margin-top:6px;}
.p2l-deck .dk-na{color:#3f4c69;}
.p2l-deck .dk-prog{margin-top:16px;}
.p2l-deck .dk-bar{height:6px;background:#131e33;border-radius:4px;
  overflow:hidden;display:flex;gap:2px;}
.p2l-deck .dk-bar i{flex:1;background:#1c2a45;border-radius:2px;}
.p2l-deck .dk-bar i.f{background:linear-gradient(90deg,#30a46c,#3ddc97);}
.p2l-deck .dk-bar i.cur{background:#5b8cff;}
.p2l-deck .dk-plab{display:flex;justify-content:space-between;font-size:10px;
  color:#7b8bac;margin-top:6px;}
@media (max-width:1100px){
  .p2l-deck .dk-body{grid-template-columns:1fr;}
  .p2l-deck .dk-colL,.p2l-deck .dk-colR{border:0;border-top:1px solid #16203a;}
  /* 窄窗口顶栏放不下「标语 + 材料清单」，与其裁成半句（「智能一致性审」），
     不如整块让位给品牌名与状态徽章 —— 信息在下面的材料区还有。 */
  .p2l-deck .dk-hd{display:none;}
}
</style>"""
