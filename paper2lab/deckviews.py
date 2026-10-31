"""指挥台详情面板的视图层 —— 「每块只说自己那一份，谁也别抄谁」。

为什么要单独一个模块
--------------------
指挥台有 **七块** 可点区域（材料 / 七位置 / 四站 / 工序 / 实时事件 / 判定计数 / 就绪度）。
原来它们的详情都往同一张 `findings` 表上倒，于是「点工序」和「点判定」看到的几乎一样 ——
那不叫交互，那叫换个标题再放一遍。

2026-10-27 第三版（用户定版：**只留核心信息**）
----------------------------------------------
这一版按用户点名的口径重排，每块只回答属于它自己的那个问题，且**最多三块**：

  七位置 → 只有 ① 运行方式（三步流程，每步带本轮真实产出数）
  工序   → ① 这一步做什么 ② 在本论文里的具体对应（参数 → 论文出处/原值 ↔ 代码出处/值）
           ③ 这一轮读到什么（含「为什么不同」「图里读出什么」）
  四站   → ① 本站完成的功能 ② 在论文/代码里的作用 ③ 输出结果
  事件   → ① 它在全流程第几步（序列条）② 这一步做了什么
  判定   → ① 具体对应还是仅名称对应 ② 逐条：两侧值 ＋ 是否具体对应
  材料   → 这份材料解析出了什么（全量清单）
  就绪度 → 分数怎么加出来的

表述口径（用户明确要求）：
  · 只摆事实，不写「已通过判定」「由此得出」这类自我背书的句子；
  · 数字一律现场派生（工序数取 len(TOOLS)、格数取 len(STEPS)），不写死；
  · 空 ≠ 0：没做的写「还没跑 / 本次没有做」并给理由，不拿 0 顶上；
  · 任何一条取不到都不许抛异常 —— 界面宁可少一行，也不能整页白掉。
"""
from __future__ import annotations

import html as _html
import re

from .agent_roles import POSITIONS
from .models import Category, Status
from .synonyms import PARAM_SYNONYMS
from .toolbox import STATION_OF_STEP, STATIONS, STEPS, TOOLS

# ---------------------------------------------------------------- 基础小件

def _e(v) -> str:
    return _html.escape("" if v is None else str(v))


def _d(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:          # noqa: BLE001
            pass
    return getattr(obj, "__dict__", {}) or {}


def _n(obj) -> int:
    try:
        return len(obj or ())
    except TypeError:
        return 0


def _na(audit, val, unit: str = "") -> str:
    """没跑的时候不写 0，写「—」。

    铁律：空 ≠ 0。0 会被读成「跑了，但什么也没产出」；
    而「还没跑」是另一件事，界面必须分得开。
    """
    if audit is None:
        return "—"
    return f"{val}{unit}"


def _g(obj, name, default=None):
    """安全取属性（对象或 dict 都行）。"""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


_ST_ZH = {
    Status.CONSISTENT: "一致",
    Status.INCONSISTENT: "两数不一致",
    Status.MISSING_IN_CODE: "论文有·代码无",
    Status.MISSING_IN_PAPER: "代码有·论文无",
    Status.NOT_FIXED: "未固定",
    Status.UNVERIFIABLE: "无法静态验证",
    Status.INTERNAL_INCONSISTENT: "内部互相矛盾",
}

_CAT_ZH = {
    Category.ENV: "环境一致性",
    Category.DATA: "数据集一致性",
    Category.PARAM: "参数一致性",
    Category.RECORD: "实验记录完整度",
    Category.RESULT: "结果一致性",
}

#: 判定色类的白话说一遍（详情面板里当图例用，不写裸 key）
_TONE_ZH = {"g": "一致", "r": "两数不一致", "b": "论文有·代码无",
            "o": "代码有·论文无", "y": "其他（未固定 / 无法验证 / 内部矛盾）"}

#: 站 → 色类（序列条、图例共用一处，避免三处各写一套）
_STG_TONE = {STATIONS[0]: "b", STATIONS[1]: "g", STATIONS[2]: "y", STATIONS[3]: "o"}


def st_zh(f) -> str:
    st = _g(f, "status")
    return _ST_ZH.get(st, str(getattr(st, "value", st) or "-"))


# ---------------------------------------------------------------- 图表（纯 CSS，无 JS）
VIEW_CSS = """<style>
/* ---- 横条图 ---- */
.dd-bars{margin:2px 0 4px;}
.dd-br{display:flex;align-items:center;gap:9px;margin:5px 0;font-size:12px;}
.dd-bl{width:168px;flex:none;color:#c3d1e8;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap;}
.dd-bt{flex:1;height:9px;background:#131e33;border-radius:5px;overflow:hidden;min-width:40px;}
.dd-bt i{display:block;height:100%;border-radius:5px;background:#5b8cff;min-width:2px;
  transition:width .3s ease;}
.dd-bt i.g{background:linear-gradient(90deg,#2f8f68,#3ddc97);}
.dd-bt i.r{background:linear-gradient(90deg,#b8453f,#ff8f8a);}
.dd-bt i.b{background:linear-gradient(90deg,#3a63c8,#8ab0ff);}
.dd-bt i.o{background:linear-gradient(90deg,#b87a1e,#f0c060);}
.dd-bt i.y{background:linear-gradient(90deg,#6d4bd0,#b49bff);}
.dd-bt i.dim{background:#2b3a55;}
.dd-bv{width:122px;flex:none;text-align:right;color:#a9b8d4;
  font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;}
.dd-bv small{color:#7b8bac;font-size:10px;margin-left:5px;}
/* ---- 堆叠条＋图例 ---- */
.dd-bk{display:flex;height:14px;border-radius:6px;overflow:hidden;
  background:#131e33;margin:4px 0 7px;gap:2px;}
.dd-bk i{display:block;height:100%;background:#5b8cff;}
.dd-bk i.g{background:#3ddc97;} .dd-bk i.r{background:#ff8f8a;}
.dd-bk i.b{background:#8ab0ff;} .dd-bk i.o{background:#f0c060;}
.dd-bk i.y{background:#b49bff;} .dd-bk i.dim{background:#2b3a55;}
.dd-lg{display:flex;gap:12px;flex-wrap:wrap;font-size:11px;color:#8b9bb8;}
.dd-lg s{text-decoration:none;display:inline-flex;align-items:center;gap:5px;}
.dd-lg s::before{content:"";width:8px;height:8px;border-radius:2px;
  background:#5b8cff;display:inline-block;}
.dd-lg s.g::before{background:#3ddc97;} .dd-lg s.r::before{background:#ff8f8a;}
.dd-lg s.b::before{background:#8ab0ff;} .dd-lg s.o::before{background:#f0c060;}
.dd-lg s.y::before{background:#b49bff;} .dd-lg s.dim::before{background:#2b3a55;}
/* ---- 流程链（横向串不起来，就竖着串，左导轨） ---- */
.dd-flow{margin:2px 0 6px;padding-left:2px;}
.dd-fs{display:flex;gap:10px;padding:0 0 10px 0;position:relative;}
.dd-fs:not(:last-child)::before{content:"";position:absolute;left:11px;top:24px;
  bottom:0;width:1px;background:#1e2b47;}
.dd-fn{width:23px;height:23px;flex:none;border-radius:50%;display:grid;
  place-items:center;font-size:11px;font-weight:800;color:#8ab0ff;
  background:rgba(91,140,255,.12);border:1px solid rgba(91,140,255,.35);
  position:relative;z-index:1;}
.dd-fb{min-width:0;flex:1;}
.dd-fb b{display:block;font-size:12px;color:#dbe6f8;font-weight:700;}
.dd-fb p{margin:1px 0 0;font-size:11px;color:#7f90b0;line-height:1.6;}
.dd-fv{flex:none;font-size:11px;color:#9be3bd;background:rgba(48,163,108,.10);
  border:1px solid rgba(48,163,108,.22);border-radius:6px;padding:1px 7px;
  align-self:flex-start;max-width:200px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;}
/* ---- 2×2 命中矩阵 ---- */
.dd-mx2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:4px 0 6px;}
.dd-cell{background:#0e1728;border:1px solid #1a2540;border-radius:9px;padding:8px 10px;}
.dd-cell b{display:block;font-size:19px;font-weight:800;color:#e8f0ff;
  font-variant-numeric:tabular-nums;line-height:1.15;}
.dd-cell span{font-size:11px;color:#6b7a9b;}
.dd-cell.g b{color:#9be3bd;} .dd-cell.r b{color:#ff9d9d;}
.dd-cell.b b{color:#a9c6ff;} .dd-cell.o b{color:#f0c060;}
/* ---- 全流程序列条：一眼看出「它在第几步、哪一站」 ---- */
.dd-strip{display:flex;gap:3px;flex-wrap:wrap;margin:3px 0 7px;}
.dd-strip s{text-decoration:none;width:23px;height:18px;border-radius:4px;
  display:grid;place-items:center;font-size:10px;font-weight:800;
  background:#16223a;color:#6b7a9b;font-variant-numeric:tabular-nums;}
.dd-strip s.g{background:rgba(61,220,151,.18);color:#9be3bd;}
.dd-strip s.b{background:rgba(138,176,255,.18);color:#a9c6ff;}
.dd-strip s.y{background:rgba(180,155,255,.18);color:#c9b6ff;}
.dd-strip s.o{background:rgba(240,192,96,.18);color:#f0c060;}
.dd-strip s.cur{outline:2px solid #5b8cff;outline-offset:1px;background:#2447a8;
  color:#fff;}
/* ---- 小注 ---- */
.dd-note{font-size:11px;color:#6b7a9b;line-height:1.65;margin:5px 0 2px;}
.dd-kv{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:#8b9bb8;margin:3px 0 6px;}
.dd-kv i{font-style:normal;color:#5f7091;}
.dd-kv b{color:#c3d1e8;font-weight:700;}
</style>"""


def bars(rows, tone: str = "b", unit: str = "", cap: int = 12) -> str:
    """横条图。rows = [(标签, 数值, 附注)]，宽度按最大值归一到 100%。

    ⚠ 数值**必须保住小数**：耗时多是 0.01–0.3 秒，早先这里 `int(v)` 一刀切，
      所有条都成了 0、宽度全是 0 —— 图上只剩空轨道，比不画还误导。
    """
    vals = []
    for lab, v, note in rows:
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        vals.append((str(lab), fv, str(note or "")))
    if not vals:
        return ""
    vals = vals[:cap]
    mx = max([abs(v) for _, v, _ in vals] + [1e-9])
    out = []
    for lab, v, note in vals:
        pct = max(0.0, min(100.0, abs(v) * 100.0 / mx))
        nt = f"<small>{_e(note)}</small>" if note else ""
        out.append(f"<div class='dd-br'><span class='dd-bl'>{_e(lab)}</span>"
                   f"<span class='dd-bt'><i class='{tone}' style='width:{pct:.1f}%'></i></span>"
                   f"<span class='dd-bv'>{v:g}{_e(unit)}{nt}</span></div>")
    return f"<div class='dd-bars'>{''.join(out)}</div>"


def stack(parts) -> str:
    """堆叠条 ＋ 图例。parts = [(标签, 数值, 色类)]，全 0 时返回空串。"""
    parts = [(str(lab), int(v), t) for lab, v, t in parts if v]
    if not parts:
        return ""
    total = sum(v for _, v, _ in parts) or 1
    seg = "".join(f"<i class='{t}' style='width:{v * 100.0 / total:.2f}%'></i>"
                  for _, v, t in parts)
    lg = "".join(f"<s class='{t}'>{_e(lab)} <b>{v}</b></s>" for lab, v, t in parts)
    return f"<div class='dd-bk'>{seg}</div><div class='dd-lg'>{lg}</div>"


def flow(steps) -> str:
    """流程链。steps = [(标题, 说明, 产出)] —— 编号由位置自动给，不手写。"""
    out = []
    for i, (title, desc, out_txt) in enumerate(steps, start=1):
        ov = f"<span class='dd-fv'>{_e(out_txt)}</span>" if out_txt else ""
        out.append(f"<div class='dd-fs'><span class='dd-fn'>{i}</span>"
                   f"<div class='dd-fb'><b>{_e(title)}</b>"
                   f"<p>{_e(desc)}</p></div>{ov}</div>")
    return f"<div class='dd-flow'>{''.join(out)}</div>"


def matrix(cells) -> str:
    """2×2 命中矩阵。cells = [(大标签, 数, 小字, 色类)]。"""
    body = "".join(f"<div class='dd-cell {t}'><b>{int(v)}</b>"
                   f"<span>{_e(lab)}{(' · ' + _e(note)) if note else ''}</span></div>"
                   for lab, v, note, t in cells)
    return f"<div class='dd-mx2'>{body}</div>"


def strip(items, cur: int = -1) -> str:
    """全流程序列条：每步一个方块，按站上色，当前这步高亮。

    items = [(编号, 站, 状态)]，编号从 1 开始；cur 是 0 基下标。
    这是「标注它在全流程中的位置」最省字的画法：不用读表就知道前后还有几步。
    """
    out = []
    for i, stg, stt in items:
        cls = _STG_TONE.get(stg, "")
        if stt == "skipped":
            cls = "dim"
        if i - 1 == cur:
            cls += " cur"
        out.append(f"<s class='{cls}'>{i:02d}</s>")
    lg = "".join(f"<s class='{t}'>{_e(s)}</s>" for s, t in _STG_TONE.items())
    return (f"<div class='dd-strip'>{''.join(out)}</div>"
            f"<div class='dd-lg'>{lg}</div>")


def kv(**pairs) -> str:
    return "<div class='dd-kv'>" + "".join(
        f"<span><i>{_e(k)}</i> <b>{_e(v)}</b></span>" for k, v in pairs.items()) + "</div>"


def tbl(headers, rows, empty: str = "本次没有数据", cap: int = 60,
        widths=None) -> str:
    """小表格。widths 给了就按百分比定列宽（不给就由浏览器自己分）。"""
    if not rows:
        return f"<div class='dd-empty'>{_e(empty)}</div>"
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>"
                   for r in rows[:cap])
    cg = ("<colgroup>" + "".join(f"<col style='width:{w}%'>" for w in widths)
          + "</colgroup>") if widths else ""
    more = (f"<div class='dd-more'>共 {len(rows)} 条，此处显示前 {cap} 条</div>"
            if len(rows) > cap else "")
    return (f"<table class='dd-tbl'>{cg}<thead><tr>{th}</tr></thead>"
            f"<tbody>{body}</tbody></table>{more}")


def _tools_of(station: str) -> list:
    return [t for t in TOOLS
            if STATION_OF_STEP.get((t.checkpoints or ("",))[0], "") == station]


def _steps_of(audit) -> list:
    return list((_g(audit, "execution_log", {}) or {}).get("steps") or [])


def _rec_of(audit, key: str) -> dict:
    return next((s for s in _steps_of(audit) if s.get("key") == key), {}) or {}


def _sec(v) -> str:
    try:
        return f"{float(v):.2f}s"
    except (TypeError, ValueError):
        return "—"


def _ratio(a, b):
    """两个归一化值的倍数关系（大/小）。取不到返回 None —— 绝不估。"""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a == 0 or b == 0:
        return None
    hi, lo = max(abs(a), abs(b)), min(abs(a), abs(b))
    return hi / lo if lo else None


def _yield(audit, key: str) -> str:
    from .deck import _yield_of
    return _yield_of(key, audit)


# ---------------------------------------------------------------- 论文对应（七位置共用）
_PAGE_RE = re.compile(r"(?:Page|第)\s*(\d+)")


def _page_of(loc) -> str:
    """从「Page 6 · Training Setup」里抠出页号；抠不到返回空串。"""
    m = _PAGE_RE.search(str(loc or ""))
    return f"第 {m.group(1)} 页" if m else ""


def _findings(audit) -> list:
    return list(_g(audit, "findings", None) or [])


def _paper_rows(audit, items, cap: int = 8) -> list:
    """[(参数, 页, 论文原值)] → 表格行。items 可以是 findings / (key, Evidence) 对。"""
    rows = []
    for it in items:
        if isinstance(it, tuple):
            key, ev = it
            rows.append([key, _page_of(_g(ev, "location", "")) or "—",
                         _g(ev, "raw_value", "") or "—"])
        else:
            ev = _g(it, "paper")
            rows.append([_g(it, "display_name", "") or _g(it, "param_key", ""),
                         _page_of(_g(ev, "location", "")) or "—",
                         _g(ev, "raw_value", "") or "—"])
    return rows[:cap]


def _page_bars(rows) -> str:
    """按论文页汇总条图：这一位置动到的条目落在哪几页。"""
    cnt: dict = {}
    for r in rows:
        if r[1] and r[1] != "—":
            cnt[r[1]] = cnt.get(r[1], 0) + 1
    if not cnt:
        return ""
    order = sorted(cnt, key=lambda x: int(re.sub(r"\D", "", x) or 0))
    return bars([(p, cnt[p], "") for p in order], "b", " 条", cap=12)


def _paper_block(audit, name: str, rows, empty: str, note: str = "") -> str:
    """「与论文的对应」：按页条图 ＋ 参数→页→原值。没有就如实说没有。"""
    if not rows:
        return f"<div class='dd-empty'>{_e(empty)}</div>"
    head = _page_bars(rows)
    body = tbl(["论文里的条目", "落在这页", "论文原值"], rows, empty)
    nt = f"<div class='dd-note'>{_e(note)}</div>" if note else ""
    return head + body + nt


# ---------------------------------------------------------------- 工序：一句话说明
#: 每件工序「做什么」的大白话。为什么写死在这里而不是从 Tool 里推：
#: Tool 的 needs/gives 是给调度员看的依赖，不是给人看的说明；这里补的正是人话那一层。
_TOOL_WHAT = {
    "parse_paper": "逐页读正文与表格，抓「X 设为 v」这类赋值句；数不合常理的不收。",
    "analyze_images": "按区域读图，把图上写着的超参数抄成结构化值，与正文层分开记。",
    "read_figures": "读图注与图里能看见的东西（趋势、峰位、坐标轴），回答「图画了什么」。",
    "chem_check": "对分子式 / CAS / SMILES / 分子量做算术对账，能自己算的就自己算。",
    "fill_missing": "带着「哪些参数还没值」回原文逐条找，有预算上限，找不到就停。",
    "analyze_code": "AST 精确解析每个赋值（file:line）；多语言走正则兜底，测试/示例单列。",
    "llm_analyze": "为每条已经定性的差异写一段人话解释（只解释，不改判定）。",
    "hypothesize": "把两侧对不上的地方变成「可疑项 ＋ 怎么证伪」的假设卡。",
    "design_experiment": "为每条假设选一件能算的工具，真算一遍，拿到实际值。",
    "align": "两侧的值归一化后逐条比：命中的写判定，缺一侧的单独记。",
    "selfcheck": "拿原始材料回查每条证据是否真的存在（审计的审计）。",
    "rootcause": "判断哪一侧的值最终生效：论文没照做，还是代码没实现。",
    "patch": "按根因写「改哪个文件、改成什么」的草案（只出草案，不落盘）。",
    "report": "把结论、证据、修正建议排成一份能给别人看的报告。",
}

#: 每一件工序对这批参数「做了什么」—— 一行一句，不与块标题重复。
#: 用户 2026-10-27 点名：「工序执行」每一项都要说清它与**这篇论文**的对应（参数等）。
_TOOL_ACT = {
    "parse_paper": "从正文抽成结构化值",
    "analyze_images": "从图上抄下来（与正文层分开记）",
    "read_figures": "在图注里读到",
    "chem_check": "按分子式 / 分子量算了一遍",
    "fill_missing": "带缺项回原文补到的值",
    "analyze_code": "在代码里找到赋值（file:line）",
    "llm_analyze": "写了归因文字（不改判定）",
    "hypothesize": "变成假设卡（可疑项＋怎么证伪）",
    "design_experiment": "配了计算工具、拿到实际值",
    "align": "归一化后逐条比对",
    "selfcheck": "回原文查了证据真假",
    "rootcause": "判了哪一侧的值最终生效",
    "patch": "写了修正草案（不落盘）",
    "report": "写进了报告",
}


def _param_disp(k: str) -> str:
    """参数显示名：同义词表里有中文别名就优先用中文（学习率 > Learning Rate）。

    为什么优先中文：用户在弹窗里要一眼认出「这是哪个参数」，
    英文键名是给程序看的；原文写法已经在「论文原值」那一列里了，不会丢。
    """
    alias = PARAM_SYNONYMS.get(k)
    if alias:
        for nm in (alias[1] or ()):
            if re.search(r"[\u4e00-\u9fff]", nm or ""):
                return nm
        return alias[0] or k
    return k


def _relate(audit, key: str):
    """这一件工序在这篇论文里**具体动到哪些参数**：参数 → 论文页＋原值 ↔ 代码位置＋值。

    为什么单列一块
    --------------
    用户点开「代码为何不同」，想看的是**这篇论文里那几个参数**（lr / batch_size…）
    在第几页写的什么、代码第几行写的什么；而不是一段关于「解析器怎么工作」的通用说明。
    所以这里只摆四条事实：哪个参数、论文哪页写的什么、代码哪行写的什么、这一步干了什么。
    页号与行号一律现场从证据出处里取，取不到写「—」，不猜、不补 0。
    """
    paper = (_g(_g(audit, "paper"), "params", {}) or {}) if audit is not None else {}
    code = (_g(_g(audit, "code"), "params", {}) or {}) if audit is not None else {}
    # 这几件工序下面已经有自己的专属表（图上参数 / 图注 / 化学式 / 两侧逐条 / 补漏轨迹），
    # 再摆一遍同一批数就是重复 —— 用户最反感「点哪儿都一样」，所以这里直接不出块。
    if key in ("analyze_images", "read_figures", "chem_check", "align", "fill_missing"):
        return ""
    fs = _findings(audit)
    by_key = {(_g(f, "param_key") or ""): f for f in fs}

    def _first(v):
        if isinstance(v, (list, tuple)):
            return v[0] if v else None
        return v

    def _keys():
        if key == "analyze_code":
            return list(code.keys())
        if key == "hypothesize":
            return [(_d(x).get("key") or "") for x in (_g(audit, "hypotheses", None) or [])]
        if key == "design_experiment":
            return [(_d(x).get("key") or "") for x in (_g(audit, "experiments", None) or [])]
        if key in ("align", "rootcause", "patch", "report", "selfcheck", "llm_analyze"):
            return [(_g(f, "param_key") or "") for f in fs]
        return list(paper.keys()) or list(code.keys())

    keys = [k for k in dict.fromkeys(_keys()) if k]
    act = _TOOL_ACT.get(key, "经手过")
    rows = []
    for k in keys[:8]:
        f = by_key.get(k)
        pe = paper.get(k) or _g(f, "paper")
        ce = _first(code.get(k)) or _g(f, "code")
        # 论文侧只认「第 N 页」（_page_of 从 "Page 3" 这类出处里取）；取不到就退回原出处，
        # 代码侧保留 "train.py:23" 原样 —— 两侧的定位方式本来就不一样，不强行统一。
        ploc = _g(pe, "location", "") or ""
        rows.append([_param_disp(k),
                     (_page_of(ploc) or ploc or "—") if pe is not None else "—",
                     _g(pe, "raw_value", "") or "—",
                     _g(ce, "location", "") or "—",
                     _g(ce, "raw_value", "") or "—"])
    n_p = sum(1 for k in keys[:8] if paper.get(k) or _g(by_key.get(k), "paper"))
    n_c = sum(1 for k in keys[:8] if code.get(k) or _g(by_key.get(k), "code"))
    head = (f"<div class='dd-sub'>这一步在这篇论文里动到 <b>{len(keys)}</b> 个参数"
            f"（论文侧 {n_p} 个、代码侧 {n_c} 个），对它做的事是「{_e(act)}」。"
            "下面每一行都是<b>这篇论文里的具体一条</b>：页号与行号现场取自证据出处，"
            "取不到就写「—」，不猜。</div>")
    body = tbl(["参数", "论文出处", "论文原值", "代码出处", "代码原值"], rows,
               "本次还没有可对照的条目", cap=8, widths=(18, 18, 22, 20, 22))
    return head + body


def _diff_why(f) -> str:
    """这条为什么不一样 —— 只用两侧已有的值与出处回答，不猜语义。

    用户在指令里点名要「代码为何不同」：所以每一条差异都要落到
    「差几倍 / 哪一侧没有 / 为什么没有」这一层，而不是把判定词再抄一遍。
    """
    st = _g(f, "status")
    p, c = _g(f, "paper"), _g(f, "code")
    if st == Status.INCONSISTENT:
        r = _ratio(_g(p, "normalized"), _g(c, "normalized"))
        base = f"两数差 {r:.2f} 倍" if r else "两数不等"
        note = (_g(f, "note") or "")[:40]
        return base + ("；" + note if note else "（写法 / 单位 / 量级不同，或确实改过）")
    if st == Status.MISSING_IN_CODE:
        return "代码里按同名变量没找到赋值（可能写死在别处，或确实没实现）"
    if st == Status.MISSING_IN_PAPER:
        return "论文正文没有声明这个参数（复现时不知道该照哪个）"
    if st == Status.INTERNAL_INCONSISTENT:
        return "同一侧多处声明互相矛盾（第二处及以后见附加证据）"
    if st == Status.NOT_FIXED:
        return "值没固定（如 seed 没设）"
    if st == Status.UNVERIFIABLE:
        return "值是运行时算出来的，静态读不到"
    return "两侧归一化后相同（写法不同不算不同）"


# ---------------------------------------------------------------- 工序视图
def tool_view(audit, key: str):
    """工序：这一步**做什么、读到了什么、为什么是这样**。最多三块。"""
    t = next((x for x in TOOLS if x.key == key), None)
    title = t.title if t else key
    rec = _rec_of(audit, key)
    stt = rec.get("status") or ("未执行" if audit is None else "未记录")
    sub = ""
    if t is not None:
        sub = (f"读什么：{_e('、'.join(t.needs) or '无需前置')}　→　交出："
               f"{_e('、'.join(t.gives) or '无')}　·　{_e(t.cost)}")
    metrics = [("状态", {"done": "已完成", "skipped": "计划跳过",
                        "failed": "执行失败"}.get(stt, stt),
                "g" if stt == "done" else ("r" if stt == "failed" else "dim")),
               ("耗时", _sec(rec.get("seconds")), ""),
               ("产物", _yield(audit, key) or "无", "b")]
    blocks = [("这一步做什么",
               f"<div class='dd-sub'>{_e(_TOOL_WHAT.get(key, '本版还没写一句说明。'))}</div>"
               + _chain(key))]
    _rel = _relate(audit, key)
    if _rel:
        blocks.append((f"这一步在本论文里的具体对应（{title}）", _rel))
    if rec.get("reason"):
        blocks.append(("为什么这次是这样", f"<div class='dd-sub'>{_e(rec['reason'])}</div>"))
    blocks.extend(_tool_content(audit, key))
    return title, sub, metrics, blocks


def _chain(key: str) -> str:
    """它在链上的前后：一行为什么要有这一步。"""
    idx = next((i for i, x in enumerate(TOOLS) if x.key == key), -1)
    if idx < 0:
        return ""
    up = TOOLS[idx - 1].title if idx > 0 else "（链头：材料）"
    dn = TOOLS[idx + 1].title if idx + 1 < len(TOOLS) else "（链尾：报告）"
    return kv(上一件=up, 下一件=dn)


def _ev_cards(pairs) -> str:
    """证据卡：原文片段 ＋ 出处，左右并排。这是「可追溯」最直白的形态。"""
    out = []
    for label, ev in pairs:
        if ev is None:
            out.append(f"<div class='dd-card'><h5>{_e(label)}</h5>"
                       f"<div class='dd-empty' style='padding:0'>这一侧没有证据</div></div>")
            continue
        out.append(f"<div class='dd-card'><h5>{_e(label)}</h5>"
                   f"<div class='dd-loc'>{_e(_g(ev, 'location', '') or '')}</div>"
                   f"<div class='dd-snip'>{_e((_g(ev, 'snippet', '') or '')[:200])}</div></div>")
    return f"<div class='dd-split'>{''.join(out)}</div>"


def _tool_content(audit, key: str) -> list:
    """每件工序自己的「读到了什么」。刻意各不相同，不做通用兜底表。"""
    p = _g(audit, "paper")
    c = _g(audit, "code")

    if key == "parse_paper":
        params = list((_g(p, "params", {}) or {}).items())[:3]
        results = list((_g(p, "results", {}) or {}).items())[:2]
        cards = [("论文原文 · 参数", ev) for _, ev in params]
        cards += [("论文原文 · 实验结果", ev) for _, ev in results]
        out = [("写进比对清单的原文（片段原样搬运，不改一个字）",
                _ev_cards(cards) if cards
                else "<div class='dd-empty'>本次没有从论文里抽到参数声明</div>")]
        thin = _g(p, "page_coverage", None) or []
        bad = [f"第 {x.get('page')} 页" for x in thin if (x.get("status") or "") == "thin"]
        if bad:
            out.append(("几乎没读到字的页（可能是扫描版）",
                        f"<div class='dd-tip'>{_e('、'.join(bad[:8]))}"
                        f"{' 等 ' + str(len(bad)) + ' 页' if len(bad) > 8 else ''}"
                        f" —— 这些页里的声明可能漏掉了。</div>"))
        return out

    if key == "analyze_images":
        meta = _g(p, "image_meta", {}) or {}
        rows = [[k, str(v)[:60]] for k, v in meta.items()
                if not isinstance(v, (list, dict))]
        img = _g(p, "image_params", {}) or {}
        return [("读图通道这次开了没有", tbl(["项", "值"], rows, "本次没有读图记录", cap=8)),
                ("图上抄下来的参数（单列，不参与正文层比对）",
                 tbl(["参数", "值", "位置"],
                     [[k, _g(v, "raw_value", ""), _g(v, "location", "")]
                      for k, v in img.items()], "这次没有从图上读到参数", cap=10))]

    if key == "read_figures":
        rows = []
        for it in (_g(audit, "figure_clues", None) or []):
            d = _d(it)
            rows.append([d.get("page") or d.get("location") or "",
                         d.get("caption") or d.get("title") or d.get("key") or "",
                         (d.get("visible") or d.get("note") or d.get("text") or "")[:80]])
        st_ = _g(audit, "figure_stats", {}) or {}
        return [("图里解析出了什么（图注原文 ＋ 图上可见信息）",
                 tbl(["页", "图注 / 标题", "读到什么"], rows, "本次没读到图表线索", cap=10)),
                ("这一项的统计", kv(**{str(k): str(v)[:30] for k, v in st_.items()})
                 if st_ else "<div class='dd-empty'>本次没有统计记录</div>")]

    if key == "chem_check":
        rows = []
        for it in (_g(audit, "chem_items", None) or []):
            d = _d(it)
            rows.append([d.get("kind", ""), d.get("value", ""), d.get("page", ""),
                         d.get("computed", ""), d.get("status", "")])
        fb = ("<div class='dd-tip'>这一批是「模型跳过该步、程序兜底补算」出来的 —— "
              "跳过可以，静默不行，所以如实标明。</div>"
              if _g(audit, "chem_fallback", False) else "")
        return [("化学身份能不能算得回来",
                 tbl(["类型", "原文值", "页", "算出来的数", "判定"], rows,
                     "本次没抽到化学标记（非化学论文属正常）", cap=12) + fb)]

    if key == "fill_missing":
        trace = _g(p, "agent_trace", None) or []
        meta = _g(p, "agent_meta", {}) or {}
        cover = (_g(audit, "coverage", {}) or {}).get("agent") or {}
        out = [("补漏轨迹（回原文逐条核对，最多几步就停）",
                tbl(["步", "做了什么"], [[i, str(_d(x))[:100]]
                                       for i, x in enumerate(trace, start=1)],
                    "本次没有补漏轨迹", cap=8))]
        out.append(("它为什么停在这里",
                    kv(原因=cover.get("stopped_reason") or "—",
                       预算步数=(cover.get("budget") or {}).get("steps", 0),
                       模型可用="是" if cover.get("llm_available") else "否")))
        rows = [[k, str(v)[:40]] for k, v in (meta or {}).items()
                if not isinstance(v, (list, dict))]
        out.append(("预算消耗", tbl(["项", "值"], rows, "本次没有预算记录", cap=6)))
        return out

    if key == "analyze_code":
        items = list((_g(c, "params", {}) or {}).items())[:3]
        cards = []
        for _, evs in items:
            ev = (evs or [None])[0]
            if ev is not None:
                cards.append(("代码原文 · 赋值", ev))
        # 「代码为何不同」：同一个参数在多处被赋成不同的值，就是代码侧自己不一致
        dup = []
        for k, evs in (_g(c, "params", {}) or {}).items():
            vals = {(_g(e, "raw_value", "") or "") for e in (evs or [])}
            if len(evs or []) > 1 and len(vals) > 1:
                dup.append([k, "、".join(sorted(v for v in vals if v)[:3]),
                            "、".join(_g(e, "location", "") for e in evs[:3]),
                            f"{len(evs)} 处"])
        out = [("证据卡（AST 精确解析到的赋值，带 file:line）",
                _ev_cards(cards) if cards
                else "<div class='dd-empty'>本次没扫到参数赋值</div>")]
        out.append(("代码里同一个参数出现在多处（写死的值 / 运行时变量）",
                    tbl(["参数", "各处的值", "位置", "处数"], dup,
                        "没有这种情况：每个参数在代码里只出现一次", cap=8)))
        fails = [[(_g(f, "file", "") or ""), (_g(f, "reason", "") or "")[:60]]
                 for f in (_g(c, "failed_files", None) or [])]
        out.append(("解析失败的文件（失败示众，不静默跳过）",
                    tbl(["文件", "原因"], fails, "全部文件解析成功", cap=8)))
        rej = [[_g(_d(x), "key", ""), _g(_d(x), "value", ""),
                _g(_d(x), "location", ""), (_g(_d(x), "reason", "") or "")[:50]]
               for x in (_g(c, "rejected", None) or [])]
        out.append(("被合理性校验拒收的抽取（为什么不要它）",
                    tbl(["参数", "值", "位置", "为什么拒"], rej, "没有拒收记录", cap=8)))
        return out

    if key == "llm_analyze":
        rows = [[_g(f, "display_name", "") or _g(f, "param_key", ""),
                 (_g(f, "ai_analysis", "") or "")[:100]]
                for f in _findings(audit) if _g(f, "ai_analysis", "")]
        rd = _g(audit, "readiness", {}) or {}
        rows2 = [[k, str(v)[:36]] for k, v in rd.items() if k.startswith("llm_")]
        return [("模型写下的归因文字（只解释，不改判定）",
                 tbl(["参数", "它怎么解释的"], rows, "本次没有模型归因文字", cap=8)),
                ("模型通道状态", tbl(["项", "值"], rows2, "没有模型通道记录", cap=8))]

    if key == "hypothesize":
        rows = [[_d(x).get("rank", ""), _d(x).get("key", ""),
                 (_d(x).get("claim") or "")[:46], _d(x).get("verdict", ""),
                 (_d(x).get("falsifiable_by") or "")[:40]]
                for x in (_g(audit, "hypotheses", None) or [])]
        st_ = _g(audit, "hypothesis_stats", {}) or {}
        out = [("假设卡（每条都写着「看到什么就推翻它」）",
                tbl(["次序", "参数", "它主张什么", "推演结论", "怎么推翻"],
                    rows, "本次没有提出假设（两侧没有矛盾时这是正确行为）", cap=10))]
        if st_:
            out.append(("推演用了什么", kv(**{str(k): str(v)[:24] for k, v in st_.items()})))
        return out

    if key == "design_experiment":
        rows = [[_d(x).get("rank", ""), _d(x).get("key", ""),
                 (_d(x).get("claim") or "")[:36], _d(x).get("status", ""),
                 _d(x).get("expected", ""), _d(x).get("actual", "")]
                for x in (_g(audit, "experiments", None) or [])]
        st_ = _g(audit, "experiment_stats", {}) or {}
        out = [("实验卡（要验什么 ＋ 论文值 ＋ 算出来）",
                tbl(["次序", "参数", "要验什么", "状态", "论文值", "算出来"],
                    rows, "本次没有设计实验", cap=10))]
        if st_:
            out.append(("实跑统计", kv(**{str(k): str(v)[:24] for k, v in st_.items()})))
        return out

    if key == "align":
        # 用户点名「代码为何不同」：所以值后面直接缀出处（论文第几页 / 代码第几行），
        # 一行就能回答「论文写的什么、代码写的什么、差在哪」。
        def _at(ev) -> str:
            v = _g(ev, "raw_value", "") or "—"
            loc = _g(ev, "location", "") or ""
            return f"{v}　@{loc}" if loc else v
        rows = [[_g(f, "display_name", "") or _g(f, "param_key", ""), st_zh(f),
                 _at(_g(f, "paper")), _at(_g(f, "code")), _diff_why(f)]
                for f in _findings(audit)]
        out = [("两侧逐条对照：这篇论文里的每一条差在哪、为什么差（出处精确到页 / 行）",
                tbl(["参数", "判定", "论文值（出处）", "代码值（出处）", "为什么不同"], rows,
                    "本次没有可比对的条目", cap=12, widths=(16, 12, 24, 24, 24)))]
        log = _g(audit, "compare_log", None) or []
        if log:
            out.append((f"程序比较日志（{len(log)} 条，审计的审计）",
                        f"<div class='dd-code'>{_e(chr(10).join(str(x) for x in log[:8]))}"
                        f"</div>"))
        return out

    if key == "selfcheck":
        sc = _g(audit, "selfcheck", {}) or {}
        out = [("拿原始材料回查每条证据是否真的存在",
                tbl(["项", "值"], [["抽检条数", sc.get("checked", 0)],
                                  ["查证到原文", sc.get("passed", 0)],
                                  ["查不到原文", sc.get("failed", 0)]],
                    "本次没有做自证", cap=4))]
        fails = sc.get("failures") or []
        if fails:
            out.append(("查不到原文的条目（有问题就说有问题）",
                        tbl(["条目"], [[str(x)[:100]] for x in fails], "", cap=8)))
        return out

    if key == "rootcause":
        rows = [[_d(x).get("display") or _d(x).get("param_key", ""),
                 _d(x).get("paper_value", ""), _d(x).get("code_value", ""),
                 _d(x).get("verdict", "") or "无法确定",
                 (_d(x).get("reason", "") or "")[:50]]
                for x in (_g(audit, "rootcause", None) or [])]
        return [("根因追查（哪一侧的值最终生效）",
                 tbl(["参数", "论文值", "代码值", "生效值", "为什么"], rows,
                     "本次没有追查根因", cap=10))]

    if key == "patch":
        rows = [[_d(x).get("display") or _d(x).get("param_key", ""),
                 _d(x).get("status", ""), _d(x).get("target", ""),
                 (_d(x).get("diff") or _d(x).get("rationale") or "")[:56]]
                for x in (_g(audit, "patches", None) or [])]
        return [("修正草案（只出草案，绝不落盘）",
                 tbl(["参数", "状态", "目标", "草案"], rows,
                     "本次没有生成草案", cap=10))]

    if key == "report":
        rd = _g(audit, "readiness", {}) or {}
        dom = rd.get("dominant_issues") or []
        sc = _g(audit, "selfcheck", {}) or {}
        return [("报告里影响最大的几条",
                 tbl(["序号", "项"], [[i, str(x)[:80]] for i, x in enumerate(dom, start=1)],
                     "本次没有主要结论", cap=8)),
                ("它写报告时用了哪些数",
                 kv(总分=rd.get("total", "—"),
                    判定条数=_n(_findings(audit)),
                    查证到原文=f"{sc.get('passed', 0)} / {sc.get('checked', 0)}"))]

    return [("这一件本版没有专门的明细视图",
             "<div class='dd-empty'>它是新加的工具，视图还没登记 —— "
             "不是没跑，是这里暂时没话说。</div>")]


# ---------------------------------------------------------------- 材料视图
def mat_view(audit, kind: str):
    p = _g(audit, "paper")
    c = _g(audit, "code")
    if kind == "paper":
        title = "论文这份资料解析出了什么"
        sub = _e(_g(p, "title", "") or "（未识别标题）")
        metrics = [("页数", str(_g(p, "page_count", 0) or 0), "b"),
                   ("参数声明", str(_n(_g(p, "params"))), "g"),
                   ("结果声明", str(_n(_g(p, "results"))), ""),
                   ("疑似声明", str(_g(p, "decl_total", 0) or 0), "dim")]
        cov = _g(p, "page_coverage", None) or []
        oc = _g(audit, "coverage", {}) or {}
        thin = oc.get("pages_thin", 0) or 0
        blocks = [
            ("每页读到了多少字（横条越短越可能是扫描版）",
             bars([(f"第 {x.get('page')} 页", x.get("chars") or 0,
                    "thin" if (x.get("status") or "") == "thin" else "")
                   for x in cov], "b", " 字", cap=40)
             + (f"<div class='dd-tip'>{thin} 页几乎没读到字，"
                f"这些页里的声明可能漏掉。</div>" if thin else "")),
            ("论文声明的参数（含页码出处）",
             tbl(["参数", "论文值", "位置", "原文"],
                 [[k, _g(ev, "raw_value", ""), _g(ev, "location", ""),
                   (_g(ev, "snippet", "") or "")[:80]]
                  for k, ev in (_g(p, "params", {}) or {}).items()],
                 "没有抽到参数声明", cap=12)),
            ("论文声明的实验结果",
             tbl(["指标", "值", "位置"],
                 [[k, _g(ev, "raw_value", ""), _g(ev, "location", "")]
                  for k, ev in (_g(p, "results", {}) or {}).items()],
                 "没有抽到结果声明", cap=10)),
        ]
        return title, sub, metrics, blocks

    if kind == "code":
        title = "代码这份资料扫描出了什么"
        sub = "只看静态结构，不运行任何代码。"
        metrics = [("文件", str(_g(c, "python_files", 0) or 0), "b"),
                   ("参数", str(_n(_g(c, "params"))), "g"),
                   ("依赖", str(_n(_g(c, "requirements"))), ""),
                   ("种子固定", "是" if _g(c, "seed_fixed", False) else "否",
                    "g" if _g(c, "seed_fixed", False) else "o")]
        rows = []
        for k, evs in (_g(c, "params", {}) or {}).items():
            rows.append([k, "、".join(_g(e, "raw_value", "") for e in evs[:3]),
                         "、".join(_g(e, "location", "") for e in evs[:3]),
                         f"{len(evs)} 处"])
        blocks = [
            ("每个参数写在哪些文件里（多处出现都保留）",
             tbl(["参数", "代码值", "位置", "出现次数"], rows,
                 "没有扫到参数赋值", cap=12)),
            ("依赖环境", tbl(["包", "声明版本"],
                             [[k, v or "未固定版本"]
                              for k, v in (_g(c, "requirements", {}) or {}).items()],
                             "没有依赖声明", cap=10)),
        ]
        return title, sub, metrics, blocks

    if kind == "result":
        ur = _g(audit, "user_results", {}) or {}
        return ("你回传的实测结果",
                "原样展示，不改一个字 —— 它是我们自己复现出来的数，不是论文的数。",
                [("条目", str(_n(ur)), "b")],
                [("原文", f"<div class='dd-code'>{_e(str(ur)[:2400])}</div>")])

    return ("还没有载入材料",
            "先用左上角的上传入口放进论文 PDF（代码包与结果文件可选）。",
            [], [("说明", "<div class='dd-empty'>没有材料就没有可摊开的内容。</div>")])


# ---------------------------------------------------------------- 位置视图（七个）
def pos_view(audit, key: str):
    """位置：这个岗位**怎么跑**。只有一块 —— 用户 2026-10-27 定版。

    为什么砍掉了「与论文哪一块对应」：七个岗位都要讲一遍论文页码，弹窗又长又像
    同一个意思换七种说法，真正该看的「它是怎么跑起来的」反而被挤到下面。
    现在每个位置只留一条三步流程（每步带本轮真实产出数），一屏看完、互不重复。
    """
    p = next((x for x in POSITIONS if x.key == key), None)
    if p is None:
        return "位置", "", [], []
    title = f"{p.cn} · 它怎么跑"
    sub = (f"能力：{_e(p.capability)}　·　归属：{_e(p.station)}"
           f"　·　交接物：{_e(p.handoff)}")
    builder = {"parser": _pos_parser, "retriever": _pos_retriever,
               "moderator": _pos_moderator, "experimenter": _pos_experimenter,
               "calculator": _pos_calculator, "reviewer": _pos_reviewer,
               "writer": _pos_writer}.get(key)
    if builder is None:
        return title, sub, [], [("说明", "<div class='dd-empty'>这个位置本版没有明细视图。</div>")]
    metrics, blocks = builder(audit, p.cn)
    # 只留运行方式那一块（其余块留给 builder 复用，但不在弹窗里出现）。
    run_blocks = [x for x in blocks if "运行方式" in x[0]]
    return title, sub, metrics, (run_blocks or blocks[:1])[:1]


def _pos_parser(audit, cn: str):
    p = _g(audit, "paper")
    c = _g(audit, "code")
    cov = _g(audit, "coverage", {}) or {}
    img = cov.get("image") or {}
    rej_p = _g(p, "rejected", None) or []
    rej_c = _g(c, "rejected", None) or []
    metrics = [("页", _na(audit, _g(p, "page_count", 0) or 0), "b"),
               ("参数", _na(audit, _n(_g(p, "params"))), "g"),
               ("代码文件", _na(audit, _g(c, "python_files", 0) or 0), ""),
               ("拒收", _na(audit, len(rej_p) + len(rej_c)),
                "o" if (rej_p or rej_c) else "dim")]
    run = [
        ("逐页读正文与表格", "读不到字的页会标成 thin，不假装读到。",
         _na(audit, _g(p, "page_count", 0) or 0, " 页")),
        ("抓赋值句、过校验", "数值区间 / 单位 / 类型不合常理的不进比对清单，但会登记。",
         _na(audit, f"收下 {_n(_g(p, 'params'))} 个")),
        ("两只眼睛分开读", "论文视野只拿 paper、代码视野只拿 code，互相看不见。",
         _na(audit, f"代码 {_g(c, 'python_files', 0) or 0} 文件")),
    ]
    rows = _paper_rows(audit, list((_g(p, "params", {}) or {}).items()))
    blocks = [(f"{cn}的运行方式（三步，每步都留产物）", flow(run)),
              (f"{cn}动的是论文的哪一块（参数声明落在哪几页）",
               _paper_block(audit, cn, rows,
                            "还没开始跑 —— 跑完这里会逐条列出「参数 → 页 → 原值」",
                            "只收正文与表格里的声明；图上的值算在「图像解析」那一件里。")),
              (f"{cn}这一轮的交出物",
               kv(参数=_na(audit, _n(_g(p, "params"))),
                  结果=_na(audit, _n(_g(p, "results"))),
                  拒收=_na(audit, len(rej_p) + len(rej_c)),
                  读图通道="开" if img.get("enabled") else "关"))]
    return metrics, blocks


def _pos_retriever(audit, cn: str):
    p = _g(audit, "paper")
    c = _g(audit, "code")
    pk = set((_g(p, "params", {}) or {}).keys())
    ck = set((_g(c, "params", {}) or {}).keys())
    both, only_p, only_c = pk & ck, pk - ck, ck - pk
    cov = _g(audit, "coverage", {}) or {}
    mx = cov.get("param_matrix") or []
    metrics = [("论文命中", _na(audit, len(pk)), "b"),
               ("代码命中", _na(audit, len(ck)), "b"),
               ("两侧配平", _na(audit, len(both)), "g" if both else "dim"),
               ("只一侧", _na(audit, len(only_p) + len(only_c)),
                "r" if (only_p or only_c) else "dim")]
    run = [
        ("展开关键词", "lr / learning_rate / 学习率 算同一个，先用同义词表摊成一组。",
         _na(audit, f"{len(both | only_p)} 组")),
        ("两侧各找一遍", "论文按赋值句找、代码按 AST 赋值找，命中位置各自记下。",
         _na(audit, f"{len(pk)} / {len(ck)}")),
        ("配对，缺的也留", "两侧都有才进比对；只有一侧的照样登记，不悄悄丢掉。",
         _na(audit, f"配上 {len(both)} 条")),
    ]
    kw_rows = []
    for k in sorted(pk | ck)[:6]:
        alias = PARAM_SYNONYMS.get(k)
        if alias:
            disp, names = alias[0], alias[1]
            kw_rows.append([disp, len(names), "、".join(names[:4])
                            + ("…" if len(names) > 4 else "")])
        else:
            kw_rows.append([k, 1, k])
    pair_rows = []
    for row in mx[:8]:
        d = _d(row)
        pair_rows.append([d.get("参数", ""),
                          str(d.get("论文侧") or "").split("@")[-1].strip(),
                          str(d.get("代码侧") or "").split("@")[-1].strip(),
                          d.get("是否进入比对", "")])
    notyet = "还没开始跑 —— 跑完这里会逐条列出来"
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}在论文与代码两侧各找到多少",
               (matrix([("两侧都找到 · 进比对", len(both), "已配对", "g"),
                        ("只在论文里 · 代码没写", len(only_p), "记缺代码",
                         "r" if only_p else "dim"),
                        ("只在代码里 · 论文没写", len(only_c), "记缺论文",
                         "o" if only_c else "dim"),
                        ("正文提过 · 两侧都没取到", _n(cov.get("kinds_unmatched")),
                         "未进比对", "dim")])
                if audit is not None else
                "<div class='dd-empty'>还没开始跑 —— 这里会显示"
                "「两侧都找到 / 只一侧 / 都没取到」各几条，现在一条都没有，"
                "不填 0 假装检索过。</div>")
               + tbl(["参数", "论文出处", "代码出处", "是否进比对"], pair_rows,
                     notyet if audit is None else "本次没有配对记录", cap=8)),
              (f"{cn}用的关键词（同义词表写死在代码里，不靠模型猜）",
               tbl(["参数", "别名数", "实际用到的关键词"], kw_rows,
                   notyet if audit is None else "本次没有可检索的参数", cap=6,
                   widths=(26, 12, 62))
               + (f"<div class='dd-note'>只在一侧找到的："
                  + _e("、".join([f"{x}（论文）" for x in sorted(only_p)]
                                + [f"{x}（代码）" for x in sorted(only_c)]))
                  + " —— 这些会被记成「缺」，不会被悄悄丢掉。</div>"
                  if (only_p or only_c) else ""))]
    return metrics, blocks


def _pos_moderator(audit, cn: str):
    ex = _g(audit, "execution_log", {}) or {}
    plan = ex.get("plan") or {}
    dec = plan.get("decisions") or []
    ran = [d for d in dec if (d.get("action") or "run") != "skip"]
    skip = [d for d in dec if (d.get("action") or "") == "skip"]
    who = "模型排计划" if plan.get("planner") == "llm" else "规则兜底排计划"
    p = _g(audit, "paper")
    c = _g(audit, "code")
    metrics = [("菜单", str(len(TOOLS)), "b"),
               ("排进计划", _na(audit, len(ran)), "g"),
               ("跳过", _na(audit, len(skip)), "o" if skip else "dim"),
               ("谁定的", "—" if audit is None else
                ("模型" if plan.get("planner") == "llm" else "规则"), "")]
    run = [
        ("接材料、看齐不齐", "只有论文？有没有代码包？有没有回传实测值？",
         _na(audit, "论文+代码" if _g(audit, "code") else "论文")),
        ("对着菜单排顺序", f"菜单上一共 {len(TOOLS)} 件，每件写着要什么前置、交出什么。",
         f"{len(TOOLS)} 件"),
        ("写理由、交计划", "每一件为什么执行、为什么跳过，都留一句话。",
         f"{len(ran)} 执行 / {len(skip)} 跳过"),
    ]
    mat_rows = [["论文页", _na(audit, _g(p, "page_count", 0) or 0, " 页")],
                ["论文参数", _na(audit, _n(_g(p, "params")))],
                ["代码参数", _na(audit, _n(_g(c, "params")))],
                ["回传实测", _na(audit, _n(_g(audit, "user_results")))]]
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}看的是材料清单，不是论文正文",
               bars([("论文声明的参数", _n(_g(p, "params")), ""),
                     ("代码里的赋值", _n(_g(c, "params")), ""),
                     ("回传的实测指标", _n(_g(audit, "user_results")), "")],
                    "b", " 条", cap=4)
               + tbl(["它看到的东西", "有多少"], mat_rows,
                     "还没开始跑 —— 跑完这里会显示材料清单", cap=6)
               + "<div class='dd-note'>它不读正文，只按「哪些前置齐了」决定跑哪几件；"
                 "所以换了材料，跑的东西会跟着变。</div>"),
              (f"{cn}排出来的计划（{who}）",
               tbl(["工序", "动作", "为什么", "谁定的"],
                   [[d.get("title") or d.get("tool", ""),
                     "跳过" if d.get("action") == "skip" else "执行",
                     (d.get("reason") or "")[:48], d.get("source") or ""]
                    for d in dec], "这一轮没有计划记录", cap=14))]
    return metrics, blocks


def _pos_experimenter(audit, cn: str):
    exp = _g(audit, "experiments", None) or []
    st_ = _g(audit, "experiment_stats", {}) or {}
    metrics = [("设计", _na(audit, _g(audit, "experiment_total", 0) or 0), "b"),
               ("实跑", _na(audit, st_.get("measured", 0) or 0), "g"),
               ("有数", _na(audit, (st_.get("agree", 0) or 0) + (st_.get("disagree", 0) or 0)), ""),
               ("工具调用", _na(audit, st_.get("calls", 0) or 0), "")]
    run = [
        ("拿假设卡", "从①站拿到「可疑项 ＋ 怎么证伪」，不自己另找问题。",
         f"{_n(_g(audit, 'hypotheses'))} 张"),
        ("挑一件能算的工具", "关系式复算 / 峰位定位 / 单位换算 / 量纲核对。",
         "、".join(st_.get("tools") or []) or "无"),
        ("真算一遍、取实际值", "算出来的数与论文值摆一起，差多少如实记，不修数。",
         f"{st_.get('measured', 0) or 0} 条有数"),
    ]
    # 它验的是论文里的哪几条 —— 假设卡的 key 对回论文侧的那一条，页号照样现场取
    hk = {(_d(x).get("key") or "") for x in (_g(audit, "hypotheses", None) or [])}
    rows_paper = _paper_rows(audit, [f for f in _findings(audit)
                                     if (_g(f, "param_key") or "") in hk])
    st_count: dict = {}
    for x in exp:
        k = str(_d(x).get("status") or "未标注")
        st_count[k] = st_count.get(k, 0) + 1
    card_rows = [[_d(x).get("rank", ""), _d(x).get("key", ""),
                  _d(x).get("status", ""),
                  ("" if _d(x).get("actual") in (None, "") else str(_d(x).get("actual"))),
                  re.sub(r"\*\*", "", _d(x).get("status_desc") or "")[:44]]
                 for x in exp[:10]]
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}验的是论文里的哪几条",
               _paper_block(audit, cn, rows_paper,
                            "本次没有可对应的论文条目（两侧没有矛盾时这是正确行为）",
                            "页号取自论文侧证据的出处；同一参数只算一次。")),
              (f"{cn}的实验卡与实跑记录",
               (stack([(k, v, "g" if "实测" in k else ("o" if "算不出" in k else "y"))
                       for k, v in st_count.items()])
                if st_count else "<div class='dd-empty'>本次没有实验卡</div>")
               + tbl(["次序", "参数", "状态", "实测量", "状态说明"], card_rows,
                     "本次没有实验卡", cap=10))]
    return metrics, blocks


def _diff_of(d) -> str:
    """实验卡：算出来的数与论文值差多少（倍数差，不给结论词）。"""
    r = _ratio(d.get("expected"), d.get("actual"))
    if r is None:
        return "缺一侧的数值"
    return "相同" if r == 1 else f"差 {r:.2f} 倍"


def _pos_calculator(audit, cn: str):
    comp = _g(audit, "compute", {}) or {}
    fs = _findings(audit)
    unver = [f for f in fs if _g(f, "status") in
             (Status.UNVERIFIABLE, Status.NOT_FIXED)]
    cov = _g(audit, "coverage", {}) or {}
    metrics = [("有值的参数", _na(audit, cov.get("kinds_valued", 0) or 0), "g"),
               ("见到的参数", _na(audit, cov.get("kinds_seen", 0) or 0), ""),
               ("无法静态验证", _na(audit, len(unver)), "o" if unver else "dim"),
               ("复算记录", _na(audit, _n(comp)), "b" if comp else "dim")]
    run = [
        ("拿两侧原值", "只用已经抽好的值，不重读论文，也不看模型说了什么。", "原值"),
        ("归一化再比", "1e-4 与 0.0001 折算成同一个数 —— 写法不同不等于数值不同。",
         f"{cov.get('kinds_valued', 0) or 0} 种"),
        ("独立算一遍", "用白名单表达式自己算，不引用论文里的中间结果。",
         "复算结果" if comp else "本次未做"),
    ]
    norm_rows = []
    for f in fs[:10]:
        pe, ce = _g(f, "paper"), _g(f, "code")
        st_ = _g(f, "status")
        same = ("两侧相同（写法不同也折算成一个）" if st_ == Status.CONSISTENT else
                "两侧不同" if st_ == Status.INCONSISTENT else "缺一侧，折不了")
        norm_rows.append([_g(f, "display_name", "") or _g(f, "param_key", ""),
                          _g(pe, "raw_value", "") or "—",
                          _g(ce, "raw_value", "") or "—",
                          same])
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}怎么把论文与代码的数放到同一把尺子上",
               bars([("见到的参数名", cov.get("kinds_seen", 0) or 0, ""),
                     ("真正取到值的", cov.get("kinds_valued", 0) or 0, "")],
                    "b", " 种", cap=4)
               + tbl(["参数", "论文原值", "代码原值", "折算后"], norm_rows,
                     "还没开始跑 —— 跑完这里逐条对照", cap=10)),
              (f"{cn}这一轮的交出物",
               kv(识别出的值=_na(audit, cov.get("kinds_valued", 0) or 0, " 种"),
                  无法静态验证=_na(audit, len(unver), " 条"),
                  闭环复算="有" if comp else "本次未做"))]
    return metrics, blocks


def _pos_reviewer(audit, cn: str):
    fs = _findings(audit)
    sc = _g(audit, "selfcheck", {}) or {}
    cov = _g(audit, "coverage", {}) or {}
    disagree = [f for f in fs if _g(f, "llm_agree") is False]
    metrics = [("判定条数", _na(audit, len(fs)), "b"),
               ("查证到原文", "—" if audit is None
                else f"{sc.get('passed', 0)} / {sc.get('checked', 0)}",
                "g" if sc.get("failed", 0) == 0 else "r"),
               ("可信度", cov.get("confidence_level") or "—", ""),
               ("分歧并列", _na(audit, len(disagree)), "o" if disagree else "dim")]
    run = [
        ("收两侧的值与出处", "只收结构化字段，不重读原文 —— 验的人不看提的人的结论。",
         f"{len(fs)} 条"),
        ("三路各查一遍", "对撞（另一观察者再看）、自检（回材料查证据真假）、复算（自己算）。",
         "三路独立"),
        ("出判定卡", "每条都带两侧值与出处；模型说的话单独一格并列，不覆盖。", "判定卡"),
    ]
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}的判定落在论文哪几页",
               _paper_block(audit, cn, _paper_rows(audit, fs),
                            "还没开始跑 —— 跑完这里会显示每条判定对应的论文页",
                            "页号取自论文侧证据的出处，同一条只算一次。")),
              (f"{cn}这一轮的交出物",
               stack([("一致", len([f for f in fs if _g(f, "status") == Status.CONSISTENT]), "g"),
                      ("两数不一致", len([f for f in fs
                                     if _g(f, "status") == Status.INCONSISTENT]), "r"),
                      ("缺一侧", len([f for f in fs if _g(f, "status") in
                                   (Status.MISSING_IN_CODE, Status.MISSING_IN_PAPER)]), "b"),
                      ("其他", len([f for f in fs if _g(f, "status") in
                                 (Status.NOT_FIXED, Status.UNVERIFIABLE,
                                  Status.INTERNAL_INCONSISTENT)]), "y")])
               if fs else "<div class='dd-empty'>本次没有判定卡</div>")]
    if disagree:
        blocks.append((f"{cn}记下的分歧（模型与规则说法不一样）",
                       tbl(["参数", "规则判定", "模型判定"],
                           [[_g(f, "display_name", "") or _g(f, "param_key", ""), st_zh(f),
                             _ST_ZH.get(_g(f, "llm_status"), "—")] for f in disagree],
                           "", cap=8)))
    return metrics, blocks


def _pos_writer(audit, cn: str):
    rc = _g(audit, "rootcause", None) or []
    pt = _g(audit, "patches", None) or []
    rec = _rec_of(audit, "report")
    loop = _g(audit, "loop", {}) or {}
    issues = [f for f in _findings(audit) if _g(f, "status") != Status.CONSISTENT]
    metrics = [("根因", _na(audit, len(rc)), "b"),
               ("草案", _na(audit, len(pt)), "g" if pt else "dim"),
               ("报告", "—" if audit is None else
                ("已生成" if rec.get("status") == "done" else "未生成"),
                "g" if rec.get("status") == "done" else "dim"),
               ("耗时", _sec(rec.get("seconds")), "")]
    run = [
        ("收判定卡", "只拿审查员签过字的判定，不自己改判定。",
         f"{_n(_findings(audit))} 条"),
        ("追根因、写草案", "哪一侧的值最终生效；改哪个文件、改成什么 —— 只出草案。",
         f"{len(rc)} 根因 / {len(pt)} 草案"),
        ("写报告", "把结论、证据、修正建议排成一份能给别人看的文档。",
         "已出报告" if rec.get("status") == "done" else "未出报告"),
    ]
    blocks = [(f"{cn}的运行方式（三步）", flow(run)),
              (f"{cn}要写的那些问题落在论文哪几页",
               _paper_block(audit, cn, _paper_rows(audit, issues),
                            "还没开始跑 —— 跑完这里会显示要写进报告的问题分布在论文哪几页",
                            "只算不通过的项；一致的项不进报告正文。")),
              (f"{cn}这一轮的交出物",
               kv(根因=_na(audit, len(rc), " 条"), 草案=_na(audit, len(pt), " 条"),
                  报告="已生成" if rec.get("status") == "done" else "未生成",
                  回边轮次=(loop.get("rounds") or "—")))]
    st_count: dict = {}
    for x in pt:
        s = str(_d(x).get("status") or "未标注")
        st_count[s] = st_count.get(s, 0) + 1
    if st_count:
        blocks.append((f"{cn}的草案都是什么状态",
                       stack([(k, v, "g" if "可用" in k or "待" in k else "o")
                              for k, v in st_count.items()])))
    return metrics, blocks


# ---------------------------------------------------------------- 四站视图
def station_view(audit, stg: str):
    """四站：这一站**完成什么功能、在论文与代码里起什么作用、输出什么**。三块。"""
    keys = _tools_of(stg)
    walked = [s for s in STEPS if STATION_OF_STEP.get(s) == stg]
    done_cp = set()
    for s in _steps_of(audit):
        if (s.get("status") or "") == "done":
            done_cp.update(s.get("checkpoints") or ())
    hit = [s for s in walked if s in done_cp]
    _order = list(STATIONS)
    i = _order.index(stg) if stg in _order else -1
    nxt = (_order[i + 1] if 0 <= i < len(_order) - 1 else "（终点：出报告）") if i >= 0 else ""
    metrics = [("站内格", f"{len(hit)} / {len(walked)}",
                "g" if walked and len(hit) == len(walked) else "b"),
               ("工序", str(len(keys)), ""),
               ("下一站", nxt, "dim"),
               ("耗时", _sec(sum(float(_rec_of(audit, t.key).get("seconds") or 0)
                                for t in keys)), "")]
    # ① 功能：这一站把什么变成什么
    feat = _ST_FEATURE.get(stg, ("", ""))
    blocks = [(f"{stg}完成的功能",
               f"<div class='dd-sub'>{_e(feat[0])}</div>"
               + flow([(t.title, _TOOL_WHAT.get(t.key, ""), _yield(audit, t.key) or "—")
                       for t in keys] or [("本版这一站没有登记工序", "", "")]))]
    # ② 作用：论文侧碰不碰 / 本站做什么 / 代码侧碰不碰
    act_rows = []
    for t in keys:
        act_rows.append([_side_zh(t.needs, t.gives, "paper"),
                         _TOOL_WHAT.get(t.key, t.title),
                         _side_zh(t.needs, t.gives, "code")])
    blocks.append((f"{stg}在论文与代码里的作用",
                   tbl(["论文这一侧", "本站做的事", "代码这一侧"], act_rows,
                       "这一站没有工序", cap=10, widths=(20, 60, 20))
                   + f"<div class='dd-note'>{_e(feat[1])}</div>"))
    # ③ 输出
    out_rows = [[t.title, _yield(audit, t.key) or "—",
                 "、".join(t.gives) or "—"] for t in keys]
    blocks.append((f"{stg}的输出结果",
                   tbl(["工序", "这一轮产出", "交接物字段"], out_rows,
                       "这一站没有工序", cap=10)
                   + f"<div class='dd-note'>整站交给：{_e(nxt or '—')}"
                     f"　·　站内 {len(hit)} / {len(walked)} 格走到过</div>"
                   + ("<div class='dd-lg' style='gap:14px'>" + "".join(
                       f"<s class='{'g' if s in done_cp else 'dim'}'>{_e(s)}</s>"
                       for s in walked) + "</div>" if walked else "")))
    return (title_of(stg),
            "三块各回答一句：这一站做什么、在论文与代码里起什么作用、把什么交给下一站。",
            metrics, blocks)


def _side_zh(needs, gives, side: str) -> str:
    """一件工序在这一侧到底碰不碰材料 —— 用 needs/gives 派生，不手写。

    为什么要看 gives：`代码扫描` 的 needs 是空的（不需要任何前置），
    它自己也产出 code —— 只看 needs 会把它写成「不读代码」，那是错的。
    """
    in_side = (side in (needs or ())) or (side in (gives or ()))
    if not in_side:
        return "不碰这一侧"
    if side in (gives or ()):
        return "读论文原文" if side == "paper" else "扫代码"
    return "用论文侧的值" if side == "paper" else "用代码侧的值"


#: 四站各一句「功能」，以及它在论文/代码两侧的作用（口径：四站的分界看输入与输出）
_ST_FEATURE = {
    STATIONS[0]: ("把原始材料变成一张张可被证伪的假设卡：论文写了什么、代码写了什么、"
                  "哪里对不上、要怎么推翻它。",
                  "论文这一侧只读正文与图表声明，代码这一侧只读静态赋值 —— 两侧在一站之内不见面。"),
    STATIONS[1]: ("把每条假设变成一次真算：选一件能算的工具，跑一遍，拿到实际值。"
                  "这一站只出数，不判断对错。",
                  "论文值只当输入用（写进实验卡的「期望值」），代码不执行，只按 AST 里读到的配置算。"),
    STATIONS[2]: ("把两侧的值与证据逐条对照，给每条一个判定，并回原始材料查证据真假。",
                  "论文页号与代码 file:line 在这一站被放在同一行，缺哪一侧都如实标出来。"),
    STATIONS[3]: ("把没对上的项追到根因，出修正草案，最后排成一份能给别人看的报告。",
                  "论文与代码不再被读取，只用上一站交来的判定卡与证据出处。"),
}


def title_of(stg: str) -> str:
    return f"{stg} · 功能 / 作用 / 输出"


# ---------------------------------------------------------------- 判定视图
def _has_both(f) -> bool:
    """两侧都有明确值 = 能落到具体数字上（「具体对应」）；否则只算名称对上。"""
    p, c = _g(f, "paper"), _g(f, "code")
    return bool(p and c and (_g(p, "raw_value") or "") and (_g(c, "raw_value") or ""))


def verdict_view(audit, tone: str):
    """判定：这一类**有没有落到具体值上**。逐条给两侧值与「是否具体对应」。"""
    want = {"g": Status.CONSISTENT, "r": Status.INCONSISTENT,
            "b": Status.MISSING_IN_CODE, "o": Status.MISSING_IN_PAPER}.get(tone)
    named = {Status.CONSISTENT, Status.INCONSISTENT,
             Status.MISSING_IN_CODE, Status.MISSING_IN_PAPER}
    fs = list(_findings(audit))
    if want is not None:
        fs = [f for f in fs if _g(f, "status") == want]
    elif tone == "y":
        fs = [f for f in fs if _g(f, "status") not in named]
    exact = [f for f in fs if _has_both(f)]
    title = f"判定 · {_TONE_ZH.get(tone, tone)}"
    sub = ("这一类逐条摆出来：两侧是不是都有明确值（能落到具体数字＝具体对应），值是不是相同。"
           "只摆两侧的原值与出处，不复述结论。")
    cat: dict = {}
    for f in fs:
        k = _CAT_ZH.get(_g(f, "category"), str(_g(f, "category") or "未分类"))
        cat[k] = cat.get(k, 0) + 1
    risk: dict = {}
    for f in fs:
        r = str(getattr(_g(f, "risk"), "value", _g(f, "risk")) or "none")
        risk[r] = risk.get(r, 0) + 1
    metrics = [("这一类", _na(audit, len(fs)), tone if tone in "grboy" else "b"),
               ("具体对应", _na(audit, len(exact)), "g" if exact else "dim"),
               ("仅名称对应", _na(audit, len(fs) - len(exact)),
                "o" if len(fs) > len(exact) else "dim"),
               ("高风险", _na(audit, risk.get("high", 0)), "r" if risk.get("high") else "dim")]
    blocks = [("具体对应还是只对上了名字",
               (matrix([("两侧都有明确值 · 具体对应", len(exact), "能落到数上", "g"),
                        ("只有一侧有值 · 仅名称对应", len(fs) - len(exact),
                         "另一侧缺", "o" if len(fs) > len(exact) else "dim"),
                        ("两侧值相同", len([f for f in exact
                                       if _g(f, "status") == Status.CONSISTENT]), "", "b"),
                        ("两侧值不同", len([f for f in exact
                                       if _g(f, "status") == Status.INCONSISTENT]), "", "r")])
                if audit is not None else
                "<div class='dd-empty'>还没开始跑 —— 这里会显示「具体对应」与"
                "「只对上名字」各几条。</div>"))]
    rows = []
    for f in fs[:14]:
        pe, ce = _g(f, "paper"), _g(f, "code")
        rows.append([_g(f, "display_name", "") or _g(f, "param_key", ""),
                     (_g(pe, "raw_value", "") or "—") + " " + (_page_of(_g(pe, "location", ""))),
                     (_g(ce, "raw_value", "") or "—") + " " + (_g(ce, "location", "") or ""),
                     "是" if _has_both(f) else "否（仅名称）"])
    blocks.append(("这一类里都有谁、是不是具体对应",
                   tbl(["参数", "论文侧", "代码侧", "具体对应"], rows,
                       "还没开始跑 —— 跑完这里逐条列出来" if audit is None
                       else "这一类没有条目", cap=14)))
    # 幅度只在「两数不一致」这一类上有意义：一致的项按定义就没有偏离，
    # 把它们也画成条只会让人以为「一致的也差了 100 倍」。
    amp = []
    if tone == "r":
        for f in exact:
            r = _ratio(_g(_g(f, "paper"), "normalized"), _g(_g(f, "code"), "normalized"))
            if r is not None and r > 1.0001:
                amp.append((_g(f, "display_name", "") or _g(f, "param_key", ""),
                            round((r - 1) * 100), f"{r:.2f} 倍"))
    if amp:
        amp.sort(key=lambda x: -x[1])
        blocks.append(("差得最多的几条（相对偏离，%，只给倍数）",
                       bars(amp, "r", "%", cap=10)))
    return title, sub, metrics, blocks


# ---------------------------------------------------------------- 事件视图
#: 不占进度格的工具（如化学核验）也要能说出「它属于哪一站」——
#: 否则时间线上会冒出一行「站 = —」，看起来像没有归属。
_TOOL_ST_FALLBACK = {"chem_check": STATIONS[0]}


def _stg_of(key, cps) -> str:
    if cps:
        stg = STATION_OF_STEP.get(cps[0], "")
        if stg:
            return stg
    return _TOOL_ST_FALLBACK.get(key or "", "")


def _strip_items(audit) -> list:
    out = []
    for i, s in enumerate(_steps_of(audit), start=1):
        out.append((i, _stg_of(s.get("key"), s.get("checkpoints")),
                    s.get("status") or ""))
    return out


def timeline_view(audit):
    """事件：整轮**按真实顺序**走了一遍，每步在哪一站、花了多久、留下什么。"""
    steps = _steps_of(audit)
    if not steps:
        return ("时间线", "跑完之后这里会按真实顺序列出每一步。", [],
                [("说明", "<div class='dd-empty'>本次还没有执行记录。</div>")])
    total = sum(float(s.get("seconds") or 0) for s in steps)
    metrics = [("总步数", str(len(steps)), "b"),
               ("总耗时", f"{total:.2f}s", ""),
               ("跳过", str(len([s for s in steps if s.get("status") == "skipped"])), "o"),
               ("失败", str(len([s for s in steps if s.get("status") == "failed"])), "r")]
    rows = []
    for i, s in enumerate(steps, start=1):
        key = s.get("key") or ""
        stg = _stg_of(key, s.get("checkpoints"))
        rows.append([i, stg or "—", s.get("title") or "",
                     _TOOL_WHAT.get(key, "")[:40],
                     _sec(s.get("seconds"))])
    blocks = [("它在全流程里的位置",
               strip(_strip_items(audit))
               + f"<div class='dd-note'>按真实执行顺序排，颜色是四站；"
                 f"整轮共 {len(steps)} 步（固定菜单是 {len(STEPS)} 格，"
                 f"计划外的闭环步也会出现在这里）。</div>"),
              ("每一步在哪一站、做了什么、留多久",
               tbl(["序", "站", "工序", "这一步做什么", "耗时"], rows,
                   "没有执行记录", cap=20))]
    slow = sorted(steps, key=lambda s: -float(s.get("seconds") or 0))[:4]
    blocks.append(("最慢的几步",
                   bars([(s.get("title") or "", round(float(s.get("seconds") or 0), 2), "")
                         for s in slow], "b", "s", cap=4)))
    return ("时间线 · 这一轮是按什么顺序走过来的",
            "只讲顺序与位置：它在第几步、属于哪一站；做了什么在「这一步做什么」一列。",
            metrics, blocks)


def event_view(audit, idx: int):
    """单条事件：它在全流程的第几步、属于哪一站、这一步做了什么。"""
    steps = _steps_of(audit)
    if not (0 <= idx < len(steps)):
        return ("时间线", "", [], [("说明", "<div class='dd-empty'>这条事件已经不在了。</div>")])
    s = steps[idx]
    total = sum(float(x.get("seconds") or 0) for x in steps) or 1.0
    sec = float(s.get("seconds") or 0)
    stg = _stg_of(s.get("key"), s.get("checkpoints"))
    t = next((x for x in TOOLS if x.key == s.get("key")), None)
    metrics = [("第几步", f"{idx + 1} / {len(steps)}", "b"),
               ("属于", stg or "闭环", ""),
               ("耗时", f"{sec:.2f}s", ""),
               ("占整轮", f"{sec * 100 / total:.1f}%", "o" if sec * 100 / total > 40 else "")]
    prev = steps[idx - 1].get("title") if idx > 0 else "（链头）"
    nxt = steps[idx + 1].get("title") if idx + 1 < len(steps) else "（链尾）"
    blocks = [("它在全流程里的位置",
               strip(_strip_items(audit), idx)
               + kv(上一步=prev, 这一步=s.get("title") or "", 下一步=nxt,
                    落在哪一站=stg or "计划外的闭环步")),
              ("这一步做了什么",
               f"<div class='dd-sub'>{_e(_TOOL_WHAT.get(s.get('key') or '', '本版还没写一句说明。'))}"
               f"</div>"
               + kv(产物=(_yield(audit, s.get("key") or "") or "无"),
                    状态={"done": "已完成", "skipped": "计划跳过",
                          "failed": "执行失败"}.get(s.get("status") or "",
                                                   s.get("status") or "—"))
               + (f"<div class='dd-note'>计划给的理由：{_e(s.get('reason'))}</div>"
                  if s.get("reason") else ""))]
    if t is not None:
        blocks.append(("它读什么、交什么",
                       kv(读=("、".join(t.needs) or "无需前置"),
                          交=("、".join(t.gives) or "无"))))
    return (f"事件 {idx + 1} · {s.get('title') or ''}",
            "只讲这一条在全流程里的位置与它做了什么；它读到的原文，点中栏同一行看。",
            metrics, blocks)


# ---------------------------------------------------------------- 就绪度视图
def ready_view(audit):
    rd = _g(audit, "readiness", {}) or {}
    cats = rd.get("categories") or {}
    conf = rd.get("audit_confidence") or rd.get("conf") or {}
    metrics = [("总分", str(rd.get("total", "—")), "g"),
               ("可信度", str(conf.get("level") or "—"), ""),
               ("维度", str(len(cats)), "b"),
               ("影响最大", str(_n(rd.get("dominant_issues"))), "o")]
    blocks = [("五个维度各多少分（缺一个维度就不给总分）",
               bars([(k, v, "") for k, v in cats.items()], "g", " /100", cap=6)
               if cats else "<div class='dd-empty'>本次一个维度都没核验，所以不给分。</div>")]
    dom = rd.get("dominant_issues") or []
    blocks.append(("拖分最多的项",
                   tbl(["项"], [[str(x)] for x in dom], "没有明显拖分项", cap=8)))
    unv = rd.get("unverified") or {}
    blocks.append(("没有核验到的维度（不是 0 分，是没算）",
                   tbl(["维度", "为什么"], [[k, str(v)[:70]] for k, v in unv.items()],
                       "所有维度都核验过了", cap=6)))
    rows = [[k, str(v)[:60]] for k, v in rd.items()
            if k not in ("score", "level", "categories", "dominant_issues",
                         "unverified", "audit_confidence", "conf")]
    blocks.append(("逐项明细", tbl(["项", "值"], rows, "本次没有就绪度明细", cap=10)))
    return ("综合复现就绪度怎么算出来的",
            "分数只由可复算的检查项按维度汇总，不做主观加权；缺维度就是缺，不补 0。",
            metrics, blocks)
