"""指挥台详情视图的离线体检（不起服务、不开浏览器，纯函数级）。

守的是什么
----------
2026-10-27 第三版（用户定版「只留核心信息」）把七块区域的内容重排成
「每块最多三块、每块只回答自己那个问题」：

  七位置 → 运行方式（三步）＋ 与论文哪一块对应（按页条图 ＋ 参数→页→原值）
  四站   → 功能 / 在论文与代码里的作用 / 输出结果
  事件   → 在流程第几步（序列条）＋ 这一步做了什么
  判定   → 具体对应还是只对上名字 ＋ 逐条两侧值

最大的风险不是崩，而是**悄悄退回到「每块都倒同一张 findings 表」**。
所以这里逐条盯五件事：
  A 七位置：流程链 ≥ 3 步 ＋ 至少一种图表 ＋ 文字，且标题点名位置；
  B 六区域：块的标题不许跨区域重复；
  C 数据只从 audit 来：视图里出现的数必须能在 audit 上找到出处；
  D 空态不炸：audit=None 时每块都能渲染，且写「还没跑」而不是填 0；
  E 表述干净：不出现「已通过判定」「由此得出」这类自我背书的句子。
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import deckviews as v          # noqa: E402
from paper2lab.agent_roles import POSITIONS   # noqa: E402
from paper2lab.pipeline import run_audit      # noqa: E402
from paper2lab.toolbox import STATIONS, TOOLS  # noqa: E402

N = 0
BAD: list[str] = []


def check(ok: bool, label: str) -> None:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        BAD.append(label)


def _txt(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "")


def _strip_first(html: str) -> str:
    """去掉 <style>…</style>（那里面全是 CSS，不该拿去比标题）。"""
    return re.sub(r"<style>.*?</style>", "", html or "", flags=re.S)


def main() -> int:
    demo = ROOT / "demo"
    audit, _ = run_audit(str(demo / "paper.pdf"), str(demo / "project.zip"),
                         user_results=json.loads(
                             (demo / "results.json").read_text(encoding="utf-8")))

    # ---------------------------------------------------------------- A 七位置
    print("\n===== A 七位置：流程 ＋ 图表 ＋ 文字 =====")
    for p in POSITIONS:
        title, sub, metrics, blocks = v.pos_view(audit, p.key)
        html = _strip_first(" ".join(h for _, h in blocks))
        steps = html.count("class='dd-fs'")
        # 流程链本身就是最直接的「运行方式可视化」：三步、带步骤号、每步挂产出数。
        chart = (("dd-flow" in html) or ("dd-bt" in html)
                 or ("dd-mx2" in html) or ("dd-bk" in html))
        words = len(_txt(html).strip())
        heads = [h for h, _ in blocks]
        check(p.cn in title, f"{p.key}：标题点名了「{p.cn}」")
        check(steps >= 3, f"{p.key}：运行方式有 {steps} 步（≥3）")
        check(chart, f"{p.key}：带可视化图表")
        check(words >= 100, f"{p.key}：文字说明 {words} 字（≥100，别写成小作文也行）")
        check(all(p.cn in h for h in heads), f"{p.key}：每块标题都点名了位置（不串台）")
        # 2026-10-27 第四版（用户定版）：七位置**只留运行方式**，论文对应整块删掉。
        check(len(blocks) == 1 and "运行方式" in heads[0],
              f"{p.key}：只保留运行方式一块（实际 {len(blocks)} 块）")
        check("交接物" in sub, f"{p.key}：写清了交接物")

    # 位置之间不许撞车：把每块的标题集合两两比一次
    heads = {p.key: {h for h, _ in v.pos_view(audit, p.key)[3]} for p in POSITIONS}
    same = [(a, b) for i, a in enumerate(heads) for b in list(heads)[i + 1:]
            if heads[a] == heads[b]]
    check(not same, f"七个位置的块标题互不相同（撞车的：{same or '无'}）")

    # ---------------------------------------------------------------- B 六区域
    print("\n===== B 六区域：各说各话，不互相抄 =====")
    regions: dict[str, set] = {}
    regions["工序"] = set()
    for t in TOOLS:
        regions["工序"] |= {h for h, _ in v.tool_view(audit, t.key)[3]}
    regions["站点"] = {h for h, _ in v.station_view(audit, STATIONS[0])[3]}
    regions["判定"] = {h for h, _ in v.verdict_view(audit, "r")[3]}
    regions["事件"] = {h for h, _ in v.timeline_view(audit)[3]}
    regions["就绪度"] = {h for h, _ in v.ready_view(audit)[3]}
    regions["位置"] = set().union(*heads.values())
    for key in ("工序", "站点", "判定", "事件", "就绪度", "位置"):
        check(bool(regions[key]), f"{key}：有明细块")
    cross = []
    names = list(regions)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = regions[a] & regions[b]
            if inter:
                cross.append((a, b, sorted(inter)[:2]))
    check(not cross, f"六区域之间没有重复的块标题（撞车的：{cross or '无'}）")

    # 每个区域都得有点「别人没有的东西」
    only = {
        "位置": ("运行方式",),
        "站点": ("功能", "作用", "输出"),
        "判定": ("具体对应",),
        "事件": ("位置",),
        "就绪度": ("维度", "拖分"),
    }
    for key, words in only.items():
        blob = " ".join(regions[key])
        check(any(w in blob for w in words), f"{key}：块标题里出现了它独有的说法")

    # ---------------------------------------------------------------- C 只读
    print("\n===== C 数据只从 audit 来（不造数） =====")
    src = inspect.getsource(v)
    check("open(" not in src and "requests" not in src,
          "视图层不读文件、不发请求（只吃 audit）")
    check(not re.search(r"\brandom\b", src), "视图层没有随机数")
    # 抽查一个具体数：检索员的「两侧都找到」必须等于两侧键的交集
    pk = set((audit.paper.params or {}).keys())
    ck = set((audit.code.params or {}).keys())
    mret = {k: val for k, val, *_ in v.pos_view(audit, "retriever")[2]}
    check(mret.get("两侧配平") == str(len(pk & ck)),
          f"检索员「两侧配平」= {len(pk & ck)}（现场算的前后一致）")
    # 判定分布的数必须加起来等于同类条目数
    fs = [f for f in audit.findings if f.status.value == "inconsistent"]
    h = " ".join(x for _, x in v.verdict_view(audit, "r")[3])
    check(str(len(fs)) in _txt(h) or not fs, f"判定「两数不一致」条数 {len(fs)} 出现在详情里")

    # ---------------------------------------------------------------- D 空态
    print("\n===== D 没跑之前点开不许炸 =====")
    for p in POSITIONS:
        try:
            t, s, m, b = v.pos_view(None, p.key)
            check(bool(t) and isinstance(b, list), f"{p.key}：空态能渲染")
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{p.key}：空态抛了 {type(exc).__name__}: {exc}")
    for t in TOOLS:
        try:
            t2, s2, m2, b2 = v.tool_view(None, t.key)
            check(isinstance(b2, list), f"{t.key}：空态能渲染")
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{t.key}：空态抛了 {type(exc).__name__}: {exc}")
    for st in STATIONS:
        try:
            v.station_view(None, st)
            check(True, f"{st}：空态能渲染")
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{st}：空态抛了 {type(exc).__name__}: {exc}")
    for fn, name in ((v.timeline_view, "时间线"), (v.ready_view, "就绪度"),
                     (v.verdict_view, "判定")):
        try:
            fn(None) if name != "判定" else fn(None, "r")
            check(True, f"{name}：空态能渲染")
        except Exception as exc:                       # noqa: BLE001
            check(False, f"{name}：空态抛了 {type(exc).__name__}: {exc}")

    # 空态不许出现「结论尚未生成」之外的分数字样（不许把没跑说成 0 分）
    h = " ".join(x for _, x in v.tool_view(None, "align")[3])
    check("0 条判定" not in _txt(h), "空态不把「没跑」写成「0 条」")

    # ---------------------------------------------------------------- E 表述
    print("\n===== E 表述干净（不自我背书） =====")
    all_txt = []
    for p in POSITIONS:
        all_txt.append(" ".join(x for _, x in v.pos_view(audit, p.key)[3]))
    for t in TOOLS:
        all_txt.append(" ".join(x for _, x in v.tool_view(audit, t.key)[3]))
    for st in STATIONS:
        all_txt.append(" ".join(x for _, x in v.station_view(audit, st)[3]))
    for tone in "grboy":
        all_txt.append(" ".join(x for _, x in v.verdict_view(audit, tone)[3]))
    all_txt.append(" ".join(x for _, x in v.timeline_view(audit)[3]))
    blob = _txt(" ".join(all_txt))
    for bad in ("已通过判定", "由此得出", "判定通过", "审核通过"):
        check(bad not in blob, f'不出现「{bad}」这类说法')

    # ---------------------------------------------------------------- F 工序↔论文
    print("\n===== F 工序执行：每一项都指明它在这篇论文里的哪一条 =====")
    # 用户 2026-10-27 点名：「工序执行」每一项都要补与本论文内容的详细对应（参数等）。
    # 守两点：① 有对照的那几件必须同时给出「论文第几页」与「代码第几行」；
    #        ② 已经自带专属表的几件（图 / 图注 / 化学 / 两侧 / 补漏）不再叠一张同数据的表。
    for k in ("parse_paper", "analyze_code", "hypothesize", "design_experiment",
              "llm_analyze", "selfcheck", "patch", "report", "rootcause"):
        h = " ".join(x for _, x in v.tool_view(audit, k)[3])
        has_page = bool(re.search(r"第\s*\d+\s*页", h))
        has_line = bool(re.search(r"\.py:\d+", h))
        check(has_page and has_line, f"{k}：写出了论文页与代码行")
    for k in ("analyze_images", "read_figures", "chem_check", "align", "fill_missing"):
        heads_k = [h for h, _ in v.tool_view(audit, k)[3]]
        check(not any("具体对应" in h for h in heads_k),
              f"{k}：不叠第二张同数据的表（已有专属表）")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 58)
    ok = N - len(BAD)
    print(f"共 {N} 项：通过 {ok}，未通过 {len(BAD)}")
    if BAD:
        for x in BAD:
            print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
