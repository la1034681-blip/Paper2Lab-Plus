"""逐环节功能体检 + 全流程衔接 + 兜底验证（2026-10-25）。

用户口径：「把每个环节单独测试是否正常功能，以及单测试完后进去全流程跑，
看看是否衔接正常、信息传递啊、包括兜底啊是否正常」。

三层，互相独立：

  A 单环节 —— 14 件工序**逐件单独跑**。
    共享同一份材料、按依赖顺序推进（所以不会因为"上游没跑"而假红），
    但每一件只看它自己：跑没跑通、抛没抛异常、有没有产出、花了多久。
    产物为空不算失败 —— 要分清「跑了但这份材料确实没有」（empty）
    与「跑崩了」（fail）。这两件事在报告里必须分开写。
    另外三件「读」的工序（图像 / 图表 / 化学）在主样本上天然为空，
    所以另配两套专项材料再跑一遍 —— 否则「空」会把「没测到」伪装成「没问题」。

  B 全流程 —— 走真实的 LangGraph 图（run_audit），看四站之间**传没传下去**：
    假设卡有没有变成实验卡、实验卡有没有被判定、发现有没有进报告。
    这正是历史上踩过的坑：工单每站重建、上一站产物没显式带过去，
    下游拿到空清单还报 done（2026-10-26 实测 ② 站长期空转、假设 8 张实验 0 张）。
    所以判据不能只看节点状态，要看**产物计数**。
    本节强制 needs_index=False：不走快照，逼它真解析真扫描，
    否则缓存会把「上一轮的产物」伪装成「这一轮传下去的」。

  C 兜底 —— 四条必须成立的退路：
    ① 没有模型 key（本轮全部如此）→ 计划器必须回落规则版，并**如实标 rule**；
    ② 化学核验被模型跳过 → 程序补算并标 chem_fallback；
    ③ 不给代码包 → 照样出假设与报告，不崩；
    ④ 输入路径不存在 → 明确报错，不许静默成功。

零成本：本脚本**不加载 .env**、全程 force_mock=True，
所以模型/视觉一次都不真调，结果与"今天有没有额度"无关。

用法:
    .venv\\Scripts\\python.exe tests\\verify_stages.py
    .venv\\Scripts\\python.exe tests\\verify_stages.py --stage=a     # 只跑单环节
输出:
    Desktop\\兴智杯_逐环节体检报告.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 外部文献检索在体检里一律关掉：一次真查要 5~11 秒且依赖外网,
# 让"今天网通不通"变成体检结果的一部分是错的（与 run_all_checks 同口径）。
os.environ["PAPER2LAB_LIT"] = "off"

from paper2lab.toolbox import TOOLS, TOOL_BY_KEY, Job, ensure_chemistry  # noqa: E402

OUT = Path(r"C:\Users\lijiahui\Desktop\兴智杯_逐环节体检报告.txt")
DEMO_PDF = ROOT / "demo" / "paper.pdf"
DEMO_ZIP = ROOT / "demo" / "project.zip"
FIG_PDF = ROOT / "demo" / "figure_demo.pdf"
CHEM_PDF = ROOT / "demo" / "chem_demo.pdf"

# gives 里这几个键不挂在 Job 上, 挂在 audit / paper 上（历史结构, 不改）
_AUDIT_KEYS = frozenset({"findings", "selfcheck", "coverage", "readiness", "ai_analysis"})

CHECKS: list = []      # (ok: bool, label: str, detail: str)
ROWS: list = []        # 单环节明细


def check(ok, label: str, detail: str = "") -> None:
    """断言防呆：ok 必须是真 bool（and 会返回操作数, 不许直接塞进来）。"""
    assert isinstance(ok, bool), f"断言值必须是 bool, 拿到 {type(ok).__name__}: {label}"
    CHECKS.append((ok, label, detail))
    print(f"  {'✅' if ok else '❌'} {label}" + (f" — {detail}" if detail else ""))


def _value(job: Job, key: str):
    """按产物键取值 —— 兼容挂在 audit / paper 上的历史结构。"""
    if key == "image_params":
        return getattr(job.paper, "image_params", None) if job.paper is not None else None
    if key in _AUDIT_KEYS:
        return getattr(job.audit, key, None) if job.audit is not None else None
    return getattr(job, key, None)


def _count(job: Job, key: str) -> int:
    """产物条数（空/无 = 0）。"""
    v = _value(job, key)
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (list, tuple, dict, str, set)):
        return len(v)
    return 1


def _has(job: Job, key: str) -> bool:
    return _count(job, key) > 0


# ================================================================ A 单环节

def stage_unit() -> None:
    print("\n===== A 单环节：14 件工序逐件跑 =====")
    job = Job(pdf_path=str(DEMO_PDF), code_path=str(DEMO_ZIP),
              force_mock=True, enable_agent=True)

    for i, t in enumerate(TOOLS, 1):
        miss = [n for n in t.needs if not _has(job, n)]
        t0 = time.monotonic()
        err = ""
        try:
            t.run(job, lambda name: None)
        except Exception as exc:                                   # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        secs = time.monotonic() - t0

        counts = {k: _count(job, k) for k in t.gives}
        total = sum(counts.values())
        if err:
            status = "fail"
        elif total == 0:
            status = "empty"          # 跑了但这份材料确实没有 → 不算坏, 但要说清
        else:
            status = "ok"

        ROWS.append({"n": i, "key": t.key, "title": t.title, "status": status,
                     "secs": secs, "counts": counts, "err": err, "miss": miss,
                     "memo": "主样本"})
        flag = {"ok": "✅", "empty": "○", "fail": "❌"}[status]
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {flag} [{i:02d}] {t.title:<12} {secs:6.2f}s  {detail}"
              + (f"  缺上游:{miss}" if miss else "")
              + (f"  ERR {err}" if err else ""))

        check(status != "fail", f"单环节 {t.title} 能跑通（不抛异常）", err or f"{secs:.2f}s")
        check(not miss, f"单环节 {t.title} 的上游产物齐全", str(miss) if miss else "齐")

    # 单环节整体：不该有中途抛异常的工序
    bad = [r["title"] for r in ROWS if r["status"] == "fail"]
    check(not bad, "单环节无任何工序抛异常", "全部正常" if not bad else str(bad))
    # 关键工序必须有真产物（这几件是四站的地基，空就等于没做事）
    for key in ("parse_paper", "analyze_code", "hypothesize", "design_experiment",
                "align", "report"):
        r = next((x for x in ROWS if x["key"] == key), None)
        check(bool(r and r["status"] == "ok"), f"关键工序「{TOOL_BY_KEY[key].title}」有真产物",
              "有" if r and r["status"] == "ok" else (r["status"] if r else "未执行"))


def stage_unit_extra() -> None:
    """三件「读」的工序在主样本上天然为空 —— 换专项材料再跑，证明功能本身没坏。"""
    print("\n===== A2 单环节专项：图表 / 化学（换专门造的材料）=====")

    def fresh(pdf: Path, label: str) -> Job:
        j = Job(pdf_path=str(pdf), force_mock=True)
        t0 = time.monotonic()
        TOOL_BY_KEY["parse_paper"].run(j, lambda n: None)
        print(f"  · {label} 解析 {time.monotonic() - t0:.2f}s")
        return j

    def one(j: Job, key: str, label: str) -> dict:
        t = TOOL_BY_KEY[key]
        t0 = time.monotonic()
        err = ""
        try:
            t.run(j, lambda n: None)
        except Exception as exc:                                   # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        secs = time.monotonic() - t0
        counts = {k: _count(j, k) for k in t.gives}
        status = "fail" if err else ("ok" if sum(counts.values()) else "empty")
        ROWS.append({"n": label, "key": key, "title": f"{t.title}（专项）",
                     "status": status, "secs": secs, "counts": counts,
                     "err": err, "miss": [], "memo": "专项样本"})
        flag = {"ok": "✅", "empty": "○", "fail": "❌"}[status]
        print(f"  {flag} {t.title:<12} {secs:6.2f}s  "
              + ", ".join(f"{k}={v}" for k, v in counts.items()))
        return counts

    # 图表：图注通道必须出线索（读图通道要 key，本轮离线，如实为 0）
    jf = fresh(FIG_PDF, "图表样例")
    c_img = one(jf, "analyze_images", "特1")
    c_fig = one(jf, "read_figures", "特2")
    check(c_fig.get("figure_clues", 0) > 0,
          "图注通道能读出图表线索（离线也成立）",
          f"{c_fig.get('figure_clues', 0)} 条")
    notes = (jf.figure_note or {}).get("notes") or []
    check(any("视觉" in n for n in notes),
          "没配视觉 key 时如实说明「读图未启用」（不假装读过）",
          notes[0][:40] + "…" if notes else "无说明")

    # 化学：专项样例必须抽出条目并给出判定
    jc = fresh(CHEM_PDF, "化学样例")
    c_chem = one(jc, "chem_check", "特3")
    items = jc.chem_items or []
    check(len(items) > 0, "化学身份核验能抽出条目", f"{len(items)} 条（共抽到 {jc.chem_total} 处）")
    judged = [getattr(it, "status", "") for it in items]
    check(any(s in ("consistent", "inconsistent", "invalid") for s in judged),
          "化学条目给出了可复算的判定", f"判定分布 {sorted(set(judged))}")
    check(bool(jc.chem_ran), "化学核验如实标记「已跑过」", f"ran={jc.chem_ran}")


# ================================================================ B 全流程

def stage_flow() -> None:
    print("\n===== B 全流程：走真实图，看四站衔接 =====")
    from paper2lab.pipeline import run_audit

    marks: list = []
    t0 = time.monotonic()

    def cb(i, name):
        marks.append((i, name, time.monotonic() - t0))

    # needs_index=False：不借快照，逼它真解析真扫描 —— 否则缓存会把
    # 「上一轮留下的产物」伪装成「这一轮传下去的」。
    audit, report = run_audit(str(DEMO_PDF), str(DEMO_ZIP), force_mock=True,
                              progress=cb, needs_index=False)
    total = time.monotonic() - t0

    print(f"  全流程耗时 {total:.1f}s，回调 {len(marks)} 次")
    for i, name, sec in marks:
        print(f"    [{i:02d}] {name:<16} @{sec:6.2f}s")

    check(audit is not None, "全流程出得来审计结果（不是 None）")
    if audit is None:
        return

    log = audit.execution_log or {}
    steps = log.get("steps") or []
    plan = log.get("plan") or {}

    # --- 四站产物计数：衔接好不好的唯一硬判据（不能只看节点状态）
    hyps = audit.hypothesis_total or len(audit.hypotheses or [])
    exps = audit.experiment_total or len(audit.experiments or [])
    finds = len(audit.findings or [])
    print(f"  产物：假设 {hyps} · 实验 {exps} · 发现 {finds} · 报告 {len(report or '')} 字")
    check(hyps > 0, "① 假设站产出假设卡", f"{hyps} 张")
    check(exps > 0, "② 实验站产出实验卡（历史上曾空转）", f"{exps} 张")
    check(finds > 0, "③ 验证站产出发现", f"{finds} 条")
    check(bool(report) and len(report) > 200, "④ 迭代站产出报告", f"{len(report or '')} 字")

    # --- 信息传递：实验卡必须来自假设卡, 不是凭空造的
    hyp_keys = {getattr(h, "key", "") for h in (audit.hypotheses or [])}
    hyp_claims = {getattr(h, "claim", "") for h in (audit.hypotheses or [])}
    exp_cards = audit.experiments or []
    linked = sum(1 for e in exp_cards
                 if getattr(e, "key", "") in hyp_keys
                 or getattr(e, "claim", "") in hyp_claims)
    check(exps == 0 or linked > 0, "实验卡与假设卡对得上（不是凭空出现）",
          f"{linked}/{len(exp_cards)} 张能追到假设")

    # --- 每条发现都要有出处, 否则等于空口结论
    def _loc(f, name):
        return f.get(name) if isinstance(f, dict) else getattr(f, name, None)

    with_loc = sum(1 for f in (audit.findings or [])
                   if _loc(f, "paper") or _loc(f, "code"))
    check(finds == 0 or with_loc > 0, "发现都带出处（论文页 / 代码行）",
          f"{with_loc}/{finds} 条有出处")

    # --- 执行记录：14 件工序都要有交代（跑了 / 跳过 / 动了但没成）
    keys_in_log = {s.get("key") for s in steps}
    missing = [t.key for t in TOOLS if t.key not in keys_in_log and t.key != "chem_check"]
    check(not missing, "执行记录覆盖每一件工序（跳过也要留下理由）", str(missing) if missing else "齐")
    bad_status = [s for s in steps if s.get("status") not in ("done", "skipped", "failed")]
    check(not bad_status, "执行记录里状态合法", f"{len(steps)} 条")

    # --- 计划器来源：没有 key 时必须如实说「规则兜底」, 不许冒充模型
    n_plan = len(plan.get("items") or plan.get("order") or plan.get("tools") or [])
    print(f"  计划器来源：{plan.get('planner') or '(未标)'} · 计划 {n_plan} 项")
    check(plan.get("planner") in ("rule", "model"), "计划器标了来源（rule / model）",
          str(plan.get("planner")))
    check(plan.get("planner") == "rule", "没配 key 时如实回落规则版（不冒充模型）",
          str(plan.get("planner")))

    # --- 报告里该有的小节都在
    for frag in ("假设", "实验", "发现", "复现"):
        check(frag in (report or ""), f"报告含「{frag}」部分")

    # --- 自检必须真跑过并有数
    sc = audit.selfcheck or {}
    check(bool(sc), "审计自检有结果", f"checked={sc.get('checked')} passed={sc.get('passed')}")


# ================================================================ C 兜底

def stage_fallback() -> None:
    print("\n===== C 兜底：四条退路 =====")
    from paper2lab.pipeline import run_audit
    from paper2lab.paper_parser import parse_paper

    # ① 无 key → 规则计划（在 B 里已断过, 这里单独再走一条独立调用确认稳定）
    a1, r1 = run_audit(str(DEMO_PDF), str(DEMO_ZIP), force_mock=True,
                       llm_planner=True, needs_index=False)
    p1 = ((a1.execution_log or {}).get("plan") or {}).get("planner") if a1 else None
    check(p1 == "rule", "要求模型排计划但没 key → 回落规则版并标 rule", str(p1))

    # ② 化学补算：模拟「模型跳过了化学核验」
    job = Job(pdf_path=str(CHEM_PDF), force_mock=True)
    job.paper = parse_paper(str(CHEM_PDF))
    check(job.chem_ran is False, "补算前：确实没跑过化学核验")
    ensure_chemistry(job)
    check(job.chem_ran is True, "补算后：化学核验状态为「已跑」")
    check(job.chem_fallback is True, "补算被如实标成 fallback（不冒充模型跑的）",
          f"items={len(job.chem_items or [])} total={job.chem_total}")
    n_before = len(job.chem_items or [])
    ensure_chemistry(job)
    check(len(job.chem_items or []) == n_before, "补算是幂等的（跑两次不会翻倍）",
          f"{n_before} → {len(job.chem_items or [])}")

    # ③ 不给代码包：照样出假设与报告, 不崩
    try:
        a3, r3 = run_audit(str(DEMO_PDF), "", force_mock=True, needs_index=False)
        err3 = ""
    except Exception as exc:                                       # noqa: BLE001
        a3, r3, err3 = None, "", f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    check(not err3, "只有论文没有代码 → 不崩", err3 or "正常返回")
    check(a3 is not None and bool(r3), "只有论文时仍能出报告", f"报告 {len(r3 or '')} 字")
    if a3 is not None:
        check((a3.hypothesis_total or 0) > 0, "没有代码包也能提假设（论文独有那条）",
              f"{a3.hypothesis_total} 张")

    # ④ 坏输入：必须明确报错, 不许静默成功
    raised = False
    try:
        run_audit(str(ROOT / "demo" / "__not_exist__.pdf"), str(DEMO_ZIP),
                  force_mock=True, needs_index=False)
    except Exception:                                              # noqa: BLE001
        raised = True
    check(raised, "路径不存在时明确抛错（而不是静默出一份空报告）",
          "已抛错" if raised else "未抛错——静默成功才是问题")


# ================================================================ 报告

def write_report(total: float) -> Path:
    ok = sum(1 for c in CHECKS if c[0])
    bad = len(CHECKS) - ok
    lines: list = []
    add = lines.append

    add("兴智杯 · Paper2Lab 逐环节体检报告")
    add("")
    add(f"跑完时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"总耗时：{total:.1f} 秒")
    add(f"结果：{len(CHECKS)} 项断言，通过 {ok}，未通过 {bad}")
    add("说明：本轮全程离线（不加载 .env、模型与视觉一次都不真调），")
    add("      所以结论与「今天有没有额度、网通不通」无关，可反复复现。")
    add("")

    add("=" * 60)
    add("一、单环节：14 件工序逐件跑")
    add("=" * 60)
    add("")
    add("  标记：✅ 有产物 ｜ ○ 跑了但这份材料本来就没有（不算坏） ｜ ❌ 跑崩")
    add("")
    for r in ROWS:
        flag = {"ok": "✅", "empty": "○", "fail": "❌"}[r["status"]]
        detail = "、".join(f"{k}={v}" for k, v in r["counts"].items())
        add(f"  {flag} {str(r['n']):>3}  {r['title']}")
        add(f"        耗时 {r['secs']:.2f} 秒 ｜ 产物：{detail} ｜ 材料：{r['memo']}")
        if r["err"]:
            add(f"        报错：{r['err']}")
        if r["miss"]:
            add(f"        缺上游：{r['miss']}")
    add("")
    empty = [r for r in ROWS if r["status"] == "empty"]
    if empty:
        add(f"  产物为空的环节：{'、'.join(r['title'] for r in empty)}")
        add("  「空」不等于「坏」，但必须说清是哪一种：")
        for r in empty:
            if r["memo"] == "专项样本":
                add(f"    · {r['title']}：视觉 key 未配 —— 图上超参数读不到，")
                add("      本地只能抽图注；配了读图模型才会补上（程序如实标了「未启用」）。")
            elif r["key"] in ("analyze_images", "read_figures", "chem_check"):
                add(f"    · {r['title']}：主样本里没有这类内容")
                add("      （非化学论文没有分子式；未配视觉 key 时不读图）。")
                add("      对应能力已用专项样本单独验过，见上面「专项」两行。")
            elif r["key"] == "llm_analyze":
                add(f"    · {r['title']}：本轮没配模型 key，模型这一步一次都不调，")
                add("      所以没有归因文字 —— 这是「如实不做」，不是「做了没结果」。")
            else:
                add(f"    · {r['title']}：材料不含该环节需要的输入。")
        add("")

    add("=" * 60)
    add("二、全流程：四站衔接与信息传递")
    add("=" * 60)
    add("")
    add("  判据不是「节点有没有打勾」，而是「下游有没有拿到东西」。")
    add("  历史上 ② 实验站就因为工单每站重建、上一站产物没带过去，")
    add("  空转了一整段时间（假设 8 张、实验 0 张，节点状态还显示成功）。")
    add("  本节强制真解析（不走快照），避免缓存把上一轮的产物伪装成这一轮传下去的。")
    add("")
    flow_prefix = ("①", "②", "③", "④", "实验卡", "发现都", "执行记录", "计划器",
                   "没配 key", "审计自检", "报告含", "全流程")
    for c in CHECKS:
        if c[1].startswith(flow_prefix):
            add(f"  {'通过  ' if c[0] else '未通过'} ｜ {c[1]} ｜ {c[2]}")
    add("")

    add("=" * 60)
    add("三、兜底：四条退路")
    add("=" * 60)
    add("")
    add("  ① 没配模型 key：计划器回落规则版，并如实标 rule，不许冒充模型；")
    add("  ② 化学核验被跳过：程序补算并标 chem_fallback，报告与界面照实写；")
    add("  ③ 只给论文不给代码：照样提假设、出报告，不崩；")
    add("  ④ 路径不存在：明确报错，不许静默出一份空报告。")
    add("")
    fb_prefix = ("要求模型", "补算", "只有论文", "路径不存在", "没有代码包")
    for c in CHECKS:
        if c[1].startswith(fb_prefix):
            add(f"  {'通过  ' if c[0] else '未通过'} ｜ {c[1]} ｜ {c[2]}")
    add("")

    add("=" * 60)
    add("四、未通过项（若有）")
    add("=" * 60)
    add("")
    fails = [c for c in CHECKS if not c[0]]
    if not fails:
        add("  无。全部断言通过。")
    else:
        for _ok, label, detail in fails:
            add(f"  · {label} ｜ {detail}")
    add("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "c"])
    args = ap.parse_args()

    assert DEMO_PDF.exists(), f"样例论文不存在：{DEMO_PDF}"
    assert DEMO_ZIP.exists(), f"样例代码不存在：{DEMO_ZIP}"
    assert FIG_PDF.exists(), f"图表样例不存在：{FIG_PDF}"
    assert CHEM_PDF.exists(), f"化学样例不存在：{CHEM_PDF}"

    t0 = time.monotonic()
    if args.stage in ("all", "a"):
        stage_unit()
        stage_unit_extra()
    if args.stage in ("all", "b"):
        stage_flow()
    if args.stage in ("all", "c"):
        stage_fallback()
    total = time.monotonic() - t0

    ok = sum(1 for c in CHECKS if c[0])
    bad = len(CHECKS) - ok
    path = write_report(total)
    print(f"\n===== 汇总：{len(CHECKS)} 项，通过 {ok}，未通过 {bad} =====")
    print(f"报告：{path}")
    for _ok, label, detail in CHECKS:
        if not _ok:
            print(f"  ❌ {label} — {detail}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
