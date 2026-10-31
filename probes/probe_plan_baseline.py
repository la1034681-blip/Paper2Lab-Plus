"""验证「模型排计划 ＋ 四站主干底线」在**真模型**上的表现（本机 intern-s1-mini）。

做三件事，只做这三件：
  1. 让真模型排一次计划，**原样打印它的原始提案**（它想让哪些工具别跑）；
  2. 把这份提案喂给 `make_plan`，看最终计划里 ① 假设 / ② 实验 还在不在；
  3. 可选：拿这份计划真跑一遍审计，核对「假设卡 / 实验卡」是不是真出得来。

用法（默认只做 1+2，不跑审计）：
    .venv/Scripts/python.exe -X utf8 probes/probe_plan_baseline.py
    .venv/Scripts/python.exe -X utf8 probes/probe_plan_baseline.py --audit
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import envfile, llm, planner                        # noqa: E402
from paper2lab.science_tools import CATALOG                        # noqa: E402
from paper2lab.toolbox import Job, TOOLS                           # noqa: E402

KEY = envfile.get("LLM_API_KEY")
URL = envfile.get("LLM_BASE_URL")
MODEL = envfile.get("LLM_MODEL")


def job() -> Job:
    return Job(fingerprint="probe-baseline",
               pdf_path=str(ROOT / "demo" / "paper.pdf"),
               code_path=str(ROOT / "demo" / "project.zip"),
               api_key=KEY, llm_base_url=URL, llm_model=MODEL,
               llm_planner=True)


def main() -> int:
    print(f"端点 {URL}  模型 {MODEL}")
    J = job()
    base = planner._rule_plan(J)
    possible = {d.tool for d in base.decisions if d.action == "run"}
    entries = [c for c in CATALOG if c.key in possible]
    prompt = planner._llm_prompt(J, base, entries)
    print(f"菜单 {len(entries)} 件；提示词 {len(prompt)} 字")

    t = time.time()
    proposal, note = llm.ask_json(prompt, KEY, mode="plan", max_tokens=900,
                                  base_url=URL, model=MODEL)
    dt = time.time() - t
    print(f"模型排计划 {dt:.1f}s，说明：{note}")
    if not isinstance(proposal, dict):
        print("❌ 模型没给出可解析的提案 —— 这一步本身没验证成功")
        return 1

    want_skip = [str(d.get("tool")) for d in proposal.get("decisions") or []
                 if isinstance(d, dict) and str(d.get("action", "")).lower() == "skip"]
    print(f"\n模型原始提案里判 skip 的：{want_skip or '（一个都没跳）'}")
    for d in proposal.get("decisions") or []:
        if isinstance(d, dict) and str(d.get("action", "")).lower() == "skip":
            print(f"   - {d.get('tool'):20s} {str(d.get('reason'))[:70]}")

    plan = planner.make_plan(J, use_llm=True, proposal=proposal)
    dm = {d.tool: d for d in plan.decisions}
    print(f"\n最终计划 planner={plan.planner} 执行 {len(plan.order)} 件")
    for d in plan.decisions:
        print(f"   {d.action:4} {d.source:5} {d.tool:20s} {d.reason[:64]}")
    print("\nnotes：")
    for n in plan.notes:
        print("   ·", n)

    ok = True
    for k, label in (("hypothesize", "① 假设生成"), ("design_experiment", "② 实验设计与真跑")):
        hit = dm.get(k)
        good = bool(hit and hit.action == "run" and k in plan.order)
        ok = ok and good
        print(f"\n{'✅' if good else '❌'} {label}：{'执行' if good else '不见了'}"
              f"（模型原本{'想跳' if k in want_skip else '没跳'}）")

    if "--audit" in sys.argv:
        # 真跑一遍：把「模型真排出来的这份计划」原样注入执行器，不再让它重新排一次
        # （否则又会多调一次模型，且排出来的东西未必一样）。
        # 计划里被留下的都是纯本地工序（解析 / 假设 / 实验复算 / 对齐 / 自检 / 报告），
        # 所以这一轮不会调模型，一两分钟就能出结果。
        from paper2lab import runner
        from dataclasses import replace as _replace

        def _run_with(plan_, tag: str, job_):
            _orig = runner.make_plan
            runner.make_plan = lambda _j, use_llm=True: plan_
            t0 = time.time()
            try:
                runner.run_job(job_)
            finally:
                runner.make_plan = _orig
            hy = len(getattr(job_, "all_hypotheses", []) or [])
            ex = int(getattr(job_, "experiment_total", 0) or 0)
            print(f"\n[{tag}] {time.time() - t0:.1f}s｜判定 "
                  f"{len(getattr(job_.audit, 'findings', []) or []) if job_.audit else 0} 条"
                  f"｜假设卡 {hy} 张（总数 {getattr(job_, 'hypothesis_total', 0)}）"
                  f"｜实验卡 {ex} 张")
            skipped = [s.get("key") for s in
                       (job_.execution_log or {}).get("steps", [])
                       if s.get("status") == "skipped"]
            print(f"      跳过：{skipped}")
            return hy, ex

        print("\n开始真跑一遍完整审计（注入上面这份计划，纯本地工序）…")
        hy, ex = _run_with(plan, "底线版（当前代码）", J)

        # 对照组：把 ② 改回 skip —— 等价于旧行为（模型说跳就跳）。
        # 两次跑同一份材料、同一份其余决策，唯一的差别就是 ② 跑不跑。
        _dm = {d.tool: d for d in plan.decisions}
        _old = _replace(
            plan,
            order=tuple(k for k in plan.order if k != "design_experiment"),
            decisions=[_replace(d, action="skip",
                                reason="对照：模拟旧行为（模型说跳就跳）")
                       if d.tool == "design_experiment" else d
                       for d in plan.decisions])
        print("\n对照组（把 ② 改回 skip，模拟改之前的行为）…")
        hy_old, ex_old = _run_with(_old, "旧行为（② 照模型跳过）", job())

        print("\n对照结论：")
        print(f"  ① 假设卡：底线版 {hy} 张 / 旧行为 {hy_old} 张")
        print(f"  ② 实验卡：底线版 {ex} 张 / 旧行为 {ex_old} 张")
        if hy > 0 and ex > 0 and ex_old == 0:
            print("  ✅ 底线确实把「② 实验站」从 0 张救回到有产物 —— 改动是有效的")
        elif hy > 0 and ex > 0:
            print("  注：旧行为也出了实验卡（本次模型没跳 ②？），但底线版两站都非空")
        else:
            print("  ❌ 底线版仍有一站为空，需复查")
            ok = False
        ok = ok and hy > 0 and ex > 0

    print("\n" + "=" * 70)
    print("结论：" + ("四站主干底线在真模型上成立" if ok else "底线没有生效，需复查"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
