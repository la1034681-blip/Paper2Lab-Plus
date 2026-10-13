"""阶段五闭环端到端验证 —— 把「搜 / 算 / 做 + 迭代回边」挂到图上跑真实样本。

这一项要证明的**唯一一件事**：新增闭环链**只加能力，不改结论**。
所以最关键的是消融对照：
  同一份 demo 数据，classic / multagent / multagent+loop 三版跑完，
  11 条结论必须逐条参数与状态一致。
对不上就说明闭环链偷偷改了审计结果 —— 那比没有这个功能更糟。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import graph, pipeline
from paper2lab.graph import _PRECONDITIONS, gate
from paper2lab.toolbox import STEPS, TOOLS

CHECKS = 0
FAILURES = []
DEMO = ROOT / "demo"

CLOSED_LOOP_NODES = ("compute", "search", "repro_plan", "loop")


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


def sec(n):
    print("\n" + "─" * 62)
    print(n)
    print("─" * 62)


def sig(audit):
    """结论指纹：逐条 (参数, 状态, 论文值, 代码值) —— 消融对照的唯一判据。"""
    return sorted((f.param_key, f.status.value,
                   f.paper.raw_value if f.paper else "",
                   f.code.raw_value if f.code else "") for f in audit.findings)


def run(**kw):
    t0 = time.time()
    a, _ = graph.run_audit_multagent(str(DEMO / "paper.pdf"),
                                     str(DEMO / "project.zip"),
                                     force_mock=True, **kw)
    return a, time.time() - t0


# ═══════════════════════════════════════ 跑三版

sec("0. 跑三版（classic / multagent / multagent+loop）")

_t0 = time.time()
classic, _ = pipeline.run_audit(str(DEMO / "paper.pdf"),
                                str(DEMO / "project.zip"), force_mock=True)
t_c = time.time() - _t0
multi, t_m = run()
looped, t_l = run(loop=True)
print("    耗时: classic %.1fs | multagent %.1fs | +loop %.1fs" % (t_c, t_m, t_l))

# ═══════════════════════════════════════ 1. 默认不动既有路径

sec("1. 默认关 —— 不启用时既有路径一字未变")

rec_m = {r["key"] for r in multi.execution_log.get("records", [])}
check(not (set(CLOSED_LOOP_NODES) & rec_m),
      "loop=False 时闭环四节点**不在** records 里（默认不动既有路径）")
check(multi.execution_log.get("compute", {}) == {}, "loop=False 时 compute 为空")
check(multi.execution_log.get("loop_enabled") is False, "loop_enabled 如实为 False")
check(multi.execution_log.get("nodes_done") == 16,
      "节点数仍为 16（阶段四原样）: %s" % multi.execution_log.get("nodes_done"))

# ═══════════════════════════════════════ 2. 打开后四节点跑起来

sec("2. loop=True —— 四个节点都跑到且为 done")

rec_l = {r["key"]: r for r in looped.execution_log.get("records", [])}
for k in CLOSED_LOOP_NODES:
    check(k in rec_l, "节点 %s 出现在 records 里" % k)
    check(rec_l.get(k, {}).get("status") == "done", "节点 %s 状态为 done" % k)
check(rec_l.get("compute", {}).get("gate") != "blocked", "compute 过了闸门")
check(looped.execution_log.get("nodes_done") == 20,
      "节点数 20 = 16 + 4: %s" % looped.execution_log.get("nodes_done"))
check(looped.execution_log.get("failed") == 0, "0 失败")
check(looped.execution_log.get("errors") == [], "errors 为空")
check(looped.execution_log.get("loop_enabled") is True, "loop_enabled 如实为 True")

# ═══════════════════════════════════════ 3. 产物挂到 audit 上

sec("3. 产物挂在 audit 上（报告与界面只拿得到 audit）")

check(bool(looped.compute), "audit.compute 有内容")
check(bool(looped.citations), "audit.citations 有内容")
check(bool(looped.repro_plan), "audit.repro_plan 有内容")
check(bool(looped.loop), "audit.loop 有内容")

check("relations" in looped.compute and "magnitudes" in looped.compute,
      "「算」产物含关系式与量纲两块")
check("checks" not in looped.compute, "「算」不写入无关键（不污染结构）")
check(bool(looped.compute.get("notes")), "「算」带说明（含边界声明）")

cov = looped.citations.get("coverage") or {}
check(cov.get("total", 0) > 0, "「搜」统计到结论总数 %s" % cov.get("total"))
check(cov.get("cited", 0) == cov.get("total", 1),
      "demo 的每条结论都有可定位出处: %s/%s" % (cov.get("cited"), cov.get("total")))
check(looped.citations.get("external", {}).get("enabled") is False,
      "外部文献检索如实记为未启用（零成本配置）")

grade = looped.repro_plan.get("grade") or {}
check(grade.get("level") in ("A", "B", "C", "D"), "「做」给出可复现性评级 %s" % grade.get("level"))
check(isinstance(grade.get("score"), int), "评级有分值")
check(bool(grade.get("deductions")), "评级带扣分理由（可逐条复核）")
check(bool(looped.repro_plan.get("markdown")), "「做」产出可直接下载的 markdown")
check("可复现性评级" in (looped.repro_plan.get("markdown") or ""), "markdown 含评级")

lp = looped.loop
check(lp.get("max_rounds") == 3, "迭代回边上限 3 轮")
check(len(lp.get("rounds") or []) <= 3, "实际轮数 %d ≤ 3" % len(lp.get("rounds") or []))
check(bool(lp.get("notes")), "迭代回边带说明")
check(bool(lp.get("markdown")), "迭代回边产出 markdown")

# 阶段五产物同时进了 execution_log（界面走这个通道）
ec = looped.execution_log
check(ec.get("compute") == looped.compute, "execution_log.compute 与 audit.compute 同源")
check(ec.get("citations") == looped.citations, "execution_log.citations 同源")
check(ec.get("repro_plan") == looped.repro_plan, "execution_log.repro_plan 同源")
check(ec.get("loop") == looped.loop, "execution_log.loop 同源")

# ═══════════════════════════════════════ 4. 消融对照（最关键）

sec("4. 消融对照 —— 加闭环链不改任何一条结论")

check(sig(classic) == sig(multi), "classic vs multagent: 结论逐条一致（阶段四既有保证）")
check(sig(multi) == sig(looped),
      "multagent vs multagent+loop: 结论逐条一致（新增链只加能力不改结论）")
check(len(looped.findings) == len(multi.findings),
      "结论条数不变: %d" % len(looped.findings))
check(sig(looped) == sig(classic), "classic vs +loop: 结论逐条一致（端到端最强证据）")

# 迭代回边只观察"审计已报出来的不一致项"，不自己造新结论
_pk = {it["param_key"] for r in (lp.get("rounds") or []) for it in r.get("items", [])}
audited = {f.param_key for f in looped.findings if f.status.value == "inconsistent"}
check(_pk <= audited, "迭代回边只观察审计报出的不一致项 %s ⊆ %s" % (sorted(_pk), sorted(audited)))
check(all(it["strategy"] in ("feasible_fix", "rootcause_candidate", "reverse_to_code")
          for r in (lp.get("rounds") or []) for it in r.get("items", [])),
      "每条留痕都标明了用的是哪一种修正策略")

# ═══════════════════════════════════════ 5. 进度口径与闸门

sec("5. 进度口径不变 + 闸门登记")

steps = looped.execution_log.get("steps") or []
steps_m = multi.execution_log.get("steps") or []
steps_c = classic.execution_log.get("steps") or []
check(all(s["key"] not in CLOSED_LOOP_NODES for s in steps),
      "闭环四节点**不占进度槽**（否则界面进度条会失真）")
check([s["key"] for s in steps] == [s["key"] for s in steps_m],
      "进度槽清单与 loop=False 时**逐项相同**（新增链不改进度口径）")
check(all(s["key"] in {t.key for t in TOOLS} for s in steps),
      "每条进度记录都对应一个真实工具（闭环节点确实没混进来）")
# 阶段四既定的流向: analyze_code → 对撞区 → align → report_findings →（阶段五）→ selfcheck，
# 因此多智能体图里 `llm_analyze` **不会执行**（进度条 9 项；classic 是 10 项）。
# 这是既有事实，如实断言下来（不粉饰、也不在本轮擅自改动阶段四的行为契约）。
_missing_tools = [t.key for t in TOOLS if t.key not in {s["key"] for s in steps}]
check(_missing_tools == ["llm_analyze"],
      "多智能体图按既定流向绕过 llm_analyze，进度条 9 项（差的就是它）: %s" % _missing_tools)
check(len(steps_c) == len(TOOLS), "classic 走全部 10 个工具: %d" % len(steps_c))
check(len(steps) == len(steps_c) - 1,
      "多智能体图比 classic 少 1 项（正是 llm_analyze）: %d vs %d" % (len(steps), len(steps_c)))
check(all(s["status"] in ("done", "skipped", "failed") for s in steps),
      "每条进度记录状态取值合法（界面按这三个值渲染）")

tools = {s["key"] for s in steps}
check(tools <= set(_PRECONDITIONS), "每个进度节点都在闸门登记表里")
for k in CLOSED_LOOP_NODES:
    check(k in _PRECONDITIONS, "闭环节点 %s 也登记了闸门（不是无闸门直通）" % k)

# 闸门真的会拦：没有结论时不许开「算」
g, reason = gate("compute", {"audit": None})
check(g == "blocked", "没有 findings 时 compute 被闸门拦下")
check("findings" in reason, "拦下理由写明缺什么: %s" % reason)

g2, _ = gate("loop", {"audit": None, "code": None})
check(g2 == "blocked", "没有 code 时迭代回边被闸门拦下")

g3, _ = gate("compute", {"audit": type("A", (), {"findings": [1]})()})
check(g3 == "pass", "产物齐备时放行")

check(gate("parse_paper", {})[0] == "pass", "无前置要求的工具直接放行（未被误伤）")

# ═══════════════════════════════════════ 6. 断点续跑不受影响

sec("6. 断点续跑仍可用（加了节点之后）")

# 注意: 断点要直接驱动图来看（run_audit_multagent 是"跑完返回"的封装，
# 它会把暂停时的状态也返回出来，看不出"停没停"）。
_chk = graph._make_checkpointer()
_g = graph.build_multagent_graph(checkpointer=_chk, interrupt_before=["compute"], loop=True)
_state = {
    "pdf_path": str(DEMO / "paper.pdf"), "code_path": str(DEMO / "project.zip"),
    "user_results": None, "api_key": "", "llm_base_url": "", "llm_model": "",
    "force_mock": True, "vision_kwargs": {}, "vision_max_regions": None,
    "enable_agent": True, "agent_budget": {}, "llm_planner": False, "llm_patch": False,
    "needs_index": None, "snapshot": None, "session_fp": "", "fingerprint": "",
    "paper": None, "code": None, "audit": None, "report": "",
    "context_screen": None, "decisions": [], "records": [], "errors": [],
    "current_step": "", "next_step": "",
    "paper_view": {}, "code_view": {}, "numeric_view": {},
    "collisions": [], "arbitrations": [], "red_team": [], "crosscheck": [],
    "role_llm": False, "role_llm_note": "",
    "loop_enabled": True, "loop_rounds": 3,
    "compute": {}, "citations": {}, "repro_plan": {}, "loop_result": {},
}
_cfg = {"configurable": {"thread_id": "closeloop-bp1"}}
_paused = None
for _ev in _g.stream(_state, _cfg, stream_mode="values"):
    _paused = _ev
check(_paused is not None, "图停在断点处")
check(_paused.get("next_step") == "compute",
      "停在 compute 前（阶段五节点也支持断点）: %s" % _paused.get("next_step"))

_g2 = graph.build_multagent_graph(checkpointer=_chk, loop=True)
_final = _state
for _ev in _g2.stream(_state, _cfg, stream_mode="values"):
    _final = _ev
check(_final.get("next_step") == "__end__", "续跑到底: %s" % _final.get("next_step"))
check(len(_final.get("records", [])) >= 20, "续跑后记录数 %d ≥ 20" % len(_final.get("records", [])))
check(_final.get("audit") is not None and bool(_final.get("compute")),
      "续跑后闭环产物也补齐了")
check(_final.get("errors") == [], "续跑无异常")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段五闭环端到端: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_closeloop.py)")
print("=" * 62)
