"""图状态「声明与实传」一致性守卫（2026-10-28 由真 bug 逼出来）。

背景（这条 bug 的实际经过）
--------------------------
`run_audit_multagent` 建了一个初始 state，里面有 `judge_llm: True`；
`_job_from_state` 也确实写了 `judge_llm=state.get("judge_llm")`。
看上去一路通到底，但 `AuditGraphState`（TypedDict）里**漏声明了 `judge_llm`** ——
LangGraph 只透传声明过的键，于是这个 True 在图入口就被丢掉，
Job 恒拿到 False，网页默认档（多智能体）的「模型先判」一次都没跑过，
界面显示「未启用」倒是实话。

经典入口 `run_audit` 不走这张图，所以那边一直是好的 ——
**一条路通、另一条路空转**，只有对两条路分别断言才发现得了。

这条守卫盯三件事
----------------
A 初始 state 里的**每个键**都必须在 `AuditGraphState` 里声明（否则会被静默丢弃）；
B 声明的**每个键**都要在初始 state 里给初值（否则下游拿到 None 也可能静默出错）；
C 关键开关（judge_llm / experiment_llm / role_llm / loop_enabled）真能落到 Job 上。

全程离线、不建图、不调模型，毫秒级。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import graph as G                                  # noqa: E402

ok = 0
bad = 0


def check(cond, label, extra=""):
    global ok, bad
    assert isinstance(cond, bool), f"断言必须是 bool: {label}"
    if cond:
        ok += 1
        print(f"  ✅ {label}{(' — ' + str(extra)) if extra else ''}")
    else:
        bad += 1
        print(f"  ❌ {label}{(' — ' + str(extra)) if extra else ''}")


SRC = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")


def _initial_state_keys() -> set[str]:
    """从源码里抠出 `state = {...}` 那一块的键（不执行建图，避免副作用）。"""
    body = SRC.split("    state = {", 1)[1].split("\n    }", 1)[0]
    return set(re.findall(r'"([a-zA-Z_]+)":', body))


print("=" * 70)
print("A 初始 state 的键必须都声明（漏了会被 LangGraph 静默丢掉）")
print("=" * 70)
ann = set(G.AuditGraphState.__annotations__)
keys = _initial_state_keys()
check(len(keys) >= 60, "抠到了初始 state（数量合理）", f"{len(keys)} 个键")
missing = sorted(keys - ann)
check(not missing, "没有未声明的键（否则开关静默失效）", f"未声明: {missing}")
# 本次 bug 的两个具体键，单独点名 —— 回归时一眼看出是哪个又漏了
for k in ("judge_llm", "fingerprint"):
    check(k in ann, f"`{k}` 已声明（2026-10-28 修的就是它）")

print()
print("=" * 70)
print("B 声明的键必须有初值（否则下游拿到 None）")
print("=" * 70)
noinit = sorted(ann - keys)
check(not noinit, "每个声明过的键都有初值", f"缺初值: {noinit}")

print()
print("=" * 70)
print("C 关键开关真能落到 Job 上（不建图、不调模型）")
print("=" * 70)
base = {
    "pdf_path": "demo/paper.pdf", "code_path": "demo/project.zip",
    "user_results": None, "api_key": "", "llm_base_url": "", "llm_model": "",
    "force_mock": True, "vision_kwargs": {}, "vision_max_regions": None,
    "enable_agent": True, "agent_budget": {}, "llm_planner": False,
    "llm_patch": False, "needs_index": False, "snapshot": None,
    "session_fp": "", "fingerprint": "abc123", "paper": None, "code": None,
    "audit": None, "report": "", "context_screen": None, "decisions": [],
    "plan_dict": {}, "records": [], "errors": [], "current_step": "",
    "next_step": "", "paper_view": {}, "code_view": {}, "numeric_view": {},
    "collisions": [], "arbitrations": [], "red_team": [], "crosscheck": [],
    "role_llm": True, "role_llm_note": "",
    "judge_llm": True, "experiment_llm": True,
    "hypotheses": [], "all_hypotheses": [], "hypothesis_total": 0,
    "hypothesis_stats": {}, "experiments": [], "experiment_total": 0,
    "experiment_stats": {}, "figure_clues": [], "figure_total": 0,
    "figure_stats": {}, "chem_items": [], "chem_total": 0, "chem_stats": {},
    "chem_fallback": False, "loop_enabled": True, "loop_rounds": 2,
    "compute": {}, "citations": {}, "repro_plan": {}, "loop_result": {},
    "auto_loop": True, "auto_round": 0, "auto_decisions": [], "auto_assume": {},
    "llm_mode": "", "llm_real_calls": 0, "llm_cache_hits": 0, "llm_error": "",
    "llm_shard": {}, "llm_model_used": "",
}
job = G._job_from_state(base)
for attr, want in (("judge_llm", True), ("experiment_llm", True)):
    got = getattr(job, attr, None)
    check(got == want, f"`{attr}` 从 state 落到了 Job 上", f"{got}")
# role_llm / loop_enabled 不进 Job：它们由节点直接读 state（对撞区 / 闭环分支）。
# 所以这里改成「读的地方确实在读 state」，而不是硬套 Job 属性（套了必然 None）。
check('"role_llm": bool(state.get("role_llm"))' in SRC,
      "`role_llm` 在对撞区节点从 state 读（没有被写死成 False）")
check('"compute" if state.get("loop_enabled")' in SRC,
      "`loop_enabled` 在分支判断时从 state 读")
check(getattr(job, "fingerprint", "") == "abc123", "fingerprint 从 state 落到了 Job 上",
      getattr(job, "fingerprint", ""))

# 反证：不给这个键时必须是 False（默认行为不变，别把默认值改开）
base2 = dict(base)
base2.pop("judge_llm")
check(getattr(G._job_from_state(base2), "judge_llm") is False,
      "没传 judge_llm 时默认关（默认行为没被改）")

print()
print(f"===== 汇总：{ok + bad} 项，通过 {ok}，未通过 {bad} =====")
sys.exit(1 if bad else 0)
