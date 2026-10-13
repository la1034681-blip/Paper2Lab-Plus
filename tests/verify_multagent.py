"""多智能体端到端验证（阶段四）——在真实 .venv 里跑三方对撞 + 仲裁 + 红队。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import graph, pipeline

CHECKS = 0
FAILURES = []

DEMO = ROOT / "demo"


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


def run_both():
    t0 = time.time()
    classic, _ = pipeline.run_audit(
        str(DEMO / "paper.pdf"), str(DEMO / "project.zip"), force_mock=True)
    classic_t = time.time() - t0
    t0 = time.time()
    multi, _ = graph.run_audit_multagent(
        str(DEMO / "paper.pdf"), str(DEMO / "project.zip"), force_mock=True)
    multi_t = time.time() - t0
    return classic, classic_t, multi, multi_t


def make_job(audit):
    from paper2lab.toolbox import Job
    return Job(fingerprint="", pdf_path=str(DEMO / "paper.pdf"),
               code_path=str(DEMO / "project.zip"), paper=audit.paper, code=audit.code)


def build_state(pdf, code):
    return {
        "pdf_path": str(pdf), "code_path": str(code),
        "user_results": None, "api_key": "", "llm_base_url": "", "llm_model": "",
        "force_mock": True, "vision_kwargs": {}, "enable_agent": True,
        "agent_budget": {}, "llm_planner": False, "llm_patch": False,
        "needs_index": None, "snapshot": None, "session_fp": "", "fingerprint": "",
        "paper": None, "code": None, "audit": None, "report": "",
        "context_screen": None, "decisions": [], "records": [], "errors": [],
        "current_step": "", "next_step": "",
        "paper_view": {}, "code_view": {}, "numeric_view": {},
        "collisions": [], "arbitrations": [], "red_team": [],
    }


print("1. 图能跑通")
classic, classic_t, multi, multi_t = run_both()
check(classic is not None, "经典版产出审计结果")
check(multi is not None, "多智能体版产出审计结果")
check(len(multi.findings) > 0, "多智能体版产出 %d 条发现" % len(multi.findings))
check(classic_t < 3, "经典版 %.2fs（离线）" % classic_t)
check(multi_t < 3, "多智能体版 %.2fs（离线）" % multi_t)

print("2. 对撞区命中了真实冲突")
params = [f.param_key for f in multi.findings]
check("learning_rate" in params, "学习率冲突被查出: %s" % params)
check(any(f.status.value == "inconsistent" and f.param_key == "learning_rate"
          for f in multi.findings), "learning_rate 状态为 inconsistent")
check(any(f.param_key == "learning_rate" and f.risk.value == "high"
          for f in multi.findings), "learning_rate 风险为 high（关键参数）")

print("3. 角色隔离（上下文层面）")
from paper2lab.agent_roles import Analyst, CodeVerifier
job = make_job(classic)
pv = Analyst()(job)
cv = CodeVerifier()(job)
check("learning_rate" in pv, "解析员看到论文侧 learning_rate: %s" % list(pv.keys()))
check("learning_rate" in cv, "代码验证员看到代码侧 learning_rate: %s" % list(cv.keys()))
check(pv["learning_rate"]["value"] == "1e-4", "论文值: %s" % pv["learning_rate"]["value"])
check(cv["learning_rate"]["value"] == "0.001", "代码值: %s" % cv["learning_rate"]["value"])

print("4. 仲裁与红队")
ec = multi.execution_log
records = ec.get("records", []) if ec else []
keys = [r["key"] for r in records]
check("roles" in keys, "roles 节点执行: %s" % keys)
check("collider" in keys, "collider 节点执行: %s" % keys)
check("arbiter" in keys, "arbiter 节点执行: %s" % keys)
check("redteam" in keys, "redteam 节点执行: %s" % keys)
check("report_findings" in keys, "report_findings 节点执行: %s" % keys)
done = [r for r in records if r["status"] == "done"]
check(len(done) >= 12, "12+ 节点 done: %d" % len(done))
failed = [r for r in records if r["status"] == "failed"]
check(len(failed) == 0, "0 节点失败: %d" % len(failed))

print("5. 消融对比：多智能体版 vs 经典版")
c_map = {f.param_key: f for f in classic.findings}
m_map = {f.param_key: f for f in multi.findings}
check(len(c_map) == len(m_map), "条数一致: classic=%d multi=%d" % (len(c_map), len(m_map)))
both = set(c_map) & set(m_map)
check(len(both) == len(c_map) == len(m_map), "参数集一致: %d" % len(both))
mismatch = []
for k in both:
    if c_map[k].status.value != m_map[k].status.value:
        mismatch.append("%s: classic=%s multi=%s" % (k, c_map[k].status.value, m_map[k].status.value))
check(not mismatch, "逐条状态一致: %s" % (mismatch or "是"))
check(not mismatch, "逐条状态一致（消融结论：调度改效率、不改结论）")

print("6. 冲突清单与仲裁可序列化")
from paper2lab.collider import collide, arbitrate_all, red_team
from paper2lab.agent_roles import Analyst, CodeVerifier, NumericVerifier
job = make_job(classic)
pv = Analyst()(job)
cv = CodeVerifier()(job)
nv = NumericVerifier()(job)
collisions = collide(pv, cv, nv)
check(len(collisions) > 0, "对撞产出 %d 条冲突" % len(collisions))
arbs = arbitrate_all(collisions)
check(len(arbs) == len(collisions), "仲裁 %d 条" % len(arbs))
notes = red_team(collisions, pv, cv)
check(isinstance(notes, list), "红队产出 %d 条笔记" % len(notes))

print("7. 双路互证（align 基线 vs 对撞区独立结论）")
_ex = multi.execution_log or {}
_cc = _ex.get("crosscheck", [])
check(len(_cc) > 0, "产出核对表: %d 行" % len(_cc))
_agree = [x for x in _cc if x["verdict"] == "互证"]
_diff = [x for x in _cc if x["verdict"] == "分歧"]
check(len(_agree) >= 8, "互证 %d 条（两条独立路径得出相同结论）" % len(_agree))
# 分歧不是 bug —— 两条独立路径本就可能在个别参数上不一致，
# 关键是这样的分歧**必须被记录**，而不是被后跑的那条路径静默覆盖。
check(len(_diff) >= 1, "分歧 %d 条被如实记录: %s"
      % (len(_diff), [(x["param_key"], x["baseline"], x["collision"]) for x in _diff]))
check(all(x["verdict"] in ("互证", "分歧", "仅基线", "仅对撞") for x in _cc),
      "核对表取值合法")
# 并集口径: 对撞没有新增参数时，findings 必须等于基线（消融成立的前提）
_baseline_keys = {f.param_key for f in classic.findings}
_multi_keys = {f.param_key for f in multi.findings}
check(_baseline_keys <= _multi_keys, "findings 是基线 ∪ 对撞（不丢基线）: %d -> %d"
      % (len(_baseline_keys), len(_multi_keys)))
# 对撞新增的必须真的进去
_only_collide = [x["param_key"] for x in _cc if x["verdict"] == "仅对撞"]
check(all(k in _multi_keys for k in _only_collide),
      "对撞独有的 %d 个参数也进了 findings: %s" % (len(_only_collide), _only_collide))

print("8. 界面契约（execution_log 与 runner 同构）")
check("steps" in _ex, "execution_log 带 steps 键（界面进度条只认这个）")
check(_ex.get("mode") == "graph", "mode 标记为 graph: %s" % _ex.get("mode"))
check(isinstance(_ex.get("steps"), list) and len(_ex["steps"]) == len(
    [r for r in _ex.get("records", []) if r["key"] in {t.key for t in __import__(
        "paper2lab.toolbox", fromlist=["TOOLS"]).TOOLS}]), "steps 条数与工具节点数一致")
_step_keys = {s["key"] for s in _ex["steps"]}
check("parse_paper" in _step_keys and "report" in _step_keys, "工具节点进了 steps")
# 对撞区节点没有 STEPS 对应项，所以不进 steps，但必须留在 records 里
_rec_keys = {r["key"] for r in _ex.get("records", [])}
check({"roles", "collider", "arbiter", "redteam"} <= _rec_keys,
      "对撞区节点留在 records 里: %s" % sorted(
          {"roles", "collider", "arbiter", "redteam"} & _rec_keys))
check(isinstance(_ex.get("red_team"), list) and len(_ex["red_team"]) > 0,
      "红队笔记随 execution_log 一起交付: %d 条" % len(_ex.get("red_team") or []))

print("9. 闸门对撞区也生效（不是无闸门直通）")
_rec = {r["key"]: r for r in _ex.get("records", [])}
check(_rec.get("roles", {}).get("gate") in ("", "pass"),
      "roles 过闸门: %r" % _rec.get("roles", {}).get("gate"))
check(all(_rec.get(k, {}).get("gate") in ("", "pass", "blocked")
          for k in ("collider", "arbiter", "redteam", "report_findings")),
      "对撞区节点都带 gate 字段")

print("10. 断点/续跑（checkpoint）")
chk = graph._make_checkpointer()
g = graph.build_multagent_graph(checkpointer=chk, interrupt_before=["collider"])
state = build_state(DEMO / "paper.pdf", DEMO / "project.zip")
config = {"configurable": {"thread_id": "bp1"}}
paused = None
for ev in g.stream(state, config, stream_mode="values"):
    paused = ev
check(paused is not None, "图停在断点处")
check(paused.get("next_step") == "collider", "停在 collider: %s" % paused.get("next_step"))
# 续跑
g2 = graph.build_multagent_graph(checkpointer=chk)
final = state
for ev in g2.stream(state, config, stream_mode="values"):
    final = ev
check(final.get("next_step") == "__end__", "续跑完成: %s" % final.get("next_step"))
check(len(final.get("records", [])) >= 12, "续跑后记录数: %d" % len(final.get("records", [])))


print("=" * 50)
print("通过 %d/%d" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败 %d: %s" % (len(FAILURES), "；".join(FAILURES)))
    sys.exit(1)
print("全部通过")
