"""独立数据层全流程矩阵（不依赖既有测试脚本，直查产物）。

作为「专业测试员」的独立验证：
  ① 跑一次真实 demo 流程，逐模块数产物（有没有真的产出，不是状态写着 done）；
  ② 跑第二次，比结论签名 —— 同输入必须同输出；
  ③ 全程对 demo/*.zip 做哈希快照，证明审计不写用户文件；
  ④ 报告里四站/修正建议/化学/复现方案这些段落是否真的都在。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.pipeline import run_audit          # noqa: E402
from paper2lab.toolbox import STEPS, TOOLS        # noqa: E402

DEMO = ROOT / "demo"
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


def digest(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:16]


def run_once():
    t0 = time.monotonic()
    a, report = run_audit(
        str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
        user_results=json.loads((DEMO / "results.json").read_text(encoding="utf-8")),
    )
    return a, report, time.monotonic() - t0


print("=" * 72)
print("A 跑一次真实流程，逐模块数产物")
print("=" * 72)
zip_before = digest(DEMO / "project.zip")
pdf_before = digest(DEMO / "paper.pdf")
a, report, cost = run_once()
zip_after = digest(DEMO / "project.zip")
pdf_after = digest(DEMO / "paper.pdf")

check(a is not None, "run_audit 返回了 audit 对象")
check(bool(report) and len(report) > 5000, "报告正文够长（不是空壳）", f"{len(report)} 字")
check(cost < 300, "demo 单次全流程在 5 分钟内", f"{cost:.1f}s")

# ① 解析层
paper_params = getattr(getattr(a, "paper", None), "params", {}) or {}
code_params = getattr(getattr(a, "code", None), "params", {}) or {}
check(len(paper_params) >= 5, "论文侧解析出参数", f"{len(paper_params)} 个")
check(len(code_params) >= 5, "代码侧解析出参数", f"{len(code_params)} 个")

# ② 判定层
fs = getattr(a, "findings", []) or []
check(len(fs) >= 8, "参数比对产出条目", f"{len(fs)} 条")
statuses = {}
for f in fs:
    s = getattr(getattr(f, "status", None), "value", None) or str(getattr(f, "status", ""))
    statuses[s] = statuses.get(s, 0) + 1
check(sum(statuses.values()) == len(fs), "每条都有判定状态", str(statuses))
check(len(statuses) >= 2, "判定不是一边倒（确实分出了不同结论）", str(statuses))

# ③ 假设 / 实验
hyps = getattr(a, "hypotheses", []) or []
exps = getattr(a, "experiments", []) or []
check(len(hyps) >= 1, "假设站产出假设卡", f"{len(hyps)} 张")
check(len(exps) >= 1, "实验站产出实验卡（不是只写 done）", f"{len(exps)} 张")

# ④ 多智能体 / 化学 / 闭环
# 多智能体对撞的产物落在 compare_log（不是 multi_agent）
clog = getattr(a, "compare_log", None)
check(isinstance(clog, (list, dict)) and len(clog) > 0, "多智能体对撞区有产物",
      f"{len(clog)} 条" if hasattr(clog, "__len__") else "")
# 化学核验：有条目，或明确记为 0 条（不适用也算正常，但要能读到这个事实）
chem_n = getattr(a, "chem_total", None)
check(chem_n is not None, "化学核验区有产物计数（0 也算可读）", f"{chem_n} 项")
audit_d = {"steps": (getattr(a, "execution_log", {}) or {}).get("steps") or []}
check(getattr(a, "readiness", None) is not None, "就绪度字段在")
check(bool(getattr(a, "execution_log", None)), "执行记录在")

# ⑤ 报告段落
must = ("假设", "实验", "复现", "修正", "证据")
missing = [m for m in must if m not in report]
check(not missing, "报告含四站关键段落", f"缺: {missing}" if missing else "全在")
check("已通过判定" not in report, "报告不自我背书（无「已通过判定」）")

# ⑥ 不写用户文件
check(zip_before == zip_after, "代码包一字未改（哈希一致）", f"{zip_before}→{zip_after}")
check(pdf_before == pdf_after, "论文一字未改（哈希一致）", f"{pdf_before}→{pdf_after}")

# ⑦ 工序覆盖率
waked = [s for s in (audit_d.get("steps") or []) if (s.get("status") or "") == "done"]
check(len(waked) >= 12, "实际走完的工序数足够", f"{len(waked)} / {len(STEPS)} 格")
check(len(TOOLS) >= 14, "工具菜单规模没缩水", f"{len(TOOLS)} 件")

print()
print("=" * 72)
print("B 再跑一次，比结论是否一致")
print("=" * 72)
b, report2, cost2 = run_once()
sig = lambda x: [(getattr(f, "param_key", ""), str(getattr(getattr(f, "status", None), "value", "")))  # noqa: E731
                 for f in (getattr(x, "findings", []) or [])]
check(sig(a) == sig(b), "两次结论逐条一致（同输入同输出）",
      f"{len(sig(a))} vs {len(sig(b))}")
check(len(report) == len(report2) or abs(len(report) - len(report2)) < 200,
      "报告长度稳定（无随机段落）", f"{len(report)} vs {len(report2)}")
check(len(getattr(a, "hypotheses", []) or []) == len(getattr(b, "hypotheses", []) or []),
      "两次假设卡数量一致")
check(digest(DEMO / "project.zip") == zip_before, "第二轮也没写代码包")

print()
print(f"===== 汇总：{ok + bad} 项，通过 {ok}，未通过 {bad} =====")
sys.exit(1 if bad else 0)
