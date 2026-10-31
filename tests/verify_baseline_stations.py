"""四站主干底线守卫（离线，不联网、不调模型、不跑审计）。

守的是什么
----------
模型排计划会「砍步骤」：实测给 demo 只排了 5 件，把「假设生成」与
「实验设计与真跑」都判成不值得跑。于是四站里的 ① ② 恒为空 ——
界面上「提 → 做 → 判 → 改」只剩后两站，等于这条流水线断了一半。

2026-10-31 用户拍板：**保留模型排计划（决定权仍归它），但 ①② 不许空。**
实现口径不是收回决定权，而是画三条底线：
  · `_BASELINE_STATIONS` = 假设生成 / 实验设计与真跑 —— 模型判 skip 也执行，
    理由栏如实写「已按底线兜底执行」，模型原话保留；
  · `_NO_RESULT_WITHOUT` = align —— 老底线（跳了整轮没有任何结论），不动；
  · 依赖闭包 —— 底线工序 needs 的产物若没人产出（典型：模型把「论文解析」也跳了），
    就恢复「规则版本要跑、被模型跳过」的那个生产者，否则底线会在前置检查处被拦、
    等于白兜底一次。
**只补这两道工序的上游，不给 align 补**：align 的 needs 里有 code，若一并算进来，
模型跳「代码扫描」就会被强行拉回 —— 那是替用户多要了一样他并不要的东西。

这个文件盯七件事：
  A 常量：底线恰好是那两件，老底线 align 没被挤掉；
  B 提示词：模型**知情**（提前说清 ①② 跳了也会被执行，免得它白写理由）；
  C 行为：模型跳 ①② → 仍执行，理由写明底线，notes 点名；
  D 不越权：模型跳别的（读图 / 报表类）→ 照跳，没人改回来；
  E 依赖闭包：上游没人产出时补一个；有人产出时**不补**（不滥用）；
  F align 的 code 依赖**不**触发补齐；
  G 结构自洽：order == run 集合、每条决策来源合法、规则兜底版本身含 ①②。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.planner import (                                  # noqa: E402
    _BASELINE, _BASELINE_STATIONS, _NO_RESULT_WITHOUT, make_plan)
from paper2lab.toolbox import Job, TOOLS                         # noqa: E402

N = 0
BAD: list[str] = []


def check(ok, label: str) -> None:
    global N
    N += 1
    ok = bool(ok)
    print(("  [OK ] " if ok else "  [FAIL] ") + label)
    if not ok:
        BAD.append(label)


def dmap(plan) -> dict:
    return {d.tool: d for d in plan.decisions}


def job(*, code: bool = True):
    return Job(pdf_path=str(ROOT / "demo" / "paper.pdf"),
               code_path=str(ROOT / "demo" / "project.zip") if code else "",
               api_key="", force_mock=True,
               vision_kwargs={"api_key": "fake-for-planning"})


J = job()

print("A 常量层：底线是哪几件")
check(set(_BASELINE_STATIONS) == {"hypothesize", "design_experiment"},
      "四站主干底线恰好是「假设生成」与「实验设计与真跑」两件: %s"
      % sorted(_BASELINE_STATIONS))
check(set(_NO_RESULT_WITHOUT) == {"align"}, "老底线 align 仍在（防空转不许被挤掉）")
check({"hypothesize", "design_experiment", "align"} <= set(_BASELINE),
      "合并后的底线表含三件: %s" % sorted(_BASELINE))
check(all(k in {t.key for t in TOOLS} for k in _BASELINE),
      "底线里的工具键都真实存在于工具箱（写错键名 = 底线静默失效）")

print("")
print("B 提示词：模型必须知情")
_src = (ROOT / "paper2lab" / "planner.py").read_text(encoding="utf-8")
check("唯一的例外" in _src and "即使判 skip 也会被执行" in _src,
      "prompt 提前说清「①② 跳了也会被执行」（否则模型白写理由、也谈不上诚实）")
check("四站主干" in _src, "prompt 里用「四站主干」说清为什么，而不是只说「不行」")

print("")
print("C 行为层：模型跳 ①②，仍然执行")
C1 = {"decisions": [{"tool": "hypothesize", "action": "skip", "reason": "论文已给全参数，不必追问"},
                    {"tool": "design_experiment", "action": "skip", "reason": "复算成本高"}],
      "notes": []}
p1 = make_plan(J, use_llm=True, proposal=C1)
d1 = dmap(p1)
check(p1.planner == "llm", "仍走模型排计划通道（没有整体回退成规则版）")
check(d1["hypothesize"].action == "run", "模型判 skip「假设生成」-> 仍执行（① 不许空）")
check("底线" in d1["hypothesize"].reason, "理由栏写明这是底线兜底: %s"
      % d1["hypothesize"].reason[:40])
check("跳过" in d1["hypothesize"].reason, "同时保留「模型原本判跳过」这一事实（可倒查）")
check(d1["hypothesize"].source == "llm",
      "来源仍记 llm（不是假装程序自己决定的）")
check(d1["design_experiment"].action == "run", "模型判 skip「实验设计与真跑」-> 仍执行（② 不许空）")
check("底线" in d1["design_experiment"].reason, "② 的理由同样写明底线")
check("design_experiment" in p1.order and "hypothesize" in p1.order,
      "两件都真的进了执行序列: %s" % (p1.order[:6],))
check(any("四站主干" in n for n in p1.notes), "notes 点名「四站主干按底线执行」（事后可倒查）")
check(any("假设生成" in n for n in p1.notes), "notes 里写清是哪两件被兜底")

C2 = {"decisions": [{"tool": "hypothesize", "action": "skip", "reason": ""}], "notes": []}
p2 = make_plan(J, use_llm=True, proposal=C2)
d2 = dmap(p2)
check(d2["hypothesize"].action == "run" and "底线" in d2["hypothesize"].reason,
      "模型跳了还不给理由 -> 照样兜底执行，且不代它编理由")

print("")
print("D 不越权：其它工具照模型说的办")
C3 = {"decisions": [{"tool": "read_figures", "action": "skip", "reason": "图注没信息量"},
                    {"tool": "selfcheck", "action": "skip", "reason": "自检没必要"},
                    {"tool": "rootcause", "action": "skip", "reason": "不必追根因"}],
      "notes": []}
p3 = make_plan(J, use_llm=True, proposal=C3)
d3 = dmap(p3)
check(d3["read_figures"].action == "skip" and d3["read_figures"].source == "llm",
      "读图不是底线 -> 模型说跳就跳（没被顺手兜底）")
check(d3["selfcheck"].action == "skip", "审计自检不是底线 -> 照跳")
check(d3["rootcause"].action == "skip", "根因追查不是底线 -> 照跳")
check("read_figures" not in p3.order, "跳掉的工具不在执行序列里")

print("")
print("E 依赖闭包：上游没人产出才补")
# E1 模型跳「论文解析」但不跳补漏 —— paper 仍由补漏产出，不该补回解析
E1 = {"decisions": [{"tool": "parse_paper", "action": "skip", "reason": "省时间"},
                    {"tool": "fill_missing", "action": "run", "reason": "让 Agent 补"}],
      "notes": []}
p4 = make_plan(J, use_llm=True, proposal=E1)
d4 = dmap(p4)
check(d4["parse_paper"].action == "skip",
      "paper 另有产出者（补漏在执行）-> 不把论文解析硬拉回来（不滥用补齐）")
check(d4["hypothesize"].action in ("run",), "假设生成照常执行")

# E2 论文解析与补漏**都**跳 —— 底线工序会缺 paper，必须补一个生产者
E2 = {"decisions": [{"tool": "parse_paper", "action": "skip", "reason": "省时间"},
                    {"tool": "fill_missing", "action": "skip", "reason": "也不用补"}],
      "notes": []}
p5 = make_plan(J, use_llm=True, proposal=E2)
d5 = dmap(p5)
_restored = [k for k in ("parse_paper", "fill_missing")
             if d5[k].action == "run" and "底线依赖补齐" in d5[k].reason]
check(len(_restored) >= 1,
      "两个 paper 产出者都被跳 -> 至少恢复一个（否则假设生成在 gate 处被拦，白兜底）: %s"
      % (_restored,))
check(any("恢复了被模型跳过的上游" in n for n in p5.notes),
      "notes 如实写明「为让底线跑得动而恢复了上游」")
_paper_makers = [k for k in ("parse_paper", "fill_missing") if d5[k].action == "run"]
check(bool(_paper_makers), "paper 产物在执行集合里确实有了产出者: %s" % (_paper_makers,))

print("")
print("F align 的 code 依赖不触发补齐")
F1 = {"decisions": [{"tool": "analyze_code", "action": "skip", "reason": "代码太乱不想扫"},
                    {"tool": "align", "action": "skip", "reason": "跳过对齐"}],
      "notes": []}
p6 = make_plan(J, use_llm=True, proposal=F1)
d6 = dmap(p6)
check(d6["align"].action == "run", "align 仍按老底线兜底为执行（跳了整轮没结论）")
check(d6["analyze_code"].action == "skip",
      "代码扫描没被 align 的 needs 拉回来（那是替用户多要一样他并不要的东西）")
check("analyze_code" not in p6.order, "因此执行序列里也没有代码扫描")

print("")
print("G 结构自洽")
ALL_SKIP = {"decisions": [{"tool": t.key, "action": "skip", "reason": "全跳"}
                          for t in TOOLS], "notes": []}
p7 = make_plan(J, use_llm=True, proposal=ALL_SKIP)
d7 = dmap(p7)
check(len(p7.decisions) == len(TOOLS), "每个工具都有一条决策（含模型没表态的默认项）")
check(len(d7) == len(TOOLS), "决策表无重复键")
check(set(p7.order) == {x.tool for x in p7.decisions if x.action == "run"},
      "执行序列 == 判 run 的集合（不多不少）")
check(list(p7.order) == [x.tool for x in p7.decisions if x.action == "run"],
      "执行序列顺序与决策表一致（不许重排）")
check(all(x.source in ("llm", "rule") for x in p7.decisions), "每条决策的来源只有 llm / rule 两种")
check(all(x.action in ("run", "skip") for x in p7.decisions), "动作只有 run / skip 两种")
check(d7["align"].action == "run" and d7["hypothesize"].action == "run"
      and d7["design_experiment"].action == "run",
      "全跳的极端提案下，三件底线仍是执行（空转与缺站都被堵住）")
check(sum(1 for x in p7.decisions if x.action == "skip") >= 5,
      "其余工具仍然照模型跳（兜底没蔓延成「全都跑」）: 仍跳过 %d 件"
      % sum(1 for x in p7.decisions if x.action == "skip"))

print("")
print("H 规则兜底版（不调模型）本身含 ①②")
p8 = make_plan(J, use_llm=False)
d8 = dmap(p8)
check(p8.planner == "rule", "use_llm=False 得到规则版")
check(d8["hypothesize"].action == "run" and d8["design_experiment"].action == "run",
      "规则版本来就会跑 ①②（底线只是不让模型把它们砍掉，不是新增步骤）")
check("hypothesize" in p8.order and "design_experiment" in p8.order, "规则版执行序列含 ①②")

print("")
if N == 0:
    print("没有跑到任何一项 —— 视为失败")
    sys.exit(1)
_o = N - len(BAD)
print("=" * 74)
print(f"共 {N} 项：通过 {_o}，未通过 {len(BAD)}")
if BAD:
    for b in BAD:
        print("  -", b)
    sys.exit(1)
print("四站主干底线成立：① 假设 / ② 实验不会被模型砍掉，其余决定权仍归模型")
