"""阶段一 1B 验证 —— 三角色转正为「七位置」编制，且结论一字不改。

守的契约:
  A 编制表: 七个位置、出场序固定、键唯一、字段齐全、承载模块真实存在
  B 隔离没松: 解析员两只眼睛互不通气（论文视野看不到代码侧，反之亦然）
  C 行为不变: Parser ≡ Analyst+CodeVerifier、Calculator ≡ NumericVerifier
  D 向后兼容: 老三角色仍在（既有测试 / 消融脚本还在直接 import 它们）
  E 开关仍在: role_llm 关着时 role_llm_note 为空（"模型没参与"也要如实说）

为什么要有这一项: 1B 是"换名分"，最怕的就是换着换着把隔离换掉了 ——
那是本项目的生死线（自审自）。所以隔离必须由**可执行的断言**盯着。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.agent_roles import (Analyst, Calculator, CodeVerifier, NumericVerifier,
                                   POSITION_BY_KEY, POSITIONS, Parser, ROLE_BY_NAME)
from paper2lab.models import CodeInfo, Evidence, PaperInfo

CHECKS = 0
FAILURES = []


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


def _ev(value, loc, snippet, source):
    return Evidence(raw_value=value, location=loc, snippet=snippet, source=source)


def _job():
    """一份两侧都有材料的 demo 式样品: 论文侧与代码侧各有对方没有的参数。"""
    paper, code = PaperInfo(), CodeInfo()
    paper.params = {
        "learning_rate": _ev("1e-4", "p3", "learning rate 1e-4", "paper"),
        "scheduler": _ev("cosine", "p4", "采用余弦退火调度", "paper"),
    }
    paper.image_params = {"warmup": [_ev("1000", "fig2", "图 2 横轴起点", "paper")]}
    code.params = {
        "learning_rate": [_ev("1e-3", "train.py:12", "LR = 1e-3", "code")],
        "seed": [_ev("42", "train.py:3", "SEED = 42", "code")],
    }
    code.aux_only_keys = set()
    return types.SimpleNamespace(paper=paper, code=code)


print("1. 编制表: 七位置 · 出场序 · 键唯一")
check(len(POSITIONS) == 7, "位置数 = 7（实得 %d）" % len(POSITIONS))
_order = [p.cn for p in POSITIONS]
_want = ["解析员", "检索员", "主持人", "实验员", "计算员", "审查员", "撰写员"]
check(_order == _want, "出场序 = 解析·检索·主持·实验·计算·审查·撰写（实得 %s）" % _order)
check(len(POSITION_BY_KEY) == 7, "位置键唯一")
_caps = [p.capability for p in POSITIONS]
check(sorted(_caps) == sorted(["读", "搜", "编排", "做", "算", "审", "写"]),
      "五能力全覆盖（含编排与审）: %s" % _caps)

print("2. 每个位置字段齐全 + 交接物驱动")
check(all(p.key and p.cn and p.capability and p.station and p.handoff
          and p.carrier and p.impl for p in POSITIONS), "七行字段均非空")
check(all(p.impl in ("复用", "待改", "待建") or p.impl.startswith(("复用", "待改", "待建"))
          for p in POSITIONS), "现状字段取值受控: %s" % sorted({p.impl for p in POSITIONS}))
check(sum(1 for p in POSITIONS if p.impl == "复用") >= 4,
      "1B 落地后多数位置已是「复用」（不写新算法）: %d 个"
      % sum(1 for p in POSITIONS if p.impl == "复用"))
check(all(p.handoff for p in POSITIONS), "每个位置都有交接物（协作 = 交接物驱动）")

print("3. 承载模块真实存在（不是纸上画饼）")
_carriers = {
    "retrieval.py": ROOT / "paper2lab" / "retrieval.py",
    "planner.py": ROOT / "paper2lab" / "planner.py",
    "repro_plan.py": ROOT / "paper2lab" / "repro_plan.py",
    "repro.py": ROOT / "paper2lab" / "repro.py",
    "compute.py": ROOT / "paper2lab" / "compute.py",
    "collider.py": ROOT / "paper2lab" / "collider.py",
    "aligner.py": ROOT / "paper2lab" / "aligner.py",
    "selfcheck.py": ROOT / "paper2lab" / "selfcheck.py",
    "report.py": ROOT / "paper2lab" / "report.py",
    "patch.py": ROOT / "paper2lab" / "patch.py",
}
_missing = [n for n, p in _carriers.items() if not p.is_file()]
check(not _missing, "七位置点到的承载模块都在: 缺 %s" % _missing)
_arsrc = (ROOT / "paper2lab" / "agent_roles.py").read_text(encoding="utf-8")
check("class Parser" in _arsrc and "class Calculator" in _arsrc,
      "解析员 / 计算员已落到新结构（不只是登记）")

print("4. 隔离没松: 解析员两只眼睛互不通气（生死线）")
job = _job()
p = Parser()
views = p(job)
check(set(views) == {"paper", "code"}, "解析员产出两张视图: %s" % sorted(views))
check("seed" not in views["paper"], "论文视野看不到代码侧独有的 seed")
check("scheduler" not in views["code"], "代码视野看不到论文侧独有的 scheduler")
check("warmup" not in views["code"], "论文图像通道的值也不泄到代码视野")
check("learning_rate" in views["paper"] and "learning_rate" in views["code"],
      "两侧共有的参数各自都能拿到（隔离 ≠ 丢数据）")
check(getattr(job, "paper_view", None) == views["paper"]
      and getattr(job, "code_view", None) == views["code"],
      "解析员顺手把两侧视图挂回 job（计算员的输入口径不变）")

print("5. 行为一字不改: Parser ≡ Analyst+CodeVerifier，Calculator ≡ NumericVerifier")
job2 = _job()
_old = (Analyst()(job2), CodeVerifier()(job2))
job3 = _job()
_new = Parser()(job3)
check(_new["paper"] == _old[0], "论文视野逐字段等于原 Analyst")
check(_new["code"] == _old[1], "代码视野逐字段等于原 CodeVerifier")
job2.paper_view, job2.code_view = _old
job3.paper_view, job3.code_view = _new["paper"], _new["code"]
check(Calculator()(job3) == NumericVerifier()(job2), "计算员逐字段等于原 NumericVerifier")

print("6. 向后兼容: 老三角色保留（既有测试与消融脚本仍在直接 import）")
check(set(ROLE_BY_NAME) == {"analyst", "verifier", "numeric"},
      "ROLE_BY_NAME 仍是三个老键: %s" % sorted(ROLE_BY_NAME))
check(Parser.name == "parser" and Calculator.name == "calculator",
      "新执行体的 name 与编制表键一致")

print("7. 留痕口径: 模型没参与也要如实说（不能沉默）")
from paper2lab.agent_roles import RoleLLM     # noqa: E402

p2 = Parser()
p2(_job())
check(p2.last_note == "", "role_llm 关着时留痕为空串（与阶段四同口径）")
# 用测试替身顶掉模型调用（不碰网络）：两只眼睛都看过之后，留痕必须合并且不丢半边
import paper2lab.llm as _llmmod     # noqa: E402

_orig_ask = _llmmod.ask_json
_llmmod.ask_json = lambda *a, **k: ({"normalized": {}}, "AI（测试替身）")
try:
    p3 = Parser(RoleLLM(api_key="fake-key"))
    p3(_job())
finally:
    _llmmod.ask_json = _orig_ask
check("角色 analyst" in p3.last_note and "角色 verifier" in p3.last_note,
      "两只眼睛各自的留痕都并进解析员一条话里（不丢半边）: %r" % p3.last_note)


print("8. node_roles 已改走新结构（换编制必须换到执行路径上，不做死代码）")
_gsrc = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")
check("from .agent_roles import Calculator, Parser, RoleLLM" in _gsrc,
      "graph.py 从 agent_roles 导入的是 Parser / Calculator")
check("_parser = Parser(_llm)" in _gsrc and "Calculator()(job)" in _gsrc,
      "node_roles 实际调用解析员与计算员")
check("_note = _parser.last_note" in _gsrc, "留痕改从解析员取（两只眼睛合并成一条）")

print("9. 1C 调用权翻转: 程序不设闸门，菜单按材料裁剪，留痕可倒查")
from paper2lab.planner import make_plan          # noqa: E402
from paper2lab.science_tools import CATALOG, SCIENCE_TOOLS, catalog_keys, menu  # noqa: E402
from paper2lab.toolbox import TOOLS, Job         # noqa: E402

check(len(CATALOG) == len(TOOLS) and catalog_keys() == {t.key for t in TOOLS},
      "工具菜单与执行体同源同数（菜单不会撒谎）: %d 条" % len(CATALOG))
check(len(menu()) == len(CATALOG) and all("key" in m and "成本" in m for m in menu()),
      "菜单是可交给模型的 JSON 友好形状（含成本，供它判断值不值得跑）")
check(len(SCIENCE_TOOLS) == 4 and {c.key for c in SCIENCE_TOOLS}
      == {t.key for t in SCIENCE_TOOLS},
      f"3A 的具名科学计算工具已登记（{len(SCIENCE_TOOLS)} 件, 菜单与执行体同源）")
check(all(c.station == "② 实验" and c.kind == "science" for c in SCIENCE_TOOLS),
      "科学工具归 ② 实验站且标 kind=science（③ 只判定、不算数 —— 2C 的口径）")
check(not ({c.key for c in CATALOG} & {c.key for c in SCIENCE_TOOLS}),
      "审计工序与科学工具不重名（重名会让模型挑到哪件说不清）")
check(all(c.station for c in CATALOG), "菜单每一条都标了归属站（四站口径）")

_PLAN_JOB = Job(pdf_path=str(ROOT / "demo" / "paper.pdf"),
                code_path=str(ROOT / "demo" / "project.zip"),
                api_key="", force_mock=True)
_rule = make_plan(_PLAN_JOB, use_llm=False)
check(_rule.planner == "rule", "use_llm=False 时是规则兜底版计划")
_MENU_KEYS = {d.tool for d in _rule.decisions if d.action == "run"}
_prop = {"decisions": [{"tool": "parse_paper", "action": "skip", "reason": "省时间"},
                       {"tool": "selfcheck", "action": "skip", "reason": ""},
                       {"tool": "no_such_tool", "action": "run", "reason": "瞎编一个"}],
         "notes": []}
_plan = make_plan(_PLAN_JOB, use_llm=True, proposal=_prop)
_dm = {d.tool: d for d in _plan.decisions}
check(_plan.planner == "llm", "use_llm=True 时走模型排计划通道")
check(_dm["parse_paper"].action == "skip" and _dm["parse_paper"].source == "llm",
      "模型想跳核心步骤 -> 照跳（1C 起没有程序闸门）")
check(_dm["selfcheck"].action == "skip" and "未给理由" in _dm["selfcheck"].reason,
      "模型没给理由也照跳，理由栏如实写「未给理由」")
check("no_such_tool" not in _plan.order,
      "清单外工具不采纳（本次不存在）——边界靠「没有那个东西」而非「门」")
check(any("决定权在模型" in n for n in _plan.notes),
      "留痕写明「决定权在模型」（程序只兜四站主干①②与结论产出），并记下模型跳了核心步骤")
check(all(d.action in ("run", "skip") for d in _plan.decisions)
      and len(_plan.decisions) == len(TOOLS), "每个工具都有决策（含模型没表态的默认项）")
_gsrc2 = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")
check('state["plan_dict"] = plan.to_dict()' in _gsrc2
      and '"plan": dict(state.get("plan_dict") or {})' in _gsrc2,
      "排计划留痕跟着走（多智能体路径原先把它丢了，现与 runner 同口径）")
check('"plan": {}' not in _gsrc2,
      "旧的写死 plan={} 已清掉（否则界面的「模型排计划」标签永远不亮）")


print("=" * 50)
print("通过 %d/%d" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败 %d: %s" % (len(FAILURES), "；".join(FAILURES)))
    sys.exit(1)
print("全部通过")
