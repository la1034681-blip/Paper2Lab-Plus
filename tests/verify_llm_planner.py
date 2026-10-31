"""模型参与决策的两条通道（阶段一 1C 起：**调用权在模型，程序不设闸门**）。

用法: .venv\\Scripts\\python.exe verify_llm_planner.py        （不调 API，全部用注入的提案）

覆盖:
  ① 模型排计划 —— 注入一份"乱来"的提案，程序**照单全收**（核心步骤也照跳），
     只做三件不参与决策的事: 递菜单 / 原样留痕 / 如实提示（跳了核心步骤、
     没给理由、上游没跑都会被点名）；清单外工具不采纳（本次不存在）。
  ② AI 提名补丁目标 —— 这条**不归 1C 管**，仍是"值可信、缺的只是选哪个"时启用，
     且必须过补丁自己的机械闸门（盖章在程序）。它管的是"补丁内容"，不是"调不调工具"。
  ③ 模型不可用 / 输出不合法 → 回落规则版并**如实标注**，不许悄悄当成模型结果。

为什么用注入而不是真调 API: 要断言的是**程序这一侧的行为**（确定性），
不是模型今天答得怎么样。真调 API 的波动会掩盖"交权后程序到底还动不动手"。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from paper2lab.planner import make_plan          # noqa: E402
from paper2lab.toolbox import TOOLS, Job         # noqa: E402

OLD = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
fails: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok:
        fails.append(label)


def dmap(plan) -> dict:
    return {d.tool: d for d in plan.decisions}


print("① 模型排计划：决定权在模型（只兜四站主干①②与结论产出，其余照它）")
# 这份 Job 刻意配上一个假读图 key: 否则规则版本来就会跳过读图（无读图通道），
# 读图压根不进菜单，就测不出"模型的成本判断被采纳"这一条了。
JOB = Job(pdf_path=str(OLD / "01_DDPM/paper.pdf"), code_path=str(OLD / "01_DDPM/code.zip"),
          api_key="", force_mock=True, vision_kwargs={"api_key": "fake-for-planning"})

# 一份"乱来"的提案：跳核心步骤、跳代码扫描（但有代码包）、跳根因（补丁却没跳）、跳归因
BOGUS = {"decisions": [
    {"tool": "parse_paper", "action": "skip", "reason": "省点时间"},
    {"tool": "analyze_code", "action": "skip", "reason": "不想扫"},
    {"tool": "llm_analyze", "action": "skip", "reason": "解释文字可以省掉"},
    {"tool": "rootcause", "action": "skip", "reason": "结论够清楚了"},
    {"tool": "patch", "action": "run", "reason": ""},
], "notes": ["注入用例"]}
p = make_plan(JOB, use_llm=True, proposal=BOGUS)
d = dmap(p)
print(f"     排计划方式={p.planner} 执行={p.to_dict()['ran']} 跳过={p.to_dict()['skipped']}")
for x in p.decisions:
    print(f"       {x.action:4} {x.source:5} {x.tool:14} {x.reason[:52]}")
check(p.planner == "llm", "确认走的是模型排计划通道")
check(d["parse_paper"].action == "skip" and d["parse_paper"].source == "llm",
      "模型想跳论文解析 -> **照跳**（核心步骤也不例外，1C 起不再有程序闸门）")
check(d["analyze_code"].action == "skip" and d["analyze_code"].source == "llm",
      "模型想跳代码扫描 -> 照跳（有代码包也不例外）")
check(d["llm_analyze"].action == "skip" and d["llm_analyze"].source == "llm",
      "模型跳归因 -> 采纳（这是允许它决定的成本项）")
check(d["rootcause"].action == "skip" and d["rootcause"].source == "llm",
      "模型跳根因 -> 照跳（旧版这里会被依赖级联否决，1C 起不再否决）")
check(d["patch"].action == "run" and "未给理由" in d["patch"].reason,
      "模型说跑却没说理由 -> 照跑，理由栏如实写「未给理由」（不代它编）")
check("parse_paper" not in p.order and "analyze_code" not in p.order,
      "被跳过的工具确实不在执行序列里: %s" % (p.order,))
check(any("模型跳过了核心步骤" in n for n in p.notes),
      "跳核心步骤被点名（如实提示，可倒查）")
check(any("前置检查拦下" in n for n in p.notes),
      "「根因没跑、补丁却要跑」被如实提示（不偷偷改决策）")
check(not any("程序否决" in n for n in p.notes), "不再有「程序否决」这类措辞")
check(any("决定权在模型" in n for n in p.notes), "排计划方式里写明「决定权在模型」")
# 2026-10-31 用户定版：程序**只兜两样**（四站主干①② ＋ 结论产出），其余照模型。
# 这句留痕必须说清兜底范围，否则事后倒查会以为程序又在四处否决。
check(any("只兜四站主干" in n for n in p.notes),
      "并写明程序只兜「四站主干①②与结论产出」这一小块（不越权）")

# 产物"多生产者"的坑：`fill_missing` 与 `parse_paper` 都产出 paper。
# 模型跳过补漏时，**读图不该被误判成"上游缺失"**（paper 仍在）。
SHARED = {"decisions": [{"tool": "fill_missing", "action": "skip",
                         "reason": "内核已抽全，补漏收益低"}], "notes": []}
p_shared = make_plan(JOB, use_llm=True, proposal=SHARED)
d_shared = dmap(p_shared)
check(d_shared["fill_missing"].action == "skip" and d_shared["fill_missing"].source == "llm",
      "模型跳过「Agent 补漏」-> 采纳")
check(d_shared["analyze_images"].action == "run" and d_shared["analyze_images"].source == "rule",
      "补漏被跳过**不会**让读图跟着跳（paper 由论文解析产出）；读图模型没表态，"
      "如实记为 rule 而非 llm")

# 合理提案：模型认为"论文太短、读图没收益" —— 采纳
OK_PROP = {"decisions": [{"tool": "analyze_images", "action": "skip",
                          "reason": "论文 8 页且正文已给全参数，读图收益低"}],
           "notes": []}
p2 = make_plan(JOB, use_llm=True, proposal=OK_PROP)
d2 = dmap(p2)
check(d2["analyze_images"].source == "llm" and d2["analyze_images"].action == "skip",
      "模型的成本判断（短论文不值得读图）-> 采纳")

# 模型想跳过却**没给理由**：照跳（决定权在它），但理由栏必须如实写"未给理由"
# —— 既不代它编理由，也不假装它没提。
NO_REASON = {"decisions": [{"tool": "analyze_images", "action": "skip", "reason": ""}],
             "notes": []}
p_nr = make_plan(JOB, use_llm=True, proposal=NO_REASON)
d_nr = dmap(p_nr)
check(d_nr["analyze_images"].action == "skip" and d_nr["analyze_images"].source == "llm",
      "模型想跳过但没给理由 -> 照跳（程序不替它做决定）")
check("未给理由" in d_nr["analyze_images"].reason and "未给理由" in "".join(p_nr.notes),
      "决策表与 notes 都措辞准确：它「提了但没理由」")
check(any("清单外工具" in n for n in p_nr.notes) is False
      and d_nr["analyze_images"].action == "skip",
      "没提清单外工具时不误报")

# 清单外工具（材料里不存在）: 模型提了也不采纳，如实记下来
OUTSIDE = {"decisions": [{"tool": "quantum_solver", "action": "run", "reason": "试试"},
                         {"tool": "analyze_images", "action": "run", "reason": "看下图"}],
           "notes": []}
p_out = make_plan(JOB, use_llm=True, proposal=OUTSIDE)
check(any("清单外工具" in n for n in p_out.notes) and "quantum_solver" in "".join(p_out.notes),
      "清单外工具不采纳，但如实记进 notes（本次不存在，不是被否决）")
check("quantum_solver" not in p_out.order, "清单外工具不会进执行序列")

print("")
print("② 无模型：必须回退确定性路径，并如实说明")
p3 = make_plan(JOB, use_llm=True)      # 无 key + force_mock
check(any("模型排计划未生效" in n for n in p3.notes), "排计划回退规则版，且写明原因")
check(p3.planner == "rule", "回落时如实标 planner=rule（不假装是模型排的）")
# 回退必须是**逐条等价**，不只是"看起来像规则版"
p4 = make_plan(JOB, use_llm=False)
_sig = lambda pl: [(d.tool, d.action, d.source) for d in pl.decisions]   # noqa: E731
check(_sig(p3) == _sig(p4), "回退后的计划与规则版**逐条一致**（工具/动作/来源都相同）")
check(p4.planner == "rule", "默认（use_llm=False）就是规则版，行为与从前一致")

print("")
print("③ AI 提名补丁目标（不归 1C 管：盖章在程序，管的是「补丁内容」不是「调不调工具」）")
from paper2lab.pipeline import run_audit          # noqa: E402
from paper2lab.patch import propose_patches       # noqa: E402
from paper2lab.rootcause import trace_findings    # noqa: E402

DETR_PDF = OLD / "02_DETR/paper.pdf"
DETR_ZIP = OLD / "02_DETR/code.zip"
_detr_res = OLD / "02_DETR/results.json"
audit, _ = run_audit(str(DETR_PDF), str(DETR_ZIP),
                     (json.loads(_detr_res.read_text(encoding="utf-8"))
                      if _detr_res.exists() else None),
                     force_mock=True)
traces = trace_findings(audit, max_items=8)
rule_only = propose_patches(audit, traces, str(DETR_ZIP), max_items=8)
print(f"     确定性通道：给方案 {sum(1 for x in rule_only if x.is_proposed)} / "
      f"拒绝 {sum(1 for x in rule_only if not x.is_proposed)}；"
      f"拒绝类型={[x.refuse_kind for x in rule_only if not x.is_proposed]}")
check(all(x.status == "proposed" for x in rule_only if x.is_proposed),
      "确定性通道给方案的那几条，status 仍是 proposed（没被 AI 通道改写）")

elig = next((x.param_key for x in rule_only
             if not x.is_proposed and x.refuse_kind in ("multi_value", "no_rootcause")), None)
if elig is None:
    print("     （本例没有可交给 AI 复判的拒绝类型，跳过 ③）")
else:
    print(f"     可交给 AI 复判的参数：{elig}（拒绝类型={[x.refuse_kind for x in rule_only if x.param_key == elig][0]}）")
    t_obj = next((t for t in traces if t.param_key == elig), None)
    pans = list((getattr(audit.paper, "params_all", {}) or {}).get(elig) or [])
    c0 = (t_obj.candidates or [None])[0] if t_obj else None

    # (a) 越界提名：报一个不存在的论文值 -> 盖章不过
    forged = {elig: {"ok": True, "paper_value": "99999",
                     "code_loc": (f"{c0.file}:{c0.line}" if c0 else "main.py:1"),
                     "why": "测试：越界提名"}}
    got = next(x for x in propose_patches(audit, traces, str(DETR_ZIP), max_items=8,
                                          proposals=forged) if x.param_key == elig)
    print(f"     (a) 越界提名 -> status={got.status} kind={got.refuse_kind}｜{got.refuse_reason[:70]}")
    check(not got.is_proposed, "(a) 提名的值不在已抽候选里 -> 盖章不过，仍不给补丁")

    # (b) 合法提名：值取自论文候选、位置取自根因候选点 -> 应产出 proposed-llm（仍需人工确认）
    if pans and c0 is not None:
        good = {elig: {"ok": True, "paper_value": str(pans[0].raw_value),
                       "code_loc": f"{c0.file}:{c0.line}",
                       "why": "测试：合法提名（取自候选）"}}
        g2 = next(x for x in propose_patches(audit, traces, str(DETR_ZIP), max_items=8,
                                             proposals=good) if x.param_key == elig)
        print(f"     (b) 合法提名 -> status={g2.status} target={g2.target} stamps={len(g2.stamps)}")
        check(g2.source == "llm", "(b) 确实走了 AI 提名通道")
        if g2.is_proposed:
            check(g2.status == "proposed-llm" and g2.needs_human,
                  "(b) 产出标为 proposed-llm 且 needs_human（采纳前必须人工确认）")
            check(bool(g2.stamps) and "没有" in g2.rationale,
                  "(b) 盖章清单非空，且理由里写明「程序没有验证它属于哪套配置」")
        else:
            check("机械闸门" in g2.refuse_reason,
                  "(b) 若被拒，必须是**盖章清单里某一条**不过（而非原判定）")
    else:
        print("     （缺候选，跳过 (b) 合法提名用例）")

print("")
# ---- 底线兜底: 模型若要跳过「唯一产出结论的那一步」, 这一轮就变成空转。
#      实测过这个真实故障: 只传论文 + 模型排计划 → 模型跳 align → job.audit 为 None
#      → run_audit 返回 (None, None) → 界面 `audit.readiness` 抛 AttributeError
#      → Streamlit 整页往下不渲染 → 用户看到"点开始一直转、结果不出来"。
#      回归断言: 即使模型明确说 skip align, 最终计划里它也必须是 run;
#      且兜底**只保这一件**, 其余步骤说跳仍照跳（否则等于把决定权又收回程序）。
print("=== 底线兜底: 不许把这一轮变成空转 ===")
_JOB_NOCODE = Job(pdf_path=str(OLD / "01_DDPM/paper.pdf"), code_path="",
                  api_key="", force_mock=True)
_SKIP_ALL = {"decisions": [{"tool": t.key, "action": "skip", "reason": "测试: 我全跳"}
                           for t in TOOLS], "notes": []}
_pf = make_plan(_JOB_NOCODE, use_llm=True, proposal=_SKIP_ALL)
_df = dmap(_pf)
print(f"     排计划方式={_pf.planner} 执行={_pf.to_dict()['ran']} 跳过={_pf.to_dict()['skipped']}")
check(_df.get("align") is not None and _df["align"].action == "run",
      "模型要跳 align -> 按底线兜底为执行（它一被跳整轮就没有任何结论）: %s"
      % (_df["align"].action if _df.get("align") else "缺失"))
check("底线兜底" in (_df["align"].reason or ""),
      "兜底执行的理由栏写明这是底线、且如实留痕: %s" % (_df["align"].reason or "")[:44])
check(_df["align"].source == "llm",
      "兜底仍如实标 source=llm（不静默改写成规则版的决定）")
check("align" in (_pf.order or ()),
      "最终执行序列里含 align —— 否则 run_audit 会返回 (None, None)")
_still_skipped = [x for x, dd in _df.items() if dd.action == "skip"]
check(len(_still_skipped) >= 1,
      f"兜底只保「结论产出 ＋ 四站主干①②」三件, 其余照模型: 仍跳过 {len(_still_skipped)} 个")
# 2026-10-31 用户定版：①② 也进底线（证据：实测 demo 只排 5 件、四站前两站恒空）
check(_df["hypothesize"].action == "run" and "底线" in _df["hypothesize"].reason,
      "模型要跳「假设生成」-> 底线兜底为执行（四站主干①不许空）")
check(_df["design_experiment"].action == "run" and "底线" in _df["design_experiment"].reason,
      "模型要跳「实验设计与真跑」-> 底线兜底为执行（四站主干②不许空）")

print("")
if fails:
    print(f"未通过 {len(fails)} 项：")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("全部通过（1C: 调不调工具由模型定、程序不设闸门；补丁内容仍由程序盖章）")
