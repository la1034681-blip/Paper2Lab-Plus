"""「做」复现方案与可复现性评级验证（阶段五 · M2）——离线、零成本。

关注点:
1. 评级是**规则算出来的**、可复算：每一分都能追到一条扣分理由；
2. 等级边界必须逐档锁死（85/70/55），否则"看起来变了"却说不出为什么；
3. **同一参数只扣一次** —— 一处赋值在代码里出现多次不该把分扣穿；
4. 方案里"跑不动的部分"必须标注「需实机验证」，且**不写任何文件**。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.models import (Category, CodeInfo, Confidence, Evidence, Finding,
                              PaperInfo, Risk, Status)
from paper2lab.repro_plan import (GRADE_RULES, build_repro_plan,
                                  grade_reproducibility)

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


def sec(n):
    print("\n" + "─" * 62)
    print(n)
    print("─" * 62)


def finding(key, disp, status, risk=Risk.HIGH, pv="1e-4", cv="0.001"):
    return Finding(
        param_key=key, display_name=disp, category=Category.PARAM, status=status,
        confidence=Confidence.CONFIRMED, risk=risk,
        paper=Evidence(source="paper", location="Page 3", snippet=f"{key} = {pv}",
                       raw_value=pv) if pv else None,
        code=Evidence(source="code", location="train.py:11", snippet=f"{key} = {cv}",
                      raw_value=cv) if cv else None,
    )


def audit(findings=None, pages=(8, 10), failed=0, citation=None, crosscheck=None):
    code = CodeInfo(python_files=5, requirements={"torch": "2.1", "numpy": ""},
                    failed_files=[{"file": "bad.py", "reason": "语法错误"}] * failed,
                    file_lines={"train.py": 20})
    paper = PaperInfo(title="Demo Paper", page_count=pages[1])
    a = SimpleNamespace(
        paper=paper, code=code, findings=list(findings or []),
        coverage={"pages_total": pages[1], "pages_covered": pages[0],
                  "code_ok": 5, "code_files": 5,
                  "confidence_level": "高", "confidence_score": 90},
        selfcheck={"passed": 5, "checked": 5}, readiness={},
        rootcause=[], patches=[], compute={}, citations={}, repro_plan={}, loop={},
        user_results={}, compare_log=[],
    )
    return a


# ═══════════════════════════════════════ 1. 基线

sec("1. 满分基线 —— 没有扣分项就是 A")

g = grade_reproducibility(audit())
check(g["score"] == 100, "无扣分项 -> 100 分")
check(g["level"] == "A", "100 分 -> A")
check(g["deductions"] == [], "扣分列表为空")
check(g["basis"]["findings_total"] == 0, "basis 里给出统计依据（可核验）")

# ═══════════════════════════════════════ 2. 逐条扣分规则

sec("2. 扣分规则逐条验证（每条都要能追到理由）")

g = grade_reproducibility(audit([finding("learning_rate", "LR", Status.INCONSISTENT,
                                         Risk.HIGH)]))
check(g["score"] == 100 - GRADE_RULES["inconsistent_high"], "高风险不一致 -> 扣 8")
check(len(g["deductions"]) == 1 and g["deductions"][0]["points"] == 8, "扣分项写明分值")
check("LR" in g["deductions"][0]["reason"], "扣分理由点名是哪个参数（可核验）")

g_mid = grade_reproducibility(audit([finding("batch_size", "BS", Status.INCONSISTENT,
                                            Risk.MEDIUM)]))
check(g_mid["score"] == 100 - GRADE_RULES["inconsistent_medium"], "中风险不一致 -> 扣 4")

g_low = grade_reproducibility(audit([finding("dropout", "DP", Status.INCONSISTENT,
                                            Risk.LOW)]))
check(g_low["score"] == 100 - GRADE_RULES["inconsistent_low"], "低风险不一致 -> 扣 2")

g_seed = grade_reproducibility(audit([finding("seed", "Seed", Status.NOT_FIXED)]))
check(g_seed["score"] == 100 - GRADE_RULES["not_fixed"], "未固定随机性 -> 扣 6")
check("不可稳定复现" in g_seed["deductions"][0]["reason"], "理由说明后果（不是光扣分）")

g_int = grade_reproducibility(audit([finding("optimizer", "Opt",
                                             Status.INTERNAL_INCONSISTENT)]))
check(g_int["score"] == 100 - GRADE_RULES["internal_inconsistent"], "内部矛盾 -> 扣 5")

g_missing = grade_reproducibility(audit([finding("input_size", "Input", 
                                                Status.MISSING_IN_CODE, pv="224", cv="")]))
check(g_missing["score"] == 100 - GRADE_RULES["missing_in_code"], "论文有代码没有 -> 扣 3")
check("无法照做" in g_missing["deductions"][0]["reason"], "理由说明为什么算问题")

# 一致项不扣分
g_ok = grade_reproducibility(audit([finding("learning_rate", "LR", Status.CONSISTENT)]))
check(g_ok["score"] == 100, "一致项不扣分")

# 同一参数多条只扣一次
dup = [finding("learning_rate", "LR", Status.INCONSISTENT, Risk.HIGH),
       finding("learning_rate", "LR", Status.INCONSISTENT, Risk.HIGH)]
g_dup = grade_reproducibility(audit(dup))
check(g_dup["score"] == 100 - GRADE_RULES["inconsistent_high"],
      "同一参数多条结论只扣一次（否则一处多处赋值就把分扣穿）")
check(g_dup["basis"]["findings_penalized"] == 1, "基里如实记只惩罚 1 条")

# ═══════════════════════════════════════ 3. 覆盖与分歧

sec("3. 覆盖类与分歧类扣分")

g_cov = grade_reproducibility(audit(pages=(4, 10)))
check(g_cov["score"] == 100 - GRADE_RULES["low_page_coverage"], "页覆盖 40% -> 扣 10")
check("覆盖" in g_cov["deductions"][0]["reason"], "理由里给出覆盖数字")

g_full = grade_reproducibility(audit(pages=(9, 10)))
check(g_full["score"] == 100, "覆盖 90% -> 不扣（阈值 80%）")

g_fail = grade_reproducibility(audit(failed=2))
check(g_fail["score"] == 100 - GRADE_RULES["code_parse_failure"], "有代码文件解析失败 -> 扣 5")

g_diff = grade_reproducibility(audit(), crosscheck=[
    {"param_key": "scheduler", "verdict": "分歧", "baseline": "consistent",
     "collision": "inconsistent"}])
check(g_diff["score"] == 100 - GRADE_RULES["crosscheck_diff"], "双路核对分歧 -> 扣 4")
check("需人工裁定" in g_diff["deductions"][0]["reason"], "理由指明明需要人工裁定")
check(g_diff["basis"]["crosscheck_diffs"] == 1, "basis 记下分歧条数")

g_nodiff = grade_reproducibility(audit(), crosscheck=[
    {"param_key": "scheduler", "verdict": "互证", "baseline": "consistent",
     "collision": "consistent"}])
check(g_nodiff["score"] == 100, "互证不扣分")

g_nocite = grade_reproducibility(audit(), citation_cov={"total": 5, "cited": 0})
check(g_nocite["score"] == 100 - GRADE_RULES["no_citations"], "全部无出处 -> 扣 3")

g_halfcite = grade_reproducibility(audit(), citation_cov={"total": 10, "cited": 4})
check(g_halfcite["score"] == 100 - GRADE_RULES["no_citations"], "出处覆盖率 <50% -> 扣 3")

g_goodcite = grade_reproducibility(audit(), citation_cov={"total": 10, "cited": 8})
check(g_goodcite["score"] == 100, "出处覆盖率 80% -> 不扣")

# ═══════════════════════════════════════ 4. 等级边界

sec("4. 等级边界逐档锁死（85 / 70 / 55）")

# 85 -> A（8 + 4 + 3 = 15）
edge_a = [finding("learning_rate", "A", Status.INCONSISTENT, Risk.HIGH),
          finding("batch_size", "B", Status.INCONSISTENT, Risk.MEDIUM),
          finding("input_size", "C", Status.MISSING_IN_CODE, pv="224", cv="")]
ga = grade_reproducibility(audit(edge_a))
check(ga["score"] == 85, "恰好 85 分（构造验证）")
check(ga["level"] == "A", "85 分 -> A（含）")

# 84 -> B（8 + 8 = 16）
edge_b = [finding("learning_rate", "A", Status.INCONSISTENT, Risk.HIGH),
          finding("batch_size", "B", Status.INCONSISTENT, Risk.HIGH)]
gb = grade_reproducibility(audit(edge_b))
check(gb["score"] == 84, "恰好 84 分（构造验证）")
check(gb["level"] == "B", "84 分 -> B（差一分掉档）")

# 55 -> C（8*5 + 5 = 45）
edge_c = [finding("p%d" % i, "P%d" % i, Status.INCONSISTENT, Risk.HIGH)
          for i in range(5)] + [finding("optimizer", "Opt", Status.INTERNAL_INCONSISTENT)]
gc = grade_reproducibility(audit(edge_c))
check(gc["score"] == 55, "恰好 55 分（构造验证）")
check(gc["level"] == "C", "55 分 -> C（含）")

# 54 -> D（8*5 + 6 = 46）
edge_d = [finding("p%d" % i, "P%d" % i, Status.INCONSISTENT, Risk.HIGH)
          for i in range(5)] + [finding("seed", "Seed", Status.NOT_FIXED)]
gd = grade_reproducibility(audit(edge_d))
check(gd["score"] == 54, "恰好 54 分（构造验证）")
check(gd["level"] == "D", "54 分 -> D")

# 0 分下限
many = [finding("p%d" % i, "P%d" % i, Status.INCONSISTENT, Risk.HIGH) for i in range(30)]
gm = grade_reproducibility(audit(many, pages=(1, 10), failed=3))
check(gm["score"] == 0, "扣穿 -> 下限 0（不出现负数）")
check(gm["level"] == "D", "0 分 -> D")

# ═══════════════════════════════════════ 5. 方案步骤

sec("5. build_repro_plan —— 步骤 / 阻断项 / 边界声明")

tmp = Path(tempfile.mkdtemp())
before = sorted(p.name for p in tmp.iterdir())
a = audit([finding("learning_rate", "Learning Rate", Status.INCONSISTENT, Risk.HIGH),
           finding("seed", "Random Seed", Status.NOT_FIXED, risk=Risk.MEDIUM,
                   pv="42", cv="")])
# seed 那条没有 code 侧证据 -> 换一条有两侧的
a.findings[1].code = Evidence(source="code", location="train.py:3",
                              snippet="没有固定种子", raw_value="")
a.findings[1].paper = Evidence(source="paper", location="Page 4",
                               snippet="We set seed = 42", raw_value="42")
plan = build_repro_plan(a)
d = plan.to_dict()

check(len(d["steps"]) == 7, "方案含 7 个有序步骤: %d" % len(d["steps"]))
check([s["n"] for s in d["steps"]] == list(range(1, 8)), "步骤编号 1..7 连续")
check(d["steps"][0]["title"] == "准备运行环境", "第 1 步是环境准备")
check(d["steps"][2]["title"] == "按对照表配置超参", "第 3 步是参数配置")
check(d["steps"][3]["title"].startswith("固定随机性"), "第 4 步是固定随机性")
check(d["steps"][4]["title"].startswith("小规模试跑"), "第 5 步是小规模试跑")
check(d["steps"][6]["title"] == "留痕与交付", "最后一步是留痕交付")

check(d["steps"][3]["blocking"] is True, "seed 未固定 -> 第 4 步标为必做（blocking）")
check(any("seed" in str(t).lower() or "随机" in str(t) for t in
          [d["steps"][3]["detail"]]), "第 4 步点名是哪个参数")
check(d["steps"][2]["blocking"] is True, "有需判断项 -> 参数配置步骤也是必做")

smoke = d["steps"][4]["detail"]
check("不执行任何训练" in smoke, "试跑步骤写明不执行训练这条边界")
check("需实机验证" in smoke, "并提示需 GPU 时间的部分要标需实机验证")

check(d["grade"]["level"] in ("A", "B", "C", "D"), "方案里带评级")
check(d["env"]["python_deps"] == {"torch": "2.1", "numpy": ""}, "环境依赖来自代码清单")

# 「做」是纯函数: 不写任何文件
after = sorted(p.name for p in tmp.iterdir())
check(before == after, "build_repro_plan 不写任何文件（快照一致）")

# markdown 可交付
md = plan.to_markdown()
check("可复现性评级" in md, "markdown 含评级")
check("扣分理由" in md, "markdown 列出扣分理由（可逐条复核）")
check("复现步骤" in md, "markdown 含复现步骤")
check("复现前必须确认的事" in md, "markdown 含待确认事项")
check("不执行任何训练" in md, "markdown 里同样写明边界")

# compute / citations 被带进来当证据
plan2 = build_repro_plan(
    a, compute={"relations": [{"rid": "warmup_unit", "formula": "w = s/T",
                               "verdict": "consistent", "detail": "换算一致"}],
                "checked": 1, "flagged": 0, "skipped": 0},
    citations={"coverage": {"total": 2, "cited": 2, "inferred_only": 0}})
check(plan2.to_dict()["compute"]["checked"] == 1, "「算」的产物被带进方案")
check("小规模复算" in plan2.to_markdown(), "方案 markdown 里有复算一节")
check(plan2.to_dict()["citations"]["coverage"]["cited"] == 2, "「搜」的产物被带进方案")

# 空 audit 也能出方案（不炸）
empty = build_repro_plan(audit())
check(len(empty.to_dict()["steps"]) == 7, "没有结论也能出一个（诚实的）方案")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段五「做」验证: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_repro_plan.py)")
print("=" * 62)
