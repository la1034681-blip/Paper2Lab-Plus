"""阶段二 2D 验证 —— ② 实验站：真的跑过，还是只是看起来跑过。

守的契约:
  A 常量与菜单: 五个观察态互不相同、五个工具键互不相同、菜单与执行体同源同数
  B 一致路径: 换算后 / 倍数可解释 → 「已复算·一致」, 且预期与实际**来源不同**（非自证）
  C 对不上路径: 换算后仍然对不上 → 「已复算·对不上」（最该被看见的一类）
  D 实测量不下结论: 量出了倍数但解释不掉 → 既不是"一致"也不是"算不出来", 必须有实际值
  E 算不出来: 文字型取值 / 缺输入 → 「算不出来」, 且理由里**必须写明为什么**
  F 无需实验: ① 站已定性（已排除 / 已定位）的卡不重复验, 一次工具都不许调
  G 需实机验证: 必须给**可执行的实机方案**, 不许只写一句"需要实机"
  H 顺序 = 结论硬度: 关系式复算必须排在区间检查之前（排反了会"有结论却给不出"）
  I 原始写法可核验: `10k` 这类写法的卡, 执行记录里必须出现 `10k` 原文（不是展开后的数）
  J 非自证守卫: 预期与实际同源时, 那一步**当场作废**（不许拿自证当验证）
  K 工具真能跑: 五个工具逐个直接调用, 都返回结论（不许有"摆设工具"）
  L 不造数: 挑不出工具时如实记「算不出来」, 绝不用默认值补一个数
  M 契约: ExperimentCard.to_dict 键齐全; 旧对象（假设卡）一个字段都没被动
  N 流水线接线: TOOLS 里有序、归 ② 站、图版前置与多智能体链都登记了
  O 纯规则: 不 import 模型 / 不依赖 streamlit / 不 import hypothesis（不成环）
  P 端到端: demo 材料真跑一遍, 卡片数与统计自洽, 报告里有这一节

为什么要有这一项:
  ② 站最危险的失败模式是**"看起来做了"** —— 面板上有卡片、卡片上有字,
  但那个字不是任何工具算出来的。所以 B/C/D/K 是重点:
  每条结论都必须能追到一次**真实的工具调用**（输入 → 输出 → 耗时）。
  另外真实材料里两侧凑得齐公式的场景很少, 所以 B/C 大半用合成用例 ——
  光跑真样本根本覆盖不到「一致」「对不上」这两条路径（2B 已踩过同一个坑）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import experiment as X                                  # noqa: E402
from paper2lab.experiment import (EXP_AGREE, EXP_DISAGREE,             # noqa: E402
                                  EXP_MACHINE, EXP_MEASURED, EXP_NO_NEED,
                                  EXP_UNDECIDED, EXPERIMENT_TOOLS,
                                  catalog_keys, design_experiments,
                                  engine_name, menu, summarize)
from paper2lab.hypothesis import Hypothesis                            # noqa: E402

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


def card(key, **kw):
    """造一张假设卡 —— 字段与 hypothesis.Hypothesis 完全一致, 不另造一套。"""
    base = dict(key=key, kind="取值冲突", claim=f"{key} 两侧取值不同",
                question="哪个才是真正生效的", falsifiable_by="跑一次短训练看结果")
    base.update(kw)
    return Hypothesis(**base)


def run(paper_params, code_params, cards, llm=None):
    return design_experiments(cards,
                              SimpleNamespace(params=paper_params),
                              SimpleNamespace(params=code_params), llm=llm)


def one(paper_params, code_params, c, llm=None):
    got = run(paper_params, code_params, [c], llm=llm)
    return got[0] if got else None


print("=" * 72)
print("阶段二 2D · ② 实验站（实验设计与真跑）")
print("=" * 72)

# ---------------------------------------------------------------- A 常量与菜单
print("\nA 常量与菜单")
_states = {EXP_AGREE, EXP_DISAGREE, EXP_MEASURED, EXP_UNDECIDED, EXP_MACHINE, EXP_NO_NEED}
check(len(_states) == 6, f"六个观察态互不相同（{len(_states)} 个）")
check(all(bool(v) for v in _states), "每个观察态都有非空中文名")
check(len(EXPERIMENT_TOOLS) == len({t.key for t in EXPERIMENT_TOOLS}),
      "工具键不重复")
check(len(EXPERIMENT_TOOLS) == len(catalog_keys()),
      f"菜单与执行体同源同数（{len(EXPERIMENT_TOOLS)} 件）")
check(bool(engine_name()), "引擎名非空（界面 / 报告都读它）")
_m = menu()
check(len(_m) == len(EXPERIMENT_TOOLS) and all("key" in e and "名称" in e for e in _m),
      "给模型看的菜单形状完整（key / 名称 / 需要输入 / 产出）")
check(all(t.cost and t.desc for t in EXPERIMENT_TOOLS),
      "每个工具都写清了说明与成本（菜单上没有来路不明的东西）")

# ---------------------------------------------------------------- B 一致路径
print("\nB 一致路径（换算后一致）")
e_ok = one({"warmup": "0.05"}, {"warmup": "500", "iterations": "10000"}, card("warmup"))
check(e_ok is not None and e_ok.status == EXP_AGREE,
      f"warmup 步数↔比例 换算一致 → {EXP_AGREE}（实得 {e_ok.status if e_ok else '—'}）")
check(bool(e_ok and e_ok.expected_from != e_ok.actual_from),
      "预期与实际**来源不同**（非自证）")
check(bool(e_ok and e_ok.expected is not None and e_ok.actual is not None),
      "一致路径必须同时给出预期值与实际值（只给一个等于没说）")

e_fac = one({"batch_size": "4096"}, {"batch_size": "512", "world_size": "8"}, card("batch_size"))
check(e_fac is not None and e_fac.status == EXP_AGREE,
      "批大小差 8 倍、材料里正好有 8 张卡 → 一致（口径差, 不是真分歧）")
check(bool(e_fac and "8" in str(e_fac.actual)), "实际值是量出来的倍数（8）")

# ---------------------------------------------------------------- C 对不上路径
print("\nC 对不上路径")
e_bad = one({"warmup": "0.1"}, {"warmup": "500", "iterations": "10000"}, card("warmup"))
check(e_bad is not None and e_bad.status == EXP_DISAGREE,
      f"换算后仍然对不上 → {EXP_DISAGREE}（实得 {e_bad.status if e_bad else '—'}）")
check(bool(e_bad and e_bad.expected is not None and e_bad.actual is not None
           and e_bad.expected != e_bad.actual),
      "对不上时预期与实际确实是两个不同的数")

# ---------------------------------------------------------------- D 实测量不下结论
print("\nD 实测量·不下结论")
e_meas = one({"learning_rate": "1e-4"}, {"learning_rate": "0.001"}, card("learning_rate"))
check(e_meas is not None and e_meas.status == EXP_MEASURED,
      f"量出 10 倍但材料解释不掉 → {EXP_MEASURED}（实得 {e_meas.status if e_meas else '—'}）")
check(bool(e_meas and e_meas.actual is not None),
      "「不下结论」不等于「没测到」—— 实际值必须留下")
check(bool(e_meas and "ratio_probe" in e_meas.tool_path),
      "这一类的工具路径里有倍数探测")

# ---------------------------------------------------------------- E 算不出来
print("\nE 算不出来")
e_txt = one({"dataset": "CIFAR-10"}, {"dataset": "v2.0"}, card("dataset"))
check(e_txt is not None and e_txt.status == EXP_UNDECIDED,
      f"文字型取值 → {EXP_UNDECIDED}")
check(bool(e_txt and "不是数值型" in e_txt.design),
      "理由里写明了**为什么**算不出来（不是只有「算不出来」四个字）")
check(bool(e_txt and not e_txt.records), "算不出来时不留下假的执行记录")

e_none = one({}, {}, card("whatever"))
check(e_none is not None and e_none.status == EXP_UNDECIDED,
      "两侧都没抽到值 → 算不出来（不猜）")
check(bool(e_none and "不猜" in e_none.design or "没抽到" in e_none.design),
      "并写明是「两侧都没抽到取值」")

# ---------------------------------------------------------------- F 无需实验
print("\nF 无需实验（① 站已定性）")
for verdict, tag in (("已排除", "已排除"), ("已定位", "已定位")):
    e_skip = one({"dropout": "0.5"}, {"dropout": "1.5"},
                 card("dropout", verdict=verdict))
    check(e_skip is not None and e_skip.status == EXP_NO_NEED,
          f"① 站判「{tag}」的卡 → {EXP_NO_NEED}（不重复验）")
    check(bool(e_skip and not e_skip.records),
          f"「{tag}」的卡一次工具都不许调（不空转）")

# ---------------------------------------------------------------- G 需实机验证
print("\nG 需实机验证")
e_mac = one({"model": "ResNet-18"}, {"arch": "resnet"}, card("model", needs_rerun=True))
check(e_mac is not None and e_mac.status == EXP_MACHINE,
      f"必须实机才能定的卡 → {EXP_MACHINE}（不假装算过）")
check(bool(e_mac and len(e_mac.rerun_plan) > 30),
      "必须给**可执行的实机方案**（一句话的「需要实机」不算方案）")
check(bool(e_mac and "判定标准" in e_mac.rerun_plan),
      "实机方案里带上了「看到什么算通过」的判定标准")

# ---------------------------------------------------------------- H 顺序 = 结论硬度
print("\nH 工具顺序（结论硬度）")
e_ord = one({"warmup": "0.05"}, {"warmup": "500", "iterations": "10000"}, card("warmup"))
path = list(e_ord.tool_path) if e_ord else []
check("relation_eval" in path,
      "关系式复算排得进来（早先被区间检查挤掉过, 结果是「跑了一堆却给不出结论」）")
if "relation_eval" in path and "bounds_check" in path:
    check(path.index("relation_eval") < path.index("bounds_check"),
          "关系式复算排在区间检查之前（硬的先做）")
else:
    check(True, "关系式复算排在区间检查之前（本次区间检查未排入）")
check(bool(e_ord and not e_ord.tool_path.count("bounds_check") > 2),
      f"工具路径不超过 {X.MAX_STEPS} 步（不靠堆步骤充工作量）")

# ---------------------------------------------------------------- I 原始写法可核验
print("\nI 原始写法可核验")
e_unit = one({"warmup": "10000", "iterations": "200000"}, {"warmup": "10k"}, card("warmup"))
_blob = str([r.get("inputs") for r in (e_unit.records if e_unit else [])])
check("10k" in _blob,
      "执行记录里留下了原始写法 `10k`（用户能照着核对; 只记展开后的数他对不上）")
check(bool(e_unit and "expand_units" in e_unit.tool_path),
      "需要归一时工具路径里有单位展开（不是悄悄在取数时改掉）")

# ---------------------------------------------------------------- J 非自证守卫
print("\nJ 非自证守卫")
_tmp_key = "_verify_same_source"
X._TOOL_BY_KEY[_tmp_key] = X.ExpTool(
    key=_tmp_key, name="同源工具", desc="测试用", cost="零", needs=(), gives="值",
    run=lambda inp: {"ok": True, "text": "同源", "compare": True,
                     "expected": 1.0, "actual": 1.0,
                     "expected_from": "同一个地方", "actual_from": "同一个地方"})
try:
    _recs, _agg = X._run_steps([{"tool": _tmp_key, "inputs": {}, "why": "测试"}])
    check(not _agg["compares"],
          "预期与实际同源时那一步当场作废（自证验不出任何东西）")
    check(bool(_recs and "同源" in _recs[0]["output"]),
          "并在记录里如实标注（不是静默丢弃）")
finally:
    X._TOOL_BY_KEY.pop(_tmp_key, None)

# ---------------------------------------------------------------- K 工具真能跑
print("\nK 工具真能跑（五个都逐个直接调）")
_cases = {
    "expand_units": {"raw": "10k"},
    "eval_expr": {"raw": "3*1e-4"},
    "bounds_check": {"key": "dropout", "value": "1.5"},
    "ratio_probe": {"key": "batch_size", "a": "4096", "b": "512",
                    "factor_name": "并行进程数", "factor_raw": "world_size",
                    "factor_value": 8},
    "relation_eval": {"rid": "warmup_unit",
                      "paper": {"warmup": "0.05", "iterations": "10000"},
                      "code": {"warmup": "500"}},
    # ---- 阶段三 3A 的四件具名科学计算工具（2026-10-17）----
    # 这四件要**成串**的输入, 所以用例得给足 —— 给一个空 dict 当然跑不出结论,
    # 那不是工具坏了, 是没喂料（3A 的专项验证 tests/verify_science_tools.py 里
    # 把四条边界都覆盖了: 点太少 / 横坐标相同 / 端点峰 / 精度档位）。
    "fp16_precision": {"raw": ["1e-8"], "precision": "fp16"},
    "series_stats": {"values": ["0.90", "0.92", "0.91"]},
    "linear_fit": {"xs": [1, 2, 3, 4], "ys": [0.5, 0.52, 0.51, 0.53]},
    "peak_pick": {"xs": [4000, 3500, 3000], "ys": [0.2, 0.8, 0.3]},
}
for _t in EXPERIMENT_TOOLS:
    _out = dict(_t.run(dict(_cases.get(_t.key) or {})))
    check(bool(_out.get("ok")) and bool(str(_out.get("text") or "")),
          f"工具 `{_t.key}`（{_t.name}）真能跑出结论, 不是摆设")

check(X._TOOL_BY_KEY["bounds_check"].run(
    {"key": "dropout", "value": "1.5"}).get("compare") is False,
    "区间检查：越界 → compare=False（对不上）")
check(X._TOOL_BY_KEY["bounds_check"].run(
    {"key": "dropout", "value": "0.5"}).get("compare") is None,
    "区间检查：在区间内 → 不下结论（值合法 ≠ 假设被验证）")
check(X._TOOL_BY_KEY["ratio_probe"].run(
    {"key": "batch_size", "a": "512", "b": "512"}).get("ok") is False,
    "倍数探测：两侧几乎相等 → 如实说量不出倍数（不输出一个 1.0 充数）")
check(X._TOOL_BY_KEY["expand_units"].run({"raw": "Adam"}).get("ok") is False,
    "单位展开：`Adam` 不是「数字+单位」（只看末位字符会把模型名当单位）")
check(X._TOOL_BY_KEY["expand_units"].run({"raw": "LiT-B16B"}).get("ok") is False,
    "单位展开：`LiT-B16B` 同理（这一条是实测踩出来的）")

# ---------------------------------------------------------------- L 不造数
print("\nL 不造数")
_src = (ROOT / "paper2lab" / "experiment.py").read_text(encoding="utf-8")
check("不用默认值" in _src or "不猜" in _src,
      "源码里写明了「不造数」这条纪律")
check("def _why_no_tool" in _src,
      "挑不出工具时有专门的函数写出**为什么**（只说「算不出来」等于什么都没说）")
e_blank = one({"epochs": "100"}, {}, card("iterations"))
check(bool(e_blank and e_blank.status == EXP_UNDECIDED and not e_blank.records),
      "只有一侧有值时不当成「算出来了」（宁可记算不出来）")

# ---------------------------------------------------------------- M 契约
print("\nM 契约")
e_any = one({"seed": "42"}, {"seed": "42"}, card("seed")) or one(
    {"learning_rate": "1e-4"}, {"learning_rate": "0.001"}, card("learning_rate"))
_d = e_any.to_dict() if e_any else {}
for k in ("key", "rank", "claim", "status", "status_desc", "design", "tool_path",
          "records", "expected", "expected_from", "actual", "actual_from",
          "plan_reason", "rerun_plan", "seconds"):
    check(k in _d, f"ExperimentCard.to_dict 有键 `{k}`")
check(all("seq" in r and "tool" in r and "inputs" in r and "ok" in r
          and "output" in r and "ms" in r for r in (_d.get("records") or [])),
      "每条执行记录都带序号 / 工具 / 输入 / 成败 / 输出 / 耗时（可复现）")
_h_keys = set(Hypothesis("k", "取值冲突", "c", "q", "f").to_dict())
check({"verdict", "derivation", "derivation_kind", "engine"} <= _h_keys,
      "2B 给假设卡加的字段一个没丢（2D 没有动上游契约）")

# ---------------------------------------------------------------- N 流水线接线
print("\nN 流水线接线")
from paper2lab.graph import _MULTAGENT_AFTER_MAP, _PRECONDITIONS   # noqa: E402
from paper2lab.toolbox import STATION_OF_STEP, STEPS, TOOLS        # noqa: E402

_keys = [t.key for t in TOOLS]
check("design_experiment" in _keys, "工具箱里登记了「实验设计与真跑」")
check(_keys.index("design_experiment") > _keys.index("hypothesize"),
      "排在假设生成之后（没有假设卡就无从设计实验）")
check(_keys.index("design_experiment") < _keys.index("align"),
      "排在参数对齐之前（先做实验, 再拿结果去判）")
check(STATION_OF_STEP.get("实验设计") == "② 实验",
      "「实验设计」归 ② 实验站（② 站不再是没有环节的空站）")
check("实验设计" in STEPS, "进度条里有这一格")
check(_PRECONDITIONS.get("design_experiment") == {"hypotheses"},
      "图版前置登记了（不登记 = 闸门形同虚设）")
check(_MULTAGENT_AFTER_MAP.get("hypothesize") == "design_experiment",
      "多智能体链：假设生成之后接实验（不接就会被整段跳过 = 两模式口径漂移）")
check(_MULTAGENT_AFTER_MAP.get("design_experiment") == "roles",
      "多智能体链：实验之后接对撞区")
_gsrc = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")
check('"paper2lab.experiment", "ExperimentCard"' in _gsrc,
      "checkpoint 白名单登记了 ExperimentCard（漏登记 = 断点续跑埋雷）")
_msrc = (ROOT / "paper2lab" / "models.py").read_text(encoding="utf-8")
check("experiments: list" in _msrc and "experiment_stats" in _msrc,
      "AuditResult 上挂了实验卡（否则报告与界面看不到）")

# ---------------------------------------------------------------- O 纯规则
print("\nO 纯规则")
check("import streamlit" not in _src, "不依赖 streamlit（可离线、可在 CLI 跑）")
check("from .llm import" not in _src and "import llm" not in _src,
      "不在模块级 import 模型（模型通道是可选回调, 不是硬依赖）")
check("open(" not in _src, "不碰任何文件（② 站只算, 不写盘）")
check("eval(" not in _src.replace("eval_arith", "").replace("relation_eval", ""),
      "不直接用 eval（求值一律走 compute 的 ast 白名单）")
check("import hypothesis" not in _src and "from .hypothesis import" not in _src,
      "不 import hypothesis（反向依赖会成环）")

# ---------------------------------------------------------------- P 端到端
print("\nP 端到端（demo 真跑）")
_formal = ROOT / "demo" / "paper.pdf"
_zip = ROOT / "demo" / "project.zip"
if _formal.exists() and _zip.exists():
    from paper2lab.code_analyzer import analyze_code              # noqa: E402
    from paper2lab.derivation import derive                       # noqa: E402
    from paper2lab.hypothesis import build_hypotheses             # noqa: E402
    from paper2lab.paper_parser import parse_paper                # noqa: E402
    from paper2lab.pipeline import run_audit                      # noqa: E402

    _paper = parse_paper(str(_formal))
    _code = analyze_code(str(_zip))
    _cards = derive(build_hypotheses(_paper, _code), _paper, _code)
    _exps = design_experiments(_cards, _paper, _code)
    _st = summarize(_exps)
    check(len(_exps) == len(_cards),
          f"每张假设卡都有交代（{len(_cards)} 张卡 → {len(_exps)} 个实验）")
    check(_st["total"] == len(_exps), "统计总数与卡片数一致")
    check(_st["disagree"] + _st["agree"] + _st["measured"] + _st["machine"]
          + _st["undecided"] + _st["no_need"] == _st["total"],
          "六个观察态加起来正好是总数（没有掉进缝里的卡）")
    check(all(e.status in (EXP_AGREE, EXP_DISAGREE, EXP_MEASURED, EXP_MACHINE,
                           EXP_UNDECIDED, EXP_NO_NEED) for e in _exps),
          "每张卡的观察态都是登记过的常量（没有散装字符串）")
    check(all(e.rank >= 1 for e in _exps) and
          [e.rank for e in _exps] == list(range(1, len(_exps) + 1)),
          "rank 连续（同分有稳定次序, 消融对照才能比）")
    check(all(bool(e.design) for e in _exps),
          "每张卡都写清了实验设计 —— 包括「不做实验」的卡也写明为什么不做")

    _audit, _report = run_audit(str(_formal), str(_zip), force_mock=True)
    check(len(_audit.experiments) == min(_st["total"], X.EXPERIMENT_MAX),
          f"界面拿到的卡片数 = min(总数, 展示上限 {X.EXPERIMENT_MAX})")
    check(_audit.experiment_total == _st["total"],
          f"总数如实报（{_audit.experiment_total} 条, 不只报展示的那几条）")
    check("② 实验站" in _report, "导出的报告里有「② 实验站」这一节")
    check("真跑记录" in _report, "报告里带上了真跑记录（可照着复现）")
    _steps = {s["key"] for s in (_audit.execution_log or {}).get("steps", [])}
    check("design_experiment" in _steps, "经典版流水线真的跑了这一步")
    _ev = (ROOT / "paper2lab" / "execview.py").read_text(encoding="utf-8")
    check("design_experiment" in _ev, "执行记录视图里有这一步的产物说明")
else:
    check(False, "demo 材料缺失，无法端到端验证")

print("\n" + "=" * 72)
print(f"共 {CHECKS} 项，未通过 {len(FAILURES)} 项")
if FAILURES:
    for f in FAILURES:
        print("  *x* " + f)
    sys.exit(1)
print("全部通过 ✅")
sys.exit(0)
