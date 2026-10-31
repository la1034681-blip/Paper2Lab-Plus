"""阶段七·评测打磨 验证 —— 契约对齐 / 金标准 / 判分 / 零重依赖。

关注点（每条对应一个"容易做错"的地方）:
1. **契约必须与书生·司南 SciEvalKit 逐字一致** —— 否则"能挂进平台"就是空话，
   接入时会因参数名/语义不同而静默跑歪。本测试直接读官方源码做签名比对。
2. **判分不能自证** —— 金标准来自论文公开超参，不是工具输出；判分器只做度量，
   不做"通过/不通过"硬断言（recall 可能合理低于 1）。
3. **诱饵必须能被抓住** —— "合法差异"被误升级为高风险，是最伤信任的错。
4. **零重依赖** —— 不能把 torch / vllm / transformers 拖进来（本地跑不动）。
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.evalsuite import goldset, scoring
from paper2lab.evalsuite.agent_adapter import ANSWER_SCHEMA, Paper2LabAgent
from paper2lab.evalsuite.audit_bench import AuditBenchDataset
from paper2lab.evalsuite.scieval_compat import (CONTRACT_SIGNATURES, SCHEMA_SOURCE,
                                                AgentBase, EvalRecord, EvalResult,
                                                EvalSample, TrajectoryStore,
                                                run_agent_eval)

CHECKS = 0
FAILURES = []


def check(ok, label):
    global CHECKS
    CHECKS += 1
    # 防呆: 参数传反会让 ok 拿到非空字符串 -> 永远打印"通过"
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


# ═══════════════════════════════════════ 1. 契约签名对齐

sec("1. 契约签名 —— 必须与 SciEvalKit 逐字一致")

#: 本地类 → 受检签名
_LOCAL = {
    "EvalSample.__init__": EvalSample.__init__,
    "AgentBase.__init__": AgentBase.__init__,
    "AgentBase.run": AgentBase.run,
    "EvalResult.__init__": EvalResult.__init__,
    "TrajectoryStore.__init__": TrajectoryStore.__init__,
    "EvalRecord.__init__": EvalRecord.__init__,
}


def _params(func) -> list:
    out = []
    for p in inspect.signature(func).parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            out.append("kwargs")
        else:
            out.append(p.name)
    return out


for key, expected in CONTRACT_SIGNATURES.items():
    got = _params(_LOCAL[key])
    check(got == expected, f"{key} 参数对齐 {expected}（实得 {got}）")

check(bool(SCHEMA_SOURCE), "契约来源已标注（写入产物便于交代口径）")

# 真机比对: 官方源码在临时克隆目录时逐字核验
_sc = Path(r"C:\Users\lijiahui\AppData\Local\Temp\scieval_probe\SciEvalKit")
if _sc.exists():
    _base = (_sc / "scieval" / "agents" / "base.py").read_text(encoding="utf-8")
    _rec = (_sc / "scieval" / "agents" / "records.py").read_text(encoding="utf-8")
    check("class EvalSample:" in _base, "官方 base.py 含 EvalSample")
    check("class AgentBase(ABC):" in _base, "官方 base.py 含 AgentBase(ABC)")
    check("def run(self, sample" in _base, "官方 AgentBase.run(self, sample) 签名一致")
    check("class EvalResult:" in _rec, "官方 records.py 含 EvalResult")
    check("class TrajectoryStore:" in _rec, "官方 records.py 含 TrajectoryStore")
    check("def save_traj(self, idx" in _rec, "官方 save_traj(self, idx, result) 一致")
    check("def score_agent_sample" in (_sc / "scieval" / "dataset" / "text_base.py")
          .read_text(encoding="utf-8"), "官方 text_base.py 含 score_agent_sample 钩子")
else:
    print("  [--] 官方源码克隆不存在，跳过真机签名比对（仅本地契约自检）")

# ═══════════════════════════════════════ 2. 金标准集

sec("2. 金标准集 —— 自建考卷的结构完整性")

_desc = goldset.describe()
check(_desc["total"] >= 8, f"考卷数 >= 8（实得 {_desc['total']}）")
check(bool(_desc["evidence_note"]), "金标准标注了依据来源（防自证）")
check("非取自本工具输出" in goldset.EVIDENCE_NOTE, "依据明写「不是工具输出」")

_kinds = {g.kind for g in goldset.GOLD_SAMPLES}
check("real" in _kinds, "含真实论文样本（real）")
check("edge" in _kinds, "含诱饵样本（edge，考精度）")
check("modeling" in _kinds, "含数学建模题样本（参数域与 ML 不同，单独标注）")

_noexpect = [g.sid for g in goldset.GOLD_SAMPLES if g.kind != "edge" and not g.expect]
check(not _noexpect, f"非诱饵样本都有期望项（缺: {_noexpect}）")

_edge = [g for g in goldset.GOLD_SAMPLES if g.kind == "edge"]
check(bool(_edge), "含诱饵/边缘样本")
check(all(g.expect or g.decoy for g in _edge), "边缘样本声明了考题（该报的或不该报的）")

# 期望项必须是合法规范化参数键
from paper2lab.synonyms import PARAM_SYNONYMS as _PS
_bad = [k for g in goldset.GOLD_SAMPLES for k in g.expect if k not in _PS]
check(not _bad, f"期望项都是合法参数键（非法: {_bad}）")

# ═══════════════════════════════════════ 3. 判分器四维

sec("3. 判分器 —— 准 / 全 / F1 / 自证 / 诱饵")

_full = {"found_params": {"paper": ["learning_rate", "batch_size"], "code": ["learning_rate"]},
         "inconsistencies": [{"param_key": "learning_rate", "verified": True}],
         "selfcheck": {"passed": 1, "checked": 1}}
_s = scoring.evaluate_answer(expect=["learning_rate", "batch_size"], decoy=[], answer=_full)
check(_s["recall"] == 1.0, "全抓 → recall 1.0")
check(_s["precision"] == 1.0, "证据站得住 → precision 1.0")
check(_s["f1"] == 1.0, "准全皆满 → f1 1.0")

_part = {"found_params": {"paper": ["learning_rate"], "code": []},
         "inconsistencies": [{"param_key": "learning_rate", "verified": False}],
         "selfcheck": {"passed": 0, "checked": 1}}
_p = scoring.evaluate_answer(expect=["learning_rate", "batch_size", "epochs"], decoy=[], answer=_part)
check(abs(_p["recall"] - 1 / 3) < 1e-3, "抓 1/3 → recall ≈0.333")
check(_p["precision"] == 0.0, "证据全不过 → precision 0.0")
# precision 为 0 时 f1 必然 0，故另构造"两头都不满"的案例
_mid = {"found_params": {"paper": ["learning_rate"], "code": []},
        "inconsistencies": [{"param_key": "a", "verified": True},
                            {"param_key": "b", "verified": False}],
        "selfcheck": {"passed": 1, "checked": 2}}
_m = scoring.evaluate_answer(expect=["learning_rate", "batch_size"], decoy=[], answer=_mid)
check(0.0 < _m["f1"] < 1.0, "precision/recall 都不满 → f1 居中")

_zero = {"found_params": {"paper": [], "code": []}, "inconsistencies": [], "selfcheck": {}}
_z = scoring.evaluate_answer(expect=["learning_rate"], decoy=[], answer=_zero)
check(_z["recall"] == 0.0, "零抓 → recall 0.0")
check(_z["precision"] == 1.0, "零误报 → precision 记 1.0（没冤枉人）")
check(_z["selfcheck_rate"] == 0.0, "无自检记录 → 自证率 0.0")

# 诱饵：报了"不该报的" -> 记 FP
_decoy_ans = {"found_params": {"paper": ["learning_rate"], "code": []},
              "inconsistencies": [{"param_key": "learning_rate", "verified": True}],
              "selfcheck": {"passed": 1, "checked": 1}}
_d = scoring.evaluate_answer(expect=["learning_rate"], decoy=["learning_rate"], answer=_decoy_ans)
check(_d["decoy_fp"] == 1.0, "诱饵被报为不一致 → decoy_fp 1")

# 数值可聚合（run_agent_eval 只会对 number 求均值）
_all_num = all(isinstance(v, float) or isinstance(v, int) for v in _s.values())
_all_notbool = not any(isinstance(v, bool) for v in _s.values())
check(bool(_all_num and _all_notbool), "判分返回值全是可聚合数值（非 bool）")

# 无期望项（诱饵样本）→ 不产出召回，避免污染均值
_none = scoring.evaluate_answer(expect=[], decoy=[], answer=_zero)
check(bool("recall" not in _none and _none["decoy_fp"] == 0.0),
      "无期望项(诱饵样本) → 不产出 recall，避免污染整体均值")

# ═══════════════════════════════════════ 4. 适配器

sec("4. 适配器 —— agent / dataset 协议")

_agent = Paper2LabAgent(mode="classic")
check(bool(_agent.name), "agent.name 非空")
check(_agent.name == "paper2lab", "agent.name = paper2lab")
check(bool(_agent.model_version), "agent.model_version 非空")
check(callable(getattr(_agent, "run", None)), "agent.run 可调用")

for m in ("classic", "multagent", "closeloop"):
    _a = Paper2LabAgent(mode=m)
    check(bool(_a.model_version.endswith(m)), f"模式 {m} 写入 model_version")

try:
    Paper2LabAgent(mode="不存在的模式")
    check(False, "未知模式应报错")
except ValueError:
    check(True, "未知模式报错（不静默跑错）")

# 缺路径 → 结构化失败，不抛异常
_bad = _agent.run(EvalSample(prompt="x", files={}, metadata={"sid": "t"}))
check(_bad.success is False, "缺文件 → success=False（不抛异常）")
check(json.loads(_bad.final_answer).get("ok") is False, "缺文件 → 作答标 ok=False 且可解析")

# 失败路径（不存在的文件）也不抛
_bad2 = _agent.run(EvalSample(prompt="x", files={"pdf": "no.pdf", "code": "no.zip"},
                              metadata={"sid": "t2"}))
check(isinstance(_bad2.success, bool), "坏路径有明确成败信号")

# dataset 协议
_samples = [s for s in goldset.list_resolved() if s.sid == "demo"]
check(bool(_samples), "本机存在 demo 考卷")
_ds = AuditBenchDataset(_samples)
check(_ds.dataset_name == "Paper2Lab-AuditBench", "dataset_name 固定")
check(len(_ds) == 1, "len(dataset) = 样本数")
_smp = _ds.build_agent_sample(0)
check(isinstance(_smp, EvalSample), "build_agent_sample 返回 EvalSample")
check(bool(_smp.files.get("pdf")), "题目携带 pdf 路径")
check(_smp.metadata.get("sid") == "demo", "题目元数据带 sid")

# 判分对脏作答不炸
_dirty = _ds.score_agent_sample(0, "这不是 JSON", )
check(_dirty["parse_ok"] == 0.0, "脏作答 → parse_ok 0（不炸）")

# ═══════════════════════════════════════ 5. 端到端（一套，快）

sec("5. 端到端 —— demo 一套真跑")

_sum = run_agent_eval(Paper2LabAgent(mode="classic"), AuditBenchDataset(_samples),
                      work_dir=str(ROOT / "eval_output_test"))
check(isinstance(_sum, dict), "返回 summary dict")
check("recall" in _sum and "precision" in _sum, "summary 含四维指标")
check(_sum["recall"] > 0.0, "demo 召回 > 0")
check(_sum["selfcheck_rate"] >= 0.9, "demo 证据自证率 >= 0.9")
check("schema_source" in json.loads(
    (ROOT / "eval_output_test" / "agent_eval" / "Paper2Lab-AuditBench" /
     "paper2lab" / "v1-classic" / "summary.json").read_text(encoding="utf-8")),
    "summary.json 落盘且标注契约来源")

# ═══════════════════════════════════════ 6. 零重依赖

sec("6. 零重依赖 —— 不拖 torch / vllm / transformers")

_ver = [m for m in ("torch", "vllm", "transformers", "tensorflow", "qwen_agent")
        if m in sys.modules]
check(not _ver, f"导入 evalsuite 未拉起重依赖（实得 {_ver}）")

# ═══════════════════════════════════════ 7. 三档消融（3C）

sec("7. 三档消融 —— 只开解析员 / 加计算员 / 全开加审查员")

from paper2lab.evalsuite.agent_adapter import (TIERS, TIER_BY_KEY,  # noqa: E402
                                               tier_keys)
from paper2lab.evalsuite.scoring import ALL_METRICS         # noqa: E402
from paper2lab.runner import ABLATION_SKIP_REASON           # noqa: E402
from paper2lab.toolbox import ABLATION_LOCKED, TOOL_BY_KEY  # noqa: E402

# --- 7.1 档位表本身
check(tier_keys() == ("parser", "calc", "review"), "三档就是三档，顺序即能力递增")
for _a, _b in zip(TIERS, TIERS[1:]):
    check(set(_a.metrics) <= set(_b.metrics),
          f"档位 {_b.key} 的指标覆盖 {_a.key}（只能加、不能丢）")
    check(set(_b.skip) <= set(_a.skip),
          f"档位 {_b.key} 跳过的工具不多于 {_a.key}")
check(len(TIER_BY_KEY["parser"].skip) == 3,
      "档一只关三件：提假设 / 设计实验 / 自检（读与比是底座，三档都在）")
_skip_all = set().union(*[set(t.skip) for t in TIERS])
check(not (_skip_all & ABLATION_LOCKED),
      "任何档位都不许跳受保护的底座（抽取 / 比对 / 报告）")

# --- 7.2 档位与 agent 的绑定
for _t in tier_keys():
    _ag = Paper2LabAgent(mode="classic", tier=_t)
    check(_ag.skip_tools == TIER_BY_KEY[_t].skip, f"档位 {_t} 的跳过清单与表一致")
    check(_ag.model_version.endswith("classic") and _t in _ag.model_version,
          f"档位 {_t} 写进 model_version（结果目录不串）")

try:
    Paper2LabAgent(mode="classic", tier="不存在的档")
    check(False, "未知档位应报错")
except ValueError:
    check(True, "未知档位报错（不静默按全开跑）")

try:
    Paper2LabAgent(mode="multagent", tier="parser")
    check(False, "档位配图版应报错")
except ValueError:
    check(True, "档位不给图版用（图版是硬边串起来的，逐件跳过会破图）")

# --- 7.3 反事实：档位是**真跳过**，不是把输出藏起来
from paper2lab import pipeline as _pl                       # noqa: E402

_audit_skip, _ = _pl.run_audit(str(_samples[0].pdf), str(_samples[0].code),
                              _samples[0].results or None, force_mock=True,
                              skip_tools=("selfcheck",))
_steps = {s["key"]: s for s in _audit_skip.execution_log.get("steps", [])}
check(_steps.get("selfcheck", {}).get("status") == "skipped",
      "自检这一件在执行记录里是 skipped（真没跑）")
check(_steps.get("selfcheck", {}).get("reason") == ABLATION_SKIP_REASON,
      "跳过理由统一标注（可核，不是悄悄少跑）")
check(bool(_audit_skip.selfcheck.get("skipped")), "audit.selfcheck 标 skipped")
check(all(f.verified is False or f.verified is True for f in _audit_skip.findings),
      "档位跳过不篡改底层 findings（verified 原值留在审计对象上）")
check(any("消融档位" in n for n in _audit_skip.execution_log.get("notes", [])),
      "执行记录备注里写明本档未开哪些能力")

try:
    _pl.run_audit(str(_samples[0].pdf), str(_samples[0].code),
                  _samples[0].results or None, force_mock=True,
                  skip_tools=("align",))
    check(False, "跳底座应报错")
except AssertionError:
    check(True, "跳受保护底座 → 当场断言失败（不出一份空答卷）")

# --- 7.4 三档真跑（同一份 demo 考卷）
_tier_sums = {}
for _t in tier_keys():
    _ds_t = AuditBenchDataset(_samples)
    _tier_sums[_t] = run_agent_eval(
        Paper2LabAgent(mode="classic", tier=_t), _ds_t,
        work_dir=str(ROOT / "eval_output_test"))
    _det = _ds_t.details.get(0, {})
    _tier_sums[_t]["_det"] = _det

_det_p = _tier_sums["parser"]["_det"]
_det_c = _tier_sums["calc"]["_det"]
_det_r = _tier_sums["review"]["_det"]

check(_det_p.get("tier") == "parser" and _det_r.get("tier") == "review",
      "作答里带档位标记（明细可追溯）")
check((_det_p.get("stats") or {}).get("n_hypotheses", 0) == 0,
      "档一：一张假设卡都没有（②站没开）")
check((_det_p.get("stats") or {}).get("n_experiments", 0) == 0,
      "档一：一张实验卡都没有")
check((_det_c.get("stats") or {}).get("n_hypotheses", 0) > 0,
      "档二：假设卡出来了（①站算过）")
check((_det_c.get("stats") or {}).get("n_experiments", 0) > 0,
      "档二：实验卡出来了（②站真跑过）")
check((_det_c.get("stats") or {}).get("n_judged", 0) == 0,
      "档二：一条都没判（③站没开）")
check((_det_r.get("stats") or {}).get("n_judged", 0) > 0,
      "档三：判定出来了（③站开了）")

# 指标白名单：没开的能力不出数，且**绝不填 0**
check("precision" not in _tier_sums["parser"], "档一不产出 precision（不是 0 分，是没测）")
check("selfcheck_rate" not in _tier_sums["parser"], "档一不产出 selfcheck_rate")
check("calc_coverage" not in _tier_sums["parser"], "档一不产出 calc_coverage")
check("calc_coverage" in _tier_sums["calc"], "档二产出 calc_coverage（算了多少条）")
check("precision" in _tier_sums["review"] and "selfcheck_rate" in _tier_sums["review"],
      "档三才产出 precision / selfcheck_rate")

# 只加能力、不改结论
check(_tier_sums["parser"].get("recall") == _tier_sums["review"].get("recall"),
      "三档召回一致：加能力不改结论")
check(_tier_sums["parser"].get("decoy_fp") == _tier_sums["review"].get("decoy_fp"),
      "三档诱饵误报一致：加能力不乱报")
check(_tier_sums["parser"].get("n_findings") == _tier_sums["review"].get("n_findings"),
      "三档结论条数一致")
check(_tier_sums["calc"].get("calc_coverage", 0.0) > 0.0,
      "档二确实算出了数（calc_coverage > 0）")
check(_tier_sums["review"].get("judged", 0.0) > 0.0, "档三确实判了")

# --- 7.5 判分器对白名单的处理（兼容旧作答）
_legacy = {"found_params": {"paper": ["learning_rate"], "code": []},
           "inconsistencies": [{"param_key": "learning_rate", "verified": True}],
           "selfcheck": {"passed": 1, "checked": 1}}
_ls = scoring.evaluate_answer(expect=["learning_rate"], decoy=[], answer=_legacy)
check(_ls["precision"] == 1.0, "没声明白名单 → 按全量算（旧作答兼容）")

_gated = dict(_legacy, metrics_available=["recall", "n_findings", "parse_ok"])
_gs = scoring.evaluate_answer(expect=["learning_rate"], decoy=[], answer=_gated)
check("precision" not in _gs, "声明不含 precision → 不产出 precision")
check("selfcheck_rate" not in _gs, "声明不含 selfcheck_rate → 不产出")
check(_gs["recall"] == 1.0, "声明含 recall → 照常产出")

_unjudged = {"found_params": {"paper": ["learning_rate"], "code": []},
             "inconsistencies": [{"param_key": "learning_rate", "verified": None}],
             "selfcheck": {"passed": 0, "checked": 0, "skipped": True},
             "metrics_available": list(ALL_METRICS)}
_us = scoring.evaluate_answer(expect=["learning_rate"], decoy=[], answer=_unjudged)
check(_us["precision"] == 0.0,
      "verified=None 不算通过（未判 ≠ 判过；靠白名单不出数，而不是靠 0 分）")
check(_us["judged"] == 0.0, "verified=None 不计入已判条数")

# ═══════════════════════════════════════ 收尾

print("\n" + "=" * 62)
if FAILURES:
    print(f"未通过 {len(FAILURES)}/{CHECKS}:")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print(f"全部通过（{CHECKS} 项）")
sys.exit(0)
