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

# ═══════════════════════════════════════ 收尾

print("\n" + "=" * 62)
if FAILURES:
    print(f"未通过 {len(FAILURES)}/{CHECKS}:")
    for f in FAILURES:
        print("   -", f)
    sys.exit(1)
print(f"全部通过（{CHECKS} 项）")
sys.exit(0)
