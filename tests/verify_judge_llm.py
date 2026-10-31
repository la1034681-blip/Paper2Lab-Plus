"""模型先判、程序后验 验证（2026-10-25）。

守的契约:
  A 默认关: `Job.judge_llm` 默认 False; 关着时一次都不调用（零成本、结论可复现）
  B 封闭集合: 假设判决三选一、参数状态七选一, 且与 derivation / Status 同源派生（不手抄）
  C 不合规就丢: id 越界 / 判决不在集合 / 不是 dict —— 一律丢弃, 不"猜一个最近的"
  D 防锚定: 喂给模型的条目里**不含** verdict / status / derivation（带着答案问等于让它抄）
  E 不覆盖: 规则结论 verdict/status 永远由规则写, 模型的话只进 llm_* 三格
  F 三态: llm_agree = True（一致）/ False（不一致）/ None（没比成：模型没判或规则没结论）
  G 顺序: 模型判在规则之前产生（`_run_hypothesize` / `_run_align` 的调用序）
  H 契约不破: Hypothesis / Finding 的 to_dict 老键一个不少, 只加三个新键（L8）
  I 往返: Finding.to_dict → from_dict 后 llm_* 仍在, 枚举不丢
  J 不造数: apply_* 只在有 judged 结果时写, 拿不到就留空并记 unjudged
  K 人话总结: summary() 必须同时说出「一致几条」和「不一致几条」, 不许只报一个
  L 端到端: demo 材料真跑一遍, 关着时全空、开着时两边都留

为什么要有这一项:
  「先判后验」最坏的结局是**模型定调、程序背书** —— 程序只复核模型说的那一项,
  模型说"一致"就只查一致性, 于是"两侧都抄错了"永远查不出来。
  所以这里守的不是"能不能判", 而是"程序那步有没有独立跑、结论有没有被覆盖"。

用法: .venv\\Scripts\\python.exe tests/verify_judge_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import judge_llm as J                                        # noqa: E402
from paper2lab.derivation import (DERIVE_EXCLUDED, DERIVE_FLAGGED,         # noqa: E402
                                  DERIVE_KEEP)
from paper2lab.hypothesis import Hypothesis, build_hypotheses               # noqa: E402
from paper2lab.models import (CodeInfo, Evidence, Finding, PaperInfo,       # noqa: E402
                              Status)
from paper2lab.toolbox import Job                                           # noqa: E402

CHECKS = 0
FAILURES = []

HYP_OLD_KEYS = ("key", "kind", "claim", "question", "falsifiable_by", "score", "why",
                "paper_value", "code_value", "paper_where", "code_where",
                "needs_rerun", "rank", "verdict", "derivation", "derivation_kind",
                "engine", "impact")
FIND_OLD_KEYS = ("param_key", "display_name", "category", "status", "confidence",
                 "risk", "paper", "code", "extra", "ai_analysis", "note",
                 "risk_reason", "verified", "verify_note")


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


def ev(raw, *, source="code", loc="train.py:1"):
    return Evidence(source=source, location=loc, snippet=f"= {raw}", raw_value=raw)


def card(key="learning_rate", pv="1e-4", cv="3e-4", verdict=""):
    c = Hypothesis(key=key, kind="取值冲突",
                   claim=f"论文写 {pv}、代码写 {cv} —— 需要定出哪个真正生效",
                   question="哪个是真的", falsifiable_by="跑一次看 loss 是否分叉",
                   paper_value=pv, code_value=cv,
                   paper_where="p3", code_where="train.py:12")
    c.verdict = verdict
    return c


def finding(key="learning_rate", status=Status.INCONSISTENT):
    return Finding(param_key=key, display_name="Learning Rate",
                   category=__import__("paper2lab.models", fromlist=["Category"]).Category.PARAM,
                   status=status,
                   confidence=__import__("paper2lab.models",
                                         fromlist=["Confidence"]).Confidence.CONFIRMED,
                   risk=__import__("paper2lab.models", fromlist=["Risk"]).Risk.HIGH)


# ---------------------------------------------------------------- A 默认关

def test_default_off():
    check(Job().judge_llm is False, "A1 Job.judge_llm 默认 False（线上零成本）")
    check(Job().judge_hyp_stat == {} and Job().judge_find_stat == {},
          "A2 产物字段默认空（关着时界面不出现这一节）")
    called = []

    def _ask(prompt):
        called.append(prompt)
        return {"judgments": [{"id": 1, "verdict": DERIVE_KEEP, "why": "x"}]}

    # 关着时调用方压根不该进这个分支 —— 用 available 的语义验一遍
    check(J.available(api_key="", force_mock=False) is False,
          "A3 没配 Key 时通道不可用（不假装判过）")
    check(J.available(api_key="sk-x", force_mock=True) is False,
          "A4 强制离线时通道不可用（force_mock 优先于 Key）")
    check(J.available(api_key="sk-x") is True, "A5 有 Key 且未离线时可用")


# ---------------------------------------------------------------- B 封闭集合

def test_closed_sets():
    check(set(J.JUDGE_VERDICTS) == {DERIVE_KEEP, DERIVE_EXCLUDED, DERIVE_FLAGGED},
          "B1 假设判决集合与 derivation 三个常量同源")
    check(len(J.JUDGE_STATUSES) == len(Status),
          "B2 参数状态集合与 Status 成员数一致（同源派生，不手抄）")
    check(set(J.JUDGE_STATUSES) == set(__import__("paper2lab.llm", fromlist=["_STATUS_ZH"])._STATUS_ZH.values()),
          "B3 参数状态就是 llm.py 那张中文标签表（全项目只有一份）")
    check(all(J._STATUS_BY_ZH.get(v) is not None for v in J.JUDGE_STATUSES),
          "B4 中文标签能反查回 Status（回填要用）")


# ---------------------------------------------------------------- C 不合规就丢

def test_parse_strict():
    ok = {"judgments": [{"id": 1, "verdict": DERIVE_KEEP, "why": "论文 1e-4"},
                        {"id": 2, "verdict": "看起来还行", "why": "自创词"},
                        {"id": 99, "verdict": DERIVE_KEEP, "why": "越界"},
                        {"id": "x", "verdict": DERIVE_KEEP, "why": "id 不是数"},
                        "不是字典"]}
    rows, dropped = J._parse(ok, 3, J.JUDGE_VERDICTS)
    check(len(rows) == 1 and dropped == 4, "C1 四类不合规输出全丢，只留合法那条")
    check(J._parse(None, 3, J.JUDGE_VERDICTS) == ([], 0), "C2 返回非 dict 时不炸、不造")
    check(J._parse({"judgments": "oops"}, 3, J.JUDGE_VERDICTS) == ([], 0),
          "C3 judgments 不是列表时不炸")
    long_why = "长" * 500
    rows2, _ = J._parse({"judgments": [{"id": 1, "verdict": DERIVE_KEEP, "why": long_why}]},
                        1, J.JUDGE_VERDICTS)
    check(len(rows2[0]["why"]) == J.WHY_MAX, "C4 理由超长就截断（不丢条目）")


# ---------------------------------------------------------------- D 防锚定

def test_no_anchor():
    c = card(verdict=DERIVE_EXCLUDED)          # 规则已判「已排除」
    item = J._hyp_item(1, c)
    keys = set(item.keys())
    check("verdict" not in keys and "derivation" not in keys,
          "D1 喂给模型的条目里没有规则结论（防锚定）")
    check("claim" in keys and "推翻判据" in keys, "D2 给的是事实与可证伪判据")

    f = finding()
    fi = J._find_item(1, f)
    check("status" not in fi and "risk" not in fi,
          "D3 Finding 的条目里没有 status/risk（那是答案）")


# ---------------------------------------------------------------- E/F 不覆盖 + 三态

def test_no_override():
    c1 = card("learning_rate", verdict=DERIVE_KEEP)
    c2 = card("batch_size", pv="4096", cv="512", verdict=DERIVE_EXCLUDED)
    c3 = card("epochs", pv="300", cv="300", verdict="")      # 规则没给结论
    cards = [c1, c2, c3]
    # ⚠ 「规则没结论」与「模型没判」是两件事, 分别对应 rule_empty 与 unjudged。
    #   这里刻意三条都被模型判了, 才能单独验 rule_empty。
    judged = {"learning_rate": {"verdict": DERIVE_KEEP, "why": "两边都是 1e-4"},
              "batch_size": {"verdict": DERIVE_KEEP, "why": "写法不同"},
              "epochs": {"verdict": DERIVE_KEEP, "why": "两侧都是 300"}}
    stat = J.apply_hypotheses(cards, judged)

    check(c1.verdict == DERIVE_KEEP and c1.llm_agree is True,
          "E1 两边一致 → 规则结论不动, llm_agree=True")
    check(c2.verdict == DERIVE_EXCLUDED and c2.llm_verdict == DERIVE_KEEP
          and c2.llm_agree is False,
          "E2 两边不一致 → **规则结论仍是规则给的**, 模型那句并存（不覆盖）")
    check(c2.llm_why == "写法不同", "E3 模型的理由如实留着（报告要两边都写）")
    check(c3.verdict == "" and c3.llm_agree is None,
          "F1 规则没结论时 llm_agree=None —— 模型的话只能当推测")
    check(stat["judged"] == 3 and stat["agree"] == 1 and stat["disagree"] == 1
          and stat["rule_empty"] == 1 and stat["unjudged"] == 0,
          "F2 统计五项各自算对（judged/agree/disagree/rule_empty/unjudged）")
    # 再验一遍 unjudged 与 rule_empty 不串味: 一条都不判时 unjudged 涨、rule_empty 不涨
    stat2 = J.apply_hypotheses([card("dropout", "0.1", "0.2", verdict="")], {})
    check(stat2["unjudged"] == 1 and stat2["rule_empty"] == 0,
          "F3 「模型没判」只进 unjudged, 不冒充 rule_empty")

    f1 = finding("learning_rate", Status.INCONSISTENT)
    f2 = finding("batch_size", Status.CONSISTENT)
    fstat = J.apply_findings([f1, f2], {"learning_rate": {"status": Status.CONSISTENT, "why": "y"},
                                        "batch_size": {"status": Status.CONSISTENT, "why": "z"}})
    check(f1.status == Status.INCONSISTENT and f1.llm_status == Status.CONSISTENT
          and f1.llm_agree is False,
          "E4 Finding 的 status 同样不被覆盖（下游全依赖它）")
    check(f2.llm_agree is True and fstat["agree"] == 1 and fstat["disagree"] == 1,
          "E5 Finding 一遍一不一致，统计对得上")


# ---------------------------------------------------------------- G 顺序

def test_order():
    """模型判必须在规则之前被调用 —— 用注入的 ask 记录"被调时规则结论还在不在"。"""
    seen = {}

    def _ask(prompt):
        # 模型被问的那一刻，卡上还不该有 verdict（否则顺序反了）
        seen["verdict_at_ask"] = [c.verdict for c in seen["cards"]]
        return {"judgments": [{"id": i + 1, "verdict": DERIVE_KEEP, "why": "看两侧原文"}
                              for i in range(len(seen["cards"]))]}

    cards = [card("learning_rate"), card("batch_size", "4096", "512")]
    seen["cards"] = cards
    judged, note = J.judge_hypotheses(cards, api_key="sk-x", ask=_ask)
    check(seen.get("verdict_at_ask") == ["", ""],
          "G1 模型被问的时候规则还没判（先判后验的顺序成立）")
    check(len(judged) == 2 and "测试注入" not in note,
          "G2 注入通道能判出两条")


# ---------------------------------------------------------------- H/I 契约与往返

def test_contract():
    c = card()
    d = c.to_dict()
    missing = [k for k in HYP_OLD_KEYS if k not in d]
    check(not missing, "H1 Hypothesis.to_dict 老键一个不少（缺: %s）" % (missing or "无"))
    check(all(k in d for k in ("llm_verdict", "llm_why", "llm_agree")),
          "H2 只加三个新键")

    f = finding()
    fd = f.to_dict()
    missing_f = [k for k in FIND_OLD_KEYS if k not in fd]
    check(not missing_f, "H3 Finding.to_dict 老键一个不少（缺: %s）" % (missing_f or "无"))
    check(fd.get("llm_status") == "" and fd.get("llm_agree") is None,
          "H4 未判时 llm_status 序列化成空串（不是 None）")

    f.llm_status = Status.CONSISTENT
    f.llm_why = "两边等价"
    f.llm_agree = True
    back = Finding.from_dict(f.to_dict())
    check(back.llm_status == Status.CONSISTENT and back.llm_agree is True,
          "I1 Finding 往返后 llm_* 仍在, 枚举不丢")
    check(back.status == f.status and back.risk == f.risk, "I2 往返后老字段不漂")


# ---------------------------------------------------------------- J 不造数

def test_no_invention():
    c = card(verdict=DERIVE_KEEP)
    stat = J.apply_hypotheses([c], {})
    check(c.llm_verdict == "" and c.llm_agree is None,
          "J1 没有模型判断时不写假话（留空）")
    check(stat["unjudged"] == 1 and stat["judged"] == 0, "J2 如实记「模型没判」")

    obj, note = J.judge_hypotheses([], api_key="sk-x")
    check(obj == {} and "没有待判" in note, "J3 没有卡时不调模型")
    obj2, note2 = J.judge_hypotheses([card()], api_key="", force_mock=False)
    check(obj2 == {} and "未启用" in note2, "J4 没 Key 时如实说没判，不编")


# ---------------------------------------------------------------- K 人话总结

def test_summary():
    s = J.summary({"judged": 5, "agree": 3, "disagree": 2, "rule_empty": 1, "unjudged": 0})
    check("一致 3 条" in s and "不一致 2 条" in s, "K1 同时报一致与不一致（不许只报一个）")
    check("推测" in s, "K2 规则没结论的那几条如实标成推测")
    check("没有参与" in J.summary({}), "K3 没参与时明说没参与")


# ---------------------------------------------------------------- L 端到端

def test_end_to_end():
    paper = PaperInfo(params={"learning_rate": ev("1e-4", source="paper", loc="p3"),
                              "batch_size": ev("4096", source="paper", loc="p3")})
    code = CodeInfo(params={"learning_rate": [ev("3e-4")], "batch_size": [ev("512")]})
    pairs = J.pairs_from_materials(paper, code)
    check(len(pairs) == 2 and {p.param_key for p in pairs} == {"learning_rate", "batch_size"},
          "L1 从两侧材料能组出配对（不依赖 Finding）")
    check(all(p.display_name for p in pairs), "L2 配对带展示名（界面用）")

    cards = build_hypotheses(paper, code)
    check(len(cards) >= 2, "L3 demo 材料能提出假设卡")

    # 关着时全空
    job_off = Job()
    check(job_off.judge_hyp_stat == {} and job_off.judge_find_stat == {},
          "L4 关着时两个统计都是空的")

    # 开着但没 Key：如实记"没判"，不报错
    job_on = Job(judge_llm=True, api_key="")
    from paper2lab.judge_llm import apply_hypotheses, judge_hypotheses
    judged, note = judge_hypotheses(cards, api_key="", force_mock=False)
    stat = apply_hypotheses(cards, judged)
    check(judged == {} and stat["unjudged"] == len(cards),
          "L5 开着但没 Key → 一条都没判, 且如实计数")


# ---------------------------------------------------------------- M 真调用路径上的顺序

def test_wired_order():
    """在 `toolbox` 的真实调用路径上验顺序。

    为什么必须在这里验一遍: 模块层的顺序对不等于**接线**对 ——
    以后有人图省事把 `judge_hypotheses(...)` 挪到 `derive(...)` 后面，
    E 段那些断言一个都不会红（它们只测 apply_*），但"先判后验"这个口径就没了。
    """
    from paper2lab import toolbox as TB

    paper = PaperInfo(params={"learning_rate": ev("1e-4", source="paper", loc="p3")})
    code = CodeInfo(params={"learning_rate": [ev("3e-4")]})

    seen: dict = {}
    orig_h, orig_f = J.judge_hypotheses, J.judge_findings

    def fake_h(cards, **kw):
        seen["verdicts_at_ask"] = [c.verdict for c in cards]
        return {}, "测试：不想真调模型"

    def fake_f(pairs, **kw):
        seen["pairs_at_ask"] = list(pairs)
        return {}, "测试：不想真调模型"

    J.judge_hypotheses, J.judge_findings = fake_h, fake_f
    try:
        # ① 假设站
        job = Job(judge_llm=True, api_key="sk-x")
        job.paper, job.code = paper, code
        TB._run_hypothesize(job, lambda *_: None)
        check(seen.get("verdicts_at_ask") and all(v == "" for v in seen["verdicts_at_ask"]),
              "M1 假设站：模型被问时规则还没判（顺序没被改反）")
        check(all(c.verdict for c in (job.all_hypotheses or [])),
              "M2 假设站：跑完之后规则结论照样在（模型没覆盖）")
        check(job.judge_hyp_note.startswith("测试"), "M3 假设站：模型没判时的说明如实记着")

        # ③ 参数配对
        job2 = Job(judge_llm=True, api_key="sk-x")
        job2.paper, job2.code = paper, code
        TB._run_align(job2, lambda *_: None)
        check(bool(seen.get("pairs_at_ask")), "M4 配对站：配对在 align 之前就组好了")
        check(job2.audit is not None and len(job2.audit.findings) >= 1,
              "M5 配对站：规则判定照样跑完（不因模型没判就少跑）")
        check(job2.judge_find_stat.get("unjudged", 0) >= 1,
              "M6 配对站：如实记「模型没判」的条数")
    finally:
        J.judge_hypotheses, J.judge_findings = orig_h, orig_f

    # 关着时两条路径一次都不该调
    seen.clear()
    J.judge_hypotheses, J.judge_findings = fake_h, fake_f
    try:
        job3 = Job(judge_llm=False)
        job3.paper, job3.code = paper, code
        TB._run_hypothesize(job3, lambda *_: None)
        TB._run_align(job3, lambda *_: None)
        check(not seen, "M7 关着时两条路径一次都不调（零成本）")
        check(job3.judge_hyp_stat == {} and job3.judge_find_stat == {},
              "M8 关着时统计为空，界面不会出现这一节")
    finally:
        J.judge_hypotheses, J.judge_findings = orig_h, orig_f


def main():
    print("=" * 62)
    print("模型先判、程序后验 —— 验证")
    print("=" * 62)
    test_default_off()
    test_closed_sets()
    test_parse_strict()
    test_no_anchor()
    test_no_override()
    test_order()
    test_contract()
    test_no_invention()
    test_summary()
    test_end_to_end()
    test_wired_order()

    print("\n" + "=" * 62)
    if FAILURES:
        print("未通过 %d / %d 项：" % (len(FAILURES), CHECKS))
        for f in FAILURES:
            print("  - " + f)
        print("=" * 62)
        sys.exit(1)
    print("全部通过：%d 项" % CHECKS)
    print("=" * 62)


if __name__ == "__main__":
    main()
