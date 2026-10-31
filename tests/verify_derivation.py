"""阶段二 2B 验证 —— 符号推演与筛排：能推的推干净，推不了的如实说推不了。

守的契约:
  A 常量与引擎: 三个判决互不相同、三类推演互不相同、引擎名非空
  B 约束检查: 一侧取值非法 → 判「已定位」（比"冲突"更硬, 且不用跑实验）
  C 倍数推演: 材料里有解释量 → 「已排除」; 没有 → 保持「待验」但必须给出方向
  D 恒等验证: 公式复算"换算后一致" → 「已排除」; 其余一律保持「待验」不许替用户下结论
  E 硬度排序: 值非法 > 公式复算 > 倍数像什么（排反会出现"已排除的卡被写成仍待验"）
  F 单位展开: `10k` / `1M` / `3万` 能展开; 别的字符串一个字符都不许动
  G 不造数: 推演只加新字段, 不改 claim / score / paper_value / code_value / needs_rerun
  H 契约不破: to_dict 旧键一个不少, 新键齐全（L8: 只加不改）
  I 筛排: 已定位在前、已排除在后、静态可判的先于需实机的; rank 连续
  J 不阻断: 关系式模块整个炸掉时, 假设卡照样发得出来（宁可这次不推, 不能不发卡）
  K 纯规则: 不 import 模型 / 不依赖 streamlit / 不 import hypothesis（不成环）
  L 端到端: demo 材料真跑一遍, 每张卡都有判决与引擎, 统计数字自洽

为什么要有这一项:
  2B 最危险的失败模式**不是报错, 是空转** —— 文件写得很像样, 规则一条也没命中。
  2026-10-15 实测踩到过: 第一版在 5 个真实样本 43 张卡上**零命中**,
  因为判据太窄（只在材料里已有并行度参数时才说话）。
  所以 C/G/J/L 四组是重点, 且**大部分用合成用例** —— 真实材料里能推的太少,
  光靠真样本根本测不到「已排除」「已定位」这两条路径。
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import derivation as D                                    # noqa: E402
from paper2lab.derivation import (DERIVE_EXCLUDED, DERIVE_FLAGGED,        # noqa: E402
                                  DERIVE_KEEP, KIND_CONSTRAINT, KIND_IDENTITY,
                                  KIND_SCALE, derive, engine_name, summarize)
from paper2lab.hypothesis import build_hypotheses                         # noqa: E402
from paper2lab.models import CodeInfo, Evidence, PaperInfo                # noqa: E402

CHECKS = 0
FAILURES = []

OLD_KEYS = ("key", "kind", "kind_desc", "claim", "question", "falsifiable_by",
            "score", "why", "paper_value", "code_value", "paper_where",
            "code_where", "needs_rerun", "rank")
NEW_KEYS = ("verdict", "derivation", "derivation_kind", "engine",
            "impact",   # 阶段二 2E: 影响量级（提示层, 只加不改）
            "llm_verdict", "llm_why", "llm_agree")   # 2026-10-25: 模型判断（只加不改）


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


def P(**params):
    return PaperInfo(params=dict(params))


def C(**params):
    return CodeInfo(params=dict(params))


def key_of(cards, key):
    return next((c for c in cards if c.key == key), None)


# ---------------------------------------------------------------- A 常量与引擎

def group_a():
    print("\nA 常量与引擎")
    verdicts = {DERIVE_KEEP, DERIVE_EXCLUDED, DERIVE_FLAGGED}
    check(len(verdicts) == 3, "三个判决常量互不相同")
    kinds = {KIND_CONSTRAINT, KIND_SCALE, KIND_IDENTITY}
    check(len(kinds) == 3, "三类推演常量互不相同")
    check(bool(engine_name().strip()), "引擎名非空（界面/报告要如实显示用的什么引擎）")
    check("sympy" in engine_name() or "内置" in engine_name(),
          "引擎名说清是内置还是第三方（不写含糊的'符号引擎'）")


# ---------------------------------------------------------------- B 约束检查

def group_b():
    print("\nB 约束检查（值本身非法 → 已定位）")
    paper, code = P(dropout=ev("0.5", source="paper")), C(dropout=[ev("1.5")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "dropout")
    check(c is not None and c.is_flagged, "论文 0.5 / 代码 1.5 → dropout 判「已定位」")
    check(c is not None and "超出" in c.derivation, "推演文案里写清了超出哪个区间")
    check(c is not None and c.derivation_kind == KIND_CONSTRAINT, "推演类型标「约束检查」")
    check(c is not None and c.needs_rerun is False, "取值非法属静态可判, 不该标「需实机」")

    paper, code = P(learning_rate=ev("1e-4", source="paper")), C(learning_rate=[ev("2.0")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "learning_rate")
    check(c is not None and c.is_flagged, "学习率 2.0 超上界 → 也判「已定位」")

    paper, code = P(dropout=ev("0.1", source="paper")), C(dropout=[ev("0.5")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "dropout")
    check(c is not None and not c.is_flagged, "两个合法值不许判「已定位」（判错比不说更糟）")


# ---------------------------------------------------------------- C 倍数推演

def group_c():
    print("\nC 倍数推演（能解释→排除; 不能解释→给方向但不排除）")
    # 材料里恰好有并行度参数, 倍数正好对得上 → 已排除
    paper = P(batch_size=ev("32", source="paper"), world_size=ev("8", source="paper"))
    code = C(batch_size=[ev("256")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and c.is_excluded, "批 32 vs 256 + 材料有 world_size=8 → 「已排除」")
    check(c is not None and "world_size" in c.derivation, "推演文案里点出了是哪个量解释的")
    check(c is not None and c.derivation_kind == KIND_SCALE, "推演类型标「倍数推演」")

    # 梯度累积同样能解释
    paper = P(batch_size=ev("64", source="paper"), gradient_accumulation_steps=ev("4", source="paper"))
    code = C(batch_size=[ev("256")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and c.is_excluded, "梯度累积步数 4 也能解释 64→256 的倍数")

    # 材料里没有解释量 → 保持待验, 但必须给出方向
    paper, code = P(batch_size=ev("32", source="paper")), C(batch_size=[ev("256")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and c.verdict == DERIVE_KEEP, "找不到解释量时**不**判「已排除」")
    check(c is not None and "2 的 3 次幂" in c.derivation, "但点出了 8 倍是 2 的幂（给方向）")
    check(c is not None and "仍待验" in c.derivation, "并明说这条仍待验（不越权下结论）")

    # 10 的幂 → 量纲方向
    paper, code = P(batch_size=ev("8", source="paper")), C(batch_size=[ev("800")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and "10 的" in c.derivation, "100 倍点出是 10 的幂（疑似量纲）")

    # 学习率不属于批大小类, 不许拿卡数去解释
    paper = P(learning_rate=ev("0.01", source="paper"), world_size=ev("3", source="paper"))
    code = C(learning_rate=[ev("0.03")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "learning_rate")
    check(c is not None and not c.is_excluded, "学习率差 3 倍不许用卡数解释掉（口径不可乱套）")

    # 差得不多 → 不写诊断（免得清单全是噪音）
    paper, code = P(batch_size=ev("32", source="paper")), C(batch_size=[ev("33")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and c.verdict == DERIVE_KEEP, "只差 1 点几倍 → 不写倍数诊断")
    check(c is not None and not c.derivation, "也不留半句话占版面")

    # 方向不许说反 —— 这条**真写反过**: 报告里出现过
    # "论文 1e-4 / 代码 0.001，论文侧更大"（1e-4 明明更小）。
    paper, code = P(batch_size=ev("4", source="paper")), C(batch_size=[ev("400")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and "代码侧更大" in c.derivation,
          "说方向: 代码值更大时写「代码侧更大」")
    paper, code = P(batch_size=ev("400", source="paper")), C(batch_size=[ev("4")])
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "batch_size")
    check(c is not None and "论文侧更大" in c.derivation,
          "说方向: 论文值更大时写「论文侧更大」")


# ---------------------------------------------------------------- D 恒等验证

def group_d():
    print("\nD 恒等验证（公式复算；不许替用户下结论）")
    # warmup: 一侧比例、一侧步数, 换算后正好一致 → 已排除
    paper, code = (P(warmup=ev("0.05", source="paper"), iterations=ev("10000", source="paper")),
                   C(warmup=[ev("500")]))
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "warmup")
    check(c is not None and c.is_excluded, "warmup 500 步 ÷ 10000 = 0.05 → 与比例侧一致 → 「已排除」")
    check(c is not None and c.derivation_kind == KIND_IDENTITY, "推演类型标「恒等验证」")
    check(c is not None and "公式" in c.derivation, "文案里写出了用的是哪条公式")

    # 换算不上 → 必须保持待验（关系式自己写着"需人工确认"）
    paper, code = (P(warmup=ev("0.05", source="paper"), iterations=ev("10000", source="paper")),
                   C(warmup=[ev("900")]))
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "warmup")
    check(c is not None and c.verdict == DERIVE_KEEP, "复算对不上时保持「待验」（不替用户下结论）")

    # lr 与 batch 同步放大 → 关系式给 hint, 也一律保持待验
    paper, code = (P(learning_rate=ev("0.01", source="paper"), batch_size=ev("32", source="paper")),
                   C(learning_rate=[ev("0.02")], batch_size=[ev("64")]))
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "learning_rate")
    check(c is not None and c.verdict == DERIVE_KEEP,
          "lr 与 batch 同步缩放（可能是刻意的）→ 仍标待验, 不替用户拍板")


# ---------------------------------------------------------------- E 硬度排序

def group_e():
    print("\nE 硬度排序（结论硬的先出）")
    # warmup 这条：倍数推演会想抢（10000 倍），但公式复算更硬, 必须由它定案
    paper, code = (P(warmup=ev("0.05", source="paper"), iterations=ev("10000", source="paper")),
                   C(warmup=[ev("500")]))
    cards = derive(build_hypotheses(paper, code), paper, code)
    c = key_of(cards, "warmup")
    check(c is not None and c.derivation_kind == KIND_IDENTITY,
          "公式能算的, 不许被倍数推演抢走（否则已排除会被写成仍待验）")


# ---------------------------------------------------------------- F 单位展开

def group_f():
    print("\nF 单位展开（10k / 1M / 3万）")
    out = D._expand_units({"a": "10k", "b": "1M", "c": "abc", "d": "0.05",
                           "e": "3万", "f": "1.5M", "g": "k", "h": "learning_rate"})
    check(float(out["a"]) == 10000, "10k → 10000")
    check(float(out["b"]) == 1e6, "1M → 1000000")
    check(out["c"] == "abc", "普通字符串一个字符都不动")
    check(out["d"] == "0.05", "纯数字不动")
    check(float(out["e"]) == 30000, "3万 → 30000")
    check(float(out["f"]) == 1.5e6, "1.5M → 1500000")
    check(out["g"] == "k", "单个字母不动（可能只是个变量名）")
    check(out["h"] == "learning_rate", "键名带下划线也不会被误展开")
    check(D._expand_units({}) == {}, "空表安全")


# ---------------------------------------------------------------- G 不造数

def group_g():
    print("\nG 不造数（推演只加字段, 不改内容）")
    paper, code = P(dropout=ev("0.5", source="paper")), C(dropout=[ev("1.5")])
    before = build_hypotheses(paper, code)
    snap = [(c.key, c.claim, c.score, c.paper_value, c.code_value, c.needs_rerun,
             c.falsifiable_by) for c in before]
    after = derive(before, paper, code)
    now = [(c.key, c.claim, c.score, c.paper_value, c.code_value, c.needs_rerun,
            c.falsifiable_by) for c in after]
    check(snap == now, "推演前后: 陈述/分数/两侧取值/判据/需实机 一字未改")
    check(all(c.verdict for c in after), "推演后每张卡都拿到了判决")
    check(all(c.engine for c in after), "每张卡都标了用的什么引擎")
    check(summarize(after)["engine"] == engine_name(), "统计里的引擎名与卡片上的一致")


# ---------------------------------------------------------------- H 契约不破

def group_h():
    print("\nH 契约不破（to_dict 只加不改）")
    paper, code = P(batch_size=ev("32", source="paper")), C(batch_size=[ev("256")])
    card = build_hypotheses(paper, code)[0]
    derive([card], paper, code)
    d = card.to_dict()
    missing = [k for k in OLD_KEYS if k not in d]
    check(not missing, f"旧键一个不少（缺: {missing}）")
    missing_new = [k for k in NEW_KEYS if k not in d]
    check(not missing_new, f"新键齐全（缺: {missing_new}）")
    check(d["verdict"] == card.verdict, "字典里的判决与卡片上的同源")
    check(set(d) == set(OLD_KEYS) | set(NEW_KEYS), "键集合 = 旧键 ∪ 新键（没有偷偷多出来的）")


# ---------------------------------------------------------------- I 筛排

def group_i():
    print("\nI 筛排（分层 + rank 连续）")
    paper = P(dropout=ev("0.5", source="paper"), batch_size=ev("32", source="paper"),
              model=ev("ResNet-18", source="paper"), world_size=ev("8", source="paper"))
    code = C(dropout=[ev("1.5")], batch_size=[ev("256")])
    cards = derive(build_hypotheses(paper, code), paper, code)

    check(cards[0].is_flagged, "「已定位」排在最前（最硬且不用动手）")
    ex = [i for i, c in enumerate(cards) if c.is_excluded]
    other = [i for i, c in enumerate(cards) if not c.is_excluded]
    check(not ex or not other or min(ex) > max(other), "「已排除」全部排在非排除项之后")
    check([c.rank for c in cards] == list(range(1, len(cards) + 1)), "rank 连续, 从 1 开始")

    # 静态可判的待验 必须排在 需实机的待验 前面（② 站少排活）
    pend = [c for c in cards if c.verdict == DERIVE_KEEP]
    static = [i for i, c in enumerate(pend) if not c.needs_rerun]
    real = [i for i, c in enumerate(pend) if c.needs_rerun]
    check(not static or not real or max(static) < min(real),
          "同是待验: 读代码就能定的排在需要真跑的之前")

    check(summarize(cards)["flagged"] + summarize(cards)["excluded"]
          + summarize(cards)["pending"] == summarize(cards)["total"],
          "统计三类相加 = 总数")


# ---------------------------------------------------------------- J 不阻断

def group_j():
    print("\nJ 不阻断（推演炸了也要发得出卡）")
    import paper2lab.compute as COMP
    paper, code = P(batch_size=ev("32", source="paper")), C(batch_size=[ev("256")])
    orig = COMP.check_relations

    def boom(*a, **k):
        raise RuntimeError("模拟关系式模块故障")

    try:
        COMP.check_relations = boom
        cards = derive(build_hypotheses(paper, code), paper, code)
        check(len(cards) > 0, "关系式整个炸掉时, 卡片照样发出来")
        check(all(c.verdict for c in cards), "没推演出结论也要给判决（不许留空白）")
        check(all(c.engine for c in cards), "引擎名照样标（不静默）")
    finally:
        COMP.check_relations = orig

    cards2 = derive(build_hypotheses(paper, code), paper, code)
    check(key_of(cards2, "batch_size") is not None, "恢复后功能正常")


# ---------------------------------------------------------------- K 纯规则

def group_k():
    print("\nK 纯规则（不依赖模型 / 界面 / 上游模块）")
    src = (ROOT / "paper2lab" / "derivation.py").read_text(encoding="utf-8")
    check(not re.search(r"^\s*(from|import)\s+streamlit", src, re.M), "不依赖 streamlit")
    check(not re.search(r"^\s*(from|import)\s+\.llm", src, re.M), "不 import 模型（纯规则零成本）")
    check(not re.search(r"^\s*(from|import)\s+\.hypothesis", src, re.M),
          "不 import hypothesis（否则成环: hypothesis 要 import 本模块取常量）")
    check("open(" not in src and "Path(" not in src, "不碰任何文件")
    check(derivation_is_pure(), "模块顶层不 import compute/collider（延迟到调用时）")


def derivation_is_pure():
    # 只查**顶层** import（行首紧跟 from/import）: 函数内的延迟 import 是设计的一部分,
    # 用 `^\s*` 会把缩进行也算进去, 断言就会永远红。
    src = (ROOT / "paper2lab" / "derivation.py").read_text(encoding="utf-8")
    return not re.search(r"^from\s+\.(compute|collider|synonyms|aligner|selfcheck)\s+import",
                         src, re.M)


# ---------------------------------------------------------------- L 端到端

def group_l():
    print("\nL 端到端（demo 材料真跑）")
    pdf, zip_ = ROOT / "demo" / "paper.pdf", ROOT / "demo" / "project.zip"
    if not pdf.exists() or not zip_.exists():
        check(False, "demo 材料缺失, 无法端到端验证")
        return
    from paper2lab.code_analyzer import analyze_code
    from paper2lab.paper_parser import parse_paper

    paper, code = parse_paper(str(pdf)), analyze_code(str(zip_))
    cards = build_hypotheses(paper, code)
    cards = derive(cards, paper, code)
    check(len(cards) > 0, f"demo 提出了 {len(cards)} 张卡")
    check(all(c.verdict for c in cards), "每张卡都有判决")
    check(all(c.engine for c in cards), "每张卡都有引擎名")
    check(all(c.rank >= 1 for c in cards), "rank 都回填了")
    stats = summarize(cards)
    check(stats["total"] == len(cards), "统计总数 = 卡片数")
    check(stats["flagged"] + stats["excluded"] + stats["pending"] == len(cards),
          "三类相加 = 总数（界面照这个说）")
    check(all(c.derivation_kind in ("", KIND_CONSTRAINT, KIND_SCALE, KIND_IDENTITY)
              for c in cards), "推演类型只能是三类之一或空")
    bad = [c.key for c in cards if c.derivation and not c.derivation_kind]
    check(not bad, f"有推演文案就必有类型（缺: {bad}）")


def main():
    print("=" * 68)
    print("阶段二 2B 验证 —— 符号推演与筛排")
    print("=" * 68)
    for fn in (group_a, group_b, group_c, group_d, group_e,
               group_f, group_g, group_h, group_i, group_j, group_k, group_l):
        fn()
    print("\n" + "=" * 68)
    print(f"共 {CHECKS} 项, 未过 {len(FAILURES)} 项")
    if FAILURES:
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("全部通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
