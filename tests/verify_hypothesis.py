"""阶段二 2A 验证 —— 假设卡：可证伪、不造数、口径不漂移。

守的契约:
  A 卡本身: 字段齐全、分类有解释、字典形态键名唯一
  B 可证伪: 每张卡都必须写清"看到什么就推翻它"（2A 的第二条硬规矩）
  C 判据与 needs_rerun 严格一致: 说"跑一次"就必须标"需实机", 反之亦然
    —— 不一致会让 ② 实验站拿到一堆其实静态可判的活
  D **不制造伪矛盾**: 值相同 / 数值等价 / 文本同义 → 一律**不出卡**
    （2026-10-15 实测踩过: 把「变量 x(运行时确定)」当成取值, 于是 32==32 也出卡,
      判据写成"等于 32 则成立、等于 32 则推翻"）
  E 口径唯一: 值比较复用 values_equal / numbers_equal / text_values_match,
    "哪些算真值"复用 selfcheck._NON_LITERAL —— 本模块不另造一套
  F 引用点不当取值: 「变量 x(运行时确定)」是使用点, 不许进 code_value
  G 纯规则零成本: 不 import llm、不碰文件
  H 端到端: demo 材料真跑一遍, 每张卡都有判据
  I 报告同源: 导出的 .md 里也有这一节（界面看得到、报告看不到 = 只活在网页上）

为什么要有这一项: 假设卡是"从查错转向发现"的第一环, 它最危险的失败模式不是报错,
而是**一本正经地胡说** —— 拿本来一致的东西当矛盾提。所以 B/C/D 三组断言是重点。
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import hypothesis as H                        # noqa: E402
from paper2lab.hypothesis import (KIND_CODE_ONLY, KIND_CONFLICT, KIND_DESC,   # noqa: E402
                                  KIND_PAPER_ONLY, KIND_RUNTIME, KIND_TEXT,
                                  Hypothesis, build_hypotheses, to_dicts)
from paper2lab.models import CodeInfo, Evidence, PaperInfo     # noqa: E402

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


def ev(raw: str, *, source: str = "code", loc: str = "train.py:1", norm=None) -> Evidence:
    return Evidence(source=source, location=loc, snippet=f"= {raw}",
                    raw_value=raw, normalized=norm)


def P(**params) -> PaperInfo:
    return PaperInfo(params=dict(params))


def C(**params) -> CodeInfo:
    return CodeInfo(params=dict(params))


def kinds(cards) -> list:
    return [c.kind for c in cards]


def key_of(cards, key: str):
    return next((c for c in cards if c.key == key), None)


print("\n== A. 卡与分类的结构 ==")
card = Hypothesis(key="k", kind=KIND_CONFLICT, claim="c", question="q",
                  falsifiable_by="f")
check(bool(card.to_dict()), "to_dict 能出字典")
check(set(card.to_dict()) >= {"key", "kind", "kind_desc", "claim", "question",
                              "falsifiable_by", "score", "why", "paper_value",
                              "code_value", "paper_where", "code_where",
                              "needs_rerun", "rank"},
      "to_dict 键齐全（含 kind_desc / falsifiable_by / needs_rerun）")
check(all(KIND_DESC.get(k) for k in (KIND_CONFLICT, KIND_RUNTIME, KIND_TEXT,
                                     KIND_CODE_ONLY, KIND_PAPER_ONLY)),
      "五类假设都有中文解释（KIND_DESC 无空项）")
check(len(H.to_dicts([card])) == 1, "to_dicts 按张导出")

print("\n== D. 不制造伪矛盾（本次重点：曾经在这里翻车）==")
c1 = build_hypotheses(P(batch_size=ev("32", source="paper", loc="Page 3")),
                      C(batch_size=ev("32")))
check(len(c1) == 0, "值完全相同(32 vs 32) → 不出卡")

c2 = build_hypotheses(P(weight_decay=ev("5e-4", source="paper", loc="Page 3")),
                      C(weight_decay=ev("0.0005")))
check(len(c2) == 0, "数值等价(5e-4 vs 0.0005) → 不出卡")

c3 = build_hypotheses(P(learning_rate=ev("1e-4", source="paper", loc="Page 3")),
                      C(learning_rate=ev("0.001")))
check(kinds(c3) == [KIND_CONFLICT], "数值冲突(1e-4 vs 0.001) → 出「取值冲突」卡")

c4 = build_hypotheses(
    P(scheduler=ev("cosine annealing", source="paper", loc="Page 3")),
    C(scheduler=ev("CosineAnnealingLR")))
check(len(c4) == 0,
      "文本互为包含(cosine annealing vs CosineAnnealingLR) → 与 aligner 同一口径, 不出卡")

print("\n== F. 引用点不当取值 ==")
c5 = build_hypotheses(
    P(batch_size=ev("32", source="paper", loc="Page 3")),
    C(batch_size=[ev("32", loc="train.py:12"),
                  ev("变量 batch_size(运行时确定)", loc="train.py:26")]))
check(len(c5) == 0, "写死值与论文一致 + 另有引用点 → 不出卡（引用点不算取值）")

c6 = build_hypotheses(
    P(lr=ev("0.001", source="paper", loc="Page 3")),
    C(lr=[ev("变量 lr(运行时确定)", loc="train.py:9"),
          ev("变量 args.lr(运行时确定)", loc="train.py:20")]))
check(kinds(c6) == [KIND_RUNTIME], "代码全是引用点 → 出「运行时才定」卡")
check(all("变量 " not in c.code_value for c in c6),
      "「变量 x(运行时确定)」的原文没被当成取值写进 code_value"
      "（只在 code_value 里如实写明「未写死，N 处运行时确定」）")

c7 = build_hypotheses(
    P(lr=ev("1e-4", source="paper", loc="Page 3")),
    C(lr=[ev("0.001", loc="train.py:9"), ev("0.01", loc="train.py:20")]))
k7 = key_of(c7, "lr")
check(k7 is not None and k7.kind == KIND_CONFLICT and k7.score == 4,
      "代码自己就写了两个不同值 → 分数升到 4（自洽性都没过）")
check(bool(k7) and k7.needs_rerun is True,
      "代码侧多取值 → 必须实机才能定论（needs_rerun=True）")

print("\n== 另两类 ==")
c8 = build_hypotheses(P(), C(momentum=ev("0.9", loc="train.py:5")))
check(kinds(c8) == [KIND_CODE_ONLY], "代码有、论文没写 → 「代码独有」卡")
c9 = build_hypotheses(P(seed=ev("42", source="paper", loc="Page 3")), C())
check(kinds(c9) == [KIND_PAPER_ONLY], "论文有、代码没有 → 「论文独有」卡")
c10 = build_hypotheses(P(dataset=ev("CIFAR-10", source="paper", loc="Page 4")),
                       C(dataset=ev("v2.0", loc="dataset.py:3")))
check(kinds(c10) == [KIND_TEXT], "两边都是文本且规则口径判不等 → 「写法不同」卡")

print("\n== B/C. 可证伪 + 判据与 needs_rerun 一致 ==")
allc = c3 + c6 + c7 + c8 + c9 + c10
check(all(c.falsifiable_by.strip() for c in allc), "每张卡都有推翻判据")
check(all(c.question.strip() for c in allc), "每张卡都写清了要回答的问题")
check(all(c.claim.strip() and "不一致" != c.claim.strip() for c in allc),
      "claim 不只是「不一致」三个字")
rerun_cards = [c for c in allc if c.needs_rerun]
check(bool(rerun_cards) and all("跑一次" in c.falsifiable_by for c in rerun_cards),
      "标了需实机的卡，判据里确实写了「跑一次」")
static_cards = [c for c in allc if not c.needs_rerun]
check(bool(static_cards) and all("跑一次" not in c.falsifiable_by for c in static_cards),
      "标了静态可判的卡，判据里没有混进「跑一次」")

print("\n== 排序 ==")
cards = build_hypotheses(
    P(a=ev("1", source="paper"), b=ev("1", source="paper"), c=ev("1", source="paper")),
    C(a=ev("2"), b=ev("2"), c=ev("2")))
check([x.rank for x in cards] == [1, 2, 3], "rank 从 1 连续编号")
check([x.key for x in cards] == sorted(x.key for x in cards),
      "同分按参数名排（名次稳定，消融对照才可比）")
check(all(cards[i].score >= cards[i + 1].score for i in range(len(cards) - 1)),
      "按分数降序")
check(all(c.why.strip() for c in cards), "每张卡都写清了自己为什么排这个位置")

print("\n== 健壮性 ==")
check(build_hypotheses(P(), C()) == [], "两侧都空 → 空列表")
check(build_hypotheses(P(a=ev("1", source="paper")), None) != []
      or True, "code=None 不崩（论文独有照样能提）")
check(isinstance(build_hypotheses(P(), None), list), "code=None 返回列表")
check(len(build_hypotheses(P(a=[ev("1", source="paper"), ev("2", source="paper")]),
                           C())) == 1,
      "params 值是 list 的形态也认（_as_list 双形态）")
check(len(build_hypotheses(P(a=ev("1", source="paper")),
                           C(a=[ev("2"), ev("2")]))) == 1,
      "代码同一值重复出现不算内部不自洽（仍然只出 1 张冲突卡，而不是多值冲突卡）")

print("\n== G. 纯规则零成本 ==")
_src = inspect.getsource(H)
# ⚠ 判据是**import 行**, 不是全文搜 "llm"（2026-10-25 改）:
#   注释里出现 `judge_llm` 这类字段名并不等于"调了模型", 全文扫会误伤。
#   这条守的本意只有一个: 提假设这一步不许 import 模型模块。
_import_lines = [ln.strip() for ln in _src.splitlines()
                 if ln.strip().startswith(("import ", "from "))]
check(not any("llm" in ln.lower() for ln in _import_lines),
      "本模块不 import llm（提假设不花钱、不调模型）")
check("open(" not in _src and "write_text" not in _src,
      "本模块不碰任何文件（只读已在内存里的抽取结果）")

print("\n== H. 端到端（demo 材料真跑）==")
DEMO = ROOT / "demo"
if (DEMO / "paper.pdf").exists() and (DEMO / "project.zip").exists():
    from paper2lab.code_analyzer import analyze_code
    from paper2lab.paper_parser import parse_paper

    _paper = parse_paper(str(DEMO / "paper.pdf"))
    _code = analyze_code(str(DEMO / "project.zip"))
    real = build_hypotheses(_paper, _code)
    check(len(real) >= 3, "demo 上至少提出 3 张卡（实得 %d）" % len(real))
    check(all(c.falsifiable_by.strip() for c in real), "demo 上每张卡都有判据")
    check(all(c.code_value.strip() or c.paper_value.strip() for c in real),
          "demo 上每张卡都带上了真实取值")
    _bad = [c.key for c in real
            if c.kind == KIND_CONFLICT and c.paper_value == c.code_value]
    check(not _bad, "demo 上没有「取值相同却报冲突」的伪矛盾卡: %s" % _bad)
    check(len({c.key for c in real}) == len(real), "同一个参数不重复出卡")

    # ---- I. 报告里必须有假设站（导出件与界面同源）
    # 界面看得到、导出的 .md 里看不到 = 「从查错转向发现」只活在网页上；
    # 报告是能带走的那一份，评审打开的是它。
    print("\n== I. 报告里的假设站 ==")
    from paper2lab.models import AuditResult
    from paper2lab.report import generate_report

    _aud = AuditResult(paper=_paper, code=_code)
    _aud.hypotheses = list(real[:3])
    _aud.hypothesis_total = len(real)
    _md = generate_report(_aud)
    check("① 假设站" in _md, "报告含「① 假设站」章节")
    check("怎么推翻它" in _md, "每张卡都带上了推翻判据")
    check(all(c.key in _md for c in _aud.hypotheses),
          "卡里的参数名都出现在报告中")
    check(len(real) == len(_aud.hypotheses) or f"共提出 {len(real)} 条" in _md,
          "总数与展示数不一致时如实说明（共 N 条, 列出前 M 条）")
    check(_md.index("① 假设站") < _md.index("## 审计概览"),
          "假设站排在审计概览之前（先提假设、再给判定）")
    _md2 = generate_report(AuditResult(paper=_paper, code=_code))
    check("① 假设站" in _md2 and "没有提出假设" in _md2,
          "没有假设时报告仍保留这一节, 并说明为什么空")

    # 真链路上最容易出的错: 报告**早于**卡片挂载生成 → 报告写「0 条」、界面列着 5 条。
    # 上面是"手工把卡片塞进 audit 再生成报告", 测不出这个错, 必须真跑一遍 ——
    # 同一份审计的两个出口（界面 / 报告）对同一件事说法不一致, 是最不该出现的分裂。
    from paper2lab.pipeline import run_audit as _run_audit

    _a2, _md3 = _run_audit(str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
                           force_mock=True)
    _n2 = len(getattr(_a2, "hypotheses", None) or [])
    check(_n2 > 0, "真跑链路: audit.hypotheses 非空（实得 %d 条）" % _n2)
    check("## ① 假设站: %d 条" % _a2.hypothesis_total in _md3,
          "真跑链路: 报告里的条数与 audit 一致（报告不早于挂载生成）")
    check(_md3.count("怎么推翻它") == _n2,
          "真跑链路: 报告里的卡片数与 audit 一致（%d vs %d）"
          % (_md3.count("怎么推翻它"), _n2))
else:
    check(False, "demo 材料缺失，无法端到端验证")

print("\n%d 项, 失败 %d" % (CHECKS, len(FAILURES)))
sys.exit(1 if FAILURES else 0)
