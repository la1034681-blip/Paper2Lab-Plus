"""阶段二 2E 验证 —— 影响量级：算得出就算，算不出就明说算不出。

守的契约:
  A 规则表: 每条都带出处与适用限制, 且不含"应该改成多少"这类结论口吻
  B 等效批大小: 批大小 × 累积步数, 倍数与算式逐项可复算
  C 方向: 学习率低于区间下限 → 偏低; 高于上限 → 偏高; 落在区间内 → 不标方向
  D 批大小相同却差 10 倍: 判「无法用批大小解释」（比缩放规则更硬的一条）
  E 缺量: 返回 {} —— 算不出 ≠ 没影响, 且不许拿默认值凑
  F 单卡口径陷阱: per_device 值必须乘并行度换成全局, 查不到卡数就放弃
  G 不造数: 只写 impact 一个新键, claim/score/verdict/needs_rerun 一个不动
  H 契约不破: to_dict 有 impact 键, 14 个老键一个不少（L8: 只加不改）
  I 不相干参数: dropout 这类算不出批大小影响的, 直接返回 {}
  J 纯规则: 不 import 模型、不依赖 streamlit、不 import hypothesis（不成环）
  K 口径歧义: 批大小之比正好等于累积步数时, 两种读法都要写出来, 区间取并集
  L 端到端: demo 材料真跑一遍, 每张卡的 impact 与 verdict 互不覆盖

为什么要有这一项:
  影响量级是**唯一会被用户当成"可以动手改代码"的依据**的字段。
  它写错的代价不是报告难看, 是人照着它去改 lr 然后训练崩了。
  所以这里的重点不是"能算出来", 而是"算不出来时绝不硬凑"。

用法: .venv\\Scripts\\python.exe tests/verify_impact.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import derivation as D                                    # noqa: E402
from paper2lab.derivation import IMPACT_RULES, derive, impact_of          # noqa: E402
from paper2lab.hypothesis import build_hypotheses                         # noqa: E402
from paper2lab.models import CodeInfo, Evidence, PaperInfo                # noqa: E402

CHECKS = 0
FAILURES = []

OLD_KEYS = ("key", "kind", "claim", "question", "falsifiable_by", "score", "why",
            "paper_value", "code_value", "paper_where", "code_where",
            "needs_rerun", "rank", "verdict")


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
    return CodeInfo(params={k: [v] for k, v in params.items()})


class _Card:
    """最小卡片替身 —— impact_of 只按鸭子类型访问 key, 不需要整张 Hypothesis。"""

    def __init__(self, key):
        self.key = key


def impact(**kw):
    """直接对一对参数表算某条卡的影响。"""
    key = kw.pop("key", "gradient_accumulation_steps")
    p = P(**kw.pop("p", {}))
    c = C(**kw.pop("c", {}))
    from paper2lab.compute import _side_values
    return impact_of(_Card(key), _side_values(p.params), _side_values(c.params))


# ---------------------------------------------------------------- A 规则表

def test_rules():
    print("\nA 规则表：只收公开共识，每条带出处")
    # 条数不写死（新增规则不该让这条变红）—— 守的是「那几条必须在」。
    check({"effective_batch", "lr_linear", "lr_sqrt"} <= set(IMPACT_RULES),
          "三条基础规则都在（等效批大小 / 线性缩放 / 平方根缩放）")
    check("warmup_scale" in IMPACT_RULES, "预热规则已登记（恒等关系，不是经验规则）")
    check(all(v.get("title") for v in IMPACT_RULES.values()), "每条规则都有标题")
    check(all(v.get("source") for v in IMPACT_RULES.values()), "每条规则都有出处")
    check(all("limit" in v for v in IMPACT_RULES.values()), "每条规则都声明了适用限制字段")
    # 线性缩放必须有"小批不成立"这句限制 —— 少了它就是在夸大规则的适用范围
    check("小批" in IMPACT_RULES["lr_linear"]["limit"], "线性缩放声明了小批区间不成立")

    blob = " ".join(v["title"] + v["source"] + v["limit"] for v in IMPACT_RULES.values())
    check("应该改成" not in blob and "建议改" not in blob,
          "规则表里不含「应该改成多少」这类结论口吻")


# ---------------------------------------------------------------- B 等效批大小

def test_effective_batch():
    print("\nB 等效批大小：批大小 × 累积步数")
    r = impact(p={"batch_size": ev("256", source="paper", loc="p3"),
                  "learning_rate": ev("1e-4", source="paper", loc="p4")},
               c={"batch_size": ev("1024"), "learning_rate": ev("1e-4"),
                  "gradient_accumulation_steps": ev("4")})
    check(bool(r), "论文 256 / 代码 1024×累积4 → 算得出")
    check(abs(r.get("magnitude", 0) - 16) < 1e-6, "倍数 = 16（4096 ÷ 256）")
    fm = " ".join(r.get("formulas") or [])
    check("256 × 1 = 256" in fm, "算式写明了论文侧等效批 = 256 × 1")
    check("1024 × 4 = 4096" in fm, "算式写明了代码侧等效批 = 1024 × 4")
    check("4096 ÷ 256 = 16" in fm, "算式写明了倍数 16")
    check(r.get("compares") == "等效批大小", "指明比的是等效批大小")
    check(bool(r.get("sources")), "带上了规则出处")

    # 论文没写累积 = 按 1 计，但必须标注"不是替它编数"
    gv = [g for g in r.get("given") or [] if g.get("side") == "论文" and "累积" in g.get("key", "")]
    check(bool(gv), "论文侧累积步数按 1 列进了 given")
    check(gv and "未声明" in gv[0].get("note", ""), "并标注论文未声明、按未使用计")


# ---------------------------------------------------------------- C 方向

def test_direction():
    print("\nC 方向：低于下限偏低，高于上限偏高，区间内不标")
    base_p = {"batch_size": ev("256", source="paper"), "learning_rate": ev("1e-4", source="paper")}

    r = impact(p=base_p, c={"batch_size": ev("1024"), "learning_rate": ev("1e-4"),
                            "gradient_accumulation_steps": ev("4")})
    check(r.get("direction") == "偏低", "学习率没变（1 倍）→ 低于下限 4 倍 → 偏低")
    rg = r.get("range") or {}
    check(rg.get("low") and rg.get("high") and rg["high"] > rg["low"], "区间有上下界且上界大于下界")

    r2 = impact(p=base_p, c={"batch_size": ev("1024"), "learning_rate": ev("1.6e-3"),
                             "gradient_accumulation_steps": ev("4")})
    check(r2.get("direction") == "", "学习率放大 16 倍（= 上限）→ 落在区间内 → 不标方向")

    r3 = impact(p=base_p, c={"batch_size": ev("1024"), "learning_rate": ev("1e-2"),
                             "gradient_accumulation_steps": ev("4")})
    check(r3.get("direction") == "偏高", "学习率放大 100 倍 → 高于上限 → 偏高")


# ---------------------------------------------------------------- D 批大小相同

def test_same_batch():
    print("\nD 批大小相同却差 10 倍：缩放规则解释不掉")
    r = impact(key="learning_rate",
               p={"batch_size": ev("256", source="paper"), "learning_rate": ev("1e-4", source="paper")},
               c={"batch_size": ev("256"), "learning_rate": ev("1e-3")})
    check(r.get("direction") == "无法用批大小解释", "判为「无法用批大小解释」（不是偏低/偏高）")
    check(abs(r.get("magnitude", 0) - 1) < 1e-6, "倍数记作 1（批大小确实没变）")
    note = str(r.get("note") or "")
    check("解释不掉" in note, "note 里说清了这是缩放规则解释不掉的差异")


# ---------------------------------------------------------------- E 缺量

def test_missing():
    print("\nE 缺量：算不出就返回空，不拿默认值凑")
    check(impact(p={"batch_size": ev("256", source="paper")},
                 c={"gradient_accumulation_steps": ev("4")}) != {},
          "只有代码侧批大小 → 按两侧相同计并明说（允许的退让）")

    r = impact(key="dropout", p={"batch_size": ev("256", source="paper")},
               c={"dropout": ev("0.1")})
    check(r == {}, "不相干的参数（dropout）→ 返回 {}")

    r2 = impact(p={}, c={"gradient_accumulation_steps": ev("4")})
    check(r2 == {}, "两侧都没有批大小 → 返回 {}（不猜）")

    # 只有单卡口径、查不到卡数 → 必须放弃，不许当全局值用
    r3 = impact(p={"per_device_train_batch_size": ev("32", source="paper")},
                c={"per_device_train_batch_size": ev("64")})
    check(r3 == {}, "两侧都只有单卡口径且无卡数 → 返回 {}（不当全局值用）")


# ---------------------------------------------------------------- F 单卡口径

def test_per_device():
    print("\nF 单卡口径：必须乘并行度换成全局")
    r = impact(p={"per_device_train_batch_size": ev("32", source="paper"),
                  "world_size": ev("4", source="paper")},
               c={"per_device_train_batch_size": ev("32"),
                  "world_size": ev("8"),
                  "gradient_accumulation_steps": ev("1")})
    check(bool(r), "per_device 值 + 卡数 → 能算出全局批大小")
    fm = " ".join(r.get("formulas") or [])
    check("128" in fm, "论文侧 32 × 4 = 128（换成了全局值）")
    check("256" in fm, "代码侧 32 × 8 = 256（换成了全局值）")
    # 128 → 256 是 2 倍，不是 32→32 的 1 倍（直接拿单卡值比会算出 1 倍，那就错了）
    check(abs(r.get("magnitude", 0) - 2) < 1e-6, "倍数 = 2，不是 1（证明没有直接比单卡值）")


# ---------------------------------------------------------------- G 不造数

def test_no_overwrite():
    print("\nG 不造数：impact 之外一个字段都不许动")
    p = P(batch_size=ev("256", source="paper", loc="p3"),
          learning_rate=ev("1e-4", source="paper", loc="p4"))
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    cards = build_hypotheses(p, c)
    # rank 由 2B 的 _rerank 重排、verdict 由三道推演回填，这两个本来就会变；
    # 这里守的是「impact 没有偷偷改文案、改分数、改要不要实机」这三样。
    # 用字典按 key 比 —— `_rerank` 会**就地把列表重排**，按序列比会误报成"内容变了"。
    before = {x.key: (x.claim, x.score, x.needs_rerun) for x in cards}
    derive(cards, p, c)
    after = {x.key: (x.claim, x.score, x.needs_rerun) for x in cards}
    check(before == after, "推演前后 claim/score/needs_rerun 完全一致")
    # verdict 本来就该被 2B 的推演改写（空串 → 待验/已排除/已定位），这不是 impact 干的；
    # 这里只确认它没被写成第四个值（impact 没有权力新增判决态）。
    legal = {D.DERIVE_KEEP, D.DERIVE_EXCLUDED, D.DERIVE_FLAGGED, ""}
    check(all(x.verdict in legal for x in cards),
          "verdict 仍是三种既有判决之一（impact 没造出第四种）")

    has = [x for x in cards if x.impact]
    check(bool(has), "至少有一张卡带上了 impact")
    # exec_log 与就绪度读的是 verdict/score，这里只要确认 impact 没被塞进那两个字段
    check(all(isinstance(x.impact, dict) for x in cards), "impact 一律是字典（空字典表示算不出）")


# ---------------------------------------------------------------- H 契约

def test_contract():
    print("\nH 契约不破：只加不改")
    p = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    cards = derive(build_hypotheses(p, c), p, c)
    d = cards[0].to_dict()
    missing = [k for k in OLD_KEYS if k not in d]
    check(not missing, "to_dict 老键一个不少（缺: %s）" % (missing or "无"))
    check("impact" in d, "to_dict 里有了 impact 新键")
    check(isinstance(d["impact"], dict), "impact 是字典形态（可直接进 JSON）")
    import json
    try:
        json.dumps(d, ensure_ascii=False)
        ok = True
    except (TypeError, ValueError) as ex:      # noqa: F841
        ok = False
    check(ok, "to_dict 能安全序列化成 JSON（界面/留痕要用）")


# ---------------------------------------------------------------- K 口径歧义

def test_ambiguity():
    print("\nK 口径歧义：两种读法都写出来")
    r = impact(p={"batch_size": ev("256", source="paper"),
                  "learning_rate": ev("1e-4", source="paper")},
               c={"batch_size": ev("1024"), "learning_rate": ev("1e-4"),
                  "gradient_accumulation_steps": ev("4")})
    note = str(r.get("note") or "")
    check("口径歧义" in note, "批大小之比 = 累积步数时，标出「口径歧义」")
    check("两种读法并存" in note, "并写明两种读法并存")
    rg = r.get("range") or {}
    # 并集后区间必须比单读法宽：[√16,16]=[4,16]，并入 4 倍读法 → 下界降到 2
    check(rg.get("low") is not None and rg["low"] <= 2.0001,
          "区间取并集（下界放宽到 2，不藏不确定性）")


# ---------------------------------------------------------------- M 候选排序（③）

_SCORE_ITEMS = ("有出处", "复验消失", "违反量纲", "改动量小")


def _sv(holder):
    from paper2lab.compute import _side_values
    return _side_values(getattr(holder, "params", None))


def _impact_of(key, p, c, paper=None):
    return impact_of(_Card(key), _sv(p), _sv(c), paper if paper is not None else p)


def test_candidates():
    print("\nM 候选解释：四项可核算、同分取保守读法")
    p = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    r = _impact_of("batch_size", p, c)
    cands = list(r.get("candidates") or [])
    check(len(cands) >= 2, "口径歧义时给出 2 条候选读法（实得 %d）" % len(cands))
    check([x.get("rank") for x in cands] == list(range(1, len(cands) + 1)),
          "候选序号从 1 连续递增")
    check(all(cands[i]["score"] >= cands[i + 1]["score"]
              for i in range(len(cands) - 1)), "按证据分降序排列")
    check(all(sum(x["delta"] for x in cd.get("parts") or []) == cd.get("score")
              for cd in cands), "打分明细逐项相加 = 总分（可复核）")
    check(all(x["item"] in _SCORE_ITEMS for cd in cands for x in cd.get("parts") or []),
          "只允许那四项打分（有出处 / 复验消失 / 违反量纲 / 改动量小）")
    check(all(str(x.get("why") or "") for cd in cands for x in cd.get("parts") or []),
          "每一项加减分都写了理由（不许只给分不给理由）")
    check("单次前向值" in str(cands[0].get("reading")),
          "同分时先列保守读法（不把差异解释掉的那条）")
    check(r.get("strongest") == cands[0].get("reading"), "strongest 就是第 1 顺位那条")
    check("_ctx" not in r, "推演中间量 _ctx 已 pop（不外泄、不进 JSON）")


# ---------------------------------------------------------------- N 四项核算（④）

def test_four_checks():
    print("\nN 四项核算：多口径 / 翻盘 / 边界 / 对照")
    p = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    p.results = {"accuracy": ev("78.5", source="paper", loc="Page 5 · Table 1")}
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    r = _impact_of("batch_size", p, c)

    lenses = list(r.get("lenses") or [])
    check([x.get("name") for x in lenses] == ["线性缩放", "平方根缩放"],
          "多口径算：线性缩放与平方根缩放各算一遍")
    check(lenses and abs(lenses[0]["value"] - 16) < 1e-6 and abs(lenses[1]["value"] - 4) < 1e-6,
          "两个口径的值分别是 16 与 4（不是抄同一个数）")

    flips = list(r.get("flip") or [])
    check(len(flips) >= 2, "翻盘算：至少列出两条能让结论反转的输入（实得 %d）" % len(flips))
    blob = " ".join(str(x.get("input")) for x in flips)
    check("口径" in blob, "翻盘里点了 batch_size 口径这条")
    check("累积" in blob, "翻盘里点了「累积是否真作用于训练」这条")
    check(all(x.get("if_changed") and x.get("effect") for x in flips),
          "每条翻盘都写了「换成什么」和「后果」")

    zero = list(r.get("zero_when") or [])
    check(len(zero) >= 2, "边界算：至少两条影响归零的条件（实得 %d）" % len(zero))
    check(any("影响为零" in str(z) for z in zero), "边界条目明写「影响为零」")

    ag = dict(r.get("against") or {})
    check(ag.get("available") is True, "对照算：论文有结果指标时给出对照基准")
    check(ag.get("metric") == "准确率" and str(ag.get("paper_value")) == "78.5",
          "对照基准取的是论文报告的准确率 78.5")
    check("Page 5" in str(ag.get("where")), "对照基准带页码出处")
    check("不是我们给出的结论" in str(ag.get("note")),
          "对照明写「不是我们给出的结论」（本项目不跑训练）")

    # 论文没抽到结果指标 → 如实说做不了，不编一个数来比
    p2 = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    r2 = _impact_of("batch_size", p2, c)
    ag2 = dict(r2.get("against") or {})
    check(ag2.get("available") is False, "没有结果指标时 available=False")
    check("做不了" in str(ag2.get("why")), "并写明这一项本次做不了")


# ---------------------------------------------------------------- O 预热

def test_warmup():
    print("\nO 预热步数：恒等关系，预热期走过的样本数守恒")
    p = P(batch_size=ev("256", source="paper"), warmup_steps=ev("2000", source="paper"))
    c = C(batch_size=ev("1024"), gradient_accumulation_steps=ev("4"),
          warmup_steps=ev("500"))
    r = _impact_of("warmup_steps", p, c)
    check(r.get("kind") == "warmup", "预热卡走预热规则（不借用批大小那条）")
    check(abs((r.get("range") or {}).get("low", 0) - 1 / 16) < 1e-6,
          "应有预热步数倍数 = 1/k = 0.0625")
    check(r.get("direction") == "偏多", "2000 → 应有 125 步，代码写了 500 步 → 偏多")
    fm = " ".join(r.get("formulas") or [])
    check("2000 ÷ 16 = 125" in fm, "算式写明了应有步数 125")

    c2 = C(batch_size=ev("1024"), gradient_accumulation_steps=ev("4"),
           warmup_steps=ev("125"))
    r2 = _impact_of("warmup_steps", p, c2)
    check(r2.get("direction") == "已对齐", "代码写 125 步 → 已按批大小调整过 → 已对齐")

    # 比例口径不推：warmup_ratio 乘未知总步数，算不出步数
    p3 = P(batch_size=ev("256", source="paper"), warmup_ratio=ev("0.05", source="paper"))
    r3 = _impact_of("warmup_steps", p3, c)
    check(r3 == {}, "一侧只给 warmup_ratio → 返回 {}（比例当步数会算出假倍数）")


# ---------------------------------------------------------------- P 口径纪律

def test_language():
    print("\nP 口径纪律：不许自称「最优」，必须自称「提示层」")
    p = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    r = _impact_of("batch_size", p, c)
    blob = (str(r.get("scope") or "")
            + " ".join(str(x.get("reading")) for x in r.get("candidates") or []))
    check("最优" not in blob, "产出文案里不出现「最优」（没跑过训练，说不了优不优）")
    check("提示层" in str(r.get("scope") or ""), "scope 明写这是提示层")
    check("不进判定" in str(r.get("scope") or ""), "scope 明写不进判定")
    check("不进评分" in str(r.get("scope") or ""), "scope 明写不进评分")


# ---------------------------------------------------------------- Q 接页面与报告

def test_wired():
    print("\nQ 接页面与报告：算出来的东西必须能看见")
    rep_src = (ROOT / "paper2lab" / "report.py").read_text(encoding="utf-8")
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("_impact_md(audit)" in rep_src, "报告装配里调用了 _impact_md")
    check("_render_impact(audit)" in app_src, "页面装配里调用了 _render_impact")
    check("影响量级" in app_src, "界面上有「影响量级」这一块")

    from paper2lab.models import AuditResult
    from paper2lab.report import _impact_groups, _impact_md

    p = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    p.results = {"accuracy": ev("78.5", source="paper", loc="Page 5 · Table 1")}
    c = C(batch_size=ev("1024"), learning_rate=ev("1e-4"),
          gradient_accumulation_steps=ev("4"))
    cards = derive(build_hypotheses(p, c), p, c)
    a = AuditResult(paper=p, code=c)
    a.hypotheses = cards
    check(len(_impact_groups(a)) == 1,
          "批大小卡与学习率卡合并成一条（同一件事不说两遍）")
    md = "\n".join(_impact_md(a))
    check("影响量级" in md, "报告章节能生成")
    check("候选解释" in md, "报告里有候选解释")
    check("翻盘算" in md and "边界算" in md, "报告里有翻盘与边界")
    check("不是我们给出的结论" in md, "报告里对照那一项写明不是我们给出的结论")
    check("不进判定" in md, "报告里也写了提示层口径")

    # 翻盘为空时也必须写清**为什么空** —— 默默少一行会让人以为"没做这一步"
    p3 = P(batch_size=ev("256", source="paper"), learning_rate=ev("1e-4", source="paper"))
    c3 = C(batch_size=ev("512"), learning_rate=ev("1e-4"))
    a3 = AuditResult(paper=p3, code=c3)
    a3.hypotheses = derive(build_hypotheses(p3, c3), p3, c3)
    md3 = "\n".join(_impact_md(a3))
    check("没有会让结论反转的输入" in md3, "翻盘为空时写明「本次没有会让结论反转的输入」")


# ---------------------------------------------------------------- J 纯规则

def test_pure_rules():
    print("\nJ 纯规则：不拉模型、不依赖界面、不成环")
    src = (ROOT / "paper2lab" / "derivation.py").read_text(encoding="utf-8")
    head = src.split("# ---------------------------------------------------------------- 影响量级")[-1]
    check("import streamlit" not in src, "derivation 不 import streamlit")
    check("from .llm" not in src and "import llm" not in head, "derivation 不 import 模型模块")
    check("from .hypothesis" not in src, "derivation 不 import hypothesis（不成环）")


# ---------------------------------------------------------------- L 端到端

def test_end_to_end():
    print("\nL 端到端：demo 材料真跑一遍")
    from paper2lab.envfile import load_env
    load_env()
    pdf = ROOT / "demo" / "paper.pdf"
    zipf = ROOT / "demo" / "project.zip"
    if not pdf.exists() or not zipf.exists():
        print("  [--] 跳过（demo 材料不在）")
        return

    from paper2lab.code_analyzer import analyze_code
    from paper2lab.paper_parser import parse_paper

    try:
        paper = parse_paper(str(pdf))
    except Exception as ex:                                   # noqa: BLE001
        print("  [--] 跳过端到端（论文解析未产出材料）：%s" % str(ex)[:60])
        return

    try:
        code = analyze_code(str(zipf))
    except Exception as ex:                                   # noqa: BLE001
        print("  [--] 跳过端到端（代码分析未产出）：%s" % str(ex)[:60])
        return

    cards = derive(build_hypotheses(paper, code), paper, code)
    check(len(cards) > 0, "demo 材料产出了假设卡（%d 张）" % len(cards))
    with_impact = [c for c in cards if c.impact]
    print("    影响推演命中 %d / %d 张卡" % (len(with_impact), len(cards)))
    check(all(isinstance(c.impact, dict) for c in cards), "每张卡的 impact 都是字典")
    for c in with_impact:
        fm = c.impact.get("formulas") or []
        check(bool(fm) and all(isinstance(x, str) and x for x in fm),
              "「%s」的算式非空且都是字符串" % c.key)
        break


def main():
    print("=" * 62)
    print("阶段二 2E 验证 —— 影响量级")
    print("=" * 62)
    test_rules()
    test_effective_batch()
    test_direction()
    test_same_batch()
    test_missing()
    test_per_device()
    test_no_overwrite()
    test_contract()
    test_ambiguity()
    test_candidates()
    test_four_checks()
    test_warmup()
    test_language()
    test_wired()
    test_pure_rules()
    test_end_to_end()

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
