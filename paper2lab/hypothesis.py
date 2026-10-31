"""假设卡（阶段二 2A · 2026-10-15）——把「两边对不上」翻译成**可被推翻的陈述**。

一句话定位:
    审计的终点是「论文说 A、代码说 B」; 科学的起点是「那到底哪个是真的、为什么不一样」。
    这一步就是那个转折 —— 不停在"不一致"三个字上, 而是给出一句**能被推翻**的话。

为什么排在 align（③ 验证）之前, 而不是之后:
    提假设**不需要先有判定**。「论文写 1e-4、代码写 3e-4」本身就是一个待验证的假设,
    而不是一条已经成立的结论 —— 判定留给 ③。
    这样排还有个好处: 四站进度条上「① 假设」是**连续的一段**（读材料 → 提假设），
    叙事与界面一致, 不会出现同一站被劈成头尾两块。

**这一模块是纯规则、零成本、不调模型、不碰任何文件。**

—————————————————————————————————————————————
口径纪律（本项目最忌讳"两套口径"，所以下面每一处比较都复用现成的）
—————————————————————————————————————————————
  值是否相同      → `synonyms.values_equal`      （先数值归一, 再规范化字符串）
  能否当数值比    → `synonyms.numbers_equal` is None 表示"不是数值型"
  文本型宽松比对  → `aligner.text_values_match`   （仅在**两侧都不是数值**时才用;
                    它对数值型一律返回 False, 直接拿来当"是否一致"会把 32==32 判成不一致）
  哪些取值算"真值"→ `selfcheck._NON_LITERAL`     （识别「变量 x(运行时确定)」这类引用点,
                    它不是赋值, 不能当成"代码把值设成了它"）

  上面三个都是**私有/既有**口径, 这里直接 import 而不复制 —— 项目先例:
  `graph.py` 就 `from .runner import _store_session_memory`。
  ⚠ 别让 `selfcheck` / `aligner` 反过来 import 本模块, 会成环。

三条硬规矩（都来自项目既有口径, 不是个人口味）:
    1. **不造数**: 卡里出现的每个值都来自已抽到的 Evidence, 带出处; 推不出来的就明说。
    2. **可证伪**: 每张卡必须写清"看到什么现象就推翻它"。写不出判据的卡不发。
    3. **判据与 `needs_rerun` 严格一致**: 说"读代码就能定"就不标"需实机验证"，
       说"跑一次"就必须标 —— 否则 ② 实验站会拿到一堆其实静态可判的活。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .aligner import text_values_match
# 阶段二 2B: 推演判决的三个常量只 import 这一次, 别处一律用常量名比较（不写字面量）。
# 方向是 hypothesis -> derivation; 反向没有依赖（derivation 只按鸭子类型访问卡片）, 不成环。
from .derivation import DERIVE_EXCLUDED, DERIVE_FLAGGED
from .selfcheck import _NON_LITERAL as NON_LITERAL_VALUE
from .synonyms import numbers_equal, values_equal

# 四类假设的固定说法（对外文案只有这一份, 别处不要另写一套）
KIND_CONFLICT = "取值冲突"
KIND_RUNTIME = "运行时才定"
KIND_TEXT = "写法不同"
KIND_CODE_ONLY = "代码独有"
KIND_PAPER_ONLY = "论文独有"

KIND_DESC: dict = {
    KIND_CONFLICT: "论文与代码对同一参数写死了不同的值 —— 需要定出到底哪个真正生效",
    KIND_RUNTIME: "代码没有写死这个值、留到运行时才定 —— 论文的静态声明未必等于实际取值",
    KIND_TEXT: "两边写的是同一个东西的不同写法，还是真的两回事 —— 需要归一后确认",
    KIND_CODE_ONLY: "代码里设了这个超参, 论文一个字没提 —— 需要确认它是否真的生效",
    KIND_PAPER_ONLY: "论文声明了这个设定, 代码里找不到对应实现 —— 需要确认是没实现还是换了名字",
}


@dataclass
class Hypothesis:
    """一张假设卡。

    字段口径（界面 / 报告 / 留痕都读这里, 别处不要再造一份）:
      key           规范化参数名, 如 "learning_rate"
      kind          KIND_* 五选一
      claim         一句可证伪的陈述（**不是**"不一致"三个字）
      question      要回答的那个问题
      falsifiable_by 看到什么就推翻它 —— 必填; 写不出来就不该发这张卡
      score         排序分（越高越该先验）; `why` 如实写清分从哪来
      needs_rerun   是否**必须实机才能定论**（静态可判的给 False, 好让 ② 站少排活）
      rank          1 = 最该先验（build_hypotheses 回填; 阶段二 2B 起由推演后的次序覆盖）
      verdict       阶段二 2B 的推演判决: 待验 / 已排除 / 已定位（derivation.derive() 回填）
      derivation    推演结论一句话（带算式, 供人工核）; 空串 = 本次没做推演

阶段二 2B 的分工（**本模块不自己推演**）:
    本模块只负责"提出可被推翻的陈述"; 推演与筛排是 `derivation.py` 的事 ——
    由「计算员」在提完之后先算一遍, 判出「真分歧 / 伪分歧 / 一侧写错」。
    这样切开的好处: 提假设的口径与推演的口径可以各自独立测、独立换
    （与 collider / aligner 分工的理由一致）。
    """
    key: str
    kind: str
    claim: str
    question: str
    falsifiable_by: str
    score: int = 0
    why: str = ""
    paper_value: str = ""
    code_value: str = ""
    paper_where: str = ""
    code_where: str = ""
    needs_rerun: bool = False
    rank: int = 0
    # ---- 阶段二 2B: 符号推演（由 derivation.derive() 回填）
    # **只加新键, 不动上面任何一个既有字段的语义**（execution_log / 界面 / 报告都按老键读）。
    # 空串 = 本次没做推演（例如推演模块不可用）—— 不假装推演过。
    verdict: str = ""          # 待验 / 已排除 / 已定位（常量见 derivation.py）
    derivation: str = ""       # 推演结论一句话（带算式, 供人工核）
    derivation_kind: str = ""  # 约束检查 / 口径换算 / 恒等验证
    engine: str = ""           # 用的什么引擎（如实标, 不写死）
    # ---- 阶段二 2E: 影响量级（发现型假设）。由 `derivation.impact_of()` 回填。
    # **纯提示层**: 不参与 score / rank / verdict, 也不进就绪度评分 ——
    # 推错了最多多一句话, 不会让人拿着假结论去改代码。
    # 空字典 = 本次算不出来（材料缺量）, **不等于"没有影响"**, 界面要如实区分。
    impact: dict = field(default_factory=dict)
    # ---- 2026-10-25: 模型先判、程序后验（`judge_llm` 模块回填）
    # **只加新键, 不动上面任何一个既有字段的语义**。
    # 口径: `verdict` 永远由 `derivation.derive()` 写（可重算）; 模型的话只存这三格。
    # `llm_agree` 三态: True=两边一致 / False=两边不一致 / None=没比成
    #   （模型没判, 或规则那边这次没给结论 —— 后者模型的话只能当推测, 不能当确证）。
    llm_verdict: str = ""
    llm_why: str = ""
    llm_agree: Optional[bool] = None

    @property
    def is_excluded(self) -> bool:
        """推演判定「这不是真分歧」——排到最后, 但仍列出来。"""
        return self.verdict == DERIVE_EXCLUDED

    @property
    def is_flagged(self) -> bool:
        """推演发现比"冲突"更硬的东西（一侧取值非法）——提到最前。"""
        return self.verdict == DERIVE_FLAGGED

    def to_dict(self) -> dict:
        return {
            "key": self.key, "kind": self.kind, "kind_desc": KIND_DESC.get(self.kind, ""),
            "claim": self.claim, "question": self.question,
            "falsifiable_by": self.falsifiable_by, "score": self.score, "why": self.why,
            "paper_value": self.paper_value, "code_value": self.code_value,
            "paper_where": self.paper_where, "code_where": self.code_where,
            "needs_rerun": self.needs_rerun, "rank": self.rank,
            # 阶段二 2B 新增（旧键一个没动）
            "verdict": self.verdict, "derivation": self.derivation,
            "derivation_kind": self.derivation_kind, "engine": self.engine,
            # 阶段二 2E 新增（同样只加不改）
            "impact": dict(self.impact or {}),
            # 2026-10-25 新增（模型判断, 只加不改）
            "llm_verdict": self.llm_verdict, "llm_why": self.llm_why,
            "llm_agree": self.llm_agree,
        }


# ---------------------------------------------------------------- 取证小工具

def _as_list(v: Any) -> list:
    """`params` 的值有两种历史形态: 单个 Evidence 或 list[Evidence]。

    （`toolbox._run_fill_missing` 里 `params_all` 存 list、`params` 存单个。）
    **两种都要认**, 否则会漏掉一半候选。
    """
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for x in v if x is not None]
    return [v]


def _split_values(evs: list) -> tuple[list[str], int]:
    """把一侧的证据拆成 **(写死的取值, 运行时才定的引用处数)**。

    「变量 batch_size(运行时确定)」是**使用点**不是赋值点 —— 把它当成"代码把值设成了
    batch_size"会直接生成"等于 32 则成立、等于 32 则推翻"这种自相矛盾的卡
    （2026-10-15 实测踩到）。判据复用 `selfcheck._NON_LITERAL`，不另写一套。
    """
    vals: list[str] = []
    refs = 0
    for ev in evs:
        raw = getattr(ev, "raw_value", None)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        if NON_LITERAL_VALUE.search(s):
            refs += 1
            continue
        if s not in vals:
            vals.append(s)
    return vals, refs


def _places(evs: list) -> str:
    """出处（页码 / 文件:行号）, 最多三条, 多了写"等 N 处" —— 不铺满整屏。"""
    locs: list[str] = []
    for ev in evs:
        loc = (getattr(ev, "location", "") or "").strip()
        if loc and loc not in locs:
            locs.append(loc)
    if not locs:
        return ""
    if len(locs) <= 3:
        return "；".join(locs)
    return "；".join(locs[:3]) + f" 等 {len(locs)} 处"


def _join(vals: list[str], limit: int = 3) -> str:
    vals = [v for v in vals if v]
    if not vals:
        return ""
    if len(vals) <= limit:
        return " / ".join(vals)
    return " / ".join(vals[:limit]) + f" 等 {len(vals)} 个"


def _same(p_raw: str, c_raw: str, pev: Any, cev: Any) -> bool:
    """两侧取值是不是"其实是同一个东西"。

    顺序很重要, 照抄 `selfcheck` 的做法:
      1 先 `values_equal`（数值归一优先, 5e-4 与 0.0005 判为相同）;
      2 **只有两侧都不是数值时**才让 `text_values_match` 出场 ——
        它对数值型一律返回 False, 无条件调用会把 32 == 32 判成"不一致"。
    """
    if values_equal(p_raw, c_raw):
        return True
    if numbers_equal(p_raw, c_raw) is not None:
        return False                       # 是数值型且不等 → 就是真冲突, 不再看文本
    if pev is not None and cev is not None:
        return bool(text_values_match(pev, cev))
    return False


# ---------------------------------------------------------------- 五类卡的生成

def _two_sides(key: str, pevs: list, cevs: list) -> Optional[Hypothesis]:
    """两侧都有这个参数。按"到底哪里不一样"分成三种, 不是笼统一张卡。"""
    pvals, _ = _split_values(pevs)
    cvals, crefs = _split_values(cevs)
    if not pvals or (not cvals and crefs == 0):
        return None
    p_raw = pvals[0]

    # ---- 情况一: 代码没写死, 全是运行时确定（静态看不出实际取值）
    if not cvals and crefs > 0:
        return Hypothesis(
            key=key, kind=KIND_RUNTIME,
            claim=(f"代码里 {key} 没有写死（{crefs} 处都是运行时才确定），"
                   f"论文却声明为 {p_raw} —— 实际训练用的值未必等于论文写的"),
            question=f"{key} 在真正跑起来时到底取到了什么值？",
            falsifiable_by=(f"跑一次短训练, 打印 {key} 的实际生效值："
                            f"等于 {p_raw} → 论文这一处没问题；不等于 → 论文声明与实际实验对不上"),
            score=3, why="代码把值推到运行时，静态看不出来，只能靠实机",
            paper_value=_join(pvals), code_value=f"（未写死，{crefs} 处运行时确定）",
            paper_where=_places(pevs), code_where=_places(cevs), needs_rerun=True,
        )

    c_raw = cvals[0]

    # ---- 情况二: 值确实不一样（写死的字面量 vs 写死的字面量）
    if len(pvals) == 1 and len(cvals) == 1 and _same(p_raw, c_raw, pevs[0], cevs[0]):
        return None                        # 同一件事 → 不是矛盾, 不制造假设

    internal = len(cvals) > 1              # 代码自己就写了多个不同的值 → 连自洽性都没过
    if numbers_equal(p_raw, c_raw) is None:
        # ---- 情况三: 两边都不是数值（optimizer/scheduler/dataset 这类）
        #      规则版已经按统一口径比过了; 不等 = 至少"是不是一回事"还没定论
        return Hypothesis(
            key=key, kind=KIND_TEXT,
            claim=(f"论文写的是「{p_raw}」，代码里是「{c_raw}」 —— "
                   f"这两处可能是同一件事的不同写法，也可能真的不是一回事"),
            question=f"代码实际在用的 {key} 与论文描述的是不是同一个？",
            falsifiable_by=(f"归一到同一口径再比一次（{c_raw} 展开成论文那种写法）："
                            f"说的是同一件事 → 本假设推翻；确实不同 → 假设成立"),
            score=3, why="文本型取值，规则口径判不等，但是否同一件事还没定论",
            paper_value=_join(pvals), code_value=_join(cvals),
            paper_where=_places(pevs), code_where=_places(cevs), needs_rerun=False,
        )

    score, why = 3, "两侧都写死了不同的字面量，是最硬的可查线索"
    if internal:
        score, why = 4, "代码同一参数就写了多个不同的值，连自洽性都还没过"
    # 判据必须与 needs_rerun(=internal) 对齐: 代码侧有多个写死值时,
    # 光看代码**定不了**"最后生效的是哪一个" —— 那是运行时的事, 必须真跑。
    falsifiable = (
        f"代码里 {key} 有多个写死值（{_join(cvals)}）—— "
        f"跑一次短训练，打印实际生效的是哪一个"
        if internal else
        f"沿 {key} 的赋值链看**最后一次**赋值（rootcause 那一步会自动追）："
        f"落在 {c_raw} → 本假设成立；被命令行/配置文件改成别的 → 推翻"
    )
    return Hypothesis(
        key=key, kind=KIND_CONFLICT,
        claim=(f"论文声明 {key} = {p_raw}，代码写死为 {c_raw} —— "
               f"推测真正生效的是 {c_raw}（论文这一处未随实现更新）"),
        question=f"训练时 {key} 到底取了哪个值？",
        falsifiable_by=falsifiable,
        score=score, why=why,
        paper_value=_join(pvals), code_value=_join(cvals),
        paper_where=_places(pevs), code_where=_places(cevs),
        needs_rerun=internal,          # 只有一个写死值时读代码即可定论；多个则必须实机
    )


def _code_only(key: str, cevs: list) -> Optional[Hypothesis]:
    """代码有、论文没写。"""
    cvals, _ = _split_values(cevs)
    if not cvals:
        return None
    return Hypothesis(
        key=key, kind=KIND_CODE_ONLY,
        claim=(f"代码把 {key} 设为 {cvals[0]}，但论文完全没有提到它 —— "
               f"需要确认它是真的参与了训练，还是只是留在代码里没被用上"),
        question=f"{key} 真的参与了这次训练吗？",
        falsifiable_by=(f"把它改成另一个值跑一次短训练：结果发生变化 → 它生效（论文漏写）；"
                        f"结果完全不变 → 它没被真正用上（论文不写反而是对的）"),
        score=2, why="代码侧有写死的取值，可用一个对照实验直接判",
        code_value=_join(cvals), code_where=_places(cevs), needs_rerun=True,
    )


def _paper_only(key: str, pevs: list) -> Optional[Hypothesis]:
    """论文写了、代码里找不到。"""
    pvals, _ = _split_values(pevs)
    if not pvals:
        return None
    return Hypothesis(
        key=key, kind=KIND_PAPER_ONLY,
        claim=(f"论文声明 {key} = {pvals[0]}，但代码里扫不到对应的赋值 —— "
               f"需要确认是代码换了名字，还是这一条根本没有实现"),
        question=f"论文声明的 {key} 在代码里到底去哪了？",
        falsifiable_by=(f"跑一次短训练，看结果是否与论文声称的 {key}={pvals[0]} 对得上："
                        f"对不上 → 代码未实现该设定；对得上 → 只是没找到入口（可能换了名字）"),
        score=2, why="论文侧有明确声明，与代码缺口直接对撞",
        paper_value=_join(pvals), paper_where=_places(pevs), needs_rerun=True,
    )


# ---------------------------------------------------------------- 对外主入口

def build_hypotheses(paper: Any, code: Any) -> list[Hypothesis]:
    """从两侧已抽好的参数里挑出**值得验证的对立点**, 产出排好序的假设卡。

    输入只用 `PaperInfo.params` / `CodeInfo.params`（align 之前的原始抽取结果），
    因此本函数**不依赖 findings、不重复做归一化判定**, 也不碰任何文件。

    返回: 按 score 降序排好的卡列表（`rank` 已回填）。**不截断** ——
    截断留给调用方（toolbox 侧按 max_items 取前 N 条），这样界面能如实说
    "共发现 N 条、列出前 M 条"。
    """
    pmap = dict(getattr(paper, "params", None) or {})
    cmap = dict(getattr(code, "params", None) or {}) if code is not None else {}

    cards: list[Hypothesis] = []
    for key in sorted(set(pmap) | set(cmap)):
        pevs, cevs = _as_list(pmap.get(key)), _as_list(cmap.get(key))
        if pevs and cevs:
            card = _two_sides(key, pevs, cevs)
        elif cevs:
            card = _code_only(key, cevs)
        elif pevs:
            card = _paper_only(key, pevs)
        else:
            card = None
        if card is not None:
            cards.append(card)

    # 排序: 分数降序；同分按参数名排 —— **同分必须给出稳定次序**,
    # 否则每次跑出来的名次会飘，消融对照就没法比了。
    cards.sort(key=lambda c: (-c.score, c.key))
    for i, c in enumerate(cards, start=1):
        c.rank = i
    return cards


def to_dicts(cards: list) -> list:
    """给界面 / 报告 / 留痕用的字典形态（键名只在 to_dict 里定义一次）。"""
    return [c.to_dict() for c in cards]
