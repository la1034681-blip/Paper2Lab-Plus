"""符号推演与筛排（阶段二 2B · 2026-10-15）——「计算员」在提完假设之后先算一遍。

一句话定位:
    2A 把「两边对不上」翻译成可被推翻的陈述, 但那份清单里混着两类东西 ——
      **真分歧**（该动手验的） 与 **伪分歧**（换个口径就一致 / 数值本身写错了）。
    2B 就是在动手之前先让计算员**算一遍**, 把这两类分开。省下 ② 实验站的力气,
    也让「待验」这两个字是有分量的。

三道推演（都不是"猜"—— 每一道都能写出算式, 界面与报告里也照算式呈现）:
    ① 约束检查  值本身落不落在该参数的合法区间里
                → 落在外面 = 某一侧的值写错了, 判「已定位」（不用跑实验就能定）
    ② 倍数推演  两侧差几倍? 这个倍数"像什么"（2 的幂 / 10 的幂 / 简单整数比）
                → 材料里若有能解释它的量（卡数 / 累积步数）正好对得上, 判「已排除」
                → 没有也不空手: 把倍数与方向写出来, 让人知道**该往哪查**
    ③ 恒等验证  把两侧已知值代进已知公式（warmup 换算 / lr 缩放）, 看等号成不成立
                → 成立 = 两者是同一件事的两种写法, 判「已排除」

⚠ 一句实话（别被"推演"两个字骗了）:
    真实材料里能推演的卡**是少数**。2026-10-15 在 5 个真实样本上实测:
    43 张卡里只有 8 张能推出东西 —— 因为论文侧往往只声明了少数超参,
    代码侧的取值又常常是"变量 x（运行时确定）", 两侧凑不齐一条公式的输入。
    所以本模块的定位是「**能推的推干净, 推不了的如实说推不了**」,
    绝不用默认值补出一个数来充数（与 `compute` 的既有纪律一致）。
    扩展方向是"接更多条关系式", 不是"放宽判据去凑命中率"。

三个判决（界面 / 报告 / 留痕都读 `Hypothesis.verdict`, 别处不要再造一份）:
    待验    推演没能下结论 —— 照原样交给 ② 站, 该动手还得动手
    已定位  推演发现了**比"冲突"更硬的东西**（一侧取值非法）—— 提到最前
    已排除  推演证明这**不是真分歧** —— 排到最后, 但**不删**

—————————————————————————————————————————————————————————
口径纪律（本项目最忌讳"两套口径", 所以下面每一处比较都复用现成的）
—————————————————————————————————————————————————————————
  合法区间    → `collider.numeric_check`    （与「计算员」在 ③ 站用的是同一张表;
                                              换一张表就会出现"① 站说合理、③ 站说不合理"）
  关系式      → `compute.check_relations`   （与闭环「算」那一步同一套公式, 不另写一份）
  两侧取数    → `compute._side_values` / `compute.to_number`
                                            （连"3*1e-4 这种表达式也算数"都是同源口径）
  归一化键    → `synonyms.canonical_key`
  ⚠ 反向不许 import: `compute` / `collider` / `hypothesis` 都不得 import 本模块, 否则成环。

不做的事（都是本项目已经付过代价的教训）:
  · **不删卡**。被排除的卡照样留在清单里, 只标明"为什么不用验" ——
    审计平台的底线是如实, 不是替用户把看不顺眼的东西藏起来。
  · **不改旧键**。只在 `Hypothesis` 上新增字段（见 hypothesis.py 的 L8 说明）。
  · 不碰文件、不调模型、不执行用户代码（要求值也只走 `ast` 白名单）。
"""
from __future__ import annotations

from fractions import Fraction

# 三个判决（常量只在这里定义一次）
DERIVE_KEEP = "待验"
DERIVE_EXCLUDED = "已排除"
DERIVE_FLAGGED = "已定位"

# 三类推演
KIND_CONSTRAINT = "约束检查"
KIND_SCALE = "倍数推演"
KIND_IDENTITY = "恒等验证"

# 「批大小类」参数才做口径换算 —— 只有批量这类量存在"单卡 / 全局"两种口径,
# 学习率差几倍不能拿卡数去解释（那是 lr 缩放规则的事, 归 ③ 恒等验证）。
_BATCHISH = ("batch", "size", "worker")

# 材料里可能出现的"倍数来源"（并行度 / 梯度累积）。命中哪一个都要能被出处追到。
_SCALE_FACTORS = (
    ("world_size", "并行进程数"), ("n_gpu", "GPU 卡数"), ("num_gpus", "GPU 卡数"),
    ("gpu", "GPU 卡数"), ("device_count", "设备数"), ("n_devices", "设备数"),
    ("num_nodes", "节点数"), ("nodes", "节点数"), ("n_nodes", "节点数"),
    ("gradient_accumulation_steps", "梯度累积步数"),
    ("accumulation_steps", "梯度累积步数"), ("accum_steps", "梯度累积步数"),
    ("grad_accum", "梯度累积步数"), ("accum", "梯度累积步数"),
)

# 关系式 → 它管的是哪些参数（用于把推演结论挂回具体的卡）
_IDENTITY_TOKENS = {
    "warmup_unit": ("warmup",),
    "lr_scaling": ("learning_rate", "lr"),
}


def engine_name() -> str:
    """本次推演用的引擎名（界面 / 报告 / 留痕都读这里, 别处不要再写一份）。

    为什么不引 sympy —— 三条理由, 都是本项目已经踩过的:
      1. **能力已经有了**。求值走 `compute.eval_arith`（`ast` 白名单, 认 `3*1e-4`、`2**10`），
         关系式走 `compute.RELATIONS`。2B 要的是"把已知的数代进已知公式",
         不是解未知量的符号方程组 —— 后者本项目当前一处也用不上。
      2. **口径不能分叉**。再引一套符号库, 就会变成"闭环那边用 ast、假设站这边用 sympy",
         同一条公式可能给出两种说法 —— 这正是本项目最忌讳的病。
      3. **不能多一条装不上的路径**。线上是 Streamlit Cloud, 依赖越少越稳。

    真需要解方程时, 在这里挂 sympy 即可 —— 调用点只有本函数与 `_check_constraint`。
    """
    return "内置符号求值(ast 白名单)"


# ---------------------------------------------------------------- 取数

def _side_values(params):
    """把一侧参数表拍平成 `{key: 值字符串}` —— 复用 compute 的同名口径, 不复制一份。

    （项目先例: `graph.py` 就 `from .runner import _store_session_memory` 直接用私有函数。）
    """
    from .compute import _side_values as _impl
    return _impl(params)


def _lookup(vals: dict, key: str):
    from .compute import _lookup as _impl
    return _impl(vals, key)


def _num(vals: dict, key: str):
    from .compute import _num as _impl
    return _impl(vals, key)


def _to_number(v):
    """把值转成数（含 `3*1e-4` 这类表达式）—— 与 compute 同源。"""
    from .compute import to_number
    return to_number(v)


def _find_factor(paper_vals: dict, code_vals: dict):
    """在两侧材料里找"能解释倍数的量"。返回 (中文名, 数值, 出处侧, 材料里的键名) 或 None。

    先按已知别名精确找, 再按子串兜底（材料里的键名五花八门, 但都躲不开这几个词根）。
    返回里带上**材料里的原始键名** —— 推演文案要写"材料里论文侧写着 world_size = 8",
    用户才能自己去那一行核对; 只写"并行进程数 = 8"他找不到。
    """
    for name, cn in _SCALE_FACTORS:
        for vals, side in ((paper_vals, "论文"), (code_vals, "代码")):
            v = _num(vals, name)
            if v is not None and v > 1:
                return cn, v, side, name
    # 子串兜底: 遍历材料里的键, 看有没有含这些词根的
    for name, cn in _SCALE_FACTORS:
        for vals, side in ((paper_vals, "论文"), (code_vals, "代码")):
            for k in vals:
                kk = str(k).lower()
                if name in kk and len(kk) <= len(name) + 12:
                    v = _to_number(vals.get(k))
                    if v is not None and v > 1:
                        return cn, v, side, str(k)
    return None


# ---------------------------------------------------------------- 精确比值与倍数诊断

def _exact(x: float) -> Fraction:
    """把浮点还原成"人写的那两个整数"。1e-4/3e-4 直接除会得到 0.3333333333333333,
    用它判断"是不是简单整数比"会永远判不出来 —— 必须先化回有理数。"""
    try:
        return Fraction(x).limit_denominator(100_000)
    except (ValueError, OverflowError):
        return Fraction(0)


def _pow_order(ratio: float, base: int):
    """ratio 恰好是 base 的整数次幂吗? 是则返回次数（≥1）, 否则 None。"""
    f = _exact(ratio)
    if f <= 1 or f.denominator != 1:
        return None
    n, k = f.numerator, 0
    while n % base == 0 and n > base:
        n //= base
        k += 1
    # 循环退出时 n 还剩最后一次没除, 所以次数是 k+1（100 = 10^2, 不是 10^1）
    return k + 1 if n == base else None


def _ratio_hint(ratio: float) -> str:
    """这个倍数"像什么"—— 给方向, 不下结论。

    ⚠ 分寸: 学习率差 10 倍是**合法且常见**的差异（不同 batch / 不同模型）,
    所以这里一律用"常见于…"的措辞, 绝不说"这是写错了"。判错比不说更糟 ——
    用户会拿着一条假结论去改代码。
    """
    if _pow_order(ratio, 2):
        k = _pow_order(ratio, 2)
        return (f"{ratio:g} 倍（2 的 {k} 次幂）—— 这类倍数常见于并行度相关口径："
                f"显卡数 / 梯度累积步数 / 批拆分方式")
    if _pow_order(ratio, 10):
        k = _pow_order(ratio, 10)
        return (f"{ratio:g} 倍（10 的 {k} 次幂）—— 这类倍数常见于单位与量纲："
                f"百分比与小数混用、漏写或错写量级")
    fr = _exact(ratio)
    if fr.denominator <= 20 and fr.numerator <= 20:
        return (f"{fr.numerator}:{fr.denominator} 的整数比 —— 常见于口径差："
                f"一侧写单卡值、另一侧写全局值（或反之）")
    return f"{ratio:g} 倍，不属于常见的口径倍数，需人工判断"


# 大小写都收: 论文里 `10k` 与 `1M` 同样常见, 只认小写会在真实材料上漏掉一半。
# ⚠ 只在"纯数字 + 单字符后缀"时才展开（见 _expand_units 的长度判断）——
#    `3ms`、`1mb` 这类带别的字母的写法一动都不动, 免得把值改错。
_UNIT_SUFFIX = {
    "k": 1e3, "K": 1e3,
    "m": 1e6, "M": 1e6,
    "b": 1e9, "B": 1e9,
    "w": 1e4, "W": 1e4,
    "万": 1e4, "亿": 1e8,
}


def _expand_units(vals: dict) -> dict:
    """把 `10k` / `1M` / `3万` 这类写法展开成纯数字。

    为什么不改 `compute` 的取数口径: 那条路是闭环「算」在用, 改它会牵动它的契约测试。
    而这类写法在真实论文里很常见（05_ViT 的 warmup 就写成 `10k`）——
    展开不了, 本该能算的关系式就会退化成"只有一侧声明了 warmup", 推演直接消失。

    只在**本模块内部**做, 原值不动（卡的 paper_value / code_value 仍显示用户看到的原样）。
    """
    out = {}
    for k, v in (vals or {}).items():
        s = str(v).strip()
        # 纯整数 + 单位后缀（10k / 1.5M / 3万）; 前后带别的字符的一律不碰
        if len(s) >= 2 and s[-1] in _UNIT_SUFFIX:
            head = s[:-1].strip()
            try:
                out[k] = f"{float(head) * _UNIT_SUFFIX[s[-1]]:g}"
                continue
            except ValueError:
                pass
        out[k] = v
    return out


# ---------------------------------------------------------------- 三道推演

def _check_constraint(card, paper_vals: dict, code_vals: dict):
    """① 约束检查: 值本身是否落在合法区间 —— 复用 `collider.numeric_check` 的同一张表。

    为什么把它排在第一道: 它给出的结论**比"两边不一致"更硬**。dropout=1.5 这种情况,
    争论"论文和代码谁对"是跑偏了 —— 这一侧的值本身就写错了。
    """
    from .collider import numeric_check

    for side, vals in (("论文", paper_vals), ("代码", code_vals)):
        v = _lookup(vals, card.key)
        if v is None or str(v).strip() == "":
            continue
        got = numeric_check(card.key, str(v))
        if got and not got.get("plausible"):
            return {
                "kind": KIND_CONSTRAINT, "verdict": DERIVE_FLAGGED,
                "text": (f"{side}侧 {card.key} = {v} —— {got.get('note') or '超出该参数的合法范围'}。"
                         f"这不是两边口径不同, 是这一侧的取值本身有问题, "
                         f"**不需要跑实验**, 读代码 / 核论文即可定论"),
            }
    return None


def _check_scale(card, paper_vals: dict, code_vals: dict):
    """② 倍数推演: 两侧差几倍, 这个倍数"像什么"。

    两种结局, 分开对待:
      · 材料里恰好有能解释它的量（卡数 / 累积步数）→ 判「已排除」: 这是口径差, 不是真分歧;
      · 解释不掉 → **不排除**, 但把方向标出来（2 的幂 / 10 的幂 / 简单整数比）。

    为什么"解释不掉"也要写: 这一步的价值在于**告诉人往哪查**, 而不是下结论。
    先前的版本在找不到解释量时直接返回 None, 结果是真实样本上 43 张卡一条推演都没有 ——
    文件看着挺像样, 实际什么也没说。那是自欺。
    """
    from .synonyms import canonical_key

    a = _to_number(_lookup(paper_vals, card.key))
    b = _to_number(_lookup(code_vals, card.key))
    if not a or not b or a <= 0 or b <= 0:
        return None
    r = b / a
    ratio = r if r >= 1 else 1.0 / r
    if ratio < 1.5:
        return None                      # 几乎相等: 不写诊断, 免得把清单变成噪音

    kk = str(canonical_key(card.key) or card.key).lower()
    # ---- 先试"能不能解释掉"（只有批大小类参数天然有"单卡 / 全局"两种口径;
    #      学习率差几倍不能用卡数解释 —— 那是 lr 缩放规则的事, 归 ③）
    if any(t in kk for t in _BATCHISH):
        found = _find_factor(paper_vals, code_vals)
        if found:
            cn, f, side, raw = found
            if abs(ratio - f) <= max(1e-9, 1e-6 * max(ratio, f)):
                return {
                    "kind": KIND_SCALE, "verdict": DERIVE_EXCLUDED,
                    "text": (f"两侧 {card.key} 相差 {ratio:g} 倍（{a:g} vs {b:g}），"
                             f"材料里 {side}侧写着 {raw} = {f:g}（{cn}）—— 倍数**正好对得上**: "
                             f"一侧写的是单卡值、另一侧写的是全局值。"
                             f"这不是真分歧, 归一化后是一致的"),
                }

    # r = 代码 / 论文。r ≥ 1 说明**代码侧**更大（这里写反过一次:
    # 报告里出现过"论文 1e-4 / 代码 0.001，论文侧更大"这种自相矛盾的话）。
    bigger = "代码" if r >= 1 else "论文"
    return {
        "kind": KIND_SCALE, "verdict": DERIVE_KEEP,
        "text": (f"两侧 {card.key} 相差 {ratio:g} 倍（论文 {a:g} / 代码 {b:g}，{bigger}侧更大）。"
                 f"{_ratio_hint(ratio)}。材料里没有能直接解释这个倍数的量，"
                 f"所以**仍待验** —— 这里只给方向，不下结论"),
    }


def _check_identity(card, relations: list):
    """③ 恒等验证: 把两侧已知值代进已知公式, 看等号成不成立。

    关系式本身由 `compute.RELATIONS` 提供（warmup 换算 / lr 缩放），这里只做一件事:
    把"算出来一致"的那条结论**挂回具体的卡**上。
    """
    from .synonyms import canonical_key

    kk = str(canonical_key(card.key) or card.key).lower()
    for r in relations or []:
        rid = r.get("rid")
        tokens = _IDENTITY_TOKENS.get(rid)
        if not tokens:
            continue
        if not any(t == kk or t in kk for t in tokens):
            continue
        verdict = r.get("verdict")
        detail = r.get("detail") or ""
        if verdict == "consistent":
            # 确定性结论: 换算之后就是同一个值 —— 这张卡不是真分歧
            return {
                "kind": KIND_IDENTITY, "verdict": DERIVE_EXCLUDED,
                "text": f"按公式 `{r.get('formula', '')}` 复算：{detail}。**不是真分歧**",
            }
        if verdict in ("mismatch", "hint", "info"):
            # ⚠ 这三种**都不是**确定性结论（关系式自己的措辞就是"需人工确认"/"不作为结论"）,
            #   所以一律保持「待验」, 只把推演过程摆出来。
            #   替用户下结论比什么都不说更糟 —— 他会拿着一条假结论去改代码。
            return {
                "kind": KIND_IDENTITY, "verdict": DERIVE_KEEP,
                "text": f"按公式 `{r.get('formula', '')}` 复算：{detail}",
            }
    return None


# ---------------------------------------------------------------- 影响量级（阶段二 2E）

# 等效规则表 —— **只收公开共识, 每条带出处**。
# 为什么这么苛: 这里算出来的东西会被用户当成"可以动手改代码的依据"。
# 一条自己发明的规则, 比什么都不说危险得多 —— 它长得像科学。
# 拿不出处方的规则一律不写; 宁可少一条。
IMPACT_RULES: dict = {
    "effective_batch": {
        "title": "等效批大小 = 批大小 × 梯度累积步数（并行度已折进批大小口径）",
        "source": "梯度累积 / 数据并行的定义式（恒等关系，无需外部文献）",
        "limit": "",
    },
    "lr_linear": {
        "title": "批大小放大 k 倍 → 学习率放大 k 倍",
        "source": "Goyal et al. 2017, Accurate, Large Minibatch SGD（线性缩放规则）",
        "limit": "小批区间不成立，放大幅度过大时须配合预热",
    },
    "lr_sqrt": {
        "title": "批大小放大 k 倍 → 学习率放大 √k 倍",
        "source": "自适应优化器场景的常用做法（平方根缩放）",
        "limit": "与线性缩放并存，两派结论不同 —— 所以只报区间",
    },
    "warmup_scale": {
        "title": "预热期样本数守恒：预热步数 × 等效批大小 = 常数 → 预热步数应 ∝ 1/k",
        "source": "Goyal et al. 2017 的 gradual warmup（预热按「走过的样本数」计, 不是按步数硬编码）",
        "limit": "只认「步数」口径；一侧写 warmup_ratio（比例）、或两侧单位不同（步数 vs 轮数）, 本例不推",
    },
}

# 参数归类用「子串」，不用同义词表 —— 这三类键在 synonyms 里**没有登记**
# （实测 canonical_key("gradient_accumulation_steps") 返回 None）。
# 累积 / 并行的键名从 _SCALE_FACTORS 派生，不手抄一份（同源派生是项目既有纪律）。
_ACCUM_KEYS = tuple(n for n, cn in _SCALE_FACTORS if "累积" in cn)
_PARALLEL_KEYS = tuple(n for n, cn in _SCALE_FACTORS if "累积" not in cn)

# 判"这张卡涉及哪类参数"用的词根。
_IMPACT_BATCH_TOKENS = ("batch",)
_IMPACT_ACCUM_TOKENS = ("accum", "grad_accum")
_IMPACT_PARALLEL_TOKENS = ("world_size", "gpu", "device", "node", "n_nodes", "rank")
_IMPACT_LR_TOKENS = ("learning_rate", "lr")

# 两侧差不到 5% 就算"一样" —— 比这个还小的差异拿批大小去解释没有意义。
_IMPACT_TOL = 0.05


def _scan_num(vals: dict, names) -> tuple:
    """在一侧的侧值表里找第一个能读成正数的量。返回 (原始键名, 数) 或 (None, None)。

    三级匹配（顺序有讲究, 不能反）:
      ① 原始键名精确命中 —— 用户写什么就认什么;
      ② 同义词归一名 —— `train_batch_size` 这类别名;
      ③ 子串兜底 —— 只认长度不超过上限的键, 免得把 `total_training_batch_size_avg` 这种
         长句子也当下标。

    为什么要带原始键名回去: 推演文案要写"代码侧写着 gradient_accumulation_steps = 4",
    用户才能自己去那一行核对。只写"梯度累积步数"他找不到。
    """
    from .synonyms import canonical_key

    vals = vals or {}
    names = tuple(str(n) for n in (names or ()))
    if not names:
        return None, None

    for k in vals:
        if str(k) in names:
            v = _to_number(vals.get(k))
            if v is not None and v > 0:
                return str(k), v
    for k in vals:
        if canonical_key(k) in names:
            v = _to_number(vals.get(k))
            if v is not None and v > 0:
                return str(k), v
    longest = max(len(n) for n in names)
    for k in vals:
        kk = str(k).lower()
        if any(n in kk for n in names) and len(kk) <= longest + 14:
            v = _to_number(vals.get(k))
            if v is not None and v > 0:
                return str(k), v
    return None, None


def _batch_of(vals: dict, parallel: float) -> tuple:
    """取这一侧的「全局批大小」。返回 (原始键名, 数, 人话说明)。

    ⚠ 口径陷阱（真实踩过, 不是假想）: 同义词表把 `per_device_train_batch_size`
    **归一到 `batch_size`** —— 于是"单卡值"和"全局值"在表里长得一模一样。
    直接拿来当全局值会小算 N 倍（N = 卡数），整条影响推演全错。

    处理办法: 先找明确不带 per_device / micro 字样的键, 找不到再按单卡口径取、
    乘并行度换全局; 连并行度都没有就**放弃这次推演**并说明原因（不许拿默认值凑）。
    """
    vals = vals or {}

    for k in vals:
        kk = str(k).lower()
        if "batch" not in kk:
            continue
        if "per_device" in kk or "per_gpu" in kk or "micro" in kk:
            continue
        v = _to_number(vals.get(k))
        if v is not None and v > 0:
            return str(k), v, "全局口径，直接采信"

    for k in vals:
        kk = str(k).lower()
        if "batch" not in kk:
            continue
        v = _to_number(vals.get(k))
        if v is None or v <= 0:
            continue
        if not parallel or parallel <= 0:
            return None, None, f"{k} 是单卡口径，材料里查不到卡数，无法换算成全局值"
        return (str(k), v * parallel,
                f"单卡 {v:g} × 并行度 {parallel:g} = 全局 {v * parallel:g}")

    return None, None, ""


def _equivalence(paper_vals: dict, code_vals: dict):
    """两侧「等效批大小」与倍数 k —— 批大小 / 学习率 / 预热三条规则共用这一份取数口径。

    为什么必须共用: 三条规则各写一份取数, 迟早会有一份把「单卡值」当「全局值」用,
    于是同一份材料在两张卡上给出互相矛盾的倍数。**只留一个口径**。

    返回 None = 材料缺量, 本次算不出（不是「没有影响」）。
    """
    par_key_p, parallel_p = _scan_num(paper_vals, _PARALLEL_KEYS)
    par_key_c, parallel_c = _scan_num(code_vals, _PARALLEL_KEYS)
    acc_key_p, accum_p = _scan_num(paper_vals, _ACCUM_KEYS)
    acc_key_c, accum_c = _scan_num(code_vals, _ACCUM_KEYS)

    # ⚠ 不许把 parallel 兜底成 1.0 —— 兜底会让 _batch_of 把「单卡值」当全局值用（实测踩到）
    accum_p_eff = accum_p if accum_p else 1.0
    accum_c_eff = accum_c if accum_c else 1.0

    p_key, p_batch, p_note = _batch_of(paper_vals, parallel_p)
    c_key, c_batch, c_note = _batch_of(code_vals, parallel_c)

    # 一侧没写批大小、另一侧写了 —— 只在这一种情况下退让: 按「两侧相同」计并明说。
    # 否则一律放弃（缺量就算不出来, 不猜）。
    fallback_note = ""
    if p_batch and not c_batch:
        c_key, c_batch = p_key, p_batch
        c_note = f"代码侧未声明批大小，按与论文相同（{p_batch:g}）计"
        fallback_note = c_note
    if c_batch and not p_batch:
        p_key, p_batch = c_key, c_batch
        p_note = f"论文侧未声明批大小，按与代码相同（{c_batch:g}）计"
        fallback_note = p_note
    if not p_batch or not c_batch or p_batch <= 0 or c_batch <= 0:
        return None

    p_eff = p_batch * accum_p_eff
    c_eff = c_batch * accum_c_eff
    return {
        "p_key": str(p_key or ""), "c_key": str(c_key or ""),
        "p_batch": p_batch, "c_batch": c_batch,
        "p_note": p_note or "", "c_note": c_note or "",
        "accum_p": accum_p, "accum_c": accum_c,
        "accum_p_eff": accum_p_eff, "accum_c_eff": accum_c_eff,
        "acc_key_p": acc_key_p, "acc_key_c": acc_key_c,
        "parallel_p": parallel_p, "parallel_c": parallel_c,
        "p_eff": p_eff, "c_eff": c_eff, "k": c_eff / p_eff,
        "fallback_note": fallback_note,
    }


def _impact_batch_lr(card, paper_vals: dict, code_vals: dict) -> dict:
    """一张卡「如果这条成立，后果有多大」—— 只算量级与方向，**不下结论**。

    返回 `{}` 表示算不出来（材料缺量）。**调用方必须如实写「本次未推演」** ——
    空字典不是"没有影响", 是"这次算不了", 两者不能混。

    三条分寸, 都是这个模块一贯的口径:
      1. **给区间不给点值**。批大小放大 k 倍, 学习率该放大几倍业内有两派
         （线性 k 倍 / 平方根 √k 倍），都算主流做法。只报一个数 = 替用户选边,
         被问"凭什么"就答不上。报区间是把不确定性摊开。
      2. **不算"应该改成多少"**。这里只说"按这派规则, 学习率本该落在 [a, b]",
         绝不写"你应该把 lr 改成 3e-4"。后者是结论, 不是提示。
      3. **论文没声明的量按 1 计, 但必须标注** —— 累积步数没写就当没使用,
         这是"论文声称的设定", 不是我们替它编的数。写进 given 让人一眼看见。

    完全不碰 claim / score / verdict / needs_rerun —— 这是提示层, 推错了也只多一句话。
    """
    from .synonyms import canonical_key

    kk = str(canonical_key(card.key) or card.key).lower()
    is_batch = any(t in kk for t in _IMPACT_BATCH_TOKENS)
    is_accum = any(t in kk for t in _IMPACT_ACCUM_TOKENS)
    is_parallel = any(t in kk for t in _IMPACT_PARALLEL_TOKENS)
    is_lr = any(t in kk for t in _IMPACT_LR_TOKENS)
    if not (is_batch or is_accum or is_parallel or is_lr):
        return {}

    # ---- 取数（每一处都留原始键名, 供用户回原文核）
    eq = _equivalence(paper_vals, code_vals)
    if not eq:
        return {}
    p_key, c_key = eq["p_key"], eq["c_key"]
    p_batch, c_batch = eq["p_batch"], eq["c_batch"]
    p_note, c_note = eq["p_note"], eq["c_note"]
    accum_p, accum_c = eq["accum_p"], eq["accum_c"]
    acc_key_p, acc_key_c = eq["acc_key_p"], eq["acc_key_c"]
    accum_p_eff, accum_c_eff = eq["accum_p_eff"], eq["accum_c_eff"]
    fallback_note = eq["fallback_note"]
    p_eff, c_eff, k = eq["p_eff"], eq["c_eff"], eq["k"]

    # 推演中间量 —— ③④ 那一步（候选排序 / 翻盘 / 边界 / 方向对照）要复用，
    # 但**不对外**：出口处会被 pop 掉，`to_dict()` 与报告只看正式字段。
    # 为什么不重新算一遍: 重算就是第二份取数口径, 迟早和这份对不上。
    ctx: dict = {
        "k": k, "quantity": "学习率",
        "p_key": str(p_key or ""), "c_key": str(c_key or ""),
        "p_batch": p_batch, "c_batch": c_batch,
        "accum_p": accum_p, "accum_c": accum_c,
        "acc_key_p": acc_key_p, "acc_key_c": acc_key_c,
        "p_eff": p_eff, "c_eff": c_eff,
    }

    given: list = []
    formulas: list = []
    for side, bk, bv, bn, ak, av in (
            ("论文", p_key, p_batch, p_note, acc_key_p, accum_p),
            ("代码", c_key, c_batch, c_note, acc_key_c, accum_c)):
        if bk:
            given.append({"side": side, "key": str(bk), "value": f"{bv:g}", "note": bn})
        if ak:
            given.append({"side": side, "key": str(ak), "value": f"{av:g}", "note": ""})
        elif side == "论文":
            given.append({"side": side, "key": "梯度累积步数", "value": "1",
                          "note": "论文未声明，按未使用计 —— 不是替它编数"})

    formulas.append(f"等效批大小（论文） = {p_batch:g} × {accum_p_eff:g} = {p_eff:g}")
    formulas.append(f"等效批大小（代码） = {c_batch:g} × {accum_c_eff:g} = {c_eff:g}")
    formulas.append(f"倍数 k = {c_eff:g} ÷ {p_eff:g} = {k:g}")

    out: dict = {
        "kind": "effective_batch",
        "rule": IMPACT_RULES["effective_batch"]["title"],
        "given": given,
        "formulas": formulas,
        "range": {},
        "direction": "",
        "sources": [f"{IMPACT_RULES['lr_linear']['source']}（{IMPACT_RULES['lr_linear']['limit']}）",
                    f"{IMPACT_RULES['lr_sqrt']['source']}（{IMPACT_RULES['lr_sqrt']['limit']}）"],
        "note": fallback_note,
        "compares": "等效批大小",
        "magnitude": round(k, 6),
    }

    # ---- 口径歧义（真实存在的两种读法, 不许只报对自己有利的那个）
    # 两侧批大小之比**正好等于**代码里的累积步数时, 代码那个 batch_size 有两种可能:
    #   · DataLoader 里的单次前向值（常规写法）→ 等效批 = batch_size × accum, 倍数 k;
    #   · 已经是累积后的等效值          → 等效批 = batch_size, 倍数小 accum 倍。
    # 这一情形下 2B 的推演判决**很可能把这张卡判成「口径差、已排除」**
    # （倍数正好被累积步数解释掉）。两个结论都留在报告里会互相打架,
    # 所以这里如实写明两种读法并存, 不挑一个说得响的。
    ambig_k = None
    if accum_c and accum_c > 1:
        raw_ratio = c_batch / p_batch
        if abs(raw_ratio - accum_c) <= max(1e-9, 1e-6 * max(raw_ratio, accum_c)):
            low_read = raw_ratio if raw_ratio >= 1 else 1.0 / raw_ratio
            ambig_k = low_read
            out["note"] = (out["note"] + "；" if out["note"] else "") + (
                f"口径歧义：两侧批大小之比 {raw_ratio:g} 正好等于代码里的累积步数 {accum_c:g}。"
                f"若代码的 batch_size 是单次前向值（DataLoader 的常规写法），倍数是 {k:g}；"
                f"若它已经是累积后的等效值，则只差 {low_read:g} 倍。两种读法并存")
    ctx["ambig_k"] = ambig_k

    # ---- 学习率落在哪: 只有两侧都取到 lr 才有话说
    p_lr = _num(paper_vals, "learning_rate")
    c_lr = _num(code_vals, "learning_rate")
    if not p_lr or not c_lr or p_lr <= 0 or c_lr <= 0:
        # 学习率取不到时, 只有「等效批大小确实变了」才值得说。
        # 若两侧批大小本来就是我按"相同"补齐的, k 必然 = 1 —— 报一个「1 倍」
        # 等于什么都没说, 还会让界面显示一条假命中。宁可返回空（算不出）, 不拿空话充数。
        if abs(k - 1) <= _IMPACT_TOL:
            return {}
        out["note"] = (out["note"] + "；" if out["note"] else "") + \
            "两侧学习率未同时取到，本次只报等效批大小的倍数，不推方向"
        return _done(out, ctx)

    ratio_lr = c_lr / p_lr
    formulas.append(f"学习率之比 = 代码 {c_lr:g} ÷ 论文 {p_lr:g} = {ratio_lr:g}")
    ctx["ratio_lr"] = ratio_lr

    if abs(k - 1) <= _IMPACT_TOL:
        # 批大小等效 —— 两派缩放规则都推出"学习率不该变"。
        # 这时 lr 还差几倍, 就是一个**任何缩放规则都解释不掉**的差异。
        if abs(ratio_lr - 1) <= _IMPACT_TOL:
            # 算过了、确实没有可推的影响 —— 这是一条**算出来的**结论, 不是"算不出",
            # 所以保留 impact 并标「无影响」, 不返回空字典（那会被当成材料缺量）。
            out["direction"] = "无影响"
            formulas.append("等效批大小一致，学习率也一致 —— 这条没有可推的影响")
            out["range"] = {"low": 1.0, "high": 1.0, "unit": "倍",
                            "subject": "学习率应有的放大倍数"}
            return _done(out, ctx)
        out["direction"] = "无法用批大小解释"
        out["range"] = {"low": 1.0, "high": 1.0, "unit": "倍",
                        "subject": "学习率应有的放大倍数"}
        formulas.append("等效批大小没变，任何一派缩放规则都推出学习率不该变 —— "
                        f"但实际差 {ratio_lr:g} 倍")
        out["note"] = (out["note"] + "；" if out["note"] else "") + \
            "这是比缩放规则更硬的一条：批大小相同却差这么多倍，缩放规则解释不掉"
        return _done(out, ctx)

    lo, hi = sorted((k ** 0.5, k))
    # 有口径歧义时, 区间取**两种读法的并集**。「给区间不给点值」这条在歧义情形下更要守 ——
    # 只报对自己有利的那个读法, 等于把不确定性藏起来, 而那正是这个字段要解决的问题。
    if ambig_k and abs(ambig_k - 1) > _IMPACT_TOL:
        lo = min(lo, min(ambig_k ** 0.5, ambig_k))
        hi = max(hi, max(ambig_k ** 0.5, ambig_k))
    out["range"] = {"low": round(lo, 4), "high": round(hi, 4), "unit": "倍",
                    "subject": "学习率应有的放大倍数"}
    formulas.append(f"学习率应有的放大区间 = [√k, k] 取并集 = [{lo:.4g}, {hi:.4g}] 倍")

    if ratio_lr < lo * (1 - _IMPACT_TOL):
        out["direction"] = "偏低"
        formulas.append(f"实际 {ratio_lr:.4g} 倍 < 下限 {lo:.4g} 倍 —— 偏低")
    elif ratio_lr > hi * (1 + _IMPACT_TOL):
        out["direction"] = "偏高"
        formulas.append(f"实际 {ratio_lr:.4g} 倍 > 上限 {hi:.4g} 倍 —— 偏高")
    else:
        formulas.append(f"实际 {ratio_lr:.4g} 倍落在区间内 —— 两派都能解释，不标方向")
    return _done(out, ctx)


# ---------------------------------------------------------------- ③ 候选解释 ＋ ④ 四项核算
#
# 这一段是「发现型假设」对外能说的那半：找出差异 ＋ 算出影响之后，还要回答两个问题 ——
#   ③ 这个差异有几种合理解释？哪一条**证据最硬**？
#   ④ 换个口径还算得出来吗？什么情况下影响归零？跟论文报的结果对不对得上？
#
# 为什么坚持说「证据最硬」而不是「最优」：
#   「最优」要拿准确率比，那是跑训练的事（本项目不跑训练，也不该假装跑过）。
#   「证据最硬」可以当场核算：材料里有没有出处、代入后冲突消不消失、量纲合不合理、
#   要不要假设材料写错了。四项都能复核，摆出来谁都能重算一遍 —— 这才经得起追问。

_SCORE_SOURCE = 3        # 有出处
_SCORE_RESOLVE = 3       # 复验消失
_SCORE_DIM = -5          # 违反量纲
_SCORE_SMALL = 2         # 改动量小
_DIM_LIMIT = 100.0       # 倍数超过两个数量级 —— 缩放规则在这个区间不成立

_METRIC_LABEL = {
    "accuracy": "准确率", "f1": "F1", "precision": "精确率",
    "recall": "召回率", "loss": "损失", "silhouette": "轮廓系数",
}

# 预热只认「步数」口径 —— `warmup_ratio` 是比例, 拿它当步数会算出一个假倍数,
# 而假倍数比不说更危险（它长得像科学）。
_WARMUP_KEYS = ("warmup_steps", "num_warmup_steps", "warmup_iters", "warmup_step")
_WARMUP_TOKENS = ("warmup",)

# 方向 → 复现时应当看到的现象。措辞与 direction 字段严格对应, 不许自由发挥：
# 这是给人「跑完之后拿什么对照」的清单, 不是我们给出的结论。
_PREDICT = {
    "偏高": "训练初期 loss 更容易震荡或发散，最终指标应低于论文报告值",
    "偏低": "收敛更慢，同等轮数下指标应低于论文报告值（表现为「跑不到论文的数」）",
    "偏少": "预热期走过的样本数不够，大 batch 下训练初期更易不稳，指标可能低于报告值",
    "偏多": "预热占掉的步数过多，有效训练步数变少，指标可能低于报告值",
    "已对齐": "预热已按批大小调整过 —— 这一项不构成问题，不用再查",
    "无法用批大小解释": "这个差异任何缩放规则都解释不掉 —— 先按原值复跑一次，看指标是否对得上",
    "无影响": "训练过程不应因此改变，指标应与论文一致；若明显不一致，原因不在这一项",
}


def _done(out: dict, ctx: dict) -> dict:
    """出口统一收口：把推演中间量挂在 `_ctx`，供 ③④ 复用（对外出口会 pop 掉）。"""
    out["_ctx"] = dict(ctx or {})
    return out


def _lenses(k: float) -> list:
    """多口径算：同一个倍数，两派缩放规则各算一遍。

    这正是「只报区间」的理由 —— 两派都是主流做法，挑一个报等于替用户选边。
    """
    if not k or k <= 0:
        return []
    return [
        {"name": "线性缩放", "value": round(k, 4),
         "source": IMPACT_RULES["lr_linear"]["source"],
         "limit": IMPACT_RULES["lr_linear"]["limit"]},
        {"name": "平方根缩放", "value": round(k ** 0.5, 4),
         "source": IMPACT_RULES["lr_sqrt"]["source"],
         "limit": IMPACT_RULES["lr_sqrt"]["limit"]},
    ]


def _steps_only(vals: dict) -> dict:
    """去掉比例类键（warmup_ratio / warmup_percent）—— 比例乘未知总步数，算不出步数。"""
    out = {}
    for k, v in (vals or {}).items():
        kk = str(k).lower()
        if "ratio" in kk or "percent" in kk or "pct" in kk:
            continue
        out[k] = v
    return out


def _readings_batch(ctx: dict) -> list:
    """这个倍数有几种**材料允许的**读法 —— 每条都要能被代入复算。

    这里不发明方案：只列材料本身允许的两种读法（代码那个 batch_size 是单次前向值、
    还是累积后的等效值）。凭空写一条「建议改成 3e-4」就是结论了，
    那是 ③ 验证站和用户的事，不是假设站能替他们做的。

    `order` 是**同分时的次序**：先列「不把差异解释掉」的那条。
    为什么: 一条「这是口径差」要是排在前面, 真问题就被一句口径抹平了 ——
    宁可先按保守读法报警, 让人自己决定要不要接受「口径差」这个解释。
    """
    k = ctx.get("k") or 0
    if k <= 0:
        return []
    ev = [x for x in (f"论文 {ctx['p_key']}" if ctx.get("p_key") else "",
                      f"代码 {ctx['c_key']}" if ctx.get("c_key") else "") if x]
    out = [{"reading": "代码的 batch_size 按「单次前向值」读（DataLoader 常规写法）",
            "low": min(k, k ** 0.5), "high": max(k, k ** 0.5),
            "evidence": list(ev), "no_edit": True, "order": 0}]
    ambig = ctx.get("ambig_k")
    if ambig and abs(ambig - k) > _IMPACT_TOL:
        ev2 = list(ev)
        if ctx.get("acc_key_c"):
            ev2.append(f"代码 {ctx['acc_key_c']}")
        out.append({"reading": "代码的 batch_size 已是「累积后等效值」，不再乘累积步数",
                    "low": min(ambig, ambig ** 0.5), "high": max(ambig, ambig ** 0.5),
                    "evidence": ev2, "no_edit": True, "order": 1})
    return out


def _candidates(ctx: dict) -> list:
    """③ 候选解释排序 —— 按**四项可核算的**证据强度打分，不按「哪个更优」。

    打分项（每一项都写清加减分理由，摆出来谁都能重算）：
      · 有出处   +3  依赖的键在材料里都找得到
      · 复验消失 +3  代入这个读法后，实际值与应有区间不再矛盾
      · 违反量纲 -5  要求的倍数超过两个数量级，缩放规则在该区间不成立
      · 改动量小 +2  只需重新解释口径，不必假设材料里某个数写错了
    """
    readings = list(ctx.get("readings") or [])
    actual = ctx.get("actual")
    out: list = []
    for r in readings:
        lo, hi = sorted((float(r["low"]), float(r["high"])))
        resolves = bool(actual) and (lo * (1 - _IMPACT_TOL) <= float(actual) <= hi * (1 + _IMPACT_TOL))
        parts: list = []
        total = 0

        ev = [x for x in (r.get("evidence") or []) if x]
        if ev:
            total += _SCORE_SOURCE
            parts.append({"item": "有出处", "delta": _SCORE_SOURCE,
                          "why": "依赖的键在材料里都找到了：" + "、".join(ev)})
        else:
            parts.append({"item": "有出处", "delta": 0,
                          "why": "依赖的键没能在材料里找到，不采信"})

        if resolves:
            total += _SCORE_RESOLVE
            parts.append({"item": "复验消失", "delta": _SCORE_RESOLVE,
                          "why": f"代入后实际 {float(actual):.4g} 倍落在 "
                                 f"[{lo:.4g}, {hi:.4g}] 内，冲突消失"})
        else:
            parts.append({"item": "复验消失", "delta": 0,
                          "why": (f"代入后实际 {float(actual):.4g} 倍仍不在 "
                                  f"[{lo:.4g}, {hi:.4g}] 内，冲突没消失"
                                  if actual else "两侧没同时取到可比的量，无法复验")})

        scale = max(abs(lo), abs(hi))
        if scale > _DIM_LIMIT or scale < 1.0 / _DIM_LIMIT:
            total += _SCORE_DIM
            parts.append({"item": "违反量纲", "delta": _SCORE_DIM,
                          "why": f"这个读法要求 {scale:.4g} 倍，超过两个数量级 —— "
                                 "缩放规则在这个区间不成立"})

        if r.get("no_edit", True):
            total += _SCORE_SMALL
            parts.append({"item": "改动量小", "delta": _SCORE_SMALL,
                          "why": "只需重新解释口径，不必假设材料里某个数写错了"})

        out.append({"reading": r["reading"],
                    "implied": (f"{lo:.4g}" if abs(hi - lo) <= 1e-12 else f"{lo:.4g}~{hi:.4g}"),
                    "quantity": ctx.get("quantity") or "",
                    "resolves": bool(resolves), "score": total, "parts": parts})

    # 同分时按规则给的 order（保守读法先），再按读法名 ——
    # 次序必须稳定, 否则同一份材料跑两遍排序会变（可复现性靠这条撑着）。
    order = {r["reading"]: int(r.get("order", 9)) for r in readings}
    out.sort(key=lambda x: (-x["score"], order.get(x["reading"], 9), x["reading"]))
    for i, c in enumerate(out, start=1):
        c["rank"] = i
    return out


def _flips(ctx: dict) -> list:
    """④ 翻盘算：哪个输入一变，结论就反过来。

    只列**材料自带的**不确定性（口径歧义 / 累积是否真作用于训练），
    不列「如果换个模型会怎样」这类凭空假设 —— 那不是核算，是空想。
    """
    items: list = []
    k = ctx.get("k") or 0
    ambig = ctx.get("ambig_k")
    if k and ambig and abs(ambig - k) > _IMPACT_TOL:
        items.append({
            "input": "代码 batch_size 的口径（单次前向值 / 累积后等效值）",
            "now": f"按单次前向值读 → 倍数 {k:g}",
            "if_changed": f"按累积后等效值读 → 倍数 {ambig:g}",
            "effect": "倍数变小，方向可能从「偏高」回到区间内 —— 这条就从问题变成没问题",
        })
    accum_c = ctx.get("accum_c")
    if accum_c and accum_c > 1 and ctx.get("p_batch") and ctx.get("c_batch"):
        raw = float(ctx["c_batch"]) / float(ctx["p_batch"])
        if abs(raw - k) > 1e-9:
            items.append({
                "input": "梯度累积是否真的作用于训练（有些实现只在评估时累积）",
                "now": f"计入累积 → 倍数 {k:g}",
                "if_changed": f"不计累积 → 倍数 {raw:g}",
                "effect": "倍数缩小，方向可能反转（累积不参与训练时，这条影响基本消失）",
            })
    return items


def _zero_when(ctx: dict) -> list:
    """④ 边界算：什么条件下这一项的影响归零。"""
    q = ctx.get("quantity") or "学习率"
    items = [f"两侧等效批大小相同时，{q}不需要跟着调整 —— 这一项影响为零"]
    accum_c = ctx.get("accum_c")
    if accum_c and accum_c > 1:
        items.append(f"论文侧其实也用了 {accum_c:g} 步梯度累积（只是正文没写）时，"
                     "倍数回到 1，影响为零")
    items.append("该参数若只在评估 / 推理阶段生效、不参与训练，它对训练过程影响为零")
    return items


def _against(paper, ctx: dict) -> dict:
    """④ 对照算：论文报的结果指标 → 复现时该看到什么。

    ⚠ 这一项**不是我们给出的结论**：它要真跑一次才能对上。本系统不做训练，
    所以这里只交「核对清单」—— 拿哪个指标比、差多少算同向、反向了怎么办。
    把它写成结论（「所以这条就是原因」）就是造假。
    """
    res = getattr(paper, "results", None) or {}
    pick = next(((m, res[m]) for m in _METRIC_LABEL if m in res), None)
    if not pick:
        return {"available": False,
                "why": "论文里没抽到可比的结果指标（准确率 / F1 / 精确率 / 召回率 / 损失 / 轮廓系数），"
                       "这一项本次做不了 —— 不编一个数来比"}
    metric, ev = pick
    return {
        "available": True,
        "metric": _METRIC_LABEL.get(metric, metric),
        "paper_value": str(getattr(ev, "raw_value", "") or ""),
        "where": str(getattr(ev, "location", "") or ""),
        "predict": _PREDICT.get(ctx.get("direction") or "",
                                "先按原值复跑一次，把实测指标与论文报告值并列看"),
        "if_match": "跑出来比报告值差 → 与推出来的方向同向，这条保留（要定案还得排除别的因素）",
        "if_opposite": "跑出来持平或更好 → 与推出来的方向反向，这条撤掉",
        "note": "对照需要真跑一次。本系统不做训练，所以这是给你（或外部实验）的核对清单，"
                "不是我们给出的结论",
    }


def _warmup_unit(key) -> str:
    """预热值的单位。返回 "steps" / "epochs" / ""（不明）。**只认 steps 才做比较。**

    为什么必须先判单位: 论文写「10k 步」、代码写「5 轮」是最常见的组合,
    直接拿 10000 和 5 比会算出「偏多 32 倍」这种看着像科学、其实无意义的结论。
    **单位不同的, 一律不比** —— 宁可少一条, 不要错认。
    """
    kk = str(key or "").lower()
    if "epoch" in kk:
        return "epochs"
    if "step" in kk or "iter" in kk:
        return "steps"
    return ""


def _impact_warmup(card, paper_vals: dict, code_vals: dict) -> dict:
    """预热步数该不该跟着批大小调整 —— **恒等关系**, 不是经验规则。

    判据: 预热期走过的样本数 = 预热步数 × 等效批大小。
    要让这个数不变, 预热步数应与等效批大小成反比。而且两种读法结论一致:
      · 按「预热样本数守恒」算     → 步数 ∝ 1/k
      · 按「预热占总步数比例不变」算 → 总步数 ∝ 1/k, 比例不变 → 步数也 ∝ 1/k
    所以这里给点值（与学习率那条不同: 那条真有两派, 只能给区间）。
    """
    eq = _equivalence(paper_vals, code_vals)
    if not eq:
        return {}
    pw_key, pw = _scan_num(_steps_only(paper_vals), _WARMUP_KEYS)
    cw_key, cw = _scan_num(_steps_only(code_vals), _WARMUP_KEYS)
    if not pw or not cw:
        return {}
    # 单位必须两侧都是「步数」才比 —— 步数和轮数混比会算出假倍数（05_ViT 就是 10k 步 vs 5 轮）
    if _warmup_unit(pw_key) != "steps" or _warmup_unit(cw_key) != "steps":
        return {}

    k = eq["k"]
    implied = pw / k
    ratio = cw / pw
    formulas = [
        f"预热期样本数（论文） = {pw:g} 步 × {eq['p_eff']:g} = {pw * eq['p_eff']:g}",
        f"等效批大小倍数 k = {k:g}（取数与批大小那条同源）",
        f"应有预热步数 = {pw:g} ÷ {k:g} = {implied:.4g}",
        f"实际预热步数之比 = 代码 {cw:g} ÷ 论文 {pw:g} = {ratio:.4g}",
    ]
    out: dict = {
        "kind": "warmup",
        "rule": IMPACT_RULES["warmup_scale"]["title"],
        "given": [
            {"side": "论文", "key": str(pw_key), "value": f"{pw:g}", "note": ""},
            {"side": "代码", "key": str(cw_key), "value": f"{cw:g}", "note": ""},
        ],
        "formulas": formulas,
        "range": {"low": round(1.0 / k, 4), "high": round(1.0 / k, 4), "unit": "倍",
                  "subject": "预热步数应有的倍数（恒等关系，给点值）"},
        "direction": "",
        "sources": [f"{IMPACT_RULES['warmup_scale']['source']}"
                    f"（{IMPACT_RULES['warmup_scale']['limit']}）"],
        "note": "",
        "compares": "预热步数",
        "magnitude": round(ratio, 6),
    }

    if abs(k - 1) <= _IMPACT_TOL:
        # 批大小没变 → 预热本来就不用调。若这时预热还差几倍, 那是**批大小解释不掉**的差异。
        if abs(ratio - 1) <= _IMPACT_TOL:
            out["direction"] = "已对齐"
            formulas.append("等效批大小没变，预热步数也一致 —— 这一项不构成问题")
        else:
            out["direction"] = "无法用批大小解释"
            formulas.append(f"等效批大小没变，任何缩放规则都推出预热不用调 —— 但实际差 {ratio:g} 倍")
    elif abs(cw - implied) <= _IMPACT_TOL * max(implied, 1e-9):
        out["direction"] = "已对齐"
        formulas.append(f"实际 {cw:g} 步与应有 {implied:.4g} 步一致 —— 预热已按批大小调整过")
    elif cw < implied:
        out["direction"] = "偏少"
        formulas.append(f"实际 {cw:g} 步 < 应有 {implied:.4g} 步 —— 偏少")
    else:
        out["direction"] = "偏多"
        formulas.append(f"实际 {cw:g} 步 > 应有 {implied:.4g} 步 —— 偏多")

    ctx = {
        "k": k, "quantity": "预热步数",
        "p_batch": eq["p_batch"], "c_batch": eq["c_batch"],
        "p_eff": eq["p_eff"], "c_eff": eq["c_eff"],
        "accum_p": eq["accum_p"], "accum_c": eq["accum_c"],
        "warmup_p": pw, "warmup_c": cw, "implied": implied, "actual": ratio,
        "readings": [{
            "reading": "按「预热期样本数守恒」读（预热步数 × 等效批大小 = 常数）",
            "low": 1.0 / k, "high": 1.0 / k,
            "evidence": [x for x in (
                f"论文 {pw_key}" if pw_key else "",
                f"代码 {cw_key}" if cw_key else "",
                f"论文 {eq['p_key']}" if eq.get("p_key") else "",
                f"代码 {eq['c_key']}" if eq.get("c_key") else "") if x],
            "no_edit": True, "order": 0,
        }],
    }
    return _done(out, ctx)


def impact_of(card, paper_vals: dict, code_vals: dict, paper=None) -> dict:
    """③④ 总入口：按卡片类型分派到对应规则，再补候选排序与四项核算。

    分派口径: 预热 / 批大小·学习率 各走各的规则, 互不借用 ——
    拿批大小规则去解释预热, 算出来的倍数看着像科学, 其实无意义。
    """
    from .synonyms import canonical_key

    kk = str(canonical_key(card.key) or card.key).lower()
    if any(t in kk for t in _WARMUP_TOKENS):
        out = _impact_warmup(card, paper_vals, code_vals)
    else:
        out = _impact_batch_lr(card, paper_vals, code_vals)
    if not out:
        return {}

    ctx = dict(out.pop("_ctx", {}) or {})
    ctx["direction"] = out.get("direction") or ""
    if out.get("kind") == "effective_batch":
        ctx["lenses"] = _lenses(ctx.get("k") or 0)
        ctx["readings"] = _readings_batch(ctx)
        ctx["actual"] = ctx.get("ratio_lr")

    cands = _candidates(ctx)
    out["lenses"] = list(ctx.get("lenses") or [])
    out["candidates"] = cands
    out["strongest"] = cands[0]["reading"] if cands else ""
    out["flip"] = _flips(ctx)
    out["zero_when"] = _zero_when(ctx)
    out["against"] = _against(paper, ctx)
    # 这一行要跟着数据走, 不只在文档里写 —— 报告/界面直接显示它, 免得看的人以为这是结论
    out["scope"] = "提示层：不进判定、不进评分、不改就绪度"
    return out


# ---------------------------------------------------------------- 排序（主持人筛排）

def _layer(card) -> int:
    """分层: 越小越该先看。

    0 已定位  —— 比"冲突"更硬, 且不用动手就能定
    1 待验·静态可判 —— 读代码就能定, 便宜
    2 待验·需实机   —— 要占 ② 站的力气
    3 已排除  —— 推演已经不认为是分歧了, 排最后（但照样列出来）
    """
    if card.verdict == DERIVE_FLAGGED:
        return 0
    if card.verdict == DERIVE_EXCLUDED:
        return 3
    return 1 if not card.needs_rerun else 2


def _breadth(card, paper, code) -> int:
    """影响面: 这个参数在两侧一共出现了多少处（出现越多, 越可能是主结果相关的设定）。"""
    n = 0
    for holder in (paper, code):
        params = getattr(holder, "params", None) or {}
        evs = params.get(card.key)
        if isinstance(evs, (list, tuple)):
            n += len(evs)
        elif evs is not None:
            n += 1
    return n


def _rerank(cards: list, paper, code) -> None:
    """重排并回填 rank —— 同分必须有稳定次序, 否则消融对照没法比。"""
    cards.sort(key=lambda c: (_layer(c), -max(int(c.score or 0), 0),
                              -_breadth(c, paper, code), c.key))
    for i, c in enumerate(cards, start=1):
        c.rank = i


# ---------------------------------------------------------------- 对外主入口

def derive(cards: list, paper=None, code=None) -> list:
    """对假设卡跑一遍三道推演, 就地写入判决并重排。返回同一个列表（方便链式调用）。

    输入:  `build_hypotheses()` 的产物 + 两侧原始材料（只为取数, 不重新抽）。
    输出:  同一个 cards 列表, 每张卡上多了 verdict / derivation / derivation_kind / engine。
    """
    paper_vals = _expand_units(_side_values(getattr(paper, "params", None)))
    code_vals = _expand_units(_side_values(getattr(code, "params", None)))

    relations: list = []
    try:
        from .compute import check_relations
        report = check_relations(paper_vals or None, code_vals or None)
        relations = list(getattr(report, "relations", None) or [])
    except Exception:      # noqa: BLE001 —— 推演失败不阻断主流程, 卡照发（宁可不推, 不能不发）
        relations = []

    eng = engine_name()
    for card in cards or []:
        # 顺序 = 结论硬度: 值非法（最硬） > 公式复算 > 倍数像什么（最软, 只是给方向）。
        # 排反了会出洋相: warmup 论文 0.05 / 代码 500 本来能被公式**算出**"换算后一致",
        # 但比值 10000 会被倍数推演先抢走, 于是一条已排除的卡被写成"仍待验"。
        hit = (_check_constraint(card, paper_vals, code_vals)
               or _check_identity(card, relations)
               or _check_scale(card, paper_vals, code_vals))
        if hit:
            card.verdict = hit["verdict"]
            card.derivation = hit["text"]
            card.derivation_kind = hit["kind"]
        else:
            card.verdict = DERIVE_KEEP
            card.derivation = ""
            card.derivation_kind = ""
        card.engine = eng
        # 阶段二 2E: 影响量级（提示层）。**只写 impact 这一个新键** ——
        # 不参与 verdict / score / rank / 就绪度, 推错也只是多一句话。
        # 这里兜住异常: 影响推演整个炸掉时, 卡照发（与 _check_* 那三段同一条纪律）。
        try:
            card.impact = impact_of(card, paper_vals, code_vals, paper)
        except Exception:      # noqa: BLE001
            card.impact = {}

    _rerank(cards or [], paper, code)
    return cards


def summarize(cards: list) -> dict:
    """一句话说清这批卡被推演分成了几堆（界面 / 报告共用, 不各写一套）。"""
    cards = list(cards or [])
    flagged = sum(1 for c in cards if getattr(c, "verdict", "") == DERIVE_FLAGGED)
    excluded = sum(1 for c in cards if getattr(c, "verdict", "") == DERIVE_EXCLUDED)
    return {
        "total": len(cards),
        "flagged": flagged,
        "excluded": excluded,
        "pending": len(cards) - flagged - excluded,
        "engine": engine_name(),
    }
