"""具名科学计算工具（阶段三 3A · 2026-10-17）—— ② 实验站的算力从「关系式复算」扩到科学计算。

一句话定位:
    2D 给了 ② 站五件工具（单位展开 / 算术求值 / 区间检查 / 倍数探测 / 关系式复算）,
    它们解决的是"**同一个数，两种写法**"—— 比大小、换单位、代公式。
    3A 补的是另一类问题: "**这串数本身说明了什么**" —— 精度会不会丢、这批测量稳不稳、
    这条趋势是不是线性的、峰在哪里。这四件事**关系式一条都算不出来**,
    所以不是重复建设, 是把工具箱从「换算器」扩成「计算器」。

为什么这四件值得做（每件都对应一类真实翻车, 不是凑数）:
    fp16_precision  论文写 `lr=1e-8`, 代码默认 fp16 —— 这个数在 fp16 里**直接下溢成 0**,
                    等于没设学习率。这是最典型的"论文写对了、代码里根本没生效"。
    series_stats    论文说"5 次运行方差很小" —— 能不能把五次的值拿出来算离散度? 算不出
                    就说明"方差很小"这句话**没有证据支撑**。
    linear_fit      论文说"精度随数据量线性增长" —— 拿三四个点拟合一次, 看 R² 到底多少。
                    说线性但 R²=0.6, 是该被记一笔的。
    peak_pick       谱图 / 曲线里峰在哪 —— 只报**图上看得见的峰位与峰值**, 不做物理解释。

⚠ 四条纪律（与 2D / 2C 同源, 别当建议）:
    1. **不造数**。输入一律由 `assemble_inputs` 从**两侧材料里取**, 取不到就如实说缺什么,
       绝不用默认值 / "典型值"补一个数出来。
    2. **不执行用户代码**。全部是纯 Python 的 `math` 运算, 不 `eval`、不 import 用户文件。
    3. **② 站只算数, 不判定**。这四件**一律不给 compare** —— 量出来的数放进"实际值"栏,
       「对得上 / 对不上」由 ③ 站下（2C 已把这条顶成结构, 别在这里破例）。
    4. **清单外不存在**。`run()` 只认注册表里的键, 提到别的直接拒（与 experiment 同款口径）。

为什么单独开一个模块而不塞进 `experiment.py`:
    `experiment.py` 的五件工具是「① 站提出的一张卡 → 怎么验它」的配套, 输入全来自那张卡;
    这四件是**通用的科学计算**, 输入来自"材料里的任意一串数"。两者的输入来源不同,
    硬塞进去会让 2D 那个已经很稳的路径变形。分开放, 各改各的, 出问题能定位到人。

口径复用（同一件事只有一处实现, 这是本项目的铁律）:
    取值转数 → `compute.to_number`（认 `3*1e-4` / `5%` / `1/2`）
    归一化键 → `synonyms.canonical_key`
    ⚠ 反向不许 import: `experiment` / `compute` / `derivation` 都不得 import 本模块。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .compute import to_number
from .synonyms import canonical_key

# ---------------------------------------------------------------- 浮点精度表
#
# 为什么只列这三档: 深度学习训练里实际只用 fp16 / bf16 / fp32。
# 再往上加（fp8 / fp4）就得引外部数值表, 而本项目不接受"为了凑工具引一个新依赖"。
#
# 表里三个数是 IEEE-754 的规格值, 不是拍脑袋:
#   最小正规数 = 2**-14 (fp16) —— 低于它进次正规区, 精度断崖
#   最小次正规数 = 2**-24        —— 低于它**直接变 0**
#   最大有限值 = 65504            —— 超过变 inf
_PRECISION: dict = {
    "fp16": {"label": "半精度 fp16", "min_normal": 2.0 ** -14,
             "min_subnormal": 2.0 ** -24, "max_finite": 65504.0,
             "digits": 3},
    "bf16": {"label": "脑浮点 bf16", "min_normal": 2.0 ** -126,
             "min_subnormal": 2.0 ** -133, "max_finite": 3.3895e38,
             "digits": 2},
    "fp32": {"label": "单精度 fp32", "min_normal": 2.0 ** -126,
             "min_subnormal": 2.0 ** -149, "max_finite": 3.4028e38,
             "digits": 7},
}
DEFAULT_PRECISION = "fp16"      # 深度学习训练的默认档; 显式写出来是为了让人一眼看见这是选择


@dataclass(frozen=True)
class SciTool:
    """一个具名科学计算工具 —— 形状与 `experiment.ExpTool` 同款（3A 不另造一套描述）。

    run(inputs) -> dict, 约定返回:
        ok    跑成没跑成（False 时 text 必须写明**为什么没跑成**、缺什么）
        text  人话结论（会进实验记录, 也会进报告）
        value 量出来的数（有就给）
        actual / actual_from —— 实测结果与它的出处（⚠ **不给 compare**, 见模块头第 3 条）
    """
    key: str
    name: str
    desc: str
    cost: str
    needs: tuple               # 需要什么输入（装配器照此从材料里取）
    gives: str
    run: Callable[[dict], dict]


# ---------------------------------------------------------------- 工具实现

def _to_float_list(raw: Any) -> tuple:
    """把一串取值转成 float 列表 —— 不认识的**逐项剔掉并说明**, 不猜、不补。

    返回 `(values, skipped)`: skipped 是被剔掉的原样字符串, 供工具如实写进 text。
    为什么剔而不是整条失败: 一串 5 个数里有 1 个是 "Adam", 剩下 4 个照样能算;
    但**剔掉这件事必须让用户看见**, 否则就成了"悄悄少算了一个"。
    """
    if raw is None:
        return [], []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    vals: list = []
    skipped: list = []
    for it in items:
        n = to_number(it)
        if n is None:
            skipped.append(str(it))
        else:
            vals.append(float(n))
    return vals, skipped


def _skipped_note(skipped: list) -> str:
    if not skipped:
        return ""
    shown = "、".join(f"`{s}`" for s in skipped[:5])
    more = f"（另有 {len(skipped) - 5} 项）" if len(skipped) > 5 else ""
    return f"　其中 {len(skipped)} 项读不出数已剔掉: {shown}{more}"


def _t_fp16_precision(inp: dict) -> dict:
    """浮点精度检查: 这个取值在指定精度下会不会丢成 0 / 溢出成 inf。

    **这是本工具箱里最容易出真问题的一件**: 论文写 `1e-8` 看着完全正常, 但 fp16 的
    最小次正规数是 `2**-24 ≈ 5.96e-8` —— 1e-8 比它还小, **在 fp16 里就是 0**。
    设了等于没设, 而且代码不报错、训练照跑, 属于最难发现的一类。
    """
    raw = inp.get("raw")
    mode = str(inp.get("precision") or DEFAULT_PRECISION).strip().lower()
    spec = _PRECISION.get(mode)
    if spec is None:
        return {"ok": False,
                "text": f"没有 `{mode}` 这一档精度（只登记了 fp16 / bf16 / fp32）, "
                        f"不替它编一档出来"}
    vals, skipped = _to_float_list([raw] if not isinstance(raw, (list, tuple)) else raw)
    if not vals:
        return {"ok": False,
                "text": f"没有可检查的数值取值（这一侧没抽到值）"
                        + _skipped_note(skipped)}
    # ⚠ 只报**最危险**的那个数, 不逐个列 —— 报了用户也不会逐个看。
    worst = min((v for v in vals if v != 0), key=abs, default=None)
    if worst is None:
        return {"ok": True, "value": 0.0,
                "text": f"取值就是 0（{spec['label']} 下同样落成 0）—— "
                        f"这一项本身就没有生效"}
    a = abs(worst)
    if a < spec["min_subnormal"]:
        return {
            "ok": True, "value": 0.0, "actual": 0.0,
            "actual_from": f"材料取值 {worst:g} 在 {spec['label']} 下的实际落值",
            "text": (f"⚠ `{worst:g}` 在 {spec['label']} 里**落成 0**"
                     f"（该格式最小非零数是 {spec['min_subnormal']:.3g}）—— "
                     f"这一项等于没设, 而且**不会报错**。"
                     f"要真用上这个量级, 得换 bf16/fp32 或改成梯度裁剪后的自适应尺度"),
        }
    if a < spec["min_normal"]:
        return {
            "ok": True, "value": worst, "actual": worst,
            "actual_from": f"材料取值 {worst:g} 在 {spec['label']} 下的实际落值",
            "text": (f"`{worst:g}` 落在 {spec['label']} 的**次正规区**"
                     f"（< {spec['min_normal']:.3g}）—— 有效数字只剩约 "
                     f"{spec['digits']} 位, 数值本身能用但精度已经明显掉"),
        }
    if a > spec["max_finite"]:
        return {
            "ok": True, "value": float("inf"), "actual": float("inf"),
            "actual_from": f"材料取值 {worst:g} 在 {spec['label']} 下的实际落值",
            "text": (f"⚠ `{worst:g}` 超过 {spec['label']} 的最大有限值 "
                     f"{spec['max_finite']:.6g} —— **溢出成 inf**"),
        }
    return {
        "ok": True, "value": worst, "actual": worst,
        "actual_from": f"材料取值 {worst:g} 在 {spec['label']} 下的实际落值",
        "text": (f"`{worst:g}` 在 {spec['label']} 下**正常可表示**"
                 f"（有效数字约 {spec['digits']} 位）"),
    }


def _t_series_stats(inp: dict) -> dict:
    """一串测量的集中趋势与离散度: n / 均值 / 中位 / 样本标准差 / 离散度。

    为什么用**样本**标准差（除以 n-1）而不是总体标准差: 论文里"跑 5 次"的 5 个数
    是从更大的分布里抽出来的样本, 除以 n 是总体口径, 会系统性地偏小 —— 这种偏差
    在 n=3 时能到 20%, 拿来当"方差很小"的证据就是假的。
    """
    vals, skipped = _to_float_list(inp.get("values"))
    if len(vals) < 2:
        return {"ok": False,
                "text": (f"只有 {len(vals)} 个可用取值（算离散度至少要 2 个）"
                         + _skipped_note(skipped)
                         + "　—— 样本量这么小, 论文里「方差很小」这类说法没有证据支撑")}
    n = len(vals)
    mean = sum(vals) / n
    srt = sorted(vals)
    med = (srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)      # 样本方差（n-1）
    sd = math.sqrt(var) if var > 0 else 0.0
    cv = abs(sd / mean) if mean else None                  # 离散度（无量纲, 可比）
    out = {
        "ok": True, "value": sd,
        "actual": (cv if cv is not None else sd),
        "actual_from": f"材料里同一项的 {n} 个取值（样本标准差 / 均值, n-1 口径）",
        "text": (f"这 {n} 个取值: 均值 {mean:.6g} / 中位 {med:.6g} / "
                 f"样本标准差 {sd:.6g}"),
    }
    if cv is not None:
        out["text"] += f" / 离散度 {cv * 100:.2f}%"
    out["text"] += f", 极差 {max(vals) - min(vals):.6g}" + _skipped_note(skipped)
    return out


def _t_linear_fit(inp: dict) -> dict:
    """最小二乘直线拟合: 斜率 / 截距 / R²。

    为什么值得做: 论文常说"精度随数据量线性增长", 但常常只有 3 个点。
    3 个点拟合出的 R² 是**能证伪这句话的最便宜办法** —— 说线性而 R² 只有 0.6,
    这句话就该被记一笔, 而不是照抄进报告。
    """
    xs, sk_x = _to_float_list(inp.get("xs"))
    ys, sk_y = _to_float_list(inp.get("ys"))
    skipped = sk_x + sk_y
    if len(xs) != len(ys):
        return {"ok": False,
                "text": (f"横纵坐标个数对不上（{len(xs)} vs {len(ys)}）—— "
                         f"拟合要求成对, 凑不齐就不硬配" + _skipped_note(skipped))}
    n = len(xs)
    if n < 3:
        return {"ok": False,
                "text": (f"只有 {n} 个点 —— 直线拟合至少要 3 个点才有意义"
                         f"（2 个点必然完美贴合, 那不是规律是巧合）"
                         + _skipped_note(skipped))}
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return {"ok": False,
                "text": ("横坐标全都一样 —— 这几个点连成的是竖线, "
                         "拟合不出斜率（不硬凑一个 0 上去）" + _skipped_note(skipped))}
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    slope = sxy / sxx
    intercept = my - slope * mx
    # R²: 总平方和为 0（所有 y 一样）时定义为 1 —— 横向一条线, 任何斜率都"解释得动",
    # 但那本身不是"线性增长", 所以如实说这种情况, 不拿 R²=1 冒充拟合得很好。
    if syy <= 0.0:
        r2 = 1.0
        note = "（所有纵坐标都相同, R² 按惯例记 1 —— 但这说明的是「没有变化」, 不是「线性关系」)"
    else:
        r2 = (sxy * sxy) / (sxx * syy)
        note = ""
    text = (f"{n} 个点最小二乘: 斜率 {slope:.6g} / 截距 {intercept:.6g} / "
            f"R² {r2:.4f}{note}")
    if r2 < 0.8 and syy > 0.0:
        text += "。⚠ R² 偏低 —— 若材料里声称这条关系是线性的, 这几个点并不支持它"
    return {"ok": True, "value": slope, "actual": r2,
            "actual_from": f"材料里这 {n} 对取值的最小二乘拟合优度",
            "text": text + _skipped_note(skipped)}


def _t_peak_pick(inp: dict) -> dict:
    """序列峰位定位: 报峰在哪、峰值多少, 以及**它是不是端点**。

    ⚠ 与 2E 图表解析的分工: 2E 是**模型看图**, 只报看得见的东西, 解释当场丢弃;
    这一件是**从一串数里找峰**, 不看图。所以它报的是峰位坐标与峰值这类**可复算的数**,
    不做"这个峰说明什么"的物理解释 —— 解释归 ① 站提假设, 判定归 ③ 站。
    """
    xs, sk_x = _to_float_list(inp.get("xs"))
    ys, sk_y = _to_float_list(inp.get("ys"))
    skipped = sk_x + sk_y
    if len(xs) != len(ys):
        return {"ok": False,
                "text": (f"横纵坐标个数对不上（{len(xs)} vs {len(ys)}）—— "
                         f"找峰要求成对" + _skipped_note(skipped))}
    n = len(ys)
    if n < 3:
        return {"ok": False,
                "text": f"只有 {n} 个点 —— 找不出「峰」（峰要求左右两侧都有更低的点）"
                        + _skipped_note(skipped)}
    order = max if "峰" in str(inp.get("mode") or "峰") else min
    bi = max(range(n), key=lambda i: ys[i]) if order is max else min(range(n), key=lambda i: ys[i])
    at_edge = bi in (0, n - 1)
    base = {"ok": True, "value": ys[bi],
            "actual": xs[bi],
            "actual_from": (f"材料里这 {n} 个点中第 {bi + 1} 个的横坐标"
                            f"（峰值 {ys[bi]:.6g}）"),
            "text": (f"{n} 个点里最高点在横坐标 {xs[bi]:.6g} 处, 峰值 {ys[bi]:.6g}"
                     + _skipped_note(skipped))}
    if at_edge:
        base["text"] += ("。⚠ 最高点落在**端点**上 —— 采样范围还没罩住整个峰, "
                         "这个位置只是边界, 不是真的峰顶")
    else:
        lo = min(ys[bi - 1], ys[bi + 1])
        base["text"] += f", 两侧邻点都更低（{min(ys[bi-1], ys[bi+1]):.6g} / {max(ys[bi-1], ys[bi+1]):.6g}）"
    return base


SCI_TOOLS: tuple = (
    SciTool(key="fp16_precision", name="浮点精度检查",
            desc="核这个取值在 fp16 / bf16 / fp32 下会不会下溢成 0 或溢出成 inf"
                 "（论文里看着正常的极小值, 在半精度下常常直接变 0）",
            cost="本地、零成本", needs=("一个或多个取值", "精度档位（默认 fp16）"),
            gives="实际落值 + 是否有精度损失", run=_t_fp16_precision),
    SciTool(key="series_stats", name="序列统计",
            desc="对同一项的多个取值算均值 / 中位 / 样本标准差 / 离散度"
                 "（论文说「方差很小」时, 这是唯一的证据来源）",
            cost="本地、零成本", needs=("同一项的多个取值（≥2）"),
            gives="均值 / 标准差 / 离散度", run=_t_series_stats),
    SciTool(key="linear_fit", name="直线最小二乘拟合",
            desc="对若干 (x, y) 点拟合直线, 报斜率 / 截距 / R²"
                 "（材料声称「线性增长」时, R² 就是这句话的可证伪性）",
            cost="本地、零成本", needs=("横坐标序列", "纵坐标序列（≥3 对）"),
            gives="斜率 / 截距 / R²", run=_t_linear_fit),
    SciTool(key="peak_pick", name="峰位定位",
            desc="在一串 (x, y) 点里定位最高 / 最低点, 并说明它是不是端点"
                 "（只报坐标与数值, 不做物理解释）",
            cost="本地、零成本", needs=("横坐标序列", "纵坐标序列（≥3 对）"),
            gives="峰位坐标 + 峰值", run=_t_peak_pick),
)

_TOOL_BY_KEY: dict = {t.key: t for t in SCI_TOOLS}


def catalog_keys() -> frozenset:
    """白名单 —— 可被调用的科学工具键全集（清单外 = 不存在）。"""
    return frozenset(t.key for t in SCI_TOOLS)


def menu() -> list:
    """给模型看的科学工具菜单（JSON 友好形状, 字段名用中文更好读）。

    与 `experiment.menu()` 同款形状 —— 上层把两份并起来递就行, 不必为 3A 再写一套描述。
    """
    return [{"key": t.key, "名称": t.name, "说明": t.desc, "成本": t.cost,
             "需要输入": list(t.needs), "产出": t.gives} for t in SCI_TOOLS]


def engine_name() -> str:
    """本次用的引擎名 —— 与 2B / 2D 同源（内置纯 Python 运算, 不引外部数值库）。

    为什么不引 sympy / numpy: 3A 要的是"把已知的数算清楚", 项目已有等价基座
    （`ast` 白名单 + 纯 Python 的 `math`）。再引一套只会多一条装不上的路径,
    还会让同一条公式出现两种算法 —— 本项目最忌讳这个（见 `derivation.engine_name`）。
    """
    return "内置科学计算工具（纯 Python math 运算, 无外部数值依赖）"


def run(tool_key: str, inputs: Optional[dict] = None) -> dict:
    """跑一件科学工具 —— **唯一的调用入口**, 且只认注册表里的键。

    为什么单独封一个入口（而不是让调用方直接拿 `_TOOL_BY_KEY`）:
      ① 白名单在这一处**物理成立** —— 提到清单外的键直接拒, 不是"记一笔然后照跑";
      ② 异常在这里**兜住** —— 单个工具炸了只该毁掉这一步, 不该拖垮整轮;
      ③ 返回形状**统一** —— ok / text 一定有, 缺了就补, 调用方不必到处判空。

    ⚠ 这里刻意**不返回 compare**: ② 站只算数, 判定归 ③ 站（2C 已把这条顶成结构）。
    """
    key = str(tool_key or "")
    tool = _TOOL_BY_KEY.get(key)
    if tool is None:
        return {"ok": False,
                "text": (f"没有名为 `{key}` 的科学计算工具 —— 菜单外的东西**本次不存在**"
                         f"（可用的: {', '.join(sorted(_TOOL_BY_KEY))}）")}
    try:
        out = dict(tool.run(dict(inputs or {})))
    except Exception as ex:      # noqa: BLE001 —— 单个工具炸了只该毁掉这一步
        return {"ok": False, "text": f"工具执行出错: {type(ex).__name__}: {ex}"}
    out.setdefault("ok", False)
    out.setdefault("text", "")
    out.pop("compare", None)      # 兜底: 即使实现里手滑写了 compare, 这里也不让它出去
    return out


# ---------------------------------------------------------------- 输入装配
#
# 纪律第 1 条的落地: **输入只能从材料里取**。这一段是"从两侧参数表 → 工具输入"的唯一通道,
# 取不到就如实说缺什么, 不猜、不补。所以模型只挑**工具键**（3A 同 2D 的口径）,
# 输入一律由这里装配 —— 模型一旦能自己填输入, "不造数"当场就破了。

# 每个科学工具要哪些参数键 —— 键名走 `synonyms.canonical_key` 归一, 与全项目同一套。
# ⚠ 只登记**常见可核**的一类: 需要成套多参数才能算的（模型结构、硬件配置）不在此列,
#   因为审计材料里极少同时抽齐, 登记了就是"菜单上有、永远跑不动"。
_NEEDS_KEYS: dict = {
    "fp16_precision": (),
    "series_stats": (),
    "linear_fit": (),
    "peak_pick": (),
}


def assemble_inputs(tool_key: str, *, card=None,
                    paper_vals: Optional[dict] = None,
                    code_vals: Optional[dict] = None) -> tuple:
    """把一件科学工具的输入从材料里装配出来 —— 返回 `(inputs, note)`。

    `note` 是**没装配成但必须告诉人的事**（例如"材料里同一项只有一个取值, 算不了离散度"）,
    它进实验**设计**那一栏, 不进执行记录 —— 同一件事两种表达, 选好那种。

    怎么装配（规则很窄, 宁可装不上也不硬凑）:
      · `fp16_precision` —— 这张卡**自己**的两侧取值就是输入（没有比"这一项"更贴的输入）;
      · `series_stats` —— 找两侧同名参数的**多个**取值（论文可能在正文和表格里各写一次）;
      · `linear_fit` / `peak_pick` —— 需要成对的 (x, y), 材料里几乎不会抽齐,
        所以这里**如实说装不上**, 不拿两侧的单值凑一条假曲线。

    ⚠ 一条纪律: 序列类工具的输入**只从材料里已有的多个取值来**, 绝不用"等间隔 1..N"
       这种合成坐标 —— 那等于自己造一条曲线再拟合它, 拟合优度全无意义。
    """
    key = str(tool_key or "")
    if key not in _TOOL_BY_KEY:
        return {}, f"没有名为 `{key}` 的科学工具, 无从装配输入"

    from .compute import _lookup      # 局部导入: 避免与 experiment 形成导入环
    pv = dict(paper_vals or {})
    cv = dict(code_vals or {})
    ckey = str(getattr(card, "key", "") or "")

    if key == "fp16_precision":
        raws = []
        for tbl, side in ((pv, "论文"), (cv, "代码")):
            v = _lookup(tbl, ckey)
            if v not in (None, ""):
                raws.append(v)
        if not raws:
            return {}, f"两侧都没抽到 `{ckey}` 的确定取值 —— 没有可检查的数"
        return ({"raw": raws,
                 "precision": _guess_precision(pv, cv) or DEFAULT_PRECISION}, "")

    if key == "series_stats":
        # 同一项的多个取值: 两张表里同名键的值, 加上 params_all 里的历史取值
        vals: list = []
        for tbl in (pv, cv):
            v = _lookup(tbl, ckey)
            if v not in (None, ""):
                vals.append(v)
        extra = _multi_values(getattr(card, "paper_values", None),
                              getattr(card, "code_values", None))
        vals.extend(extra)
        uniq = list(dict.fromkeys(str(v) for v in vals))
        if len(uniq) < 2:
            return ({"values": uniq} if uniq else {},
                    f"材料里 `{ckey}` 只出现 1 次 —— 算离散度至少要 2 个取值。"
                    f"论文若声称这项「很稳定」, 本次**拿不出证据**")
        return {"values": uniq}, ""

    # linear_fit / peak_pick —— 如实说装不上, 不合成坐标
    return {}, (f"`{key}` 需要成对的 (x, y) 序列; 审计材料抽出来的是**单个超参数取值**, "
                f"凑不成一条曲线。不用等间隔坐标硬凑 —— 那样拟合出来的 R² 是自己造的")


def _multi_values(*seqs) -> list:
    """从一个或多个取值序列里摊出可用值（每个元素本身可能是列表）。"""
    out: list = []
    for s in seqs:
        if not s:
            continue
        if isinstance(s, (list, tuple)):
            for it in s:
                if isinstance(it, (list, tuple)):
                    out.extend(it)
                else:
                    out.append(it)
        else:
            out.append(s)
    return out


def _guess_precision(pv: dict, cv: dict) -> Optional[str]:
    """材料里**明写了**精度档位就照它走, 没写就返回 None（由上层落回默认档）。

    ⚠ 只认明确的写法（`fp16` / `float16` / `half` …）, 拿"数值很小"去**猜**精度是编造 ——
       同一个值在小数下可能是真小, 也可能是 fp32 下的正常值。
    """
    for tbl in (pv, cv):
        for k, v in (tbl or {}).items():
            kk = str(k).lower()
            vv = str(v).lower()
            if "fp16" in kk or "float16" in kk or "fp16" in vv or "float16" in vv:
                return "fp16"
            if "bf16" in kk or "bfloat16" in kk or "bf16" in vv or "bfloat16" in vv:
                return "bf16"
            if "fp32" in kk or "float32" in kk or "fp32" in vv or "float32" in vv:
                return "fp32"
    return None


def summarize(records: list) -> dict:
    """一句话说清这批科学工具调用的情况（界面 / 报告共用, 不各写一套）。"""
    recs = list(records or [])
    used: list = []
    calls = 0
    ok_n = 0
    for r in recs:
        calls += 1
        if r.get("ok"):
            ok_n += 1
        k = str(r.get("tool") or "")
        if k and k not in used:
            used.append(k)
    return {"calls": calls, "ok": ok_n, "tools": used, "engine": engine_name()}
