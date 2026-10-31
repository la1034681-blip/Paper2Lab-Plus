"""实验设计与真跑（阶段二 2D · 2026-10-15）—— ② 实验站。

一句话定位:
    ① 站（`hypothesis.py`）说"我们怀疑什么"; 2B（`derivation.py`）先**不动手**推了一遍,
    把真分歧 / 伪分歧 / 一侧写错分开。2D 是接着往下走的那一步 ——
    对**推不掉的**待验假设, **真的动手算一遍**, 并把「怎么算的、喂了什么、算出什么」
    一条条记下来。

为什么必须"真跑", 而不只是"再推一遍":
    推演是**说理**（"按公式应该对得上"）; 实验是**执行**（"我调了这个工具、输入是这些、
    输出是这个、耗时 0.3ms"）。两者在报告里的可信度不是一回事 ——
    前者读者只能选择信或不信, 后者读者可以**照着复现**。
    ② 站的产出因此是一份**实验记录**, 不是又一段论断。

② 站的实验卡上必须有四段, 缺一段就不算做过实验:
    设计    —— 要算什么、为什么这么算（不写"研究一下"这种话）
    调用    —— 用了哪些**具名工具**, 顺序也记下来（先归一 → 再核区间 → 再量倍数）
    记录    —— 每一步的输入 / 输出 / 耗时 / 成没成（可核验、可复现）
    结果    —— 实际值 + 它是从哪来的（**绝不能和预期值同源**, 否则就是自证）

⚠ 三条纪律（都是本项目已经付过代价的, 别当建议）:
    1. **不造数**。输入只从已抽到的值里取; 凑不齐就记「算不出来」, 并写明缺哪一项。
       **绝不**用默认值 / "常见取值"补一个数出来充场面。
    2. **不执行用户代码**。所有"算"都走 `compute.eval_arith` 的 `ast` 白名单,
       以及 `compute` / `collider` / `derivation` 已有的纯函数。② 站不跑训练、不装环境。
    3. **非自证**。预期值与实际值必须来自**不同的**输入（例: 预期 = 论文声明的比例,
       实际 = 用代码的步数除出来的比例）。同源的两个数按理必然相等, 那种"验证"
       什么都验不出来 —— `compute` 模块头把这条写得很清楚, 这里照办, 并加了运行时守卫。

五个观察态（注意用词: 这是**观察**, 不是"假设成立 / 不成立" —— 判定是 ③ 站的事）:
    已复算·一致        工具跑出来的数与预期吻合
    已复算·对不上      跑出来的数与预期不符（最该看的一类）
    已实测量·不下结论  量出了可核验的数（倍数 / 区间 / 复算过程）,
                       但它本身**不构成结论** —— ② 站只负责测, 判定归 ③ 站
    算不出来          输入凑不齐 / 写法无法归一 —— 如实说, 不硬凑（并写明缺的是什么）
    需实机验证        这条要 GPU / 训练才能定; 本步**不假装算过**, 改给一份可执行的实机方案
    （另加一个「无需实验」= ① 站推演已定性, 不重复验 —— 好的实验设计首先是"哪些不用做"）

口径纪律（与 2B 同源, 不另造一套 —— 本项目最忌讳"同一件事两种说法"）:
    取值归一 → `compute.to_number` / `eval_arith`（认 `3*1e-4` 这种写法）
    单位展开 → `derivation._expand_units`（认 `10k` / `1M` / `3万`）
    合法区间 → `collider.numeric_check`（与「计算员」在 ③ 站用的是同一张表）
    倍数来源 → `derivation._find_factor`
    关系式   → `compute.check_relations`
    ⚠ 反向不许 import: `compute` / `collider` / `hypothesis` / `derivation` 都不得 import 本模块。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .collider import numeric_check
from .compute import RELATIONS, check_relations, eval_arith, to_number
from .derivation import (DERIVE_EXCLUDED, DERIVE_FLAGGED, _expand_units,
                         _find_factor)
# 阶段三 3A: 具名科学计算工具（浮点精度 / 序列统计 / 直线拟合 / 峰位定位）。
# 本模块只做**适配层**（把 scifuncs 的工具包成 ② 站能调的形状）＋一条规则触发,
# 真正的实现在 scifuncs —— 不复制一份, 免得同一件事两种算法。
from .scifuncs import _PRECISION      # noqa: F401  (测试与调试常引用阈值)
from . import scifuncs

# ---------------------------------------------------------------- 观察态

EXP_NO_NEED = "无需实验"
EXP_AGREE = "已复算·一致"
EXP_DISAGREE = "已复算·对不上"
EXP_MEASURED = "已实测量·不下结论"
EXP_UNDECIDED = "算不出来"
EXP_MACHINE = "需实机验证"

STATUS_DESC: dict = {
    EXP_DISAGREE: ("工具跑出来的数与预期**不符** —— 这是最该看的一类"
                   "（本步只做观察, 成立与否的判定归 ③ 站）"),
    EXP_MACHINE: "这条要 GPU / 训练才能定 —— 本步不假装算过, 改给一份可执行的实机方案",
    EXP_MEASURED: ("量出了可核验的数（倍数 / 区间 / 复算过程）, 但**它本身不构成结论** —— "
                   "② 站只负责测, 判定归 ③ 站"),
    EXP_AGREE: "工具跑出来的数与预期吻合 —— 这条分歧在数值上已被抹平",
    EXP_UNDECIDED: "输入凑不齐或写法无法归一 —— 如实记为算不出来, 未用默认值补数",
    EXP_NO_NEED: "① 站推演已定性（已排除 / 已定位）—— 不重复验, 避免空转",
}

# 排序: 越靠前越该被人先看到。
# 对不上（真问题）> 要动手的实机 > 量出来待判 > 已抹平 > 算不出来 > 不用做。
_STATUS_ORDER = {EXP_DISAGREE: 0, EXP_MACHINE: 1, EXP_MEASURED: 2,
                 EXP_AGREE: 3, EXP_UNDECIDED: 4, EXP_NO_NEED: 5}

# 每张实验卡的展示上限（与 hypothesis.HYPOTHESIS_MAX / patch 的 max_items 同口径）
EXPERIMENT_MAX = 8

# 工具路径上限 —— 防止一张卡被拼出十几步来刷"工作量"。三步足够: 归一 → 核区间 → 量倍数。
MAX_STEPS = 3


# ---------------------------------------------------------------- 具名计算工具
#
# 这份清单就是 ② 站能"调"的全部计算工具（一张菜单, 不是一道闸门）:
# 清单外的东西根本不存在, 所以"乱调工具"在物理上不成立。
# 每个工具都**真能跑**, 且跑的当天就是 `compute` / `collider` / `derivation` 里的同一批函数 ——
# 绝不为了"看起来有工具"而包一层空壳。

@dataclass(frozen=True)
class ExpTool:
    """一个可被 ② 站调用的具名计算工具。

    run(inputs) -> dict, 约定返回:
        ok      这一步跑成没跑成（False 时 text 必须写明**为什么没跑成**）
        text    人话结论（会进实验记录, 也会进报告）
        value   数值结果（有就给）
        compare 与预期是否吻合: True / False / None（None = 这一步不下"一致/不一致"的判断）
        expected / actual / expected_from / actual_from —— compare 不为 None 时必填,
                且**两个来源必须不同**（非自证, 见模块头第 3 条）
    """
    key: str
    name: str
    desc: str
    cost: str
    needs: tuple
    gives: str
    run: Callable[[dict], dict]


def _t_expand_units(inp: dict) -> dict:
    """单位后缀展开: `10k` → 10000、`1M` → 1000000、`3万` → 30000。"""
    raw = str(inp.get("raw") or "").strip()
    if not raw:
        return {"ok": False, "text": "没有可展开的取值（这一侧没抽到值）"}
    out = _expand_units({"v": raw}).get("v")
    if out is None or str(out) == raw:
        return {"ok": False,
                "text": f"`{raw}` 不是「数字 + 单位后缀」的写法, 单位展开动不了它"}
    v = to_number(out)
    if v is None:
        return {"ok": False, "text": f"`{raw}` 展开成 `{out}` 后仍读不出数"}
    return {"ok": True, "value": v, "text": f"`{raw}` 按单位后缀展开 = {v:g}"}


def _t_eval_expr(inp: dict) -> dict:
    """算术表达式求值: `3*1e-4` → 0.0003、`2**10` → 1024。

    只走 `ast` 白名单 —— 代码包是用户上传的, `eval` 等于把执行权交出去。
    """
    raw = str(inp.get("raw") or "").strip()
    if not raw:
        return {"ok": False, "text": "没有可求值的取值（这一侧没抽到值）"}
    v = eval_arith(raw)
    if v is None:
        return {"ok": False,
                "text": (f"`{raw}` 不是可求值的算术表达式"
                         f"（只放行数字、四则运算、乘方与 pi/e 两个常量）")}
    return {"ok": True, "value": v, "text": f"`{raw}` 求值 = {v:g}"}


def _t_bounds_check(inp: dict) -> dict:
    """合法区间检查 —— 复用 `collider.numeric_check` 的同一张表（与 ③ 站同源）。"""
    key = str(inp.get("key") or "")
    raw = str(inp.get("value") or "").strip()
    if not raw:
        return {"ok": False, "text": "没有可检查的取值（这一侧没抽到值）"}
    r = numeric_check(key, raw)
    note = str(r.get("note") or "")
    if "无预设合理区间" in note:
        return {"ok": False,
                "text": f"`{key}` 没有预设的合法区间表项 —— 不硬编一个区间来凑这个检查"}
    if "非数值型" in note:
        return {"ok": False, "text": f"`{key}` 是非数值型参数, 区间检查不适用"}
    ok = bool(r.get("plausible"))
    base: dict = {"ok": True, "value": ok, "text": f"核 `{key} = {raw}`：{note}"}
    if not ok:
        # ⚠ 只有**越界**才当成"对不上"报出来。
        #   在区间内**不下结论** —— 值合法 ≠ 这条假设被验证了。
        #   （早先版本给合法也打 compare=True, 于是每张卡都变成「已复算·一致」,
        #     用户看到的是"验过了", 实际只说明"这个数不离谱"。那是自欺。）
        base.update(compare=False, expected="落在合法区间内",
                    expected_from="该参数的合法区间表（collider 与 ③ 站共用）",
                    actual=note, actual_from=f"{key} = {raw}（材料原文取值）")
    return base


def _t_ratio_probe(inp: dict) -> dict:
    """倍数探测: 两侧差几倍, 材料里有没有能解释这个倍数的量。

    这是 ② 站最常用的一步 —— 批大小 / 步数 / 并行度这类"差整数倍"的分歧,
    多半是口径差（一侧写单卡、一侧写全局）而不是真写错。
    """
    key = str(inp.get("key") or "")
    a, b = inp.get("a"), inp.get("b")
    na, nb = to_number(a), to_number(b)
    if not na or not nb or na <= 0 or nb <= 0:
        return {"ok": False,
                "text": f"两侧取值 `{a}` / `{b}` 里有一侧读不出正数, 量不出倍数"}
    r = nb / na
    ratio = r if r >= 1 else 1.0 / r
    bigger = "代码" if r >= 1 else "论文"
    if ratio < 1.5:
        # 几乎相等: 量倍数没有意义（防呆 —— 计划阶段也会拦, 但模型单独挑这个工具时得挡住）
        return {"ok": False,
                "text": (f"两侧取值几乎相等（论文 {na:g} / 代码 {nb:g}, 相差 {ratio:g} 倍）, "
                         f"量倍数没有意义")}

    fx = inp.get("factor_value")
    fname = str(inp.get("factor_name") or "")
    fraw = str(inp.get("factor_raw") or "")
    if fx and abs(ratio - fx) <= max(1e-9, 1e-6 * max(ratio, fx)):
        return {
            "ok": True, "value": ratio, "compare": True,
            "expected": float(fx),
            "expected_from": f"材料里声明的{fname}（{fraw} = {fx:g}）",
            "actual": ratio,
            "actual_from": f"两侧取值之比（代码 {nb:g} ÷ 论文 {na:g}）",
            "text": (f"两侧 {key} 相差 {ratio:g} 倍（论文 {na:g} / 代码 {nb:g}, {bigger}侧更大）, "
                     f"材料里写着 {fraw} = {fx:g}（{fname}）—— 倍数**正好对得上**: "
                     f"一侧是单卡值、另一侧是全局值, 不是真分歧"),
        }
    return {
        "ok": True, "value": ratio,
        # 解释不掉 ≠ 没测到: 倍数本身就是**实测结果**, 必须留在"实际值"那一栏,
        # 只是不替它下"一致 / 不一致"的判断（compare 留空）。
        "actual": ratio,
        "actual_from": f"两侧取值之比（代码 {nb:g} ÷ 论文 {na:g}）",
        "text": (f"两侧 {key} 相差 {ratio:g} 倍（论文 {na:g} / 代码 {nb:g}, {bigger}侧更大）。"
                 + (f"材料里 {fraw} = {fx:g}（{fname}）解释不掉这个倍数。"
                    if fx else "材料里没有与并行度 / 累积相关的量可供比对。")
                 + "本步只把倍数量出来, 不外推原因"),
    }


def _t_relation_eval(inp: dict) -> dict:
    """关系式复算: 把两侧已知值代进已登记的公式（warmup 换算 / lr 缩放 / 每轮步数）。

    输入是**两侧参数表**（不是单值）—— 关系式往往需要别的参数配合（如 warmup 需要总步数）。
    真跑走 `compute.check_relations`, 与闭环「算」那一步是同一套公式、同一个入口。
    """
    rid = str(inp.get("rid") or "")
    paper, code = inp.get("paper"), inp.get("code")
    try:
        rep = check_relations(paper or None, code or None)
    except Exception as ex:      # noqa: BLE001
        return {"ok": False, "text": f"关系式复算没能跑起来: {type(ex).__name__}: {ex}"}
    hit = next((r for r in (rep.relations or []) if r.get("rid") == rid), None)
    if hit is None:
        return {"ok": False, "text": f"本次没有登记名为 `{rid}` 的关系式, 复算不了"}
    verdict = hit.get("verdict")
    formula = str(hit.get("formula") or "")
    detail = str(hit.get("detail") or "")
    if verdict == "skipped":
        return {"ok": False, "text": f"关系式 `{rid}` 本次未算：{detail}"}
    base = {
        "ok": True, "value": hit.get("actual"),
        "text": (f"按公式 `{formula}` 复算：{detail}"
                 f"（本次喂进去的输入: {hit.get('inputs') or '—'}）"),
    }
    if verdict == "consistent":
        base.update(compare=True, expected=hit.get("expected"), actual=hit.get("actual"),
                    expected_from=f"公式预期值（{formula} 的左侧）",
                    actual_from="另一侧自己声明的值（材料原文）")
    elif verdict == "mismatch":
        base.update(compare=False, expected=hit.get("expected"), actual=hit.get("actual"),
                    expected_from=f"公式预期值（{formula} 的左侧）",
                    actual_from="另一侧自己声明的值（材料原文）")
    else:
        # hint / info —— 关系式自己写着"需人工确认 / 不作为结论", 本步**不许**替它下结论。
        # 但复算出来的两个数照样要留下（测到了就要显示, 只是不给 compare）。
        base["text"] += "。这一条关系式本身标着「需人工确认」, 所以本步只呈现复算过程, 不下判断"
        if hit.get("actual") is not None or hit.get("expected") is not None:
            base.update(expected=hit.get("expected"), actual=hit.get("actual"),
                        expected_from=f"公式预期值（{formula} 的左侧）",
                        actual_from="另一侧自己声明的值（材料原文）")
    return base


def _sci_runner(tool_key: str) -> Callable[[dict], dict]:
    """3A 适配层: 把 `scifuncs` 的一件工具包成 ② 站能调的形状。

    为什么是「包一层」而不是把 scifuncs 的 run 直接摆进清单:
      · ② 站的 `_run_steps` 期望的返回形状比 scifuncs **宽**（多 ok/text 兜底、异常兜底）,
        这层保证单件工具炸了只毁掉这一步, 不会拖垮整条卡;
      · scifuncs 那边的纪律（**永不给 compare**）由 `scifuncs.run` 自己顶住, 这层不再重复。
    真正的实现、边界与不造数纪律全在 `scifuncs`, 这里一行逻辑都不复制 —— 免得两处漂。
    """
    from . import scifuncs

    def _run(inp: dict) -> dict:
        return scifuncs.run(tool_key, inp)
    _run.__name__ = f"_sci_{tool_key}"
    return _run


EXPERIMENT_TOOLS: tuple = (
    ExpTool(key="expand_units", name="单位展开",
            desc="把 10k / 1M / 3万 这类写法摊成纯数字, 让它能参与比较",
            cost="本地、零成本", needs=("待归一的取值",), gives="数值",
            run=_t_expand_units),
    ExpTool(key="eval_expr", name="算术求值",
            desc="把 3*1e-4 / 2**10 这类表达式算成数（只走 ast 白名单, 不执行用户代码）",
            cost="本地、零成本", needs=("算术表达式",), gives="数值",
            run=_t_eval_expr),
    ExpTool(key="bounds_check", name="合法区间检查",
            desc="核对该参数取值是否落在合法区间内（与 ③ 站同一张表）",
            cost="本地、零成本", needs=("参数名", "取值"), gives="是否合法",
            run=_t_bounds_check),
    ExpTool(key="ratio_probe", name="倍数探测",
            desc="量出两侧相差几倍, 并看材料里有没有能解释这个倍数的量（并行度 / 累积步数）",
            cost="本地、零成本", needs=("两侧取值",), gives="倍数 + 是否可解释",
            run=_t_ratio_probe),
    ExpTool(key="relation_eval", name="关系式复算",
            desc="把两侧已知值代进已登记的公式复算一遍（warmup 换算 / lr 缩放 / 每轮步数）",
            cost="本地、零成本", needs=("两侧参数表",), gives="复算值与依据",
            run=_t_relation_eval),
    # ---- 阶段三 3A：四件具名科学计算工具（执行体在 scifuncs）----
    # 为什么并进这份清单而不是留一份平行菜单: ② 站是**唯一**该调它们的地方
    # （模型排的是工序, 算数的是这一站）。分成两套菜单, 模型就得"先选站、再选工具"，
    # 而实际只需要"在②站这一堆里挑要用的那几件"。
    # 下面四行只是**适配层** —— 真正的实现、边界与不造数纪律都在 `scifuncs` 里, 这里不复制。
    ExpTool(key="fp16_precision", name="浮点精度检查",
            desc="核取值在 fp16 / bf16 / fp32 下会不会下溢成 0 或溢出成 inf",
            cost="本地、零成本", needs=("取值", "精度档位"), gives="实际落值",
            run=_sci_runner("fp16_precision")),
    ExpTool(key="series_stats", name="序列统计",
            desc="对同一项的多个取值算均值 / 中位 / 样本标准差 / 离散度",
            cost="本地、零成本", needs=("同一项的多个取值",), gives="标准差 / 离散度",
            run=_sci_runner("series_stats")),
    ExpTool(key="linear_fit", name="直线最小二乘拟合",
            desc="对若干 (x, y) 点拟合直线, 报斜率 / 截距 / R²",
            cost="本地、零成本", needs=("横纵坐标序列（≥3 对）",), gives="斜率 / R²",
            run=_sci_runner("linear_fit")),
    ExpTool(key="peak_pick", name="峰位定位",
            desc="在一串点里定位最高 / 最低点, 并说明它是不是端点",
            cost="本地、零成本", needs=("横纵坐标序列（≥3 对）",), gives="峰位 / 峰值",
            run=_sci_runner("peak_pick")),
)

_TOOL_BY_KEY: dict = {t.key: t for t in EXPERIMENT_TOOLS}


def menu() -> list:
    """给模型看的计算工具菜单（JSON 友好形状, 字段名用中文更好读）。

    与 `toolbox.TOOLS` 的 `science_tools.menu()` 同款形状 —— 上层（规划器 / 3A 的
    科学工具登记）可以直接把它并进去, 不必为 ② 站再写一套描述。
    """
    return [{"key": t.key, "名称": t.name, "说明": t.desc, "成本": t.cost,
             "需要输入": list(t.needs), "产出": t.gives} for t in EXPERIMENT_TOOLS]


def catalog_keys() -> frozenset:
    """白名单 —— ② 站可调用的工具键全集（清单外 = 不存在）。"""
    return frozenset(t.key for t in EXPERIMENT_TOOLS)


# ---------------------------------------------------------------- 计划（挑工具）

def _unit_suffix() -> dict:
    """单位后缀表 —— 复用 derivation 的那一份（不复制, 否则两边迟早不一样）。"""
    from .derivation import _UNIT_SUFFIX
    return _UNIT_SUFFIX


def _norm_tool(raw: Any) -> Optional[str]:
    """这一侧的写法需要先归一吗? 需要的话用哪个工具。

    判据很窄: **只有**"读不出数、但能被某个工具读出数"才排归一 ——
    凡是 `to_number` 已经能给数的（`0.05`、`5%`、`1/2`、`1e-4`）一律不排,
    免得实验记录里全是"把 0.05 归一成 0.05"这种废话。

    ⚠ 单位后缀必须**前缀是纯数字**才算数。只看末位字符会闹笑话:
    `Adam` 结尾是 m、`LiT-B16B` 结尾是 B、`v2.0` 里也有数字 ——
    这些是模型名 / 版本号, 不是"数字+单位"。早先版本就是这么漏的,
    于是实验记录里出现了"把 Adam 归一成数"这种荒唐的步骤。
    """
    s = str(raw or "").strip()
    if not s or to_number(s) is not None:
        return None
    if len(s) >= 2 and s[-1] in _unit_suffix():
        try:
            float(s[:-1].strip())      # 前面必须是纯数字（10k / 1.5M / 3万）
            return "expand_units"
        except ValueError:
            pass
    if eval_arith(s) is not None:
        return "eval_expr"
    return None


def _has_bounds(key: str) -> bool:
    """这个参数在合法区间表里吗? —— 用一次探针问出来, 不直接去读私有表。"""
    note = str((numeric_check(key, "1") or {}).get("note") or "")
    return ("无预设合理区间" not in note) and ("非数值型" not in note)


# 关系式管哪些参数（与 derivation 的 _IDENTITY_TOKENS 同一份口径, 直接借过来）
def _relation_for(key: str) -> Optional[str]:
    from .derivation import _IDENTITY_TOKENS
    kk = str(key or "").lower()
    for rid, tokens in _IDENTITY_TOKENS.items():
        if any(t == kk or t in kk for t in tokens):
            return rid
    return None


def _relation_ready(rid: str, paper_vals: dict, code_vals: dict) -> tuple:
    """关系式这次**跑得动吗** —— 先探一次路, 缺输入就别排那一步。

    为什么要探: 排一个**明知道会失败**的步骤, 只会让实验记录里多一行"未算"。
    缺什么输入是有价值的信息, 但它应该写在**实验设计**里（人能看懂), 而不是
    伪装成一次失败的执行。这是同一件事的两种表达, 选好那种。

    （探路走的是关系式自己的 `gather`, 与真跑同一个入口 —— 不另写一份"能不能算"的判断。）
    """
    rel = next((r for r in RELATIONS if r.rid == rid), None)
    if rel is None:
        return False, f"没有名为 {rid} 的关系式"
    try:
        inputs, reason = rel.gather(paper_vals, code_vals)
    except Exception as ex:      # noqa: BLE001
        return False, f"关系式探路出错: {type(ex).__name__}"
    if inputs is None:
        return False, str(reason or "输入不齐")
    return True, ""


def _needs_fp16_check(*vals) -> bool:
    """这个取值**小到可能丢精度**吗 —— 判据只认"进得了半精度的危险区"。

    阈值直接取 fp16 的规格值, 不另定一个"看起来很小"的数:
      · < 2**-24 (5.96e-8): **落成 0**, 必然要报;
      · < 2**-14 (6.10e-5): 进次正规区, 有效数字掉到 3 位, 值得报一句;
      · 再大就**不报** —— fp32 下正常, 报它是噪声（2D 早就踩过"判据太宽变噪声"）。
    ⚠ 0 不算: 取值本来就是 0 时这不是"精度问题", 是"这项没设", 由别处报。
    """
    lo_norm = _PRECISION["fp16"]["min_normal"]
    for v in vals:
        n = to_number(v)
        if n is not None and n != 0 and abs(n) < lo_norm:
            return True
    return False


def _plan_for(card, paper_raw: dict, code_raw: dict,
              paper_vals: dict, code_vals: dict) -> tuple:
    """给一张卡挑一条工具路径（规则版）—— 这就是 ② 站的「自主调用」在规则档的样子。

    顺序 = **结论硬度**, 与 2B 的推演排序同一条原则（下面每条都踩过坑）:
        ① 归一        前提。`10k` 不先摊成数, 后面量出来的是"字符串比字符串"
        ② 关系式复算   最硬 —— 能直接给出"换算后一致 / 对不上"
        ③ 倍数探测    次硬 —— 能量出倍数、并看材料能不能解释
        ④ 区间检查    最软 —— 通过只说明"这个数不离谱", 所以**没别的可排时才排它**

    ⚠ 早先版本把区间检查排在关系式前面, 后果是: 一对 warmup 卡先被两次区间检查占满三步,
    关系式（唯一能判"一致 / 对不上"的一步）**根本排不上** —— 真实场景里就表现为
    "实验站跑了一堆, 却一条结论都给不出"。顺序错了, 功能就等于没有。

    两个算表（`paper_raw` / `paper_vals`）都要传进来, 各有用处:
        raw  —— 用户在材料里**看到的原样**（`10k` / `3*1e-4`）, 用于排"归一"这一步、以及记录输入;
        vals —— 展开后的表, 给需要算数的那几步用。
    早先只传展开后的表, 于是"归一"这一步永远排不出来（值在取数时就已经被摊平了）——
    实验记录里因此看不到原始写法, 用户没法照着核对。

    返回 `(steps, notes)`: notes 是**没排进路径但必须告诉人的事**
    （例如"关系式 warmup 这次缺 total_steps, 算不动"）—— 它进实验设计, 不进执行记录。
    """
    from .compute import _lookup

    steps: list = []
    notes: list = []
    rp_raw = _lookup(paper_raw, card.key)
    rc_raw = _lookup(code_raw, card.key)
    rp = _lookup(paper_vals, card.key)
    rc = _lookup(code_vals, card.key)

    # ① 归一：哪一侧写法读不出数, 就为它排一步（两侧都要就排两步）
    for side, raw in (("论文", rp_raw), ("代码", rc_raw)):
        tool = _norm_tool(raw)
        if tool:
            steps.append({
                "tool": tool, "inputs": {"raw": raw, "key": card.key},
                "why": (f"{side}侧写作 `{raw}`, 直接读不出数 —— 先归一成数才能继续比较"
                        f"（不替它猜一个值）"),
            })
    if len(steps) >= MAX_STEPS:
        return steps[:MAX_STEPS], notes

    # ② 关系式复算：这张卡涉及的参数有没有已登记的公式, 且这次**跑得动**
    rid = _relation_for(card.key)
    if rid:
        ready, why_not = _relation_ready(rid, paper_vals, code_vals)
        if ready:
            steps.append({
                "tool": "relation_eval",
                "inputs": {"rid": rid, "paper": paper_vals, "code": code_vals},
                "why": (f"两侧取值命中已登记的关系式 `{rid}` —— 代进去复算一遍。"
                        f"这是本张卡结论最硬的一步（能直接给出换算后一致 / 对不上）, 所以排在前面"),
            })
        else:
            notes.append(f"关系式 `{rid}` 这次算不动：{why_not}"
                         f"（不排这一步, 免得记录里多一行无效执行）")
    if len(steps) >= MAX_STEPS:
        return steps[:MAX_STEPS], notes

    # ②' 浮点精度（阶段三 3A）—— 只在**取值小到可能丢精度**时才排, 排它的理由很硬:
    #   一个数在训练用的精度里落成 0, 意味着"论文写了、代码里等于没设"。
    #   这条比倍数 / 区间都硬, 所以排在它们前面; 而判据窄（只看极小值）,
    #   所以绝大多数卡不会走到这里 —— 不排无意义的一步, 是 2D 就定下的原则。
    _sci_in, _sci_note = scifuncs.assemble_inputs(
        "fp16_precision", card=card, paper_vals=paper_vals, code_vals=code_vals)
    if _sci_in and _needs_fp16_check(rp, rc):
        steps.append({
            "tool": "fp16_precision", "inputs": _sci_in,
            "why": ("核这两侧的取值在训练常用的半精度下会不会**落成 0** —— "
                    "极小值（如 1e-8）在 fp16 里就是 0, 设了等于没设, 且不报错"),
        })
    if _sci_note:
        notes.append(f"浮点精度检查这次排不上：{_sci_note}")
    if len(steps) >= MAX_STEPS:
        return steps[:MAX_STEPS], notes

    # ③ 倍数探测：两侧都能读出数、且**确实差得开**才量（几乎相等的不用量）
    na, nb = to_number(rp), to_number(rc)
    if na and nb and na > 0 and nb > 0:
        ratio = max(nb / na, na / nb)
        if ratio >= 1.5:
            found = _find_factor(paper_vals, code_vals)
            inputs = {"key": card.key, "a": rp, "b": rc}
            if found:
                cn, f, side, raw_key = found
                inputs.update({"factor_name": cn, "factor_raw": raw_key, "factor_value": f})
            steps.append({
                "tool": "ratio_probe", "inputs": inputs,
                "why": ("量出两侧相差几倍, 并看材料里有没有能解释这个倍数的量"
                        "（并行度 / 梯度累积）—— 只有它能解释时才叫口径差, 否则只看倍数"),
            })
        else:
            notes.append(f"两侧取值几乎相等（相差 {ratio:g} 倍）—— 量倍数没有意义, 不排这一步")
    if len(steps) >= MAX_STEPS:
        return steps[:MAX_STEPS], notes

    # ④ 区间检查：没别的可排时才排 —— 它通过只说明"这个数不离谱", 不构成结论
    if _has_bounds(card.key):
        for side, raw in (("论文", rp_raw), ("代码", rc_raw)):
            if raw not in (None, ""):
                steps.append({
                    "tool": "bounds_check",
                    "inputs": {"key": card.key, "value": raw},
                    "why": f"核 {side}侧取值是否落在该参数的合法区间内（与 ③ 站同一张表）",
                })
    return steps[:MAX_STEPS], notes


def _llm_choose(card, llm: Optional[Callable]) -> Optional[list]:
    """模型版挑工具（**可选通道, 默认不走**）。

    为什么留这条通道: 赛题要求"计算工具自主调用"。本项目的既定口径是
    「模型全权规划、程序只递菜单 + 留痕」（1C）。这里同款:
    把**工具菜单**与这张卡的两侧取值交给模型, 让它给出工具键序列;
    模型不可用 / 输出不合法 / 提到了清单外的工具 → 返回 None, 上层回落规则版并**如实标**
    `planner="rule"`, 绝不假装模型参与过。

    ⚠ 模型只能挑**工具键**, 不能自己填输入 —— 输入一律由规则从材料里取。
       否则模型一旦编一个不存在的值, "不造数"这条纪律当场破。
    """
    if llm is None:
        return None
    try:
        import json
        prompt = (
            "你在给一次『论文↔代码一致性审计』的**实验站**挑计算工具。\n"
            "待验的这一条:\n"
            f"  参数: {card.key}\n  待验陈述: {card.claim}\n"
            f"  论文侧取值: {card.paper_value or '（无）'}\n"
            f"  代码侧取值: {card.code_value or '（无）'}\n\n"
            "下面是你能调用的**全部计算工具**; 清单之外的工具本次不存在, 不要臆造。\n"
            f"{json.dumps(menu(), ensure_ascii=False)}\n\n"
            "后四件是**具名科学计算工具**（阶段三 3A）: 前五件解决「同一个数两种写法」, "
            "后四件解决「这串数本身说明什么」。挑够用且必要的即可, 别为了凑数全挑。\n\n"
            "请挑出**必要且够用**的工具, 按执行顺序给出（最多 3 个）。"
            "若一条都用不上, 给空数组。\n"
            '只输出 JSON: {"tools":["工具key", ...], "reason":"一句话理由"}'
        )
        got = llm(prompt)
        keys = list((got or {}).get("tools") or [])
        seq: list = []
        for k in keys[:MAX_STEPS]:
            if k not in _TOOL_BY_KEY:
                return None          # 提到了清单外的工具 → 整条不采纳, 回落规则版
            seq.append(k)
        return seq or None
    except Exception:      # noqa: BLE001 —— 模型这一路出任何问题都不该拖垮主流程
        return None


# ---------------------------------------------------------------- 真跑

def _run_steps(steps: list) -> tuple:
    """逐步真跑, 返回 (执行记录, 汇总)。

    汇总里收集 expected / actual —— 只取**最后一个**给出 compare 的成功步骤:
    一张卡可能跑三步, 但最终的"实际值"只能有一个来源, 否则报告里就成了拼凑。
    """
    records: list = []
    agg = {"expected": None, "actual": None, "expected_from": "", "actual_from": "",
           "compares": [], "values": []}
    for i, s in enumerate(steps or [], start=1):
        tool = _TOOL_BY_KEY.get(str(s.get("tool")))
        if tool is None:
            continue
        t0 = time.perf_counter()
        try:
            out = dict(tool.run(dict(s.get("inputs") or {})))
        except Exception as ex:      # noqa: BLE001 —— 单个工具炸了只该毁掉这一步, 不该毁掉整条
            out = {"ok": False, "text": f"工具执行出错: {type(ex).__name__}: {ex}"}
        ms = (time.perf_counter() - t0) * 1000.0
        out.setdefault("ok", False)
        records.append({
            "seq": i, "tool": tool.key, "name": tool.name,
            "why": s.get("why", ""),
            # 只记能进 JSON 的输入形状（两侧参数表只记键名, 免得整张表灌进留痕里）
            "inputs": {k: (v if isinstance(v, (str, int, float, bool)) or v is None
                           else f"<{type(v).__name__} 共 {len(v)} 项>")
                       for k, v in dict(s.get("inputs") or {}).items()},
            "ok": bool(out.get("ok")), "output": str(out.get("text") or ""),
            "ms": round(ms, 4),
        })
        if not out.get("ok"):
            continue
        v = out.get("value")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            agg["values"].append(v)
        exp_from = str(out.get("expected_from") or "")
        act_from = str(out.get("actual_from") or "")
        # ---- 非自证守卫: 预期与实际一旦同源, 这个"验证"什么都验不出来 —— 直接作废。
        #      （`compute` 模块头把这条写死在关系式定义里; 这里是运行时兜底, 防新工具踩坑。）
        if exp_from and exp_from == act_from:
            records[-1]["output"] += "（⚠ 预期与实际同源, 本条不作验证依据）"
            continue
        # 两条收集规则, 分开管两件事（合在一起会互相擦掉 —— 实测踩过）:
        #   ① **结论性**的一步（有 compare）→ 覆盖预期 / 实际。它是这一步的最终答案。
        #   ② **只测到、没下判断**的一步（有 actual 没 compare）→ 只在**还没有实际值**时填。
        #      否则会出现这种自相矛盾: 第 1 步复算出"0.05 一致", 第 2 步量出"差 10000 倍",
        #      后者把 expected 擦成 None —— 卡片说"一致", 预期栏却是空的。
        #      测到了就该显示（早先只在有 compare 时记, 结果是量出 10 倍的卡实际值写着「—」）。
        if out.get("compare") is not None:
            agg["compares"].append(bool(out["compare"]))
            agg["expected"] = out.get("expected")
            agg["actual"] = out.get("actual")
            agg["expected_from"] = exp_from
            agg["actual_from"] = act_from
        elif out.get("actual") is not None and agg["actual"] is None:
            agg["actual"] = out.get("actual")
            agg["actual_from"] = act_from
    return records, agg


def _why_no_tool(card, paper_vals: dict, code_vals: dict) -> str:
    """挑不出工具时说清**为什么** —— "算不出来"这三个字必须带原因, 否则等于什么都没说。"""
    from .compute import _lookup

    rp = _lookup(paper_vals, card.key)
    rc = _lookup(code_vals, card.key)
    if rp in (None, "") and rc in (None, ""):
        return "两侧都没抽到这个参数的确定取值（不猜、不拿默认值补）"
    if to_number(rp) is None and to_number(rc) is None:
        p, c = (rp or "—"), (rc or "—")
        return (f"有效值不是数值型（论文 {p} / 代码 {c}）—— "
                f"② 站的计算工具对文字型取值无从下手, 这类分歧要靠读原文或实机")
    return "取值凑不齐一条可跑的算式（缺另一侧配合的参数, 或没有对应的已知公式）"


def _decide_static(card) -> Optional[tuple]:
    """① 站已经定性的卡, ② 站不重复验（好的实验设计首先是"哪些不用做"）。"""
    verdict = str(getattr(card, "verdict", "") or "")
    if verdict == DERIVE_EXCLUDED:
        return EXP_NO_NEED, ("① 站推演已证明这不是真分歧（已排除）—— 本步不重复验, "
                             "避免把力气花在已经说清的事上")
    if verdict == DERIVE_FLAGGED:
        return EXP_NO_NEED, ("① 站推演已给出比「两边不一致」更硬的结论"
                             "（已定位: 一侧取值本身有问题）—— "
                             "本步不重复验, 该去核那一侧的原文")
    return None


# 实机方案模板 —— 按"这条假设属于哪一类"给, 每一句都必须是**可以照着做**的。
_RERUN_BY_KIND: dict = {
    "取值冲突": ("固定随机种子与数据子集, 用两侧各自的取值各跑一次短跑（建议 200 步以内）, "
                 "比对这条参数**直接影响的量**: 学习率看前 200 步 loss 曲线, "
                 "批大小看每步耗时与显存, 丢弃率看训练/验证 gap。"
                 "两条曲线在噪声范围内重合 → 差异不影响结论; 出现系统性分叉 → 差异确实生效。"),
    "运行时才定": ("先在训练启动处打印这个变量的实际取值（一行日志即可）—— "
                   "大多数情况下不必真的跑完训练, 看到实际值就能定。"
                   "若打印出来的值与论文声明一致, 这条假设当场推翻。"),
    "写法不同": ("把两侧的写法各自喂给同一段归一逻辑, 看归一到同一个值后是否一致; "
                 "一致的说明只是写法差异, 不一致才需要实机验证。"),
    "代码独有": ("确认这个参数在代码里**有没有消费者**（谁读它）: 没人读 = 设了也不生效; "
                 "有人读 = 关掉它跑一次短跑, 看指标是否变化, 变化说明它确实起作用。"),
    "论文独有": ("先在代码里按同义名搜一遍（含缩写与嵌套配置）, 确认是不是改了名字; "
                 "确实没有实现 → 在代码里补一个默认实现再跑, 比对补前补后的指标差异。"),
}


def _rerun_plan(card) -> str:
    """给"必须实机"的卡配一份可执行的方案 —— 这本身就是「实验设计」的产物。"""
    base = _RERUN_BY_KIND.get(str(getattr(card, "kind", "") or ""),
                              "按该参数最直接影响的指标做一次短跑对照（同种子、同数据子集）。")
    falsify = str(getattr(card, "falsifiable_by", "") or "").strip()
    if falsify:
        base += f" 判定标准（来自 ① 站）: {falsify}"
    return base


# ---------------------------------------------------------------- 主入口

@dataclass
class ExperimentCard:
    """一张实验卡 —— ② 站的产物（界面 / 报告 / 留痕都读这里, 别处不要再造一份）。"""
    key: str
    rank: int = 0                          # _rerank 回填（构造时为 0）
    claim: str = ""
    status: str = ""
    status_desc: str = ""
    design: str = ""                       # 实验设计: 要算什么、为什么
    tool_path: tuple = ()                  # 调了哪些具名工具（顺序）
    records: list = field(default_factory=list)   # 逐步执行记录
    expected: Any = None
    expected_from: str = ""
    actual: Any = None
    actual_from: str = ""
    plan_reason: str = ""                  # 谁挑的工具路径（模型 / 规则兜底）+ 挑法
    rerun_plan: str = ""                   # 需实机时的可执行方案
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "rank": self.rank, "claim": self.claim,
            "status": self.status, "status_desc": self.status_desc,
            "design": self.design, "tool_path": list(self.tool_path),
            "records": self.records,
            "expected": self.expected, "expected_from": self.expected_from,
            "actual": self.actual, "actual_from": self.actual_from,
            "plan_reason": self.plan_reason, "rerun_plan": self.rerun_plan,
            "seconds": self.seconds,
        }


def design_experiments(cards, paper=None, code=None, *,
                       llm: Optional[Callable] = None) -> list:
    """对假设卡做实验设计与真跑, 返回实验卡列表（已排序）。

    输入: `build_hypotheses()` + `derive()` 之后的卡片, 以及两侧原始材料（只为取数）。
    输出: [ExperimentCard], 每个都带**实际跑过的**执行记录（或"为什么没跑"的如实说明）。

    llm: 可选的模型回调（`llm(prompt) -> dict`）。给了就走模型挑工具; 不可用 /
          输出不合法 → 回落规则版, 并在 `plan_reason` 里如实写明。**默认 None = 纯规则。**
    """
    # 两张表都要: 原样（用户在材料里看到的写法, 用于排"归一"与记录输入）
    # ＋ 展开后的（单位摊平, 给需要算数的那几步用）。
    # 只留展开后的表会出问题: 实验记录里看不到用户原来的写法, 他没法照着核对。
    paper_raw = _side_values(getattr(paper, "params", None))
    code_raw = _side_values(getattr(code, "params", None))
    paper_vals = _expand_units(paper_raw)
    code_vals = _expand_units(code_raw)

    out: list = []
    for card in cards or []:
        t0 = time.perf_counter()
        static = _decide_static(card)
        if static is not None:
            status, why = static
            out.append(ExperimentCard(
                key=card.key, claim=card.claim, status=status,
                status_desc=STATUS_DESC.get(status, ""),
                design=f"不做实验：{why}",
                plan_reason="规则判定（① 站已定性）", seconds=0.0))
            continue

        steps, notes = _plan_for(card, paper_raw, code_raw, paper_vals, code_vals)
        planner = "规则版（按卡片类型挑工具路径）"
        chosen = _llm_choose(card, llm)
        if chosen:
            # 模型只给**工具键序列**, 输入仍由规则从材料里取（不造数）。
            by_key = {s["tool"]: s for s in steps}
            steps = [by_key[k] for k in chosen if k in by_key][:MAX_STEPS]
            planner = "模型挑的工具路径（输入仍由规则从材料取, 模型不造数）"

        if not steps:
            # 挑不出任何工具 —— 要么这条本来就只能实机定, 要么输入凑不齐
            machine = bool(getattr(card, "needs_rerun", False))
            status = EXP_MACHINE if machine else EXP_UNDECIDED
            why_txt = ("① 站已标「必须实机才能定论」—— 本步不假装算过, 改给一份可执行的实机方案"
                       if machine else _why_no_tool(card, paper_vals, code_vals))
            design = f"本步挑不出可跑的工具：{why_txt}"
            if notes:
                design += "　｜　" + "；".join(notes)
            out.append(ExperimentCard(
                key=card.key, claim=card.claim, status=status,
                status_desc=STATUS_DESC.get(status, ""), design=design,
                plan_reason=planner, rerun_plan=_rerun_plan(card) if status == EXP_MACHINE else "",
                seconds=round(time.perf_counter() - t0, 4)))
            continue

        records, agg = _run_steps(steps)
        ok_records = [r for r in records if r["ok"]]
        if agg["compares"]:
            status = EXP_AGREE if all(agg["compares"]) else EXP_DISAGREE
        elif ok_records:
            # 跑成了, 但这一步本身不构成"一致 / 不一致"（例如量出了倍数却解释不掉）
            # —— 如实记「已实测量·不下结论」, 不硬塞进"一致"里充成绩。
            status = EXP_MEASURED
        else:
            status = EXP_UNDECIDED

        parts: list = []
        for s in steps:
            tk = _TOOL_BY_KEY.get(s["tool"])
            if tk:
                why = s.get("why") or ""
                parts.append(f"{tk.name}：{why}")
        design = "　→　".join(parts)
        if notes:
            # 没排进路径、但必须让人知道的事（例如"关系式缺 total_steps"）
            design += "　｜　" + "；".join(notes)
        out.append(ExperimentCard(
            key=card.key, rank=0, claim=card.claim,
            status=status, status_desc=STATUS_DESC.get(status, ""),
            design=design, tool_path=tuple(s["tool"] for s in steps),
            records=records,
            expected=agg["expected"], expected_from=agg["expected_from"],
            actual=agg["actual"], actual_from=agg["actual_from"],
            plan_reason=f"{planner}；本次 {len(records)} 步, 成了 {len(ok_records)} 步",
            rerun_plan=_rerun_plan(card) if status == EXP_MACHINE else "",
            seconds=round(time.perf_counter() - t0, 4)))

    _rerank(out)
    return out


def _rerank(cards: list) -> None:
    """按"值不值得先看"排序, 并回填 rank（同分必须有稳定次序, 否则消融对照没法比）。"""
    cards.sort(key=lambda c: (_STATUS_ORDER.get(c.status, 9), c.key))
    for i, c in enumerate(cards, start=1):
        c.rank = i


def summarize(cards: list) -> dict:
    """一句话说清这批实验分成了几堆（界面 / 报告共用, 不各写一套）。"""
    cards = list(cards or [])
    by = {s: 0 for s in (EXP_DISAGREE, EXP_MACHINE, EXP_MEASURED,
                         EXP_AGREE, EXP_UNDECIDED, EXP_NO_NEED)}
    calls = 0
    tools: list = []
    seconds = 0.0
    for c in cards:
        by[c.status] = by.get(c.status, 0) + 1
        for r in c.records or []:
            calls += 1
            if r["tool"] not in tools:
                tools.append(r["tool"])
        seconds += float(c.seconds or 0.0)
    return {
        "total": len(cards),
        "disagree": by[EXP_DISAGREE], "agree": by[EXP_AGREE],
        "machine": by[EXP_MACHINE], "measured": by[EXP_MEASURED],
        "undecided": by[EXP_UNDECIDED], "no_need": by[EXP_NO_NEED],
        "calls": calls, "tools": tools, "seconds": round(seconds, 4),
        "engine": engine_name(),
    }


def engine_name() -> str:
    """本次实验用的引擎名 —— 与 2B 的推演同源（内置求值, 不引外部符号库）。

    为什么不引 sympy: 2D 要的是"把已知的数代进已知公式", 项目已有等价基座
    （`ast` 白名单求值 + 关系式注册表）。再引一套只会多一条装不上的路径,
    还会让同一条公式出现两种算法 —— 本项目最忌讳这个（详见 `derivation.engine_name`）。
    """
    return "内置计算工具（ast 白名单求值 + 关系式注册表）"


def _side_values(params):
    """把一侧参数表拍平成 `{key: 值字符串}` —— 复用 compute 的同名口径, 不复制一份。"""
    from .compute import _side_values as _impl
    return _impl(params)
