"""「算」科学计算工具（阶段五 · M2 闭环扩展）。

**为什么需要这一步**：审计只回答"两个值一样不一样"，但科研里不少"不一致"其实是
**同一件事的两种写法**——论文写 `warmup = 1000 步`，代码写 `warmup_ratio = 0.05`，
总步数 20000 时两者完全等价。只比字面值就会把这种"写法差异"报成冲突
（阶段四双路核对抓到的 `scheduler` 属于语义写法差异，这一类属于**数值口径差异**）。

**所以「算」干三件事**（全部纯规则、零成本、可复现、离线可跑）：

1. **关系式核验** —— 已登记的关系式，输入齐备就复算一遍，拿复算值去核对另一侧的声明值。
   典型：warmup 的「步数 ↔ 比例」换算、由 epochs + iterations 反推每轮步数、
   学习率与批大小的线性缩放推演。
2. **量纲/数量级提示** —— 两侧都是数但恰好差 10 的整数次幂时，提示"疑似单位或量纲写错"
   （`1e-4` vs `1e-1` 这种，比笼统的"不一致"更有诊断价值）。
3. **数值合理性复核** —— 复用对撞口 `collider.numeric_check` 的区间表，不另立一套。

**三条边界（答辩会被追问，所以写在最前面）**：
1. **只做小规模**：大模型训练跑不动（要 GPU / 时间 / 钱）。本模块**不执行任何训练、
   不运行用户代码**，只做"把已知的数代进已知公式"这种小规模复算 + 静态推演。
   跑不动的，在结论里明确标注「需实机验证」，不假装算过。
2. **算不出来就说算不出来**：输入不全一律记 `skipped` 并写明缺哪个输入，
   **绝不用默认值补出一个数**——与第一期"宁可不报一个值"同源。
3. **核验与推演分开标注**：确凿输入比出来的叫 `consistent` / `mismatch`；
   带假设的推测（如"可能是有意按批大小缩放"）一律 `hint` + `confidence=inferred`，
   由下游决定是否采信，绝不混进确定性结论。

**一条实现上的自我约束**（写下来防止再犯）：
每一条关系式的 `gather` 必须从**两侧各自**取值，判定的 expected 与 actual 必须来自
**不同的**输入。若两者由同一个数推出来，就成了自证循环 —— 恒等成立、什么都验不出来。
"""
from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# 复用对撞口的数值归一化与合理性区间表 —— 不另写一份，否则两处口径会漂移
from .collider import _to_number, numeric_check
from .synonyms import canonical_key

__all__ = [
    "eval_arith", "to_number", "magnitude_hint", "check_relations",
    "review_numbers", "ComputeReport", "RELATIONS", "Relation",
]


# ---------------------------------------------------------------- 安全求值

_BIN_OPS: dict = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTS: dict = {"pi": math.pi, "e": math.e}


def eval_arith(expr: Any) -> Optional[float]:
    """把一个算术表达式算出来（`3*1e-4`、`2**10`、`512*512`）。

    为什么不用 `eval`：代码包是**用户上传的**，`eval` 等于把执行权交出去。
    这里走 `ast` 白名单 —— 只放行数字常量、四则运算、乘方、取模、整除与两个数学常量，
    其它一律返回 None（不抛异常，调用方按"算不出来"处理）。
    """
    if expr is None:
        return None
    s = str(expr).strip()
    if not s or len(s) > 120:
        return None
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError:
        return None
    try:
        return _ev(tree.body)
    except Exception:      # noqa: BLE001  —— 越界/除零/非法节点都按"算不出来"处理
        return None


def _ev(node: Any) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("bool 不算数")
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("非常量")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return float(_BIN_OPS[type(node.op)](_ev(node.left), _ev(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_ev(node.operand)))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return float(_CONSTS[node.id])
    raise ValueError(f"不允许的节点: {type(node).__name__}")


def to_number(v: Any) -> Optional[float]:
    """把任意侧的值转成数：先常规归一（含百分比/分数），不行再试算术求值。

    与 `collider._to_number` 的关系：那个只认字面量，这个多兜一层"表达式"
    （`3*1e-4` 这种在代码里很常见）。两个都失败才算"这不是个数"。
    """
    n = _to_number(v)
    if n is not None:
        return n
    return eval_arith(v)


def magnitude_hint(paper_val: Any, code_val: Any) -> Optional[str]:
    """两侧都是数、且比值恰为 10 的整数次幂时，给出量纲提示。

    为什么单拎出来：`1e-4` 与 `1e-1` 在报告里都只是"不一致"，但"差 1000 倍"
    这件事本身最有诊断价值（通常是 `%` 与小数混用，或漏写量级）。

    **阈值取 10^2 起**：学习率差 10 倍是极常见的合法差异（不同 batch / 不同模型），
    拿它当"量纲疑似写错"会变成噪声；差 100 倍以上才是真线索。
    比值不是 10 的整数次幂（2 倍、1.5 倍）也不提示 —— 那属于正常差异。
    """
    p, c = to_number(paper_val), to_number(code_val)
    if p is None or c is None or p == 0 or c == 0:
        return None
    ratio = p / c
    if ratio <= 0:
        return None
    k = math.log10(ratio)
    kr = round(k)
    if abs(kr) < 2 or abs(k - kr) > 1e-9 or abs(kr) > 12:
        return None
    return (f"两侧相差 10^{kr} 倍（{paper_val} vs {code_val}）—— "
            f"疑似单位/量纲写错，而不只是取值不同")


# ---------------------------------------------------------------- 侧值收集

def _side_values(params: Any) -> dict:
    """把一侧的参数表拍平成 `{key: 值字符串}`。

    兼容三种形态（项目里三种都存在，不能只认一种）:
      · `{key: Evidence}`             —— PaperInfo.params
      · `{key: [Evidence, ...]}`      —— CodeInfo.params（多处赋值，取第一处）
      · `{key: "1e-4"}` / `{key: 数值}` —— 对撞区视图与测试里的简写
    """
    out: dict = {}
    if not params:
        return out
    for k, v in dict(params).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = v[0] if v else None
            if v is None:
                continue
        if isinstance(v, dict):
            v = v.get("value") or v.get("raw_value")
            if v is None:
                continue
        if hasattr(v, "raw_value"):      # Evidence
            v = v.raw_value
        s = str(v).strip()
        if s:
            out[str(k)] = s
    return out


def _lookup(vals: dict, key: str) -> Optional[str]:
    """在一个侧值表里找某个键（先精确、再按同义词归一）。"""
    if key in vals:
        return vals[key]
    for k, v in vals.items():
        if canonical_key(k) == key:
            return v
    return None


def _num(vals: dict, key: str) -> Optional[float]:
    v = _lookup(vals, key)
    return to_number(v) if v is not None else None


# ---------------------------------------------------------------- 关系式

@dataclass
class Relation:
    """一条可复算的参数关系式 —— **执行与判定分两段**（阶段二 2C）。

    gather(pv, cv) -> (inputs, reason)
      pv / cv 是两侧的侧值表；inputs 是算好的输入字典，reason 是"差什么"。
      返回 inputs=None 表示这次不适用，reason 会被如实写进结果（不静默跳过）。
    run(inputs) -> dict
      **② 实验站的活**：把两侧的数代进公式，**只产出 expected / actual 两个数**，
      不碰 verdict（不给结论、不贴标签）。跑不出来就抛异常或回 {"undecidable": 理由}。
    judge(measured) -> dict
      **③ 验证站的活**：拿 ② 已经跑出来的数**下判断**（一致 / 对不上 / 推测 / 仅推演），
      **不重新算数** —— 判定函数只能看 measured 里已有的数，缺哪个就如实说缺。

    分两段是为了让「③不造数」这条红线在**代码结构上**成立：
    以前 judge 一次返回 actual＋verdict，判定和算数混在一个函数里，
    说"判定不造数"只能靠注释；现在 run 是唯一的算数口，judge 无权产出数。
    """
    rid: str
    title: str
    formula: str
    gather: Callable[[dict, dict], tuple]
    judge: Callable[[dict], dict]
    run: Optional[Callable[[dict], dict]] = None   # ② 算数；缺省时回落成 judge 兼办（旧关系式兼容）

    def measure(self, inputs: dict) -> dict:
        """统一入口：返回**只有数、没有结论**的测量结果。"""
        if callable(self.run):
            return self.run(inputs)
        # 旧关系式没拆：judge 兼算数兼判定，这里跑一次再把结论剔掉。
        res = dict(self.judge(inputs) or {})
        for k in ("verdict", "detail", "confidence"):
            res.pop(k, None)
        return res

    def decide(self, measured: dict) -> dict:
        """统一入口：只下判断。拆过的走 judge；没拆的走兼容路径（judge 再跑一次取结论）。"""
        return self.judge(measured)


def _g_warmup(pv: dict, cv: dict) -> tuple:
    """warmup 的「步数 ↔ 比例」换算核验。

    判定哪一侧是比例、哪一侧是步数，用的不是"谁是论文谁是代码"，
    而是**量级**：≤1 视为比例（warmup_ratio），>1 视为步数（warmup_steps）。
    warmup 步数为 1 的真实论文不存在，这个分界足够安全，且不依赖任何外部约定。
    """
    total = _num(pv, "iterations")
    if total is None:
        total = _num(cv, "iterations")
    if not total or total <= 0:
        return None, "缺少 total_steps（iterations）无法换算"
    wp, wc = _num(pv, "warmup"), _num(cv, "warmup")
    if wp is None or wc is None:
        return None, "只有一侧声明了 warmup"
    same_scale = (wp <= 1.0) == (wc <= 1.0)
    if same_scale:
        return {"mode": "same_scale", "paper": wp, "code": wc,
                "total_steps": total}, ""
    # 一侧步数、一侧比例 —— 这正是需要换算的场景
    ratio = wp if wp <= 1.0 else wc
    steps = wc if wp <= 1.0 else wp
    return {"mode": "convert", "ratio": ratio, "steps": steps,
            "total_steps": total, "ratio_side": "paper" if wp <= 1.0 else "code"}, ""


def _r_warmup(inp: dict) -> dict:
    """② 站：warmup 的**换算** —— 只算数，不下结论。

    两种情形都把 expected / actual 摆齐：
      · convert   → expected = 步数 ÷ 总步数（公式算出来的），actual = 另一侧声明的比例
      · same_scale → 两侧同为比例或同为步数，expected=None（无从比较，如实留空）
    判"一致 / 对不上"是 ③ 站的活，这里不碰 verdict。
    """
    if inp.get("mode") == "same_scale":
        return {"expected": None, "actual": None,
                "both_same_scale": True,
                "note_shape": ("比例" if inp["paper"] <= 1.0 else "步数"),
                "paper": inp["paper"], "code": inp["code"],
                "total_steps": inp.get("total_steps")}
    return {"expected": inp["steps"] / inp["total_steps"],   # 由「步数侧」推出来的比例
            "actual": inp["ratio"],                            # 「比例侧」自己声明的比例
            "both_same_scale": False,
            "steps": inp["steps"], "total_steps": inp["total_steps"],
            "ratio_side": inp.get("ratio_side", "")}


def _j_warmup(measured: dict) -> dict:
    """③ 站：拿 ② 跑出来的数**下判断** —— 本函数不读两侧参数表，也不再做除法。"""
    base = {"rid": "warmup_unit", "confidence": "confirmed",
            "expected": measured.get("expected"), "actual": measured.get("actual")}
    if measured.get("both_same_scale"):
        base.update(verdict="info",
                    detail=(f"两侧 warmup 同为{measured.get('note_shape', '比例')}形态"
                            f"（{measured.get('paper', '—')} / {measured.get('code', '—')}），"
                            f"无需换算 —— 直接按字面比对即可"))
        return base
    exp, act = base["expected"], base["actual"]
    steps, total = measured.get("steps"), measured.get("total_steps")
    if exp is None or act is None:
        base.update(verdict="info", detail="② 站没有给出可比的数（缺 expected 或 actual），本次不下结论")
        return base
    if abs(exp - act) <= max(1e-9, 1e-6 * max(abs(exp), abs(act), 1.0)):
        base.update(verdict="consistent",
                    detail=(f"{steps:g} 步 ÷ {total:g} 总步数 = {exp:.4g}，"
                            f"与另一侧声明的比例 {act:.4g} 一致 —— "
                            "两者是同一件事的两种写法"))
    else:
        base.update(verdict="mismatch",
                    detail=(f"按总步数 {total:g} 换算应为 {exp:.4g}，"
                            f"实际声明 {act:.4g} —— 换算后仍然对不上"))
    return base


def _g_steps_per_epoch(pv: dict, cv: dict) -> tuple:
    """由「轮数 + 总步数」反推每轮步数 —— 纯推演，没有可比对的声明值。"""
    epochs = _num(pv, "epochs") if _num(pv, "epochs") is not None else _num(cv, "epochs")
    total = _num(pv, "iterations") if _num(pv, "iterations") is not None else _num(cv, "iterations")
    missing = [n for n, v in (("epochs", epochs), ("iterations", total)) if not v]
    if missing:
        return None, f"缺少 {'、'.join(missing)}"
    return {"epochs": epochs, "total_steps": total}, ""


def _r_steps_per_epoch(inp: dict) -> dict:
    """② 站：由「轮数 + 总步数」算出每轮步数 —— 纯除法，没有预期值可比。"""
    return {"expected": None, "actual": inp["total_steps"] / inp["epochs"],
            "epochs": inp["epochs"], "total_steps": inp["total_steps"]}


def _j_steps_per_epoch(measured: dict) -> dict:
    """③ 站：给推演值贴标签 —— 只有 actual、没有 expected，因此**只能标「推演」不作为结论**。"""
    v = measured.get("actual")
    if v is None:
        return {"rid": "steps_per_epoch", "expected": None, "actual": None,
                "verdict": "info", "confidence": "none",
                "detail": "② 站没有算出每轮步数（输入缺轮数或总步数），本次不下结论"}
    return {"rid": "steps_per_epoch", "expected": None, "actual": v,
            "verdict": "info", "confidence": "inferred",
            "detail": (f"{measured['total_steps']:g} 总步数 ÷ {measured['epochs']:g} 轮 ≈ {v:g} 步/轮"
                       "（推演值，供人工核对，不作为结论）")}


def _g_lr_scaling(pv: dict, cv: dict) -> tuple:
    """学习率与批大小的线性缩放推演（两侧都要有 lr 与 batch 才能算）。"""
    vals = {"lr_paper": _num(pv, "learning_rate"), "lr_code": _num(cv, "learning_rate"),
            "bs_paper": _num(pv, "batch_size"), "bs_code": _num(cv, "batch_size")}
    missing = [k for k, v in vals.items() if not v]
    if missing:
        return None, f"缺少 {'、'.join(missing)}（两侧都要有 lr 与 batch 才能做缩放推演）"
    if vals["lr_paper"] == vals["lr_code"] or vals["bs_paper"] == vals["bs_code"]:
        return None, "两侧 lr 或 batch 相同，无缩放可推演"
    return vals, ""


def _r_lr_scaling(inp: dict) -> dict:
    """② 站：量出两个倍率 —— 批大小几倍、学习率几倍。**不判断它们是否吻合。**"""
    return {"expected": inp["bs_code"] / inp["bs_paper"],     # 批大小倍率
            "actual": inp["lr_code"] / inp["lr_paper"],        # 学习率倍率
            "lr_paper": inp["lr_paper"], "lr_code": inp["lr_code"],
            "bs_paper": inp["bs_paper"], "bs_code": inp["bs_code"]}


def _j_lr_scaling(measured: dict) -> dict:
    """③ 站：只比两个倍率像不像 —— 不重算倍率。"""
    exp, act = measured.get("expected"), measured.get("actual")
    base = {"rid": "lr_scaling", "confidence": "inferred",
            "expected": exp, "actual": act}
    if exp is None or act is None:
        base.update(verdict="info", detail="② 站没有量出两个可比的倍率，本次不下结论")
        return base
    if abs(act - exp) <= max(1e-9, 0.02 * max(abs(act), abs(exp), 1.0)):
        base.update(verdict="hint",
                    detail=(f"批大小放大 {exp:.4g} 倍、学习率放大 {act:.4g} 倍，两者吻合 —— "
                            "疑似**刻意按批大小线性缩放**，不是写错（需人工确认，"
                            "不作为确定性结论）"))
    else:
        base.update(verdict="info",
                    detail=(f"批大小放大 {exp:.4g} 倍、学习率放大 {act:.4g} 倍，"
                            "两者不吻合 —— 线性缩放解释不成立"))
    return base


# 登记表: 加一条关系式只要在这里加一项（并补测试），下游代码一行不改。
# run = ② 站算数；judge = ③ 站判定（2C 起两段分开，run 可缺省但建议补上）。
RELATIONS: tuple = (
    Relation(rid="warmup_unit", title="warmup 步数 ↔ 比例 换算核验",
             formula="warmup_steps / total_steps ?= warmup_ratio",
             gather=_g_warmup, run=_r_warmup, judge=_j_warmup),
    Relation(rid="steps_per_epoch", title="每轮步数推演",
             formula="steps_per_epoch = total_steps / epochs",
             gather=_g_steps_per_epoch, run=_r_steps_per_epoch, judge=_j_steps_per_epoch),
    Relation(rid="lr_scaling", title="学习率 × 批大小 线性缩放推演",
             formula="lr_code / lr_paper ?= batch_code / batch_paper",
             gather=_g_lr_scaling, run=_r_lr_scaling, judge=_j_lr_scaling),
)

_VERDICT_LABEL = {
    "consistent": "✅ 换算后一致",
    "mismatch": "⚠️ 复算对不上",
    "hint": "💡 推测（需人工确认）",
    "info": "ℹ️ 推演（不作为结论）",
    "skipped": "— 未算",
}


# ---------------------------------------------------------------- 核验入口

@dataclass
class ComputeReport:
    """一次「算」的完整产物（供报告/界面消费）。"""
    relations: list = field(default_factory=list)     # 关系式核验/推演结果
    magnitudes: list = field(default_factory=list)    # 量纲提示
    plausibility: list = field(default_factory=list)  # 数值合理性复核
    notes: list = field(default_factory=list)
    checked: int = 0        # 实际算出来的条数（未算的不计）
    flagged: int = 0        # 算出来且提示异常的条数
    skipped: int = 0        # 因缺输入没算的条数

    def to_dict(self) -> dict:
        return {
            "relations": self.relations, "magnitudes": self.magnitudes,
            "plausibility": self.plausibility, "notes": self.notes,
            "checked": self.checked, "flagged": self.flagged,
            "skipped": self.skipped,
        }

    def to_markdown(self) -> str:
        md = ["## 「算」小规模复算与静态推演", "",
              "> 只做「把已知的数代进已知公式」的小规模复算与静态推演；",
              "> **不执行任何训练、不运行用户代码**。输入不全的一律如实记为「未算」，",
              "> 不用默认值补出一个数。", "",
              "> **执行与判定分开（2C）**：② 实验站只负责把数算出来，",
              "> ③ 验证站只负责下结论 —— 判定环节拿不到两侧原始参数，**没有能力造数**。", ""]
        if self.relations:
            md += ["| 关系式 | ② 算出的数 | 结果 | 说明 |",
                   "| --- | --- | --- | --- |"]
            for r in self.relations:
                lab = _VERDICT_LABEL.get(r.get("verdict", ""), r.get("verdict", ""))
                exp, act = r.get("expected"), r.get("actual")
                if exp is None and act is None:
                    nums = "—（未算出）"
                elif exp is None:
                    nums = f"实得 {act:.6g}" if isinstance(act, (int, float)) else f"实得 {act}"
                else:
                    e = f"{exp:.6g}" if isinstance(exp, (int, float)) else str(exp)
                    a = f"{act:.6g}" if isinstance(act, (int, float)) else str(act)
                    nums = f"预期 {e} / 实得 {a}"
                md.append(f"| `{r.get('formula', '')}` | {nums} | {lab} | {r.get('detail', '')} |")
            md.append("")
        if self.magnitudes:
            md += ["**量纲提示**", ""]
            md += [f"- `{m['param']}`：{m['detail']}" for m in self.magnitudes]
            md.append("")
        if self.plausibility:
            md += ["**数值合理性复核**", ""]
            md += [f"- `{p['param']}`（{p['side']} 侧）：{p['note']}" for p in self.plausibility]
            md.append("")
        if self.notes:
            md += [f"- {n}" for n in self.notes]
        return "\n".join(md)


def check_relations(paper_params: Any = None, code_params: Any = None,
                    *, relations: Optional[tuple] = None) -> ComputeReport:
    """对一对参数表跑一遍登记的关系式 —— **② 算数 → ③ 判定，两段分开记账**。

    输入可以是 `PaperInfo.params` / `CodeInfo.params`，也可以是 `{key: 值}` 的简写
    （对撞区视图与测试里都用得上）。

    2C 的分工（写在代码结构里，不只写在注释里）：
      ① `gather` 收输入 → ② `run` 出**数**（expected / actual）→ ③ `judge` 只下**结论**。
      判定阶段拿不到两侧参数表，也就**没有能力重新造数** —— 「③不造数」由此成立。
      ② 算数失败时如实记「未算」并跳到下一条，**不用默认值补数**。
    """
    pv, cv = _side_values(paper_params), _side_values(code_params)
    rep = ComputeReport()
    for rel in (relations or RELATIONS):
        try:
            inputs, reason = rel.gather(pv, cv)
        except Exception as ex:      # noqa: BLE001
            inputs, reason = None, f"关系式执行出错: {type(ex).__name__}: {ex}"
        if inputs is None:
            rep.skipped += 1
            rep.relations.append({
                "rid": rel.rid, "title": rel.title, "formula": rel.formula,
                "verdict": "skipped", "expected": None, "actual": None,
                "confidence": "none", "inputs": {},
                "computed_by": "② 实验站", "judged_by": "",
                "detail": f"本次未算：{reason}（不用默认值补数）",
            })
            continue
        # ---- ② 算数：只出数，不出结论
        try:
            measured = rel.measure(inputs)
        except Exception as ex:      # noqa: BLE001
            rep.skipped += 1
            rep.notes.append(f"关系式 {rel.rid} 算数失败: {type(ex).__name__}: {ex}")
            continue
        measured = dict(measured or {})
        measured.setdefault("rid", rel.rid)
        measured.setdefault("expected", None)
        measured.setdefault("actual", None)
        # ---- ③ 判定：拆过的只看 measured 里的数（拿不到 inputs, 也就没法造数）；
        #      没拆的旧关系式还得靠 inputs 才能跑（兼容路径, 判据靠标注而非结构）。
        _for_judge = measured if callable(rel.run) else inputs
        try:
            res = rel.decide(_for_judge)
        except Exception as ex:      # noqa: BLE001
            rep.skipped += 1
            rep.notes.append(f"关系式 {rel.rid} 判定失败: {type(ex).__name__}: {ex}")
            continue
        rep.checked += 1
        res = dict(res or {})
        res.setdefault("title", rel.title)
        res.setdefault("formula", rel.formula)
        # 判定函数的产出**只允许**有结论与措辞；数一律以 ② 站算出的为准（L8：不被判定改写）
        res["expected"] = measured.get("expected")
        res["actual"] = measured.get("actual")
        res["measured"] = {k: v for k, v in measured.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool)}
        res["inputs"] = {k: v for k, v in inputs.items() if isinstance(v, (int, float))}
        res["computed_by"] = "② 实验站"
        res["judged_by"] = "③ 验证站"
        rep.relations.append(res)
        if res.get("verdict") in ("mismatch", "hint"):
            rep.flagged += 1
    return rep


def _finding_parts(f: Any) -> tuple:
    """把一条结论（Finding 对象或同构 dict）拍成 (param_key, display_name, 论文原值, 代码原值)。

    两种形态都得认：图里传下来的是 Finding，测试里常常直接给 dict。
    """
    def _g(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)

    def _raw(side) -> Optional[str]:
        if side is None:
            return None
        if isinstance(side, dict):
            side = side.get("value") or side.get("raw_value")
        if side is None:
            return None
        if hasattr(side, "raw_value"):
            side = side.raw_value
        s = str(side).strip()
        return s or None

    return (_g(f, "param_key"), _g(f, "display_name"),
            _raw(_g(f, "paper")), _raw(_g(f, "code")))


def review_numbers(findings: Optional[list] = None,
                   paper_params: Any = None,
                   code_params: Any = None) -> dict:
    """对每条结论做一遍"算"的复核，汇总成一份可直接进报告/界面的结构。

    返回与 `ComputeReport.to_dict()` 同构的 dict。
    """
    rep = check_relations(paper_params, code_params)

    # ---- 量纲提示（逐条结论，只要两侧都取到了数）
    for f in (findings or []):
        key, name, p, c = _finding_parts(f)
        hint = magnitude_hint(p, c)
        if hint:
            rep.magnitudes.append({"param": name or key, "param_key": key,
                                   "detail": hint, "paper": p, "code": c})
    rep.flagged += len(rep.magnitudes)

    # ---- 数值合理性复核（复用对撞口的区间表，不另立一套）
    for f in (findings or []):
        key, name, p, c = _finding_parts(f)
        if not key:
            continue
        for side, raw in (("paper", p), ("code", c)):
            if raw is None:
                continue
            r = numeric_check(key, raw)
            if not r.get("plausible"):
                rep.plausibility.append({"param": name or key, "param_key": key,
                                         "side": side, "value": raw,
                                         "note": r.get("note", "")})
    rep.flagged += len(rep.plausibility)

    rep.notes.append(
        "「算」只做小规模复算 + 静态推演：训练级复算（要 GPU / 时间 / 钱）"
        "一律标注「需实机验证」，本模块不执行任何用户代码。")
    if rep.checked == 0:
        rep.notes.append("本次没有任何关系式的输入齐备 —— 如实记为「未算」，未用默认值补数。")
    return rep.to_dict()
