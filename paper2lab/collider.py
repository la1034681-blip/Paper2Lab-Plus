"""对撞引擎（阶段四）——把三方独立产出的结构化结论并排比对，产出冲突清单。

为什么需要独立的一层:
  总纲 §6 锁死「上下文隔离」: 解析员只看论文、代码验证员只看代码，
  彼此看不见。如果让它们在同一个 prompt 里"自己对上"，就退化成
  "一个模型自审自"，独立性归零。
  对撞口 = 程序把三份结构化结论（论文侧 / 代码侧 / 数值合理性）
  并排放在一起，用确定性规则比对 —— 比对本身不依赖任何模型，
  因此可复现、可审计、可做消融对照。

三条设计取舍:
  A. 喂线索不喂原文 —— 三个 Agent 各自已经抽好了"值 + 出处 + 原文片段"，
     对撞口只接字段，不接大段文本。上下文不会随角色数翻倍。
  B. 参数名先归一 —— lr / learning_rate / 学习率 是同一个东西，
     先归到 canonical key 再比对，否则"对不上"其实是"没对上"。
  C. 数值合理性单独一档 —— 论文写 lr=1e-4、代码写 0.001，
     程序能判定"两边都是合法的学习率取值"，但"它们不相等"是冲突；
     反过来论文写 lr=999，那是"值本身不合理"，与代码无关，
     两条结论必须分开报，不能互相吞掉。

与对齐引擎（aligner）的关系:
  aligner 是"单智能体循环"的比对（阶段三，已验证）；
  collider 是"多智能体对撞"的比对（阶段四）。
  两者口径必须一致（同样的 status / risk / confidence 枚举），
  这样消融对比才有意义: 同数据跑 classic / graph / collider 三版，
  结论应当一致，只改调度方式。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Category, Confidence, Risk, Status
from .synonyms import PARAM_SYNONYMS, canonical_key, values_equal
from .risk import assess_risk

# ---------------------------------------------------------------- 视图

# 一个 Agent 产出的"我看到的这一侧"——只有字段，没有原文长段落。
# value: 抽到的原始值（字符串）
# source: 出处（"Page 3 · §4" 或 "train.py:11"）
# doc:   原文片段（截断后的短句，供仲裁员复核用，不超过 120 字）
# confidence: "已确认"（文本层抽到）| "疑似"（图像/表格层推断）
# location: 原始位置（与 source 同源，供精确回查）
SideView = dict[str, dict[str, str]]

# 数值合理性检查的结果
# plausible: 这个值在该参数的合理范围内吗
# note: 一句话说明
NumericCheck = dict[str, dict]


# ---------------------------------------------------------------- 对撞记录

@dataclass
class Collision:
    """一次对撞的结果（一条参数冲突/一致）。"""
    param_key: str                 # 规范化参数名
    display_name: str              # 展示名
    category: Category             # 分类
    status: Status                 # 一致性状态
    confidence: Confidence         # 置信级
    risk: Risk                     # 风险级
    paper: Optional[str] = None    # 论文侧值（原始字符串）
    paper_source: str = ""         # 论文侧出处
    paper_doc: str = ""            # 论文侧原文片段
    code: Optional[str] = None     # 代码侧值
    code_source: str = ""          # 代码侧出处
    code_doc: str = ""             # 代码侧原文片段
    numeric_note: str = ""         # 数值合理性说明
    note: str = ""                 # 补充说明（如多处候选）
    risk_reason: str = ""          # 风险分级理由
    # 这条结论是哪（几）侧贡献的 —— 仲裁与红队据此判断"谁说的"
    evidence_origin: str = ""      # "paper" | "code" | "both" | "numeric"
    # 两侧的"规范化标记"（阶段四·模型版产出，可空）—— 只在**两侧都有**时才参与比对:
    # 原值写法不同、规范化后相同 => 判一致，并在 note 里如实写明"经规范化后等价"。
    # 规则版永远不填这两个字段，因此行为与模型版加入之前**逐字节一致**。
    paper_normalized: str = ""
    code_normalized: str = ""

    def to_dict(self) -> dict:
        return {
            "param_key": self.param_key, "display_name": self.display_name,
            "category": self.category.value, "status": self.status.value,
            "confidence": self.confidence.value, "risk": self.risk.value,
            "paper": self.paper, "paper_source": self.paper_source,
            "paper_doc": self.paper_doc, "code": self.code,
            "code_source": self.code_source, "code_doc": self.code_doc,
            "numeric_note": self.numeric_note, "note": self.note,
            "risk_reason": self.risk_reason,
            "evidence_origin": self.evidence_origin,
            "paper_normalized": self.paper_normalized,
            "code_normalized": self.code_normalized,
        }

    @staticmethod
    def from_dict(d: dict) -> "Collision":
        return Collision(
            param_key=d["param_key"], display_name=d["display_name"],
            category=Category(d["category"]), status=Status(d["status"]),
            confidence=Confidence(d["confidence"]), risk=Risk(d["risk"]),
            paper=d.get("paper"), paper_source=d.get("paper_source", ""),
            paper_doc=d.get("paper_doc", ""), code=d.get("code"),
            code_source=d.get("code_source", ""), code_doc=d.get("code_doc", ""),
            numeric_note=d.get("numeric_note", ""), note=d.get("note", ""),
            risk_reason=d.get("risk_reason", ""),
            evidence_origin=d.get("evidence_origin", ""),
            paper_normalized=d.get("paper_normalized", ""),
            code_normalized=d.get("code_normalized", ""),
        )


# ---------------------------------------------------------------- 数值合理性

# 常见参数的"合理区间"——只拦明显离谱的，不拦合法变体。
# (下界, 上界, 是否必须整数)
_NUMERIC_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "learning_rate": (1e-9, 1.0, False),
    "lr": (1e-9, 1.0, False),
    "batch_size": (1, 1 << 20, True),
    "epochs": (1, 1 << 20, True),
    "dropout": (0.0, 1.0, False),
    "weight_decay": (0.0, 1.0, False),
    "momentum": (0.0, 1.0, False),
    "warmup": (0.0, 1 << 20, False),       # 可以是步数(整数)也可以是比例
    "input_size": (1, 1 << 16, True),
    "num_workers": (0, 1 << 10, True),
    "seed": (0, 2 ** 31 - 1, True),
}


def _to_number(v: str) -> Optional[float]:
    """把抽到的值转成数字（科学计数法 / 百分比 / 分数都认）。"""
    if v is None:
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    # 百分比: 95% -> 0.95
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return None
    # 分数: 1/2 -> 0.5
    if "/" in s and s.count("/") == 1:
        try:
            a, b = s.split("/")
            if b != "0":
                return float(a) / float(b)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def numeric_check(key: str, value: str, *,
                  allow_none: bool = True) -> dict:
    """检查一个值对某个参数来说"合不合理"。

    返回 {plausible, note}。plausible=False 不等于"冲突"，
    它只说明"这个值本身可疑"，要由仲裁员结合两侧一起判。
    """
    if value is None or str(value).strip() == "":
        return {"plausible": True, "note": "该侧未给出确定值，不做数值合理性判断"}

    n = _to_number(value)
    if n is None:
        # 非数值型（优化器名 / 数据集名 / 模型名）: 不做数值检查
        return {"plausible": True, "note": "非数值型参数，不做数值合理性判断"}

    bound = _NUMERIC_BOUNDS.get(key) or _NUMERIC_BOUNDS.get(canonical_key(key))
    if bound is None:
        return {"plausible": True, "note": "无预设合理区间，不做数值合理性判断"}

    lo, hi, must_int = bound
    ok_lo = lo <= n <= hi
    ok_int = (not must_int) or abs(n - round(n)) < 1e-9
    kind = "整数" if must_int else "数值"
    if ok_lo and ok_int:
        return {"plausible": True, "note": f"{value} 在{kind}合理区间 [{lo:g}, {hi:g}] 内"}
    reasons = []
    if not ok_lo:
        reasons.append(f"{value} 超出{kind}合理区间 [{lo:g}, {hi:g}]")
    if not ok_int:
        reasons.append(f"{value} 不是该参数常见的{kind}")
    return {"plausible": False, "note": "；".join(reasons)}


# ---------------------------------------------------------------- 归一

def _canon(key: str) -> str:
    """把别名归到 canonical key（找不到原样返回，不猜）。"""
    return canonical_key(key) or key.strip()


# ---------------------------------------------------------------- 对撞

def collide(paper_view: SideView, code_view: SideView,
            numeric: Optional[NumericCheck] = None) -> list[Collision]:
    """三方对撞: 论文侧 vs 代码侧 vs 数值合理性。

    参数:
      paper_view: 论文侧 Agent 的抽取结果 {param_key: {value, source, doc, confidence, location}}
      code_view:  代码侧 Agent 的抽取结果（同上）
      numeric:    数值侧 Agent 的合理性检查 {param_key: {plausible, note}}（可空）

    返回:
      [Collision, ...]，按风险从高到低排（与 aligner 同源，方便对比）。
    """
    out: list[Collision] = []

    # 全部见过的参数名（并集），先归一
    keys: dict[str, None] = {}
    for k in paper_view:
        keys[_canon(k)] = None
    for k in code_view:
        keys[_canon(k)] = None
    if numeric:
        for k in numeric:
            keys[_canon(k)] = None

    for key in keys:
        pv = paper_view.get(key) or {}
        cv = code_view.get(key) or {}

        disp = PARAM_SYNONYMS.get(key, (key, []))[0]
        category = _category(key)

        p_val = str(pv.get("value", "")).strip() if pv else ""
        c_val = str(cv.get("value", "")).strip() if cv else ""
        p_conf = str(pv.get("confidence", "已确认")) if pv else "未确认"
        c_conf = str(cv.get("confidence", "已确认")) if cv else "未确认"
        # 规范化标记（模型版填；规则版恒为空串 -> _judge 走原来的分支，行为不变）
        p_norm = str(pv.get("normalized", "") or "").strip() if pv else ""
        c_norm = str(cv.get("normalized", "") or "").strip() if cv else ""

        # ---- 两侧都缺: 无事可做（不报噪声）
        if not p_val and not c_val:
            continue

        # ---- 数值合理性（独立一档，与"是否一致"正交）
        nc = (numeric or {}).get(key, {})
        numeric_note = nc.get("note", "") if nc else ""

        # ---- 判定
        status, risk, note, origin = _judge(
            key, p_val, c_val, p_conf, c_conf,
            nc.get("plausible") if nc else None, numeric_note,
            p_norm, c_norm)

        out.append(Collision(
            param_key=key, display_name=disp, category=category,
            status=status,
            confidence=_joint_conf(p_conf, c_conf), risk=risk,
            paper=p_val if p_val else None,
            paper_source=str(pv.get("source", "")) if pv else "",
            paper_doc=str(pv.get("doc", ""))[:120] if pv else "",
            code=c_val if c_val else None,
            code_source=str(cv.get("source", "")) if cv else "",
            code_doc=str(cv.get("doc", ""))[:120] if cv else "",
            numeric_note=numeric_note, note=note,
            risk_reason=_risk_reason(status, key, risk),
            evidence_origin=origin,
            paper_normalized=p_norm, code_normalized=c_norm,
        ))

    # 高风险在前（与 aligner 对齐）
    order = {Risk.HIGH: 0, Risk.MEDIUM: 1, Risk.LOW: 2, Risk.NONE: 3}
    out.sort(key=lambda c: (order.get(c.risk, 9), c.param_key))
    return out


def _category(key: str) -> Category:
    from .aligner import _PARAM_CATEGORY
    return _PARAM_CATEGORY.get(key, Category.PARAM)


def _joint_conf(p_conf: str, c_conf: str) -> Confidence:
    """两侧置信度的交集: 有一侧是"疑似"，整体就不报"已确认"。"""
    if "疑似" in (p_conf, c_conf):
        return Confidence.INFERRED
    return Confidence.CONFIRMED


def _judge(key: str, p_val: str, c_val: str, p_conf: str, c_conf: str,
           plausible: Optional[bool], numeric_note: str,
           p_norm: str = "", c_norm: str = ""
           ) -> tuple[Status, Risk, str, str]:
    """核心比对规则。

    **比对本身仍是纯确定的**（不打模型）—— 模型只在角色层产出一个"规范化标记"
    当作输入（见 paper2lab/agent_roles.canonicalize）。标记为空则完全等价于原行为。
    """
    # 两侧都给值了 -> 比对
    if p_val and c_val:
        pn, cn = _to_number(p_val), _to_number(c_val)
        if pn is not None and cn is not None:
            same = abs(pn - cn) <= 1e-12 * max(abs(pn), abs(cn), 1.0)
        else:
            same = values_equal(p_val, c_val)
        via = ""
        # 原值判不等时，再看两侧的规范化标记：标记都有且等价 => 判一致。
        # 这是"模型版"唯一能改结论的地方（例如论文写"余弦退火"、代码写
        # "CosineAnnealingLR"）。两侧任一侧没有标记就走不到这里，规则版不受影响。
        if not same and p_norm and c_norm and values_equal(p_norm, c_norm):
            same = True
            via = f"（写法不同，经规范化后等价: {p_norm}）"
        if same:
            return (Status.CONSISTENT, Risk.NONE,
                    f"论文 {p_val} 与代码 {c_val} 一致{via}", "both")
        # 不相等: 先看数值合理性，再定风险
        note = f"论文 {p_val} 与代码 {c_val} 不相等"
        if plausible is False:
            note += "；且" + numeric_note.replace("；", "，")
        return (Status.INCONSISTENT, Risk.LOW, note, "both")

    # 只有论文侧
    if p_val and not c_val:
        note = f"论文声明 {p_val}，代码侧未找到对应取值"
        r, why = assess_risk(Status.MISSING_IN_CODE, key)
        return (Status.MISSING_IN_CODE, r, note, "paper")

    # 只有代码侧
    if c_val and not p_val:
        note = f"代码取值 {c_val}，论文侧未声明"
        r, why = assess_risk(Status.MISSING_IN_PAPER, key)
        return (Status.MISSING_IN_PAPER, r, note, "code")

    return (Status.CONSISTENT, Risk.NONE, "两侧均未给出确定值", "")


def _risk_reason(status: Status, key: str, risk: Risk) -> str:
    try:
        _, why = assess_risk(status, key)
        return why
    except Exception:      # noqa: BLE001
        return ""


# ---------------------------------------------------------------- 仲裁（规则版）

@dataclass
class Arbitration:
    """仲裁员对一条冲突的裁决（规则版兜底，AI 可覆盖）。"""
    collision: Collision
    verdict: str            # "维持冲突" | "降级为疑似" | "撤销冲突" | "升级"
    reason: str             # 一句话理由
    confidence: Confidence  # 裁决后的置信度
    requires_human: bool    # 是否需要人工确认

    def to_dict(self) -> dict:
        return {
            "param_key": self.collision.param_key,
            "verdict": self.verdict, "reason": self.reason,
            "confidence": self.confidence.value,
            "requires_human": self.requires_human,
            "collision": self.collision.to_dict(),
        }


def arbitrate(collision: Collision) -> Arbitration:
    """规则版仲裁: 不是简单投票，而是按证据强度裁决。

    三条硬规则:
      1. 两侧都有值且不相等 -> 维持冲突（数值合理性只作补充，不推翻比对）；
      2. 只有一侧有值 -> 降为"疑似缺失"（缺的那一侧可能只是没抽到，
         不代表真的不存在）；
      3. 数值合理性不通过 + 两侧值又不等 -> 升级，标"需要人工确认"。
    """
    status = collision.status
    if status == Status.CONSISTENT:
        return Arbitration(collision, "一致，无需仲裁", "两侧取值一致",
                           collision.confidence, False)
    if status == Status.INCONSISTENT:
        if collision.numeric_note and "超出" in collision.numeric_note:
            return Arbitration(
                collision, "升级",
                "两侧取值不相等，且至少一侧的值超出该参数的合理区间",
                Confidence.CONFIRMED, True)
        return Arbitration(
            collision, "维持冲突",
            f"论文 {collision.paper} 与代码 {collision.code} 取值不同，"
            f"两侧证据出处分别为 {collision.paper_source} 与 {collision.code_source}",
            Confidence.CONFIRMED, False)
    if status in (Status.MISSING_IN_CODE, Status.MISSING_IN_PAPER):
        return Arbitration(
            collision, "降级为疑似",
            "仅一侧有证据，另一侧可能未被正确抽取，需补充证据后再定",
            Confidence.INFERRED, True)
    return Arbitration(collision, "维持原状", "无冲突",
                       collision.confidence, False)


def arbitrate_all(collisions: list[Collision]) -> list[Arbitration]:
    return [arbitrate(c) for c in collisions]


# ---------------------------------------------------------------- 红队（规则版）

@dataclass
class RedTeamNote:
    """红队挑出的一条漏洞。"""
    param_key: str
    flaw: str           # 漏洞类型
    detail: str
    severity: str       # high | medium | low

    def to_dict(self) -> dict:
        return {"param_key": self.param_key, "flaw": self.flaw,
                "detail": self.detail, "severity": self.severity}


def red_team(collisions: list[Collision],
             paper_view: SideView, code_view: SideView) -> list[RedTeamNote]:
    """红队: 专挑结论漏洞（对抗式验证，降幻觉）。

    挑漏洞的维度（全部是程序能查的，不是模型"觉得"）:
      1. 证据出处缺失 — 报了冲突却说不清在哪看到的；
      2. 单侧定案 — 只凭一侧就下了结论；
      3. 数值离谱 — 值本身超出合理区间却仍被当作有效值；
      4. 同义词未归并 — lr 与 learning_rate 被当成两个参数；
      5. 置信度不足 — 疑似值被当作已确认上报。
    """
    notes: list[RedTeamNote] = []
    for c in collisions:
        # 1. 证据出处缺失
        if c.status == Status.INCONSISTENT and (not c.paper_source or not c.code_source):
            notes.append(RedTeamNote(
                c.param_key, "证据出处缺失",
                f"报出 {c.display_name} 冲突，但出处不完整"
                f"（论文侧: {c.paper_source or '无'}，代码侧: {c.code_source or '无'}）",
                "high"))
        # 2. 单侧定案
        if c.status in (Status.MISSING_IN_CODE, Status.MISSING_IN_PAPER):
            notes.append(RedTeamNote(
                c.param_key, "单侧定案",
                f"{c.display_name} 仅一侧有证据，"
                "另一侧缺失可能是抽取得不够，不一定是论文/代码的问题",
                "medium"))
        # 3. 数值离谱
        if c.numeric_note and "超出" in c.numeric_note:
            notes.append(RedTeamNote(
                c.param_key, "数值离谱",
                f"{c.display_name} 的值 {c.paper or c.code} "
                f"{c.numeric_note}，建议人工复核该值是否为笔误",
                "medium"))
        # 4. 同义词未归并（检查视图里是否存在同义不同名的键）
        aliases = PARAM_SYNONYMS.get(c.param_key, (c.param_key, []))[1]
        seen_in_paper = [k for k in paper_view
                         if k.lower() != c.param_key.lower() and k.lower() in [a.lower() for a in aliases]]
        seen_in_code = [k for k in code_view
                        if k.lower() != c.param_key.lower() and k.lower() in [a.lower() for a in aliases]]
        if seen_in_paper or seen_in_code:
            notes.append(RedTeamNote(
                c.param_key, "同义词未归并",
                f"{c.display_name} 在视图里还以别名出现 "
                f"（论文侧: {seen_in_paper or '无'}，代码侧: {seen_in_code or '无'}），"
                "可能导致重复比对或漏比",
                "low"))
        # 5. 置信度不足
        if c.confidence == Confidence.INFERRED and c.status == Status.INCONSISTENT:
            notes.append(RedTeamNote(
                c.param_key, "置信度不足",
                f"{c.display_name} 的冲突基于推断值（非文本层确认），"
                "不应作为定案依据",
                "medium"))
    return notes