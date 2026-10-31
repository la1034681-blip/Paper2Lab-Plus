"""角色层（阶段四 → 阶段一 1B 转正为七位置）——独立 Agent，各只看自己该看的材料。

总纲 §6 锁死的生死线:
  上下文隔离 = 每个角色只看该看的，防污染、保独立性。
  若能互相看见或串行传递，就退化成"一个模型自审自"。

这条线在本文件是**强制的（代码层面）**，不是靠提示词求它别看:
  Analyst   只读 job.paper（论文侧），碰不到 code
  Verifier  只读 job.code（代码侧），碰不到 paper
  Numeric   只读两侧已经抽好的值，不做任何抽取

阶段一 1B 的口径变化（**只换名分与编制，不换行为**）:
  对外统一讲**七个智能体协作** —— 解析·检索·主持·实验·计算·审查·撰写
  （按出场序，见文件末尾 POSITIONS 编制表）。原来三个角色一个没废:
    Analyst → 解析员 · 论文视野 ／ CodeVerifier → 解析员 · 代码视野
    NumericVerifier → 计算员
  也就是隔离**从角色边界下移到解析员内部**，一点没松。
  执行体（Role 子类）与编制表（Position）刻意分开登记，两条线各自可改。

为什么规则版也能叫"Agent":
  本文件是"角色接口 + 规则实现"。接口（__call__ 接收材料、产出结构化视图）
  是 Agent 的契约；实现目前是确定性的（离线可测、可消融），
  未来把三个类的 __call__ 换成 LLM 调用，接口不变、下游不变。
  规则版同时是总纲 §6 说的"兜底"——AI 不可用时回退它，结论不漂移。

与 aligner 的关系:
  aligner 是单智能体循环的比对（阶段三），collider 是多智能体对撞（阶段四）。
  角色层只负责"抽"，不负责"比"——比对在 collider 里，
  这样"抽"与"比"可以独立替换、独立评测。
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from .collider import NumericCheck, SideView, numeric_check
from .models import CodeInfo, Evidence, PaperInfo
from .synonyms import canonical_key, normalize_number

# 代码证据可信度打分: 训练入口/配置文件 > 其他 > 测试/示例/教程。
# 与 aligner._evidence_score 同源（同一口径，避免两套规则给出不同结论）。
_AUX_PATH = re.compile(r"(^|[/\\_.\-])(test|tests|eval|demo|tmp|spec|example|examples|"
                       r"tutorial|sample|colab|benchmark|notebook)", re.IGNORECASE)


def _code_evidence_score(ev: Evidence) -> int:
    loc = str(ev.location).lower()
    s = 0
    if re.search(r"\btrain", loc):
        s += 4
    if "config" in loc or "option" in loc:
        s += 2
    if re.search(r"(^|[/\\_.\-])(test|tests|eval|demo|tmp|spec|example|examples|"
                 r"tutorial|sample|colab|benchmark|notebook)", loc) or "[cell" in loc:
        s -= 5
    if getattr(ev, "normalized", None) is not None:
        s += 2
    return s


def _pick_best_code_evidence(evs: list) -> Optional[dict]:
    """多处出现时挑最可信的一处; 测试/示例文件的不作为实现依据。"""
    if not evs:
        return None
    usable = [e for e in evs if _code_evidence_score(e) >= 0]
    if not usable:
        return None
    best = max(usable, key=_code_evidence_score)
    return best


def _entry(ev: Any, source: str, confidence: str) -> dict:
    """把一条证据转成 Agent 视图的一格。"""
    return {
        "value": getattr(ev, "raw_value", None),
        "source": getattr(ev, "location", ""),
        "doc": (getattr(ev, "snippet", "") or "")[:120],
        "confidence": confidence,
    }


# ---------------------------------------------------------------- 角色契约

class Role(ABC):
    """一个角色的契约: 接收 Job，产出结构化视图。

    规则版**不调模型**；构造时传入 RoleLLM 即切到模型版（见文件末尾的
    "模型后端"一节）。切换点只有这一处 —— 接口不变、下游不变。
    """
    name: str = ""

    def __init__(self, llm: Optional["RoleLLM"] = None):
        """llm: 可选的模型后端。默认 None = 纯规则版（线上默认）。

        模型版里角色**仍然只做规范化，不做抽取** —— 交给模型的只有
        "本侧已抽到的值 + 出处 + 片段"，隔离因此没有被破坏。
        """
        self.llm = llm
        self.last_note = ""      # 本次是否用上模型（如实留痕，供节点记录）

    @abstractmethod
    def __call__(self, job: Any) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------- 解析员（只看论文）

class Analyst(Role):
    """解析员: 只看论文，抽参数值。"""
    name = "analyst"

    def __call__(self, job: Any) -> SideView:
        view: SideView = {}
        paper = getattr(job, "paper", None)
        if paper is None:
            return view
        # 正文文本层: 已确认
        self._fill(paper.params, view, "paper", "已确认")
        # 多处声明: params_all 里取首个为主（其余在比对时由对撞口处理）
        for key, evs in (getattr(paper, "params_all", None) or {}).items():
            if key not in view and evs:
                view[key] = _entry(evs[0], "paper", "已确认")
        # 图像/表格通道: 未经文本层验证，降为"疑似"
        for key, evs in (getattr(paper, "image_params", None) or {}).items():
            if key not in view and evs:
                view[key] = _entry(evs[0], "paper", "疑似")
        # 模型版（可选）: 只把"本侧的值"规范化成可比对的标记，不抽取、不比对。
        # 送进模型的只有上面这张本侧视图 —— 代码验证员看到的东西这里看不到。
        if self.llm is not None and self.llm.ready:
            view, self.last_note = canonicalize(
                view, side="paper", role_name=self.name, llm=self.llm)
        return view

    @staticmethod
    def _fill(src: dict, view: SideView, source: str, confidence: str) -> None:
        for key, ev in (src or {}).items():
            if key not in view:
                view[key] = _entry(ev, source, confidence)


# ---------------------------------------------------------------- 代码验证员（只看代码）

class CodeVerifier(Role):
    """代码验证员: 只看代码，抽参数值。"""
    name = "verifier"

    def __call__(self, job: Any) -> SideView:
        view: SideView = {}
        code = getattr(job, "code", None)
        if code is None:
            return view
        aux_only = getattr(code, "aux_only_keys", None) or set()
        for key, evs in (getattr(code, "params", None) or {}).items():
            best = _pick_best_code_evidence(evs)
            if best is None:
                continue
            # 只在测试/示例文件里出现的参数: 如实报告，但标"疑似"
            # （这对应总纲 §7.3 的铁律: 无出处/出处不可靠不入库）
            confidence = "疑似" if key in aux_only else "已确认"
            view[key] = _entry(best, "code", confidence)
        # 模型版（可选）: 同上，只规范化本侧的值；模型看不到论文侧的任何一个字。
        if self.llm is not None and self.llm.ready:
            view, self.last_note = canonicalize(
                view, side="code", role_name=self.name, llm=self.llm)
        return view


# ---------------------------------------------------------------- 数值验证员（只看值）

class NumericVerifier(Role):
    """数值验证员: 对两侧已抽好的值做合理性检查，不自己抽值。"""
    name = "numeric"

    def __call__(self, job: Any) -> NumericCheck:
        checks: NumericCheck = {}
        paper_view = getattr(job, "paper_view", None) or {}
        code_view = getattr(job, "code_view", None) or {}
        for role, view in (("paper", paper_view), ("code", code_view)):
            for key, entry in view.items():
                val = entry.get("value")
                if val is None or str(val).strip() == "":
                    continue
                key_canon = canonical_key(key) or key
                c = numeric_check(key_canon, str(val))
                prev = checks.get(key_canon)
                # 同一参数两侧都检查: 只要有一侧不通过，整体就不通过
                if prev is None or not c["plausible"]:
                    checks[key_canon] = {
                        "plausible": c["plausible"],
                        "note": f"[{role}] {c['note']}",
                    }
        return checks


# ---------------------------------------------------------------- 角色表

ROLES = (Analyst(), CodeVerifier(), NumericVerifier())
ROLE_BY_NAME = {r.name: r for r in ROLES}


# ---------------------------------------------------------------- 七位置团队（阶段一 · 1B）
#
# 为什么要有这一层:
#   阶段四是「三个角色」—— 按**材料**分（只看论文 / 只看代码 / 只看值），
#   回答的是"谁能看什么"。阶段一转正为「七个位置」—— 按**能力**分
#   （读 / 搜 / 编排 / 做 / 算 / 审 / 写），回答的是"谁负责哪一段"。
#   隔离没有丢，只是**从角色边界下移到解析员内部**: 解析员手里仍是两只
#   互不通气的眼睛（论文视野 / 代码视野），仍是代码层强制，不靠提示词。
#
# 本层是**编制表**，不是新的执行体:
#   能真跑的还是下面那些类 / 模块（Agent 契约不变、结论不变）；
#   Position 只登记「出场序 / 能力 / 站 / 交接物 / 谁承载 / 现状」，
#   给界面、文档、消融当**唯一口径来源** —— 之前"越改越乱"就是两套编号并行。
#   落地节奏: 1B 只登记 ＋ 解析员 / 计算员改挂到新结构；其余五个位置的实现
#   由 1C（主持）、2A 2B（假设 / 筛排）、2D（实验）、3A（具名工具）逐个接管。


@dataclass(frozen=True)
class Position:
    """一个智能体位置的契约（阶段一 1B 定型）。

    与执行体（Analyst / CodeVerifier / ...）的分工:
      Position —— 编制表。出场序、能力、归属站、交接物、承载模块、现状。
      Role     —— 执行体。真正被节点调用、真正产出视图的那一层。
    两者分开的理由: 位置会随叙事变、承载模块会随分期换，
    但"谁读什么、谁看不见什么"这条线不能跟着变 —— 登记与执行各自独立改。
    """
    key: str          # 英文键（留痕 / 消融用）
    cn: str           # 中文名（对外统一口径）
    capability: str   # 五能力中的哪一个: 读 / 搜 / 编排 / 做 / 算 / 审 / 写
    station: str      # 归属站（贯穿全程 = 不固定在某站）
    handoff: str      # 交出去的产物（协作方式 = 交接物驱动）
    carrier: str      # 实现由谁承载
    impl: str         # 现状: 复用 / 待改 / 待建


POSITIONS: tuple = (
    Position("parser", "解析员", "读", "① 假设",
             "结构化线索 ＋ 疑点清单",
             "Analyst（论文视野）/ CodeVerifier（代码视野）", "复用"),
    Position("retriever", "检索员", "搜", "① 假设",
             "带出处的值（论文页 / 代码 file:line）",
             "retrieval.py", "复用"),
    Position("moderator", "主持人", "编排", "贯穿全程",
             "调用序列 / 排好序的假设卡",
             "planner.py（规则兜底）＋ LLM 规划调度", "待改（1C）"),
    Position("experimenter", "实验员", "做", "② 实验",
             "实验卡（设计 ＋ 实跑记录 ＋ 实际值）",
             "repro_plan.py + repro.py", "待改（2D）"),
    Position("calculator", "计算员", "算", "② ③",
             "实际值 / 独立复算结果",
             "NumericVerifier ＋ compute.py", "复用"),
    Position("reviewer", "审查员", "审", "③ 验证",
             "判定卡（支持 / 推翻 / 待验 ＋ 证据出处）",
             "collider.py + aligner.py + selfcheck.py", "复用"),
    Position("writer", "撰写员", "写", "④ 迭代",
             "发现报告 ＋ 修正建议",
             "report.py + patch.py", "复用"),
)
POSITION_BY_KEY = {p.key: p for p in POSITIONS}

# 编制表的防呆: 七个就是七个，多一个少一个都必须显式改这里（不靠文档数）
assert len(POSITIONS) == 7, "七位置 —— 增删位置必须同步改 POSITIONS 与文档口径"
assert len(POSITION_BY_KEY) == 7, "位置键必须唯一"


class Parser(Role):
    """解析员（读）: 内部两只眼睛 —— 论文视野 / 代码视野，互不通气。

    与阶段四的差别**只在包装**:
      阶段四: 节点里直接 new Analyst() + CodeVerifier()；
      阶段一: 归到一个「解析员」名下，两只眼睛仍各读各的那一个字段。
    隔离因此一点没松 —— 论文视野的入参里没有可用的 code 字段，
    代码视野也拿不到 paper；模型版同样只喂本侧（见 canonicalize）。
    """
    name = "parser"

    def __init__(self, llm: Optional["RoleLLM"] = None):
        super().__init__(llm)
        self.paper_eye = Analyst(llm)        # 论文视野（原 Analyst）
        self.code_eye = CodeVerifier(llm)    # 代码视野（原 CodeVerifier）

    def __call__(self, job: Any) -> dict:
        paper_view = self.paper_eye(job)
        code_view = self.code_eye(job)
        # 数值侧只吃这两张已抽好的视图 —— 与阶段四接线同口径，未改
        job.paper_view = paper_view
        job.code_view = code_view
        # 模型这次到底参没参与，两只眼睛各说各的，这里合并成一条留痕
        self.last_note = "；".join(
            n for n in (self.paper_eye.last_note, self.code_eye.last_note) if n)
        return {"paper": paper_view, "code": code_view}


class Calculator(Role):
    """计算员（算）: 只对两侧已抽好的值做合理性检查，自己**不抽值**。

    「下场真跑」（小规模复算 / 调具名工具）在 ② 实验站，由 compute.py 承担；
    本类对应的是「不造数、只查数」的那一半 —— 行为与 NumericVerifier 一致。
    """
    name = "calculator"

    def __init__(self, llm: Optional["RoleLLM"] = None):
        super().__init__(llm)
        self.numeric_eye = NumericVerifier()

    def __call__(self, job: Any) -> NumericCheck:
        return self.numeric_eye(job)


# ---------------------------------------------------------------- 模型后端（可选）
#
# 为什么要接模型:
#   规则版只会"对字面"。论文写"采用余弦退火调度"、代码写 CosineAnnealingLR(...)，
#   规则版判冲突，人一看就知道是一回事 —— 模型能读懂这种"同义不同写法"。
#
# 但它**只做这一件事**: 给本侧的值一个规范化标记（cosine / adamw / cifar10），
# 交给 collider 比对时使用。抽取、比对、仲裁、红队全部不动。
#
# 三条自我约束:
#   1. **只喂本侧材料** —— prompt 里只有这一侧已抽到的 (值 + 出处 + 片段)，
#      另一侧一个字符都不给。上下文隔离是这条改造的生死线，不能为了"让模型更聪明"破掉。
#   2. **数值型不送模型** —— 1e-4 与 0.0001 的关系程序算得比模型准，也省钱。
#   3. **失败一律回退规则版并说明原因** —— "这次模型没参与"必须告诉用户，不能静默。

_SIDE_CN = {"paper": "论文", "code": "代码"}

_ROLE_CN = {"analyst": "解析员（只看论文侧）",
            "verifier": "代码验证员（只看代码侧）",
            "numeric": "数值验证员（只看两侧已抽好的值）"}

_NORM_RULES = (
    "你是科研复现审计中的{role}。你**只能看到{side}侧的材料**，"
    "另一侧的材料你看不到、也不需要（这是刻意设置的隔离，不要索要）。\n"
    "你的唯一任务: 给下面每个参数的值一个**规范化标记**，供另一侧比对使用。\n"
    "硬性要求:\n"
    "1. 同一件事的不同写法必须给出**完全相同**的标记 —— 例如 "
    "'CosineAnnealingLR' / 'cosine annealing' / '余弦退火' 都写 'cosine'；"
    "'AdamW' / 'adam_w' / 'Adam with weight decay' 都写 'adamw'。\n"
    "2. 标记用简短的小写英文或数字（不超过 20 个字符），不要写成句子。\n"
    "3. 只依据给出的材料判断；材料不足以判断就给空字符串，**绝对不要猜**、不要编。\n"
    # 注意: 这段要过 str.format()，字面量花括号必须写两遍（{{ }}）。
    '4. 只输出 JSON: {{"normalized": {{"参数键": "标记"}}}}，不要任何解释文字。'
)


@dataclass
class RoleLLM:
    """角色可选的模型后端。

    **默认不启用** —— 线上"零成本"就靠这个默认值（不传 llm 就是纯规则版）。
    启用后角色仍然只做规范化、不做抽取，输入只有本侧已抽到的值。

    与 llm.LLMAnalyzer 的关系: 共用同一套 API / 缓存 / 降级（llm.ask_json），
    但用途不同 —— 那边是"给已确认的发现写风险解释"，这边是"给值写规范化标记"。
    """
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    force_mock: bool = False
    max_params: int = 12        # 一次最多送几条（控成本；超出的如实留在规则版）

    @property
    def ready(self) -> bool:
        return bool(self.api_key) and not self.force_mock


def _is_numeric(value: str) -> bool:
    """值能不能被程序解析成数字（能解析就轮不到模型上场）。"""
    return normalize_number(value) is not None


def _clean_token(tok: Any) -> str:
    """把一个标记收干净；不合规（空 / 太长 / 带换行）就丢弃（返回空串）。"""
    s = str(tok or "").strip()
    if not s or len(s) > 40 or "\n" in s or "\r" in s:
        return ""
    s = re.sub(r"\s+", " ", s)
    return s.lower() if s.isascii() else s


def _norm_prompt(side: str, role_name: str, items: list) -> str:
    """拼给模型的 prompt —— 内容**只有本侧**（隔离靠这个函数的入参保证）。"""
    lines = []
    for i, (key, entry) in enumerate(items, 1):
        val = str(entry.get("value", "") or "").strip()
        src = str(entry.get("source", "") or "")
        doc = str(entry.get("doc", "") or "")[:120]
        line = f'{i}. 参数键 "{key}"｜值: {val}｜出处: {src}'
        if doc:
            line += f'｜原文片段: "{doc}"'
        lines.append(line)
    return (_NORM_RULES.format(role=_ROLE_CN.get(role_name, role_name),
                               side=_SIDE_CN.get(side, side))
            + "\n\n【本侧参数】\n" + "\n".join(lines))


def canonicalize(view: SideView, *, side: str, role_name: str,
                 llm: "RoleLLM") -> tuple[SideView, str]:
    """用模型把**这一侧**的值规范化成可比对的标记。返回 (view', note)。

    · 送进模型的只有本侧视图；另一侧的内容不在 prompt 里（隔离不破）。
    · 数值型参数不送模型（程序归一更准，且免费）。
    · 任何一步出问题（没 key / 调用失败 / 返回不是 JSON / 标记不合规）
      都**回退规则版**，并把原因写进 note —— 不静默、不假装模型参与了。
    """
    if llm is None or not llm.ready:
        return view, f"角色 {role_name}: 未启用模型（规则版）"

    todo = [(k, e) for k, e in (view or {}).items()
            if str(e.get("value", "") or "").strip()
            and not _is_numeric(str(e.get("value", "")))
            and not e.get("normalized")]
    if not todo:
        return view, f"角色 {role_name}: 本侧无待规范化项（空值或数值型），未调用模型"
    todo = todo[: max(1, int(llm.max_params))]
    prompt = _norm_prompt(side, role_name, todo)

    try:
        from .llm import ask_json
        obj, note = ask_json(prompt, llm.api_key, mode=f"role_{role_name}",
                             max_tokens=500, force_mock=llm.force_mock,
                             base_url=llm.base_url, model=llm.model)
    except Exception as ex:      # noqa: BLE001
        return view, (f"角色 {role_name}: 模型调用异常"
                      f"（{type(ex).__name__}），回退规则版")
    if not obj:
        return view, f"角色 {role_name}: {note}，回退规则版"
    raw = obj.get("normalized")
    if not isinstance(raw, dict):
        return view, f"角色 {role_name}: 模型未返回 normalized 字段，回退规则版"

    out = dict(view)
    applied = 0
    for key, entry in todo:
        tok = _clean_token(raw.get(key))
        if not tok:
            continue
        e2 = dict(entry)
        e2["normalized"] = tok
        e2["normalized_by"] = "llm"
        out[key] = e2
        applied += 1
    if not applied:
        return view, f"角色 {role_name}: 模型未给出可用标记，回退规则版"
    return out, f"角色 {role_name}: {note}，规范化 {applied}/{len(todo)} 项"