"""上下文装配器（阶段三第2步）——把记忆里的线索拼进模型眼前的那一屏。

两条铁律（与 session.py 同源）:
  1. 记忆只记**线索**（原文里的值 + 出处 + 整句原文），不记**结论**；
  2. 装配器只做"装"，不做"判" —— 它决定喂什么，结论仍由模型/规则产出。

设计动机（2026-10-13）:
  之前 memory_lines 只是把线索拼成一段人话塞进 planner.notes。
  这对"提示"够用，但不够"对齐" —— 模型的上下文里论文侧和代码侧混在一起，
  得让模型自己把两边对上。装配器做的事:
    按 channel 把线索分成论文侧 / 代码侧，再按参数名对齐，
    拼成"参数名 | 论文侧说了什么 | 代码侧说了什么"的对照表。
  这样模型一眼就能看见"论文说 1e-4，代码写 1e-3"，而不是从两段话里自己翻。

  为什么不是"直接喂 Finding" —— 因为装配器在 make_plan 之前跑（计划阶段
  就要知道历史线索），那时还没做对齐，没有 Finding。它喂的是**线索**，
  不是结论。

  为什么不是"把全部线索一股脑塞进 prompt" —— 三条代价:
    不准（塞太多，关键几句被淹没）· 不快（token 越慢）· 不稳（每次装的不同 → 结论飘）
  装配器给每侧设上限，超了先裁"低置信/重复"的，且始终按参数名分组。

与 LangGraph 的关系（阶段三第3步预留）:
  本模块是纯函数，不依赖 LangGraph。第3步的状态机里，
  "上下文装配" 是一个独立节点，输入 (job, recall结果) → 输出 ContextScreen，
  交给后续的工具节点用。这里的设计已经按"一次装配、多次消费"写的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# 每侧最多给模型看几条线索（防止上下文膨胀）
_MAX_HINTS_PER_SIDE = 8
# 每个参数最多展示几处出处（内部不一致时够用即可）
_MAX_SOURCES_PER_PARAM = 3
# 原文片段截断长度（与 session._clip 同源，避免一条线索占半屏）
_MAX_DOC_LEN = 160

# 通道的展示名
_CHANNEL_LABEL = {"paper": "论文侧", "code": "代码侧"}
# 线索的判定类别
_LABEL = {
    "unset": "论文未给出确定值",
    "multi": "论文内多处声明不一致",
    "page": "参数出现在这些页",
    "channel": "上次从这条通道取到值",
    "code_multi": "代码内多处赋值",
    "code_aux": "只在测试/示例文件里出现",
}


@dataclass
class MemorySide:
    """某一侧（论文/代码）的线索摘要。"""
    param: str = ""
    label: str = ""          # 人类可读的类别
    sources: list = field(default_factory=list)   # [{location, doc}]
    value: str = ""
    confidence: str = ""     # "已确认" / "疑似" / "未定"


@dataclass
class ContextScreen:
    """装配好的一屏上下文。"""
    system: str = ""            # 系统提示（固定）
    overview: str = ""          # 本次任务的摘要一句话
    param_table: list = field(default_factory=list)   # [AlignedParam]
    history: str = ""           # 历史线索摘要（人话）
    budget_note: str = ""       # 裁剪说明（为什么有些没喂）

    def to_prompt(self) -> str:
        """拼成最终喂给模型的 prompt。"""
        parts = [self.system]
        if self.overview:
            parts.append(self.overview)
        if self.param_table:
            parts.append("【历史线索对照表】")
            for p in self.param_table:
                parts.append(p.to_line())
        if self.history:
            parts.append("【历史线索】")
            parts.append(self.history)
        if self.budget_note:
            parts.append(self.budget_note)
        return "\n\n".join(parts)


@dataclass
class AlignedParam:
    """一个参数在论文侧与代码侧的对照。"""
    param_key: str = ""
    paper: Optional[MemorySide] = None
    code: Optional[MemorySide] = None

    def to_line(self) -> str:
        """把这一行拼成 prompt 里的文本（不换行，每条参数一行）。"""
        out = f"参数 {self.param_key}"
        for side in (self.paper, self.code):
            if not side:
                continue
            label = _CHANNEL_LABEL.get(side.param, "")
            out += f" · {label}: {side.label}"
            if side.value:
                out += f"（{side.value}）"
            if side.sources:
                locs = [s["location"] for s in side.sources[:_MAX_SOURCES_PER_PARAM]]
                out += f" 出处: {'、'.join(locs)}"
        return out


# ---------------------------------------------------------------- 装配

def assemble(job: Any, recall: Optional[list] = None) -> ContextScreen:
    """把记忆线索装配成模型眼前的一屏。

    参数:
      job: 当前的 Job（取 paper/code 的指纹与参数名，决定"这次要看哪些参数"）
      recall: session.recall(fp) 的结果（外部查好传入，避免这里重复连库）。
              由调用方决定用哪一侧的指纹（论文-only 或论文+代码），
              装配器本身不决定指纹口径。

    返回:
      ContextScreen，调用方 .to_prompt() 后喂给模型，或把字段拆给不同节点。
    """
    screen = ContextScreen()

    # 系统提示: 角色 + 铁律 + 输出格式（固定不变，~1K token）
    screen.system = (
        "你是一名论文↔代码一致性审计员。你的任务是比对论文与代码里的参数取值，"
        "找出对不上的地方，并给出出处。"
        "铁律: 只依据给定的线索与原文做判断，不得编造值、不得脑补出处；"
        "找不到对应值就如实说'未找到'，不要用别的参数顶替。"
        "输出格式: 只输出 JSON，字段为 {findings:[{param_key, status, paper, code, note}]}。"
    )

    # 任务卡: 这次要查哪些参数（每步不同，~200 token）
    fp = getattr(job, "fingerprint", "") or ""
    paper = getattr(job, "paper", None)
    code = getattr(job, "code", None)
    keys = _param_keys(paper, code)
    if keys:
        screen.overview = (
            f"本次审计对象: 论文 {getattr(paper, 'title', '—') or '未命名'}\n"
            f"待比对参数（{len(keys)} 个）: {', '.join(keys)}"
        )

    # ---- 线索装配（核心）

    if not recall:
        screen.history = "未检索到历史线索（首次审计）"
        return screen

    # 按 channel 分两侧
    by_channel: dict[str, dict] = {"paper": {}, "code": {}}
    for r in recall:
        ch = str(r.get("channel", "")).lower()
        if ch not in by_channel:
            continue          # 陌生通道不掺入，防止污染
        by_channel[ch][str(r.get("param", "")).strip()] = r

    # 按参数名对齐，论文侧优先（论文是"标准答案"的来源）
    # 每侧先截到上限，防止上下文膨胀（预算说明在下游如实告知）
    cutoff = _MAX_HINTS_PER_SIDE
    keys: list[str] = []
    for ch in ("paper", "code"):
        for k in list(by_channel[ch].keys())[:cutoff]:
            if k not in keys:
                keys.append(k)
    screen.param_table = [_build_aligned(k, by_channel.get("paper", {}),
                                         by_channel.get("code", {}))
                          for k in keys]

    # ---- 线索摘要（人话，给 planner 的 notes 用）

    lines = _human_lines(by_channel)
    if lines:
        screen.history = "；".join(lines)
    else:
        screen.history = "未检索到历史线索（首次审计）"

    # ---- 预算: 超了的话如实告知裁掉了什么

    cutoff = _MAX_HINTS_PER_SIDE
    n_paper = len(by_channel.get("paper", {}))
    n_code = len(by_channel.get("code", {}))
    if n_paper > cutoff or n_code > cutoff:
        screen.budget_note = (
            f"线索预算: 每侧上限 {cutoff} 条，"
            f"论文侧 {n_paper} 条、代码侧 {n_code} 条，"
            f"超出部分已按参数名去重后截断（未改变任何已确认的结论）。"
        )

    return screen


def _param_keys(paper: Any, code: Any) -> list[str]:
    """本次要看的参数名（来自已解析产物；未解析时为空）。"""
    keys: list[str] = []
    for src in (paper, code):
        if src is None:
            continue
        params = getattr(src, "params", None) or {}
        keys.extend(k for k in params if k not in keys)
        all_p = getattr(src, "params_all", None) or {}
        for k in all_p:
            if k not in keys:
                keys.append(k)
    return keys


def _build_aligned(key: str, paper: dict, code: dict) -> AlignedParam:
    """把同一个参数在两侧的线索对齐。"""
    out = AlignedParam(param_key=key)
    if key in paper:
        out.paper = _build_side("paper", paper[key])
    if key in code:
        out.code = _build_side("code", code[key])
    return out


def _build_side(channel: str, row: dict) -> MemorySide:
    """把一条线索记录转成某一侧的摘要。"""
    side = MemorySide()
    side.param = channel
    side.value = str(row.get("value", "")).strip()
    side.sources = [{"location": str(row.get("source", "")),
                     "doc": _clip(str(row.get("doc", "")))}]

    # 类别: 按线索的内容判定
    if not side.value:
        side.label = _LABEL["unset"]
        side.confidence = "未定"
    elif side.value.count("；") > 0 or side.value.count(",") > 1:
        side.label = _LABEL["multi"]
        side.confidence = "已确认"
    else:
        side.label = "有声明"
        side.confidence = "已确认"
    return side


def _one_line(k: str, v: dict) -> str:
    """单条线索的人话写法（避开 f-string 嵌套括号）"""
    val = v.get("value", "（无值）")
    src = v.get("source", "—")
    return f"{k} → {val}（{src}）"


def _human_lines(by_channel: dict[str, dict]) -> list[str]:
    """把线索整理成人话提示行（与 memory.hint_lines 同源，但按 channel 分侧）。"""
    lines: list[str] = []
    for ch in ("paper", "code"):
        items = by_channel.get(ch, {})
        if not items:
            continue
        label = _CHANNEL_LABEL[ch]
        sample = "；".join(
            _one_line(k, v) for k, v in list(items.items())[:_MAX_HINTS_PER_SIDE]
        )
        lines.append(f"{label}历史线索: {sample}")
    return lines


def _clip(text: str, limit: int = _MAX_DOC_LEN) -> str:
    """把原文片段截成一段（≤limit 字符），避免一条线索占半屏。"""
    if not text or len(text) <= limit:
        return text
    # 以词为单位截断，避免在汉字中间或英文单词中间切开
    i = limit
    while i > 0 and not text[i].isspace() and text[i] not in (
        "，", "。", "、", ";", "；", "）", ")"
    ):
        i -= 1
    if i == 0:
        i = limit
    return text[:i].rstrip() + "…"