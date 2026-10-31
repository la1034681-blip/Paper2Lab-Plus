"""模型先判、程序后验（2026-10-25 · 试点两处）。

## 这是什么

用户口径（2026-10-25 定）：
    「凡是 LLM 出力的地方给它判断的权限，但后面要程序化重新检查它的判断
      是否正确，复验兜底」，且**顺序是模型先判、程序后判**。

本模块落地**最小试点两处**：
    ① 假设真假 —— 这张卡是「真分歧 / 伪分歧（写法或单位不同）/ 一侧写错」
    ③ 参数配对 —— 这一项是「一致 / 不一致 / 一侧没找到 / 运行时才定 / 内部矛盾」

## 顺序版长什么样（与"并列版"的区别）

    build_hypotheses → **模型先判**（写 llm_verdict）→ derive（规则判，写 verdict）
                     → reconcile（比对，写 llm_agree）

模型的判断**排在规则之前产生**，但规则那一步**仍然独立跑完整判定**，
不是"只复核模型说的话"。这条是硬约束，见下面「两条铁律」。

## 两条铁律（写进代码，不是建议）

**铁律一：程序独立跑。**
    复验 = 把规则判定原样再跑一遍（它本来就独立于模型：`Relation.judge` 只吃算好的数、
    `derive`/`align` 只吃两侧取值）。**禁止**写成"只检查模型说的那一项对不对" ——
    那样模型说"一致"程序就只查一致性，会漏掉"两侧其实都抽错了"。
    顺序版最大的风险就是这个（确认偏误的工程版），本模块用「规则先于比对、口径完全独立」堵住。

**铁律二：不一致时按既有两级走，不新造级别。**
    模型判 + 规则验通过            → 结论照旧（`Confidence.CONFIRMED`，可重算）
    模型判 + 规则验不通过          → **分歧并列**：`verdict`/`status` 仍由规则写
                                     （下游报告、断言、界面全依赖它，覆盖等于全链重做），
                                     模型那句存进 `llm_*`，并标 `llm_agree=False`
    规则算不出（空档）+ 模型有判断 → 认模型的，但**只能标推测**（`Confidence.INFERRED`）、
                                     **不进结论段**、不参与评分 —— 它是线索，不是证据

## 防锚定

prompt 里**只有事实**（两侧取值、出处、可证伪判据），**不给规则结论**。
给了它就会抄，A/B 就没有意义了。

## 默认关

与 `experiment_llm` / `role_llm` 同款口径：线上零成本、结论可复现是硬要求。
关着时本模块**一次都不调用**，`llm_*` 全空，报告与界面不出现这一节。

## 成本口径

一次调用打包 `BATCH` 条（不是一条一问）。八套样本一套 1~3 次调用，
配合 `llm.ask_json` 的内容哈希缓存，同一份材料重复跑不重复花钱。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .derivation import DERIVE_EXCLUDED, DERIVE_FLAGGED, DERIVE_KEEP
# 中文标签表全项目只有这一份（llm.py 里那张）——这里复用，不另抄一遍。
from .llm import _STATUS_ZH

# ---- 封闭集合（模型输出越界一律丢弃，不"猜一个最近的"）
JUDGE_VERDICTS: tuple = (DERIVE_KEEP, DERIVE_EXCLUDED, DERIVE_FLAGGED)
_STATUS_BY_ZH: dict = {v: k for k, v in _STATUS_ZH.items()}
JUDGE_STATUSES: tuple = tuple(_STATUS_ZH.values())

# 一批最多判多少条（控 token；超了分批）
BATCH = 10
# why 最长多少字（模型爱写长，截断而不是丢）
WHY_MAX = 120


def available(*, api_key: str = "", force_mock: bool = False) -> bool:
    """通道能不能用 —— 有 key 且没被强制离线。"""
    return bool(str(api_key or "").strip()) and not force_mock


# ---------------------------------------------------------------- 问答通道

def _ask(prompt: str, key: str, *, mode: str, force_mock: bool = False,
         base_url: str = "", model: str = "",
         ask: Optional[Callable[[str], Any]] = None) -> tuple:
    """一次 JSON 问答 —— 返回 `(dict 或 None, 说明)`。**绝不抛异常。**

    `ask` 是离线测试注入（收 prompt、返回 dict），线上不传。
    """
    if ask is not None:
        try:
            return ask(prompt), "测试注入"
        except Exception as ex:            # noqa: BLE001
            return None, f"注入的问答出错（{type(ex).__name__}）"
    from .llm import ask_json
    return ask_json(prompt, key, mode=mode, max_tokens=1400,
                    force_mock=force_mock, base_url=base_url, model=model)


def _parse(obj: Any, n: int, allowed: tuple) -> tuple:
    """从模型回复里取判断 —— 返回 `(list[dict], 丢弃条数)`。

    严格三关（过不了就丢，不修补）：
      ① 顶层有 judgments 列表；② id 落在 1..n；③ verdict 属于封闭集合。
    """
    if not isinstance(obj, dict):
        return [], 0
    rows = obj.get("judgments")
    if not isinstance(rows, list):
        return [], 0
    out, dropped = [], 0
    for r in rows:
        if not isinstance(r, dict):
            dropped += 1
            continue
        try:
            i = int(r.get("id"))
        except (TypeError, ValueError):
            dropped += 1
            continue
        v = str(r.get("verdict") or "").strip()
        if not (1 <= i <= n) or v not in allowed:
            dropped += 1
            continue
        out.append({"id": i, "verdict": v,
                    "why": str(r.get("why") or r.get("reason") or "").strip()[:WHY_MAX]})
    return out, dropped


def _batches(seq: list, size: int = BATCH):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ---------------------------------------------------------------- ① 假设真假

_HYP_PROMPT = """你是科研复现审计的判定员。
下面每条是「论文与代码在同一个参数上对不上」的待判假设。**只根据给出的两侧原文**判断它属于哪种：

- {keep}：给的信息不足以断定它对不对，需要实机跑一次才能定
- {excluded}：两侧其实是同一件事的不同写法（含单位换算、步数与比例互换），等价，不是真分歧
- {flagged}：能明确指认哪一侧写错了，或哪一侧才是实际生效的

纪律（违反的条目不采纳）：
1. 只能选这三种词之一，不许自创、不许加修饰。
2. 每条给 why，不超过 40 字，**必须引用具体数值或原文片段**。
3. 信息不足一律判「{keep}」，不要猜、不要补数、不要用"典型值"。
4. 你没看到代码全文，只能就给出的两段原文判断；拿不准就是「{keep}」。

待判条目：
{items}

只输出 JSON：
{{"judgments":[{{"id":1,"verdict":"{keep}","why":"..."}}]}}"""


def _hyp_item(i: int, c) -> dict:
    """把一张卡压成模型要看的事实 —— **不含 verdict / derivation**（防锚定）。"""
    return {
        "id": i,
        "param": getattr(c, "key", ""),
        "claim": getattr(c, "claim", "")[:160],
        "paper": (f"{getattr(c, 'paper_value', '')}（{getattr(c, 'paper_where', '')}）"
                  if getattr(c, "paper_value", "") else "论文里没找到这一项"),
        "code": (f"{getattr(c, 'code_value', '')}（{getattr(c, 'code_where', '')}）"
                 if getattr(c, "code_value", "") else "代码里没找到这一项"),
        "推翻判据": getattr(c, "falsifiable_by", "")[:160],
    }


def judge_hypotheses(cards: list, *, api_key: str = "", force_mock: bool = False,
                     base_url: str = "", model: str = "",
                     ask: Optional[Callable[[str], Any]] = None) -> tuple:
    """模型先判一批假设卡 —— 返回 `(dict[key -> {verdict, why}], 说明)`。

    **不抛异常、不造判断**：拿不到就返回 `({}, 说明)`，由调用方如实标"没判"。
    """
    cards = [c for c in (cards or []) if getattr(c, "key", "")]
    if not cards:
        return {}, "没有待判的假设卡"
    if not available(api_key=api_key, force_mock=force_mock) and ask is None:
        return {}, "模型未启用（没配 Key 或已强制离线）"

    got: dict = {}
    notes: list = []
    dropped = 0
    for chunk in _batches(cards):
        items = [_hyp_item(i + 1, c) for i, c in enumerate(chunk)]
        prompt = _HYP_PROMPT.format(
            keep=DERIVE_KEEP, excluded=DERIVE_EXCLUDED, flagged=DERIVE_FLAGGED,
            items=json.dumps(items, ensure_ascii=False))
        obj, note = _ask(prompt, api_key, mode="judge-hyp", force_mock=force_mock,
                         base_url=base_url, model=model, ask=ask)
        rows, bad = _parse(obj, len(chunk), JUDGE_VERDICTS)
        dropped += bad
        if not rows:
            notes.append(note or "模型没给出可用判断")
            continue
        for r in rows:
            c = chunk[r["id"] - 1]
            got[c.key] = {"verdict": r["verdict"], "why": r["why"]}
        if "缓存命中" in (note or ""):
            notes.append("缓存命中")
    tail = f"，丢弃 {dropped} 条不合规输出" if dropped else ""
    return got, (f"模型判了 {len(got)} 条" + tail if got else "；".join(notes) or "没判出结果")


# ---------------------------------------------------------------- ③ 参数配对

_FIND_PROMPT = """你是科研复现审计的判定员。
下面每条是「论文声明的取值」与「代码实现」的对照。请判断每一项属于哪种：

- {consistent}：两侧写成同一个值（写法或单位不同但等价也算）
- {inconsistent}：两侧写死了不同的值
- {mic}：论文声明了，代码里找不到对应实现
- {mip}：代码里设了，论文一个字没提
- {not_fixed}：论文要求固定（如随机种子），代码没固定
- {unverifiable}：代码用变量/表达式动态确定，看你给的这段原文定不了实际取值
- {internal}：同一侧内部多处声明互相矛盾

纪律（违反的条目不采纳）：
1. 只能选这七个词之一，不许自创。
2. 每条给 why，不超过 40 字，必须引用具体数值或代码片段。
3. **只凭给出的原文判断**，看不到运行结果就不要推断运行时取值 —— 那属于「{unverifiable}」。
4. 拿不准判「{unverifiable}」，不要猜。

待判条目：
{items}

只输出 JSON：
{{"judgments":[{{"id":1,"verdict":"{consistent}","why":"..."}}]}}"""


def _find_item(i: int, f) -> dict:
    """把一条 Finding 压成事实 —— **不含 status / risk / ai_analysis**（防锚定）。"""
    def _side(ev) -> str:
        if ev is None:
            return "这一侧没有找到"
        return (f"{getattr(ev, 'raw_value', '')}（{getattr(ev, 'location', '')}；"
                f"原文：{str(getattr(ev, 'snippet', ''))[:80]}）")
    return {"id": i, "param": getattr(f, "display_name", "") or getattr(f, "param_key", ""),
            "paper": _side(getattr(f, "paper", None)),
            "code": _side(getattr(f, "code", None))}


def judge_findings(findings: list, *, api_key: str = "", force_mock: bool = False,
                   base_url: str = "", model: str = "",
                   ask: Optional[Callable[[str], Any]] = None) -> tuple:
    """模型先判一批参数配对 —— 返回 `(dict[param_key -> {status, why}], 说明)`。"""
    todo = [f for f in (findings or []) if getattr(f, "param_key", "")]
    if not todo:
        return {}, "没有待判的参数对照"
    if not available(api_key=api_key, force_mock=force_mock) and ask is None:
        return {}, "模型未启用（没配 Key 或已强制离线）"

    got: dict = {}
    notes: list = []
    dropped = 0
    for chunk in _batches(todo):
        items = [_find_item(i + 1, f) for i, f in enumerate(chunk)]
        prompt = _FIND_PROMPT.format(
            consistent=_STATUS_ZH[_st("CONSISTENT")],
            inconsistent=_STATUS_ZH[_st("INCONSISTENT")],
            mic=_STATUS_ZH[_st("MISSING_IN_CODE")],
            mip=_STATUS_ZH[_st("MISSING_IN_PAPER")],
            not_fixed=_STATUS_ZH[_st("NOT_FIXED")],
            unverifiable=_STATUS_ZH[_st("UNVERIFIABLE")],
            internal=_STATUS_ZH[_st("INTERNAL_INCONSISTENT")],
            items=json.dumps(items, ensure_ascii=False))
        obj, note = _ask(prompt, api_key, mode="judge-find", force_mock=force_mock,
                         base_url=base_url, model=model, ask=ask)
        rows, bad = _parse(obj, len(chunk), JUDGE_STATUSES)
        dropped += bad
        if not rows:
            notes.append(note or "模型没给出可用判断")
            continue
        for r in rows:
            f = chunk[r["id"] - 1]
            got[f.param_key] = {"status": _STATUS_BY_ZH[r["verdict"]], "why": r["why"]}
        if "缓存命中" in (note or ""):
            notes.append("缓存命中")
    tail = f"，丢弃 {dropped} 条不合规输出" if dropped else ""
    return got, (f"模型判了 {len(got)} 条" + tail if got else "；".join(notes) or "没判出结果")


def _st(name: str):
    """按名字取 Status 成员 —— 避免在这个模块里再抄一遍中文标签。"""
    from .models import Status
    return Status[name]


# ---------------------------------------------------------------- 规则之前的配对

@dataclass
class JudgePair:
    """给模型看的「待判配对」—— 只有两侧事实，**没有任何判定字段**。

    为什么不在 `align()` 之后再判: 用户口径是「模型先判、程序后判」，
    所以配对必须在规则产出 Finding **之前**就组好。
    这里刻意不复用 `Finding`: 它自带 `status`，带着答案去问模型等于锚定。
    """

    param_key: str
    display_name: str = ""
    paper: Any = None
    code: Any = None


def pairs_from_materials(paper: Any, code: Any) -> list:
    """从两侧解析结果里组「待判配对」—— **纯读，不改任何东西**。

    ⚠ 取代码侧证据复用 `aligner._pick_best_code_evidence`（同一个口径）。
       项目先例: `graph.py` 就 `from .runner import _store_session_memory`，
       私有函数跨模块复用是允许的，**另写一份**才会让两处口径漂移。
    """
    from .aligner import PARAM_SYNONYMS, _pick_best_code_evidence
    p = dict(getattr(paper, "params", None) or {})
    c = dict(getattr(code, "params", None) or {})
    # 顺序: 先论文侧出现顺序, 再补代码侧独有的 —— 与 align 的遍历口径一致
    keys = list(dict.fromkeys(list(p.keys()) + list(c.keys())))
    out = []
    for k in keys:
        ce = _pick_best_code_evidence(list(c.get(k) or []))
        disp = PARAM_SYNONYMS.get(k, (k, []))[0]
        out.append(JudgePair(param_key=k, display_name=disp,
                             paper=p.get(k), code=ce))
    return out


# ---------------------------------------------------------------- 复验与回填

def apply_hypotheses(cards: list, judged: dict) -> dict:
    """把模型判断回填进卡，并与**已经跑完的**规则结论比对。

    ⚠ 必须在 `derive()` **之后**调用 —— 规则结论没出来之前，没有可比的对手，
       那就不是"复验"，只是"记录"。
    ⚠ 只写 `llm_*` 三个字段，**绝不动 `verdict`**（那条由规则写，见模块头铁律二）。
    """
    stat = {"judged": 0, "agree": 0, "disagree": 0, "rule_empty": 0, "unjudged": 0}
    for c in (cards or []):
        j = judged.get(getattr(c, "key", ""))
        if not j:
            stat["unjudged"] += 1
            continue
        c.llm_verdict = j["verdict"]
        c.llm_why = j["why"]
        stat["judged"] += 1
        if not getattr(c, "verdict", ""):
            # 规则这次没给出结论 → 模型的话只能当**推测**，不进结论段（铁律二）
            c.llm_agree = None
            stat["rule_empty"] += 1
        elif c.verdict == c.llm_verdict:
            c.llm_agree = True
            stat["agree"] += 1
        else:
            c.llm_agree = False
            stat["disagree"] += 1
    return stat


def apply_findings(findings: list, judged: dict) -> dict:
    """同 `apply_hypotheses`，对象换成 Finding（只写 `llm_*`，不动 `status`）。"""
    stat = {"judged": 0, "agree": 0, "disagree": 0, "rule_empty": 0, "unjudged": 0}
    for f in (findings or []):
        j = judged.get(getattr(f, "param_key", ""))
        if not j:
            stat["unjudged"] += 1
            continue
        f.llm_status = j["status"]
        f.llm_why = j["why"]
        stat["judged"] += 1
        if not getattr(f, "status", None):
            f.llm_agree = None
            stat["rule_empty"] += 1
        elif f.status == j["status"]:
            f.llm_agree = True
            stat["agree"] += 1
        else:
            f.llm_agree = False
            stat["disagree"] += 1
    return stat


def summary(stat: dict) -> str:
    """一句话人话总结（报告 / 界面 / 日志共用这一份措辞）。"""
    if not stat or not stat.get("judged"):
        return "模型这一轮没有参与判定"
    s = (f"模型判了 {stat['judged']} 条，与程序算出来的一致 "
         f"{stat.get('agree', 0)} 条、不一致 {stat.get('disagree', 0)} 条")
    if stat.get("rule_empty"):
        s += f"、程序没算出结论 {stat['rule_empty']} 条（这是模型的推测，不是确证）"
    if stat.get("unjudged"):
        s += f"、模型没判 {stat['unjudged']} 条"
    return s
