"""自主回头（2026-10-25）：让模型决定「这一趟跑完了，要不要再走一遍」。

## 这是什么

用户口径（2026-10-25 定）：
    「改成 LLM 自主选是否回头，给他自主权但最后的结果要程序验证就好，
      就是尽量往自主智能体这边改造，程序化对结果或者中途的过程进行验证。」

本模块把图里那条**原本硬编码为单向的**边（`loop -> selfcheck`）交出去：
模型看完「哪些条目推不动、已经试过什么」，自己决定**收尾**还是**回头再算一遍**，
回头时要给出**这一遍换什么**（针对哪条、把哪一侧假设成什么值）。

所以自主权是实的：走不走、走几遍、换什么，都是它定的。
但每一句都要过闸门——**模型说的是意见，能不能执行由程序判**。

## 为什么回头必须带「一个新值」

这是本模块最要紧的一条工程约束。

我们的「算」是**确定性复算**：输入不变，重跑一百遍输出一模一样。
如果只让模型说「再走一遍」而不给它改变输入的权力，那条回边就是**空转**——
跑完一轮，数字一个没变，还白花几分钟。

所以 `retry` 的合法形态只有一种：**带一个还没试过的假设值**。
`{"action":"retry","focus":["learning_rate"],"assume":{"side":"code","value":"0.03"}}`
读作「假设代码侧其实是 0.03，那关系式还成不成立」——这是反事实推演，
换个值结果真会变，回边才有意义。

## 七道闸门（程序验证「中途的过程」）

模型输出要逐条过，任何一条不过就退回收尾，并**如实记下被拒的理由**：

    G1 轮数上限   —— 无论它多想继续，回头最多 MAX_AUTO_ROUNDS 次
    G2 有事可做   —— 没有未收敛项时，"回头"无事可做，一律拒
    G3 不是幻觉   —— focus 里的键必须真实存在于未收敛清单（不在的过滤掉）
    G4 目标合法   —— assume.key 必须落在 focus 内；side 只能是 paper/code
    G5 值可解析   —— value 必须是个正数，不能是 NaN/inf/文字
    G6 不得空转   —— 这个值不能等于两侧现有值、也不能是已经试过的值
    G7 格式合规   —— action 属于封闭集合；retry 时必须三件套齐全

「结果」由谁验？—— 回头跑完，收敛数由 `loop.run_loop` 重新算（纯规则），
模型一个字都改不了。`record_outcome()` 把「这次回头多收敛了几条」如实回填，
**没多收敛也照实写**，那就是"这次回头没换来新证据"。

## 默认关 / 零成本

与 `role_llm` / `judge_llm` / `experiment_llm` 同款：开关关着时本模块一次都不调用。
且本模块**只在 `loop=True` 时有意义**——闭环没开就没有"回头"可言。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# 回头次数硬上限（与 loop 内部那 3 轮策略是两回事：那 3 轮是"换种查法"，
# 这里的 1 次 = 整条链重新算一遍，代价高得多）。
MAX_AUTO_ROUNDS = 2

# 动作封闭集合（越界一律丢弃，不"猜一个最近的"）
ACTIONS: tuple = ("retry", "stop")
SIDES: tuple = ("paper", "code")
# why 最长多少字（模型爱写长，截断而不是丢）
WHY_MAX = 120


def available(*, api_key: str = "", force_mock: bool = False) -> bool:
    """通道能不能用 —— 有 key 且没被强制离线。"""
    return bool(str(api_key or "").strip()) and not force_mock


# ---------------------------------------------------------------- 决策对象

@dataclass
class RetryDecision:
    """一次「要不要回头」的决定 + 它过闸门的全过程（报告与界面都读这份）。"""

    action: str = "stop"                 # stop | retry
    target: str = "selfcheck"            # compute | selfcheck
    focus: list = field(default_factory=list)
    assume: dict = field(default_factory=dict)
    why: str = ""                        # 采纳的理由（模型的原话，或规则给的）
    source: str = "rule"                 # llm（模型定）| rule（兜底）| gate（被闸门改判）
    gates: list = field(default_factory=list)   # [{name, ok, note}]
    note: str = ""                       # 一句话结论
    round_no: int = 0                    # 这是第几次回头决策（从 1 起）
    # 模型原始意见 —— 被闸门拒了也留着，报告里如实写「它想继续，程序没让」
    llm_action: str = ""
    llm_why: str = ""
    # 回头之后回填：这一遍换来多少新证据（record_outcome 写）
    before_resolved: int = -1
    after_resolved: int = -1
    outcome: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action, "target": self.target,
            "focus": list(self.focus), "assume": dict(self.assume),
            "why": self.why, "source": self.source,
            "gates": [dict(g) for g in self.gates],
            "note": self.note, "round_no": self.round_no,
            "llm_action": self.llm_action, "llm_why": self.llm_why,
            "before_resolved": self.before_resolved,
            "after_resolved": self.after_resolved, "outcome": self.outcome,
        }

    @property
    def rejected(self) -> bool:
        """模型想回头、但被闸门按下了 —— 这是最有说服力的留痕，报告要给一行。"""
        return self.llm_action == "retry" and self.action != "retry"


# ---------------------------------------------------------------- 问答通道

def _ask(prompt: str, key: str, *, force_mock: bool = False,
         base_url: str = "", model: str = "",
         ask: Optional[Callable[[str], Any]] = None) -> tuple:
    """一次 JSON 问答 —— 返回 `(dict 或 None, 说明)`。**绝不抛异常。**"""
    if ask is not None:
        try:
            return ask(prompt), "测试注入"
        except Exception as ex:            # noqa: BLE001
            return None, f"注入的问答出错（{type(ex).__name__}）"
    from .llm import ask_json
    return ask_json(prompt, key, mode="autonomy", max_tokens=900,
                    force_mock=force_mock, base_url=base_url, model=model)


# ---------------------------------------------------------------- 闸门

def _num(v: Any) -> Optional[float]:
    """把模型给的 value 解析成正数 —— 解析不了就返回 None（G5）。"""
    try:
        from .compute import to_number
        x = to_number(v)
    except Exception:                      # noqa: BLE001
        try:
            x = float(str(v).replace(",", ""))
        except (TypeError, ValueError):
            return None
    if x is None:
        return None
    try:
        if x != x or x in (float("inf"), float("-inf")):   # NaN / inf
            return None
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def _same(a: Any, b: Any) -> bool:
    """两个值是否等价（按数值比，1e-4 与 0.0001 算同一个 —— 否则 G6 形同虚设）。"""
    x, y = _num(a), _num(b)
    if x is not None and y is not None:
        return abs(x - y) <= max(abs(x), abs(y)) * 1e-9
    return str(a or "").strip() == str(b or "").strip()


def _gate(name: str, ok: bool, note: str, gates: list) -> bool:
    gates.append({"name": name, "ok": bool(ok), "note": note})
    return bool(ok)


def _unresolved_keys(loop_result: dict) -> dict:
    """未收敛清单 → `{param_key: {display, paper, code, reason}}`（G2/G3 的唯一依据）。"""
    out: dict = {}
    for u in (loop_result.get("unresolved") or []):
        if not isinstance(u, dict):
            continue
        k = str(u.get("param_key") or "")
        if k:
            out[k] = {"display": u.get("display") or k,
                      "paper": str(u.get("paper") or ""),
                      "code": str(u.get("code") or ""),
                      "reason": str(u.get("reason") or "")[:160]}
    return out


def _tried(loop_result: dict) -> dict:
    """已经试过的值 → `{param_key: [值, ...]}`（G6 用：换了个一样的值等于没换）。"""
    out: dict = {}
    for it in (loop_result.get("items") or []):
        if not isinstance(it, dict):
            continue
        k, v = str(it.get("param_key") or ""), it.get("attempt")
        if k and v not in (None, ""):
            out.setdefault(k, []).append(v)
    return out


# ---------------------------------------------------------------- 给模型看的事实

_PROMPT = """你是科研复现审计的**调度员**。这一趟已经跑完，现在由你决定：**收尾，还是回头再算一遍**。

背景：这是确定性复算系统——同样的输入跑几遍结果都一样。
所以「回头」**只有在带一个还没试过的假设值时才可能得到不同结果**，否则纯属空转。
每一条的「已试过的值」列在下面，那些再提就无效。

可选项只有两个：
- `stop`：收尾出报告。**大多数情况应该选这个**，尤其当推不动的条目缺关键信息时。
- `retry`：整条链重新算一遍，你必须给出 `focus`（针对哪几条）和 `assume`（换什么值）。

纪律（违反的会被程序拒绝，并记入报告）：
1. `action` 只能是 `stop` 或 `retry`，不许自创。
2. 选 `retry` 时必须同时给 `focus`（参数键列表）与 `assume`（含 key / side / value）。
   `side` 只能是 `paper` 或 `code`；`value` 必须是正数，不许写「约」「见论文」。
3. `assume.key` 必须出现在 `focus` 里；`focus` 只能取下面列出的参数键。
4. **不许把 value 写成两侧现有的值或已试过的值**（那等于没改，程序会拒）。
5. `why` 不超过 40 字，必须说清「换这个值想验证什么」，不许写空话。
6. 没有任何未收敛项时不要选 `retry`。
7. 拿不准就 `stop`——空转的成本是别人的时间。

当前是第 {round_no} 次回头决策（上限 {max_rounds} 次）。

未收敛条目（推不动的）：
{items}

已试过的取值：
{tried}

只输出 JSON：
{{"action":"stop","why":"..."}}
或
{{"action":"retry","focus":["参数键"],"assume":{{"key":"参数键","side":"code","value":"0.03"}},"why":"..."}}"""


def _facts(unres: dict, tried: dict) -> str:
    if not unres:
        return "（无 —— 所有不一致项都在轮内收敛了）"
    lines = []
    for k, u in unres.items():
        tv = tried.get(k) or []
        lines.append(
            f"- 键 `{k}`（{u['display']}）：论文 `{u['paper'] or '—'}` / "
            f"代码 `{u['code'] or '—'}`；卡在：{u['reason']}；"
            f"已试过：{'、'.join(str(x) for x in tv) if tv else '无'}")
    return "\n".join(lines)


def _parse(obj: Any) -> dict:
    """从模型回复里取决定 —— 只做**格式**层面的取用，合法性全交给闸门。"""
    if not isinstance(obj, dict):
        return {}
    act = str(obj.get("action") or "").strip().lower()
    if act not in ACTIONS:
        return {}
    out: dict = {"action": act,
                 "why": str(obj.get("why") or obj.get("reason") or "").strip()[:WHY_MAX]}
    focus = obj.get("focus")
    if isinstance(focus, list):
        out["focus"] = [str(x).strip() for x in focus if str(x).strip()]
    assume = obj.get("assume")
    if isinstance(assume, dict):
        out["assume"] = {"key": str(assume.get("key") or "").strip(),
                         "side": str(assume.get("side") or "").strip().lower(),
                         "value": assume.get("value")}
    return out


# ---------------------------------------------------------------- 主入口

def decide(loop_result: dict, *, round_no: int = 1,
           max_rounds: int = MAX_AUTO_ROUNDS,
           api_key: str = "", force_mock: bool = False,
           base_url: str = "", model: str = "",
           ask: Optional[Callable[[str], Any]] = None) -> RetryDecision:
    """模型决定「收尾还是回头」，再由七道闸门逐条验证。

    **无论模型说什么、无论通道通不通，本函数都返回一个可直接用的决定**：
    拿不到模型意见就按规则收尾（`source="rule"`），绝不假装它判过。
    """
    loop_result = dict(loop_result or {})
    unres = _unresolved_keys(loop_result)
    tried = _tried(loop_result)
    d = RetryDecision(round_no=int(round_no))
    gates: list = d.gates

    # 没有可判的对象时，连模型都不必问（省一次调用）
    if not available(api_key=api_key, force_mock=force_mock) and ask is None:
        d.source = "rule"
        d.why = "模型未启用（没配 Key 或已强制离线）"
        d.note = "按规则收尾：本趟不再回头"
        return d

    prompt = _PROMPT.format(round_no=d.round_no, max_rounds=int(max_rounds),
                            items=_facts(unres, tried),
                            tried=json.dumps(tried, ensure_ascii=False))
    obj, note = _ask(prompt, api_key, force_mock=force_mock,
                     base_url=base_url, model=model, ask=ask)
    got = _parse(obj)
    if not got:
        d.source = "rule"
        d.why = f"模型没给出可用决定（{note or '空回复'}）"
        d.note = "按规则收尾：本趟不再回头"
        return d

    d.llm_action = got.get("action", "")
    d.llm_why = got.get("why", "")
    d.source = "llm"
    d.why = d.llm_why

    # ---- stop 直接采纳（不需要 focus/assume）
    # G7 记在这里而不是解析完之后立刻记：报告要按 G1→G7 编号引用，
    # 先记它会让清单变成 G7、G1、G2… 读起来是乱的（G7 是"格式关"，天然排最后）。
    if d.llm_action == "stop":
        _gate("G7 格式合规", True, "action=stop", gates)
        d.action, d.target = "stop", "selfcheck"
        d.note = "模型判断证据已经够了，收尾"
        return d

    # ---- 从这里起都是 retry 意图，逐条过闸门
    focus = list(got.get("focus") or [])
    assume = dict(got.get("assume") or {})

    if not _gate("G1 轮数上限", d.round_no <= int(max_rounds),
                 f"第 {d.round_no} 次 / 上限 {max_rounds} 次", gates):
        return _fall(d, "回头次数已达上限，强制执行收尾")

    if not _gate("G2 有事可做", bool(unres),
                 f"未收敛 {len(unres)} 条" if unres else "没有推不动的条目，回头无事可做",
                 gates):
        return _fall(d, "没有未收敛项，回头等于空转")

    kept = [k for k in focus if k in unres]
    if not _gate("G3 不是幻觉", bool(kept),
                 (f"命中真实条目 {len(kept)} 条" if kept
                  else f"给出的 {len(focus)} 个键都不在未收敛清单里"), gates):
        return _fall(d, "模型指名的条目不存在，退回按规则收尾")
    d.focus = kept

    akey = str(assume.get("key") or "")
    aside = str(assume.get("side") or "")
    if not _gate("G4 目标合法", akey in kept and aside in SIDES,
                 (f"针对 `{akey}` 的 {aside} 侧" if akey in kept and aside in SIDES
                  else f"key `{akey or '—'}` 不在 focus 内，或 side `{aside or '—'}` 非法"),
                 gates):
        return _fall(d, "模型的回退目标不合法，退回按规则收尾")

    val = _num(assume.get("value"))
    if not _gate("G5 值可解析", val is not None,
                 f"假设值 {val}" if val is not None
                 else f"值 `{assume.get('value')}` 不是正数", gates):
        return _fall(d, "模型给的值不能用，退回按规则收尾")

    u = unres.get(akey) or {}
    cur = u.get(aside, "")
    seen = [str(x) for x in (tried.get(akey) or [])]
    same_cur = _same(val, cur)
    same_seen = any(_same(val, s) for s in seen)
    if not _gate("G6 不得空转", not (same_cur or same_seen),
                 ("这个值两侧现有值或已试过的值里已经有，换了等于没换"
                  if (same_cur or same_seen) else "是个没试过的新值"), gates):
        return _fall(d, "模型提的值换不出新信息，退回按规则收尾")

    d.action, d.target = "retry", "compute"
    d.assume = {"key": akey, "side": aside, "value": str(assume.get("value"))}
    d.why = d.llm_why or f"假设 {aside} 侧是 {val} 再算一遍"
    d.before_resolved = int((loop_result or {}).get("resolved") or 0)
    d.note = f"模型判断该再算一遍：假设 `{akey}` 的 {aside} 侧为 {val}"
    _gate("G7 格式合规", True, f"action=retry，三件套齐全（key/side/value）", gates)
    return d


def _fall(d: RetryDecision, note: str) -> RetryDecision:
    """被闸门改判为收尾 —— 模型的原话留着（`rejected` 属性据此为真）。"""
    _gate("G7 格式合规", True, f"action={d.llm_action}，格式本身没问题", d.gates)
    d.action, d.target = "stop", "selfcheck"
    d.source = "gate"
    d.note = note
    return d


# ---------------------------------------------------------------- 结果回填

def record_outcome(decisions: list, resolved_now: int) -> str:
    """回头跑完之后回填「这一遍换来多少新证据」—— **没换来也照实写**。

    ⚠ 收敛数由 `loop.run_loop` 重新算，模型碰不到（这是"结果由程序验证"的落点）。
    """
    if not decisions:
        return ""
    d = decisions[-1]
    if d.get("action") != "retry" or d.get("after_resolved", -1) >= 0:
        return ""
    before = int(d.get("before_resolved") or 0)
    after = int(resolved_now or 0)
    d["after_resolved"] = after
    if after > before:
        d["outcome"] = (f"回头有意义：收敛数 {before} → {after}，"
                        f"多收敛 {after - before} 条")
    else:
        d["outcome"] = (f"回头没换来新证据：收敛数仍是 {after} 条"
                        "（这一遍花的算力没换到结论，如实记下）")
    return d["outcome"]


def to_markdown(decisions: list) -> str:
    """把几次决策渲染成 markdown —— 追加在闭环那一节后面（界面下载件也带这段）。"""
    rows = [d for d in (decisions or []) if isinstance(d, dict)]
    if not rows:
        return ""
    out = ["", "## 自主回头（模型决定走不走，程序验每一句）", "",
           f"> 上限 {MAX_AUTO_ROUNDS} 次。回头**必须带一个没试过的假设值** —— "
           "确定性系统里换了输入才会换来不同输出，换了等于没换的一律拦下。", ""]
    for i, d in enumerate(rows, 1):
        want = d.get("llm_action") or d.get("action") or ""
        if want == "retry":
            a = d.get("assume") or {}
            want_txt = (f"再算一遍（假设 `{a.get('key', '—')}` 的"
                        f"{a.get('side', '—')} 侧为 {a.get('value', '—')}）")
        else:
            want_txt = "收尾"
        if d.get("action") == "retry":
            got = "**采纳**"
        elif d.get("source") == "gate":
            got = "**拦下**"
        else:
            got = "收尾"
        out.append(f"- 第 {i} 次：模型想{want_txt} → {got}。{d.get('note') or ''}")
        for g in (d.get("gates") or []):
            if not g.get("ok"):
                out.append(f"  - {g.get('name', '')}：{g.get('note', '')}")
        if d.get("outcome"):
            out.append(f"  - 事后核账：{d['outcome']}")
    s = summary(rows)
    if s:
        out += ["", f"**合计**：{s}。"]
    return "\n".join(out)


def summary(decisions: list) -> str:
    """一句话人话总结（报告 / 界面 / 日志共用这一份措辞）。"""
    if not decisions:
        return ""
    retried = [d for d in decisions if d.get("action") == "retry"]
    rejected = [d for d in decisions if d.get("llm_action") == "retry"
                and d.get("action") != "retry"]
    parts = []
    if retried:
        usef = [d for d in retried if int(d.get("after_resolved", -1)) >
                int(d.get("before_resolved", 0))]
        parts.append(f"模型要求回头 {len(retried)} 次，其中 {len(usef)} 次换到了新结论")
    if rejected:
        names = "、".join((d.get("gates") or [{}])[-1].get("name", "") for d in rejected)
        parts.append(f"{len(rejected)} 次被程序挡下（{names}）")
    if not parts:
        return "模型判断证据已够，本趟没有回头"
    return "；".join(parts)
