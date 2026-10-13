"""迭代回边（阶段五 · M2 闭环）——「对不上 → 定位根因 → 修正 → 再验证」，有界 ≤3 轮。

**这一步补的是整条链最后缺的一环**：M1 打通了"验证"（论文 vs 代码对不对得上），
但结论止于"这里不一样"。真做科研的人下一步一定是：**为什么不一样？改了会怎样？**

**三轮 = 三种逐级降档的修正策略**（不是把同一件事重试三遍）：

| 轮次 | 策略 | 要回答的问题 |
|---|---|---|
| 第 1 轮 | 可落地修正核验 | 这条"改代码对齐论文"的修正**能不能真落地**——根因定得到唯一生效点吗？补丁给得出来吗？ |
| 第 2 轮 | 换根因候选点 | 会不会是**抽取取错了赋值点**——真正的生效值其实是另一个候选、本来就和论文一样？ |
| 第 3 轮 | 反向观察（不计收敛） | 若目标是复现**代码行为**，按代码值对齐后是否自洽？——只供你决策，不算解决 |

三轮后仍未收敛的，**不硬凑**，升级为「待人工确认项」。这就是 M2 要求的"有界自主"：
能收敛的自己收敛，收敛不了的如实交回给人。

**为什么第 1 轮不是"把代码改成论文值再看是否相等"**（这条极容易写错）：
那样做**恒等成立**，任何一条都能"收敛"，收敛率永远 100%，这个指标就废了。
真正非平凡的问题是**可行性**：根因追不到唯一生效点、或补丁被程序闸门拒了，
就等于"知道该改成什么也改不了"，必须如实记为未收敛。

**四条硬约束**：
1. **绝不写文件、绝不改用户代码**：所谓"修正"是在**内存里**构造假设值重跑对撞口，
   是**反事实推演**，不是真改代码（补丁也只生成草案，沿用 patch.py 的"绝不落盘"）。
2. **有界**：`max_rounds` 上限 3，且**任何一轮没有改善就立即停止**，防死循环。
3. **复验走同一套引擎**：再验证复用阶段四的 `collider.collide`，不另写一份比对逻辑，
   否则"推演结论"会与"审计结论"各说各话。
4. **算不出来就停在原地**：根因追不出、补丁被拒，一律如实记录并保留原状态。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Status

__all__ = ["run_loop", "LoopReport", "LoopItem", "STRATEGIES", "MAX_ROUNDS"]

MAX_ROUNDS = 3

# 根因置信度达到这个档位才算"定得到唯一生效点"（rootcause 的口径: high | medium | low | none）
_TRUSTED_CONF = ("high", "medium")

# 三轮策略表（顺序即轮次）。
# observational=True 的轮次**只观察、不判收敛** —— 因为"把论文改成代码"是换目标，
# 不是解决问题；算作收敛的话"收敛率"永远 100%。
STRATEGIES: tuple = (
    {"round": 1, "key": "feasible_fix",
     "title": "可落地修正核验",
     "ask": "根因能否定位唯一生效点、补丁能否给出（能 ⇒ 这条修正可以真落地）",
     "observational": False},
    {"round": 2, "key": "rootcause_candidate",
     "title": "换根因候选点",
     "ask": "真正的生效值是否是另一个候选点（是 ⇒ 属抽取取错点，不是真冲突）",
     "observational": False},
    {"round": 3, "key": "reverse_to_code",
     "title": "反向观察：按代码修正论文",
     "ask": "若目标是复现代码行为，按代码值对齐后是否自洽（仅供决策，不计收敛）",
     "observational": True},
)

_VIEW_CONF = "已确认"


@dataclass
class LoopItem:
    """一条结论在某一轮里的推演记录。"""
    param_key: str = ""
    display: str = ""
    round: int = 0
    strategy: str = ""
    strategy_title: str = ""
    before: str = ""          # 推演前的状态
    after: str = ""           # 推演后的状态
    attempt: str = ""         # 这次假设改成了什么值
    target_loc: str = ""      # 假设改在哪个位置（根因追到的位置）
    side: str = ""            # 改的是哪一侧
    improved: bool = False
    resolved: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"param_key": self.param_key, "display": self.display,
                "round": self.round, "strategy": self.strategy,
                "strategy_title": self.strategy_title, "before": self.before,
                "after": self.after, "attempt": self.attempt,
                "target_loc": self.target_loc, "side": self.side,
                "improved": self.improved, "resolved": self.resolved,
                "reason": self.reason}


@dataclass
class LoopReport:
    """迭代回边的完整留痕。"""
    rounds: list = field(default_factory=list)      # 每轮汇总
    items: list = field(default_factory=list)       # 逐条推演记录
    resolved: int = 0
    unresolved: list = field(default_factory=list)  # 升级为待人工确认项
    max_rounds: int = MAX_ROUNDS
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"rounds": self.rounds, "items": self.items,
                "resolved": self.resolved, "unresolved": self.unresolved,
                "max_rounds": self.max_rounds, "notes": self.notes}

    def to_markdown(self) -> str:
        md = ["## 迭代回边（对不上 → 定位根因 → 修正 → 再验证）", "",
              f"> 有界自主：上限 {self.max_rounds} 轮；任何一轮没有改善就立即停止。",
              "> 所谓「修正」是**内存里的反事实推演** —— **不写任何文件、不改你的代码**。",
              ""]
        for r in self.rounds:
            md.append(f"**第 {r['round']} 轮 · {r['title']}** —— {r['ask']}")
            md.append(f"尝试 {r['tried']} 条，收敛 {r['improved']} 条，"
                      f"仍未解决 {r['remaining']} 条")
            for it in r.get("items", []):
                mark = "✅" if it["resolved"] else ("↑" if it["improved"] else "·")
                md.append(f"- {mark} `{it['display']}`：{it['before']} → {it['after']}"
                          f" —— {it['reason']}")
            md.append("")
        if self.unresolved:
            md += ["**仍未收敛 —— 升级为待人工确认项**（系统不硬凑）：", ""]
            md += [f"- `{u['display']}`（论文 `{u['paper'] or '—'}` / "
                   f"代码 `{u['code'] or '—'}`）：{u['reason']}"
                   for u in self.unresolved]
            md.append("")
        if self.notes:
            md += [f"- {n}" for n in self.notes]
        return "\n".join(md)


# ---------------------------------------------------------------- 复验

def _view_of(finding: Any, side: str, value: str) -> dict:
    """把一条 Finding 的一侧包成对撞口认的**取值格** `{value, source, doc, confidence}`。"""
    ev = getattr(finding, side, None)
    if ev is None:
        return {}
    return {"value": value,
            "source": getattr(ev, "location", ""),
            "doc": getattr(ev, "snippet", ""),
            "confidence": _VIEW_CONF}


def _reverify(finding: Any, side: str, new_value: str,
              paper_value: str, code_value: str) -> tuple:
    """在内存里把 `side` 侧改成 `new_value`，重跑对撞口，返回 (状态, 说明)。

    这是整个模块最核心、也最容易写错的动作：
      · 只改**一个**参数的一侧，其余保持原样 —— 否则分不清是哪条改动起的作用；
      · 复用 `collider.collide` + `numeric_check`，与审计同一套判定口径；
      · 视图是**两层**结构 `{参数键: {value, ...}}`，少包一层会直接 AttributeError
        （collide 里会对每个键 `.get("value")`）。
    """
    from .collider import collide, numeric_check

    p_val = paper_value if side != "paper" else new_value
    c_val = code_value if side != "code" else new_value

    key = getattr(finding, "param_key", "")
    pv = {key: _view_of(finding, "paper", p_val)} if p_val else {}
    cv = {key: _view_of(finding, "code", c_val)} if c_val else {}
    # 两侧数值合理性取"任一不合理就算不合理"（与 node_roles 的串联方式一致）
    nc: dict = {}
    for v in (p_val, c_val):
        if not v:
            continue
        r = numeric_check(key, v)
        if not r.get("plausible"):
            nc = r
            break
    cols = collide(pv, cv, {key: nc} if nc else None)
    c = next((x for x in cols if x.param_key == key), None)
    if c is None:
        return "", "推演后该参数两侧都没值了，退出比对"
    return c.status.value, c.note


# ---------------------------------------------------------------- 主流程

def _actionable(audit: Any) -> list:
    """挑出"值得迭代"的结论：值不一致，且两侧都取到了值（否则无从修正）。"""
    findings = list(getattr(audit, "findings", None) or [])
    order = {"high": 0, "medium": 1, "low": 2, "none": 3}
    findings = sorted(findings, key=lambda f: order.get(
        getattr(getattr(f, "risk", None), "value", ""), 9))
    out: list = []
    seen: set = set()
    for f in findings:
        if getattr(f, "status", None) != Status.INCONSISTENT:
            continue
        key = getattr(f, "param_key", "")
        if key in seen or getattr(f, "paper", None) is None or getattr(f, "code", None) is None:
            continue
        seen.add(key)
        out.append(f)
    return out


def _try_proposal(finding: Any, trace: Any, code_path: str) -> tuple:
    """试着为一条结论生成补丁草案（只读代码、绝不落盘）。返回 (草案, 拒绝理由)。"""
    if trace is None:
        return None, "根因追查不可用"
    try:
        from .patch import propose_patch
        p = propose_patch(finding, trace, code_path or "")
    except Exception as ex:      # noqa: BLE001
        return None, f"补丁草案生成失败: {type(ex).__name__}: {ex}"
    if p is None or not p.is_proposed:
        return p, (getattr(p, "refuse_reason", "") or "补丁被程序闸门拒绝")
    return p, ""


def run_loop(audit: Any, code_path: str = "", *,
             max_rounds: int = MAX_ROUNDS) -> LoopReport:
    """跑迭代回边。

    `audit` 需要 `findings` 与 `code`；`code_path` 用于根因追查与补丁草案
    （**只读**，绝不写）。返回 `LoopReport`（含逐轮留痕与未收敛清单）。
    """
    rep = LoopReport(max_rounds=max(0, min(max_rounds, MAX_ROUNDS)))
    items = _actionable(audit)
    if not items:
        rep.notes.append("没有「值不一致且两侧都取到值」的结论 —— 无可迭代项，如实记为未执行。")
        return rep
    if rep.max_rounds == 0:
        rep.notes.append("max_rounds=0 —— 未执行任何迭代。")
        return rep

    # 逐条状态账（跨轮共享；收敛后不再进入后续轮次）
    state: dict = {}
    for f in items:
        state[f.param_key] = {
            "finding": f, "status": Status.INCONSISTENT.value, "resolved": False,
            "paper": getattr(f.paper, "raw_value", ""),
            "code": getattr(f.code, "raw_value", ""),
            "history": [],
        }

    # 根因追查只做一次（只读代码）：贵一点，但不该每轮重跑
    traced: dict = {}
    try:
        from .rootcause import trace_param
        code = getattr(audit, "code", None)
        if code is not None:
            for f in items:
                traced[f.param_key] = trace_param(
                    f.param_key, code, paper_value=getattr(f.paper, "raw_value", ""),
                    display=f.display_name)
    except Exception as ex:      # noqa: BLE001
        rep.notes.append(f"根因追查不可用（{type(ex).__name__}: {ex}）"
                         "—— 第 1/2 轮将如实记为未收敛。")

    for strat in STRATEGIES[:rep.max_rounds]:
        pending = [f for f in items if not state[f.param_key]["resolved"]]
        if not pending:
            break
        rnd: dict = {"round": strat["round"], "key": strat["key"],
                     "title": strat["title"], "ask": strat["ask"],
                     "tried": 0, "improved": 0, "remaining": 0, "items": []}
        improved_this_round = 0

        for f in pending:
            st = state[f.param_key]
            before = st["status"]
            trace = traced.get(f.param_key)
            key = f.param_key

            if strat["key"] == "feasible_fix":
                item, improved = _round1_feasible(f, st, trace, code_path, before)
            elif strat["key"] == "rootcause_candidate":
                item, improved = _round2_candidate(f, st, trace, before)
            else:
                item, improved = _round3_observe(f, st, before)

            rnd["items"].append(item.to_dict())
            improved_this_round += 1 if improved else 0

        rnd["tried"] = sum(1 for it in rnd["items"] if it.get("attempt"))
        rnd["improved"] = improved_this_round
        rnd["remaining"] = sum(1 for f in items if not state[f.param_key]["resolved"])
        rep.rounds.append(rnd)
        rep.items.extend(rnd["items"])

        # ---- 有界：任何一轮没有改善就立即停止（防死循环）。
        # 观察轮（第 3 轮）按设计不产生"改善"，所以不参与这个判断。
        if improved_this_round == 0 and not strat.get("observational"):
            rep.notes.append(
                f"第 {strat['round']} 轮没有任何改善 —— 按「有界自主」提前停止，"
                "不把同一件事重试到上限。")
            break

    rep.resolved = sum(1 for k in state if state[k]["resolved"])
    for f in items:
        st = state[f.param_key]
        if st["resolved"]:
            continue
        last = st["history"][-1] if st["history"] else {}
        if last and last.get("attempt"):
            why = (f"仍未收敛（当前状态 {st['status']}）；最近一次尝试："
                   f"改 {last.get('side', '—')} 侧为 {last['attempt']}，"
                   f"结果 {last.get('after', '—')}")
        elif last:
            why = (f"仍未收敛（当前状态 {st['status']}）；最后一轮没有可用的候选值可试"
                   "（根因未定位到其它生效点）")
        else:
            why = "没有任何可推演的修正方向（根因与取值均不足）"
        rep.unresolved.append({
            "param_key": f.param_key, "display": f.display_name,
            "paper": st["paper"], "code": st["code"], "status": st["status"],
            "reason": why,
        })
    rep.notes.append(
        f"共观察 {len(items)} 条不一致项，{rep.resolved} 条在 {len(rep.rounds)} 轮内收敛，"
        f"{len(rep.unresolved)} 条升级为待人工确认项。")
    rep.notes.append("「修正」全部是内存中的反事实推演：**没有写任何文件、没有改任何代码**。")
    rep.notes.append("补丁草案只生成不落盘（沿用第三期 patch.py 的约束），应用与否由人决定。")
    return rep


def _round1_feasible(f: Any, st: dict, trace: Any, code_path: str,
                     before: str) -> tuple:
    """第 1 轮：这条"改代码对齐论文"的修正能否**真落地**。

    三个条件全过才算收敛（缺任何一个都说明"知道该改什么但改不了"）：
      ① 根因定得到唯一/较可信的生效点（trace.confidence ∈ high/medium）；
      ② 补丁草案给得出来（未被程序闸门拒绝）；
      ③ 按该值复验后冲突确实消失、且数值合理。
    """
    key = f.param_key
    propose_val = st["paper"]
    loc = getattr(trace, "verdict_loc", "") if trace else ""
    conf = getattr(trace, "confidence", "none") if trace else "none"

    # 条件 ①②
    proposal, refuse = _try_proposal(f, trace, code_path)
    blockers: list = []
    if conf not in _TRUSTED_CONF:
        blockers.append(f"根因未能定位唯一生效点（置信度 {conf}）")
    if proposal is None or not proposal.is_proposed:
        blockers.append(f"补丁不可给：{refuse}")

    after, note = _reverify(f, "code", str(propose_val), st["paper"], st["code"])
    if after != Status.CONSISTENT.value:
        blockers.append(f"按该值复验仍未一致（{after or '无结论'}）")

    resolved = not blockers
    item = LoopItem(
        param_key=key, display=f.display_name, round=1, strategy="feasible_fix",
        strategy_title="可落地修正核验", before=before,
        after=Status.CONSISTENT.value if resolved else before,
        attempt=str(propose_val), target_loc=loc, side="code",
        improved=resolved, resolved=resolved,
        reason=("可落地：把 `%s` 的 %s 改成 %s（补丁草案 %s）—— 复验冲突消失"
                % (loc or "代码", st["code"], propose_val,
                   getattr(proposal, "status", "—"))
                if resolved else "不可落地：" + "；".join(blockers)),
    )
    if resolved:
        st["status"] = Status.CONSISTENT.value
        st["resolved"] = True
    st["history"].append({"round": 1, "before": before, "after": item.after,
                          "attempt": str(propose_val), "side": "code", "loc": loc,
                          "observational": False})
    return item, resolved


def _round2_candidate(f: Any, st: dict, trace: Any, before: str) -> tuple:
    """第 2 轮：会不会是**抽取取错了赋值点**——另一个候选值本来就与论文一致。

    这一轮回答的是一个很实际的问题：报告说"代码是 0.001、论文是 1e-4"，
    但代码里有三处赋值，其中一处恰好是 1e-4 —— 那可能压根不是冲突，
    而是抽取选错了点。换候选值复验，一致就说明"属抽取问题，不是真冲突"。
    """
    key = f.param_key
    cands = list(getattr(trace, "candidates", None) or []) if trace else []
    hit = None
    for c in cands:
        if str(c.value) == str(st["code"]):
            continue      # 当前的抽取值，换了没意义
        after, _note = _reverify(f, "code", str(c.value), st["paper"], st["code"])
        if after == Status.CONSISTENT.value:
            hit = (c, after)
            break

    if hit is None:
        item = LoopItem(
            param_key=key, display=f.display_name, round=2,
            strategy="rootcause_candidate", strategy_title="换根因候选点",
            before=before, after=before, side="code",
            reason=("根因候选里没有与论文一致的取值"
                    + ("（该参数代码侧没有其它赋值点）" if not cands
                       else f"（共 {len(cands)} 个候选，逐个换过都不一致）")
                    + " —— 本项确属真冲突，不是抽取取错点"),
        )
        st["history"].append({"round": 2, "before": before, "after": before,
                              "attempt": "", "side": "code", "loc": ""})
        return item, False

    c, after = hit
    st["status"] = Status.CONSISTENT.value
    st["resolved"] = True
    st["history"].append({"round": 2, "before": before, "after": after,
                          "attempt": str(c.value), "side": "code",
                          "loc": f"{c.file}:{c.line}"})
    return LoopItem(
        param_key=key, display=f.display_name, round=2,
        strategy="rootcause_candidate", strategy_title="换根因候选点",
        before=before, after=after, attempt=str(c.value),
        target_loc=f"{c.file}:{c.line}", side="code", improved=True, resolved=True,
        reason=(f"第 1 轮不可落地，但代码里 `{c.file}:{c.line} = {c.value}` 与论文一致 —— "
                "**属抽取取错了赋值点，不是真冲突**（建议核对该处是否就是生效点）"),
    ), True


def _round3_observe(f: Any, st: dict, before: str) -> tuple:
    """第 3 轮：反向观察（**不计收敛**）。

    "把论文改成代码"是换复现目标，不是解决冲突。所以这一轮只产出一个供人决策的
    观察结论，绝不把它算作"已收敛"——否则收敛率永远 100%，指标就失去意义了。
    """
    after, note = _reverify(f, "paper", str(st["code"]), st["paper"], st["code"])
    if after == Status.CONSISTENT.value:
        reason = (f"按代码值 `{st['code']}` 对齐论文后字面一致 —— 仅说明"
                  "「若以复现代码行为为目标，取值口径是自洽的」；"
                  "**这是换目标不是解决问题，故不计入收敛**，仍列为待人工确认。")
    else:
        reason = (f"按代码值对齐后仍不一致（{after or '无结论'}）"
                  + (f"；{note}" if note else "")
                  + " —— 该处连「以代码为准」这条退路都不成立，务必人工处理。")
    st["history"].append({"round": 3, "before": before, "after": after,
                          "attempt": str(st["code"]), "side": "paper", "loc": ""})
    return LoopItem(
        param_key=f.param_key, display=f.display_name, round=3,
        strategy="reverse_to_code", strategy_title="反向观察：按代码修正论文",
        before=before, after=after, attempt=str(st["code"]), side="paper",
        improved=False, resolved=False, reason=reason,
    ), False
