"""问答窗口 · 后端（纯逻辑，不依赖 Streamlit）。

为什么单独一个模块，而不是塞进 app.py：
  ① 「问什么、拿哪些事实去问」是**口径**，和界面长什么样无关 ——
     口径要能被单独测（tests/verify_qa.py 就是直接调这里的函数）；
  ② 界面可以换（弹窗 / 侧栏 / 独立页），口径不该跟着改。

三条铁律（与项目其它部分完全一致）：
  · **事实只有一个来源**：所有数字都从 `audit` 里取，模型只负责组织语言，
    不许自己算、不许估、不许补充"常识" —— 否则用户会拿到一个看着很顺、
    但对不上这份材料的答案，比不给答案更糟；
  · **不改判定**：问答是解释层，任何回答都不写回 findings / readiness；
  · **拿不到模型就如实说**：没 key / 调用失败时不编答案，直接把检索到的
    审计事实摊开（这不是"降级回答"，是"证据检索"），并在文首写明未生成。
"""
from __future__ import annotations

import re
import time

from .llm import (RunCancelled, _STATUS_ZH, _call_deepseek, check_run,
                  endpoint_url_hint, is_local_endpoint, looks_non_api_reply,
                  qa_budget)
from .models import Status
from .synonyms import PARAM_SYNONYMS

#: 上下文预算（字符）。超了就截断 —— 模型上下文有限，塞太满反而会漏掉关键几条。
MAX_CTX = 7000

#: 云端限流的退避等待（秒）。**必须放模块级**：晚于函数定义再赋值，
#: 闭包就会捕到 None（LangGraph 那套踩过的坑，这里同理）。
#: 为什么是 3s、6s：DeepSeek 的 429 多数是分钟级的密度限制，
#: 实测"审计刚跑完紧接着问"最容易撞上，等几秒就恢复；等太久用户以为卡死。
_RETRY_WAITS = (3.0, 6.0)


def _is_throttle(ex: BaseException) -> bool:
    """这个异常是不是「请求太密集」那种可重试的限流。

    为什么不按 SDK 的异常类判断：`_call_deepseek` 是我方封装，
    抛出来的类名随 openai 版本变化（RateLimitError / APIStatusError / 自建的都可能），
    而**文本里的 429 是稳定的**。宁可多认几种写法，也不要因为类名变了就不重试。
    """
    _low = str(ex).lower()
    _name = type(ex).__name__.lower()
    return ("ratelimit" in _name or "rate limit" in _low or "429" in _low
            or "too many requests" in _low or "temporarily" in _low)

#: 一轮最多带几轮历史（用户问得越深，越需要前文，但也不能无限长）
MAX_TURNS = 8

# ---------------------------------------------------------------- 快捷问题
# 口径：**问的都是这份材料本身**，不是"这个平台怎么用"。
# 四条覆盖四站：判定 → 原因 → 动手 → 遗漏。
QUICK = (
    "这几个数为什么对不上？",
    "我要复现这篇论文，最先该改哪里？",
    "这次审计有没有漏掉什么？",
    "假设卡和实验卡都说明了什么？",
)


# ---------------------------------------------------------------- 取值小工具
def _g(obj, name, default=None):
    """取属性/字典键，取不到给默认值 —— 问答不该因为某个字段缺失就整段崩。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _s(v, limit: int = 160) -> str:
    t = re.sub(r"\s+", " ", str(v if v is not None else "")).strip()
    return t if len(t) <= limit else t[:limit] + "…"


def _ev(ev) -> str:
    """一条证据 -> 「原值（位置）」；没有就说没有，不编。"""
    if ev is None:
        return "—"
    raw = _s(_g(ev, "raw_value") or "", 40)
    loc = _s(_g(ev, "location") or "", 60)
    snip = _s(_g(ev, "snippet") or "", 70)
    out = raw or "（未取到值）"
    if loc:
        out += f"（{loc}）"
    if snip and snip not in out:
        out += f" 原文「{snip}」"
    return out


def _st(f) -> str:
    st = _g(f, "status")
    return _STATUS_ZH.get(st, str(st))


# ---------------------------------------------------------------- 事实摘要
def facts(audit, meta: dict | None = None, report: str = "") -> str:
    """把「这一轮审计到底查出了什么」压成一段可读的事实（给模型当唯一依据）。"""
    if audit is None:
        return "（本次还没有审计结论：用户尚未运行审计，或这一轮没有产出结论。）"

    L: list[str] = []
    meta = meta or {}
    paper = _g(audit, "paper")
    code = _g(audit, "code")
    r = _g(audit, "readiness") or {}
    findings = _g(audit, "findings") or []

    # ---- 材料
    L.append("【材料】")
    if meta.get("source"):
        L.append(f"来源：{meta['source']}")
    if meta.get("files"):
        L.append(f"文件：{meta['files']}")
    L.append(f"论文：《{_s(_g(paper, 'title') or '未识别标题', 80)}》，共 {_g(paper, 'page_count', 0)} 页")
    if code is not None:
        L.append(f"代码：{_g(code, 'python_files', 0)} 个 Python 文件；"
                 f"依赖清单 {len(_g(code, 'requirements') or {})} 项；"
                 f"是否固定随机种子：{'是' if _g(code, 'seed_fixed') else '未固定/未查到'}")

    # ---- 结论
    L.append("\n【结论】")
    total = r.get("total")
    L.append(f"综合复现就绪度：{'—（一个维度都没核验，不给分）' if total is None else str(total) + ' /100（只统计已核验的维度）'}")
    cats = r.get("categories") or {}
    if cats:
        L.append("分维度：" + "；".join(
            f"{k} {'未核验' if v is None else str(v)}" for k, v in cats.items()))
    unv = r.get("unverified") or {}
    for k, v in list(unv.items())[:5]:
        L.append(f"未核验原因 · {k}：{_s(v, 120)}")
    dom = r.get("dominant_issues") or []
    if dom:
        L.append("最该先看的问题：" + "、".join(dom))

    # ---- 逐条对表
    ok = [f for f in findings if _g(f, "status") == Status.CONSISTENT]
    L.append(f"\n【逐条对表】共 {len(findings)} 条，其中对得上 {len(ok)} 条，其余 {len(findings) - len(ok)} 条：")
    for i, f in enumerate(findings, 1):
        if _g(f, "status") == Status.CONSISTENT:
            continue
        line = (f"{i}. [{_st(f)}] {_s(_g(f, 'display_name') or _g(f, 'param_key'), 40)}"
                f"｜论文 {_ev(_g(f, 'paper'))}"
                f"｜代码 {_ev(_g(f, 'code'))}")
        note = _s(_g(f, "note") or "", 100)
        if note:
            line += f"｜说明：{note}"
        L.append(line)

    # ---- 三道核验 / 执行记录
    sc = _g(audit, "selfcheck") or {}
    if sc:
        L.append(f"\n【核验自证】检查 {sc.get('checked', 0)} 项，通过 {sc.get('passed', 0)} 项，"
                 f"未通过 {sc.get('failed', 0)} 项")
    ex = _g(audit, "execution_log") or {}
    if ex:
        plan = ex.get("plan") or {}
        ran = [t for t in (ex.get("tools") or []) if isinstance(t, dict)]
        skipped = [t for t in ran if str(t.get("action") or "") in ("skip", "skipped")]
        failed = [t for t in ran if t.get("status") == "fail" or t.get("error")]
        L.append(f"【执行】计划由{'模型' if plan.get('planner') == 'llm' else '规则'}排定；"
                 f"执行 {len(ran) - len(skipped)} 件，跳过 {len(skipped)} 件，失败 {len(failed)} 件"
                 + (f"（失败：{'、'.join(_s(t.get('key'), 20) for t in failed[:4])}）" if failed else ""))

    # ---- 其它产物（只说"有没有、几条"，细节由 focus() 按问题展开）
    hyp = _g(audit, "hypothesis_total") or len(_g(audit, "hypotheses") or [])
    exp = _g(audit, "experiment_total") or len(_g(audit, "experiments") or [])
    fig = _g(audit, "figure_total") or len(_g(audit, "figure_clues") or [])
    chem = _g(audit, "chem_total") or len(_g(audit, "chem_items") or [])
    other = [
        f"假设卡 {hyp} 张" if hyp else "",
        f"实验卡 {exp} 张" if exp else "",
        f"图表线索 {fig} 条" if fig else "",
        f"化学身份 {chem} 条" if chem else "",
    ]
    other = [x for x in other if x]
    if other:
        L.append("【其它产物】" + "；".join(other))
    return "\n".join(L)


# ---------------------------------------------------------------- 按问题展开细节
#: 关键词 -> 要补的那一块。命中越多，补得越深（但总量受 MAX_CTX 约束）。
_FOCUS = (
    (("参数", "数值", "对不上", "不一致", "为什么", "差", "lr", "学习率", "batch",
      "seed", "种子", "超参"), "param"),
    (("复现", "怎么做", "改哪", "动手", "跑起来", "第一步", "建议"), "repro"),
    # 「为什么不对 / 怎么改」要的是能动手的对象（2026-10-30 加）—— 只摆事实，改法让模型判断
    (("怎么改", "怎么解决", "怎么修", "修复", "解决方案", "怎么补", "该怎么办", "原因"), "fix"),
    (("假设", "证伪", "验一验", "推断"), "hyp"),
    (("实验", "跑数", "实际值", "复跑"), "exp"),
    (("化学", "分子式", "分子量", "cas", "smiles"), "chem"),
    (("图", "表", "配图", "曲线"), "fig"),
    (("跳过", "没跑", "计划", "调度", "为什么没"), "plan"),
    (("智能体", "对撞", "分歧", "角色", "审查"), "multi"),
    (("闭环", "迭代", "修正", "几轮"), "loop"),
    (("漏", "覆盖", "没查到", "完整"), "cover"),
)


def focus(audit, question: str) -> str:
    """按问题命中的领域，补一段更细的证据。没命中就返回空串。"""
    if audit is None or not question:
        return ""
    q = question.lower()
    want = set()
    for words, key in _FOCUS:
        if any(w in q for w in words):
            want.add(key)
    if not want:
        return ""

    out: list[str] = []
    findings = _g(audit, "findings") or []

    if "param" in want:
        bad = [f for f in findings if _g(f, "status") != Status.CONSISTENT][:6]
        if bad:
            out.append("【参数细节（原文与位置）】")
            for f in bad:
                p, c = _g(f, "paper"), _g(f, "code")
                out.append(
                    f"· {_g(f, 'display_name') or _g(f, 'param_key')}：判定「{_st(f)}」\n"
                    f"  论文侧：{_ev(p)}\n"
                    f"  代码侧：{_ev(c)}\n"
                    f"  风险理由：{_s(_g(f, 'risk_reason') or '未给出', 140)}")

    if "fix" in want:
        # 「为什么不对 / 怎么改」要的就是**能动手的对象**：每条不一致的两侧原值就是
        # 「改成什么」的参照；实验卡的跑法、复现方案给「改完怎么验」。
        # 这里**只摆事实不做判断** —— 该改哪一侧、为什么错，留给模型解释（它要标注是推断）。
        out.append("【可动手的对象（本轮审计给出的）】")
        n = 0
        for f in findings:
            if _g(f, "status") == Status.CONSISTENT:
                continue
            out.append(f"· {_g(f, 'display_name') or _g(f, 'param_key')}："
                       f"论文 {_g(_g(f, 'paper'), 'raw_value') or '—'}"
                       f"｜代码 {_g(_g(f, 'code'), 'raw_value') or '—'}"
                       f"｜判定「{_st(f)}」")
            n += 1
            if n >= 6:
                break
        for e in (_g(audit, "experiments") or [])[:3]:
            design = _s(_g(e, "design") or _g(e, "claim") or "", 120)
            rerun = _s(_g(e, "rerun_plan") or "", 120)
            if design or rerun:
                out.append("· 实验卡：" + (design or "（未写设计）")
                           + (f"｜跑法：{rerun}" if rerun else ""))
        rp = _g(audit, "repro_plan") or {}
        if rp:
            out.append("· 复现方案：" + _s(rp.get("summary") or rp.get("plan") or rp, 400))

    if "repro" in want:
        rp = _g(audit, "repro_plan") or {}
        if rp:
            out.append("【复现方案】" + _s(rp.get("summary") or rp.get("plan") or rp, 500))
        for t in (_g(audit, "rootcause") or [])[:3]:
            out.append(f"· 根因（{_s(_g(t, 'param_key') or '', 24)}）：{_s(_g(t, 'conclusion') or _g(t, 'summary') or t, 220)}")
        for p in (_g(audit, "patches") or [])[:3]:
            out.append(f"· 修正草案（{_s(_g(p, 'param_key') or '', 24)}）："
                       f"{_s(_g(p, 'diff') or _g(p, 'patch') or _g(p, 'summary') or p, 220)}")

    if "hyp" in want:
        hs = _g(audit, "hypotheses") or []
        if hs:
            out.append(f"【假设卡（共 {_g(audit, 'hypothesis_total') or len(hs)} 张，下列前 {min(6, len(hs))} 张）】")
            for h in hs[:6]:
                out.append(
                    f"· [{_s(_g(h, 'verdict') or '待验', 12)}] {_s(_g(h, 'claim'), 110)}"
                    f"｜怎么才算被推翻：{_s(_g(h, 'falsifiable_by'), 90)}")

    if "exp" in want:
        es = _g(audit, "experiments") or []
        if es:
            out.append(f"【实验卡（共 {_g(audit, 'experiment_total') or len(es)} 张，下列前 {min(5, len(es))} 张）】")
            for e in es[:5]:
                out.append(f"· {_s(_g(e, 'title') or _g(e, 'key') or '', 60)}"
                           f"｜实跑值：{_s(_g(e, 'actual') or _g(e, 'measured') or '未跑', 40)}"
                           f"｜状态：{_s(_g(e, 'status') or '', 20)}")

    if "chem" in want:
        cis = _g(audit, "chem_items") or []
        if cis:
            out.append(f"【化学身份核验（共 {_g(audit, 'chem_total') or len(cis)} 条）】")
            for c in cis[:6]:
                out.append(f"· {_s(_g(c, 'formula') or _g(c, 'raw') or '', 40)}"
                           f"｜判定：{_s(_g(c, 'status') or _g(c, 'verdict') or '', 20)}"
                           f"｜说明：{_s(_g(c, 'note') or _g(c, 'reason') or '', 100)}")

    if "fig" in want:
        fs = _g(audit, "figure_clues") or []
        if fs:
            out.append(f"【图表线索（共 {_g(audit, 'figure_total') or len(fs)} 条）】")
            for f in fs[:5]:
                out.append(f"· {_s(_g(f, 'caption') or _g(f, 'title') or _g(f, 'text') or '', 100)}")

    if "plan" in want:
        plan = (_g(audit, "execution_log") or {}).get("plan") or {}
        steps = plan.get("steps") or []
        if steps:
            out.append(f"【本次计划（planner={plan.get('planner') or '—'}）】")
            for s in steps[:20]:
                out.append(f"· {_s(_g(s, 'key') or _g(s, 'tool') or '', 24)}："
                           f"{_s(_g(s, 'action') or '', 12)}"
                           f"（理由：{_s(_g(s, 'reason') or _g(s, 'why') or '—', 70)}）")

    if "multi" in want:
        col = _g(audit, "collider") or {}
        if isinstance(col, dict) and col:
            out.append("【多智能体对撞】" + _s(
                f"互证 {col.get('agree', '—')} 项，分歧 {col.get('disagree', '—')} 项；"
                + ("分歧点：" + "、".join(map(str, (col.get("diffs") or [])[:4]))
                   if col.get("diffs") else ""), 400))

    if "loop" in want:
        lp = _g(audit, "loop") or {}
        if lp:
            out.append("【复现闭环】" + _s(
                f"轮次 {lp.get('rounds', '—')}；收敛：{lp.get('converged', '—')}；"
                f"记录 {len(lp.get('history') or [])} 条", 300))

    if "cover" in want:
        cov = _g(audit, "coverage") or {}
        if cov:
            out.append("【覆盖情况】" + _s(
                f"页覆盖率 {cov.get('page_ratio', cov.get('pages', '—'))}；"
                f"声明覆盖率 {cov.get('decl_ratio', '—')}；"
                f"审计可信度 {cov.get('confidence_level', '—')}"
                f"（{_s(cov.get('confidence_score'), 8)}）", 300))

    return "\n".join(out)


def _hit_alias(q: str, alias: str) -> bool:
    """问题里有没有提到这个别名。

    ⚠ 英文别名必须按**词边界**匹配：`eta` 是学习率的别名，但直接 `in` 判定会被
      `beta` / `theta` 这类词命中；`bs` 也会命中 `subsample`。中文别名没有词边界
      概念，用包含判断即可。
    """
    a = (alias or "").strip().lower()
    if not a:
        return False
    if re.search(r"[a-z0-9]", a):
        return re.search(rf"(?<![a-z0-9_]){re.escape(a)}(?![a-z0-9_])", q) is not None
    return a in q


def top_hits(audit, question: str, k: int = 5) -> list[str]:
    """按问题挑出最相关的几条审计事实（纯规则打分，不经模型）。

    用途：① 没配模型时直接摊开给用户看；② 有模型时也是"到底引了哪几条"的凭据。

    ⚠ 2026-10-27：原来只拿 `display_name` / `param_key` 比对，而这两个都是英文
      （Learning Rate / learning_rate）—— 用户问「学习率差多少」命中 0 条，
      降级清单直接空白。改成连**同义别名表**一起比（`synonyms.PARAM_SYNONYMS`，
      与对齐引擎用的是同一张表），中文问法才对得上。
    """
    if audit is None:
        return []
    q = (question or "").lower()
    scored: list[tuple[int, str]] = []
    for f in (_g(audit, "findings") or []):
        name = str(_g(f, "display_name") or _g(f, "param_key") or "")
        key = str(_g(f, "param_key") or "")
        score = 0
        if name and name.lower() in q:
            score += 5
        if key and key.lower() in q:
            score += 4
        aliases = (PARAM_SYNONYMS.get(key) or ("", []))[1]
        if any(_hit_alias(q, a) for a in aliases):
            score += 4
        for ev in (_g(f, "paper"), _g(f, "code")):
            raw = str(_g(ev, "raw_value") or "")
            if raw and raw.lower() in q:
                score += 3
        bad = _g(f, "status") != Status.CONSISTENT
        if bad and any(w in q for w in ("不一致", "对不上", "错", "问题", "风险", "为什么")):
            score += 2
        if score <= 0:
            continue
        scored.append((score,
                       f"[{_st(f)}] {name or key}｜论文 {_ev(_g(f, 'paper'))}"
                       f"｜代码 {_ev(_g(f, 'code'))}"))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


# ---------------------------------------------------------------- 提问
#: 口径演进（2026-10-30，用户定版）：
#: 原来只许「复述审计事实」，于是问「为什么不对」只能答「数据里没有」—— 把答疑助手做成了
#: 检索引擎。现在放开两件事：**允许解释成因、允许给修复方案**，但加一条死线 +
#: 一条呈现纪律：数字与出处仍不许编；模型自己的判断必须显式标出来（「最可能/我判断」），
#: 让用户分得清哪句是材料原文、哪句是模型推断。放开的是"能推断"，不是"能编"。
_SYS = """你是 Paper2Lab 的答疑助手。用户刚看完一轮「论文—代码—实验结果」一致性审计，现在要你像一个带他复现的师兄那样回答：是什么、为什么、接下来怎么动。

【总纲】凡是与本材料相关的问题，你都要**同时**用两样东西回答：① 本材料与本轮审计结论；② 你自己的领域知识。只念材料就是把助手做成检索引擎（会被回一句「没覆盖」），只凭自己知识就是脱离这份材料空谈。但材料是**边界**：自身知识只能用来补背景、作参照，不能拿它改写或覆盖材料结论。

【怎么组织答案】按问题需要取用，不必凑满四段：
· 结论 —— 第一句直接给判断，不要铺垫。
· 依据 —— 引用【本次审计数据】里的原值与出处（论文页码 / 代码文件:行号 / 参数名）。
· 为什么 —— 你对成因的判断：字段错位、单位换算、默认值没写、版本差异、论文笔误、
  代码改写、度量口径不同、数据划分不同…。材料不足以下定论时，列出最可能的两种到三种，
  并说清各自怎么快速证伪（看哪一页、跑哪一行、该问作者什么）。
· 怎么改 —— 落到具体对象的动作，按「先改什么」排序。写「把 train.py:11 的 lr 从 0.001
  改成 1e-4（论文 Page 3）」这种能直接动手的；不写「检查一下超参」这种空话。

【每一句都要能说清来源】四种来源，写法必须不同：
· 材料结论 —— 出自【本次审计数据】的原值和判定，必须带出处（论文页码 / 文件:行号）。
· 我的推断 —— 以材料为依据的判断，用「最可能 / 我判断 / 一种可能是」开头。
· 通用知识 —— 材料里没有、靠你自身知识补的背景（领域常识、常见做法、这个方法一般怎么调、
  同类工作的通常取值），必须单独成句并冠以「〔通用知识〕」，让它一眼能看出不是本材料结论。
  例：「〔通用知识〕CIFAR-10 上 ResNet 的初始学习率常见量级在 1e-1~1e-2，但这份论文写的是
  1e-4，以论文为准。」
· 两者都对不上 —— 直接说没有，并说清补什么材料才能判断。

【硬性要求】
1. 材料数字与出处只能用【本次审计数据】里出现过的原值：不许估、不许编。
   需要算倍数或差值时可以算，但要把算式写出来（例：0.001 ÷ 1e-4 = 10 倍）。
   通用知识的数值必须写在「〔通用知识〕」那一句里，不许混进材料结论的句子里当事实。
2. 允许你做成因解释、给修复方案 —— 但必须让用户分得清哪句是材料原文、哪句是你的判断：
   引原文时写出处；写推断时用「最可能 / 我判断 / 一种可能是」，不要写成既成事实。
3. 「没有」只允许用在两种情况：① 数据里确实没有这一项，且你也推不出大概、领域常识也给不出
   可靠参照（要说清这次没覆盖到什么、补什么才能判断）；② 问题本身不成立。
   除此之外不许拿「没覆盖」回避：能从材料推出大概的给推断，属于领域常识的标「〔通用知识〕」，
   两者都给的，先说材料再补常识。
4. 不要改变任何判定结论（对得上/对不上是审计算的），但可以解释它为什么这么判。
5. 不要输出思考过程：不写 Thinking Process / Analyze the Request / 英文分析，不复述这段
   提示词，第一个字就是答案。
6. 篇幅：结论一句话；整体一般 300–700 字，确有必要可到 900 字。引用原文每处不超过 30 字，
   够定位就行（不要把整段摘要抄进来）；不注水、不重复前面说过的内容。
7. **材料是边界**：凡是超出材料的内容，一律只作背景并标「〔通用知识〕」；不得用它反驳、
   修正或替代审计结论。两边冲突时以材料结论为准，并指出冲突点（哪一条对表、哪一页）。"""

#: 模型把"思考过程"写进正文的痕迹（书生 intern-s2 实测会这么做）
_THINK = re.compile(r"(Thinking Process|Analyze the Request|Analysis\s*:|思考过程|推理过程)", re.I)
#: 只有这种**强锚点**才值得信任（「回答：」太常见，思考过程里自己也会写，会切错）
_ANSWER_MARK = re.compile(r"(最终回答|最终答案|最终结论)\s*[:：]")
#: 自我检查 / 复述提示词的痕迹 —— 有这些说明切在了草稿中段，宁可丢掉
_META = re.compile(r"(Constraint|Char count|Check |Draft|I should|Let me|Wait,|"
                   r"提示词|硬性要求|自我检查|约束)", re.I)


def strip_thinking(text: str) -> tuple[str, bool]:
    """识别「把推理写进正文」的模型回复：能救就救，救不了就交给证据检索。

    返回 `(给用户看的正文, 是否检出思考)`。

    为什么要有这一步（2026-10-30 实测）：书生 intern-s2 会把推理**整段写进 content**
    （"Thinking Process: Analyze the Request: …"）。原来 `max_tokens=700` 时，
    这 700 个 token 全被英文思考吃掉 —— 用户拿到的「回答」里**一句中文答案都没有**，
    只有模型的草稿。这比"答错"更糟：看着像有内容，实际什么都没答。

    两档判据（**必须先能证明是答案，才敢给用户看**）：
      · 有「最终回答／最终答案／最终结论」这种强锚点，且锚点之后中文够多、
        不含自我检查的痕迹 → 用锚点之后那一段；
      · 其余一律返回空串 ⇒ 调用方退回「证据检索」，并如实说明"模型只给了推理"。
    实测（`.trash_tmp/internlm_raw.txt`）：这类回复里草稿、字数自检、提示词复述混在一起，
    任何"猜着切一刀"都可能把答案切掉一半 —— 宁可不给话，也不给半截草稿。
    """
    t = (text or "").strip()
    if not t or not _THINK.search(t):
        return t, False

    last = None
    for m in _ANSWER_MARK.finditer(t):
        last = m
    if last is not None:
        body = t[last.start():].strip()
        if (len(re.findall(r"[\u4e00-\u9fff]", body)) >= 60
                and not _META.search(body)):
            return body, True
    return "", True


#: 「模型自带知识」的标注（2026-10-30 用户定版：材料是边界，超出部分必须标来源）
GENERAL_MARK = "〔通用知识〕"


def split_general(text: str) -> tuple[str, str]:
    """把一段回答切成 `(材料/推断部分, 通用知识部分)`。

    为什么需要它：放开「可以补通用知识」之后，「回答里每个数字都要在事实里找到出处」
    这条硬校验会**误伤** —— 通用知识里的数字（如「这类任务常用 1e-3」）本来就不在本材料里。
    但也不能因此整体放行（那就等于放开编数）。所以按句切：标了〔通用知识〕的句子单独
    拿出来，**剩下的部分仍然要过「数字必须有出处」**。

    测试（`verify_qa_live.py` 的数字校验）和后续界面都调这一份，别各写一套切法。
    """
    t = text or ""
    if GENERAL_MARK not in t:
        return t, ""
    keep: list[str] = []
    moved: list[str] = []
    for part in re.split(r"(?<=[。；！？\n])", t):
        (moved if GENERAL_MARK in part else keep).append(part)
    return "".join(keep), "".join(moved)


def build_prompt(audit, question: str, history: list | None = None,
                 meta: dict | None = None, report: str = "") -> str:
    """把「事实 + 历史 + 这次的问题」拼成一次调用。**多轮靠把历史写进提示词实现**。"""
    ctx = facts(audit, meta, report)
    extra = focus(audit, question)
    if extra:
        ctx += "\n\n" + extra
    if len(ctx) > MAX_CTX:
        ctx = ctx[:MAX_CTX] + "\n…（数据过长已截断，只保留了前面这些）"

    lines = []
    for turn in (history or [])[-MAX_TURNS:]:
        role = "用户" if turn.get("role") == "user" else "助手"
        lines.append(f"{role}：{_s(turn.get('content'), 400)}")
    hist = "\n".join(lines) if lines else "（这是第一轮提问）"

    return (f"{_SYS}\n\n【本次审计数据】\n{ctx}\n\n"
            f"【之前的对话】\n{hist}\n\n"
            f"【用户这次的问题】\n{question}")


#: 没有审计结论时的固定答复（**不调模型**，见 answer() 里的说明）
_NO_AUDIT = ("还没有审计结论，这个问题我答不了 —— 我只能依据这一轮算出来的东西回答。\n\n"
             "**怎么开始**：① 上方放材料（不传也行，会用内置样例）→ ② 点「开始科研审计」"
             "→ ③ 跑完再回来问。\n\n"
             "为什么不再「凭常识」答：那样给出的页码和文件行号都是编的，比不答更糟。")


def _log_qa_error(ex: BaseException, base_url: str, model: str, qlen: int) -> None:
    """把一次失败的**原始凭据**追加进 `cache/qa_errors.log`。

    为什么必须落盘（2026-10-31）：用户两天里说了三遍「云端明明有钱，怎么会不行」，
    而用同一个 key 在命令行实测全通 —— 界面只显示一行异常类名，等于没有证据，
    双方都只能靠猜。把状态码、响应体、当时的地址/模型/问题长度留下来，
    下次直接在文件里看：429 是限流、401 是 key 失效、400 是参数/上下文超限。

    **绝不抛异常**：日志写不进去是小事，不能让它影响回答。
    """
    try:
        import datetime as _dt
        from pathlib import Path

        cache = Path(__file__).resolve().parent.parent / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        _code = getattr(ex, "status_code", "") or ""
        _body = ""
        _resp = getattr(ex, "response", None)
        if _resp is not None:
            try:
                _body = (_resp.text or "")[:600]
            except Exception:          # noqa: BLE001
                _body = ""
        _line = (f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] {type(ex).__name__} "
                 f"status={_code} url={base_url} model={model} chars={qlen}\n"
                 f"  msg : {str(ex)[:600]}\n"
                 + (f"  body: {_body}\n" if _body else ""))
        with open(cache / "qa_errors.log", "a", encoding="utf-8") as _f:
            _f.write(_line + "\n")
    except Exception:                  # noqa: BLE001
        pass


def answer(audit, question: str, history: list | None = None, *,
           api_key: str = "", base_url: str = "", model: str = "",
           meta: dict | None = None, report: str = "") -> tuple:
    """回答一轮，返回 `(正文, 说明)`。

    `说明` 必须如实，界面会原样显示：是模型真答的、还是没 key 没答。
    **绝不抛异常** —— 问答坏了不能影响审计页。
    """
    question = (question or "").strip()
    if not question:
        return "", "空问题，未调用"
    if audit is None:
        # ⚠ 2026-10-30 实测：没有结论却配了 key 时，模型会**编出看起来很像的出处**
        #   （"论文 3.2 节 / 代码 train.py:L45" —— 那份材料根本没读过）。
        #   所以这里直接不调模型：省一次钱，更重要的是不给用户假出处。
        #   要问「平台怎么用」也不吃亏 —— 固定答复里本来就写了三步。
        return _NO_AUDIT, "未跑审计：不调用模型（避免编造出处）"
    if api_key:
        # 一次调用的最小封装：预算随端点自动定（远程 1800/75s，本机 1000/360s）。
        # ⚠ 预算**不许在调用点写死** —— 2026-10-30 踩过两次：
        #   · 700 token 时，会写思考过程的模型（书生 intern-s2）把额度全花在英文推理上，
        #     正文里一句中文答案都没有；
        #   · 1800 token 配 30s 超时，40s 以上的回答必然被判失败。
        #   现在统一由 `llm.qa_budget()` 按端点定 —— 一处真相，调用方只取用。
        def _once(_k: str, _u: str, _m: str) -> str:
            _mt, _to = qa_budget(_u)
            return _call_deepseek(
                build_prompt(audit, question, history, meta, report),
                _k, max_tokens=_mt, timeout=_to, base_url=_u, model=_m)

        # 原 `_used_m`（实际作答模型名）已随说明行里的模型名一起删掉（2026-10-31）：
        # 既然不在界面上显示，就不必再维护一份"到底是谁答的"的副本 ——
        # 真正要查时看 `cache/qa_errors.log` 与调用方传进来的 model 即可。
        _fb = ""                     # 附加说明（例如"限流后重试成功"）
        _fail = None
        _tries = 0                   # 因为限流重试了几次
        # ---- 云端限流（429）自动重试（2026-10-30 晚用户口径修正）--------------
        # 用户原话：「我就是故意用 dsh 的密钥进行提问，不是本地」。
        #   ⇒ **绝不许偷偷换成本机模型**：那等于把他特意选的云端降级成 8B 小模型，
        #     还慢一百秒，比报错更糟。之前写过一版"自动改打本机"，已撤回。
        # 该做的其实是另一件事：429 多数是**瞬时**的（同一分钟请求密度高，
        #   比如审计刚跑完几十次模型调用、紧接着就问），等几秒自己就好了。
        #   所以这里退避重试两次（3s、6s），仍不行才如实报错。
        # 什么时候不重试：
        #   · 本来就在用本机 —— 本机一次生成约 100s，重试只是再等一遍，白耗时间；
        #   · 不是限流类错误（例如 prompt 超上下文那种 400）—— 原样重发结果一样；
        #   · 对端根本不是 API（2026-10-31 实锤：Base URL 填成 platform.deepseek.com
        #     这种网页控制台时，WAF 回 HTML + 429，SDK 包成 RateLimitError ——
        #     重试一万次也通不了，白等 9 秒，还会被误诊成「云端限流」）。
        while True:
            try:
                raw = _once(api_key, base_url, model)
                break
            except RunCancelled:
                # 2026-10-30 晚：「点击即终止上一次」——被新的一次点击作废**不是失败**，
                #   不能显示"模型调用失败"，否则用户会以为程序坏了。原样抛给 UI 静默处理。
                raise
            except Exception as ex:          # noqa: BLE001
                if (_tries < len(_RETRY_WAITS) and not is_local_endpoint(base_url)
                        and _is_throttle(ex) and not looks_non_api_reply(ex)
                        and not endpoint_url_hint(base_url)):
                    _wait = _RETRY_WAITS[_tries]
                    _tries += 1
                    time.sleep(_wait)
                    # 睡醒先确认这一轮还算不算数：等待期间用户可能又点了一次发送，
                    # 那这一轮就该作废，不能拿旧问题继续跑。
                    check_run()
                    continue
                _fail = ex
                break
        if _fail is None and _tries:
            _fb = f"（云端限流，自动重试 {_tries} 次后成功）"

        if _fail is not None:
            # 先留凭据再讲人话：文字是给人看的，日志是给下次排查看的。
            _log_qa_error(_fail, base_url, model, len(question))
            # 只写异常类名对用户没用（「BadRequestError」看不出该干什么）。
            # 把最常见的两类挑出来直说（2026-10-30 晚补 429：用户截图里只有
            #   「模型调用失败（RateLimitError）」，等于什么都没说）：
            #   · 账号额度/配额不足 → 充值或换 key；
            #   · 429 请求太多 → 已经自动重试过了，告诉他等多久。
            _msg = str(_fail)
            _low = _msg.lower()
            # 先排「地址根本不是 API」再谈限流/额度（2026-10-31：这两类的提示对它是误诊）。
            _url_hint = endpoint_url_hint(base_url)
            if _url_hint:
                _hint = f" —— {_url_hint}，改对 Base URL 再试（这不是限流，重试无用）"
            elif looks_non_api_reply(_fail):
                _hint = (" —— 对端回的不是 API 应答而是一张网页，"
                         "Base URL 很可能填成了控制台/门户地址，核对后再试")
            elif ("配额" in _msg or "quota" in _low or "-20080" in _msg
                    or "insufficient" in _low or "balance" in _low):
                _hint = " —— 账号额度/配额不足，去后台充值或换一个 key"
            elif ("ratelimit" in _low or "rate limit" in _low or "429" in _msg
                  or "too many requests" in _low):
                _hint = (" —— 云端在限流（请求太密集）"
                         + (f"，已自动重试 {_tries} 次仍未通过" if _tries else "")
                         + "；过半分钟再问一次就行，钱和 key 都没问题")
            else:
                _hint = ""
            text, note = "", (f"模型调用失败（{type(_fail).__name__}）{_hint}，"
                              "下面是直接检索到的审计事实")
        else:
            # 2026-10-30 晚：生成完了**再确认一次**这一轮还算不算数。
            # 为什么必须再查：本地一次问答要 100 秒，用户中途又点了一次发送，
            # 旧的那次这时才刚生成完 —— 不查就会把**已作废**的答案当成新答案写进对话。
            check_run()
            text, cut = strip_thinking(raw)
            if text:
                # ⚠ 说明行里**不能出现 GENERAL_MARK 的原文**（要写「通用知识」不加方括号）：
                #   这一行会被一起抓进弹窗文本，若含「〔通用知识〕」，`split_general()`
                #   就会把它当成模型的通用知识段 —— 那么"有没有标来源"这项检查永远为真，
                #   成了假绿（2026-10-30 自查发现）。
                # 2026-10-31 用户要求：说明行里的**模型名**去掉（「模型作答 · deepseek-chat」
                #   → 「模型作答」）。理由：模型名在「高级设置」三个框里已有，这里再挂一遍
                #   既重复又容易让人误以为是"另一条通道"。
                # ⚠ 「模型作答」这四个字**不能删**：`probes/shot_qa.py`、`probes/qa_check.py`、
                #   `tests/verify_qa_live.py` 都靠它当"这一问真的答出来了"的判据。
                #   删了模型名只是少一段尾巴，判据必须原样留着。
                return text, ("模型作答"
                              "（带出处＝材料原文｜「最可能 / 我判断」＝模型推断｜"
                              "标「通用知识」的是模型自带知识，不是本材料结论）"
                              + ("（已隐去思考过程）" if cut else "")
                              + _fb)
            # 走到这里 = 模型把推理写进了正文、没给出可用的答案。
            # 口径：不给半截草稿，退回证据检索并说清为什么（顺手告诉他换哪个模型能好）
            note = ("该模型把推理过程写进了回答（已隐去，不给半截草稿），"
                    "下面是直接检索到的审计事实 —— 换成直接作答的模型"
                    "（如 intern-s1 / deepseek-chat）就能拿到自然语言答案"
                    ) if raw else "模型返回为空，下面是直接检索到的审计事实"
    else:
        note = "未配置模型 Key：下面是直接检索到的审计事实（不是生成的话）"

    # 没模型 / 失败：不编答案，把命中的事实按条摊开
    hits = top_hits(audit, question)
    # ⚠ 2026-10-27：只列 top_hits 会答错一类问题 —— 问「这次有没有漏查什么」，
    #   没有哪条 finding 能"命中"，于是回一句「本轮没有与这一问直接相关的条目」，
    #   而覆盖情况（cov）里其实有数据。补上 focus() 的原文/记录那一块。
    extra = focus(audit, question)
    parts: list[str] = []
    if hits:
        parts.append("**这一问相关的审计事实（原样列出，未做改写）**\n\n"
                     + "\n".join(f"- {h}" for h in hits))
    if extra:
        parts.append("**这一问相关的原文与记录（同样原样列出，未做改写）**\n\n" + extra)
    if parts:
        body = "\n\n".join(parts)
    elif audit is not None:
        body = "本轮审计里没有与这一问直接相关的条目 —— 不是我藏起来了，是这次确实没查到这一块。"
    else:
        body = "还没有审计结论。先点「开始科研审计」跑一轮，再回来追问就有依据了。"
    return body, note
