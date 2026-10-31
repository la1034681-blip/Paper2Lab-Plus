"""审计报告生成——Markdown 模板(浏览器打印即 PDF)。

报告结构(含交叉审查章节):
就绪度 -> 解析覆盖与可信度 -> **① 假设站** -> 审计概览 -> 逐项详情 -> 解析拒收清单
-> 审计自检(证据自证) -> 根因追查/补丁草案 -> 待确认问题 -> 附录(比较日志)

「① 假设站」排在概览**之前**是刻意的: 四站叙事是「① 假设 → ③ 判定」,
报告里也该先给"我们要验什么", 再给"验出来是什么"。它由 `_hypotheses_md` 生成,
字段口径与界面同源（都只读 Hypothesis.to_dict()）。
"""

from __future__ import annotations

from datetime import datetime

from .models import AuditResult, Finding, Status
from .synonyms import PARAM_SYNONYMS

_DISPLAY = {k: v[0] for k, v in PARAM_SYNONYMS.items()}

_STATUS_ICON = {
    Status.CONSISTENT: "🟢",
    Status.INCONSISTENT: "🔴",
    Status.MISSING_IN_CODE: "🟡",
    Status.MISSING_IN_PAPER: "🟡",
    Status.NOT_FIXED: "🟠",
    Status.UNVERIFIABLE: "🔵",
    Status.INTERNAL_INCONSISTENT: "🟣",
}
_STATUS_ZH = {
    Status.CONSISTENT: "一致",
    Status.INCONSISTENT: "不一致",
    Status.MISSING_IN_CODE: "代码未找到",
    Status.MISSING_IN_PAPER: "论文未声明",
    Status.NOT_FIXED: "未固定",
    Status.UNVERIFIABLE: "动态确定·无法静态验证",
    Status.INTERNAL_INCONSISTENT: "内部不一致",
}
_CONF_ZH = {"confirmed": "🟢 Confirmed(规则验证)", "inferred": "🔵 Inferred(AI 推断)"}
_RISK_ZH = {"high": "高", "medium": "中", "low": "低", "none": "—"}

# 状态释义: 让读者不用查文档就知道这条结论意味着什么
_STATUS_MEANING = {
    Status.CONSISTENT: "论文声明值与代码实现值经归一化后相等, 规则验证通过。",
    Status.INCONSISTENT: "论文声明值与代码实现值经归一化后不相等——这是本次审计的核心风险, "
                         "通常会导致实验无法复现出论文指标。",
    Status.MISSING_IN_CODE: "论文明确声明了该参数, 但静态扫描未在代码中找到对应实现"
                            "(既非命名赋值也非内联判定)。可能由框架默认值/外部配置注入, 或确实被遗漏。",
    Status.MISSING_IN_PAPER: "代码中存在该配置, 但论文未声明——表现为实验记录不完整, "
                             "他人复现时无法得知该取值依据。",
    Status.NOT_FIXED: "论文要求固定该随机性来源, 但代码中未找到固定动作, 实验结果不可稳定复现。",
    Status.UNVERIFIABLE: "代码中的取值由运行时变量/外部输入决定, 无法静态验证; "
                         "系统如实标注而非猜测, 需查运行配置确认。",
    Status.INTERNAL_INCONSISTENT: "同一载体(论文或代码)内部出现多个不同取值, 存在自相矛盾; "
                                  "若属灵敏度分析/分组实验可忽略。",
}

# 结果类结论的释义(对照侧是"用户上传的实测结果", 不是代码实现)
_RESULT_MEANING = {
    Status.CONSISTENT: "论文声明的实验结果与上传的实测结果一致(相对差异 ≤1%), 复现结果可信。",
    Status.INCONSISTENT: "论文声明的实验结果与上传的实测结果存在明显差异——"
                         "说明实际复现未能达到论文指标, 需结合参数差异定位原因。",
}
_CHECK_HINT = {
    Status.CONSISTENT: "无需人工干预; 如需引用可将本项作为「已核对」证据。",
    Status.INCONSISTENT: "建议人工确认以哪一侧为准: 先看代码是否为论文对应的实验配置, "
                         "再决定修改代码还是修正论文描述。",
    Status.MISSING_IN_CODE: "建议在训练入口显式声明该参数并与论文对齐; "
                            "若确认由框架默认值提供, 请在论文/README 中注明。",
    Status.MISSING_IN_PAPER: "建议在论文或 README 中补充该配置, 便于他人复现。",
    Status.NOT_FIXED: "建议在训练入口显式固定随机种子(如 torch.manual_seed / random_state)。",
    Status.UNVERIFIABLE: "建议核对实际运行配置(命令行参数/环境变量/日志)后再判断是否一致。",
    Status.INTERNAL_INCONSISTENT: "建议确认哪一处是实际生效的配置; 若为多组实验请分别标注。",
}


def _finding_md(f: Finding) -> str:
    _is_result = f.param_key.startswith("result_")
    _meaning = (_RESULT_MEANING if _is_result else _STATUS_MEANING).get(f.status, "")
    lines = [f"### {_STATUS_ICON[f.status]} {f.display_name} — {_STATUS_ZH[f.status]}",
             "",
             f"**结论释义**: {_meaning}",
             "",
             f"**风险等级**: {_RISK_ZH[f.risk.value]}　"
             f"**置信**: {_CONF_ZH[f.confidence.value]}　"
             f"**证据自证**: {'✅ 通过' if f.verified else '❌ 未通过'}",
             ""]
    if f.risk_reason:
        lines += [f"**风险判定依据**: {f.risk_reason}", ""]
    lines += ["#### 证据链",
              ""]
    if f.paper:
        lines += [
            f"- **论文侧证据**: `{f.paper.location}`",
            f"  > {f.paper.snippet}",
            f"  - 声明值: `{f.paper.raw_value}`"
            + (f"　归一化后: `{f.paper.normalized}`" if f.paper.normalized is not None else ""),
        ]
    else:
        lines.append("- **论文侧证据**: 未声明(论文中未找到该参数的数值)")
    if f.code:
        if _is_result:
            lines += [
                f"- **对照侧证据(用户上传实测结果)**: `{f.code.location}`",
                f"  > {f.code.snippet}",
                f"  - 实测值: `{f.code.raw_value}`"
                + (f"　归一化后: `{f.code.normalized}`" if f.code.normalized is not None else ""),
            ]
        else:
            lines += [
                f"- **代码侧证据**: `{f.code.location}`",
                f"  > {f.code.snippet}",
                f"  - 实现值: `{f.code.raw_value}`"
                + (f"　归一化后: `{f.code.normalized}`" if f.code.normalized is not None else ""),
            ]
    else:
        lines.append("- **代码侧证据**: 未找到(静态扫描未命中)")
    for i, ev in enumerate(f.extra, 1):
        lines.append(f"- **附加证据 {i}**({ev.source}): `{ev.location}`"
                     f"　> {ev.snippet}　- 值: `{ev.raw_value}`")
    lines.append("")
    lines.append("#### 比对过程")
    lines.append("")
    if _is_result and f.paper and f.code:
        try:
            pv, av = float(f.paper.normalized), float(f.code.normalized)
            _d = abs(av - pv)
            _rel = _d / max(abs(pv), 1e-12)
            lines.append(f"- 结果核对: 论文声明 `{pv}` vs 实测 `{av}` → 绝对差异 `{_d:.4g}`、"
                         f"相对差异 `{_rel:.2%}`"
                         + ("（≤1% 视为一致）" if f.status == Status.CONSISTENT
                            else "（>1% 判定不一致）"))
        except (TypeError, ValueError):
            lines.append(f"- 结果核对: 论文声明 `{f.paper.raw_value}` "
                         f"vs 实测 `{f.code.raw_value}`")
    elif f.paper and f.code and f.status == Status.INCONSISTENT:
        lines.append(f"- 归一化对比: `{f.paper.normalized}` ≠ `{f.code.normalized}` "
                     f"→ 判定 **不一致**(差异见上表原文, 可直接复算)")
    elif f.paper and f.code and f.status == Status.CONSISTENT:
        lines.append(f"- 归一化对比: `{f.paper.normalized}` = `{f.code.normalized}` "
                     "→ 判定 **一致**(规则验证通过)")
    elif f.status == Status.UNVERIFIABLE:
        lines.append("- 代码侧取值由运行时变量决定, 静态分析无法给出确定值 → 标记为"
                     " **动态确定·无法静态验证**(不计入评分)")
    elif f.status == Status.INTERNAL_INCONSISTENT:
        lines.append("- 同一套实验设置里给了两个不同取值 → 标记为 **内部不一致**"
                     "(论文须同页同章节; 代码须可能同时生效。按任务分别取值/分目录的各套配置不在此列)")
    elif f.status == Status.MISSING_IN_CODE:
        lines.append("- 论文侧有值、代码侧无证据 → 标记为 **代码未找到**(中风险, 需人工确认)")
    elif f.status == Status.MISSING_IN_PAPER:
        lines.append("- 代码侧有值、论文侧无声明 → 标记为 **论文未声明**(实验记录完整度扣分)")
    elif f.status == Status.NOT_FIXED:
        lines.append("- 论文要求固定随机性、代码无固定动作 → 标记为 **未固定**")
    lines.append("")
    lines.append("#### 证据自证与人工核查")
    lines.append("")
    if f.verified:
        vfy_txt = "✅ 通过——原文片段中确实包含所声明的值, 且一致性结论可由归一化比较复现"
    else:
        vfy_txt = f"❌ 未通过——{f.verify_note}"
    lines.append(f"- 证据自证: {vfy_txt}")
    lines.append(f"- **人工核查建议**: {_CHECK_HINT.get(f.status, '')}")
    if f.note:
        lines.append(f"- 系统说明: {f.note}")
    parsers = {ev.parser for ev in ([f.paper, f.code] + list(f.extra)) if ev}
    if parsers - {"ast"}:
        zh = {"ast": "Python AST 精确解析", "regex": "多语言通用通道(正则)",
              "table": "PDF 表格通道", "ocr": "图像 OCR", "vlm": "视觉模型读图",
              "text": "PDF 正文文本", "result": "用户上传实测结果"}
        lines.append("- 抽取通道: " + "、".join(zh.get(p, p) for p in sorted(parsers)))
    if f.ai_analysis:
        lines += ["", f"**AI 归因分析**: {f.ai_analysis}"]
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _hypotheses_md(audit: AuditResult) -> list[str]:
    """① 假设站的章节（阶段二 2A 提出 ＋ 2B 推演筛排）。

    为什么报告必须有这一节: 报告是**能带走的那一份**。界面上看得到假设卡、
    导出的 .md 里没有, 等于"从查错转向发现"这个动作只活在网页上 ——
    评审把报告打开时, 看到的还是旧的"逐项比对"体系。

    口径与界面**同源**: 字段只读 `Hypothesis.to_dict()`（那里定义过一次）,
    本函数不另写一套说法, 否则界面与报告迟早对不上。推演那一列同理 ——
    判决与文案都来自卡片字段, 报告不做第二次判断。
    """
    cards = list(getattr(audit, "hypotheses", None) or [])
    total = int(getattr(audit, "hypothesis_total", 0) or len(cards))
    stats = dict(getattr(audit, "hypothesis_stats", None) or {})
    head = f"## ① 假设站: {total} 条可证伪的待验假设"
    if stats.get("total"):
        bits = []
        if stats.get("flagged"):
            bits.append(f"{stats['flagged']} 条已定位为取值非法")
        if stats.get("excluded"):
            bits.append(f"{stats['excluded']} 条推演已排除")
        bits.append(f"{stats['pending']} 条待验")
        head += "　—　" + "、".join(bits)
    out: list[str] = [
        "---", "", head, "",
        "> 这一节**不是结论**, 是研究的起点。逐项审计回答「两侧一不一样」; "
        "这里回答「因此我们假设什么、以及看到什么就推翻它」。"
        "**写不出推翻判据的话不会出现在这里** —— 不可证伪的不是假设, 是意见。", "",
    ]
    if stats.get("total"):
        out += [f"> 推演引擎: {stats.get('engine', '—')}。动手之前先算一遍 —— "
                "能算掉的先剔（推演已排除）、一侧写错的先标（已定位），省下实验的力气。", ""]
    if not cards:
        out += ["本次没有提出假设: 已抽取到的参数在论文与代码两侧没有值得追问的分歧。",
                "（这不等于「全部通过」—— 判定见下面的审计概览。）", ""]
        return out
    if total > len(cards):
        out += [f"共提出 {total} 条, 按「最该先验」排序后列出前 {len(cards)} 条。", ""]

    rows = [h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in cards]
    out += ["| # | 参数 | 类型 | 推演判决 | 推演类型 | 需实机验证 | 排序分 |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for d in rows:
        out.append(f"| {d.get('rank')} | {d.get('key')} | {d.get('kind')} | "
                   f"{d.get('verdict') or '—'} | {d.get('derivation_kind') or '—'} | "
                   f"{'是' if d.get('needs_rerun') else '否'} | {d.get('score')} |")
    out.append("")

    for d in rows:
        need = "必须实机验证" if d.get("needs_rerun") else "读代码即可定论"
        out += [f"### #{d.get('rank')}　{d.get('key')} — {d.get('kind')}（{need}）", "",
                f"- **待验陈述**: {d.get('claim')}",
                f"- **要回答的问题**: {d.get('question')}",
                f"- **怎么推翻它**: {d.get('falsifiable_by')}"]
        sides: list[str] = []
        if d.get("paper_value"):
            sides.append(f"论文 `{d['paper_value']}`"
                         + (f"（{d['paper_where']}）" if d.get("paper_where") else ""))
        if d.get("code_value"):
            sides.append(f"代码 `{d['code_value']}`"
                         + (f"（{d['code_where']}）" if d.get("code_where") else ""))
        if sides:
            out.append("- **两侧取值**: " + "　｜　".join(sides))
        if d.get("kind_desc"):
            out.append(f"- **类型说明**: {d['kind_desc']}")
        if d.get("derivation"):
            kind = d.get("derivation_kind") or "推演"
            out.append(f"- **推演判决（{d.get('verdict')} · {kind}）**: {d['derivation']}")
        if d.get("why"):
            out.append(f"- **为什么排在这个位置**: {d['why']}")
        out.append("")
    return out


def _judge_md(audit: AuditResult) -> list[str]:
    """模型先判、程序后验的**分歧留档**（2026-10-25）。

    只在至少有一条模型判断时出现 —— 关着时不占版面（与 `judge_llm` 默认关同款口径）。

    口径（必须写进报告, 这是评审最会追问的地方）:
      · **结论以程序为准**: 规则判定可重算, 换个模型、换台机器都是同一个答案;
      · **模型的话如实并存**: 不一致的条目两边都写, 不做静默取舍;
      · 规则没算出结论、只有模型有说法的, 只能当**线索**, 不进结论段。
    """
    cards = list(getattr(audit, "hypotheses", None) or [])
    finds = list(getattr(audit, "findings", None) or [])
    h_rows = []
    for c in cards:
        d = c.to_dict() if hasattr(c, "to_dict") else dict(c)
        if d.get("llm_verdict"):
            h_rows.append(d)
    f_rows = [f for f in finds if getattr(f, "llm_status", None) is not None]
    if not h_rows and not f_rows:
        return []

    def _agree(v) -> str:
        return "一致" if v is True else ("不一致" if v is False else "程序没结论")

    out: list[str] = [
        "---", "", "## 模型判断与程序复验", "",
        "> 这一节回答「模型怎么看、程序怎么算、两边差在哪」。"
        "**结论以程序为准** —— 规则判定可重算；模型的话如实并存，不做静默取舍。"
        "程序没算出结论、只有模型有说法的，只能当**线索**，不进结论。", "",
    ]

    if h_rows:
        n_dis = sum(1 for d in h_rows if d.get("llm_agree") is False)
        out += [f"### 假设真假: 模型判了 {len(h_rows)} 条, 与程序不一致 {n_dis} 条", "",
                "| 参数 | 模型说 | 程序算 | 比对 | 模型的理由 |",
                "| --- | --- | --- | --- | --- |"]
        for d in h_rows:
            out.append(f"| {d.get('key')} | {d.get('llm_verdict')} | "
                       f"{d.get('verdict') or '—'} | {_agree(d.get('llm_agree'))} | "
                       f"{d.get('llm_why') or '—'} |")
        out.append("")

    if f_rows:
        n_dis = sum(1 for f in f_rows if f.llm_agree is False)
        out += [f"### 参数配对: 模型判了 {len(f_rows)} 条, 与程序不一致 {n_dis} 条", "",
                "| 参数 | 模型说 | 程序算 | 比对 | 模型的理由 |",
                "| --- | --- | --- | --- | --- |"]
        for f in f_rows:
            out.append(f"| {f.display_name} | {_STATUS_ZH[f.llm_status]} | "
                       f"{_STATUS_ZH[f.status]} | {_agree(f.llm_agree)} | "
                       f"{f.llm_why or '—'} |")
        out.append("")

    out += ["一条纪律: 两个说法**都留着** —— 只留一个，等于替读者做了他没授权的取舍。", ""]
    return out


def _impact_groups(audit: AuditResult) -> list:
    """把「同一件事」的影响合并 —— 批大小卡与学习率卡算出来的是同一条。

    为什么必须合并: 两张卡说的是同一个倍数（批大小变了 lr 该不该跟着变）, 分开列
    读者会以为系统把一件事说了两遍, 或者以为漏了一张卡。合并时把两张卡的名字都写上。
    """
    groups: dict = {}
    for h in (getattr(audit, "hypotheses", None) or []):
        im = getattr(h, "impact", None) or {}
        if not im:
            continue
        sig = "|".join(str(im.get(x, "")) for x in ("kind", "compares", "magnitude", "direction"))
        g = groups.setdefault(sig, {"impact": im, "keys": []})
        key = str(getattr(h, "key", "") or "")
        if key and key not in g["keys"]:
            g["keys"].append(key)
    return list(groups.values())


_KIND_IMPACT_ZH = {
    "effective_batch": "等效批大小 → 学习率该不该跟着变",
    "warmup": "预热步数该不该跟着批大小调整",
}


def _impact_md(audit: AuditResult) -> list[str]:
    """① 假设站「算」这一步的**影响量级**章节（阶段二 2E）。

    为什么单独成节: 逐项比对只说「两边不一样」, 读者不知道「不一样有多大后果」。
    这一节回答五件事: 折成倍数是多少、方向朝哪、有几种合理解释、哪种证据最硬、
    什么情况下影响归零、复现时该拿什么去对照。

    三条边界写进正文（不只写在文档里）: 提示层 / 给区间不给点值 / 「证据最硬」不是「最优」。
    """
    groups = _impact_groups(audit)
    if not groups:
        return []
    out: list[str] = [
        "---", "",
        f"## 影响量级: {len(groups)} 条可核算的影响（发现型假设）", "",
        "> 这一节回答「找出来的差异**有多大后果**」。三条边界先写清楚，免得被当成结论：",
        "> ① **提示层** —— 不进判定、不进评分、不改就绪度，推错也只是多一句话；",
        "> ② **给区间不给点值** —— 批大小放大后学习率该放大几倍，业内有两派都算主流做法，",
        ">    只报一个数等于替读者选边；",
        "> ③ **「证据最硬」不是「最优」** —— 优不优要跑训练比准确率，本项目不跑训练；",
        ">    这里排的是「哪个解释证据更硬」，四项加减分都能当场重算。", "",
    ]

    for i, g in enumerate(groups, start=1):
        im = g["impact"]
        kind_zh = _KIND_IMPACT_ZH.get(str(im.get("kind", "")), str(im.get("kind", "")))
        out += [f"### 影响 {i}　{'、'.join(g['keys'])} — {kind_zh}", ""]
        out.append(f"- **比的是什么**: {im.get('compares') or '—'}"
                   f"（倍数 `{im.get('magnitude')}`）")
        out.append(f"- **方向**: {im.get('direction') or '不标方向（落在两派区间内，没有可指的方向）'}")
        out.append(f"- **依据的规则**: {im.get('rule') or '—'}")
        if im.get("formulas"):
            out.append("- **算式（可手工复算）**:")
            out += [f"  - `{f}`" for f in im["formulas"]]
        if im.get("given"):
            out.append("- **取数口径**:")
            for v in im["given"]:
                note = f"（{v['note']}）" if v.get("note") else ""
                out.append(f"  - {v['side']} `{v['key']} = {v['value']}`{note}")
        if im.get("note"):
            out.append(f"- **必须一起看的不确定性**: {im['note']}")
        if im.get("sources"):
            out += [f"- **出处**: {s}" for s in im["sources"]]

        cands = list(im.get("candidates") or [])
        if cands:
            out += ["", "**候选解释（按证据硬度排序，不是「最优」）**", "",
                    "| 序 | 读法 | 应有倍数（两派区间） | 冲突是否消失 | 证据分 |",
                    "| --- | --- | --- | --- | --- |"]
            for c in cands:
                out.append(f"| {c.get('rank')} | {c.get('reading')} | "
                           f"{c.get('quantity') or ''} {c.get('implied')} 倍 | "
                           f"{'是' if c.get('resolves') else '否'} | {c.get('score')} |")
            out.append("")
            for c in cands:
                if c.get("rank") != 1:
                    continue
                out.append(f"- **证据最硬的一条（第 1 顺位）**: {c.get('reading')}")
                for p in c.get("parts") or []:
                    out.append(f"  - {p['item']} {int(p['delta']):+d}：{p['why']}")
            others = [c for c in cands if c.get("rank") != 1]
            for c in others:
                out.append(f"- 第 {c.get('rank')} 顺位: {c.get('reading')}"
                           f"（{c.get('score')} 分）")
            out.append("")

        if im.get("flip"):
            out.append("**翻盘算：哪个输入一变，结论就反过来**")
            for f in im["flip"]:
                out.append(f"- {f['input']}：{f['now']} → {f['if_changed']}；{f['effect']}")
            out.append("")
        else:
            # 空着不写会让人以为「没做这一步」。四项核算是承诺, 空的也要说清为什么空。
            out += ["**翻盘算：本次没有会让结论反转的输入**"
                    "（口径无歧义，梯度累积也没有改变倍数）", ""]
        if im.get("zero_when"):
            out.append("**边界算：什么条件下影响归零**")
            out += [f"- {z}" for z in im["zero_when"]]
            out.append("")

        ag = im.get("against") or {}
        if ag.get("available"):
            where = f"，{ag['where']}" if ag.get("where") else ""
            out += ["**对照算：复现时该看到什么**",
                    f"- 对照基准: 论文报告的 {ag['metric']} = `{ag['paper_value']}`{where}",
                    f"- 预期现象: {ag['predict']}",
                    f"- 同向怎么办: {ag['if_match']}",
                    f"- 反向怎么办: {ag['if_opposite']}",
                    f"- ⚠ {ag['note']}", ""]
        else:
            out += ["**对照算: 本次做不了**", f"- {ag.get('why') or '缺可比的结果指标'}", ""]

    out += [f"> 口径提示: {groups[0]['impact'].get('scope') or ''}", ""]
    return out


def _figures_md(audit: AuditResult) -> list[str]:
    """① 假设站「读」这一步的**图表线索**章节（阶段二 2E）。

    为什么单独成节: 报告若只写"参数对不上", 读者不知道系统**看过图没有**。
    这一节如实交代两件事: ① 论文图注原文摘录（逐字, 出处是页码）;
    ② 有视觉通道时读到的图上可见信息（坐标轴 / 趋势 / 峰位 / 图例）。
    **不写结论** —— 不做化合物鉴定、不推断机理, 出现解释性表述的线索在采信前就被丢了。
    """
    clues = list(getattr(audit, "figure_clues", None) or [])
    total = int(getattr(audit, "figure_total", 0) or len(clues))
    st = getattr(audit, "figure_stats", None) or {}
    out: list[str] = [
        "---", "", f"## 图表线索: {total} 条（图注原文 ＋ 图上可见信息）", "",
        "> 这一节回答「**论文里的图到底画了什么**」: 图注逐字摘录 ＋ 图上看得见的",
        "> 坐标轴 / 趋势 / 峰位 / 图例 / 标注值。**不做鉴定、不推断机理、不写结论**;",
        "> 读不出的就说读不出, 不补数 —— 判定仍归 ③ 验证站。", "",
    ]
    if not clues:
        out += ["本次没有读到图表线索 —— 论文正文里没找到独立成行的图注"
                "（也可能本就不含图注/图）。", ""]
        for n in (st.get("notes") or []):
            out += [f"> {n}", ""]
        return out

    rows = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in clues]

    def _cell(v) -> str:
        return str(v if v is not None else "—").replace("|", "／").replace("\n", " ")

    kinds = st.get("kinds") or {}
    if kinds:
        bits = "　｜　".join(f"{k} {v}" for k, v in kinds.items())
        out += [f"**图种类分布**: {bits}", ""]
    src = []
    if st.get("captions"):
        src.append(f"图注原文 {st['captions']} 条")
    if st.get("vision"):
        src.append(f"读图 {st['vision']} 条（两次独立读取取交集）")
    if src:
        out += [f"**来源**: {'　｜　'.join(src)}　·　引擎: {st.get('engine', '—')}", ""]
    if total > len(clues):
        out += [f"共读到 {total} 条, 下面列出前 {len(clues)} 条。", ""]

    out += ["| # | 图 | 线索类型 | 内容（看得见的） | 图上依据 | 出处 |",
            "| --- | --- | --- | --- | --- | --- |"]
    for d in rows:
        out.append(f"| {_cell(d.get('rank'))} | {_cell(d.get('figure'))} | "
                   f"{_cell(d.get('clue_type'))} | {_cell(d.get('statement'))} | "
                   f"{_cell(d.get('anchor'))} | {_cell(d.get('page_ref'))} |")
    out.append("")
    for n in (st.get("notes") or []):
        out += [f"> {n}", ""]
    return out


def _chem_md(audit: AuditResult) -> list[str]:
    """① 假设站「读」这一步的**化学身份核验**章节（阶段二 2E-化学）。

    为什么单独成节: 化学论文里"分子式 ↔ 分子量"是一组**确定性**关系 —— 算得出来、
    对得上就对得上。这一节把"论文声称的化学身份"和"按原子量复算的结果"摆在一起,
    对不上的地方直接标出来（CAS 校验位、SMILES 语法坏、分子式与分子量不符）。

    **边界**: 本系统不从图像像素猜分子结构, 也不给化合物命名;
    它只核对论文**自己写的**式子 / 编号 / 质量是否自洽。
    """
    from . import chem

    items = list(getattr(audit, "chem_items", None) or [])
    total = int(getattr(audit, "chem_total", 0) or len(items))
    st = getattr(audit, "chem_stats", None) or {}
    fallback = bool(getattr(audit, "chem_fallback", False))
    out: list[str] = [
        "---", "", f"## 化学身份核验: {total} 条（分子式 / CAS / SMILES / 分子量）", "",
        "> 这一节回答「**论文写的化学身份自洽吗**」: 分子式按内置 IUPAC 原子量复算分子量,",
        "> CAS 号按校验位算法验, SMILES 做语法体检, 再与论文声明的数对账。",
        "> **判据取作者自己的精度**: 论文写 180.16 就按两位小数舍入后比较, 不另设阈值。",
        "> **不从图像像素猜分子结构、不给化合物命名** —— 认结构靠猜, 对账靠算。", "",
    ]
    # 如实留痕：这批结果不是模型主动跑出来的，而是它跳过该步后由程序补算的。
    # 跳过本身允许（调度权在模型），但**不许结果静默消失**，也不许把"跳过"读成"论文里没有"。
    if fallback and items:
        out += ["> ⚠ 本次规划**跳过了「化学身份核验」这一步**，以下 {n} 条是程序在生成报告前"
                "按「免费且必做」口径补算的 —— 结论有效，但这步的跳过已如实记录。".format(n=total),
                ""]
    if not items:
        out += ["本次没有抽到化学身份记录 —— 正文里没有分子式 / CAS / SMILES / 分子量声明"
                "（非化学领域论文属正常）。", ""]
        for n in (st.get("notes") or []):
            out += [f"> {n}", ""]
        return out

    rows = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in items]
    summary = chem.summarize(items, st)

    def _cell(v) -> str:
        return str(v if v is not None else "—").replace("|", "／").replace("\n", " ")

    kinds = summary.get("by_kind") or {}
    if kinds:
        out += [f"**类型分布**: {'　｜　'.join(f'{k} {v}' for k, v in kinds.items())}", ""]
    by = summary.get("by_status") or {}
    if by:
        bits = "　｜　".join(f"{chem.STATUS_ZH.get(k, k)} {v}" for k, v in by.items())
        out += [f"**判定**: {bits}　·　可判定 {summary.get('checked', 0)} 条，"
                f"其中对不上 / 不合规 **{summary.get('bad', 0)}** 条", ""]
    out += [f"**引擎**: {st.get('engine', chem.engine_name())}", ""]
    if total > len(items):
        out += [f"共抽到 {total} 条, 下面列出前 {len(items)} 条。", ""]

    out += ["| # | 类型 | 论文原文 | 页 | 判定 | 依据（算出来的数） |",
            "| --- | --- | --- | --- | --- | --- |"]
    for d in rows:
        status = chem.STATUS_ZH.get(d.get("status"), d.get("status"))
        out.append(f"| {_cell(d.get('rank'))} | {_cell(d.get('kind'))} | "
                   f"`{_cell(d.get('value'))}` | {_cell(d.get('page'))} | "
                   f"{_cell(status)} | {_cell(d.get('detail'))[:160]} |")
    out.append("")
    for n in (st.get("notes") or []):
        out += [f"> {n}", ""]
    return out


def _experiments_md(audit: AuditResult) -> list[str]:
    """② 实验站的章节（阶段二 2D）。

    为什么报告里必须有这一节: ② 站的产物是**一份实验记录**（设计 ＋ 真跑 ＋ 实际值）。
    它若只活在网页上, 评审打开报告时看到的还是"提了假设 → 给了判定"两段 ——
    中间"我们真的动手算过、这是算的过程"这一环就断了。报告是**能带走的那一份**。

    口径与界面**同源**: 字段只读 `ExperimentCard.to_dict()`（那里定义过一次）,
    本函数不另写一套说法, 否则界面与报告迟早对不上。
    """
    cards = list(getattr(audit, "experiments", None) or [])
    total = int(getattr(audit, "experiment_total", 0) or len(cards))
    st = getattr(audit, "experiment_stats", None) or {}
    out: list[str] = [
        "---", "", f"## ② 实验站: {total} 个实验（设计与真跑）", "",
        "> ① 站回答「我们怀疑什么」; 这一节回答「**那怎么验**」—— 而且**真的算了**:",
        "> 每一步都留下调用的工具、喂进去的输入、算出来的结果与耗时, 可以照着复现。",
        "> **不执行任何训练、不运行用户代码**; 小规模复算一律走算术白名单。",
        "> ② 站只负责**测**, 不下「假设成立 / 不成立」的判定 —— 判定是 ③ 站的事。", "",
    ]
    if not cards:
        out += ["本次没有做实验: 没有可设计的假设卡（① 站没提出假设）, 或实验站点被跳过。", ""]
        return out
    if st:
        bits = []
        for label, key in (("复算对不上", "disagree"), ("需实机验证", "machine"),
                           ("实测量·不下结论", "measured"), ("复算一致", "agree"),
                           ("算不出来", "undecided"), ("无需实验", "no_need")):
            if st.get(key):
                bits.append(f"{label} {st[key]}")
        if bits:
            out += ["**结果分布**: " + "　｜　".join(bits), ""]
        if st.get("calls"):
            tools = "、".join(st.get("tools") or []) or "—"
            out += [f"**真跑**: 共调用计算工具 {st['calls']} 次, 用到 {tools}; "
                    f"引擎 {st.get('engine', '—')}（合计 {float(st.get('seconds') or 0):g} 秒）", ""]
    if total > len(cards):
        out += [f"共设计 {total} 个实验, 下面列出前 {len(cards)} 个。", ""]

    rows = [c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in cards]

    def _cell(v) -> str:
        # 表格里的单元格绝不能带竖线（会把表格切碎）—— 与 app.py 的 _cell 同一个理由
        return str(v if v is not None else "—").replace("|", "／").replace("\n", " ")

    # 列名用「实测结果」而不是「实际值」: 这一栏里既有复算出的比值、也有区间核查的结论,
    # 笼统叫"实际值"会让人以为每个数都是同一种东西（下面是逐卡的完整来源说明）。
    out += ["| # | 参数 | 观察结果 | 调用的工具 | 实测结果 |", "| --- | --- | --- | --- | --- |"]
    for d in rows:
        tools = " → ".join(d.get("tool_path") or []) or "—"
        out.append(f"| {_cell(d.get('rank'))} | {_cell(d.get('key'))} | {_cell(d.get('status'))} | "
                   f"`{_cell(tools)}` | {_cell(d.get('actual'))} |")
    out.append("")

    for d in rows:
        out += [f"### #{_cell(d.get('rank'))}　{d.get('key')} — {d.get('status')}", "",
                f"- **待验陈述**: {d.get('claim')}",
                f"- **实验设计**: {d.get('design')}"]
        recs = d.get("records") or []
        if recs:
            out.append("- **真跑记录**（可照着复现）:")
            for r in recs:
                ok = "✓" if r.get("ok") else "×"
                out.append(f"  - {ok} `{r.get('tool')}` {r.get('name')}｜"
                           f"输入 `{_cell(r.get('inputs'))}`｜{r.get('output')}｜{r.get('ms')} ms")
        if d.get("actual") is not None or d.get("expected") is not None:
            out.append(f"- **预期值**: {_cell(d.get('expected'))}　← {d.get('expected_from')}")
            out.append(f"- **实际值**: {_cell(d.get('actual'))}　← {d.get('actual_from')}")
        if d.get("rerun_plan"):
            out.append(f"- **实机方案**: {d['rerun_plan']}")
        out.append(f"- **谁挑的工具路径**: {_cell(d.get('plan_reason'))}")
        out.append("")
    return out


def generate_report(audit: AuditResult) -> str:
    """生成完整 Markdown 审计报告。"""
    r = audit.readiness or {}
    cats = r.get("categories", {})
    cov = audit.coverage or {}
    img = cov.get("image") or {}
    sc = audit.selfcheck or {}
    n_issue = sum(1 for f in audit.findings if f.status != Status.CONSISTENT)
    n_ok = len(audit.findings) - n_issue

    parts = [
        "# Paper2Lab 科研实验一致性审计报告",
        "",
        f"- 论文: **{audit.paper.title or '(未识别标题)'}**",
        f"- 审计时间: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 论文页数: {audit.paper.page_count}　代码文件: {audit.code.python_files} 个(含 notebook cell)",
        f"- 本次审计可信度: **{cov.get('confidence_level', '—')}**"
        f"(得分 {cov.get('confidence_score', '—')}/100)",
        "",
        "---",
        "",
        f"## 复现就绪度: {'—' if r.get('total') is None else r.get('total')} / 100",
        "",
        "> 就绪度评价**被审对象**的可复现程度; 审计可信度评价**本次审计过程**的覆盖与可靠程度, 两者独立。",
        "",
        "| 维度 | 得分 |",
        "| --- | --- |",
    ]
    for k, v in cats.items():
        parts.append(f"| {k} | {'**未核验**' if v is None else v} |")
    parts += [""]
    unv = r.get("unverified") or {}
    if unv:
        parts.append("**「未核验」说明**: 该维度本次没有可比对的对象, **不计入总分**, "
                     "既不是扣分也不代表通过。")
        parts.append("")
        for k, why in unv.items():
            parts.append(f"- **{k}**：{why}")
        parts.append("")
    if r.get("dominant_issues"):
        parts.append("**当前阻碍复现的主要因素**: " + "、".join(r["dominant_issues"]))
        parts.append("")

    # ---- 解析覆盖与可信度(交叉审查: 明确"抽没抽到")
    parts += [
        "---",
        "",
        "## 解析覆盖与审计可信度",
        "",
        "| 项目 | 数值 |",
        "| --- | --- |",
        f"| 论文内容覆盖(全通道) | {cov.get('pages_covered', 0)} / {cov.get('pages_total', 0)}"
        f"({(cov.get('pages_cover_ratio') or 0):.0%})"
        + (f"，其中读图覆盖 {cov.get('pages_covered_by_vision')} 页"
           if cov.get("pages_covered_by_vision") else "") + " |",
        f"| 其中文本层可解析页 | {cov.get('pages_parsed', 0)} / {cov.get('pages_total', 0)} |",
        f"| 其中薄页(图表/公式/扫描区) | {cov.get('pages_thin', 0)} |",
        f"| 论文参数种类覆盖 | {cov.get('kinds_valued', 0)} / {cov.get('kinds_seen', 0)} 类"
        f"({(cov.get('kinds_ratio') or 0):.0%}) |",
        f"| 文中疑似声明句式(参考) | {cov.get('decl_total', 0)} 处, 结构化 {cov.get('decl_matched', 0)} 处 |",
        f"| 代码文件成功解析 | {cov.get('code_ok', 0)} / {cov.get('code_files', 0)} |",
        f"| 代码文件解析失败 | {cov.get('code_failed', 0)} |",
        f"| 多语言通用通道(非 Python) | {len(cov.get('multilang') or [])} 个文件 |",
        f"| 图像/扫描区域 | {img.get('regions', 0)} / 候选 {img.get('candidates', 0)} 个"
        + ("(该论文无位图/扫描区域, 图形均为矢量图, 文本层已覆盖)"
           if not img.get("candidates")
           else ("(已读图 " + str(img.get("regions_read", 0)) + " 个)"
                 if img.get("enabled") else "(图像通道未启用)"))
        + " |",
        f"| 文件格式无法处理 | {len(cov.get('unsupported') or [])} 个 |",
        "",
    ]
    # 各类参数的取值来自哪条通道(让"抽到了"也有出处, 不只是"没抽到"才说)
    kbc = cov.get("kinds_by_channel") or {}
    if any(kbc.values()):
        seg = []
        if kbc.get("text"):
            seg.append(f"正文/shell 等文本通道 {len(kbc['text'])} 类"
                       f"({', '.join(kbc['text'])})")
        if kbc.get("table"):
            seg.append(f"PDF 表格通道 {len(kbc['table'])} 类({', '.join(kbc['table'])})")
        if kbc.get("vision"):
            seg.append(f"视觉读图通道 {len(kbc['vision'])} 类({', '.join(kbc['vision'])})")
        if kbc.get("agent"):
            seg.append(f"**抽取 Agent 补漏** {len(kbc['agent'])} 类({', '.join(kbc['agent'])})")
        parts.append("**取值来源分布**: " + "; ".join(seg) + "。")
        parts.append("")
    if not img.get("regions"):
        pass
    elif not img.get("enabled"):
        parts.append("⚠️ **图像通道未启用**: 论文中发现 "
                     f"{img.get('regions')} 个图片/扫描区域, 但未配置视觉模型 key —— "
                     "这些区域的内容**未参与本次审计**。")
        parts.append("")
    else:
        parts.append(f"**图像通道**(视觉模型 {img.get('model', '—')}, 读图 "
                     f"{img.get('regions_read', 0)} 个区域; 采信 {img.get('kept', 0)} 条, "
                     f"丢弃 {img.get('dropped', 0)} 条):")
        parts.append("- AI 只提议: 读图结果必须附图中原文; 同一区域**两次独立读图取交集**, "
                     "只保留两次一致的值; 该通道结论标为 🔵 Inferred 且**不计入就绪度评分**。")
        for n in img.get("notes", []):
            parts.append(f"- 提示: {n}")
        parts.append("")

    # ---- 抽取 Agent 轨迹(第一期): 每一次尝试都留痕, 可回放
    ag = cov.get("agent") or {}
    if ag:
        steps = ag.get("steps") or []
        filled = ag.get("filled") or {}
        parts += ["**抽取 Agent(第一期)**: "
                  f"待补 {len(ag.get('targets') or [])} 类参数, 补回 {len(filled)} 类; "
                  f"共 {len(steps)} 步尝试, 消耗 LLM {ag.get('budget', {}).get('llm_calls', 0)} 次、"
                  f"耗时 {ag.get('budget', {}).get('seconds', 0)}s; "
                  f"结束原因: {ag.get('stopped_reason', '—')}",
                  "",
                  "> 机制: 每个待补参数按「文本紧邻 → 文本近邻 → 否定式 → LLM 提议 → 聚焦读图」"
                  "逐级尝试; **任一候选都必须过闸门**（回原文核对 + 取值在证据中 + 合理性先验）"
                  "才能入库, 否则丢弃并留痕。宁可不报, 也不报一个可能错的值。",
                  ""]
        if steps:
            parts += ["| # | 参数 | 工具 | 结果 | 取值 | 页 | 说明 |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for s in steps:
                parts.append(f"| {s.get('seq')} | {s.get('display')} | {s.get('tool')} | "
                             f"{s.get('status')} | {s.get('value') or '—'} | "
                             f"{s.get('page') or '—'} | "
                             f"{(s.get('note') or '').replace('|', '/')[:90]} |")
            parts.append("")
        if filled:
            parts.append("**Agent 补回的取值(均已过闸门)**:")
            for k, v in filled.items():
                parts.append(f"- `{k}` = `{v}`")
            parts.append("")

    if cov.get("multilang"):
        parts.append("**多语言通用通道解析的文件**(MATLAB/R/C++/Julia/Fortran/Shell/INI/TOML 等, "
                     "正则通道, 置信度降级):")
        for m in cov["multilang"]:
            parts.append(f"- `{m['file']}` ({m['lang']}) — 抽取参数 {m['params']} 处")
        parts.append("")
    if cov.get("confidence_notes"):
        parts.append("**可信度扣分原因**:")
        for n in cov["confidence_notes"]:
            parts.append(f"- {n}")
        parts.append("")
    if cov.get("kinds_unmatched"):
        parts.append(f"**论文提到但未取到数值的参数种类**: "
                     + "、".join(_DISPLAY.get(k, k) for k in cov["kinds_unmatched"]))
        parts.append("(这些种类的值在**所有通道**里都没取到: 正文句式未匹配、表格/图片里也没有。"
                     "若确认它们出现在图片区域, 可调大读图区域上限后重跑。)")
        parts.append("")
    if cov.get("thin_page_list"):
        parts.append(f"**文本稀薄页清单**: 第 {', '.join(map(str, cov['thin_page_list']))} 页"
                     "(文字层极少, 无法走文本解析)")
        _tread = cov.get("thin_pages_read") or []
        _tunread = cov.get("thin_pages_unread") or []
        if _tread:
            parts.append(f"- ✅ 已由**视觉读图**覆盖: 第 {', '.join(map(str, _tread))} 页")
        if _tunread:
            parts.append(f"- ⚠️ **未读图**: 第 {', '.join(map(str, _tunread))} 页 —— "
                         + (cov.get("thin_unread_reason") or "未参与审计"))
        parts.append("")
    if cov.get("code_failed_list"):
        parts.append("**解析失败文件(其内容未参与审计)**:")
        for ff in cov["code_failed_list"]:
            parts.append(f"- `{ff['file']}` — {ff['reason']}, {ff['lines']} 行")
        parts.append("")
    if cov.get("unsupported"):
        parts.append("**本期无法处理的格式(已识别, 未参与审计)**:")
        for u in cov["unsupported"]:
            parts.append(f"- `{u['file']}` ({u['lang']})")
        parts.append("")

    # ---- ① 假设站（阶段二 2A）
    #      放在「审计概览」之前: 四站叙事是「① 假设 → ③ 判定」,
    #      报告里也应先给"我们要验什么", 再给"验出来是什么"。
    #      2E 的图表线索是「读」这一步的产物, 排在假设卡**之前**（先看图, 再提假设）。
    parts += _figures_md(audit)
    # 2E-化学：同一件「读」工序的第二类产物（分子式 / CAS / SMILES / 分子量对账）
    parts += _chem_md(audit)
    parts += _hypotheses_md(audit)
    # 2026-10-25：模型先判、程序后验的**分歧留档**（关着时本函数返回空，不占版面）。
    # 排在假设卡之后、影响量级之前: 先看「要验什么」, 再看「两边各怎么说」。
    parts += _judge_md(audit)
    # 2E：假设站「算」的产物 —— 影响量级 / 候选解释 / 翻盘 / 边界 / 对照。
    # 排在假设卡之后: 先说要验什么, 再说"如果成立后果有多大"。
    parts += _impact_md(audit)
    # 阶段二 2D：② 实验站（就排在 ① 假设站之后 —— 四站的顺序: 提假设 → 做实验 → 判定）
    parts += _experiments_md(audit)

    # ---- 审计概览
    parts += [
        "---",
        "",
        f"## 审计概览: 发现 {n_issue} 项复现风险, {n_ok} 项一致",
        "",
        "| 参数 | 论文 | 代码/实际 | 状态 | 风险 | 自证 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for f in audit.findings:
        pv = f.paper.raw_value if f.paper else "—"
        cv = f.code.raw_value if f.code else "—"
        parts.append(
            f"| {f.display_name} | `{pv}` | `{cv}` | "
            f"{_STATUS_ICON[f.status]} {_STATUS_ZH[f.status]} | {_RISK_ZH[f.risk.value]} | "
            f"{'✅' if f.verified else '❌'} |")

    # ---- 逐项详情
    parts += ["", "---", "", "## 逐项审计详情(含证据链)", "",
              f"本节对全部 {len(audit.findings)} 项审计发现逐条给出: 结论释义 · 双侧证据"
              "(论文页码·章节 / 代码文件:行号 · 原文片段) · 比对过程 · 证据自证 · 人工核查建议。", ""]
    for f in audit.findings:
        parts.append(_finding_md(f))

    # ---- 参数覆盖清单(解析透明度)
    pm = cov.get("param_matrix") or []
    if pm:
        hit = sum(1 for r in pm if str(r["是否进入比对"]).startswith("✓"))
        parts += ["---", "",
                  f"## 参数覆盖清单({len(pm)} 项已知参数: {hit} 项进入比对, "
                  f"{len(pm) - hit} 项未进入)", "",
                  "每一项写明「抽到了什么」或「为什么没抽到」, 不做「看起来都审过了」的假象:", "",
                  "| 参数 | 论文侧 | 代码侧 | 是否进入比对 | 状态 | 风险 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in pm:
            parts.append(f"| {r['参数']} | {r['论文侧']} | {r['代码侧']} | "
                         f"{r['是否进入比对']} | {r['状态']} | {r['风险']} |")
        parts.append("")

    # ---- 需人工核查清单
    mc = cov.get("manual_check") or []
    parts += ["---", "", f"## 需人工核查清单({len(mc)} 项)", "",
              "以下条目无法由程序单方面给出确定结论, 或存在未参与审计的内容 —— "
              "系统**不假装都已审过**, 请按下表逐项人工确认:", "",
              "| 级别 | 类别 | 对象 | 原因 | 建议动作 |",
              "| --- | --- | --- | --- | --- |"]
    for m in mc:
        parts.append(f"| {m['level']} | {m['kind']} | {m['item']} | {m['why']} | {m['action']} |")
    if not mc:
        parts.append("| — | — | — | 无 | 本次审计未产生需人工核查的条目 |")
    parts.append("")

    # ---- 未参与审计的部分(覆盖边界如实交代)
    not_covered: list[str] = []
    if img.get("regions") and not img.get("enabled"):
        not_covered.append(f"**图像/扫描区域 {img['regions']} 个**: 未配置视觉模型, 内容未参与审计")
    if img.get("enabled") and img.get("regions_read", 0) < img.get("regions", 0):
        not_covered.append(f"**图像区域 {img.get('regions', 0) - img.get('regions_read', 0)} 个**: "
                           "超出耗时预算未读图")
    for fl in cov.get("code_failed_list") or []:
        not_covered.append(f"**代码文件** `{fl['file']}`: {fl['reason']}")
    for u in cov.get("unsupported") or []:
        not_covered.append(f"**格式暂不支持** `{u['file']}` ({u['lang']})")
    if cov.get("thin_page_list"):
        _tread = cov.get("thin_pages_read") or []
        _tunread = cov.get("thin_pages_unread") or []
        not_covered.append(
            f"**论文文本稀薄页**: 第 {', '.join(map(str, cov['thin_page_list']))} 页"
            + (f"(其中第 {', '.join(map(str, _tread))} 页已由读图覆盖)" if _tread else "")
            + (f"; **未读图**: 第 {', '.join(map(str, _tunread))} 页 — "
               + (cov.get("thin_unread_reason") or "") if _tunread else ""))
    if cov.get("kinds_unmatched"):
        not_covered.append("**论文提到但未取到数值的参数种类**（所有通道均未取到）: "
                           + "、".join(_DISPLAY.get(k, k) for k in cov["kinds_unmatched"]))
    if cov.get("rejected"):
        not_covered.append(f"**被合理性校验拒收的抽取 {len(cov['rejected'])} 处**"
                           "(判定为错位抽取, 未纳入比对)")
    # ---- 执行计划与跨任务记忆(第二期)
    ex = getattr(audit, "execution_log", None) or {}
    if ex:
        # 2026-10-25 删掉「经典模式」后只剩一条路：先排计划再执行。
        # 这里如实区分**谁排的计划**（模型 / 规则版兜底），不再有"固定全序"这种说法。
        _by = ((ex.get("plan") or {}).get("planner") or "rule")
        _mode = ("模型排计划（全权调度，程序不设闸门）" if _by == "llm"
                 else "规则版计划（模型未参与，按材料事实排）")
        parts += ["---", "", "## 执行计划", "",
                  f"- 执行模式: **{_mode}**",
                  f"- 执行 {ex.get('ran', 0)} 个工具 / 跳过 {ex.get('skipped', 0)} 个 / "
                  f"覆盖 {ex.get('nodes_total', 0)} 个进度节点 / 总耗时 {ex.get('seconds', 0)}s",
                  "",
                  "| 工具 | 结果 | 耗时 | 说明 |",
                  "| --- | --- | --- | --- |"]
        _mark = {"done": "执行", "skipped": "计划跳过", "failed": "失败"}
        for s in ex.get("steps", []):
            parts.append(f"| {s['title']} | {_mark.get(s['status'], s['status'])} "
                         f"| {s['seconds']}s | {s['reason'] or '—'} |")
        parts.append("")
        for n in ex.get("notes", []):
            parts.append(f"> {n}")
        if ex.get("notes"):
            parts.append("")
        from .toolbox import STEPS as _STEPS, TOOLS as _TOOLS
        parts.append(f"> {len(_STEPS)} 个进度节点由 {len(_TOOLS)} 件工具完成"
                     "——「参数归一化 / 交叉匹配 / 实验结果核验」"
                     "在实现上是同一次参数对齐的三个职责。")
        parts.append("")

    # ---- 分片并发记录(第二期第四批): 让"怎么调度"也可见
    sh = (audit.readiness or {}).get("llm_shards") or {}
    if sh and sh.get("total"):
        _mode = "分片并发" if sh.get("mode") == "sharded" else "串行"
        parts += [f"> **AI 归因调度**：{_mode}"
                  + (f"（{sh.get('workers')} 路）" if sh.get("mode") == "sharded" else "")
                  + f"，处理 {sh.get('total')} 条，耗时 {sh.get('seconds')}s"
                  + (f"，失败 {sh.get('failed')} 条" if sh.get("failed") else "")
                  + "。每片仍是一条独立调用（prompt 里只有该条自己的证据），"
                    "上下文互不干扰；结果**保序**，与串行逐字一致。", ""]

    # ---- 读图通道的分片记录(同上, 让"怎么调度"可见)
    _vsh = ((audit.coverage or {}).get("image") or {}).get("shard") or {}
    if _vsh and _vsh.get("total"):
        _vmode = "分片并发" if _vsh.get("mode") == "sharded" else "串行"
        parts += [f"> **读图调度**：{_vmode}"
                  + (f"（{_vsh.get('workers')} 路）" if _vsh.get("mode") == "sharded" else "")
                  + f"，处理 {_vsh.get('total')} 个区域，耗时 {_vsh.get('seconds')}s"
                  + (f"，失败 {_vsh.get('failed')} 个" if _vsh.get("failed") else "")
                  + "。每个区域仍是**两次独立读图取交集**，区域顺序与结果均保序。", ""]

    # ---- 外部文献候选（4A）: 真实存在、可点开核，但**不是结论的证据**
    # 只在真查到东西时才写这一段 —— 没查到就不占篇幅（界面那边会如实说"没检索到"）。
    _ext = (getattr(audit, "citations", None) or {}).get("external") or {}
    _lit = _ext.get("items") or []
    if _lit:
        parts += ["---", "", f"## 外部文献候选({len(_lit)} 条, 均未核验)", "",
                  "> 这些是在 6 个公开学术索引(OpenAlex / Crossref / Europe PMC / arXiv /"
                  " DataCite / DOAJ)里查到的**真实记录**，点开 DOI 就能核。",
                  "> 但它们与结论只有**关键词层面的相关** —— 「这篇文献是否支持某条结论」"
                  "机器不判，引用前请自己读摘要。", "",
                  "| 年份 | 标题 | 收录库 | 链接 |", "| --- | --- | --- | --- |"]
        for _it in _lit[:10]:
            _title = str(_it.get("title") or "").replace("|", "／")[:90]
            _link = (f"[{_it['doi']}](https://doi.org/{_it['doi']})"
                     if _it.get("doi") else (_it.get("url") or "—"))
            parts.append(f"| {_it.get('year') or '—'} | {_title} | "
                         f"{'、'.join(_it.get('found_in') or []) or '—'} | {_link} |")
        parts.append("")

    # ---- 待确认问题(第二期第二批): 把"系统说不清的地方"交回给用户
    try:
        from .inquiry import KIND_LABEL, build_questions
        qs = build_questions(audit)
    except Exception:      # noqa: BLE001
        qs = []
    if qs:
        parts += ["---", "", f"## 待确认问题({len(qs)} 项, 需要你判断)", "",
                  "> 这些问题系统**无法自行判断**，也不会替你猜答案。页面「待确认问题」面板"
                  "可直接填写答复，答复会作为**人工确认记录**附在本报告里"
                  "（只记录，不自动改写任何结论）。", ""]
        for i, q in enumerate(qs, 1):
            parts.append(f"**{i}. [{KIND_LABEL.get(q.kind, q.kind)}] {q.title}**")
            parts.append("")
            parts.append(f"- 为什么问: {q.basis}")
            for e in q.evidence:
                parts.append(f"- 依据: {e}")
            if q.suggestions:
                parts.append(f"- 可能的答复: {' / '.join(q.suggestions)}")
            parts.append(f"- 不回答的影响: {q.impact}")
            parts.append("")
    if getattr(audit, "clarifications", None):
        parts += ["### 人工确认记录", "",
                  "| 待确认问题 | 用户答复 | 记录时间 |", "| --- | --- | --- |"]
        for c in audit.clarifications:
            parts.append(f"| {c.get('question', '')} | {c.get('answer', '')} "
                         f"| {c.get('time', '')} |")
        parts.append("")

    parts += ["---", "", f"## 未参与本次审计的部分({len(not_covered)} 项)", "",
              "审计结论只覆盖**已成功解析**的内容; 以下部分未参与比对, 不应被视为「已核对」:", ""]
    if not_covered:
        for x in not_covered:
            parts.append(f"- {x}")
    else:
        parts.append("- 无: 本次审计的载体内容已全部纳入解析")
    parts.append("")

    # ---- 被拒收的抽取(合理性校验示众)
    if cov.get("rejected"):
        parts += ["---", "", "## 被合理性校验拒收的抽取", "",
                  "以下数值超出该参数的领域先验范围, 判定为错位抽取, 未纳入审计:",
                  ""]
        for rj in cov["rejected"]:
            parts.append(f"- `{rj['key']}` = `{rj['value']}` @ {rj['location']} — {rj['reason']}")
        parts.append("")

    # ---- 审计自检
    if sc:
        parts += [
            "---",
            "",
            "## 审计自检(证据自证与结论回放)",
            "",
            f"- 校验发现数: {sc.get('checked', 0)}",
            f"- 自证通过: {sc.get('passed', 0)}",
            f"- 自证未通过: {sc.get('failed', 0)}",
            "",
        ]
        if sc.get("failures"):
            parts.append("**未通过清单(结论需人工复核)**:")
            for fl in sc["failures"]:
                parts.append(f"- {fl['param']}: {'；'.join(fl['reasons'])}")
            parts.append("")
        else:
            parts.append("全部结论均通过证据自证与结论复算: 每条证据的原文片段中确实包含所声明的值, "
                         "且一致性结论可由归一化比较复现。")
            parts.append("")

    # ---- 根因追查 + 补丁草案（第三期 3.2 / 3.3）
    # 这两节由工具箱的 rootcause / patch 两站产出，是"报告的下一步"：
    # 用户看完结论的第一个问题就是"那到底该改哪一行"，而在此之前它俩只出现在 CLI 汇总里。
    traces = list(getattr(audit, "rootcause", None) or [])
    patches = list(getattr(audit, "patches", None) or [])
    if traces:
        located = [t for t in traces if getattr(t, "confidence", "none") != "none"]
        parts += [
            "---", "",
            f"## 根因追查（{len(located)} / {len(traces)} 项可定位）", "",
            "> 论文与代码不一致时，同一参数在代码里常有多处赋值。本节从这些赋值点中判断"
            "**最终生效**的那一个（**只读代码，不修改任何文件**）；判断不了的"
            "**明说「无法定位」，绝不猜**。", "",
            "| 参数 | 最终生效值 | 位置 | 置信 | 依据 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for t in traces:
            if getattr(t, "confidence", "none") == "none":
                parts.append(f"| {t.display} | — | — | **无法定位** | {t.reason} |")
            else:
                parts.append(f"| {t.display} | `{t.verdict}` | `{t.verdict_loc}` "
                             f"| {t.confidence} | {t.reason} |")
        parts.append("")
    if patches:
        def _is_prop(p) -> bool:
            # PatchProposal 是 dataclass，`p in list` 会走**逐字段相等**——
            # 两条内容相同的提案会被判成"同一个"，所以一律用 status 判定，不用成员测试。
            return bool(getattr(p, "is_proposed", getattr(p, "status", "") == "proposed"))
        proposed = [p for p in patches if _is_prop(p)]
        refused = [p for p in patches if not _is_prop(p)]
        n_ai = sum(1 for p in proposed if getattr(p, "source", "rule") == "llm")
        parts += [
            "---", "",
            f"## 补丁草案（给方案 {len(proposed)} 条 ｜ 拒绝 {len(refused)} 条）"
            + (f"　其中 **{n_ai} 条的目标由 AI 提名**（见标注，采纳前须人工确认）" if n_ai else ""),
            "",
            "> **只展示，绝不落盘**：本报告不会改动任何代码文件，是否采纳由你决定。"
            "只改**值**、且只对**配置项**；拿不准的一律拒绝并写明理由 —— "
            "宁可不给，也不给一个可能改错的方案。", "",
        ]
        for p in proposed:
            tag = ("### ⚠️ " if getattr(p, "source", "rule") == "llm" else "### ✅ ")
            parts += [f"{tag}{p.display} @ `{p.target}`", "",
                      "```diff", (p.diff or "").strip(), "```",
                      f"- 依据: {p.rationale}"]
            if getattr(p, "stamps", None):
                parts.append("- 程序盖章清单:")
                for s in p.stamps:
                    parts.append(f"  - {s}")
            parts.append("")
        if refused:
            parts += [f"**拒绝的补丁（{len(refused)} 条，逐条写明理由）**", ""]
            for p in refused:
                parts.append(f"- **{p.display}** —— {p.refuse_reason}")
            parts.append("")

    # ---- 附录
    if audit.compare_log:
        parts += ["---", "", "## 附录: 程序比较日志", "",
                  "所有一致性判定均由程序完成, 以下为完整比较记录(可逐条复核):", "",
                  "```text"]
        parts += audit.compare_log
        parts += ["```", ""]

    parts += [
        "---",
        "",
        "## 审计说明",
        "",
        "本报告由 Paper2Lab 自动生成。一致性判断由程序化规则验证完成"
        "(AST 静态分析 + 数值归一化比对), AI 仅负责风险解释与归因。"
        "报告中的每一条结论都经过: ① 合理性先验校验(拒收错位抽取) "
        "② 证据自证(原文片段必须包含所声明的值) ③ 结论复算(归一化比较可重现)。"
        "本期审计范围为可自动验证的数值/配置一致性, 语义级方法审计属于产品路线图。",
    ]
    return "\n".join(parts)
