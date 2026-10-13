"""「做」复现方案设计（阶段五 · M2 闭环扩展）。

**与 `repro.py` 的关系**（两个都要留，不是重复）：
  · `repro.py`（第三期 3.1）= **复现准备清单**：回答"装什么、配什么值、还要确认什么"；
  · 本模块 = **可执行的复现方案**：在清单之上补三样东西 ——
      ① 有序步骤（第几步做什么、做完了什么算通过）；
      ② **可复现性评级**（A/B/C/D + 分值 + 逐条扣分理由，规则算出来，不是拍的）；
      ③ 把「算」的复算结论与「搜」的引证覆盖接进来当证据。

**为什么要评级**：报告里说"这条有风险"是定性判断，评审要的是可比较的量。
评级把"有多少条不一致、风险多高、覆盖多全、有没有未消解分歧"折算成一个分数，
并且**把每一条扣分理由写出来**——分数可复算，理由可逐条反驳，不是魔法数字。

**三条约束**：
1. **不替用户拍板**：不一致项给两个候选（按论文 / 按代码），标注各自适用目标；
2. **跑不动的明说**：大模型训练级复现一律标注「需实机验证（要 GPU / 时间 / 钱）」，
   本模块只给方案，**不执行任何训练、不写任何文件**；
3. **每个数值都带出处**：沿用 `repro.py` 的口径，没有出处的项不许出现在"照着做"的步骤里。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .models import Risk, Status
from .repro import build_repro_pack

__all__ = ["build_repro_plan", "grade_reproducibility", "ReproPlan", "GRADE_RULES"]


# ---------------------------------------------------------------- 评级规则

# 扣分表: 每一项都是"可复算的规则"，不是主观印象。
# 数值来源（为什么是这些值）：
#   参数类扣分按"这条差异对能否复现出论文结论的影响"×"证据确凿度"分档（与 risk.py 同源）；
#   覆盖类扣分按"解析盲区有多大"（页覆盖 <80% 或代码有文件没读进去）。
GRADE_RULES: dict = {
    "inconsistent_high": 8,     # 不一致 且 高风险
    "inconsistent_medium": 4,
    "inconsistent_low": 2,
    "not_fixed": 6,             # 论文要求固定而代码没固定（结果不可稳定复现）
    "internal_inconsistent": 5,  # 论文/代码内部自相矛盾（连"以哪个为准"都定不下来）
    "missing_in_code": 3,       # 论文声明了但代码没有
    "crosscheck_diff": 4,       # 双路核对出的未消解分歧（两条推理路径不一致）
    "low_page_coverage": 10,    # 论文页覆盖 < 80%
    "code_parse_failure": 5,    # 有代码文件解析失败（内容没进审计）
    "no_citations": 3,          # 有结论拿不到任何引证段落
}

_GRADE_BANDS = ((85, "A"), (70, "B"), (55, "C"))


def _risk_of(f) -> str:
    r = getattr(f, "risk", None)
    return getattr(r, "value", str(r or ""))


def grade_reproducibility(audit: Any, *, crosscheck: Optional[list] = None,
                          citation_cov: Optional[dict] = None) -> dict:
    """按 `GRADE_RULES` 算一个可复现性评级。

    返回 `{level, score, deductions: [{rule, reason, points}], basis}`。
    分数 = 100 - 扣分之和（下限 0）；等级 A≥85 / B≥70 / C≥55 / D<55。
    """
    findings = list(getattr(audit, "findings", None) or [])
    cov = dict(getattr(audit, "coverage", None) or {})
    code = getattr(audit, "code", None)
    deductions: list = []
    seen_keys: set = set()

    def _deduct(rule: str, reason: str, points: Optional[int] = None):
        deductions.append({"rule": rule, "reason": reason,
                           "points": points if points is not None else GRADE_RULES[rule]})

    # ---- 逐条结论扣分（同一参数只扣最重的一次，避免"多处赋值"把分扣穿）
    for f in findings:
        key = getattr(f, "param_key", "")
        st = getattr(getattr(f, "status", None), "value", "")
        if st == Status.CONSISTENT.value:
            continue
        if key in seen_keys:
            continue
        name = getattr(f, "display_name", "") or key
        rk = _risk_of(f)
        if st == Status.INCONSISTENT.value:
            rule = {"high": "inconsistent_high", "medium": "inconsistent_medium"}.get(
                rk, "inconsistent_low")
            _deduct(rule, f"{name}：论文与代码取值不一致（风险 {rk or '未分级'}）")
            seen_keys.add(key)
        elif st == Status.NOT_FIXED.value:
            _deduct("not_fixed", f"{name}：论文要求固定，代码未固定 → 结果不可稳定复现")
            seen_keys.add(key)
        elif st == Status.INTERNAL_INCONSISTENT.value:
            _deduct("internal_inconsistent", f"{name}：内部多处声明互相矛盾")
            seen_keys.add(key)
        elif st == Status.MISSING_IN_CODE.value:
            _deduct("missing_in_code", f"{name}：论文声明了但代码里没有 → 他人无法照做")
            seen_keys.add(key)

    # ---- 双路核对的未消解分歧
    for row in (crosscheck or []):
        if row.get("verdict") == "分歧":
            _deduct("crosscheck_diff",
                    f"{row.get('param_key')}：基线（{row.get('baseline')}）与对撞区"
                    f"（{row.get('collision')}）结论不一致，需人工裁定")

    # ---- 解析覆盖
    pages_total = cov.get("pages_total") or 0
    pages_cov = cov.get("pages_covered") or 0
    if pages_total and pages_cov / pages_total < 0.8:
        _deduct("low_page_coverage",
                f"论文页覆盖 {pages_cov}/{pages_total}（<80%）→ 可能有未读到的声明")
    if code is not None and (getattr(code, "failed_files", None) or []):
        _deduct("code_parse_failure",
                f"代码有 {len(code.failed_files)} 个文件解析失败，内容未进审计")

    # ---- 引证覆盖
    if citation_cov:
        total = citation_cov.get("total") or 0
        cited = citation_cov.get("cited") or 0
        if total and cited == 0:
            _deduct("no_citations", "全部结论都拿不到可定位出处 → 无法被独立复查")
        elif total and cited / total < 0.5:
            _deduct("no_citations",
                    f"仅 {cited}/{total} 条结论有可定位出处（<50%）")

    score = max(0, 100 - sum(d["points"] for d in deductions))
    level = next((g for lo, g in _GRADE_BANDS if score >= lo), "D")
    return {
        "level": level, "score": score, "deductions": deductions,
        "basis": {
            "findings_total": len(findings),
            "findings_penalized": len(seen_keys),
            "pages": f"{pages_cov}/{pages_total}" if pages_total else "—",
            "crosscheck_diffs": sum(1 for r in (crosscheck or [])
                                    if r.get("verdict") == "分歧"),
            "cited": (f"{citation_cov.get('cited')}/{citation_cov.get('total')}"
                      if citation_cov else "—"),
        },
    }


# ---------------------------------------------------------------- 方案

@dataclass
class ReproPlan:
    """可执行的复现方案。"""
    title: str = ""
    grade: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    env: dict = field(default_factory=dict)
    params: list = field(default_factory=list)
    must_confirm: list = field(default_factory=list)
    compute: dict = field(default_factory=dict)
    citations: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"title": self.title, "grade": self.grade, "steps": self.steps,
                "env": self.env, "params": self.params,
                "must_confirm": self.must_confirm, "compute": self.compute,
                "citations": self.citations, "notes": self.notes}

    def to_markdown(self) -> str:
        g = self.grade or {}
        md = [f"# 复现方案 —— 《{self.title or '(未识别标题)'}》", "",
              f"**可复现性评级：{g.get('level', '—')}（{g.get('score', '—')}/100）**", ""]
        if g.get("deductions"):
            md += ["扣分理由（每一条都可复核）：", ""]
            md += [f"- −{d['points']} {d['reason']}" for d in g["deductions"]]
            md.append("")
        else:
            md += ["- 无扣分项。", ""]

        md += ["---", "", "## 一、复现步骤", ""]
        for s in self.steps:
            flag = "**（必做）**" if s.get("blocking") else ""
            md.append(f"### {s['n']}. {s['title']}{flag}")
            md.append("")
            md.append(s["detail"])
            md.append("")
            for e in (s.get("evidence") or []):
                md.append(f"- 依据：{e}")
            if s.get("evidence"):
                md.append("")

        md += ["## 二、关键超参对照表", "",
               "| 参数 | 论文声明 | 代码实现 | 状态 | 复现建议 |",
               "| --- | --- | --- | --- | --- |"]
        for p in self.params:
            md.append(f"| {p['name']} | `{p['paper']}` | `{p['code']}` | "
                      f"{p.get('status_label', p['status'])} | {p['advice']} |")
        md.append("")

        if self.compute.get("relations"):
            md += ["## 三、小规模复算与静态推演", "",
                   "| 关系式 | 结论 | 说明 |", "| --- | --- | --- |"]
            for r in self.compute["relations"]:
                md.append(f"| `{r.get('formula', '')}` | {r.get('verdict', '')} | "
                          f"{r.get('detail', '')} |")
            md.append("")
        if self.citations:
            c = self.citations.get("coverage") or {}
            md += ["## 四、引证覆盖", "",
                   f"- 有可定位出处：{c.get('cited', 0)} / {c.get('total', 0)}",
                   f"- 仅 AI 推断无原文出处：{c.get('inferred_only', 0)}", ""]

        md += ["## 五、复现前必须确认的事", ""]
        if self.must_confirm:
            md += [f"- [ ] **{q.get('title')}** —— {str(q.get('basis', ''))[:90]}"
                   for q in self.must_confirm]
        else:
            md.append("- 无。材料齐备，未发现需要人工判断的问题。")
        md.append("")

        if self.notes:
            md += ["---", "", "## 说明", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(md)


def _env_step(env: dict) -> dict:
    deps = env.get("python_deps") or {}
    lines = []
    if deps:
        lines.append("按代码里的依赖清单安装（版本以代码为准，缺失的标为待确认）：")
        lines.append("")
        lines.append("```")
        for k, v in deps.items():
            lines.append(f"{k}{('==' + str(v)) if v else '  # 版本未指定，需确认'}")
        lines.append("```")
    else:
        lines.append("代码里**没有发现依赖清单** —— 需要你自己确认运行环境（这一步无法自动化）。")
    ev = [f"代码规模 {env.get('code_files', 0)} 个可解析文件"]
    if env.get("langs"):
        ev.append("涉及语言：" + "、".join(env["langs"]))
    if env.get("failed_files"):
        ev.append(f"⚠️ {len(env['failed_files'])} 个文件解析失败，其内容未进审计")
    return {"n": 1, "title": "准备运行环境", "blocking": True,
            "detail": "\n".join(lines), "evidence": ev}


def _data_step(params: list, paper: Any) -> dict:
    ds = [p for p in params if "ataset" in str(p["name"]) or "数据" in str(p["name"])]
    if ds:
        lines = ["按下方对照表 `Dataset` 一项确定数据集与版本后再下载：", ""]
        for p in ds:
            lines.append(f"- **{p['name']}**：论文 `{p['paper']}`"
                         f"（{p['paper_src'] or '无页码'}）| 代码 `{p['code']}`"
                         f"（{p['code_src'] or '无位置'}）—— {p['advice']}")
        ev = [f"{p['name']}：{p['paper_src'] or p['code_src']}" for p in ds]
    else:
        lines = ["**本次没有抽到数据集声明** —— 数据集与版本需要你自己确认，"
                 "系统不替你猜一个。"]
        ev = []
    pc = getattr(paper, "page_count", 0) if paper is not None else 0
    if pc:
        ev.append(f"论文共 {pc} 页")
    return {"n": 2, "title": "准备数据", "blocking": not ds,
            "detail": "\n".join(lines), "evidence": ev}


def _config_step(params: list) -> dict:
    ok = [p for p in params if p["status"] == "consistent"]
    bad = [p for p in params if p["status"] != "consistent"]
    lines = []
    if ok:
        lines.append(f"**一致项（{len(ok)} 个）直接按表配置**："
                     + "、".join(p["name"] for p in ok))
        lines.append("")
    if bad:
        lines.append(f"**需判断项（{len(bad)} 个）**：先看「复现前必须确认的事」，"
                     "再按目标选一侧 —— 复现**论文结论**按论文值，"
                     "复现**代码实际行为**按代码值：")
        lines.append("")
        for p in bad:
            lines.append(f"- **{p['name']}**：论文 `{p['paper']}`"
                         f"（{p['paper_src'] or '—'}） vs 代码 `{p['code']}`"
                         f"（{p['code_src'] or '—'}）—— {p['status_label']}")
    if not lines:
        lines = ["本次没有抽到可比对的参数 —— 说明解析覆盖不足，需先补材料。"]
    return {"n": 3, "title": "按对照表配置超参", "blocking": bool(bad),
            "detail": "\n".join(lines),
            "evidence": [f"{p['name']}：{p['status_label']}" for p in bad[:8]]}


def _seed_step(params: list) -> dict:
    seed = [p for p in params if p["status"] in ("not_fixed", "missing_in_code", "missing_in_paper")
            and ("seed" in p["name"].lower() or "随机" in str(p["name"]))]
    if seed:
        p = seed[0]
        return {"n": 4, "title": "固定随机性（否则结果不可复现）", "blocking": True,
                "detail": f"`{p['name']}` 当前状态为 **{p['status_label']}** —— "
                          "务必在代码里显式固定（PyTorch 还要设 "
                          "`torch.backends.cudnn.deterministic`、NumPy / Python `random`），"
                          "否则每次跑出来的结果都不一样。",
                "evidence": [f"论文声明：{p['paper'] or '未声明'}"]
                            + ([f"代码位置：{p['code_src']}"] if p["code_src"] else [])}
    return {"n": 4, "title": "固定随机性", "blocking": False,
            "detail": "本次未发现「随机性未固定」的结论 —— 若你改了代码或数据，仍建议显式固定种子。",
            "evidence": []}


def _smoke_step(compute: dict) -> dict:
    rels = (compute or {}).get("relations") or []
    lines = [
        "**这一步必须先说清楚边界**：模型训练级复现要 GPU / 时间 / 钱，"
        "本系统**不执行任何训练、不运行你的代码**，「算」只做小规模复算与静态推演。",
        "",
        "所以这一步做的是「小规模试跑 + 静态核对」：",
        "1. 用**极小规模**（几十步 / 几百样本）把流程跑通，确认不报错；",
        "2. 把实际生效的取值打印出来，与上表逐条核对；",
        "3. 需要 GPU 时间才能验证的部分，明确标为「需实机验证」，不要假装算过。",
    ]
    ev = [f"`{r.get('formula')}` → {r.get('verdict')}：{r.get('detail', '')[:80]}"
          for r in rels if r.get("verdict") in ("consistent", "mismatch", "hint")]
    return {"n": 5, "title": "小规模试跑 + 静态核对", "blocking": False,
            "detail": "\n".join(lines), "evidence": ev}


def _result_step(audit: Any) -> dict:
    results = dict(getattr(getattr(audit, "paper", None), "results", None) or {})
    if results:
        lines = ["论文声明的结果，跑完后逐项对照（差多少、在不在误差范围内）：", ""]
        for k, ev in results.items():
            loc = getattr(ev, "location", "")
            raw = getattr(ev, "raw_value", "")
            lines.append(f"- **{k}** = `{raw}`" + (f"（{loc}）" if loc else ""))
        ev = [f"{k}: {getattr(v, 'location', '')}" for k, v in results.items()]
    else:
        lines = ["**本次没有抽到论文的实验结果声明** —— 请人工补充「应与什么对照」，"
                 "系统不替你编一个基准。"]
        ev = []
    lines += ["", "差异超出一致性范围时，**回到第 3 步核对取值**，而不是直接调参凑数。"]
    return {"n": 6, "title": "核对实验结果", "blocking": not results,
            "detail": "\n".join(lines), "evidence": ev}


def _log_step() -> dict:
    return {"n": 7, "title": "留痕与交付", "blocking": False,
            "detail": "跑完把**实际生效的取值**记下来（不要只记论文声明的）："
                      "凡与论文不一致的地方都要写明，这样别人才能判断你的结果是"
                      "「复现失败」还是「按不同设定跑的」。", "evidence": []}


def build_repro_plan(audit: Any, *, compute: Optional[dict] = None,
                     citations: Optional[dict] = None,
                     crosscheck: Optional[list] = None) -> ReproPlan:
    """从审计结果 + 「算」+ 「搜」的产物，拼出一份可执行的复现方案。"""
    pack = build_repro_pack(audit)
    grade = grade_reproducibility(
        audit, crosscheck=crosscheck,
        citation_cov=(citations or {}).get("coverage"))
    env = dict(pack.env or {})
    steps = [
        _env_step(env),
        _data_step(pack.params, getattr(audit, "paper", None)),
        _config_step(pack.params),
        _seed_step(pack.params),
        _smoke_step(compute or {}),
        _result_step(audit),
        _log_step(),
    ]
    notes = [
        "本方案由审计证据自动生成：每个数值都附出处，不一致项由你决定以哪一方为准。",
        "**系统不执行任何训练、不写任何文件**；需要 GPU 时间的部分统一标「需实机验证」。",
        "可复现性评级是规则算出来的（见扣分理由），不是主观印象；"
        "改一处材料后重新生成，分数会跟着变。",
    ]
    return ReproPlan(
        title=getattr(getattr(audit, "paper", None), "title", "") or "",
        grade=grade, steps=steps, env=env, params=pack.params,
        must_confirm=list(pack.must_confirm or []),
        compute=dict(compute or {}), citations=dict(citations or {}),
        notes=notes,
    )
