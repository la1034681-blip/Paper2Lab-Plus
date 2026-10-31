"""八套样本的影响推演统计（阶段二 2E 实测）。

只跑「论文解析 → 代码扫描 → 假设卡 → 推演」这一段，**不跑完整流水线** ——
要回答的问题是"影响推演在真实材料上能命中多少条"，不是"审计结论对不对"。

统计口径（七个数, 一个都不许美化）:
    cards      这批材料共出多少张假设卡
    impact     其中算得出影响量级的（impact 非空）
    direction  其中还能标出方向的（偏低 / 偏高 / 无法用批大小解释）
    warmup     其中走「预热步数」规则的（新增覆盖面, 与批大小那条同源取数）
    blank      算不出的（材料缺量 —— 这不是失败, 是如实说算不出）
    cand       候选解释能列出来的（③ 那一步有东西可排）
    against    对照基准能给出的（论文里抽到结果指标, ④ 那一步才做得了）

用法: .venv\\Scripts\\python.exe tests/report_impact_samples.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.envfile import load_env                                   # noqa: E402

load_env()

from paper2lab.code_analyzer import analyze_code                          # noqa: E402
from paper2lab.derivation import derive                                   # noqa: E402
from paper2lab.hypothesis import build_hypotheses                         # noqa: E402
from paper2lab.paper_parser import parse_paper                            # noqa: E402

SAMPLES_ROOT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
HERE = ROOT
OUT = SAMPLES_ROOT / "impact_samples_report.md"

CASES = [
    ("01_DDPM", SAMPLES_ROOT / "01_DDPM/paper.pdf", SAMPLES_ROOT / "01_DDPM/code.zip"),
    ("02_DETR", SAMPLES_ROOT / "02_DETR/paper.pdf", SAMPLES_ROOT / "02_DETR/code.zip"),
    ("03_NeRF", SAMPLES_ROOT / "03_NeRF/paper.pdf", SAMPLES_ROOT / "03_NeRF/code.zip"),
    ("04_PointNet", SAMPLES_ROOT / "04_PointNet/paper.pdf", SAMPLES_ROOT / "04_PointNet/code.zip"),
    ("05_ViT", SAMPLES_ROOT / "05_ViT/paper.pdf", SAMPLES_ROOT / "05_ViT/code.zip"),
    ("demo", HERE / "demo/paper.pdf", HERE / "demo/project.zip"),
    ("D题", HERE / "demo/D题论文.pdf", HERE / "demo/D题代码.zip"),
    ("E题", Path(r"C:\Users\lijiahui\Desktop\E题论文.pdf"),
     Path(r"C:\Users\lijiahui\Desktop\E题作答.zip")),
]


def one(name: str, pdf: Path, zipf: Path) -> dict:
    t0 = time.monotonic()
    row = {"name": name, "ok": False, "note": "", "cards": 0, "impact": 0,
           "direction": 0, "warmup": 0, "blank": 0, "cand": 0, "against": 0,
           "items": []}
    if not pdf.exists() or not zipf.exists():
        row["note"] = "材料缺失"
        return row
    try:
        paper = parse_paper(str(pdf))
        code = analyze_code(str(zipf))
    except Exception as ex:                                  # noqa: BLE001
        row["note"] = f"解析失败：{type(ex).__name__}: {str(ex)[:80]}"
        row["seconds"] = round(time.monotonic() - t0, 1)
        return row

    cards = derive(build_hypotheses(paper, code), paper, code)
    row["ok"] = True
    row["cards"] = len(cards)
    for c in cards:
        imp = c.impact or {}
        if not imp:
            row["blank"] += 1
            continue
        row["impact"] += 1
        d = str(imp.get("direction") or "")
        if d:
            row["direction"] += 1
        if str(imp.get("kind") or "") == "warmup":
            row["warmup"] += 1
        if imp.get("candidates"):
            row["cand"] += 1
        ag = imp.get("against") or {}
        if ag.get("available"):
            row["against"] += 1
        row["items"].append({
            "key": c.key, "kind": c.kind, "verdict": c.verdict,
            "mag": imp.get("magnitude"), "direction": d or "（未标）",
            "range": imp.get("range") or {},
            "note": str(imp.get("note") or "")[:60],
            "ncand": len(imp.get("candidates") or []),
            "nflip": len(imp.get("flip") or []),
            "nzero": len(imp.get("zero_when") or []),
            "against": "有" if ag.get("available") else "无",
        })
    row["seconds"] = round(time.monotonic() - t0, 1)
    return row


def main():
    rows = []
    print("=" * 78)
    print("八套样本 · 影响推演统计")
    print("=" * 78)
    for name, pdf, zipf in CASES:
        r = one(name, pdf, zipf)
        rows.append(r)
        if r["ok"]:
            print(f"{name:12s} 卡 {r['cards']:3d} | 影响 {r['impact']:2d} | "
                  f"方向 {r['direction']:2d} | 预热 {r['warmup']:2d} | "
                  f"候选 {r['cand']:2d} | 对照 {r['against']:2d} | "
                  f"算不出 {r['blank']:3d} | {r['seconds']}s")
        else:
            print(f"{name:12s} -- {r['note']}")

    tot = {k: sum(r[k] for r in rows if r["ok"])
           for k in ("cards", "impact", "direction", "warmup", "cand", "against", "blank")}
    print("-" * 78)
    print(f"合计 卡 {tot['cards']} | 影响 {tot['impact']} | 方向 {tot['direction']} | "
          f"预热 {tot['warmup']} | 候选 {tot['cand']} | 对照 {tot['against']} | "
          f"算不出 {tot['blank']}")
    if tot["cards"]:
        print(f"影响命中率 {tot['impact'] / tot['cards']:.1%}｜"
              f"方向命中率 {tot['direction'] / tot['cards']:.1%}（占全部卡）")

    lines = ["# 八套样本 · 影响推演统计", "",
             f"生成时间：2026-10-25　｜　样本 {len([r for r in rows if r['ok']])} / {len(CASES)} 套跑通", "",
             "| 样本 | 假设卡 | 算得出影响 | 标出方向 | 其中预热 | 有候选解释 | 有对照基准 | 算不出 | 耗时 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["ok"]:
            lines.append(f"| {r['name']} | {r['cards']} | {r['impact']} | "
                         f"{r['direction']} | {r['warmup']} | {r['cand']} | "
                         f"{r['against']} | {r['blank']} | {r.get('seconds')}s |")
        else:
            lines.append(f"| {r['name']} | - | - | - | - | {r['note']} |")
    lines += ["",
              f"**合计**：假设卡 {tot['cards']} 张，算得出影响 {tot['impact']} 张，"
              f"其中标出方向 {tot['direction']} 张（含预热规则 {tot['warmup']} 张），"
              f"有候选解释可排 {tot['cand']} 张，有对照基准 {tot['against']} 张，"
              f"算不出 {tot['blank']} 张。", "",
              "> 口径：算不出 ≠ 没影响。分类型参数（dataset / optimizer / model）只有「一样 / 不一样」，",
              "> 没有「影响多大」可言；两侧缺量的也一律返回空。**宁可漏，不要错认。**", ""]

    detail = [r for r in rows if r["ok"] and r["items"]]
    if detail:
        lines += ["## 明细（只列算得出影响的卡）", ""]
        for r in detail:
            lines.append(f"### {r['name']}")
            lines.append("")
            lines.append("| 参数 | 规则 | 判决 | 倍数 | 方向 | 应有区间 | 候选 | 翻盘 | 边界 | 对照 | 备注 |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for it in r["items"]:
                rg = it["range"] or {}
                rng = (f"[{rg.get('low')}, {rg.get('high')}] {rg.get('unit', '')}"
                       if rg else "-")
                lines.append(f"| `{it['key']}` | {it['kind']} | {it['verdict']} | "
                             f"{it['mag']} | {it['direction']} | {rng} | "
                             f"{it['ncand']} 条 | {it['nflip']} 条 | {it['nzero']} 条 | "
                             f"{it['against']} | {it['note'] or '-'} |")
            lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入 {OUT}")


if __name__ == "__main__":
    main()
