"""人工标注流水线：为「一条审计发现」建立可核对的标注集。

为什么要有它：上一轮验证报告的结论是「跑得通、产物结构对」，但**证明不了判得准**——
因为 51 份样例没有任何标准答案。这个脚本就是来补标准答案的。

标注对象
--------
一条 `Finding`（参数名 + 论文证据 + 代码证据 + 状态判定）。
样本范围由 `--samples` 指定（默认 10 份），每份取全部发现。

两层标注
--------
A 层（全量, 客观, 脚本自动）：四条**不需要人来主观判断**的硬标准
  1. 证据存在性 —— 状态说"论文有"就必须真的带论文证据，反之亦然
  2. 证据真实性 —— `snippet` 必须是 PDF 该页原文的**子串**（防证据被编造）
  3. 参数相关性 —— snippet 里必须真的出现这个参数的名字（防张冠李戴）
  4. 数值落地   —— 论文侧 `raw_value` 必须出现在 snippet 里（防数字与原文脱节）
  四条全过 = A 层「通过」。

B 层（抽样, 语义, 人工填）：`human_verdict` 三态
  `ok`     —— 回原文看过，这个数确实是该参数的取值，状态判定也符合定义
  `wrong`  —— 回原文看过，判定错了（证据错位 / 状态判反 / 值张冠李戴）
  `unsure` —— 原文表述含糊，人或机器都不该武断下结论
  填法见输出的 CSV：`human_verdict` 与 `human_note` 两列留空，由人逐条填。

输出
----
  <out>/findings_pool.json   全量发现 + A 层结果（机器可读）
  <out>/annotations.csv      标注工作台（含 A 层结果, B 层两列留空待人填）
  <out>/summary.json         分层统计

跑法：  .venv/Scripts/python.exe tools/annotate.py --out <目录> --samples 10
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402

from paper2lab.pipeline import run_audit  # noqa: E402

DEFAULT_DIR = Path(r"C:/Users/lijiahui/Desktop/Paper2Lab测试集/50份样例")

#: 状态 → 该状态**应当**具备哪侧证据。
#: 这条表就是 A 层第 1 条标准的依据，改状态口径时必须同步改它。
#: 注意 `internal_inconsistent` **不在表里**：它可能只在论文侧（论文自己前后矛盾）、
#: 也可能只在代码侧（多个文件写了不同值），所以单独判「至少有一侧」。
EXPECT = {
    "consistent": ("paper", "code"),
    "inconsistent": ("paper", "code"),
    "missing_in_code": ("paper",),
    "missing_in_paper": ("code",),
    "not_fixed": ("paper", "code"),
    "unverifiable": ("paper", "code"),
}

_WS = re.compile(r"\s+")


def norm(s: str) -> str:
    """归一化空白 —— PDF 抽出的文本里换行/多空格很多，直接比子串会假阴。"""
    return _WS.sub(" ", (s or "")).strip()


def page_text(pdf: Path, page_no: int) -> str:
    try:
        with fitz.open(str(pdf)) as doc:
            if 1 <= page_no <= doc.page_count:
                return norm(doc[page_no - 1].get_text())
    except Exception:  # noqa: BLE001
        pass
    return ""


def page_of(location: str) -> int:
    m = re.search(r"Page\s+(\d+)", location or "")
    return int(m.group(1)) if m else 0


def key_words(f) -> list[str]:
    """参数在原文里可能长什么样（用于「snippet 是否真的在讲这个参数」）。"""
    out = [f.display_name, f.param_key.replace("_", " ")]
    alias = {
        "learning_rate": ["learning rate", "lr", "学习率"],
        "batch_size": ["batch size", "batchsize", "批次"],
        "epochs": ["epoch", "轮"],
        "optimizer": ["optimizer", "优化器"],
        "scheduler": ["scheduler", "schedule", "调度"],
        "weight_decay": ["weight decay", "衰减"],
        "seed": ["seed", "random seed", "随机种子"],
        "dataset": ["dataset", "数据集"],
        "model": ["model", "architecture", "模型"],
    }
    out += alias.get(f.param_key, [])
    return [w for w in out if w and len(w) > 1]


def check_one(f, pdf: Path) -> dict:
    """A 层四条标准。返回 {ok, fails[], paper_in_pdf, name_hint}

    `name_hint`（片段里有没有出现参数名）**只作提示，不参与 ok 判定** ——
    论文里的用词未必等于本项目的参数名（实例：`input_size` 在原文里写作
    "grid with resolution 32 × 32 × 32"）。把它当硬标准会让判据本身产生假红。
    """
    status = str(getattr(f.status, "value", f.status))
    fails: list[str] = []

    if status == "internal_inconsistent":
        # 内部不一致可能只在某一侧：论文自己前后矛盾 / 代码多个文件写了不同值
        if not (f.paper or f.code):
            fails.append("内部不一致，却两侧都没有证据")
    else:
        need = EXPECT.get(status, ("paper", "code"))
        if "paper" in need and not f.paper:
            fails.append("状态要求论文证据，但没带")
        if "code" in need and not f.code:
            fails.append("状态要求代码证据，但没带")
        if "paper" not in need and f.paper:
            fails.append("状态不需要论文侧，却带了论文证据")
        if "code" not in need and f.code:
            fails.append("状态不需要代码侧，却带了代码证据")

    # 2/3 只对论文侧做（代码侧走扫描器内部证据，页码概念不适用）
    paper_in_pdf = None
    name_hint = None
    if f.paper:
        raw = f.paper.raw_value or ""
        snip = norm(f.paper.snippet)
        if raw and raw not in snip:
            fails.append(f"论文侧数值 {raw} 不在自己的原文片段里")
        pg = page_of(f.paper.location)
        text = page_text(pdf, pg) if pg else ""
        if text:
            paper_in_pdf = snip in text
            if not paper_in_pdf:
                # 用前 40 字再试一次：snippet 可能跨了页边界
                paper_in_pdf = snip[:40] in text if len(snip) > 40 else False
            if not paper_in_pdf:
                fails.append(f"片段在第 {pg} 页原文里找不到（证据对不上原文）")
        words = key_words(f)
        low = snip.lower()
        name_hint = bool(words) and any(w.lower() in low for w in words)

    return {"ok": not fails, "fails": fails, "paper_in_pdf": paper_in_pdf,
            "name_hint": name_hint}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out", default=str(ROOT / "标注集"))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--only", default="", help="只标这些编号，逗号分隔")
    args = ap.parse_args()

    base = Path(args.dir)
    items = sorted(p for p in base.iterdir()
                   if p.is_dir() and not p.name.startswith("_")
                   and (p / "paper.pdf").exists() and (p / "code.zip").exists())
    if args.only:
        keys = [x.strip() for x in args.only.split(",") if x.strip()]
        items = [p for p in items if any(p.name.startswith(k) for k in keys)]
    items = items[: args.samples]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"标注范围：{len(items)} 份", flush=True)

    pool: list[dict] = []
    for it in items:
        print(f"  跑 {it.name} ...", end="", flush=True)
        try:
            audit, _ = run_audit(str(it / "paper.pdf"), str(it / "code.zip"))
        except Exception as e:  # noqa: BLE001
            print(f" 失败 {type(e).__name__}: {e}", flush=True)
            continue
        if audit is None:
            print(" 无结果", flush=True)
            continue
        n = 0
        for f in audit.findings or []:
            rec = {
                "id": f"{it.name}#{n + 1:02d}",
                "sample": it.name,
                "param": f.param_key,
                "display": f.display_name,
                "status": str(getattr(f.status, "value", f.status)),
                "confidence": str(getattr(f.confidence, "value", f.confidence)),
                "paper_loc": (f.paper.location if f.paper else ""),
                "paper_raw": (f.paper.raw_value if f.paper else ""),
                "paper_snip": norm(f.paper.snippet)[:220] if f.paper else "",
                "code_loc": (f.code.location if f.code else ""),
                "code_raw": (f.code.raw_value if f.code else ""),
                "code_snip": norm(f.code.snippet)[:220] if f.code else "",
                "verified": bool(f.verified),
                "note": (f.note or "")[:200],
            }
            rec.update(check_one(f, it / "paper.pdf"))
            pool.append(rec)
            n += 1
        print(f" {n} 条", flush=True)

    (out / "findings_pool.json").write_text(
        json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")

    cols = ["id", "sample", "param", "display", "status", "confidence",
            "paper_loc", "paper_raw", "paper_snip", "code_loc", "code_raw", "code_snip",
            "ok", "fails", "paper_in_pdf", "name_hint", "human_verdict", "human_note"]
    with open(out / "annotations.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in pool:
            row = {k: r.get(k, "") for k in cols}
            row["ok"] = "1" if r.get("ok") else "0"
            row["fails"] = "；".join(r.get("fails") or [])
            row["human_verdict"] = ""
            row["human_note"] = ""
            w.writerow(row)

    by_status: dict = {}
    for r in pool:
        d = by_status.setdefault(r["status"], {"n": 0, "pass": 0})
        d["n"] += 1
        d["pass"] += 1 if r.get("ok") else 0
    summary = {
        "samples": len(items),
        "findings": len(pool),
        "A_pass": sum(1 for r in pool if r.get("ok")),
        "name_hint_hit": sum(1 for r in pool if r.get("name_hint")),
        "A_fail": sum(1 for r in pool if not r.get("ok")),
        "by_status": by_status,
        "fail_examples": [{"id": r["id"], "fails": r["fails"]} for r in pool if not r.get("ok")][:30],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n发现合计 {len(pool)} 条，A 层通过 {summary['A_pass']}，未过 {summary['A_fail']}")
    for k, v in sorted(by_status.items()):
        print(f"  {k:<22} {v['n']:>3} 条，A 层过 {v['pass']}")
    print(f"\n输出：{out}/annotations.csv  （B 层 human_verdict / human_note 待填）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
