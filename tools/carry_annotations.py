"""把上一版标注集里的人工判定，按**内容**搬到新版标注集（2026-10-29 新建）。

为什么需要它
------------
`annotate.py` 的编号是「样本名 + 顺序号」（`S07_...#09`）。审计逻辑一改，
某条发现消失，后面所有编号整体前移 —— 上一轮填的 `human_verdict` 全部错位，
人工判定看起来"过期"了，实际是**对错了行**。

这个工具按内容对号入座。只有**状态相同**才搬，两级匹配：
    1. 样本 + 参数 + 状态 + 论文值 + 代码值   —— 最可靠
    2. 样本 + 参数 + 状态                     —— 唯一命中才采用

**不搬「同参数但状态变了」的行**：状态一变，说的事就不一样了（原来是"这条
判成内部不一致是错的"，新版本那条成了"不一致"，把旧结论贴上去是造假的）。
搬不动的（新版里这条已不存在）单独列出来 —— 那通常正是**被修掉的误报**。

输出
----
  <to>/annotations_filled.csv   新版工作台 + 搬过来的人工判定（原文件不动）
  <to>/carry_report.json        搬了几条、按哪种强度、哪些搬不动

跑法：
  .venv/Scripts/python.exe tools/carry_annotations.py \
      --from "C:/Users/lijiahui/Desktop/Paper2Lab测试集/标注集" \
      --to   "C:/Users/lijiahui/Desktop/Paper2Lab测试集/标注集_v2"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(d: Path) -> list[dict]:
    """读标注工作台。**优先读填过的** —— 人工判定在 `annotations_filled.csv` 里，
    `annotations.csv` 是待填的空白工作台（第一次搬就踩过这个坑：读空白文件，
    搬了 0 条还以为"判定都过期了"）。"""
    for name in ("annotations_filled.csv", "annotations.csv"):
        p = d / name
        if p.exists():
            with open(p, encoding="utf-8-sig") as fh:
                return list(csv.DictReader(fh))
    raise FileNotFoundError(f"没有标注工作台：{d}")


def _keys(row: dict) -> list[tuple]:
    """一条发现的两级索引键（严 → 宽）。状态必须相同，否则不搬。"""
    base = (row.get("sample", ""), row.get("param", ""), row.get("status", ""))
    return [
        base + (row.get("paper_raw", ""), row.get("code_raw", "")),
        base,
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True, help="上一版标注集目录")
    ap.add_argument("--to", dest="dst", required=True, help="新版标注集目录")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    old = _load(src)
    new = _load(dst)
    humans = [r for r in old if (r.get("human_verdict") or "").strip()]

    # 建立 键 -> 新版行号 的索引（只在唯一命中时才可信）
    index: dict[tuple, list[int]] = {}
    for i, row in enumerate(new):
        for k in _keys(row):
            index.setdefault(k, []).append(i)

    carried: list[dict] = []
    for r in humans:
        hit_i, how = None, ""
        for level, k in enumerate(_keys(r)):
            cand = index.get(k, [])
            if len(cand) == 1:
                hit_i, how = cand[0], ["精确", "同参数同状态", "同参数"][level]
                break
        if hit_i is None:
            carried.append({"old_id": r.get("id", ""), "sample": r.get("sample", ""),
                            "param": r.get("param", ""), "how": "未搬（新版已无此条）",
                            "verdict": r["human_verdict"], "note": r.get("human_note", "")})
            continue
        tgt = new[hit_i]
        if (tgt.get("human_verdict") or "").strip():
            continue  # 新版已填过，不覆盖
        tgt["human_verdict"] = r["human_verdict"]
        tgt["human_note"] = r.get("human_note", "")
        carried.append({"old_id": r.get("id", ""), "new_id": tgt.get("id", ""),
                        "sample": r.get("sample", ""), "param": r.get("param", ""),
                        "how": how, "verdict": r["human_verdict"]})

    cols = list(new[0].keys())
    out = dst / "annotations_filled.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(new)

    filled = [r for r in new if (r.get("human_verdict") or "").strip()]
    stat = {"ok": 0, "wrong": 0, "unsure": 0}
    for r in filled:
        stat[r["human_verdict"]] = stat.get(r["human_verdict"], 0) + 1
    judged = stat["ok"] + stat["wrong"]
    report = {
        "old_findings": len(old), "new_findings": len(new),
        "old_human_marks": len(humans), "carried": sum(1 for c in carried if c["how"] != "未搬（新版已无此条）"),
        "dropped": sum(1 for c in carried if c["how"] == "未搬（新版已无此条）"),
        "verdicts": stat,
        "agreement": round(stat["ok"] / judged, 4) if judged else None,
        "detail": carried,
    }
    (dst / "carry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"上一版发现 {len(old)} 条 → 新版 {len(new)} 条；人工判定 {len(humans)} 条")
    print(f"搬到新版 {report['carried']} 条，搬不动 {report['dropped']} 条（新版已无此条）")
    print(f"新版判定分布: 对 {stat['ok']} / 错 {stat['wrong']} / 存疑 {stat['unsure']}")
    if report["agreement"] is not None:
        print(f"一致率（只算对与错）: {report['agreement']:.1%}")
    print(f"输出：{out}")
    print(f"      {dst / 'carry_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
