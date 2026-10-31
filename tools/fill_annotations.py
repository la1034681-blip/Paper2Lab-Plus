"""B 层人工标注结果回填（2026-10-29 执行记录）。

标注员：本轮执行者（AI），逐条读 `paper_snip` / `paper_raw` / `note` 后给出三态判定。
判定标准（与报告里写的一致）：

  ok     —— 证据在原文里支持这个结论：片段确实在讲这个参数、值对得上、状态符合定义
  wrong  —— 证据与结论不匹配（张冠李戴 / 状态语义用错 / 判反）
  unsure —— 原文表述含糊，或需要看代码全貌才能定论；人也不该武断

抽样方式：把 126 条按等间隔取 28 条（`findings_pool.json` 的下标 0,4,8,…）。
不做"只挑好核的"的挑选 —— 等间隔是为了避免主观筛选。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

OUT = Path(r"C:/Users/lijiahui/Desktop/Paper2Lab测试集/标注集")

#: id -> (verdict, note)
V: dict[str, tuple[str, str]] = {
    # ---- 论文未声明：证据是"代码有、论文没有" ----
    "S02_charlesq34-pointnet#01": ("ok", "代码 200(part_seg/train.py:19)，论文侧确实无 epochs 声明"),
    "S09_xingyizhou-CenterNet#16": ("ok", "代码 seed=317，论文正文未写随机种子"),
    "S17_microsoft-Swin-Transformer#07": ("ok", "代码 dropout=0.0，论文未声明该值"),
    "S17_microsoft-Swin-Transformer#15": ("ok", "代码 seed=0，论文未写"),
    "S25_tensorflow-tensor2tensor#01": ("ok", "代码里 batch_size 由命令行给定，论文侧无声明"),
    "S25_tensorflow-tensor2tensor#17": ("ok", "代码 seed 为运行时变量，论文未写"),

    # ---- 论文有、代码未找到 ----
    "S07_google-research-bert#01": ("ok", "论文写 GLUE；官方仓库按任务拆脚本，扫描器未命中 GLUE 键，字面成立"),
    "S13_facebookresearch-detr#03": ("unsure", "DETR 仓库其实有 --dataset_file coco，属扫描器没识别到，"
                                           "状态字面成立但容易让读者误以为代码里没有 COCO"),
    "S17_microsoft-Swin-Transformer#03": ("unsure", "论文侧取自标题里的 Swin Transformer，"
                                                   "把标题当模型声明偏宽；代码侧确实没有模型名常量"),
    "S32_google-research-bert#03": ("unsure", "论文侧取自标题里的 Transformers，与上一条同类"),

    # ---- 无法静态验证 ----
    "S02_charlesq34-pointnet#05": ("ok", "论文 ShapeNet，代码里 data 由运行时参数决定，判不了是对的口径"),
    "S07_google-research-bert#05": ("ok", "论文 batch size 32，代码 batch_size 由命令行给定"),

    # ---- 一致 ----
    "S02_charlesq34-pointnet#09": ("ok", "片段「initial learning rate 0.001, momentum 0.9 and batch size 32」与代码 0.001 对得上"),
    "S07_google-research-bert#13": ("ok", "片段「L2 weight decay of 0.01」与代码 0.01 一致"),
    "S13_facebookresearch-detr#15": ("ok", "片段「DETR with AdamW」与代码 ADAMW 一致"),
    "S17_microsoft-Swin-Transformer#19": ("ok", "片段「trained on ImageNet-22K」与代码 imagenet 一致"),
    "S25_tensorflow-tensor2tensor#21": ("ok", "片段「the Transformer, based solely on attention」与代码 transformer 一致"),

    # ---- 不一致 ----
    "S17_microsoft-Swin-Transformer#11": ("ok", "片段「initial learning rate of 0.001」与代码 0.0005 确实不同"),
    "S25_tensorflow-tensor2tensor#09": ("ok", "片段「dropout rate Pdrop = 0.1」与代码 0.0 确实不同"),

    # ---- 内部不一致（本轮问题最集中的一类）----
    "S09_xingyizhou-CenterNet#12": ("ok", "代码多个配置给了不同模型(dla_34 等)，确属代码内部多值"),
    "S13_facebookresearch-detr#07": ("ok", "代码多个 config 给了不同 dropout 默认值，确属代码内部多值"),
    "S25_tensorflow-tensor2tensor#13": ("ok", "代码多处 momentum 取值不同，确属代码内部多值"),
    "S07_google-research-bert#09": ("wrong",
        "论文里 5e-5(GLU E)、2e-5(SWAG)、1e-4(预训练)是**不同任务/阶段**的取值，不是同一实验的矛盾。"
        "状态应为一 致或按场景拆分，写成「内部不一致」会让读者以为论文自相矛盾"),
    "S09_xingyizhou-CenterNet#04": ("wrong",
        "128/29/4 是论文里不同实验设置的 batch size，代码 32/1 亦然；这是多实验而非矛盾"),
    "S09_xingyizhou-CenterNet#08": ("wrong",
        "140(主实验)与 70(retinanet)分属不同实验；代码 140/230/70 同理。判「矛盾」不成立"),
    "S32_google-research-bert#07": ("wrong",
        "3(GLUE 微调)与 2(另一任务)是不同阶段的值，不属于同一实验内部矛盾"),

    # ---- 单条最明确的错 ----
    "S13_facebookresearch-detr#11": ("wrong",
        "论文侧片段是「sine and cosine functions … positional embedding」，讲的是**位置编码**，"
        "不是学习率调度器；拿它跟代码 StepLR 比属于张冠李戴（正是 `_TEXT_VETO` 本该挡住的情形）"),
}


def main() -> int:
    pool = json.loads((OUT / "findings_pool.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(OUT / "annotations.csv", encoding="utf-8-sig")))
    n_fill = 0
    for row in rows:
        v = V.get(row["id"])
        if v:
            row["human_verdict"], row["human_note"] = v[0], v[1]
            n_fill += 1

    cols = list(rows[0].keys())
    with open(OUT / "annotations.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    filled = [r for r in rows if r["human_verdict"]]
    cnt: dict = {}
    for r in filled:
        cnt[r["human_verdict"]] = cnt.get(r["human_verdict"], 0) + 1
    by_status: dict = {}
    for r in filled:
        d = by_status.setdefault(r["status"], {"ok": 0, "wrong": 0, "unsure": 0, "n": 0})
        d[r["human_verdict"]] += 1
        d["n"] += 1
    judged = cnt.get("ok", 0) + cnt.get("wrong", 0)
    acc = (cnt.get("ok", 0) / judged * 100) if judged else 0.0

    summary = {
        "annotated": len(filled),
        "verdicts": cnt,
        "accuracy_on_decided": round(acc, 1),
        "by_status": by_status,
    }
    (OUT / "human_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"回填 {n_fill} 条（总计 {len(rows)} 条，已标 {len(filled)} 条）")
    print(f"判定分布：{cnt}")
    print(f"在有把握的 {judged} 条里，与系统一致 {cnt.get('ok', 0)} 条，"
          f"判错 {cnt.get('wrong', 0)} 条 → 一致率 {acc:.1f}%")
    print("\n按状态拆开：")
    for k, v in sorted(by_status.items(), key=lambda x: -x[1]["wrong"]):
        print(f"  {k:<22} 标 {v['n']:>2} 条｜对 {v['ok']}｜错 {v['wrong']}｜存疑 {v['unsure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
