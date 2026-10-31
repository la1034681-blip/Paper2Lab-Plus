"""「内部不一致」的判定尺度守卫（2026-10-29 新建）。

守的是什么
----------
一个状态叫「内部不一致」（论文或代码自己前后矛盾），过去只要**同一个参数出现两个不同
取值**就报。人工标注抽了 7 条，**4 条判错** —— 错法完全一样：

    BERT 论文里 5e-5 是 GLUE 的、2e-5 是 SWAG 的、1e-4 是预训练的
    CenterNet 论文里 128 是主实验的、29 和 4 是另一组设置的

按任务/阶段分别给参数，是论文的常规写法，不是自相矛盾。这种误报比漏报更糟：
读者拿着报告回原文，找一个根本不存在的错误，然后不再信这份报告。

新尺度（论文侧）
----------------
只有当两处**确实是同一套实验设置**的两个不同取值，才叫矛盾 —— 三条同时满足：

    ┌ 同一页          （跨页 = 章节已经翻过去了）
    ├ 同一章节        （「Training」 vs 「Reti-」是两回事）
    └ 片段里其余数字完全一致（都在说 8 张卡、140 轮，值却不一样）

不满足就只出一条**提示**：「论文里这个参数有多处取值（A @ 第 5 页；B @ 第 12 页）」，
挂在主结论的补充说明里，不算矛盾、不参与扣分。

代码侧尺度不变：同一个仓库里两个文件给了不同值，仍是真问题（配置被覆盖，正是
`demo/edge` 那个例子）。只有测试/示例文件里的取值不算数。

代码侧（2026-10-29 晚补）：判据同样换成「会不会在同一次运行里同时生效」——
同一个文件、同一目录下的入口脚本与配置、根目录平铺，都算真问题（后写的覆盖先写的，
复现的人不知道该填哪个）；分处不同任务目录、或落在 `experiments/` 这类
「一个文件一套实验」的目录，只出提示「代码里这个参数有 N 处取值」。

跑法：  .venv/Scripts/python.exe tests/verify_internal_scope.py
离线、不联网、不读 PDF（全部用构造出来的证据）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.aligner import _internal_conflict, align        # noqa: E402
from paper2lab.models import (                                  # noqa: E402
    Category, CodeInfo, Evidence, PaperInfo, Status,
)

OK = 0
BAD = 0
FAILS: list[str] = []


def check(ok, label: str, extra: str = "") -> None:
    global OK, BAD
    assert isinstance(ok, bool), f"断言值必须是 bool，拿到 {type(ok).__name__}"
    if ok:
        OK += 1
        print(f"  [ok] {label}" + (f"  —— {extra}" if extra else ""))
    else:
        BAD += 1
        FAILS.append(label)
        print(f"  [!!] {label}" + (f"  —— {extra}" if extra else ""))


def sec(title: str) -> None:
    print(f"\n{'─' * 66}\n{title}\n{'─' * 66}")


def pev(val, loc: str, snip: str, excl: bool = False) -> Evidence:
    """论文侧证据。"""
    return Evidence(source="paper", location=loc, snippet=snip, raw_value=str(val),
                    normalized=float(val), context_excluded=excl, parser="regex")


def cev(val, loc: str, snip: str) -> Evidence:
    """代码侧证据。"""
    return Evidence(source="code", location=loc, snippet=snip, raw_value=str(val),
                    normalized=float(val), parser="ast")


def conflict_paper(evs) -> tuple:
    return _internal_conflict("learning_rate", "Learning Rate", Category.PARAM, evs, "paper")


def conflict_code(evs) -> tuple:
    return _internal_conflict("learning_rate", "Learning Rate", Category.PARAM, evs, "code")


def main() -> int:

    # ============================================================ 论文侧：该报的还是要报
    sec("A. 论文侧 —— 同一套设置给两个值，仍然判矛盾")

    a = pev(0.001, "Page 5 · Training",
            "we train the model with a learning rate of 0.001 on 8 GPUs for 140 epochs")
    b = pev(0.0001, "Page 5 · Training",
            "we train the model with a learning rate of 0.0001 on 8 GPUs for 140 epochs")
    f, hint = conflict_paper([a, b])
    check(f is not None, "同页同章节、其余细节一致、只有取值不同 → 判矛盾")
    check(f is not None and f.status == Status.INTERNAL_INCONSISTENT,
          "判矛盾时状态就是 internal_inconsistent")
    check(hint == "", "判矛盾时不再另出提示（两者不并存）")

    # 引用号/表号不同不算「另一套设置」
    c = pev(0.001, "Page 5 · Training",
            "we train with a learning rate of 0.001 on 8 GPUs for 140 epochs [12]")
    d = pev(0.0001, "Page 5 · Training",
            "we train with a learning rate of 0.0001 on 8 GPUs for 140 epochs [13]")
    f2, _ = conflict_paper([c, d])
    check(f2 is not None, "只差引用编号 [12]/[13] → 仍算同一套设置（编号不算细节）")

    # ============================================================ 论文侧：不该报的收住
    sec("B. 论文侧 —— 按任务/阶段分别取值，不判矛盾")

    e = pev(5e-5, "Page 6 · BERTLARGE",
            "fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32")
    g = pev(0.0001, "Page 13 · Pre-training Procedure",
            "Adam with learning rate of 1e-4, L2 weight decay of 0.01")
    f3, hint3 = conflict_paper([e, g])
    check(f3 is None, "跨页（第 6 页 vs 第 13 页）→ 不判矛盾")
    check("多处取值" in hint3, "跨页时给出一条「多处取值」提示")
    check("Page 6" in hint3 and "Page 13" in hint3, "提示里带全部取值位置，便于人工回原文",
          hint3[:60])

    h = pev(5e-5, "Page 7 · BERTLARGE",
            "fine-tune with a learning rate of 5e-5 and a batch size of 48")
    i = pev(2e-5, "Page 7 · SWAG",
            "fine-tune with a learning rate of 2e-5 and a batch size of 16")
    f4, _ = conflict_paper([h, i])
    check(f4 is None, "同页但不同章节（BERTLARGE vs SWAG）→ 不判矛盾")

    j = pev(0.0005, "Page 5 · Training",
            "in with a batch-size of 128 (on 8 GPUs) and learning rate 5e-4 for 140 epochs")
    k = pev(0.00025, "Page 5 · Training",
            "and use batch-size 29 (on 5 GPUs) and learning rate 2.5e-4 for 50 epochs")
    f5, hint5 = conflict_paper([j, k])
    check(f5 is None, "同页同章节，但 8 张卡 / 140 轮 与 5 张卡 / 50 轮 是两套设置 → 不判矛盾")
    check("不判为矛盾" in hint5, "这种情形提示里写明「不判为矛盾」，避免读者误读")

    f6, hint6 = conflict_paper([a])
    check(f6 is None and hint6 == "", "只有一处声明 → 什么都不出（既不判也不提示）")

    sens = pev(0.01, "Page 5 · Training",
               "learning rate of 0.01 on 8 GPUs for 140 epochs", excl=True)
    f7, _ = conflict_paper([a, sens])
    check(f7 is None, "灵敏度语境里的取值本就不参与 → 不判矛盾")

    # ============================================================ 代码侧：会不会同时生效
    sec("C. 代码侧 —— 看「会不会在同一次运行里同时生效」")

    c1 = cev(0.001, "train.py:24", "lr = 0.001")
    c2 = cev(0.0001, "config.py:3", "lr = 0.0001")
    f8, _ = conflict_code([c1, c2])
    check(f8 is not None, "根目录下 train.py 与 config.py 各一个值 → 判矛盾（edge 演示靠这条）")
    check(f8 is not None and f8.risk.value in ("low", "medium", "high"),
          "代码侧仍给出风险等级", f8.risk.value if f8 else "")

    d1 = cev(0.001, "edge/train.py:2", "LEARNING_RATE = 0.001")
    d2 = cev(0.0001, "edge/config.json:1", '"learning_rate": 0.0001')
    f10, _ = conflict_code([d1, d2])
    check(f10 is not None, "同一目录下的入口脚本与配置文件 → 判矛盾（浏览器演示就是这一例）")

    e1 = cev(0.001, "model/train.py:24", "lr = 0.001")
    e2 = cev(0.0001, "model/train.py:41", "lr = 0.0001")
    f11, _ = conflict_code([e1, e2])
    check(f11 is not None, "同一个文件里给了两个值 → 判矛盾（后写的覆盖先写的）")

    g1 = cev(128, "src/lib/opts.py:89", "batch_size = 128")
    g2 = cev(29, "sem_seg/train.py:24", "batch_size = 29")
    f12, hint12 = conflict_code([g1, g2])
    check(f12 is None, "分属两个任务目录（检测 vs 语义分割）→ 不判矛盾")
    check("多套配置" in hint12, "跨任务目录时给出一条「多套配置」提示，不丢信息")

    h1 = cev(140, "src/lib/opts.py:87", "epochs = 140")
    h2 = cev(230, "experiments/ctdet_coco_dla_2x.sh:3", "epochs = 230")
    f13, _ = conflict_code([h1, h2])
    check(f13 is None, "落在 experiments/ 这类「一个文件一套实验」的目录 → 不判矛盾")

    c3 = cev(0.5, "tests/test_train.py:9", "lr = 0.5")
    f9, _ = conflict_code([c1, c3])
    check(f9 is None, "测试文件里的取值不算实现配置 → 不判矛盾")

    i1 = cev(24, "sem_seg/train.py:24", "batch_size = 24")
    i2 = cev(1, "sem_seg/batch_inference.py:12", "batch_size = 1")
    f14, _ = conflict_code([i1, i2])
    check(f14 is None, "训练脚本与推理脚本同目录、批大小不同 → 不判矛盾（推理常设 1）")

    j1 = cev(32, "lib/opts.py:89", "add_argument('--batch_size', type=int, default=32)")
    j2 = cev(1, "lib/opts.py:256", "opt.batch_size = 1")
    f15, _ = conflict_code([j1, j2])
    check(f15 is None, "命令行默认值（add_argument default）不算实际取值 → 不判矛盾")

    k1 = cev(32, "train.py:18", "batch_size = 32")
    k2 = cev(1, "main.py:44", "DataLoader(Dataset(opt, 'val'), batch_size=1, shuffle=False)")
    f16, _ = conflict_code([k1, k2])
    check(f16 is None, "验证集 DataLoader 的 batch_size 不是训练批大小 → 不判矛盾")

    # ============================================================ 端到端：状态与提示
    sec("D. 端到端 —— 多值论文的结论该是什么")

    paper = PaperInfo(params={"learning_rate": e},
                      params_all={"learning_rate": [e, g]},
                      page_count=20)
    code = CodeInfo(params={"learning_rate": [c2]}, file_lines={"config.py": 40})
    res = align(paper, code)
    lr = [x for x in res.findings if x.param_key == "learning_rate"]
    check(len(lr) == 1, "一个参数只给一条主结论（不再额外多一条「内部不一致」）")
    check(bool(lr) and lr[0].status != Status.INTERNAL_INCONSISTENT,
          "论文多值 + 代码单值 → 状态按论文↔代码比对给出，不是「内部不一致」",
          lr[0].status.value if lr else "")
    check(bool(lr) and "多处取值" in (lr[0].note or ""),
          "「多处取值」提示挂在主结论的补充说明里（界面上是 ℹ️ 那一行）")
    check(any("内部核对" in x for x in res.compare_log),
          "比对过程里留了一条记录，可追溯")

    # 真矛盾仍要能被看见
    paper2 = PaperInfo(params={"learning_rate": a},
                       params_all={"learning_rate": [a, b]},
                       page_count=20)
    res2 = align(paper2, code)
    lr2 = [x for x in res2.findings if x.param_key == "learning_rate"]
    check(any(x.status == Status.INTERNAL_INCONSISTENT for x in lr2),
          "同一套设置给两个值 → 仍报「内部不一致」（没把功能关掉）")

    print(f"\n{'=' * 66}")
    print(f"内部不一致判定尺度: {OK + BAD} 项，通过 {OK}，未通过 {BAD}")
    if FAILS:
        print("未通过项：")
        for x in FAILS:
            print(f"  · {x}")
    print("=" * 66)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
