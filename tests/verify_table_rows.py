"""表格结果通道：按坐标重组的四条边界。

为什么要有这套（2026-10-30）:
    26 份论文一条结果指标都抽不到，根因不在正则，在**文本层是按列输出的** ——
    表格里「AP」和它对应的数字隔着几十行，按句子切永远凑不到一起。
    修法是按 (x, y) 坐标把页面重组成「行 × 单元格」，再按列号对齐取指标值。

    这个功能最容易出两类错，两类都要守住:
      ① **漏**: 真表认不出来（DETR 的 Table 3 就是标准形态）;
      ② **误**: 把正文碎片当成表（实测 GPT-2 的列名恰好叫 AP、DINOv2 的
         "Accuracy | Accuracy 60 | Accuracy"）。

    PDF 由本脚本现场生成，插入顺序**刻意按列**（模拟 arXiv 双栏论文的病态文本层），
    不依赖桌面上的样例文件，离线可跑、结论稳定。

跑法:  .venv/Scripts/python.exe tests/verify_table_rows.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz                                              # noqa: E402
from paper2lab.paper_parser import (_extract_rows,       # noqa: E402
                                    _page_rows, _rows_to_results)

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


#: 列的左边界（pt）与行距。列的间隔给足 60pt，保证坐标切分不会糊在一起。
COL_X = [60.0, 200.0, 290.0, 380.0]
ROW_Y0 = 120.0
ROW_DY = 16.0


def make_pdf(path: Path, cols: list[list[str]], *, by_column: bool = True) -> None:
    """把 cols（每行一个 list）写进 PDF。

    by_column=True 时**按列写入**：先写完第 1 列的所有行，再写第 2 列……
    这正是 arXiv 双栏论文表格的文本层顺序 —— `get_text("text")` 拿到的是
    「AP / AP50 / FPS / 32.8 / 40.1 / ...」这种一列一列的碎片。
    """
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    rows = len(cols)
    n_col = max(len(r) for r in cols)
    if by_column:
        order = [(ri, ci) for ci in range(n_col) for ri in range(rows)]
    else:
        order = [(ri, ci) for ri in range(rows) for ci in range(n_col)]
    for ri, ci in order:
        if ci >= len(cols[ri]):
            continue
        txt = cols[ri][ci]
        if not txt:
            continue
        page.insert_text((COL_X[ci], ROW_Y0 + ri * ROW_DY), txt, fontsize=9)
    doc.save(str(path))
    doc.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="p2l_tbl_"))

    # ============================================================ 一、能还原
    sec("一、病态文本层（按列输出）—— 必须还原成表")
    good = tmp / "good.pdf"
    make_pdf(good, [
        ["Method", "AP", "AP50", "FPS"],
        ["DETR", "38.2", "60.1", "28"],
        ["DETR-DC5", "41.6", "63.0", "24"],
        ["Ours", "42.0", "62.4", "26"],
    ])
    rows = _extract_rows(str(good))
    flat = [r for page_rows in rows.values() for r in page_rows]
    hits = _rows_to_results(flat)
    check(len(flat) >= 4, "按列写入的表格被重组成行", f"{len(flat)} 行")
    check(any(len(r) == 4 for r in flat), "行里切出了 4 个单元格",
          f"最宽一行 {max((len(r) for r in flat), default=0)} 个")
    check(len(hits) >= 1, "认出了指标表", f"{[h[0] for h in hits]}")
    got = dict((m, v) for m, v, _s in hits)
    check(got.get("ap") == "42", "AP 取到表内最大值（论文主张的最好结果）",
          f"ap={got.get('ap')}")
    check("fps" not in got and "speed" not in got,
          "FPS 这类非指标列不被当成结果", f"{sorted(got)}")

    # 逐单元格顺序写入（PDF 本身正常）时也要能还原 —— 不能只会处理病态输入
    normal = tmp / "normal.pdf"
    make_pdf(normal, [["Method", "AP", "AP50"], ["A", "30.1", "51.0"],
                      ["B", "33.3", "54.2"]], by_column=False)
    hits_n = _rows_to_results([r for p in _extract_rows(str(normal)).values() for r in p])
    check(any(m == "ap" for m, _v, _s in hits_n), "正常顺序的 PDF 同样能认出来")

    # ============================================================ 二、误报闸
    sec("二、误报闸 —— 正文碎片不许当成表")

    # 反例 1: 只有一个指标列（GPT-2 的列名恰好叫 AP）
    one = tmp / "one.pdf"
    make_pdf(one, [["SimLex999", "MEN", "AP"],
                   ["0.194", "0.216", "0.321"],
                   ["0.185", "0.201", "0.305"]])
    check(_rows_to_results([r for p in _extract_rows(str(one)).values() for r in p]) == [],
          "只有一个指标列 → 不认（列名撞上缩写的情形）")

    # 反例 2: 表头同一个词重复（DINOv2 的 Accuracy 碎片）
    dup = tmp / "dup.pdf"
    make_pdf(dup, [["Accuracy", "Accuracy", "Accuracy"],
                   ["64", "58", "60"],
                   ["63", "57", "59"]], by_column=False)
    check(_rows_to_results([r for p in _extract_rows(str(dup)).values() for r in p]) == [],
          "表头指标全是同一个词 → 不认（正文被切段的形态）")

    # 反例 3: 正文句子（含指标词但没有对齐的数值列）
    prose = tmp / "prose.pdf"
    make_pdf(prose, [["we report accuracy on the", "dev set and the test set"],
                     ["which improves accuracy by a", "large margin over the baseline"],
                     ["and we also report F1 for", "every single task in the suite"]],
             by_column=False)
    check(_rows_to_results([r for p in _extract_rows(str(prose)).values() for r in p]) == [],
          "正文句子 → 不认")

    # 反例 4: 表头认得出、但下面没有数据行（孤零零一个表头）
    lonely = tmp / "lonely.pdf"
    make_pdf(lonely, [["Method", "AP", "AP50"]], by_column=False)
    check(_rows_to_results([r for p in _extract_rows(str(lonely)).values() for r in p]) == [],
          "只有表头没有数据行 → 不认")

    # ============================================================ 三、取值方向
    sec("三、取值方向 —— 越大越好取最大，越小越好取最小")
    low = tmp / "low.pdf"
    make_pdf(low, [["Model", "PPL", "BLEU"],
                   ["base", "24.5", "25.1"],
                   ["ours", "18.2", "28.4"],
                   ["big", "20.1", "27.0"]], by_column=False)
    got_low = {m: v for m, v, _s in
               _rows_to_results([r for p in _extract_rows(str(low)).values() for r in p])}
    check(got_low.get("ppl") == "18.2", "PPL（越小越好）取表内最小值",
          f"ppl={got_low.get('ppl')}")
    check(got_low.get("bleu") == "28.4", "BLEU（越大越好）取表内最大值",
          f"bleu={got_low.get('bleu')}")

    # ============================================================ 四、可核验
    sec("四、证据要能回原文核 —— snippet 同时给出表头与取值口径")
    snip = next((s for m, _v, s in hits if m == "ap"), "")
    check("表头" in snip and "取值行" in snip, "snippet 里有表头和取值行原文", snip[:70])
    check("42" in snip.split("取值行")[-1], "取值行写的就是取到 42 的那一行（不是首行）")
    check("→" not in snip and len(snip) <= 240, "snippet 长度受控", f"{len(snip)} 字")

    # ============================================================ 五、不拖慢
    sec("五、边界 —— 空页 / 无文字页不崩")
    blank = tmp / "blank.pdf"
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(str(blank))
    doc.close()
    check(_extract_rows(str(blank)) == {}, "空白页返回空行集，不抛异常")

    print(f"\n{'=' * 66}")
    print(f"表格结果通道: {OK + BAD} 项，通过 {OK}，未通过 {BAD}")
    if FAILS:
        print("未通过项：")
        for f in FAILS:
            print(f"  · {f}")
    print("=" * 66)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
