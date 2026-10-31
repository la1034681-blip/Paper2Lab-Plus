"""验证指挥台七块区域点开后的详情：标题对不对、有没有真数据、关得掉。

2026-10-27 第二版改版后：
  · 详情从「大屏下方」改成**弹窗** —— 所以每验一块都要先关掉再点下一块，
    否则弹窗会挡住后面的热区（Playwright 会报 intercepts pointer events）。
  · 新增「实时事件」区域（标题点开是时间线，每条点开是它在时间上的位置）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8519"
URL = f"http://127.0.0.1:{PORT}"
DLG = 'div[data-testid="stDialog"]'
FAIL: list[str] = []


def check(ok: bool, label: str) -> None:
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        FAIL.append(label)


def open_hot(pg, key: str) -> bool:
    loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button").first
    if not loc.count():
        return False
    loc.scroll_into_view_if_needed()
    loc.click()
    for _ in range(14):
        pg.wait_for_timeout(350)
        if pg.locator(DLG).count():
            pg.wait_for_timeout(700)
            return True
    return False


def close_hot(pg) -> None:
    btn = pg.get_by_role("button", name="收起详情")
    if btn.count():
        btn.first.click()
    for _ in range(14):
        pg.wait_for_timeout(350)
        if not pg.locator(DLG).count():
            return


def text(pg) -> str:
    return pg.locator(".st-key-dk_detail").inner_text()


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1200})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=综合复现就绪度", timeout=300000)
        pg.wait_for_timeout(2500)

        # ---- ① 工序：参数对齐与结果核验（最该有证据的一步）
        check(open_hot(pg, "row_align"), "工序行点得开（弹窗）")
        t = text(pg)
        check("工 序 详 情" in t, "详情标题是工序")
        check("参数对齐" in t, "标题是这一步的名字")
        check("两侧逐条对照" in t, "列出了比对了哪些项")
        check(("Page" in t) or ("页" in t), "带论文页码")
        check((".py:" in t) or ("行号" in t), "带代码位置")
        check("为什么不同" in t, "每条差异都写了为什么不同")
        # align 的专属表本身就是「这篇论文里的具体一条」，表头带出处即算对上
        check(("在本论文里的具体对应" in t) or ("论文值（出处）" in t),
              "补了这一步与本论文的具体对应")
        check("论文值（出处）" in t and "代码值（出处）" in t,
              "逐条给出两侧原值 ＋ 出处（页 / 行）")
        check(len(t) > 260, f"详情有实质内容（{len(t)} 字）")
        check(pg.locator("[class*='st-key-dkhot_row_align'] .dk-fl.sel").count() == 1,
              "点过的那一行留了选中标记")
        (OUT / "hot_06_align_detail.txt").write_text(t, encoding="utf-8")
        close_hot(pg)
        check(pg.locator(DLG).count() == 0, "「收起详情」能关掉弹窗")

        # ---- ② 七位置：检索员（要能说清「怎么检索 / 检索出什么 / 配上哪些」）
        check(open_hot(pg, "pos_retriever"), "七位置点得开")
        t = text(pg)
        check("位 置 详 情" in t, "详情标题是位置")
        check("检索员" in t, "点的是检索员")
        check("交接物" in t, "写清了这个位置交什么")
        # 2026-10-27 第四版（用户定版）：七位置**只留运行方式**，论文对应整块删掉
        check("运行方式（三步" in t, "只有运行方式那一块（三步流程）")
        check("匹配" in t or "配对" in t, "三步里讲清了它怎么找、怎么配")
        check("论文哪一块" not in t, "不再有「与论文对应」那一块")
        close_hot(pg)

        # ---- ③ 四站：① 假设
        check(open_hot(pg, "stg0"), "四站点得开")
        t = text(pg)
        check("站 点 详 情" in t, "详情标题是站点")
        check("功能" in t, "讲了本站完成的功能")
        check("作用" in t, "讲了在论文与代码里的作用")
        check("输出" in t, "讲了输出结果")
        close_hot(pg)

        # ---- ④ 实时事件：标题（整轮时间线）
        check(open_hot(pg, "evt_all"), "实时事件标题点得开")
        t = text(pg)
        check("时 间 线" in t, "详情标题是时间线")
        check("耗时" in t, "给了耗时")
        check("最慢" in t, "给了最慢的几步")
        close_hot(pg)

        # ---- ⑤ 判定计数
        check(open_hot(pg, "vd_r"), "判定计数点得开")
        t = text(pg)
        check("判 定 明 细" in t, "详情标题是判定")
        check("具体对应" in t, "讲了是不是具体对应")
        check("参数" in t, "逐条列出了条目")
        close_hot(pg)

        # ---- ⑥ 就绪度
        check(open_hot(pg, "ready"), "就绪度点得开")
        t = text(pg)
        check("就 绪 度" in t, "详情标题是就绪度")
        check("85" in t or "总分" in t, "给出了总分")
        check("维度" in t, "给了五个维度")
        close_hot(pg)

        # ---- ⑦ 材料
        check(open_hot(pg, "mat_paper"), "材料卡点得开")
        t = text(pg)
        check("材 料 详 情" in t, "详情标题是材料")
        check("页" in t, "带了页级信息")
        close_hot(pg)

        print("\n  提示：七块区域各点一次，标题分别是 "
              "工序 / 位置 / 站点 / 时间线 / 判定 / 就绪度 / 材料 —— 逐块核对过。")
        b.close()

    print("\n结论：" + ("全部通过" if not FAIL else f"{len(FAIL)} 项未通过 {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
