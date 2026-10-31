"""本轮视觉巡检截图（统一落桌面临时目录，测完删除）。

覆盖：首屏 / 跑完的大屏 / 工序详情 / 位置详情 / 四站 / 事件 / 判定 /
二级页面（返回按钮）/ 窄窗口（1024 宽，验证侧栏收起后仍能返回）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab_本轮测试截图")
OUT.mkdir(parents=True, exist_ok=True)
PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"


def shot(pg, name, full=False):
    pg.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("  截图", name)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)

    # ---------- 常规宽度 ----------
    pg = b.new_page(viewport={"width": 1680, "height": 1250})
    pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=上传论文资料", timeout=60000)
    pg.wait_for_timeout(1500)
    shot(pg, "01_first_screen")

    pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
    pg.wait_for_selector("text=● 已完成", timeout=600000)
    pg.wait_for_timeout(2600)
    shot(pg, "02_deck_done", full=True)

    for key, name in (("row_align", "03_tool_align"),
                      ("pos_retriever", "04_pos"),
                      ("stg0", "05_station"),
                      ("evt_all", "06_events"),
                      ("vd_r", "07_verdict"),
                      ("ready_score", "08_readiness")):
        loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button").first
        try:
            loc.wait_for(state="visible", timeout=15000)
        except Exception:                                   # noqa: BLE001
            print("  缺热区", key)
            continue
        loc.scroll_into_view_if_needed()
        loc.click()
        pg.wait_for_timeout(1700)
        shot(pg, name)
        close = pg.get_by_role("button", name="收起详情")
        if close.count():
            close.first.click()
        pg.wait_for_timeout(900)

    # 二级页面：返回按钮
    pg.get_by_role("button", name="逐条对表").first.scroll_into_view_if_needed()
    pg.get_by_role("button", name="逐条对表").first.click()
    pg.wait_for_timeout(3500)
    shot(pg, "09_second_level", full=True)
    pg.get_by_role("button", name="← 返回指挥台").first.click()
    pg.wait_for_timeout(2500)
    shot(pg, "10_back_to_deck")
    pg.close()

    # ---------- 窄窗口 ----------
    pg2 = b.new_page(viewport={"width": 1024, "height": 900})
    pg2.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg2.wait_for_selector("text=上传论文资料", timeout=60000)
    pg2.get_by_role("button", name="开 始 科 研 审 计").first.click()
    pg2.wait_for_selector("text=● 已完成", timeout=600000)
    pg2.wait_for_timeout(2500)
    shot(pg2, "11_narrow_deck", full=True)
    pg2.get_by_role("button", name="逐条对表").first.scroll_into_view_if_needed()
    pg2.get_by_role("button", name="逐条对表").first.click()
    pg2.wait_for_timeout(3500)
    shot(pg2, "12_narrow_second")
    has_back = pg2.get_by_role("button", name="← 返回指挥台").count() > 0
    print("  窄窗口二级页有返回按钮:", has_back)
    if has_back:
        pg2.get_by_role("button", name="← 返回指挥台").first.click()
        pg2.wait_for_timeout(2500)
        shot(pg2, "13_narrow_back")
    b.close()
print("done")
