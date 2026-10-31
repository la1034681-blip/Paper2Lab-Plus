"""探测「可点击指挥台」：首屏 -> 跑完 -> 点一行工序看详情。"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)
PORT = sys.argv[1] if len(sys.argv) > 1 else "8519"
URL = f"http://127.0.0.1:{PORT}"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1150})
        errs: list[str] = []
        pg.on("pageerror", lambda e: errs.append(str(e)))

        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "hot_00_landing.png"))
        print("[1] 首屏已拍")

        btn = pg.get_by_role("button", name="开始科研审计")
        print("    主按钮个数:", btn.count())
        btn.first.scroll_into_view_if_needed()
        btn.first.click()
        print("[2] 已点开始")

        pg.wait_for_selector("text=综合复现就绪度", timeout=300000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "hot_01_deck.png"))
        print("[3] 指挥台已出")

        n_hot = pg.locator("[class*='st-key-dkhot_']").count()
        print("    可点热区个数:", n_hot)

        row = pg.locator("[class*='st-key-dkhot_row_']").first
        row.scroll_into_view_if_needed()
        box = row.bounding_box()
        print("    第一行位置:", box)
        pg.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        pg.wait_for_timeout(3000)
        body = pg.inner_text("body")
        print("    详情面板出现:", "工 序 详 情" in body)
        pg.screenshot(path=str(OUT / "hot_02_detail.png"))

        # 再点左栏的一张材料卡（用 locator.click：它会自己滚到可见再点中心，
        # 比手算坐标稳 —— 详情面板插进来之后，下面的元素会整体位移）
        mat = pg.locator("[class*='st-key-dkhot_mat_paper'] button").first
        if mat.count():
            mat.click()
            pg.wait_for_timeout(2500)
            print("    材料详情:", "材 料 详 情" in pg.inner_text("body"))
            pg.screenshot(path=str(OUT / "hot_03_mat.png"))

        # 点四站第一站
        stg = pg.locator("[class*='st-key-dkhot_stg0'] button").first
        if stg.count():
            stg.click()
            pg.wait_for_timeout(2500)
            print("    站点详情:", "站 点 详 情" in pg.inner_text("body"))
            pg.screenshot(path=str(OUT / "hot_04_station.png"))

        # 点判定计数（不一致那条）
        vd = pg.locator("[class*='st-key-dkhot_vd_r'] button").first
        if vd.count():
            vd.click()
            pg.wait_for_timeout(2500)
            print("    判定明细:", "判 定 明 细" in pg.inner_text("body"))
            pg.screenshot(path=str(OUT / "hot_05_verdict.png"))

        print("控制台错误:", errs[:3] if errs else "无")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
