"""给七位置 / 四站 / 判定 / 事件 四类详情拍真机截图。"""
from __future__ import annotations
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(r"C:/Users/lijiahui/Desktop/Paper2Lab测试集/ui_screenshots")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1400})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=已走", timeout=300000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "v3_deck.png"), full_page=True)
        for key, name in (("pos_retriever", "pos"), ("pos_experimenter", "exp"),
                          ("stg0", "stg"), ("vd_g", "vdg"), ("evt_all", "evt")):
            loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button").first
            if not loc.count():
                print("缺热区", key)
                continue
            loc.scroll_into_view_if_needed()
            loc.click()
            pg.wait_for_timeout(1600)
            pg.screenshot(path=str(OUT / f"v3_{name}.png"))
            pg.get_by_role("button", name="收起详情").first.click()
            pg.wait_for_timeout(1000)
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
