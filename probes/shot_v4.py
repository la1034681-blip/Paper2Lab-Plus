"""第四轮改版截图：主屏字体 ＋ 工序的「本论文具体对应」＋ 七位置只留运行方式。"""
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
        pg = b.new_page(viewport={"width": 1680, "height": 1250})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=● 已完成", timeout=600000)
        pg.wait_for_timeout(2600)
        pg.screenshot(path=str(OUT / "v4_deck.png"), full_page=True)
        for key, name in (("row_parse_paper", "tool_parse"),
                          ("row_analyze_code", "tool_code"),
                          ("pos_retriever", "pos_run")):
            loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button").first
            # 关掉上一个弹窗后 app 会 rerun 一次，DOM 正在重建；直接 count() 会读到 0。
            try:
                loc.wait_for(state="visible", timeout=20000)
            except Exception:                            # noqa: BLE001
                print("缺热区", key)
                continue
            loc.scroll_into_view_if_needed()
            loc.click()
            pg.wait_for_timeout(1700)
            pg.screenshot(path=str(OUT / f"v4_{name}.png"))
            btn = pg.get_by_role("button", name="收起详情")
            if btn.count():
                btn.first.click()
            pg.wait_for_timeout(900)
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
