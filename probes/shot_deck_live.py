"""抓「跑的过程中」的指挥台——验证它是一步一步长出来的，不是跑完才一次性出现。

用法： .venv/Scripts/python.exe probes/shot_deck_live.py [端口]
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8502"
URL = f"http://localhost:{PORT}"
OUT = Path.home() / "Desktop" / "Paper2Lab测试集" / "ui_screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1100})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(8000)

        btn = None
        for x in pg.get_by_role("button").all():
            t = (x.inner_text() or "").replace(" ", "").replace("\n", "")
            if "开始科研审计" in t:
                btn = x
                break
        if btn is None:
            print("没找到开始按钮")
            b.close()
            return 1
        btn.scroll_into_view_if_needed()
        btn.click()
        print("已点击，开始连拍")

        # 等指挥台容器真的出现（.p2l-deck 是它唯一的类名），出现后连拍 3 张。
        # 判据用元素而不是文本 —— 「审计中」这几个字别处也可能有。
        shots = 0
        appear = 0.0
        for i in range(120):
            pg.wait_for_timeout(250)
            try:
                n = pg.locator(".p2l-deck").count()
            except Exception:      # noqa: BLE001
                n = 0
            if n:
                appear = (i + 1) * 0.25
                break
        print(f"  指挥台容器出现 @{appear:.2f}s（count={n}）")
        if n:
            for k in range(3):
                pg.wait_for_timeout(400)
                shots += 1
                path = OUT / f"deck_live_{shots:02d}.png"
                pg.screenshot(path=str(path))
                print(f"  过程态 #{shots}")
        print(f"过程态张数：{shots}")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
