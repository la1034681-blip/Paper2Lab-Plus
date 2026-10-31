"""主界面（深色指挥台）首屏截图 —— 只看「一进来长什么样」，不跑审计。

用法: .venv\\Scripts\\python.exe probes\\shot_entry.py [端口]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8599"
URL = f"http://127.0.0.1:{PORT}"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1100})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "entry_01_top.png"))
        body = pg.inner_text("body")
        print("首屏前 600 字：", body[:600].replace("\n", " | "))
        print("控件：switch =", pg.locator('[role="switch"]').count(),
              " file =", pg.locator('input[type=file]').count(),
              " 热区 =", pg.locator("[class*='st-key-dkhot_']").count())
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
