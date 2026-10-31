"""跑完一轮之后的主界面截图（深色指挥台 · 已完成态）。

用法: .venv\\Scripts\\python.exe probes\\shot_entry_done.py [端口]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8599"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1150})
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=综合复现就绪度", timeout=300000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "entry_02_done.png"))
        # 点一下「主持人」，看详情面板是不是真的摊开
        pg.locator("[class*='st-key-dkhot_pos_moderator'] button").first.click()
        pg.wait_for_timeout(2200)
        pg.screenshot(path=str(OUT / "entry_03_pos.png"))
        print(pg.locator(".st-key-dk_detail").inner_text()[:400].replace("\n", " | "))
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
