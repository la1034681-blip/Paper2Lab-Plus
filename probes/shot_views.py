"""点开七块区域各一次，看详情弹窗里到底说了什么（同时截图）。"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8599"
URL = f"http://127.0.0.1:{PORT}"
TARGETS = [("tool", "row_align"), ("pos", "pos_retriever"), ("station", "stg0"),
           ("verdict", "vd_r"), ("event", "evt_all"), ("event", "evt8"),
           ("ready", "ready"), ("mat", "mat_paper")]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1080})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=综合复现就绪度", timeout=300000)
        pg.wait_for_timeout(2500)

        for i, (kind, key) in enumerate(TARGETS):
            loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button")
            if not loc.count():
                print(f"✗ {key}: 没有热区")
                continue
            loc.first.scroll_into_view_if_needed()
            loc.first.click()
            ok = False
            for _ in range(12):
                pg.wait_for_timeout(350)
                if pg.locator('div[data-testid="stDialog"]').count():
                    ok = True
                    break
            if not ok:
                print(f"✗ {key}: 弹窗没打开")
                continue
            pg.wait_for_timeout(900)
            txt = pg.locator('div[data-testid="stDialog"]').inner_text()
            head = " / ".join(x for x in txt.split("\n")[:3] if x.strip())
            print(f"✓ {key}: {len(txt)} 字 | {head[:80]}")
            (OUT / f"view_{i}_{key}.txt").write_text(txt, encoding="utf-8")
            pg.screenshot(path=str(OUT / f"view_{i}_{key}.png"))
            pg.get_by_role("button", name="收起详情").first.click()
            pg.wait_for_timeout(1200)
            if pg.locator('div[data-testid="stDialog"]').count():
                print(f"  ✗ {key}: 关不掉")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
