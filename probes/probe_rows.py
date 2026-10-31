"""逐个点开工序行的详情弹窗，报告哪几行点不开 / 点开是空的。"""
from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"

KEYS = ["parse_paper", "analyze_images", "read_figures", "chem_check",
        "fill_missing", "analyze_code", "llm_analyze", "hypothesize",
        "design_experiment", "align", "selfcheck", "rootcause", "patch", "report"]


def main() -> int:
    bad = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1200})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_selector("text=综合复现就绪度", timeout=300000)
        pg.wait_for_timeout(2500)

        for k in KEYS:
            loc = pg.locator(f"[class*='st-key-dkhot_row_{k}'] button")
            if not loc.count():
                print(f"  ✗ {k}: 没有热区按钮")
                bad.append(k)
                continue
            box = loc.first.bounding_box()
            if not box or box["height"] < 6:
                print(f"  ✗ {k}: 按钮高度 {box}")
                bad.append(k)
                continue
            loc.first.scroll_into_view_if_needed()
            loc.first.click()
            opened = False
            for _ in range(12):
                pg.wait_for_timeout(350)
                if pg.locator('div[data-testid="stDialog"]').count():
                    opened = True
                    break
            if not opened:
                print(f"  ✗ {k}: 点了没弹详情（h={box['height']:.0f}）")
                bad.append(k)
                continue
            pg.wait_for_timeout(700)
            txt = pg.locator(".st-key-dk_detail").inner_text()
            sel = pg.locator(f"[class*='st-key-dkhot_row_{k}'] .dk-fl.sel").count()
            print(f"  ✓ {k}: {len(txt)} 字 | 选中标记 {sel} | "
                  f"{txt.split(chr(10))[2][:26] if chr(10) in txt else ''}")
            if len(txt) < 120 or not sel:
                bad.append(k)
            pg.get_by_role("button", name="收起详情").first.click()
            pg.wait_for_timeout(900)
            if pg.locator('div[data-testid="stDialog"]').count():
                print(f"  ✗ {k}: 关不掉")
                bad.append(k)

        print("\n问题行：", bad or "无")
        b.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
