"""探针: 2A 假设卡在**真机页面**上的渲染验证（跑演示数据即可）。

用法:
    .venv\\Scripts\\python.exe probes\\probe_hypothesis_ui.py

检查点: ① 假设站的标题、"怎么推翻它"、至少一张卡的具体内容, 并留截图。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

MISSING: list[str] = []


def wait_text(page, text: str, timeout: int = 60000) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:      # noqa: BLE001
        return False


def start_button(page):
    for _ in range(20):
        loc = page.locator('button[kind="primary"]')
        if loc.count():
            return loc.first
        page.wait_for_timeout(1500)
    loc = page.get_by_role("button", name=re.compile(r"科\s*研\s*审\s*计"))
    return loc.first if loc.count() else None


def main() -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))

        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        assert wait_text(page, "论文—代码—实验结果", 45000), "首屏没渲染"
        btn = start_button(page)
        assert btn is not None, "没找到主按钮"
        btn.click()

        t0 = time.time()
        ok = wait_text(page, "综合复现就绪度", 180000)
        print(f"演示数据出结果: {ok} ({time.time() - t0:.1f}s)")
        assert ok, "演示数据没跑出结果"

        checks = {
            "① 假设站标题": wait_text(page, "① 假设站", 15000),
            "可证伪假设计数": wait_text(page, "条可证伪假设", 5000),
            "推翻判据标签": wait_text(page, "怎么推翻它", 10000),
            "具体假设(learning_rate)": wait_text(page, "learning_rate", 5000),
            "实机/静态标记": (wait_text(page, "读代码即可定论", 5000)
                              or wait_text(page, "必须实机验证", 5000)),
            "排序理由": wait_text(page, "排这个位置的理由", 5000),
        }
        for k, v in checks.items():
            print(f"  {'[ok]' if v else '[x] '} {k}")
            if not v:
                MISSING.append(k)

        # 滚到假设卡那一段再截：full_page 太长，缩略后看不清卡片细节
        try:
            page.get_by_text("① 假设站", exact=False).first.scroll_into_view_if_needed()
            page.wait_for_timeout(1200)
        except Exception as ex:      # noqa: BLE001
            print("滚动定位失败:", ex)
        page.screenshot(path=str(OUT / "07_hypothesis_cards.png"), full_page=False)
        print("截图:", OUT / "07_hypothesis_cards.png")
        if errs:
            print("pageerror:", errs[:3])
        b.close()

    print("\n缺失项:", MISSING or "无")
    sys.exit(1 if MISSING else 0)


if __name__ == "__main__":
    main()
