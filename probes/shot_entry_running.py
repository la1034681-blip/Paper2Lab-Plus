"""跑的过程中截一张 —— 确认实时大屏也长在深色壳里（不是掉到页面底下）。

用法: .venv\\Scripts\\python.exe probes\\shot_entry_running.py [端口]
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
        pg = b.new_page(viewport={"width": 1680, "height": 1050})
        pg.goto(f"http://127.0.0.1:{PORT}", wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)
        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        pg.wait_for_timeout(2600)
        pg.screenshot(path=str(OUT / "entry_00_running.png"))
        body = pg.inner_text("body")
        print("实时画面含「审计中」:", "审计中" in body)
        print("实时画面含「结论尚未生成」:", "结论尚未生成" in body)
        # 大屏在壳里吗？比一下两者的 y 坐标
        js = """() => {
          const e=document.querySelector('.st-key-dk_entry');
          const d=document.querySelector('.st-key-dk_entry .p2l-deck');
          const r=e?e.getBoundingClientRect():null, r2=d?d.getBoundingClientRect():null;
          return {entry:r?[Math.round(r.y),Math.round(r.height)]:null,
                  deck:r2?[Math.round(r2.y),Math.round(r2.height)]:null};
        }"""
        print(pg.evaluate(js))
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
