"""截图：实验室指挥台（跑一次演示数据，抓完成态与过程态）。

用法： .venv/Scripts/python.exe probes/shot_deck.py [端口]
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
        pg = b.new_page(viewport={"width": 1680, "height": 1200})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(9000)

        body = pg.inner_text("body")
        print("[首屏] 侧栏有指挥台:", "指挥台" in body)
        pg.screenshot(path=str(OUT / "deck_00_start.png"), full_page=False)

        # 找「开始科研审计」按钮。注意 Streamlit 会把中文按字加空格渲染
        # （"🚀 开 始 科 研 审 计"），所以先去掉所有空白再比。
        btn = None
        for x in pg.get_by_role("button").all():
            t = (x.inner_text() or "").replace(" ", "").replace("\n", "")
            if "开始科研审计" in t:
                btn = x
                break
        if btn is None:
            print("[警告] 没找到按钮，当前可见按钮：")
            for x in pg.get_by_role("button").all()[:20]:
                print("   -", (x.inner_text() or "").strip()[:30])
            b.close()
            return 1
        btn.scroll_into_view_if_needed()
        btn.click()
        print("[点击] 已开始审计")

        # 等指挥台在结果区出现（body 里出现「七 位 置」说明大屏渲染了）
        t0 = time.time()
        ok = False
        while time.time() - t0 < 240:
            pg.wait_for_timeout(2000)
            txt = pg.inner_text("body")
            if "七 位 置" in txt and "已 完 成".replace(" ", "") in txt.replace(" ", ""):
                ok = True
                break
            if "没有产出审计结论" in txt:
                print("[失败] 页面报「没有产出审计结论」")
                pg.screenshot(path=str(OUT / "deck_err.png"), full_page=True)
                b.close()
                return 1
        print(f"[等待] {time.time() - t0:.1f}s 完成态={ok}")

        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(OUT / "deck_01_done.png"), full_page=True)

        # 只截指挥台那一块
        try:
            el = pg.locator(".p2l-deck").first
            el.scroll_into_view_if_needed()
            pg.wait_for_timeout(600)
            el.screenshot(path=str(OUT / "deck_02_panel.png"))
            print("[截图] 指挥台单块 OK")
        except Exception as ex:      # noqa: BLE001
            print("[截图] 单块失败:", ex)

        txt = pg.inner_text("body")
        for k in ("① 假设", "② 实验", "③ 验证", "④ 迭代", "工 序 执 行",
                  "实 时 事 件", "判 定 计 数", "解析员", "撰写员"):
            print(f"  {k}: {k.replace(' ', '') in txt.replace(' ', '')}")
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
