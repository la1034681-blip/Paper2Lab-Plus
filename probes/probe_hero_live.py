"""临时探针：确认线上 hero 是否渲染出「N 步」，用于区分「未推送旧版」与「hero 未渲染」。"""
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = "https://paper2lab.streamlit.app/"


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
        fr = None
        for _ in range(40):
            for f in pg.frames:
                if "/~/" in (f.url or ""):
                    fr = f
                    break
            if fr:
                break
            time.sleep(1)
        print("frame:", bool(fr), flush=True)
        if not fr:
            try:
                print("标题:", pg.title(), flush=True)
                print("顶层正文片段:", pg.inner_text("body")[:400].replace("\n", " | "), flush=True)
            except Exception as exc:  # noqa: BLE001
                print("读顶层正文失败:", exc, flush=True)
        text = ""
        for i in range(12):
            time.sleep(5)
            if not fr:
                break
            try:
                text = fr.inner_text("body")
            except Exception:
                text = ""
            hits = re.findall(r".{0,25}步.{0,25}", text)
            print(f"[{i}] len={len(text)} hits={hits[:3]}", flush=True)
            if hits:
                break
        print("=== 前 500 字 ===", flush=True)
        print(text[:500], flush=True)
        for kw in ["步", "载体", "智能体", "Paper2Lab", "STEPS", "演示数据"]:
            print(f"含「{kw}」: {kw in text}", flush=True)
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
