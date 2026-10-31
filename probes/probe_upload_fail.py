"""探针: 复现"上传路径迟迟不出结果"的既有缺口, 抓页面异常与服务端日志。

目的：把"300s 超时"从一个时间现象变成一条可读的根因（异常栈 / 卡在哪一步）。

用法:
    .venv\\Scripts\\python.exe probes\\probe_upload_fail.py [等待秒数]

前置: 服务需已在本机 8501 运行, 且**日志已落盘**(见 probe_upload_srv.log)。
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
SAMPLES = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
PDF = SAMPLES / "01_DDPM" / "paper.pdf"
ZIP = SAMPLES / "01_DDPM" / "code.zip"
OUT = Path(__file__).resolve().parent.parent / ".trash_tmp"
OUT.mkdir(exist_ok=True)
WAIT_S = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def wait_text(page, text: str, timeout: int = 60000) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:      # noqa: BLE001
        return False


def start_button(page):
    loc = page.locator('button[kind="primary"]')
    if loc.count():
        return loc.first
    loc = page.get_by_role("button", name=re.compile(r"科\s*研\s*审\s*计"))
    if loc.count():
        return loc.first
    return None


def main() -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1000})
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        print("首屏:", wait_text(page, "论文—代码—实验结果", 45000))

        # 首次加载时组件是渐进渲染的: hero 文字出现 ≠ 按钮已挂上, 轮询等它。
        btn = start_button(page)
        for _ in range(20):
            if btn:
                break
            page.wait_for_timeout(1500)
            btn = start_button(page)
        print("主按钮:", bool(btn))
        if not btn:
            print("⛔ 主按钮始终未出现, 放弃")
            page.screenshot(path=str(OUT / "probe_no_button.png"), full_page=True)
            b.close()
            return
        btn.click()
        t0 = time.time()
        print(f"演示结果: {wait_text(page, '综合复现就绪度', 90000)} "
              f"({time.time() - t0:.1f}s)")

        # ---- B. 上传真实样本
        inputs = page.locator("input[type=file]")
        print("上传控件数:", inputs.count())
        inputs.nth(0).set_input_files(str(PDF))
        page.wait_for_timeout(1500)
        inputs.nth(1).set_input_files(str(ZIP))
        page.wait_for_timeout(2000)
        btn = start_button(page)
        print("上传后按钮:", bool(btn))
        t0 = time.time()
        btn.click()

        step = 5
        for i in range(1, WAIT_S // step + 1):
            page.wait_for_timeout(step * 1000)
            els = page.locator('[data-testid="stException"]')
            if els.count():
                print(f"[{time.time() - t0:.1f}s] ⛔ 页面出现异常框 x{els.count()}")
                print("========== 页面异常文本 ==========")
                print(els.first.inner_text()[:6000])
                print("========== END ==========")
                page.screenshot(path=str(OUT / "probe_upload_fail.png"), full_page=True)
                b.close()
                print("pageerror:", errs[:5])
                return
            txt = page.inner_text("body")
            print(f"[{time.time() - t0:.1f}s] 当前审计对象={'当前审计对象' in txt} "
                  f"准备中={'准备中' in txt} 综合复现就绪度={'综合复现就绪度' in txt}")
            if "当前审计对象" in txt:
                print("✅ 结果已渲染")
                break
        page.screenshot(path=str(OUT / "probe_upload_fail.png"), full_page=True)
        b.close()
        print("pageerror:", errs[:5])


if __name__ == "__main__":
    main()
