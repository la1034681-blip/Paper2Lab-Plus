"""看一眼网页上「模型 API」三个框里到底是什么值、状态卡说的是哪一档。

为什么要这个脚本（2026-10-30 晚）：
    用户截图里问一问报 RateLimitError，而弹窗顶部写着 `deepseek-chat`。
    后端 `.env` 已经是本机 intern-s1-mini（`resolve_endpoint`/`probe_local` 实测都对），
    所以只剩一种可能：**页面上的输入框里还是云端那一套**。
    Streamlit 的 `value=` 只在 widget 第一次创建时生效，session 里已有的值不会被代码改写
    —— 光看后端配置永远对，必须到页面上读一遍实际值。

判据（人话）：
    · 三个框的值 = `local` / `http://127.0.0.1:8080/v1` / `intern-s1-mini` → 走本机；
    · 模型名是 `deepseek-chat` 之类 → 本次请求会打到云端，余额/限流就会报 429。
"""
from __future__ import annotations

import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"


def main() -> int:
    with sync_playwright() as p:
        br = p.chromium.launch(channel="msedge", headless=True)
        pg = br.new_page(viewport={"width": 1440, "height": 1000})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_timeout(6000)

        # 展开「高级设置 · 模型 API」
        try:
            exp = pg.get_by_text("高级设置", exact=False).first
            exp.scroll_into_view_if_needed(timeout=8000)
            exp.click()
            pg.wait_for_timeout(2500)
        except Exception as ex:      # noqa: BLE001
            print(f"[warn] 没点开高级设置：{type(ex).__name__} {ex}")

        vals = []
        for lb in ("模型 API Key", "模型 Base URL", "模型名"):
            try:
                box = pg.locator(f"input[aria-label*='{lb}']").first
                vals.append((lb, box.input_value(timeout=5000)))
            except Exception as ex:  # noqa: BLE001
                vals.append((lb, f"<读不到：{type(ex).__name__}>"))

        print("=== 页面三个框的实际值 ===")
        for k, v in vals:
            if "Key" in k and len(v) > 8:
                v = v[:6] + "…" + v[-4:]
            print(f"  {k:16s}= {v!r}")

        body = pg.inner_text("body")
        print("\n=== 页面里的判定文案 ===")
        for kw in ("本机", "页面填写", "服务端", "没有可用", "已部署", "RateLimit"):
            if kw in body:
                for line in body.splitlines():
                    if kw in line and len(line) < 120:
                        print(f"  [{kw}] {line.strip()}")
        br.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
