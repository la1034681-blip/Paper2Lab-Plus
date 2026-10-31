"""探针: 连续多次「点击 → 出结果」, 定位卡顿是否随次数累积。

用法:
    .venv\\Scripts\\python.exe probes\\probe_repeat_click.py [轮数] [每轮超时秒]

为什么写这个: 单轮实测 11.8s 很快, 但用户反馈「点击后等好久」——
区别在于**服务已经连续跑了很久**。本探针连点, 看第几轮开始退化。
"""
from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def wait_text(page, text: str, timeout: int) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:                   # noqa: BLE001
        return False


def main_button(page):
    # ⚠ 不能取 button[kind="primary"] 的第一个: 页眉「开始审计」导航也是 primary,
    #    排在 DOM 前面, 点它只是锚点跳转, 不会触发运行。
    loc = page.get_by_role("button", name=re.compile(r"开\s*始\s*科\s*研\s*审\s*计"))
    return loc.first if loc.count() else None


def main() -> None:
    times: list[float] = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        assert wait_text(page, "论文—代码—实验结果", 60000), "首屏没渲染"

        # 点击前先确认页面是**未运行**的干净态: 否则 wait_text 会命中上一轮的残留,
        # 报出 0.1s 这种假快（实测踩过）。
        assert not page.get_by_text("综合复现就绪度").count(), \
            "首屏就有结果面板, 上一轮残留没清掉 —— reload 不够, 得换新会话"

        for i in range(1, ROUNDS + 1):
            # 等待信号的选择（试过三种, 记录下来免得再踩）:
            #   ① 「综合复现就绪度」是**常驻**面板 —— 但每轮开头 reload 过, 点击前
            #      页面是干净的, 所以它可以作为"这一轮跑完了"的信号。前提是**必须 reload**。
            #   ② 「准备中…」整轮只 0.3s, 一闪而过, 抓不到。
            #   ③ 探针轮次标记 —— 页面上并不存在, 是想当然。
            # 所以: 每轮先 reload（清结果）→ 点精确按钮 → 等常驻面板回来。
            btn = main_button(page)
            if btn is None:
                print(f"第 {i} 轮: 找不到按钮")
                break
            t0 = time.time()
            btn.click()
            ok = wait_text(page, "综合复现就绪度", LIMIT)
            dt = time.time() - t0
            times.append(dt)
            print(f"第 {i} 轮: {'ok' if ok else '超时'} {dt:.1f}s")
            if not ok:
                break
            page.reload(wait_until="domcontentloaded", timeout=60000)
            wait_text(page, "论文—代码—实验结果", 60000)

        b.close()

    if times:
        print(f"\n各轮耗时: {['%.1f' % t for t in times]}")
        print(f"最快 {min(times):.1f}s | 最慢 {max(times):.1f}s")
        if len(times) > 1 and times[-1] > times[0] * 2:
            print("⚠ 逐轮退化: 耗时随点击次数明显上升")
        else:
            print("无明显退化")
    sys.exit(0)


if __name__ == "__main__":
    main()
