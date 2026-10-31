"""探针: 测「点击开始科研审计 → 出结果」的真实耗时, 并记录每一步到哪一步。

用法:
    .venv\\Scripts\\python.exe probes\\probe_slow_click.py [超时秒]

检查点: 分段计时 —— 首屏、点击、出结果, 以及期间 CPU 是否在动。
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def cpu_of(pid: int) -> float:
    """取进程累计 CPU 秒 —— 用来区分「在算」和「在等」。"""
    try:
        out = subprocess.run(
            ["wmic", "process", "where", f"ProcessId={pid}",
             "get", "KernelModeTime,UserModeTime"],
            capture_output=True, text=True, timeout=8).stdout
        nums = [int(x) for x in re.findall(r"\d+", out)]
        return sum(nums) / 1e7        # 100ns 单位 -> 秒
    except Exception:                   # noqa: BLE001
        return -1.0


def listening_pid() -> int:
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if ":8501" in line and "LISTENING" in line:
            return int(line.split()[-1])
    return 0


def wait_text(page, text: str, timeout: int) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:                   # noqa: BLE001
        return False


def main() -> None:
    pid = listening_pid()
    print("服务 PID:", pid, "| 起始累计 CPU:", round(cpu_of(pid), 1), "s")

    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))

        t_start = time.time()
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        assert wait_text(page, "论文—代码—实验结果", 60000), "首屏没渲染"
        print(f"首屏渲染: {time.time() - t_start:.1f}s")

        # ⚠ 不能用 button[kind="primary"] 的第一个 —— 页眉那个「开始审计」导航也是
        #    primary, 排在 DOM 前面, 点它只是锚点跳转, 不会触发运行。
        #    必须按文本精确定位底部的主按钮。
        btn = page.get_by_role("button", name=re.compile(r"开\s*始\s*科\s*研\s*审\s*计"))
        for _ in range(30):
            if btn.count():
                break
            page.wait_for_timeout(1000)
        assert btn.count(), "没找到主按钮"
        cpu0 = cpu_of(pid)
        t0 = time.time()
        btn.first.click()
        print(f"已点击 | 点击前累计 CPU {cpu0:.1f}s")

        # 边等边报进度: 每 10s 打一次页面已推进到哪个环节
        step_marks = ["① 假设站", "综合复现就绪度"]
        deadline = t0 + LIMIT
        got: list[str] = []
        while time.time() < deadline:
            if page.locator("text=正在运行").count() or page.locator('[data-testid="stSpinner"]').count():
                pass
            for m in step_marks:
                if m not in got and wait_text(page, m, 1000):
                    got.append(m)
                    print(f"  [{time.time() - t0:6.1f}s] 出现: {m}")
            if "综合复现就绪度" in got:
                break
            page.wait_for_timeout(1000)

        elapsed = time.time() - t0
        cpu1 = cpu_of(pid)
        print(f"\n出结果: {'综合复现就绪度' in got} | 耗时 {elapsed:.1f}s")
        print(f"服务 CPU 增量: {cpu1 - cpu0:.1f}s "
              f"({'在真算' if cpu1 - cpu0 > elapsed * 0.5 else '多数时间在等 —— 不是算得慢'})")

        page.screenshot(path=str(OUT / "08_slow_click.png"), full_page=False)
        if errs:
            print("pageerror:", errs[:3])
        b.close()


if __name__ == "__main__":
    main()
