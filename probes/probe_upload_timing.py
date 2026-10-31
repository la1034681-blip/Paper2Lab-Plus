"""探针: 走**上传路径**（真上传文件）, 测点击 → 出结果的耗时。

用法:
    .venv\\Scripts\\python.exe probes\\probe_upload_timing.py [超时秒] [材料目录]

为什么单独写: 用户反馈的「点击后等好久」发生在上传路径;
而演示数据命中 session 快照快路径（0.1s）会把问题完全掩盖掉。
所以必须用**新材料**（换指纹 → 不命中快照）跑真实计时。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SRC = Path(sys.argv[2] if len(sys.argv) > 2
          else r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\01_DDPM")
SHOTS = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
SHOTS.mkdir(parents=True, exist_ok=True)


def wait_text(page, text: str, timeout: int) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:                   # noqa: BLE001
        return False


def wait_button(page, timeout: int = 60000) -> None:
    """⚠ 不能取 button[kind="primary"] 的第一个: 页眉「开始审计」导航也是 primary,
    排在 DOM 前面, 点它只是锚点跳转, 不会触发运行。必须按文本精确定位。"""
    loc = page.get_by_role("button", name="开 始 科 研 审 计")
    for _ in range(timeout // 1000):
        if loc.count():
            return
        page.wait_for_timeout(1000)
    raise AssertionError("主按钮没出现")


def main() -> None:
    pdf, zipf = SRC / "paper.pdf", SRC / "code.zip"
    assert pdf.exists(), f"缺材料: {pdf}"

    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        assert wait_text(page, "论文—代码—实验结果", 60000), "首屏没渲染"

        # ---- 关掉演示数据开关, 走上传路径
        # ⚠ Streamlit 的 switch 是个 <div role="switch">, 真正的可点区域是它内部的
        #   track; 直接 .click() 会报 intercepts pointer events（点到外层容器）。
        #   点它的 label 文本最稳 —— 点了必然触发 onChange。
        demo_label = page.get_by_text("使用内置演示数据", exact=False).first
        if demo_label.count():
            if page.get_by_role("switch").first.is_checked():
                demo_label.click(force=True)
                page.wait_for_timeout(2000)
                print("已关掉演示数据开关 | 当前 checked =",
                      page.get_by_role("switch").first.is_checked())
        else:
            print("⚠ 没找到演示开关标签, 保持默认")

        inputs = page.locator('input[type="file"]')
        n = inputs.count()
        print(f"文件上传框: {n} 个")
        for i in range(min(n, 2)):
            f = pdf if i == 0 else (zipf if zipf.exists() else None)
            if f is None:
                continue
            inputs.nth(i).set_input_files(str(f))
            print(f"  上传 {f.name} ({f.stat().st_size // 1024} KB)")
            page.wait_for_timeout(1200)

        wait_button(page)
        assert not page.get_by_text("综合复现就绪度").count(), "点之前就有结果面板, 状态不干净"

        # ⚠ 点击方式踩坑记录（Streamlit 1.64, 全部实测）:
        #   ① button[kind="primary"] 的**第一个**是页眉「开始审计」导航, 只做锚点跳转
        #      → 页面完全不动。必须按文本精确定位。
        #   ② .click(force=True) 绕开命中测试派发事件 → Streamlit 不认, 没反应。
        #   ③ JS 手工 dispatchEvent 五件套 → 同样没反应。
        #   ④ **scroll_into_view + page.mouse.click(坐标)** → 有效（2s 出结果）。
        #   另: 等结果别选常驻面板文本, 也别选「准备中…」(整轮几秒, 一闪而过);
        #   实测应等 inner_text 里出现「综合复现就绪度」并逐秒轮询。
        t0 = time.time()
        btn = page.get_by_role("button", name="开 始 科 研 审 计").first
        btn.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        box = btn.bounding_box()
        assert box, "拿不到按钮坐标"
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        print(f"已在 ({box['x']:.0f},{box['y']:.0f}) 派发真实鼠标点击", flush=True)

        ok = False
        for k in range(LIMIT):
            page.wait_for_timeout(1000)
            if "综合复现就绪度" in page.inner_text("body"):
                ok = True
                break
        dt = time.time() - t0
        print(f"\n上传路径出结果: {ok} | 耗时 {dt:.1f}s")

        if ok:
            for probe in ("① 假设站", "条可证伪假设", "怎么推翻它", "风险项"):
                print(f"  [{'ok' if page.get_by_text(probe).count() else 'x'}] {probe}")
        page.screenshot(path=str(SHOTS / "09_upload_timing.png"), full_page=False)
        if errs:
            print("pageerror:", errs[:3])
        b.close()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
