"""单点验：问答弹窗用右上角 X 关，能真正关掉吗（on_dismiss 生效）。

为什么单独一个脚本
------------------
这件事塞在 `probe_web_suite.py` 里验，一次要连带跑完整轮审计（4 分钟）。
它就是「关窗」这一个动作，两分钟能跑完，出问题也只在这一件事上。

判据（两条都要）：
  ① 点 X 之后，弹窗元素消失（**轮询等待**，重跑本来就慢，别把慢判成关不掉）；
  ② 关掉之后再点一次「问 一 问」能重新打开 —— 说明开关是真的落回 False，
     而不是靠"页面重画时碰巧没画"糊过去。

用法：
    .venv\\Scripts\\python.exe -X utf8 probes\\probe_qa_xclose.py [端口]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright          # noqa: E402

from _serve import ensure_server, stop_server            # noqa: E402

PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
DIALOG = 'div[data-testid="stDialog"]'
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")

N = 0
BAD: list[str] = []


def check(ok, label: str, note: str = "") -> bool:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label + (f"   [{note}]" if note else ""), flush=True)
    if not ok:
        BAD.append(label)
    return ok


def main() -> int:
    proc = ensure_server(PORT)
    print("=" * 66)
    print("问一问弹窗 · 右上角 X 关窗验证")
    print("=" * 66)
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(viewport={"width": 1600, "height": 1000})
            pg.goto(f"http://127.0.0.1:{PORT}", wait_until="domcontentloaded", timeout=60000)
            pg.wait_for_selector("text=上传论文资料", timeout=90000)
            pg.wait_for_timeout(1500)

            pg.get_by_role("button", name="问 一 问").first.click()
            pg.wait_for_selector(DIALOG, timeout=20000)
            pg.wait_for_timeout(1200)
            check(pg.locator(DIALOG).count() > 0, "问答弹窗能打开")

            t0 = time.time()
            pg.locator(f"{DIALOG} button[aria-label='Close']").first.click()
            gone = False
            while time.time() - t0 < 25:
                if pg.locator(DIALOG).count() == 0:
                    gone = True
                    break
                pg.wait_for_timeout(700)
            check(gone, "点 X 后弹窗消失", f"{time.time() - t0:.1f}s")
            if not gone:
                pg.screenshot(path=str(OUT / "qa_xclose_fail.png"))

            # 再点一次：开关真的落回 False 才能重新打开
            try:
                pg.get_by_role("button", name="问 一 问").first.click()
                pg.wait_for_selector(DIALOG, timeout=20000)
                pg.wait_for_timeout(1000)
                check(pg.locator(DIALOG).count() > 0, "关掉之后还能重新打开")
                # 用「✕ 结束问答」收尾
                if pg.locator(".st-key-qa_close button").count():
                    pg.locator(".st-key-qa_close button").first.click()
                    pg.wait_for_timeout(2500)
                check(pg.locator(DIALOG).count() == 0, "「✕ 结束问答」也能关掉")
            except Exception as ex:        # noqa: BLE001
                check(False, "重新打开失败", f"{type(ex).__name__}")

            pg.screenshot(path=str(OUT / "qa_xclose_ok.png"))
            b.close()
    finally:
        stop_server(proc)

    print("\n" + "=" * 66)
    print(f"共 {N} 项：通过 {N - len(BAD)}，未通过 {len(BAD)}")
    for x in BAD:
        print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
