"""真机验证（阶段五）: 界面上打开「复现闭环」后，搜/算/做/迭代回边是否真的出现在结果页。

守的契约:
  A 不开开关 -> 结果页**没有**闭环面板（经典路径不受影响）
  B 打开开关 -> 出现「复现闭环」面板，含评级 / 复算表 / 引证 / 步骤 / 逐轮推演 / 下载入口
  C 打开闭环会自动带起「多智能体对撞」（如实提示，不悄悄改设定）

⚠️ 等渲染一律等**目标文本**（不能等 body 变长）：早先有过三次假通过。
⚠️ Streamlit 的 toggle 要点它的 label，点 input 往往不生效。
⚠️ st.dataframe / download_button 的文字在 DOM 里，但 dataframe 单元格是画布渲染 ——
   所以表格断言只数控件个数，关键结论另有纯文本再说一遍（页面里本来就写了）。
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = 8516
URL = f"http://127.0.0.1:{PORT}"
SAMPLES = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
OUT_DIR = SAMPLES / "ui_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "closeloop_panel.png"
log: list[str] = []


def say(s: str) -> None:
    log.append(s)
    print(s, flush=True)


def _health() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


proc = subprocess.Popen(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-m", "streamlit", "run",
     "app.py", "--server.headless", "true", "--server.address", "127.0.0.1",
     "--server.port", str(PORT)],
    cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

failures: list[str] = []


def check(ok: bool, label: str) -> None:
    assert isinstance(ok, bool), f"[{label}] ok 不是布尔值: {ok!r}"
    assert isinstance(label, str)
    if not ok:
        failures.append(label)
        say(f"  x {label}")
    else:
        say(f"  v {label}")


try:
    from playwright.sync_api import sync_playwright

    ready = False
    for _ in range(45):
        if _health():
            ready = True
            break
        time.sleep(2)
    say(f"服务就绪: {ready}")
    if not ready:
        raise SystemExit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)

        def run_case(closeloop: bool) -> tuple:
            page = browser.new_page(viewport={"width": 1500, "height": 1400})
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(URL, timeout=60000)
            try:
                page.wait_for_selector("text=论文 PDF", timeout=45000)
            except Exception:
                pass
            page.wait_for_timeout(1500)

            if closeloop:
                lab = page.locator("label").filter(has_text="复现闭环")
                if lab.count():
                    lab.first.click()
                    page.wait_for_timeout(2500)
                else:
                    say("  ! 没找到「复现闭环」开关")
                # 打开闭环应自动带起多智能体：子开关「对撞区角色用模型」出现 = 分流生效
                page.wait_for_timeout(1200)
                sub = page.locator("label").filter(has_text="对撞区角色用模型")
                check(sub.count() >= 1,
                      "打开闭环后自动带起多智能体（子开关出现）")
                check("已自动启用" in page.inner_text("body"),
                      "页面如实提示「已自动启用多智能体」")

            page.get_by_text("开 始 科 研 审 计", exact=False).click()
            got = False
            for _ in range(90):
                page.wait_for_timeout(2000)
                if "综合复现就绪度" in page.inner_text("body"):
                    got = True
                    break
            page.wait_for_timeout(3500)
            body = page.inner_text("body") if got else ""
            n_df = page.locator('[data-testid="stDataFrame"]').count()
            n_dl = page.locator('[data-testid="stDownloadButton"]').count()
            if errs:
                say(f"  ! 控制台异常 {len(errs)}: {errs[:2]}")
            if closeloop:
                page.screenshot(path=str(OUT), full_page=True)
            page.close()
            return body, n_df, n_dl

        say("")
        say("=== A. 不打开开关（经典路径）===")
        a, n_df_a, n_dl_a = run_case(closeloop=False)
        check("综合复现就绪度" in a, "结果页出得来")
        # ⚠️ 不能用「复现闭环」这个词断言 —— 开关标签本身就在同一个页面正文里。
        #    必须用**只可能出现在面板里**的文案。
        check("小规模复算与静态推演" not in a,
              "经典路径没有「算」一节（开关真的在起作用）")
        check("复现方案与可复现性评级" not in a, "经典路径没有「做」一节")
        check("迭代回边（对不上" not in a, "经典路径没有迭代回边一节")
        check("离线可跑" not in a, "经典路径没有闭环面板的说明段（面板整体不存在）")

        say("")
        say("=== B. 打开开关（闭环路径）===")
        b, n_df_b, n_dl_b = run_case(closeloop=True)
        check("综合复现就绪度" in b, "结果页出得来")
        check("复现闭环" in b, "出现「复现闭环」面板")
        check("可复现性" in b, "面板显示可复现性评级")
        check("小规模复算与静态推演" in b, "面板有「算」一节")
        check("文献依据检索与引证核验" in b, "面板有「搜」一节")
        check("复现方案与可复现性评级" in b, "面板有「做」一节")
        check("迭代回边" in b, "面板有「迭代回边」一节")
        check("不执行任何训练" in b, "如实声明「算」不执行训练")
        check("不写任何文件" in b, "如实声明迭代回边不写文件")
        check("扣分理由" in b, "评级给出可复核的扣分理由")
        check(n_df_b > n_df_a, "开开关后多出表格（%d -> %d）" % (n_df_a, n_df_b))
        check(n_dl_b >= n_dl_a + 2,
              "面板新增 2 个下载入口（复现方案 / 推演留痕）: %d -> %d" % (n_dl_a, n_dl_b))
        check(("待人工确认" in b) or ("收敛" in b), "面板给出可读的闭环结论")

        say("")
        say("=== C. 面板截图 ===")
        say(f"  截图: {OUT}  （存在: {OUT.exists()}）")

        browser.close()

    say("")
    say("=" * 50)
    if failures:
        say(f"失败 {len(failures)}: {'；'.join(failures)}")
        raise SystemExit(1)
    say("全部通过(verify_closeloop_ui.py)")
finally:
    try:
        proc.terminate()
    except Exception:
        pass
