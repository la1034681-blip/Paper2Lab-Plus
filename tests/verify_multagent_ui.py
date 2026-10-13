"""真机验证（阶段四）: 界面上打开「多智能体对撞」开关后，对撞区是否真的出现在结果页。

守的契约:
  A 不开开关 -> 结果页**没有**对撞区面板（经典路径不受影响）
  B 开开关   -> 出现「多智能体对撞区」，含角色留痕 / 双路核对 / 仲裁裁决 / 红队笔记

⚠️ 等渲染一律等**目标文本**（不能等 body 变长）：早先有过三次假通过。
⚠️ Streamlit 的 toggle 要点它的 label，点 input 往往不生效。
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = 8514
URL = f"http://127.0.0.1:{PORT}"
# 截图落到桌面测试集（与其它 UI 脚本一致），不要落在工程根目录里
SAMPLES = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
OUT_DIR = SAMPLES / "ui_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "multagent_panel.png"
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

        def run_case(multagent: bool) -> tuple[str, int]:
            page = browser.new_page(viewport={"width": 1500, "height": 1400})
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(URL, timeout=60000)
            # 等首屏真的渲染出来，不要定时长
            try:
                page.wait_for_selector("text=论文 PDF", timeout=45000)
            except Exception:
                pass
            page.wait_for_timeout(1500)

            if multagent:
                lab = page.locator("label").filter(has_text="多智能体对撞")
                if lab.count():
                    lab.first.click()
                    page.wait_for_timeout(2500)
                else:
                    say("  ! 没找到「多智能体对撞」开关")

            page.get_by_text("开 始 科 研 审 计", exact=False).click()
            # 等目标文本（结果页标志），不用"等 body 变长"
            got = False
            for _ in range(90):
                page.wait_for_timeout(2000)
                if "综合复现就绪度" in page.inner_text("body"):
                    got = True
                    break
            page.wait_for_timeout(3000)
            body = page.inner_text("body") if got else ""
            # ⚠️ Streamlit 的 st.dataframe 是画布渲染，**文字不在 inner_text 里**，
            #    所以表格内容只能靠"表格控件数量"来断言，不能靠抓文本。
            n_df = page.locator('[data-testid="stDataFrame"]').count()
            if errs:
                say(f"  ! 控制台异常 {len(errs)}: {errs[:2]}")
            if multagent:
                page.screenshot(path=str(OUT), full_page=True)
            page.close()
            return body, n_df

        say("")
        say("=== A. 不打开开关（经典路径）===")
        a, n_df_a = run_case(multagent=False)
        check("综合复现就绪度" in a, "结果页出得来")
        check("多智能体对撞区" not in a, "经典路径没有对撞区面板（开关真的在起作用）")

        say("")
        say("=== B. 打开开关（多智能体路径）===")
        b, n_df_b = run_case(multagent=True)
        check("综合复现就绪度" in b, "结果页出得来")
        check("多智能体对撞区" in b, "出现「多智能体对撞区」面板")
        check("角色执行留痕" in b, "面板里有角色执行留痕")
        check(n_df_b >= 4, "面板里有 4 张表（角色留痕/双路核对/仲裁/红队），实测 %d 张" % n_df_b)
        check(n_df_b > n_df_a, "开开关比不开多出表格（经典 %d -> 多智能体 %d）"
              % (n_df_a, n_df_b))
        check(("分歧参数：" in b) or ("结论被互证" in b),
              "面板给出可读的核对结论（分歧点名 / 互证确认）")
        check("隔离在代码层面强制" in b, "如实说明了上下文隔离的实现方式")
        check("纯规则、不调模型" in b, "如实说明了可复现性来源")
        check("红队笔记" in b, "面板里有红队笔记")
        check("仲裁裁决" in b, "面板里有仲裁裁决")

        say("")
        say("=== C. 面板截图 ===")
        say(f"  截图: {OUT}  （存在: {OUT.exists()}）")

        browser.close()

    say("")
    say("=" * 50)
    if failures:
        say(f"失败 {len(failures)}: {'；'.join(failures)}")
        raise SystemExit(1)
    say("全部通过(verify_multagent_ui.py)")
finally:
    try:
        proc.terminate()
    except Exception:
        pass
