"""第三套：网页端功能真机验证（Playwright + 真实服务，不 mock）。

覆盖八件事，全部以「页面上真的出现了什么」为判据：
  W1 首屏指挥台      深色大屏渲染出来（七位置 / 四站 / 工序 都在）
  W2 上传区          论文 / 代码 / 实测结果 三个载体入口在
  W3 详情弹窗        点大屏某一块 → 弹窗摊开 → 「收起详情」能关
  W4 弹窗名额        详情弹窗开过、后用 X 关掉，再点「问一问」→ 不许崩
                     （2026-10-31 00:40 就是这里崩的：一轮开两个 st.dialog 会让整页白掉）
  W5 跑一轮审计      点「开始科研审计」→ 顶栏「● 已完成」+ 四站产物 + 就绪度有数
  W6 导航切页        总览 / 参数对表 / 审计发现 / 审计报告 / 执行记录 五页都有真实内容
  W7 问一问          发一个问题 → 出现答复（模型作答 或 审计事实）
  W8 返回入口        非指挥台页面上有「← 返回指挥台」，点了能回去

用法：
    .venv\\Scripts\\python.exe -X utf8 probes\\probe_web_suite.py [端口]
    （不带端口就自己起一个 8501，跑完自己关）
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright.sync_api import sync_playwright          # noqa: E402

from _serve import ensure_server, stop_server            # noqa: E402

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)
PORT = sys.argv[1] if len(sys.argv) > 1 else ""
DIALOG = 'div[data-testid="stDialog"]'
CRASH = ("StreamlitInvalidLayoutContextError", "Only one dialog",
         "StreamlitAPIException", "Traceback (most recent call last)")

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


def body(pg) -> str:
    try:
        return pg.inner_text("body")
    except Exception:            # noqa: BLE001
        return ""


def crashed(pg) -> str:
    t = body(pg)
    for c in CRASH:
        if c in t:
            return c
    return ""


def _qa_open(pg) -> bool:
    """问答弹窗是否还开着。

    ⚠ 不能拿「有没有 stDialog 元素」当判据：关掉问答之后，**详情弹窗会按设计
      自动摊开**（名额让出来了、焦点还在），页面照样有一个 dialog。
      2026-10-31 就因为这条把「详情报到」误判成「问答关不掉」。
    """
    if pg.locator(DIALOG).count() == 0:
        return False
    try:
        head = pg.locator(DIALOG).first.inner_text()[:80]
    except Exception:            # noqa: BLE001
        return True
    return ("问 一 问" in head) or ("问一问" in head) or ("深入答疑" in head)


def main() -> int:
    proc = ensure_server(PORT or "8501")
    port = PORT or "8501"
    url = f"http://127.0.0.1:{port}"
    print("=" * 74)
    print(f"第三套 · 网页端真机验证   {url}")
    print("=" * 74)

    try:
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(viewport={"width": 1680, "height": 1150})
            pg.goto(url, wait_until="domcontentloaded", timeout=60000)
            pg.wait_for_selector("text=上传论文资料", timeout=90000)
            pg.wait_for_timeout(2000)

            # ---------------------------------------------------------- W1 首屏
            print("\nW1 首屏指挥台")
            t = body(pg)
            check("材料" in t and ("四站" in t or "① " in t), "大屏三栏在（材料 / 四站 / 工序）")
            check("七位置" in t or "位置" in t, "七位置那一栏在")
            check("就绪度" in t or "待命" in t or "未载入" in t, "空态有如实说明（未载入/待命）")
            pg.screenshot(path=str(OUT / "web_01_entry.png"))

            # ---------------------------------------------------------- W2 上传区
            print("\nW2 上传区")
            check(pg.locator("text=上传论文资料").count() > 0, "「上传论文资料」标题在")
            check("代码" in t, "代码载体入口在")

            # ---------------------------------------------------------- W3 详情弹窗
            print("\nW3 详情弹窗")
            try:
                pg.locator("[class*='st-key-dkhot_pos_moderator'] button").first.click(force=True)
                pg.wait_for_selector(DIALOG, timeout=20000)
                pg.wait_for_timeout(1500)
                dt = pg.locator(DIALOG).inner_text()
                check(len(dt.strip()) > 40, "详情内容摊开了", f"{len(dt)} 字")
                pg.screenshot(path=str(OUT / "web_02_detail.png"))
                # 「收起详情」清焦点
                pg.get_by_role("button", name="收起详情").first.click()
                pg.wait_for_timeout(1800)
                closed = pg.locator(DIALOG).count() == 0
                check(closed, "「收起详情」能关掉弹窗")
            except Exception as ex:      # noqa: BLE001
                check(False, "详情弹窗打不开", f"{type(ex).__name__}: {ex}")

            # ---------------------------------------------------------- W4 弹窗名额（崩溃点）
            print("\nW4 弹窗名额（2026-10-31 崩溃点）")
            try:
                pg.locator("[class*='st-key-dkhot_pos_moderator'] button").first.click(force=True)
                pg.wait_for_selector(DIALOG, timeout=20000)
                pg.wait_for_timeout(1200)
                # 用右上角 X 关（**不清焦点** —— 这正是会诱发"两个弹窗同开"的路径）
                try:
                    pg.locator(f"{DIALOG} button[aria-label='Close']").first.click()
                except Exception:        # noqa: BLE001
                    pg.keyboard.press("Escape")
                pg.wait_for_timeout(1500)
                # 此时焦点还在 → 再点问一问
                pg.get_by_role("button", name="问 一 问").first.click()
                pg.wait_for_timeout(2500)
                bad = crashed(pg)
                check(not bad, "详情开过之后再开问答，页面没崩", bad or "无异常")
                check(pg.locator(DIALOG).count() > 0, "问答弹窗正常打开")
                pg.screenshot(path=str(OUT / "web_03_qa_after_detail.png"))
                # 先用右上角 X 关一次 —— 这一步专门验 2026-10-31 修的 on_dismiss：
                # 改之前 X 关完会立刻自己重开，窗关不掉、页面全被盖住。
                try:
                    pg.locator(f"{DIALOG} button[aria-label='Close']").first.click()
                    # ⚠ 关窗后**必须轮询等待**：回调触发的是整页重跑，这页本身很重
                    #   （大屏 + 审计结果），一次重跑 3~8 秒都正常。原来只等 2 秒就断言，
                    #   会把「慢」判成「关不掉」（2026-10-31 本轮就误判了一次）。
                    gone = False
                    t_x = time.time()
                    while time.time() - t_x < 20:
                        if not _qa_open(pg):
                            gone = True
                            break
                        pg.wait_for_timeout(700)
                    check(gone, "右上角 X 能真正关掉问答（on_dismiss 生效）",
                          f"{time.time() - t_x:.1f}s")
                except Exception as ex:      # noqa: BLE001
                    check(False, "X 关窗失败", f"{type(ex).__name__}")

                # 关掉任何还开着的弹窗，别让它挡住后面的按钮
                for _ in range(3):
                    if pg.locator(DIALOG).count() == 0:
                        break
                    try:
                        if pg.get_by_role("button", name="收起详情").count():
                            pg.get_by_role("button", name="收起详情").first.click()
                        elif pg.locator(".st-key-qa_close button").count():
                            pg.locator(".st-key-qa_close button").first.click()
                        else:
                            pg.locator(f"{DIALOG} button[aria-label='Close']").first.click()
                    except Exception:        # noqa: BLE001
                        pass
                    pg.wait_for_timeout(1800)
                check(pg.locator(DIALOG).count() == 0, "问答弹窗已关闭")
            except Exception as ex:      # noqa: BLE001
                check(False, "弹窗名额场景没能走完", f"{type(ex).__name__}: {ex}")

            # ---------------------------------------------------------- W5 跑审计
            print("\nW5 跑一轮审计（本机模型）")
            t0 = time.time()
            try:
                pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
                pg.wait_for_selector("text=● 已完成", timeout=900000)
                dt = time.time() - t0
                pg.wait_for_timeout(3000)
                t = body(pg)
                check(True, "跑到「● 已完成」", f"{dt:.0f}s")
                check("就绪度" in t, "首屏出现就绪度")
                check(re.search(r"\d+\s*条", t) is not None, "判定条数有真实数字")
                for k in ("假设", "实验", "判定"):
                    check(k in t, f"四站里的「{k}」在页面上出现")
                check(not crashed(pg), "跑完没有异常栈")
                pg.screenshot(path=str(OUT / "web_04_audited.png"))
            except Exception as ex:      # noqa: BLE001
                check(False, "审计没能跑完", f"{type(ex).__name__}: {str(ex)[:120]}")
                pg.screenshot(path=str(OUT / "web_04_audited_fail.png"))

            # ---------------------------------------------------------- W6 导航切页
            print("\nW6 导航切页")
            # ⚠ 两个前置，缺一个就会假失败（2026-10-31 第三轮就是这么挂的）：
            #   ① 先把可能还开着的弹窗关干净（有遮罩时点不到侧栏，报错却只写"等元素"）；
            #   ② 等侧栏导航真的渲染出来（跑完审计那一刻页面在连续重跑，DOM 一直在换）。
            for _ in range(3):
                if pg.locator(DIALOG).count() == 0:
                    break
                try:
                    if pg.get_by_role("button", name="收起详情").count():
                        pg.get_by_role("button", name="收起详情").first.click()
                    elif pg.locator(".st-key-qa_close button").count():
                        pg.locator(".st-key-qa_close button").first.click()
                    else:
                        pg.locator(f"{DIALOG} button[aria-label='Close']").first.click()
                except Exception:            # noqa: BLE001
                    pass
                pg.wait_for_timeout(1500)
            _nav = 'section[data-testid="stSidebar"] [role="radiogroup"] label'
            try:
                pg.wait_for_selector(_nav, timeout=60000)
            except Exception:                # noqa: BLE001
                pass
            check(pg.locator(_nav).count() >= 2, "侧栏导航渲染出来了",
                  f"{pg.locator(_nav).count()} 项")
            for label, must in (("审计总览", "提取到可比对项"),
                                ("参数对表", "参数"),
                                ("审计发现", "发现"),
                                ("审计报告", "报告"),
                                ("执行记录", "执行")):
                ok = False
                for _try in range(2):
                    try:
                        pg.locator(_nav, has_text=label).first.scroll_into_view_if_needed()
                        pg.locator(_nav, has_text=label).first.click(timeout=15000)
                        pg.wait_for_timeout(2200)
                        t = body(pg)
                        ok = len(t) > 800 and must in t
                        if ok:
                            break
                    except Exception:        # noqa: BLE001
                        pg.wait_for_timeout(1200)
                check(ok, f"「{label}」页有真实内容",
                      f"{len(body(pg))} 字" if ok else "两次尝试都没进去")
            pg.screenshot(path=str(OUT / "web_05_report_page.png"))

            # ---------------------------------------------------------- W8 返回入口（先于问答，避免弹窗遮挡）
            print("\nW8 返回指挥台")
            try:
                btn = pg.get_by_role("button", name=re.compile("返回指挥台"))
                check(btn.count() > 0, "非指挥台页面上有「返回指挥台」")
                btn.first.click()
                pg.wait_for_timeout(2200)
                check("逐条对表" in body(pg) or "材料和" in body(pg) or
                      pg.locator("[class*='st-key-dkhot_']").count() > 0,
                      "点了能回到指挥台")
            except Exception as ex:      # noqa: BLE001
                check(False, "返回入口不可用", f"{type(ex).__name__}")

            # ---------------------------------------------------------- W7 问一问
            print("\nW7 问一问（本机模型作答）")
            try:
                pg.get_by_role("button", name="问 一 问").first.click()
                pg.wait_for_selector(DIALOG, timeout=20000)
                pg.wait_for_timeout(1500)
                pg.locator(f"{DIALOG} textarea").first.fill("这份材料复现最先该改哪一处？")
                pg.get_by_role("button", name=re.compile("发送")).first.click()
                t0 = time.time()
                got = False
                while time.time() - t0 < 420:
                    t = pg.locator(DIALOG).inner_text()
                    if "模型作答" in t or "直接检索到的审计事实" in t:
                        got = True
                        break
                    if "调用失败" in t:
                        break
                    pg.wait_for_timeout(3000)
                dt = time.time() - t0
                t = pg.locator(DIALOG).inner_text()
                check(got, "问答给出了答复", f"{dt:.0f}s")
                check("调用失败" not in t, "不是「模型调用失败」")
                check(len(t) > 200, "答复有实质长度", f"{len(t)} 字")
                pg.screenshot(path=str(OUT / "web_06_qa.png"))
            except Exception as ex:      # noqa: BLE001
                check(False, "问一问走不通", f"{type(ex).__name__}: {str(ex)[:120]}")
                pg.screenshot(path=str(OUT / "web_06_qa_fail.png"))

            b.close()
    finally:
        stop_server(proc)

    print("\n" + "=" * 74)
    ok = N - len(BAD)
    print(f"共 {N} 项：通过 {ok}，未通过 {len(BAD)}")
    for x in BAD:
        print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
