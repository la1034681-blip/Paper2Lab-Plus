"""真机验证：「影响量级」这块在网页上真的显示出来了（阶段二 2E · ③④）。

为什么单独做这一层：
  `verify_impact.py` 验的是内存里的字段与报告文字；
  界面渲染是另一条路（`app.py::_render_impact` → `_impact_card` → st.html）。
  历史教训：组件一旦抛异常，**整页往下全不渲染**（2026-10-24 st.dataframe 混类型）。
  所以"报告里有"不等于"页面上看得见"—— 必须在真浏览器里看一遍。

判据（都用**界面文案**，不用报告章节名 —— 2026-10-24 踩过"拿报告文案断言页面"）：
  · 出现「① 假设站 · 算（影响量级）」
  · 有候选解释、有「证据最硬」、有翻盘 / 边界 / 对照
  · 明写「提示层」（不许被当成结论）
  · 更靠后的区块仍在（反证没被吞）

用法: python tests/verify_impact_ui.py
退出码: 0 全过 / 1 有异常（供 run_all_checks 判定）
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = 8531
URL = f"http://127.0.0.1:{PORT}"
OUT_DIR = ROOT / ".trash_tmp"
OUT_DIR.mkdir(exist_ok=True)
SHOT = OUT_DIR / "_impact_ui.png"
BODY = OUT_DIR / "_impact_ui.txt"

errors: list[str] = []


def say(s: str) -> None:
    print(s, flush=True)


def check(ok, label: str) -> None:
    ok = bool(ok)          # `and` 会返回操作数, 一律 bool 收口（防呆）
    say(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok:
        errors.append(label)


def goto_nav(page, label: str) -> bool:
    """2026-10-26 界面改成「左侧导航 + 分页」后新增：影响量级在「假设卡」页，
    跑完默认落在「审计总览」页 ⇒ 断言前必须点过去，否则就是拿总览页的文本去判红。

    为什么不 get_by_text：导航文案与页内标题同名时会点错，报 outside of viewport。
    只点侧栏 radiogroup 里的 label 才稳。
    """
    try:
        lab = page.locator('[data-testid="stSidebarUserContent"] [role="radiogroup"] label, [data-testid="stSidebar"] [role="radiogroup"] label'
                           ).filter(has_text=label)
        if not lab.count():
            return False
        lab.first.scroll_into_view_if_needed()
        page.wait_for_timeout(300)
        lab.first.click()
        page.wait_for_timeout(2500)
        return True
    except Exception:
        return False


def main() -> int:
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-m", "streamlit",
         "run", "app.py", "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        up = False
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3) as r:
                    if r.status == 200:
                        up = True
                        break
            except Exception:
                time.sleep(1)
        say(f"## 服务启动: {'✅' if up else '❌'}")
        if not up:
            return 1

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch(channel="msedge", headless=True)
            page = br.new_page(viewport={"width": 1500, "height": 1300})
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(URL, timeout=60000)
            page.wait_for_timeout(9000)

            # 用内置演示数据（默认开着）—— demo 材料本来就算得出 1 条影响。
            # 上传优先级低于这个开关，所以这里不碰它，免得测的根本不是演示材料。
            page.get_by_text("开 始 科 研 审 计", exact=False).click()

            body = ""
            t0 = time.time()
            # 等两个关键块**同时**出现再开始断言。
            # 2026-10-26 踩过：只等「执行记录」会在流式渲染中途就放行 ——
            # 那一次体检里这一项偶发 exit=1（19.6s，正常 6.4s），单独重跑三次全过。
            # 根因是等待条件太乐观，不是功能坏；所以改等待，不改判据。
            # 2026-10-26 二次改：界面变分页后，等「执行记录 + 影响量级」会永远等不到
            # （一个在「执行记录」页、一个在「假设卡」页）⇒ 先等结果页立住，再切页。
            for _ in range(60):
                page.wait_for_timeout(2000)
                body = page.inner_text("body")
                if "复现就绪度" in body:
                    break
            _moved = goto_nav(page, "假设卡")
            say(f"## 切到假设卡页: {'✅' if _moved else '❌'}　"
                f"用时 {time.time() - t0:.1f}s")
            # 等渲染稳定（连续 4 次长度不变，且至少等满 4s）——
            # 卡片是先出标题再出内容的，稳定判据太松会拿半成品去断言。
            last, same = -1, 0
            for _ in range(30):
                page.wait_for_timeout(500)
                n = len(page.inner_text("body"))
                same = same + 1 if n == last else 0
                last = n
                if same >= 4:
                    break
            body = page.inner_text("body")
            BODY.write_text(body, encoding="utf-8")
            page.screenshot(path=str(SHOT), full_page=True)
            say(f"## 出结果: 用时 {time.time() - t0:.1f}s　body {len(body)} 字符")

            check("假设站 · 算（影响量级）" in body, "页面出现「① 假设站 · 算（影响量级）」")
            check("候选解释" in body, "列出候选解释")
            check("证据最硬" in body, "明写「证据最硬」（不是「最优」）")
            check("翻盘" in body, "有翻盘算")
            check("边界" in body, "有边界算")
            check("对照" in body, "有对照算")
            check("提示层" in body, "明写这是提示层")
            check("不进判定" in body, "明写不进判定")
            # 「最优」只许以**否定形式**出现（…不是「最优」）——
            # 我们自己就写了这句, 于是简单查"有没有"会假红。改成数次数对齐。
            n_all, n_neg = body.count("最优"), body.count("不是「最优」")
            check(n_all == n_neg and n_all >= 1,
                  "「最优」只以否定形式出现（全 %d 处 / 否定 %d 处）" % (n_all, n_neg))
            # 组件抛错会让整页往下全不渲染 —— 分页后改成「换个页也照样渲染」来反证。
            # ⚠ 2026-10-26：原来判「执行记录」这四个字，改分页后导航里就有它，
            #   永远为真 ⇒ 变成假绿。改判切页后目标页的专属标题。
            _moved2 = goto_nav(page, "实验卡")
            _body2 = page.inner_text("body")
            check(_moved2 and "② 实验站" in _body2,
                  "切到实验卡页仍正常渲染（整页没有被吞掉）")
            check(not errs, f"页面无脚本报错（实际 {len(errs)} 条）")
            if errs:
                for e in errs[:3]:
                    say(f"   err: {e[:160]}")

            br.close()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass

    say("")
    if errors:
        say(f"结论：❌ {len(errors)} 项异常")
        for e in errors:
            say(f"  - {e}")
        return 1
    say("结论：✅ 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
