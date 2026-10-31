"""真机验证「化学身份核验」在网页上真的显示出来了（2026-10-24）。

为什么单独做这一层：
  `verify_chem.py` 验的是模块与报告文字，都是"内存里的东西"；
  但界面渲染是另一条路（`app.py::_render_chemistry` + `_table()` 过一遭）。
  历史教训：组件一旦抛异常，**整页往下全不渲染**（st.dataframe 混类型）。
  所以"报告里有化学章节"不等于"页面上看得见"—— 必须在真浏览器里点一遍。

判据（必须是"只有化学论文才有"的内容，否则旧 DOM 会假通过）：
  · 「化学身份核验」标题出现
  · 页面出现 `7732-18-6`（那行 CAS 校验位错）且判为「本身不合规」
  · 页面出现 `C8H10N4O2` 且判为「对不上」
  · 页面**没有**残留 `mimo` 字样（2026-10-24 已下线，留着说明改漏了）

用法: python tests/verify_chem_ui.py
退出码: 0 全过 / 1 有异常（供 run_all_checks 判定）
"""
from __future__ import annotations
import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PORT = 8527
URL = f"http://127.0.0.1:{PORT}"
# ⚠ 2026-10-26：指挥台上线后**默认落在「指挥台」页**，而化学核验渲染在「① 假设站」页。
#   用 `?view=hyp` 直接落到那一页 —— 界面本来就支持这个参数（也是分享链接用的），
#   比在测试里模拟点侧栏稳，不受渲染时序影响。
URL_VIEW = URL + "?view=hyp"
PDF = ROOT / "demo" / "chem_demo.pdf"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

log: list[str] = []
errors: list[str] = []


def note(s: str) -> None:
    print(s, flush=True)
    log.append(s)


def check(ok, label: str) -> None:
    # 防呆：`and` 会返回操作数而不是 bool，一律 bool() 收口
    ok = bool(ok)
    note(f"  [{'OK ' if ok else 'FAIL'}] {label}")
    if not ok:
        errors.append(label)


def wait_text(page, text: str, timeout: int = 60000) -> bool:
    try:
        page.wait_for_selector(f"text={text}", timeout=timeout)
        return True
    except Exception:
        return False


def start_button(page):
    """找到**正文**那个「开始科研审计」。

    ⚠️ 别用 `button[kind="primary"].first` —— 页眉右上角也有一个 primary 按钮，
    排在它前面；点页眉那个不会触发审计（2026-10-24 实测：点了 180s 没动静，
    截图里流水线全是「未执行」，极易误判成"化学章节没渲染"）。
    """
    for loc in (page.get_by_role("button", name=re.compile(r"开\s*始\s*科\s*研\s*审\s*计")),
                page.get_by_text("开始科研审计", exact=False)):
        if loc.count():
            return loc.first
    loc = page.locator('button[kind="primary"]')
    return loc.last if loc.count() else None


def goto_nav(page, label: str) -> bool:
    """2026-10-26 界面改成「左侧导航 + 分页」后新增：内容不在第一屏了，得先点过去。

    为什么不用 get_by_text：导航文案和页内标题可能同名（「审计报告」既是导航项
    也是下载页标题），get_by_text 会点到不可见的那个，报 "outside of the viewport"。
    专点侧栏里的 radiogroup label，是唯一稳定的落点。

    ⚠ 2026-10-26 二改：固定等 2.5s 会点空 —— 侧栏是结果渲染完之后才挂上去的，
    结果刚出现时导航还没进页面。必须**等导航容器真的出现**再找。
    """
    try:
        page.wait_for_selector('[data-testid="stSidebarUserContent"] [role="radiogroup"], [data-testid="stSidebar"] [role="radiogroup"]',
                               timeout=30000)
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


def click_real(page, el) -> bool:
    """滚到可见再用**坐标**点。Streamlit 的按钮常被浮层/吸顶栏盖住，
    直接 .click() 会点到别的东西（且不报错）。"""
    try:
        el.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        box = el.bounding_box()
        if not box:
            return False
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        return True
    except Exception:
        return False


def main() -> int:
    if not PDF.exists():
        note(f"❌ 样例缺失: {PDF}")
        return 1

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
        note(f"## 服务启动: {'✅' if up else '❌'}")
        if not up:
            return 1

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            br = p.chromium.launch(channel="msedge", headless=True)
            page = br.new_page(viewport={"width": 1500, "height": 1200})
            console_errs: list[str] = []
            page.on("console", lambda m: console_errs.append(m.text)
                    if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errs.append(str(e)))
            page.goto(URL_VIEW, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)

            # 上传化学样例（只有一个 PDF，没有代码包 —— 顺带验证"无代码也能跑"）
            inputs = page.locator("input[type=file]")
            note(f"## 上传控件: {inputs.count()} 个")
            if not inputs.count():
                errors.append("没找到上传控件")
                br.close()
                return 1
            inputs.nth(0).set_input_files(str(PDF))
            page.wait_for_timeout(2500)

            btn = start_button(page)
            check(btn is not None, "找到主按钮")
            if btn is None:
                br.close()
                return 1

            t0 = time.time()
            clicked = click_real(page, btn)
            check(clicked, "点击主按钮（坐标点击）")
            # 判据用**页面上真实存在的文案**（不是报告里的章节名）：
            #   卡片标题是「🧪 ① 假设站 · 化学身份核验」，副标题「核验 N 条化学身份（…）」。
            # 2026-10-24 踩过：探针按「化学身份核验」找，页面其实写的是「读（化学）」，
            # 于是功能明明在、断言却全红 —— 断言必须对着界面文案，不能对着报告文案。
            # 2026-10-25 起化学核验拆成独立工具，卡片标题随之改成「化学身份核验」；
            # 这里判「假设站 · 化学身份核验」（报告里没有这个组合，天然防假通过）。
            # 2026-10-26 界面改成「左侧导航 + 分页」：跑完默认落在「审计总览」页，
            # 化学核验在「假设卡」页 ⇒ 断言前必须先点导航过去（否则误判成"没产出"）。
            # ⚠ 2026-10-26：本脚本用 `?view=hyp` 一进来就在「假设卡」页，
            #    而「综合复现就绪度」那一块只在「指挥台 / 总览」页 —— 等它会一直等到超时。
            #    改用「当前审计对象」：它在**每一页**顶部都出现，且只在出结果之后才出现。
            _ok_total = wait_text(page, "当前审计对象", 300000)
            note(f"## 出结果: {'✅' if _ok_total else '❌'} 用时 {time.time() - t0:.1f}s")
            _moved = goto_nav(page, "假设卡")
            note(f"## 切到假设卡页: {'已在本页' if _ok_total and not _moved else ('✅' if _moved else '❌')}")
            ok = wait_text(page, "假设站 · 化学身份核验", 60000)
            # 等渲染稳定（连续 3 次 body 长度不变）—— 卡片是先出标题再出表格的，
            # 早一步取 body 会只看到「核验 N 条」而没有具体条目，误判成"没抽到"。
            last, same = -1, 0
            for _ in range(40):
                page.wait_for_timeout(500)
                n = len(page.inner_text("body"))
                same = same + 1 if n == last else 0
                last = n
                if same >= 3:
                    break
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / "10_chem_section.png"), full_page=True)

            body = page.inner_text("body")
            # 留一份现场快照，出问题能直接看页面到底写了什么（不用再跑一遍）
            (OUT.parent / "body_chem_ui.txt").write_text(body, encoding="utf-8")
            check(ok, "页面出现化学卡片（① 假设站 · 化学身份核验）")
            check("核验" in body and "条化学身份" in body, "卡片标题报出核验条数")
            check("7732-18-6" in body, "列出 CAS 号 7732-18-6")
            check("本身不合规" in body, "CAS 校验位错被判为「本身不合规」")
            check("C8H10N4O2" in body, "列出咖啡因分子式 C8H10N4O2")
            check("对不上" in body, "分子量错被判为「对不上」")
            check("194.19" in body or "194.194" in body, "给出算出来的数（194.19x）")
            check("化学信息没问题" in body or "猜" in body,
                  "页面写明了边界（不猜结构 / 抓到才算数）")
            check("mimo" not in body.lower(), "页面不再出现已下线的 mimo")

            # 组件抛错会让整页往下的内容全不渲染 —— 用"更靠后的区块还在"反证没被吞。
            # ⚠ 文案里不要出现「异常」二字：全套监控把关键词命中当警示，
            #   断言标题自己写「异常」会被误报成 ⚠️（2026-10-24 踩过，改文案比加白名单干净）。
            # ⚠ 2026-10-26：原来判「审计报告」二字，改成分页后导航里就有这四个字，
            #   永远为真 ⇒ 变成假绿。改判同一页里更靠后的「影响量级」卡片。
            check("影响量级" in body, "后面的区块仍在（整页没有被吞掉）")
            check(not console_errs, f"控制台无错误（实际 {len(console_errs)} 条）")
            if console_errs:
                for e in console_errs[:3]:
                    note(f"   console: {e[:160]}")

            br.close()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            pass

    note("")
    if errors:
        note(f"结论：❌ {len(errors)} 项异常")
        for e in errors:
            note(f"  - {e}")
        (OUT.parent / "chem_ui_report.md").write_text("\n".join(log), encoding="utf-8")
        return 1
    note("结论：✅ 全部通过（化学身份核验在真机上可见）")
    (OUT.parent / "chem_ui_report.md").write_text("\n".join(log), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
