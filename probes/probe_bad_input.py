"""异常输入与空材料两条兜底路径的真机验收（2026-10-28 本轮新增）。

① 坏文件：上传一个内容不是 PDF 的 .pdf（乱码），页面必须给出可读错误，
   且**整页不能变白/卡死**（Streamlit 组件抛异常会让下方整块不渲染）。
② 空材料：什么都不传直接点开始 —— 按现在的产品口径应当走内置样例，
   且页面上要提前写明「本次用内置样例」，不能悄悄换成别的东西。
"""
from __future__ import annotations

import atexit
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from _serve import ensure_server, stop_server

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab_本轮测试截图")
OUT.mkdir(parents=True, exist_ok=True)
PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"
# 2026-10-29 补：以前默认"8501 上一定有人跑着服务"，一旦上一个界面检查先失败，
# 这个探针就报 ERR_CONNECTION_REFUSED，看着像界面崩了。现在没有就自己起、跑完自己关
# （别人起的服务不动，见 probes/_serve.py）。
_SRV = ensure_server(PORT)
atexit.register(stop_server, _SRV)
ok = 0
bad = 0


def check(cond, label, extra=""):
    global ok, bad
    assert isinstance(cond, bool), label
    if cond:
        ok += 1
        print(f"  ✅ {label}{(' — ' + str(extra)) if extra else ''}")
    else:
        bad += 1
        print(f"  ❌ {label}{(' — ' + str(extra)) if extra else ''}")


bad_pdf = Path(tempfile.gettempdir()) / "p2l_broken.pdf"
bad_pdf.write_bytes(b"%NOT-A-PDF\n" + b"garbage" * 200)

with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)

    # ---------- ① 坏文件 ----------
    pg = b.new_page(viewport={"width": 1600, "height": 1100})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=上传论文资料", timeout=60000)
    # ⚠ 2026-10-29 修: 文案出现时上传控件可能还没挂到 DOM（实测读到 0 个，
    # 紧接着 set_input_files 又成功）—— 先等控件本身，再计数。
    try:
        pg.wait_for_selector('input[type="file"]', timeout=30000)
    except Exception:                       # noqa: BLE001
        pass                                # 真没有的话下面那条断言会报出来
    inputs = pg.locator('input[type="file"]')
    for _ in range(10):                     # 兜底轮询，别把渲染时序当成缺陷
        if inputs.count():
            break
        pg.wait_for_timeout(500)
    check(inputs.count() >= 1, "上传控件存在", f"{inputs.count()} 个")
    inputs.nth(0).set_input_files(str(bad_pdf))
    pg.wait_for_timeout(1800)
    pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
    # ⚠ 2026-10-28 踩到：原来只要页面上出现「PDF」就判定"提示到了" ——
    #   上传区标签本来就有 PDF 两个字，于是 5 秒就往下走，正好赶上还在跑，
    #   结论自然是「没提示」。判据必须等**真正的完成锚点**：
    #   要么出现结论性提示，要么顶栏回到「待运行 / 已完成」。
    found = ""
    for _ in range(60):
        pg.wait_for_timeout(1500)
        t = pg.inner_text("body")
        if "没有产出审计结论" in t:
            found = "没有产出审计结论"
            break
        badge = pg.locator(".dk-top").inner_text() if pg.locator(".dk-top").count() else ""
        if ("待运行" in badge or "已完成" in badge) and "已走 0 / 15 格" in t:
            break                      # 跑完了但没给结论 —— 也算到了终态
    body = pg.inner_text("body")
    pg.screenshot(path=str(OUT / "20_bad_pdf.png"), full_page=True)
    # 关键判据（2026-10-28 收紧）：必须出现**明确的结论性提示**，
    # 不能只是页面上恰好有「PDF」这两个字（上传区标签就有）。
    check("没有产出审计结论" in body, "给出了「这一轮没有产出审计结论」的明确提示",
          "在页面上" if "没有产出审计结论" in body else "没找到")
    check(bool(found), "提示里说明了原因（文件/代码/路径三类）", f"命中「{found}」")
    badge = ""
    if pg.locator(".dk-top").count():
        badge = pg.locator(".dk-top").inner_text()
    check("已完成" not in badge, "顶栏不再把「没产出」错标成「已完成」",
          badge.replace("\n", " ")[:40])
    check(len(body) > 200, "整页仍在（没有被组件异常吞掉）", f"{len(body)} 字")
    check("上传论文资料" in body or "开始科研审计" in body or "开始 科 研 审 计" in body,
          "上传区/主按钮仍在，可以重试")
    check(len(errs) == 0, "浏览器控制台没有 JS 报错", f"{len(errs)} 条")
    pg.close()

    # ---------- ② 空材料 ----------
    pg2 = b.new_page(viewport={"width": 1600, "height": 1100})
    pg2.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg2.wait_for_selector("text=上传论文资料", timeout=60000)
    txt0 = pg2.inner_text("body")
    check("内置" in txt0 or "样例" in txt0, "首屏就写明了「不传就走内置样例」")
    pg2.get_by_role("button", name="开 始 科 研 审 计").first.click()
    # ⚠ 只认顶栏「● 已完成」。等 `text=逐条对表` 会假命中（左侧导航页名，一加载就有）。
    pg2.wait_for_selector("text=● 已完成", timeout=600000)
    # 「当前审计对象」这一条是**后渲染**的（它跟着结果块一起出，比「逐条对表」晚一拍）。
    # 只等固定 2 秒会偶发读不到，判成"页面没标注演示数据" —— 2026-10-29 实测：
    # 手动跑同样的流程，这一条稳定存在，所以是等待时机问题，不是文案缺失。
    try:
        pg2.wait_for_selector("text=当前审计对象", timeout=20000)
    except Exception:                       # noqa: BLE001
        pass                                # 真没有的话下面那条断言会报出来
    pg2.wait_for_timeout(800)
    t2 = pg2.inner_text("body")
    pg2.screenshot(path=str(OUT / "21_no_upload.png"), full_page=True)
    check("已走" in t2, "空材料也能跑完（走内置样例）")
    check("demo/paper.pdf" in t2, "材料区显示的是内置样例文件")
    check("当前审计对象" in t2 and "内置" in t2, "页面如实标注「内置演示数据」")
    pg2.close()
    b.close()

print(f"\n===== 兜底路径：{ok + bad} 项，通过 {ok}，未通过 {bad} =====")
sys.exit(1 if bad else 0)
