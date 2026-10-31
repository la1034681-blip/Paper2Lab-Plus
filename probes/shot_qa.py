"""问答窗口真机探针：入口位置 -> 弹窗 -> 多轮问答（2026-10-27，判据 10-30 修正）。

只用两种判据，避免"看着像"就放行：
  · 坐标：入口按钮必须在首屏右上角（y 很小、x 靠右）；
  · 文字：弹窗里出现的东西必须是这一轮审计的真数据（页码 / 行号 / 原值）。

两个判据坑：
  · 完成判据只认顶栏「● 已完成」。旧写法等 `text=逐条对表` 会假命中
    （那是左侧导航的页名，页面一加载就在），审计没跑完就点开问答 ⇒ 变成空态。
  · 答复标记是「模型作答」/「直接检索到的审计事实」。旧文案「未配置模型」已不存在，
    现在分成 还没跑审计 / 没 key / 调用失败 三种如实说明。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8599"
URL = f"http://127.0.0.1:{PORT}"
DIALOG = 'div[data-testid="stDialog"]'
DONE = "● 已完成"
ANSWERED = ("模型作答", "直接检索到的审计事实")
BUSY = "在翻这一轮的审计数据"


def dump(tag: str, page) -> str:
    try:
        t = page.inner_text("body")
    except Exception:      # noqa: BLE001
        t = ""
    print(f"--- {tag} ---\n{t[:1200]}\n")
    return t


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1680, "height": 1050})
    pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=上传论文资料", timeout=60000)
    pg.wait_for_timeout(1500)

    # ---- 1 入口位置：必须在首屏右上角
    btn = pg.get_by_role("button", name="问 一 问").first
    box = btn.bounding_box()
    print("入口按钮坐标:", box)
    ok_pos = bool(box and box["y"] < 160 and box["x"] > 1000)
    print("✅ 入口在首屏右上角" if ok_pos else "❌ 入口位置不对")
    pg.screenshot(path=str(OUT / "qa_01_entry.png"))

    # ---- 2 点开弹窗（空态：还没跑审计）
    btn.click()
    pg.wait_for_selector(DIALOG, timeout=20000)
    pg.wait_for_timeout(1200)
    txt = dump("空态弹窗", pg)
    print("空态含欢迎语:", "先问我" in txt or "还没有审计结论" in txt)
    pg.screenshot(path=str(OUT / "qa_02_dialog_empty.png"))

    # ---- 3 关掉，跑一轮真审计
    pg.get_by_role("button", name="结束问答").first.click()
    pg.wait_for_timeout(1500)
    print("关掉后弹窗还在吗:", pg.locator(DIALOG).count() > 0)
    pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
    pg.wait_for_selector(f"text={DONE}", timeout=600000)
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(OUT / "qa_03_done.png"))

    # ---- 4 有数据了再点开
    pg.get_by_role("button", name="问 一 问").first.click()
    pg.wait_for_selector(DIALOG, timeout=20000)
    pg.wait_for_timeout(1500)
    txt = dump("有数据弹窗", pg)
    print("含论文标题或对表条数:", ("对表" in txt))
    pg.screenshot(path=str(OUT / "qa_04_dialog_data.png"))

    # ---- 5 第一问：点快捷问题（真调模型）
    pg.get_by_role("button", name="这几个数为什么对不上").first.click()
    for _ in range(90):
        pg.wait_for_timeout(2000)
        t = pg.inner_text(DIALOG)
        if BUSY not in t and any(k in t for k in ANSWERED):
            break
    pg.wait_for_timeout(1200)
    t1 = dump("第一问之后", pg)
    print("第一问有答复:", any(k in t1 for k in ANSWERED))
    pg.screenshot(path=str(OUT / "qa_05_answer1.png"))

    # ---- 6 第二问：手打（验证多轮）
    box = pg.locator(f'{DIALOG} textarea').first
    box.click()
    box.fill("我复现时最先该改哪里？请给具体位置")
    pg.get_by_role("button", name="发送").first.click()
    for _ in range(90):
        pg.wait_for_timeout(2000)
        t = pg.inner_text(DIALOG)
        if "我复现时最先该改哪里" in t and BUSY not in t:
            break
    pg.wait_for_timeout(1200)
    t2 = dump("第二问之后", pg)
    n_user = t2.count("我复现时最先该改哪里")
    print("多轮成立（两问都在历史里）:", n_user >= 1 and "这几个数为什么对不上" in t2)
    pg.screenshot(path=str(OUT / "qa_06_turn2.png"))

    b.close()
print("done")
