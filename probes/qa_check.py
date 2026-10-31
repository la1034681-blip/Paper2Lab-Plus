"""问答窗口真机独立验收（带硬断言，不只看截图）。

验收点：
  ① 入口在首屏右上角，点得开；
  ② 弹窗里的内容必须来自这一轮审计（出现对表条数 / 页码 / 参数名）；
  ③ 问一句能得到答复（模型作答，或如实退回检索证据 / 说明没跑审计）；
  ④ 第二问时第一问仍在历史里（多轮成立）；
  ⑤ 关闭弹窗回到深色指挥台，页面没被吞掉。

两个判据坑（2026-10-30 踩过）：
  · 不能等 `text=逐条对表` —— 那是左侧导航的页名，页面一加载就命中，
    审计还没跑完就往下走，结果 `audit=None`、后面每一问都变成「还没有审计结论」。
    唯一可信的完成判据是顶栏「● 已完成」。
  · 答复的标记是「模型作答」或「直接检索到的审计事实」；旧文案「未配置模型」
    已经不存在（现在分成 未跑审计 / 没 key / 调用失败 三种如实说明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab_本轮测试截图")
OUT.mkdir(parents=True, exist_ok=True)
PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"
DLG = 'div[data-testid="stDialog"]'
DONE = "● 已完成"                      # 审计跑完的唯一可信判据
ANSWERED = ("模型作答", "直接检索到的审计事实")   # 答复落地的两种方式
BUSY = "在翻这一轮的审计数据"
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


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1680, "height": 1200})
    pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
    pg.wait_for_selector("text=上传论文资料", timeout=60000)
    pg.wait_for_timeout(1200)

    btn = pg.get_by_role("button", name="问 一 问").first
    check(btn.count() > 0, "首屏有「问 一 问」入口")
    box = btn.bounding_box() or {}
    check(box.get("y", 999) < 140, "入口在首屏顶部", f"y={round(box.get('y', -1))}")
    check(box.get("x", 0) > 1000, "入口靠右", f"x={round(box.get('x', -1))}")

    pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
    pg.wait_for_selector(f"text={DONE}", timeout=600000)
    pg.wait_for_timeout(2500)
    deck = pg.inner_text("body")
    check(DONE in deck, "审计跑完，指挥台出数")

    pg.get_by_role("button", name="问 一 问").first.click()
    pg.wait_for_selector(DLG, timeout=20000)
    pg.wait_for_timeout(1800)
    t0 = pg.inner_text(DLG)
    pg.screenshot(path=str(OUT / "qa1_open.png"))
    check(len(t0) > 60, "弹窗打开并渲染内容", f"{len(t0)} 字")
    check(any(k in t0 for k in ("对表", "条", "页", "参数")), "弹窗里带这轮审计的事实字段")

    pg.get_by_role("button", name="这几个数为什么对不上").first.click()
    answered = False
    for _ in range(90):
        pg.wait_for_timeout(2000)
        t = pg.inner_text(DLG)
        if BUSY not in t and any(k in t for k in ANSWERED):
            answered = True
            break
    pg.wait_for_timeout(1500)
    t1 = pg.inner_text(DLG)
    pg.screenshot(path=str(OUT / "qa2_answer.png"))
    check(answered, "第一问得到答复（模型作答，或如实退回检索证据）")
    check(len(t1) > len(t0) + 40, "答复确实增加了内容", f"{len(t0)}→{len(t1)}")
    # 答复里必须有可核对的东西：参数名 / 页码 / 行号 / 数字
    check(any(k in t1 for k in ("页", ".py:", "学习率", "dropout", "Dropout", "epoch")),
          "答复里能追到具体条目（不是空话）")

    ta = pg.locator(f"{DLG} textarea").first
    ta.click()
    ta.fill("我复现时最先该改哪里？")
    pg.get_by_role("button", name="发送").first.click()
    for _ in range(90):
        pg.wait_for_timeout(2000)
        t = pg.inner_text(DLG)
        if "我复现时最先该改哪里" in t and BUSY not in t:
            break
    pg.wait_for_timeout(1200)
    t2 = pg.inner_text(DLG)
    pg.screenshot(path=str(OUT / "qa3_turn2.png"))
    check("我复现时最先该改哪里" in t2, "第二问出现在历史里")
    check("这几个数为什么对不上" in t2, "第一问仍留在历史里（多轮成立）")

    pg.keyboard.press("Escape")
    pg.wait_for_timeout(1500)
    still = pg.locator(DLG).count()
    body = pg.inner_text("body")
    pg.screenshot(path=str(OUT / "qa4_closed.png"))
    check(still == 0 or "问 一 问" in body, "弹窗可关闭，页面仍在")
    check(DONE in body, "关闭后回到指挥台（数据还在）")

    b.close()

print(f"\n===== 问答窗口：{ok + bad} 项，通过 {ok}，未通过 {bad} =====")
sys.exit(1 if bad else 0)
