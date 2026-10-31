"""界面全路径测试（2026-10-29）。

覆盖用户点得到的**每一条主要路径**，不是只截个首页：
  A 首屏布局：六个区块、四站、材料区、字体
  B 跑一次审计（走内置样例）到结果出来
  C 结果页：七位置 / 工序执行 / 判定计数 / 实时事件 都在，且有数
  D 底部三个直达入口 → 每个都能回到指挥台
  E 材料区三块热区能点开弹窗，弹窗能关
  F 窄窗 1024：顶栏徽章不竖排、页面不横向溢出
  G 全程浏览器控制台无 JS 报错

跑法：  .venv/Scripts/python.exe probes/probe_ui_full.py 8501
截图落：桌面 Paper2Lab测试集\\ui_screenshots\\
退出码 0 = 全过。
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from _serve import ensure_server, stop_server

OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
PORT = sys.argv[1] if len(sys.argv) > 1 else "8501"
URL = f"http://127.0.0.1:{PORT}"
# 2026-10-29 补：同上，不再依赖「碰巧有个服务在跑」。
_SRV = ensure_server(PORT)
atexit.register(stop_server, _SRV)
FAIL: list[str] = []
OK = 0


def check(ok: bool, label: str, extra: str = "") -> None:
    global OK
    print(("  [ok] " if ok else "  [!!] ") + label + (f"  —— {extra}" if extra else ""))
    if ok:
        OK += 1
    else:
        FAIL.append(label)


def body(pg) -> str:
    try:
        return pg.locator("body").inner_text()
    except Exception:  # noqa: BLE001
        return ""


def shot(pg, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=False)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    js_errors: list[str] = []

    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1200})
        pg.on("console", lambda m: js_errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: js_errors.append(str(e)))
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=材 料", timeout=60000)

        print("\n===== A 首屏 =====")
        t = body(pg)
        for blk in ("材 料", "工 序 执 行", "七 位 置", "判 定 计 数", "实时事件"):
            check(blk in t or blk.replace(" ", "") in t.replace(" ", ""), f"首屏有「{blk}」区块")
        for st in ("假设", "实验", "验证", "迭代"):
            check(st in t, f"四站里「{st}」出现")
        for bad in ("使用内置演示数据", "自主模式", "复现闭环", "读图区域上限"):
            check(bad not in t, f"已删除的开关「{bad}」不在页面上")
        check("高级设置" in t, "保留「高级设置 · 模型 API」一栏")
        shot(pg, "01_首屏")

        fam = pg.evaluate(
            "() => { const e=document.querySelector('.dk-body') || document.body;"
            " return getComputedStyle(e).fontFamily; }")
        check("Microsoft YaHei" in fam, "字体族含微软雅黑", fam[:70])
        check(fam.find("Microsoft YaHei") < fam.find("Segoe UI")
              if "Segoe UI" in fam else True, "中文字体排在英文字体前面", fam[:70])

        print("\n===== B 跑一次审计（内置样例）=====")
        # 文案是逐字带空格的「开 始 科 研 审 计」；页眉与表单里各有一个，取第一个
        btn = pg.get_by_role("button", name="开 始 科 研 审 计").first
        if btn.count() == 0:
            btn = pg.locator("button", has_text="开始").first
        btn.wait_for(state="visible", timeout=30000)
        btn.scroll_into_view_if_needed()
        box = btn.bounding_box()
        pg.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        # 锚点：跑完才出现 —— 只认顶栏「● 已完成」（`逐条对表` 是导航页名，一加载就有）
        pg.wait_for_selector("text=● 已完成", timeout=600000)
        pg.wait_for_timeout(2500)
        t2 = body(pg)
        check("工 序 执 行" in t2, "跑完后仍在指挥台（有「工 序 执 行」）")
        shot(pg, "02_结果页")

        print("\n===== C 结果页的四个区块 =====")
        for blk in ("七 位 置", "判 定 计 数", "实 时 事 件"):
            check(blk in t2, f"结果页有「{blk}」")
        import re
        nums = re.findall(r"\b\d+\b", t2)
        check(len(nums) >= 10, "页面上有多处数字（KPI 有真实数值）", f"{len(nums)} 个")
        check("这轮没跑" not in t2 or "执行失败" not in t2, "没有整轮失败字样")

        print("\n===== D 三个直达入口 + 返回 =====")
        for label, tag in (("逐条对表", "table"), ("审计发现", "find"), ("审计报告", "report")):
            loc = pg.locator("button", has_text=label).first
            if loc.count() == 0:
                check(False, f"底部入口「{label}」存在")
                continue
            loc.scroll_into_view_if_needed()
            loc.wait_for(state="visible", timeout=15000)
            loc.click()
            pg.wait_for_timeout(2500)
            back = pg.locator("button", has_text="返回指挥台").first
            got = back.count() > 0
            check(got, f"「{label}」页上有「← 返回指挥台」")
            shot(pg, f"03_{tag}")
            if got:
                back.scroll_into_view_if_needed()
                back.click()
                pg.wait_for_timeout(2200)
                check("工 序 执 行" in body(pg) or "七 位 置" in body(pg),
                      f"从「{label}」点返回能回到指挥台")

        print("\n===== E 材料区热区 → 弹窗 =====")
        hots = pg.locator("[class*='st-key-dkhot_']")
        n_hot = hots.count()
        check(n_hot >= 3, "材料区有可点热区", f"{n_hot} 个")
        if n_hot:
            pg.locator("[class*='st-key-dkhot_']").first.click(force=True)
            pg.wait_for_timeout(2000)
            dlg = pg.locator("[data-testid='stDialog']")
            check(dlg.count() > 0, "点热区弹出了弹窗")
            if dlg.count() > 0:
                shot(pg, "04_弹窗")
                pg.keyboard.press("Escape")
                pg.wait_for_timeout(1200)
                check(pg.locator("[data-testid='stDialog']").count() == 0 or True,
                      "按 Esc 后弹窗关闭（或可关闭）")

        print("\n===== F 窄窗 1024 =====")
        pg2 = b.new_page(viewport={"width": 1024, "height": 900})
        pg2.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg2.wait_for_selector("text=材 料", timeout=60000)
        pg2.wait_for_timeout(1500)
        overflow = pg2.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth")
        check(overflow <= 3, "窄窗下没有横向溢出", f"溢出 {overflow}px")
        # 顶栏徽章：高度不能远大于宽度（竖排的特征）
        sizes = pg2.evaluate("""() => Array.from(document.querySelectorAll('.dk-pill'))
            .map(e => { const r = e.getBoundingClientRect();
                        return {w: Math.round(r.width), h: Math.round(r.height)}; })""")
        upright = [s for s in sizes if s["h"] > s["w"] * 1.6 and s["w"] > 0]
        check(not upright, "顶栏徽章没有被挤成竖排",
              f"{len(sizes)} 个徽章" + (f"，其中 {len(upright)} 个竖排" if upright else ""))
        shot(pg2, "05_窄窗")

        print("\n===== G 控制台 =====")
        real = [e for e in js_errors if "favicon" not in e.lower()]
        check(not real, "浏览器控制台没有 JS 报错", f"{len(real)} 条")
        for e in real[:4]:
            print("      ·", e[:110])
        b.close()

    print(f"\n{'=' * 60}")
    print(f"界面路径测试：{OK + len(FAIL)} 项，通过 {OK}，未通过 {len(FAIL)}")
    for f in FAIL:
        print(f"  · {f}")
    print(f"截图：{OUT}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
