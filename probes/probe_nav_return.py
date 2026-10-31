"""探针：底部三个直达入口跳出去之后，能不能一键回到深色指挥台。

2026-10-27 用户报的 bug：「逐条对表 / 审计发现 / 审计报告」点完就回不到主页面。
原因是侧栏在窗口不够宽时会被 Streamlit 自己收起，二级页面上就找不到出口。
修法：二级页面最上面固定一颗「← 返回指挥台」。这里按真机点一遍。
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
# 2026-10-29 补：以前这个探针假设「8501 上一定有人跑着服务」，于是它的成败取决于
# **前面一项检查有没有恰好留下一个活着的实例** —— 前一环失败它就报 ERR_CONNECTION_REFUSED，
# 看着像"界面挂了"，其实是没人开服务。现在没有就自己起、跑完自己关。
_SRV = ensure_server(PORT)
atexit.register(stop_server, _SRV)
FAIL: list[str] = []


def check(ok: bool, label: str) -> None:
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        FAIL.append(label)


def body(pg) -> str:
    try:
        return pg.locator("body").inner_text()
    except Exception:                                    # noqa: BLE001
        return ""


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 1680, "height": 1200})
        pg.goto(URL, wait_until="domcontentloaded", timeout=60000)
        pg.wait_for_selector("text=上传论文资料", timeout=60000)

        t = body(pg)
        print("\n===== A 那一排开关是不是真的没了 =====")
        for bad in ("使用内置演示数据", "自主模式（模型定流程", "复现闭环（搜",
                    "让模型自主决定「要不要再走一遍」", "角色用模型",
                    "模型先判、程序后验", "让模型挑计算工具",
                    "读图区域上限", "读图耗时预算"):
            check(bad not in t, f"页面上不再有「{bad}」")
        check("高级设置" in t, "还留着「高级设置 · 模型 API 与 图像通道」这一栏")

        pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
        # ⚠ 只认顶栏「● 已完成」。等 `text=逐条对表` 会假命中 —— 那是左侧导航的页名，
        #   页面一加载就在，会拿半成品页面去断言（假红）。也不能等「已走 N / M 格」，
        #   那行数字在跑的过程中一直在变。
        pg.wait_for_selector("text=● 已完成", timeout=600000)
        pg.wait_for_selector("text=已走", timeout=60000)
        pg.wait_for_timeout(2000)
        check("工 序 执 行" in body(pg), "首屏是深色指挥台（有「工 序 执 行」）")

        print("\n===== B 三个直达入口：出去 / 回来 =====")
        for label, shot in (("逐条对表", "table"), ("审计发现", "find"),
                            ("审计报告", "report")):
            # ⚠ 用「带文字的 button」定位 + 等它可见：
            #   跑完那一刻 app.py 会主动 st.rerun() 一次（把输入区折起来），
            #   这一瞬间 DOM 正在重建，直接 count() 会读到 0（本轮踩到的假红）。
            btn = pg.locator("button", has_text=label).first
            try:
                btn.wait_for(state="visible", timeout=30000)
            except Exception:                            # noqa: BLE001
                check(False, f"找得到「{label}」按钮")
                continue
            check(True, f"找得到「{label}」按钮")
            btn.scroll_into_view_if_needed()
            btn.click()
            pg.wait_for_timeout(2200)
            t2 = body(pg)
            check("二级页面" in t2, f"点「{label}」进到二级视图（有返回入口）")
            check("返回指挥台" in t2, f"「{label}」这一页看得见「返回指挥台」")
            pg.screenshot(path=str(OUT / f"nav2_{shot}.png"))

            back = pg.get_by_role("button", name="← 返回指挥台").first
            if not back.count():
                check(False, f"「{label}」页上点得到返回按钮")
                continue
            back.click()
            pg.wait_for_timeout(2600)
            t3 = body(pg)
            check("工 序 执 行" in t3 and "实 时 事 件" in t3,
                  f"从「{label}」回到了深色指挥台")
            pg.screenshot(path=str(OUT / f"nav2_back_{shot}.png"))

        print("\n===== C 材料三块仍然点得开 =====")
        for key, name in (("mat_paper", "论文"), ("mat_code", "代码")):
            loc = pg.locator(f"[class*='st-key-dkhot_{key}'] button").first
            if not loc.count():
                check(False, f"材料卡「{name}」有热区")
                continue
            loc.scroll_into_view_if_needed()
            loc.click()
            pg.wait_for_timeout(1500)
            dlg = pg.locator('div[data-testid="stDialog"]')
            check(dlg.count() > 0, f"材料卡「{name}」点开有内容")
            if dlg.count():
                txt = pg.locator(".st-key-dk_detail").inner_text()
                check("解析" in txt or "扫描" in txt or "载入" in txt,
                      f"材料卡「{name}」讲的是扫描/解析情况")
                pg.get_by_role("button", name="收起详情").first.click()
                pg.wait_for_timeout(1200)
        b.close()

    print("\n结论：" + ("全部通过" if not FAIL else f"{len(FAIL)} 项未通过 {FAIL}"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
