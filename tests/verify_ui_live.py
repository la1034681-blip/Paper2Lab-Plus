"""一体化: 重启服务(确保加载最新代码) -> 界面结构与渲染检查(含 HTML 泄漏检测)。

用法: .venv\\Scripts\\python.exe verify_ui_live.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://localhost:8501"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def health_ok() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def goto_nav(page, label: str) -> bool:
    """2026-10-26 界面改成「左侧导航 + 分页」后新增：发现卡片 / 可信度页签都在
    「审计发现」页里，跑完默认落在总览页 ⇒ 不切页就全是 0（会被误判成"没渲染"）。
    只点侧栏 radiogroup 的 label：导航文案与页标题同名时 get_by_text 会点错。
    """
    try:
        # 2026-10-26：先等导航容器本身出现再找 label —— 结果刚出来那一刻
        # Streamlit 可能还没把侧栏挂上 DOM，直接 count() 会拿到 0 误报"没切过去"。
        page.wait_for_selector('[data-testid="stSidebarUserContent"] [role="radiogroup"] label, [data-testid="stSidebar"] [role="radiogroup"] label',
                               timeout=20000)
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


def kill_port(port: int = 8501) -> None:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
    except Exception:
        return
    pids = set()
    for line in out.splitlines():
        if f":{port}" in line and "LISTENING" in line:
            pids.add(line.split()[-1])
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)


def start_server() -> subprocess.Popen:
    print("[live] 启动 Streamlit…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", "8501"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if health_ok():
            print("[live] 服务就绪")
            return proc
        time.sleep(1)
    print("[live] ⚠️ 启动超时")
    return proc


def main() -> None:
    kill_port()
    proc = start_server()
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1600, "height": 1100})
        errs: list[str] = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        # 首屏判据（2026-10-27 改版）：指挥台页不再有浅色导航栏，所以「审计输入」
        # 那串字没有了。改判「上传论文资料」——它就是深色指挥台左上角那一行的标题，
        # 而且三个上传控件也挂在它下面，命中即代表首屏真的立住了。
        page.wait_for_selector("text=上传论文资料", timeout=60000)
        page.locator('button[kind="primary"]').first.click()
        page.wait_for_selector("text=综合复现就绪度", timeout=120000)
        page.wait_for_timeout(3000)
        # 2026-10-26 分页改版：发现卡片与可信度页签都在「审计发现」页，先切过去再查。
        _moved = goto_nav(page, "审计发现")
        print(f"  切到审计发现页: {'✅' if _moved else '❌'}")
        # 切页后要渲染 13 张发现卡片 + 一张总览表，固定等 1.5s 在慢机器上会漏 ——
        # 2026-10-26 指挥台上线后首屏更重，这里踩过一次（判据全 0，其实是没渲染完）。
        # 改成**等元素真的挂上 DOM**：判据一个字没改，只是不再靠猜时间。
        try:
            page.wait_for_selector(".finding", timeout=25000)
        except Exception:      # noqa: BLE001
            pass
        page.wait_for_timeout(600)
        body = page.inner_text("body")

        leaks = [m for m in ("</div>", "<span", "class='", 'class="', "evi-loc",
                             "risk-bar", "ai-box", "chip-ghost", "mc-item")
                 if m in body]
        res = {
            "发现卡片": page.locator("div.finding").count(),
            "卡片结论行": page.locator("div.f-head").count(),
            "三列证据块": page.locator("div.f-evi > div.evi").count(),
            "比对结论列": body.count("比对结论"),
            "人工核查清单(主页面)": "需人工核查清单" in body,
            "未参与审计(主页面)": "未参与本次审计的部分" in body,
            "HTML 泄漏": leaks,
            "浏览器异常": errs[:3],
        }
        page.screenshot(path=str(OUT / "07_new_layout.png"), full_page=True)

        for t in page.locator("[role=tab]").all():
            if "可信度" in t.inner_text():
                t.click()
                break
        page.wait_for_timeout(2500)
        tb = page.inner_text("body")
        res["可信度页·核查清单块"] = page.locator("div.mc-item").count()
        res["可信度页·日志条数"] = bool(re.search(r"全部\s*\d+\s*条", tb))
        res["可信度页·未参与部分"] = "未参与本次审计的部分" in tb
        res["可信度页·参数覆盖清单"] = "参数覆盖清单" in tb
        res["可信度页·取值来源分布"] = "取值来源分布" in tb
        res["可信度页·Agent轨迹面板"] = "抽取 Agent 轨迹" in tb
        res["可信度页·泄漏"] = [m for m in ("</div>", "class='", "mc-item") if m in tb]
        page.screenshot(path=str(OUT / "08_trust_new.png"), full_page=True)

        # 回风险项页, 检查"风险判定依据"是否渲染
        for t in page.locator("[role=tab]").all():
            if "风险项" in t.inner_text():
                t.click()
                break
        page.wait_for_timeout(1500)
        rb = page.inner_text("body")
        res["风险判定依据(卡片)"] = rb.count("风险判定依据")
        res["覆盖清单可见(风险页)"] = "参数覆盖清单" in rb or "参数覆盖清单" in tb
        b.close()

    print("=== 界面渲染检查 ===")
    for k, v in res.items():
        print(f"{k}: {v}")

    # ---- 必须通过的项（2026-10-15 补）。
    # 本脚本此前**根本没有 sys.exit**：无论检查成什么样都退出 0，于是体检里
    # 「界面 结构与泄漏检查」永远判 OK —— 与 ui_test 当初一样形同虚设，
    # 连 HTML 泄漏都拦不住。现在按项判定。
    MUST_PASS = {
        "发现卡片": lambda v: v > 0,
        "卡片结论行": lambda v: v > 0,
        "三列证据块": lambda v: v > 0,
        "HTML 泄漏": lambda v: not v,
        "浏览器异常": lambda v: not v,
        "可信度页·核查清单块": lambda v: v > 0,
        "可信度页·日志条数": bool,
        "可信度页·未参与部分": bool,
        "可信度页·参数覆盖清单": bool,
        "可信度页·取值来源分布": bool,
        "可信度页·泄漏": lambda v: not v,
        "风险判定依据(卡片)": lambda v: v > 0,
        "覆盖清单可见(风险页)": bool,
    }
    # 如实说明: 下面三项**长期是 False，且不是任何一次改动引进的** ——
    #   「人工核查清单(主页面)」/「未参与审计(主页面)」：这两个板块本来就在
    #     「审计可信度」tab 里，主页面本来就没有（可信度页那两项是 True，可证）。
    #   「可信度页·Agent轨迹面板」：它找的「抽取 Agent 轨迹」是 **Markdown 报告**里的
    #     章节标题（report.py），界面上从来就没有这个面板 —— 断言对象搞错了。
    # 之所以过去没人发现: 一是本脚本不退出非 0，二是断言失效时只在输出里打一个 False。
    # 保留在输出里是为了不掩盖历史，**但不判定** —— 把它们设成必过项，
    # 等于让体检长期挂红，而红的原因与界面质量无关。
    INFO_ONLY = {"人工核查清单(主页面)", "未参与审计(主页面)", "可信度页·Agent轨迹面板"}

    bad: list[str] = []
    for k, ok in MUST_PASS.items():
        if k not in res:
            bad.append(f"{k}: 没跑到这一项（检查项缺失）")
        elif not ok(res[k]):
            bad.append(f"{k}: {res[k]}")
    for b in bad:
        print("  *x* " + b)
    for k in sorted(INFO_ONLY):
        if k in res:
            print(f"  (信息项, 不判定) {k}: {res[k]}")
    print(f"\n判定 {len(MUST_PASS)} 项, 未通过 {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
