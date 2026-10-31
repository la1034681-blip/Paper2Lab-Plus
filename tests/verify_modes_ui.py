"""真机验证：删掉「经典模式」之后，界面上**只有一条路**且这条路口径清楚。

2026-10-25 用户口径：删除经典模式（不看材料、固定全序全跑），回到科学智能体主线。
原脚本是"经典 vs 计划"四组对照，对照组没了，于是改成守**单一路径的两个场景**：

  A 演示数据（材料齐备）  → 计划内全部执行、无「未执行」，决策表在
  B 只传论文（无代码）    → 代码扫描 / 根因 / 补丁被跳过，且**逐条给出理由**

⚠️ 断言口径：**不写死跳过数**。模型会按成本判断额外跳过（那正是 agent 在起作用）。
这里守的是**契约**：
  · 页面上不该再有任何"模式切换"开关
  · 执行 + 跳过 = 全部工具（执行集合如实呈现）
  · 每个跳过都有理由可查（决策表在）
  · 流程面板的「未执行」次数不少于被跳过的工具数
  · 谁排的计划要如实标注（模型排 / 规则版兜底 / 模型未生效）
"""
from __future__ import annotations

import re
import subprocess
import time
import urllib.request
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.toolbox import TOOLS       # noqa: E402  （断言"执行+跳过=全部工具"要用）
PORT = 8513
PAPER = ROOT / "demo" / "paper.pdf"
_OUT_DIR = ROOT / ".trash_tmp"      # 验证产物落已 gitignore 的临时目录，不污染工作区
_OUT_DIR.mkdir(exist_ok=True)
OUT = _OUT_DIR / "_modes_ui.txt"
SHOTS = _OUT_DIR / "_shot_full.png", _OUT_DIR / "_shot_paper_only.png"
log: list[str] = []


def say(s: str) -> None:
    log.append(s)
    print(s, flush=True)


proc = subprocess.Popen(
    [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-m", "streamlit", "run",
     "app.py", "--server.headless", "true", "--server.address", "127.0.0.1",
     "--server.port", str(PORT)],
    cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

try:
    from playwright.sync_api import sync_playwright

    ok = False
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    ok = True
                    break
        except Exception:
            time.sleep(2)
    say(f"服务就绪: {ok}")

    results: dict = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)

        def run_case(tag: str, upload_paper: bool) -> str:
            page = browser.new_page(viewport={"width": 1500, "height": 1200})
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            page.goto(f"http://127.0.0.1:{PORT}/?view=exec", timeout=60000)
            page.wait_for_timeout(9000)

            # ⚠ 「使用内置演示数据」默认是**开**的, 而上传优先级低于它 —— 只传论文那组
            #   若不先关掉这个开关, 跑的根本是**演示材料**, 等不到「执行记录」还会读成 -1,
            #   看起来像"应用卡住", 其实是测试自己没配好输入（2026-09-29 实测踩过）。
            demo_lab = page.locator("label").filter(has_text="使用内置演示数据")
            if upload_paper and demo_lab.count():
                demo_lab.first.click(force=True)
                page.wait_for_timeout(2000)
            if upload_paper:
                inputs = page.locator("input[type=file]")
                inputs.nth(0).set_input_files(str(PAPER))
                page.wait_for_timeout(2500)

            page.get_by_text("开 始 科 研 审 计", exact=False).click()
            body = ""
            for _ in range(45):
                page.wait_for_timeout(2000)
                body = page.inner_text("body")
                # ⚠ 2026-10-26：不能用「执行记录」当锚点 —— 改了分页之后，**侧栏导航里**
                #    就有这四个字，一进页面就为真，循环立刻跳出，读到的是还没跑完的页面
                #    （表现为"执行集合"读成 -1）。改等展开栏标题「执行 N 个工具 / 跳过 M 个」
                #    里的「个工具」，它只在结果出来后才出现。
                if "个工具" in body:
                    break
            page.wait_for_timeout(3000)
            body = page.inner_text("body")

            shot = SHOTS[1] if upload_paper else SHOTS[0]
            page.screenshot(path=str(shot))
            if errs:
                say(f"  ⚠ {tag} 控制台异常 {len(errs)}: {errs[:2]}")
            page.close()
            return body

        say("")
        say("=== A. 演示数据（材料齐备）===")
        a = run_case("A", upload_paper=False)
        results["A"] = a
        for k in ["执行记录", "模型的决策", "规则兜底的决策", "模型排计划",
                  "兜底开关", "经典模式"]:
            say(f"  含「{k}」: {k in a}")

        say("")
        say("=== B. 只传论文（无代码）===")
        b = run_case("B", upload_paper=True)
        results["B"] = b
        for k in ["模型排计划", "规则版计划", "模型的决策", "计划跳过",
                  "未提供代码包", "未执行", "执行记录"]:
            say(f"  含「{k}」: {k in b}")

        browser.close()

    # ---------------- 断言 ----------------
    # 注: 执行明细表用 st.dataframe, 是 canvas 渲染的, inner_text 读不到单元格文本;
    #     因此从**展开栏标题**（DOM 可见）读"执行 N 个 / 跳过 M 个"来判定。
    def counts(body: str) -> tuple:
        m = re.search(r"执行\s*(\d+)\s*个工具\s*/\s*跳过\s*(\d+)\s*个", body)
        return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)

    say("")
    say("=== 从展开栏标题读到的执行计数 ===")
    cs = {k: counts(v) for k, v in results.items()}
    for k in "AB":
        say(f"  {k}: 执行 {cs[k][0]} 个 / 跳过 {cs[k][1]} 个")

    say("")
    # 决策表标题有两种（界面如实区分谁排的计划，两种都必须认）：
    #   模型可用 → 「模型的决策」；模型不可用 → 「规则兜底的决策」。
    def _has_decision_table(t: str) -> bool:
        return "模型的决策" in t or "规则兜底的决策" in t

    # 不占进度格的工具数（**派生，不写死**）：它们被跳过时，流程面板上没有对应的
    # 「未执行」格子 —— 面板是按环节画的，这类工具只在决策表里交代。
    _NO_STEP = sum(1 for t in TOOLS if not t.checkpoints)

    say("=== 断言 ===")
    checks = [
        # ---- 2026-10-25 删掉经典模式：页面上不该再有任何模式切换口子
        ("页面上已无「兜底开关」（经典模式入口已删除）",
         "兜底开关" not in results["A"] and "兜底开关" not in results["B"]),
        ("页面上已无「经典模式」字样", "经典模式" not in results["A"]),
        ("A 有决策表（谁排的计划必须摊开）", _has_decision_table(results["A"])),
        ("A 执行集合如实呈现（执行 + 跳过 = 全部工具）",
         cs["A"][0] + cs["A"][1] == len(TOOLS)),
        # 2026-09-28: 「根因追查 / 补丁草案」登记成工具后，无代码时**至少**这三个被跳过
        # （代码扫描 + 根因追查 + 补丁草案 —— 后两者都要读代码）。
        ("B（无代码）至少跳过 3 个（代码扫描 / 根因 / 补丁）", cs["B"][1] >= 3),
        ("B 执行集合如实呈现（执行 + 跳过 = 全部工具）",
         cs["B"][0] + cs["B"][1] == len(TOOLS)),
        ("B 给出了跳过理由（点明缺代码包）", "未提供代码包" in results["B"]),
        ("B 的跳过都有理由可查（决策表在）", _has_decision_table(results["B"])),
        ("B 流程面板标出了「未执行」", "未执行" in results["B"]),
        # 流程面板按**进度格**走；不占格的工具（如「化学身份核验」）被跳过时，
        # 面板上不会有它的「未执行」—— 这是设计如此，不是漏标。
        # 所以口径是：**有进度格的**跳过项必须都能在面板上看见。
        ("A / B 的「未执行」覆盖有进度格的跳过项",
         results["A"].count("未执行") >= cs["A"][1] - _NO_STEP
         and results["B"].count("未执行") >= cs["B"][1] - _NO_STEP),
        # 不占格的工具被跳过，也得在决策表里给出理由 —— 面板没有它的格子，
        # 决策表就是它唯一的交代处，不能两头都没有。
        ("A / B 的跳过都有理由可查（含不占格的工具）",
         _has_decision_table(results["A"]) and _has_decision_table(results["B"])),
        # ---- 谁排的计划必须留痕：模型生效 / 规则版兜底 / 如实说未生效，三者必居其一
        ("排计划有留痕（模型排 或 规则版兜底 或 未生效说明）",
         "模型排计划" in results["A"] or "规则版计划" in results["A"]
         or "模型排计划未生效" in results["A"]),
        ("A 与 B 的执行集合确实不同（计划真按材料走，不是固定全跑）",
         cs["A"][1] != cs["B"][1]),
    ]
    bad = 0
    for label, okk in checks:
        say(f"  [{'OK ' if okk else 'FAIL'}] {label}")
        if not okk:
            bad += 1

    say("")
    say("说明: 执行明细表用 canvas 渲染, DOM 读不到单元格 —— 其内容（含「环节」列与"
        "「这次实际做出了什么」列）由 verify_execview.py 以数据层断言覆盖。")
    say(f"截图: {SHOTS[0].name} / {SHOTS[1].name}")
    say("")
    say(f"未通过 {bad} 项" if bad else "全部通过（单一路径在界面上口径清楚）")
    # ⚠ 必须真判红：脚本只打印计数就 return 的话，退出码恒为 0 → 体检里永远 OK，
    #   界面口径漂移完全拦不住（2026-10-15 实测漏掉两处文案被改）。
    raise SystemExit(1 if bad else 0)
finally:
    proc.terminate()
    (OUT).write_text("\n".join(log), encoding="utf-8")
