"""验证：key 的状态显示是否正确，且**密钥绝不进浏览器**（不打印明文密钥）。

流程：杀掉旧服务 → 起新服务 → 打开页面 → 检查状态文案 → 断言唯一的 key 输入框**为空**
      （2026-10-30 起页面只留一组密钥，第二个视觉密钥框已删除）→ 保留服务运行。

为什么断言"必须为空"(2026-09-30 加固)：原先 `value=<服务端key>` 会把真 key
随页面发到浏览器，链接公开后任何人打开开发者工具就能读走。现在恒为空，
服务端 key 照用（`api_key or None` 回落到环境变量），状态改由文案告知。

⚠ 2026-10-26 新增「本机自用可预填」模式后的处理：
   用户要求本机页面直接带出 `.env` 里的 key，省得每次手填。这是**显式开关**：
   `.env` 里 `PAPER2LAB_PREFILL_KEYS=1` 才预填，默认与上线态都是关的
   （见 app.py 里那段注释与页面提示）。
   但这条安全底线不能被悄悄磨掉，所以本脚本**固定以 `PAPER2LAB_PREFILL_KEYS=0` 起服务**，
   验的就是"关掉预填时确实一个字符都不给浏览器"；另外再做一次源码守卫，
   要求预填模式**必须在页面上写明风险**，不许静默泄漏。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
URL = "http://localhost:8501"
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)

log: list[str] = []
errors: list[str] = []


def note(s: str):
    print(s)
    log.append(s)


def kill_8501():
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
        for line in out.splitlines():
            if ":8501" in line and "LISTENING" in line:
                pid = line.split()[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                note(f"  已结束旧服务进程 {pid}")
    except Exception as e:
        note(f"  结束旧进程失败(忽略): {e}")


def start_server():
    """后台起服务(独立进程组, 不随本脚本退出而结束)。"""
    py = HERE / ".venv" / "Scripts" / "python.exe"
    # 日志写到 .trash_tmp/，不在工程根目录留临时文件（工作区卫生）
    (HERE / ".trash_tmp").mkdir(exist_ok=True)
    logf = open(HERE / ".trash_tmp/_svc_keys_log.txt", "w", encoding="utf-8")
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    # ⚠ 显式把预填关掉：`.env` 不会覆盖**已存在的**环境变量（见 paper2lab/envfile.py），
    #    所以这里传进去的 0 一定生效 —— 本机开了预填也不影响这条安全验证。
    env = dict(os.environ, PAPER2LAB_PREFILL_KEYS="0")
    subprocess.Popen(
        [str(py), "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", "8501"],
        cwd=str(HERE), stdout=logf, stderr=subprocess.STDOUT, creationflags=flags,
        env=env)


def wait_health(timeout=90) -> bool:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/_stcore/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def main():
    kill_8501()
    time.sleep(2)
    start_server()
    if not wait_health():
        note("## 服务启动: ❌ 健康检查未通过")
        sys.exit(1)
    note("## 服务启动: ✅ 健康检查 200")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        page = b.new_page(viewport={"width": 1500, "height": 1100})
        console_errs: list[str] = []
        page.on("console", lambda m: console_errs.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errs.append(str(e)))

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)

        body = page.inner_text("body")
        # 1) 主页面（**折叠态**）
        note("## 1. 主页面（高级设置未展开时）")
        # 2026-10-30 第二次改口径：汇总行「读图通道｜归因引擎｜决策引擎」**整行已删除**
        #   （用户明确要求"彻底去除"）。连带后果写在这里，免得后人误判成 bug：
        #   **不展开高级设置时，页面上已完全看不到通道状态** —— 以前那行正是
        #   "不展开也能看到"的唯一入口，删掉它就等于把状态收进折叠区。
        #   所以本节只守两件事：① 输入区标题还在；② 三样状态字眼在折叠态都不许出现。
        #   读图通道与模型名的检查全部下移到第 2 节（展开之后）才做。
        ok = "模型 API" in body
        note(f"   - 输入区用的是通用名「模型 API」: {'✅' if ok else '❌'}")
        if not ok:
            errors.append("主页面缺少: 模型 API")
        for gone in ("读图通道", "归因引擎", "决策引擎"):
            _still = gone in body
            note(f"   - 折叠态页面无「{gone}」状态行: {'❌ 仍在' if _still else '✅'}")
            if _still:
                errors.append(f"汇总行未删干净: 折叠态页面仍有「{gone}」")

        # 2) 展开高级设置
        note("## 2. 高级设置内的密钥状态")
        exp = page.get_by_text("高级设置", exact=False)
        if exp.count():
            exp.first.click()
            page.wait_for_timeout(1500)
        body2 = page.inner_text("body")
        # 2026-10-30：页面只剩一组密钥，「视觉读图通道：已配置」改写为「读图通道：已就绪」
        for label, key in (("读图通道状态", "读图通道：已就绪"),
                           ("明确不影响结论", "不影响结论")):
            ok = key in body2
            note(f"   - {label}: {'✅' if ok else '❌'}")
            if not ok:
                errors.append(f"高级设置缺少: {label}")
        # 读图通道后面必须跟着一个**具体模型名**（不能只显示 ✅ 了事）。
        # 两种合法排版都要认，不绑品牌、不绑固定分隔符：
        #   · 多模态     → 「…（多模态）· intern-s1-mini　不需要第二组密钥」
        #   · 服务端另配 → 「…　intern-s1-mini　来源：.env」
        # 判据 = 「已就绪」之后 140 字内出现一个像模型名的英文 token。
        _m = re.search(r"读图通道：已就绪(.{0,140})", body2 or "", re.S)
        _seg = (_m.group(1) if _m else "")
        _mm = re.search(r"[A-Za-z][A-Za-z0-9._\-]{2,}", _seg)
        _vis_got = (_mm.group(0) if _mm else "")
        note(f"   - 读图通道带出具体模型名({_vis_got or '空'}): "
             f"{'✅' if _vis_got else '❌'}")
        if not _vis_got:
            errors.append("高级设置缺少: 读图通道未显示模型名")
        # 来源必须说清「读图到底用的谁」。**两条合法文案都要认**（界面本来就有两套分支）：
        #   · 服务端另配了视觉通道 → 来源：.env / 环境变量 / Secrets
        #   · 模型自己会读图      → 由上面这个模型自己读图（多模态）
        # 2026-10-24：`deepseek-chat` 被判为多模态后会自动走后者，页面自然没有 ".env"
        # 字样 —— 原先只认 ".env" 属于断言漏了分支，不是功能回归。
        src_ok = (".env" in body2) or ("由上面这个模型自己读图" in body2)
        note(f"   - 读图来源已说明（.env / 模型自己读图）: {'✅' if src_ok else '❌'}")
        if not src_ok:
            errors.append("高级设置未说明读图来源（既没写 .env，也没写模型自己读图）")

        # 3) 密钥输入框**必须为空**——这是安全要求，不是可用性偏好
        #    2026-09-30 加固：原先 value=<服务端key> 会把真 key 随页面发到浏览器，
        #    链接一旦公开，任何人打开开发者工具（或点"眼睛"图标）就能读走。
        #    现在恒为空，服务端 key 仍照用（api_key or None 回落到环境变量）。
        #    2026-10-30：页面上只有**一个**密钥框（第二组视觉密钥已整块删除）。
        pws = page.locator("input[type=password]")
        n = pws.count()
        note(f"## 3. 密钥输入框（共 {n} 个密码框）—— 期望**只有 1 个且为空**")
        vals = [pws.nth(i).input_value() for i in range(n)]
        for i, v in enumerate(vals):
            label = "模型 API" if i == 0 else f"第{i + 1}个（多出来的，不该有）"
            if v:
                note(f"   - 第 {i + 1} 个({label}): ❌ 被预填了（长度 {len(v)}）"
                     f"—— key 会随页面发到浏览器，存在泄漏风险")
                errors.append(f"{label} key 输入框被预填，key 会被发到浏览器")
            else:
                note(f"   - 第 {i + 1} 个({label}): ✅ 空（正确，key 不进浏览器）")
        if n != 1:
            errors.append(f"密钥输入框应为 1 个（页面只留一组），实际 {n} 个")
        # 服务端 key 的状态仍必须让用户看得见（靠文案，而不是靠预填）
        has_srv_note = "服务端" in body2
        note(f"   - 服务端 key 状态文案可见: {'✅' if has_srv_note else '❌'}")
        if not has_srv_note:
            errors.append("缺少『服务端 key 状态』说明文案（用户无法判断是否已配置）")

        # 4) HTML 泄漏 + 控制台错误
        leaks = [m for m in ("</div>", "class='", "unsafe_allow_html") if m in body2]
        note(f"## 4. HTML 标记泄漏: {'✅ 0 处' if not leaks else '❌ ' + str(leaks)}")
        if leaks:
            errors.append(f"HTML 泄漏: {leaks}")
        note(f"## 5. 浏览器控制台错误: {'✅ 0 条' if not console_errs else '❌ ' + str(console_errs[:3])}")

        page.screenshot(path=str(OUT / "09_keys_status.png"), full_page=True)
        b.close()

    # 6) 源码守卫：预填模式必须在页面上写明"会发到浏览器、仅本机自用"
    #    守的是"以后有人把预填变成默认、又不给提示" —— 那才是真正会出事的样子。
    src = (HERE / "app.py").read_text(encoding="utf-8", errors="replace")
    has_flag = "PAPER2LAB_PREFILL_KEYS" in src
    has_warn = ("随页面发到浏览器" in src) or ("仅本机自用" in src)
    note("## 6. 预填开关与风险提示（源码守卫）")
    note(f"   - 有开关 `PAPER2LAB_PREFILL_KEYS`: {'✅' if has_flag else '❌'}")
    note(f"   - 预填时页面写明风险: {'✅' if has_warn else '❌'}")
    if not has_flag:
        errors.append("app.py 里找不到 PAPER2LAB_PREFILL_KEYS 开关（预填无法关闭）")
    if not has_warn:
        errors.append("app.py 里找不到预填的风险提示文案（会静默泄漏 key）")

    note("")
    note(f"## 结论: {'✅ 全部通过' if not errors else '❌ ' + str(len(errors)) + ' 项异常'}")
    for e in errors:
        note(f"   - {e}")
    (OUT.parent / "ui_keys_report.md").write_text("\n".join(log), encoding="utf-8")
    print("REPORT SAVED")
    # ⚠ 必须有退出码：本项目已四次踩过"脚本打印 ❌ 但退出 0"，
    #   全套回归会把它当通过（2026-10-26 又踩一次，本次补上）。
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
