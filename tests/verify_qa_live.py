"""「问一问」真机可用性与准确性体检（2026-10-30）。

用户口径：
  "围绕几个典型问题发起提问，检查回答是否相关、完整、准确，并说明响应速度、
   交互流程和整体体验，最后汇总测试结果与发现的问题。"

为什么不能只跑离线那套（tests/verify_qa.py）：
  离线只验「口径」（事实从哪来、没 key 会不会编），验不了**真机流程**（点不点得开、
  卡不卡、多轮历史在不在）也验不了**真回答好不好**。这里两档都跑：

  第 1 档（默认配置）——用 .env 里的默认模型（deepseek-chat）问第一句。**这一档自适应**
                      两种结果都算合格，但判据不同：
                        · 答得出 → 判「相关 + 有实质内容 + 数字有出处」；
                        · 调不通 → 判「如实说调用失败 + 摊开本轮证据 + 不编」。
                      以前这里写死"账号欠费（402）必然失败"，充值后就变成假红了 ——
                      别把外部账号状态当成断言前提（2026-10-30 充值后改）。
  第 2 档（切书生） ——在页面上填书生的 Key/URL/模型名（intern-s1），再问同样几类
                      问题，这才是"模型作答"档：判相关性 / 完整性 / 准确性。
  第 2b 档（s2）    ——换成 intern-s2（会把推理写进正文的那个）：平台必须兜住 ——
                      不许把英文思考过程当答案丢给用户，要退回证据原文并说明。
  第 3 档（无结论） ——新开一个会话、不跑审计直接问，看它会不会把"没有数据"编成答案。

准确性怎么判（不靠"看着像"）：
  · 先在本进程里跑一遍**完全相同的审计**（rule 档，几秒），用 `qa.facts()` 拼出
    「这一轮的事实全集」当基准；
  · 回答里出现的每个数字，都要在这份事实全集里找得到 —— 找不到的单列出来人工复核，
    这才是"有没有编数"的硬判据，而不是让模型自证。

⚠ 两个判据坑（都踩过，别再改回去）：
  · **"审计跑完"不能用 `text=逐条对表` 判** —— 左侧导航里就有这个页名，一秒就命中，
    结果在审计还在跑的时候就点了问答，把这一轮打断，问答拿到 audit=None，
    所有回答都变成"还没有审计结论"。正解：等顶栏状态变成「● 已完成」（它等价于
    `session_state["audit"] is not None`）。
  · **说明行要取最后一条** —— 多轮里前面那句"模型调用失败"一直在历史里，
    取第一条会把后面的"模型作答"误判成失败。

用法:
    .venv\\Scripts\\python.exe tests\\verify_qa_live.py
退出码: 0 全过 / 1 有异常（⚠ 必须有退出码，本项目踩过"打印❌却退 0"四次）
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8513
URL = f"http://127.0.0.1:{PORT}"
DLG = 'div[data-testid="stDialog"]'
OUT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集\ui_screenshots")
OUT.mkdir(parents=True, exist_ok=True)
LOGF = ROOT / ".trash_tmp" / "qa_live.json"

fails: list[str] = []
skipped: list[str] = []
log: list[str] = []
records: list[dict] = []

#: 模型额度/配额被打光时，后续断言转 [SKIP]（不计失败，也不算通过）。
#: 2026-10-30 实测：书生免费账号会报 400 `-20080 您已超过输入 tokens 配额`，
#: 连 8 token 的探测都过不去 —— 那是**账号额度**，不是我们代码的问题。
#: 额度状态不该让测试红成一片，但也不能假装通过，所以单列一档。
SKIP_MODE = [False]


def check(ok: bool, label: str, detail: str = "") -> None:
    assert isinstance(ok, bool), f"断言值必须是 bool：{label}"
    if SKIP_MODE[0]:
        print(f"  [SKIP] {label}" + (f" —— {detail}" if detail else ""), flush=True)
        log.append(f"SKIP | {label}")
        skipped.append(label)
        return
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f" —— {detail}" if detail else ""),
          flush=True)
    log.append(f"{'OK' if ok else 'FAIL'} | {label}" + (f" —— {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def note(s: str) -> None:
    print(s, flush=True)
    log.append(s)


# ================================================================ 基准事实
def build_ground_truth():
    from paper2lab import qa
    from paper2lab.pipeline import run_audit

    # ⚠ 必须带上 demo/results.json：页面的内置样例是「论文 + 代码 + 实测结果」三件套
    #   （app.py 里读的），只传论文+代码少一条"结果核验"，基准就会缺 0.38 / 91.7 / 94.3
    #   这些合法原值 —— 那是**基准不全**，会被误报成"模型编数"（第一版就这样错过一次）。
    import json as _json
    results = _json.loads((ROOT / "demo/results.json").read_text(encoding="utf-8"))
    t0 = time.time()
    audit, report = run_audit(str(ROOT / "demo/paper.pdf"), str(ROOT / "demo/project.zip"),
                              results, api_key=None)
    cost = time.time() - t0
    corpus = qa.facts(audit, {"source": "内置样例",
                              "files": "demo/paper.pdf + demo/project.zip + demo/results.json"},
                      report)
    qs = ("这几个数为什么对不上", "我要复现这篇论文，最先该改哪里",
          "这次审计有没有漏掉什么", "假设卡和实验卡都说明了什么",
          "学习率在论文和代码里各是多少", "那第 2 条具体该改哪里", "seed")
    for q in qs:
        corpus += "\n" + qa.focus(audit, q)
        # ⚠ 降级档用的是 top_hits（按问题挑几条最相关的），它会把**一致项**也列出来，
        #   而 facts() 只列不一致项 —— 不把 top_hits 算进基准，就会把合法原值
        #   （0.38 / 91.7 这些结果数）误报成"编造"。这是基准不全，不是产品问题。
        corpus += "\n" + "\n".join(qa.top_hits(audit, q, k=8))
    note(f"基准：进程内同一份审计 {cost:.1f}s，findings {len(audit.findings)} 条，"
         f"就绪度 {(audit.readiness or {}).get('total')}，事实全集 {len(corpus)} 字")
    return audit, report, corpus


# ⚠ 必须认科学计数法：原正则 `\d+(?:\.\d+)?` 会把 `1e-2` 拆成「1」「2」、`1e-3` 拆成
#   「1」「3」，两个相邻的抽出来就拼成「31」这种根本不存在的数字 —— 2026-10-30 实测
#   因此误报过「模型编数」，实际是校验器自己拆错了。改数字正则时先跑一下
#   `numbers_in('1e-2~1e-3')`，结果必须是 ['1e-2', '1e-3']。
_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_SMALL = {str(i) for i in range(0, 30)}


def numbers_in(text: str) -> list[str]:
    return _NUM.findall(text or "")


def _unmatched_from(nums: list[str], corpus: str) -> list[str]:
    """给一串数字，挑出事实全集里找不到的（0–29 这类序数/小整数不算）。

    额外放行**由两个原值直接算出来的结果**，两种都算：
      · 商 —— 问「差几倍」时 0.001 ÷ 1e-4 = 10 是对的；
      · 差 —— 问「差多少」时 94.3 − 91.7 = 2.6 个百分点也是对的。
    以前只放行商，于是「差 2.6 个百分点」被报成编数（2026-10-30 实测误报）。
    """
    have = set(numbers_in(corpus))
    vals: list[float] = []
    for x in have:
        try:
            v = float(x)
        except ValueError:
            continue
        if v >= 1e-4:                      # 太小的数不参与派生，免得小整数凑出各种结果
            vals.append(v)

    def derived(n: str) -> bool:
        try:
            v = float(n)
        except ValueError:
            return False
        if v < 1e-4:
            return False
        tol = max(0.005, v * 0.01)
        for a in vals:
            for b in vals:
                if b and abs(a / b - v) <= tol:              # 两原值之商
                    return True
                if abs(abs(a - b) - v) <= tol:               # 两原值之差
                    return True
        return False

    bad = []
    for n in nums:
        if n in have or n in _SMALL or derived(n):
            continue
        if any(n in h for h in have):      # 1e-4 会被拆开，允许"是某个事实数字的子串"
            continue
        bad.append(n)
    return sorted(set(bad))


def unmatched_numbers(answer: str, corpus: str) -> list[str]:
    """整段回答里的无出处数字 —— **观察用**，不单独判失败。

    放开「可以补领域常识 + 给方案」之后，整段严查必然误报：模型会写「训练 300 轮验证」
    （自己设计的验证量）、示例代码里的通道数 64（通用知识）。这些不是编造材料结论。
    真正要守的是 `unmatched_cited`（引用处的数字）和 `general_claims_material`（冒用出处）。
    """
    return _unmatched_from(numbers_in(answer), corpus)


def cited_numbers(answer: str) -> list[str]:
    """紧贴出处的数字（`Page N` / `xxx.py:行` 前面 40 字内）—— 真正「引用材料」的数字。"""
    keep, _ = _qa().split_general(answer)
    out: list[str] = []
    for m in re.finditer(r"(?:Page\s*\d+|[\w./\\-]+\.py:\s*\d+)", keep):
        out.extend(numbers_in(keep[max(0, m.start() - 40):m.start()]))
    return out


def unmatched_cited(answer: str, corpus: str) -> list[str]:
    """只查「引用处」的数字：方案里的量（训练 300 轮、通道数 64）不算编数。"""
    return _unmatched_from(cited_numbers(answer), corpus)


def _qa():
    from paper2lab import qa
    return qa


def unmatched_material(answer: str, corpus: str) -> list[str]:
    """只校验**材料侧**的数字：标了〔通用知识〕的句子不参与。

    放开「可以补领域常识」之后（2026-10-30 用户定版），常识里的数字本来就不在本材料里，
    整段一起查会把「这类任务常用 1e-3」误报成编数。但也不能整体放行 —— 所以按句切，
    材料侧照旧一个数字都不许编。
    """
    keep, _ = _qa().split_general(answer)
    return unmatched_numbers(keep, corpus)


def general_claims_material(answer: str, corpus: str = "") -> list[str]:
    """通用知识那几句里**编造**的材料出处。

    ⚠ 判据不是「不许出现出处」：拿真实出处做对比是正确的写法
    （例「〔通用知识〕…(论文 Page 1 Abstract 写的是 2.7M)」）。
    要抓的是**编出来的**出处 —— 那句里出现 `Page N` / `xxx.py:行`，
    但事实全集里根本没有这个位置。2026-10-30 实测被这条误判过一次。
    """
    _, gen = _qa().split_general(answer)
    if not gen:
        return []
    out: list[str] = []
    if not corpus:
        return out
    flat = re.sub(r"\s+", " ", corpus)
    for m in re.finditer(r"Page\s*(\d+)", gen):
        if f"Page {m.group(1)}" not in flat:
            out.append(m.group(0))
    for m in re.finditer(r"[\w./\\-]+\.py:\s*\d+", gen):
        if re.sub(r"\s+", "", m.group(0)) not in re.sub(r"\s+", "", flat):
            out.append(m.group(0))
    return out


def entities_of(audit) -> list[str]:
    out: list[str] = []
    for f in (audit.findings or []):
        out.append(str(getattr(f, "display_name", "") or getattr(f, "param_key", "") or ""))
        for ev in (getattr(f, "paper", None), getattr(f, "code", None)):
            if ev is None:
                continue
            out.append(str(getattr(ev, "raw_value", "") or ""))
            out.append(str(getattr(ev, "location", "") or ""))
    return [x for x in out if x and len(x) >= 2]


# ================================================================ 起服务
def start_server():
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    (ROOT / ".trash_tmp").mkdir(exist_ok=True)
    logf = open(ROOT / ".trash_tmp/_svc_qa_log.txt", "w", encoding="utf-8")
    env = dict(os.environ, no_proxy="localhost,127.0.0.1", NO_PROXY="localhost,127.0.0.1")
    proc = subprocess.Popen(
        [str(py), "-X", "utf8", "-m", "streamlit", "run", "app.py",
         "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT, env=env)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3) as r:
                if r.status == 200:
                    note(f"服务就绪：{URL}")
                    return proc
        except Exception:      # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("服务起不来")


def ask(pg, *, chip: str = "", text: str = "", pairs_before: int, timeout: float = 180):
    """提一个问题，等到"多了一对消息且不再转圈"，返回 (耗时, 回答, 说明, 消息对数)。"""
    t0 = time.time()
    if chip:
        pg.get_by_role("button", name=chip).first.click()
    else:
        ta = pg.locator(f"{DLG} textarea").first
        ta.click()
        ta.fill(text)
        pg.locator(DLG).get_by_role("button", name="发送").first.click()

    msgs = pg.locator(f'{DLG} [data-testid="stChatMessage"]')
    end = t0 + timeout
    while time.time() < end:
        pg.wait_for_timeout(400)
        t = pg.inner_text(DLG)
        try:
            n = msgs.count()
        except Exception:      # noqa: BLE001
            n = 0
        if "在翻这一轮的审计数据" not in t and n >= pairs_before + 2:
            break
    cost = time.time() - t0
    pg.wait_for_timeout(600)
    t = pg.inner_text(DLG)
    try:
        n = msgs.count()
        body = msgs.nth(n - 1).inner_text() if n else ""
    except Exception:      # noqa: BLE001
        body = ""
    found = re.findall(r"(模型作答[^\n]*|模型调用失败[^\n]*|未配置模型[^\n]*)", t)
    note_txt = found[-1].strip() if found else ""
    # 额度/配额被打光 → 后面的模型类断言全部转 SKIP（否则会红成一片，掩盖真正的问题）
    if "配额" in note_txt or "额度" in note_txt:
        SKIP_MODE[0] = True
    return cost, body, note_txt, n


def open_qa(pg) -> float:
    t0 = time.time()
    pg.get_by_role("button", name="问 一 问").first.click()
    pg.wait_for_selector(DLG, timeout=30000)
    pg.wait_for_timeout(900)
    return time.time() - t0


def fill_model(pg, key: str, url: str, model: str) -> None:
    # ⚠ 两处折叠，且**不能无脑点**：
    #   ① 跑完一轮后上传/设置整块收进「换材料 / 调整选项后重新审计」；
    #   ② 里面还有一层「高级设置」。
    #   折叠区是 toggle —— 已经展开了再点一次会把它**收回去**（第二次调用就这样失败过），
    #   所以一律"看不见才点"。
    key_input = pg.locator("input[aria-label^='模型 API Key']").first
    if not key_input.is_visible():
        outer = pg.get_by_text("换材料 / 调整选项后重新审计", exact=False)
        if outer.count():
            outer.first.click()
            pg.wait_for_timeout(1200)
    if not key_input.is_visible():
        exp = pg.get_by_text("高级设置", exact=False)
        if exp.count():
            exp.first.click()
            pg.wait_for_timeout(1200)

    def put(prefix: str, val: str):
        # 全页找就行（这几个 aria-label 是唯一的），不用先锁定在哪个折叠区里 ——
        # 折叠区嵌套两层之后 `.stExpander` 的 first/last 都不可靠。
        loc = pg.locator(f"input[aria-label^='{prefix}']").first
        loc.fill(val)
        loc.press("Enter")
        pg.wait_for_timeout(900)

    put("模型 API Key", key)
    put("模型 Base URL", url)
    put("模型名", model)
    pg.wait_for_timeout(1500)

    # ⚠ 必须让**服务端**认下来，不能只看输入框里的字：Streamlit 的文本框要回车/失焦
    #   才提交，rerun 期间第三次 put 偶尔丢掉 —— 输入框显示 s2、服务端还是 s1
    #   （实测："模型作答 · intern-s1" 却问的是 s2）。判据用页面状态行「归因引擎：模型名」。
    def landed() -> bool:
        try:
            pg.locator("body").click(position={"x": 6, "y": 6})
        except Exception:      # noqa: BLE001
            pass
        pg.wait_for_timeout(1500)
        return f"归因引擎：{model}" in pg.inner_text("body")

    if not landed():
        note(f"  （模型名没被服务端接收，重填一次 → {model}）")
        box2 = pg.locator("input[aria-label^='模型名']").first
        box2.fill(model)
        box2.press("Enter")
        for _ in range(6):
            if landed():
                break


def main() -> int:
    audit, report, corpus = build_ground_truth()
    ents = entities_of(audit)
    note(f"可识别实体 {len(ents)} 个（参数名/原文值/出处）")

    from paper2lab.envfile import load_env, get
    load_env()
    ik, iu = (get("INTERNLM_API_KEY") or "", get("INTERNLM_BASE_URL") or "")
    # ⚠ 主测用 intern-s1：实测 s2 会把推理写进正文（见第 3 档），s1 是直接作答的。
    #   这不是"挑好模型测"，而是把两件事分开测：s1 测「答得好不好」，
    #   s2 单独测「吐草稿时平台怎么兜」。
    im = "intern-s1"
    im_think = "intern-s2"

    proc = start_server()
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            pg = b.new_page(viewport={"width": 1680, "height": 1200})
            pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
            pg.wait_for_selector("text=上传论文资料", timeout=90000)
            pg.wait_for_timeout(1200)

            # ---------------- 0 入口
            btn = pg.get_by_role("button", name="问 一 问").first
            check(btn.count() > 0, "首屏有「问 一 问」入口")
            box = btn.bounding_box() or {}
            check(bool(box.get("y", 999) < 160 and box.get("x", 0) > 900),
                  "入口在首屏右上角（不用滚就能看见）",
                  f"x={round(box.get('x', -1))} y={round(box.get('y', -1))}")

            # ---------------- 1 跑一轮真审计
            note("\n=== 跑一轮审计（之后所有问答都以这轮为准） ===")
            t_audit = time.time()
            pg.get_by_role("button", name="开 始 科 研 审 计").first.click()
            # ⚠ 唯一可信的完成判据：顶栏「● 已完成」（等价于 session_state["audit"] 已写入）
            pg.wait_for_selector("text=● 已完成", timeout=600000)
            pg.wait_for_timeout(2500)
            audit_cost = time.time() - t_audit
            check(bool("● 已完成" in pg.inner_text("body")),
                  "审计真的跑完（顶栏「● 已完成」）", f"{audit_cost:.1f}s")

            # ---------------- 2 打开弹窗
            note("\n=== 交互流程（打开 / 上下文 / 快捷问题） ===")
            open_cost = open_qa(pg)
            d0 = pg.inner_text(DLG)
            check(open_cost < 12, "弹窗打开够快", f"{open_cost:.1f}s")
            check(bool("对表" in d0 and "就绪度" in d0),
                  "弹窗顶部写明「拿哪份材料、几条、就绪度」（用户知道在问什么）",
                  d0.split("\n")[1][:40] if "\n" in d0 else d0[:40])
            n_quick = sum(1 for q in ("这几个数为什么对不上？", "我要复现这篇论文，最先该改哪里？",
                                      "这次审计有没有漏掉什么？", "假设卡和实验卡都说明了什么？")
                          if pg.get_by_role("button", name=q).count())
            check(n_quick == 4, "四个快捷问题都在（新用户不用想怎么问）", f"{n_quick}/4")
            pg.screenshot(path=str(OUT / "qa_live_01_open.png"))

            # ---------------- 3 第 1 档：默认模型 —— 能答就答，答不了不许编
            note("\n=== 第 1 档 · 默认模型（deepseek-chat）：能答就答，答不了不许编 ===")
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c1, a1, n1, pairs = ask(pg, chip="这几个数为什么对不上？", pairs_before=pairs)
            check(c1 < 120, "第 1 问有响应（不是一直转圈）", f"{c1:.1f}s")
            rel1 = any(e in a1 for e in ents)
            bad1 = unmatched_cited(a1, corpus)      # 引用处的数字（硬判据）
            obs1 = unmatched_material(a1, corpus)   # 整段的无出处数字（观察，含方案量）
            failed1 = bool("模型调用失败" in (n1 or ""))
            if failed1:
                # 调不通的支线：必须如实说，并把这一轮的证据原样摊开
                note(f"  （默认模型这次调不通：{n1}）")
                check(bool(n1 and len(n1) > 8), "失败时给出明确原因（不是只写一句「失败」）", n1)
                check(rel1, "降级时摊开的是**这一轮的**证据（相关）",
                      [e for e in ents if e in a1][:3])
                check(not bad1, "降级时引用处的数字都有出处", f"待复核：{bad1}" if bad1 else "全部有出处")
                stage1 = "默认(降级)"
            else:
                # 通了（账号已充值）：按"真作答"的标准判 —— 这才是用户关心的一档
                check(bool("模型作答" in (n1 or "")), "默认模型真作答（不是降级、不是空转）", n1 or "-")
                check(rel1, "第 1 问点到这轮审计里的具体对象（相关）",
                      [e for e in ents if e in a1][:3])
                check(bool(len(a1) >= 60), "第 1 问有实质内容（不是一句话敷衍）", f"{len(a1)} 字")
                check(not bad1, "第 1 问引用处的数字都能对上事实（不编数）",
                      f"待人工复核：{bad1}" if bad1 else "引用处全部对上")
                stage1 = "默认"
            if obs1:
                note(f"  （观察：另有 {len(obs1)} 个数字属方案/常识、不在材料里：{obs1[:4]}）")
            records.append(dict(stage=stage1, q="这几个数为什么对不上？", cost=c1,
                                note=n1, answer=a1, rel=rel1, bad=bad1))
            pg.screenshot(path=str(OUT / "qa_live_02_fallback.png"))

            # ---------------- 3b 解释力 / 方案 / 来源标注（用**默认模型**验）
            # ⚠ 这一段原来挂在书生档上，2026-10-30 书生免费账号把「输入 tokens 配额」用光
            #   （400 -20080，连 8 token 都过不去），核心能力就跟着一起红。
            #   教训：**核心能力不能押在一个额度不稳定的账号上**。默认模型已充值，改挂这里。
            note("\n=== 第 1b 段 · 问「为什么」「怎么改」「材料之外」：解释力与来源标注 ===")
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            cw, aw, nw, pairs = ask(pg, text="学习率这个数为什么对不上？具体是什么原因造成的？",
                                    pairs_before=pairs, timeout=240)
            check(bool("模型作答" in (nw or "")), "问「为什么」由模型作答", nw or "-")
            whyw = [w for w in ("可能", "原因", "我判断", "推断", "因为") if w in aw]
            check(bool(whyw), "给出了成因解释（不是把审计事实念一遍）", whyw[:3])
            markw = [w for w in ("最可能", "我判断", "一种可能", "推断", "推测") if w in aw]
            check(bool(markw), "成因被标成「模型的判断」（不冒充材料原文）", markw[:3])
            badw = unmatched_cited(aw, corpus)
            check(not badw, "解释里引用处的数字都能对上事实（放开推断 ≠ 放开编数）",
                  f"待人工复核：{badw}" if badw else "引用处全部对上")

            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            cz, az, nz, pairs = ask(pg, text="那我该怎么改？给我能直接动手的步骤。",
                                    pairs_before=pairs, timeout=240)
            check(bool("模型作答" in (nz or "")), "问「怎么改」由模型作答", nz or "-")
            actz = [w for w in ("train.py", ".py:", "default", "1e-4", "0.001") if w in az]
            check(bool(actz), "方案落到具体对象（文件/参数/改成什么）", actz[:3])
            check(bool(len(az) >= 80), "方案有实质篇幅（不是一句话打发）", f"{len(az)} 字")
            badz = unmatched_cited(az, corpus)
            check(not badz, "方案里引用处的数字都能对上事实",
                  f"待人工复核：{badz}" if badz else "引用处全部对上")

            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            cg, ag, ng, pairs = ask(
                pg, text="CIFAR-10 这类任务通常用什么学习率？和这篇比算高还是低？",
                pairs_before=pairs, timeout=240)
            check(bool("模型作答" in (ng or "")),
                  "问材料之外的常识：仍作答（不拿「没覆盖」打发）", ng or "-")
            keepg, geng = _qa().split_general(ag)
            check(bool(geng), "补的外部常识被标成「〔通用知识〕」（不冒充材料结论）",
                  f"{len(geng)} 字" if geng else "没标")
            fakeg = general_claims_material(ag, corpus)
            check(not fakeg, "通用知识段里没有编造的材料出处（引用真实页码做对比是允许的）",
                  fakeg or "无编造")
            badg = unmatched_cited(ag, corpus)
            check(not badg, "材料侧引用处的数字仍然都能对上事实",
                  f"待人工复核：{badg}" if badg else "引用处全部对上")
            check(bool("1e-4" in ag or "0.001" in ag),
                  "仍以本材料数值为准（没有拿通用常识替代结论）")
            records.append(dict(stage="默认", q="为什么 / 怎么改 / 材料之外（解释力）",
                                cost=cw + cz + cg, note=f"{nw} ｜ {nz} ｜ {ng}",
                                answer=ag, bad=badg))
            pg.screenshot(path=str(OUT / "qa_live_08_explain.png"))

            # ---------------- 4 切书生 intern-s1（会直接作答的那个），测"模型作答"档
            note("\n=== 第 2 档 · 切书生 intern-s1：真模型作答 ===")
            pg.get_by_role("button", name="结束问答").first.click()
            pg.wait_for_timeout(1200)
            fill_model(pg, ik, iu, im)
            adv = pg.inner_text("body")
            check("已识别为多模态模型" in adv or "已识别为单模态模型" in adv,
                  "切模型后自动重新识别模态（配置区提示同步）")
            check(bool(im in adv), "页面确实换成了书生模型名", im)

            open_cost2 = open_qa(pg)
            d1 = pg.inner_text(DLG)
            check(bool("这几个数为什么对不上" in d1),
                  "关掉再打开：前面的对话还在（不用重问一遍）", f"重开 {open_cost2:.1f}s")

            # 第 2 问：复现先改哪（手打 —— 快捷按钮只在对话刚开始时出现）
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c2, a2, n2, pairs = ask(pg, text="我要复现这篇论文，最先该改哪里？",
                                    pairs_before=pairs, timeout=240)
            check(bool("模型作答" in (n2 or "")), "第 2 问由模型作答（不是降级）", n2 or "-")
            rel2 = any(e in a2 for e in ents)
            check(rel2, "第 2 问点到这轮审计里的具体对象（相关）",
                  [e for e in ents if e in a2][:3])
            check(bool(len(a2) >= 60), "第 2 问有实质内容（不是一句话敷衍）", f"{len(a2)} 字")
            records.append(dict(stage="书生", q="我要复现这篇论文，最先该改哪里？", cost=c2,
                                note=n2, answer=a2, rel=rel2))
            pg.screenshot(path=str(OUT / "qa_live_03_internlm.png"))

            # 第 3 问：具体数值（准确性硬考）
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c3, a3, n3, pairs = ask(pg, text="学习率在论文和代码里各是多少？差几倍？",
                                    pairs_before=pairs, timeout=240)
            check(bool("模型作答" in (n3 or "")), "第 3 问由模型作答", n3 or "-")
            bad3 = unmatched_cited(a3, corpus)
            hit3 = [e for e in ents if e in a3]
            check(bool(hit3), "第 3 问引用了论文/代码里的原值或出处", hit3[:3])
            check(not bad3, "第 3 问引用处的数字都能对上事实（不编数）",
                  f"待人工复核：{bad3}" if bad3 else "引用处全部对上")
            records.append(dict(stage="书生", q="学习率在论文和代码里各是多少？差几倍？",
                                cost=c3, note=n3, answer=a3, rel=bool(hit3), bad=bad3))

            # 第 4 问：覆盖类（最容易被答成"没有相关条目"）
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c4, a4, n4, pairs = ask(pg, text="这次审计有没有漏掉什么？",
                                    pairs_before=pairs, timeout=240)
            if "模型调用失败" in (n4 or ""):
                note(f"  ⚠ 第 4 问首次失败（{c4:.1f}s），重试一次看是不是必现")
                pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
                c4b, a4b, n4b, pairs = ask(pg, text="这次审计有没有漏掉什么？",
                                           pairs_before=pairs, timeout=240)
                records.append(dict(stage="书生", q="这次审计有没有漏掉什么？（首次）",
                                    cost=c4, note=n4, answer=a4))
                c4, a4, n4 = c4b, a4b, n4b
            check(bool("模型作答" in (n4 or "")), "第 4 问由模型作答", n4 or "-")
            ok4 = bool("没有与这一问直接相关" not in a4 and len(a4) >= 60)
            check(ok4, "问「有没有漏」给出实质回答（不是一句「没有相关条目」）",
                  a4[:50].replace("\n", " "))
            records.append(dict(stage="书生", q="这次审计有没有漏掉什么？", cost=c4,
                                note=n4, answer=a4, rel=ok4))

            # 第 5 问：追问（考多轮上下文）
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c5, a5, n5, pairs = ask(pg, text="那第 2 条具体该改哪里？", pairs_before=pairs,
                                    timeout=240)
            check(bool("模型作答" in (n5 or "")), "第 5 问（追问）由模型作答", n5 or "-")
            check(bool(len(a5) >= 40), "追问有实质内容（不是空白/复读欢迎语）", f"{len(a5)} 字")
            records.append(dict(stage="书生", q="那第 2 条具体该改哪里？", cost=c5,
                                note=n5, answer=a5))

            # 第 6 问：换成 intern-s2（会把推理写进正文）→ 平台必须**兜住**，不许把草稿给用户
            note("\n=== 第 2b 档 · 换成 intern-s2（会吐思考过程）：平台怎么兜 ===")
            pg.get_by_role("button", name="结束问答").first.click()
            pg.wait_for_timeout(1200)
            fill_model(pg, ik, iu, im_think)
            cur = pg.inner_text("body")
            switched = bool(f"归因引擎：{im_think}" in cur)
            note(f"  （切换到 {im_think}：{'成功' if switched else '未生效 → 这一档只作观察，不计入通过'}）")
            open_qa(pg)
            pairs = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            c6, a6, n6, pairs = ask(pg, text="代码里随机种子固定了吗？", pairs_before=pairs,
                                    timeout=300)
            # 这一档只做**观察**，不计入通过：切模型要过 Streamlit 的"回车提交 + 整页
            # rerun"，脚本里偶发抢跑（页面状态行已显示 s2，问答拿到的还是上一轮的 s1）。
            # s2 的兜底逻辑由 tests/verify_qa.py 的 F 段离线锁死（喂真实原始输出）。
            note(f"  （本次这一问由 {n6 or '未知'} 作答，耗时 {c6:.1f}s，回答 {len(a6)} 字）")
            check(bool("Thinking Process" not in a6 and "Analyze the Request" not in a6),
                  "s2 的英文思考过程**没有**出现在回答里", f"{c6:.1f}s")
            # 两条都算合格：s2 短问题会直接作答（就照常给），长问题吐草稿（就退回证据）
            clean = "模型作答" in (n6 or "")
            fallback = any(w in (n6 or "") for w in ("已隐去", "思考", "调用失败"))
            check(bool(clean or fallback),
                  "s2 要么给干净答案、要么如实退回证据（两种都不吐草稿）",
                  (n6 or "-")[:46])
            check(bool(len(a6) > 20), "退回后仍有可读内容（证据原文，不是空白）", f"{len(a6)} 字")
            records.append(dict(stage="书生s2", q="代码里随机种子固定了吗？", cost=c6,
                                note=n6, answer=a6, rel=bool(len(a6) > 20)))
            pg.screenshot(path=str(OUT / "qa_live_06_s2.png"))

            # ---------------- 5 交互边界
            # 这几项不调模型，额度耗尽也必须照验 —— 先解除 SKIP
            SKIP_MODE[0] = False
            note("\n=== 交互边界（空提交 / 清空 / 关闭） ===")
            before = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            pg.locator(DLG).get_by_role("button", name="发送").first.click()
            pg.wait_for_timeout(2500)
            after = pg.locator(f'{DLG} [data-testid="stChatMessage"]').count()
            check(after == before, "空问题点发送：不产生空对话（不浪费一次调用）",
                  f"{before}→{after}")

            pg.get_by_role("button", name="清空对话").first.click()
            cleared = False
            for _ in range(12):
                pg.wait_for_timeout(1000)
                if "我只看这一轮审计算出来的东西" in pg.inner_text(DLG):
                    cleared = True
                    break
            check(cleared, "清空对话后回到欢迎语（真的清空）")

            pg.get_by_role("button", name="结束问答").first.click()
            pg.wait_for_timeout(1800)
            check(pg.locator(DLG).count() == 0, "点「结束问答」真的关掉弹窗")
            deck = pg.inner_text("body")
            check(bool("● 已完成" in deck), "关掉弹窗后结论页还在（没把页面吞掉）")
            pg.screenshot(path=str(OUT / "qa_live_04_after_close.png"))

            # ---------------- 6 第 3 档：没跑审计就来问（会不会编）
            note("\n=== 第 3 档 · 新会话未跑审计 + 已配模型：会不会把「没有数据」编成答案 ===")
            ctx2 = b.new_context(viewport={"width": 1680, "height": 1200})
            pg2 = ctx2.new_page()
            pg2.goto(URL, wait_until="domcontentloaded", timeout=90000)
            pg2.wait_for_selector("text=上传论文资料", timeout=90000)
            pg2.wait_for_timeout(1200)
            fill_model(pg2, ik, iu, im)
            open_qa(pg2)
            check(bool("还没有审计结论" in pg2.inner_text(DLG)),
                  "空态弹窗顶部就写明「还没有审计结论」")
            t0 = time.time()
            ta = pg2.locator(f"{DLG} textarea").first
            ta.click()
            ta.fill("学习率在论文和代码里各是多少？")
            pg2.locator(DLG).get_by_role("button", name="发送").first.click()
            end = time.time() + 90
            body2 = ""
            note2 = ""
            while time.time() < end:
                pg2.wait_for_timeout(600)
                t = pg2.inner_text(DLG)
                m2 = pg2.locator(f'{DLG} [data-testid="stChatMessage"]')
                if "在翻这一轮的审计数据" in t or m2.count() < 2:
                    continue
                body2 = m2.nth(m2.count() - 1).inner_text()
                mm = re.findall(r"(模型作答[^\n]*|模型调用失败[^\n]*|未配置模型[^\n]*|未跑审计[^\n]*)", t)
                note2 = mm[-1].strip() if mm else ""
                break
            cost6 = time.time() - t0
            says_no = bool("还没有审计结论" in body2 or "先跑" in body2
                           or "没有覆盖" in body2 or "没有数据" in body2)
            bad6 = unmatched_numbers(body2, corpus)
            check(says_no, "没有结论时明确说「先跑一轮」，不硬答",
                  body2[:50].replace("\n", " "))
            check(bool("未跑审计" in note2), "没有结论时**根本不调模型**（省一次钱，也不给假出处）",
                  note2 or "-")
            check(not bad6, "没有结论时**不报任何具体数值/假出处**（不编）",
                  f"凭空出现：{bad6}" if bad6 else "无凭空数字")
            records.append(dict(stage="无审计", q="学习率在论文和代码里各是多少？", cost=cost6,
                                note=note2, answer=body2, rel=says_no, bad=bad6))
            pg2.screenshot(path=str(OUT / "qa_live_05_no_audit.png"))
            ctx2.close()
            b.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:      # noqa: BLE001
            proc.kill()

    note("\n=== 响应速度 ===")
    for r in records:
        note(f"  {r['cost']:6.1f}s | {r['stage']:9s} | {r['q'][:24]:24s} | "
             f"{(r.get('note') or '')[:24]}")

    LOGF.write_text(json.dumps(dict(log=log, records=records, fails=fails, skipped=skipped),
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 62, flush=True)
    if skipped:
        # 额度/配额导致的跳过必须单列：既不能算通过（没验到），也不该算失败（不是我们的错）
        print(f"⚠ {len(skipped)} 项因「模型额度/配额不足」跳过，未验证：", flush=True)
        for s in skipped:
            print(f"   · {s}", flush=True)
    if fails:
        print(f"结论：❌ {len(fails)} 项未通过（另有 {len(skipped)} 项跳过）", flush=True)
        for f in fails:
            print(f"   - {f}", flush=True)
        return 1
    print(f"结论：✅ 全部通过（另有 {len(skipped)} 项因额度跳过）" if skipped
          else "结论：✅ 全部通过", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
