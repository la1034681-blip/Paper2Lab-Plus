"""真机验证：模型 API 区**只有一组**输入（Key / Base URL / 模型名），
多模态与否由程序按模型名**自动识别**，并把「能读什么、要注意什么」写在下面。

用户口径（2026-10-30）：
  "在填写 API 模型密钥的配置区域，只需提供一个用于填写 LLM 的 API 密钥、API URL 和
   模型名称的输入项。系统需根据填写的模型名称自动识别该模型是多模态还是单模态，
   并据此在下方显示相应的提示信息，说明该模型支持的输入类型及使用注意事项。"

仍然成立的老口径（2026-09-30）：支持任意 OpenAI 兼容模型；多模态不用视觉 key，
单模态才需要；用没视觉能力的模型时**如实提示无法读图**，不许假装读过图。

分两层验证（都真跑，不是"应该没问题"）：

第 1 层 · 纯函数：`paper2lab.llm.resolve_vision` 的每个分支逐一断言，外加
  `looks_multimodal` / `modality_hit`（识别结论 + 识别依据）。
  为什么必须单独测：本机 `.env` 里配了视觉 key，于是"完全没有读图通道"这类分支
  在**页面上永远走不到**（会被 .env 兜住）。线上 Secrets 是空的 —— 那才是评委看到
  的状态，所以这些分支必须在函数层锁死，不能靠"页面上试了一下没事"。

第 2 层 · 真机：起本地服务、用 Edge 打开，断言
  · 展开区**只有三个输入项 / 一个密码框**，且密码框为空（安全：key 不进浏览器）；
  · 「视觉模型 API Key」「视觉 Base URL」这两个框**已删除**（不再要第二组密钥）；
  · 填 gpt-4o → 提示"已识别为多模态"＋"文字 + 图片"＋读图由该模型自己完成；
  · 换 llama-3-70b → 提示"已识别为单模态"＋"只有文字"＋不再说由模型自己读图。
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                 # 脚本已不在项目根，需把根加回来
    sys.path.insert(0, str(ROOT))

PORT = 8521
URL = f"http://127.0.0.1:{PORT}"
fails: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}", flush=True)
    if not ok:
        fails.append(label)


# ================================================================ 第 1 层：纯函数分支
def layer1() -> None:
    from paper2lab.llm import resolve_vision as rv

    print("\n=== 第 1 层 · resolve_vision 分支（纯函数，不依赖 .env）===", flush=True)
    cases = [
        # 名字, 关键字参数, 期望 ok, 期望 source 片段, 期望 reason 片段
        ("① 页面成对填视觉 key+url（老调用方保留）",
         dict(page_key="vk", page_url="https://v/v1", page_model="mimo-v2.5"),
         True, "页面输入", ""),
        ("② 页面只填视觉 key、没填 url",
         dict(page_key="vk", page_url=""),
         False, "", "Base URL"),
        ("③ 服务端成对（.env 有 key+url）",
         dict(env_key="ek", env_url="https://e/v1", env_model="mimo-v2.5"),
         True, ".env", ""),
        ("④ 服务端厂商专用变量（有已知端点）",
         dict(env_key="ek", env_url="", env_key_has_known_url=True),
         True, ".env", ""),
        ("⑤ 服务端配了 key 却没 url",
         dict(env_key="ek", env_url="", env_key_has_known_url=False),
         False, "", "Base URL"),
        ("⑥ 单模态模型 + 无视觉配置 → 无法读图",
         dict(llm_key="lk", llm_url="https://l/v1", llm_model="llama-3-70b",
              llm_multimodal=False),
         False, "", "单模态"),
        ("⑦ 多模态模型 → 复用模型 API",
         dict(llm_key="lk", llm_url="https://l/v1", llm_model="gpt-4o",
              llm_multimodal=True),
         True, "复用模型 API", ""),
        ("⑧ 自称多模态却没填 Base URL → 不能复用",
         dict(llm_key="lk", llm_url="", llm_model="gpt-4o", llm_multimodal=True),
         False, "", "Base URL"),
        ("⑨ 什么都没配",
         dict(),
         False, "", "未配置任何模型 API"),
    ]
    for name, kw, want_ok, want_src, want_reason in cases:
        r = rv(**kw)
        good = (r["ok"] == want_ok
                and (want_src in r["source"] if want_src else r["source"] == "")
                and (want_reason in r["reason"] if want_reason else r["reason"] == ""))
        check(good, f"{name} → ok={r['ok']} source={r['source'] or '-'} "
                    f"reason={r['reason'][:34] or '-'}")

    # 多模态识别：这是"要不要额外配视觉 key"的唯一判据（页面已无勾选框可覆盖），
    # 判错的代价不对称 —— 猜成能读图而实际不能，读图会整片失败。
    from paper2lab.llm import looks_multimodal as lm
    check(lm("gpt-4o") and lm("qwen-vl-max") and lm("glm-4v-plus") and lm("deepseek-chat"),
          "视觉模型名被识别为多模态（gpt-4o / qwen-vl-max / glm-4v-plus / deepseek-chat）")
    check(lm("intern-s2") and lm("intern-s1-pro"),
          "书生系列被识别为多模态（intern-s2 / intern-s1-pro，比赛要求的那条线）")
    # 2026-10-24：`deepseek-v4-pro` 收图**不报错**、只回 content=""（reasoning 里说
    # "Unsupported Image"）→ 必须是 False，否则读图通道会"看起来配好了、实际一条没读到"。
    # `mimo-v2.5` 已下线（服务端 400 Unsupported model），不再当成视觉模型。
    check(not lm("deepseek-v4-pro") and not lm("llama-3-70b")
          and not lm("mimo-v2.5") and not lm(""),
          "纯文本模型名**不**被误判为多模态（v4-pro / llama / 已下线的 mimo / 空名）")

    # 识别依据要说得出来（页面会写"依据：模型名含 xxx"）—— 只给结论不给依据，
    # 用户换模型后结果变了就只能怀疑程序坏了。
    from paper2lab.llm import modality_hit as mh
    check(mh("qwen-vl-max") == "-vl" and mh("gpt-4o") == "gpt-4o"
          and mh("llama-3-70b") == "",
          "识别依据可复述：qwen-vl-max→`-vl`、gpt-4o→`gpt-4o`、llama→空")
    check(mh("  GPT-4O  ") == "gpt-4o", "模型名带空格/大写也能识别（先去空白再小写）")


# ================================================================ 第 2 层：真机
def layer2() -> None:
    print("\n=== 第 2 层 · 真机（Edge 打开本地页面）===", flush=True)
    # ⚠ 本机开了「从 .env 预填密钥」(`PAPER2LAB_PREFILL_KEYS=1`) 之后，下面几条断言
    #    会全部失效 —— 它们都假设"一进来输入框是空的，测试自己往里填"。这里**显式关掉**
    #    预填再验（`.env` 不覆盖已存在的环境变量，见 paper2lab/envfile.py）。
    #    预填模式本身另有守卫：tests/verify_keys_ui.py 第 6 段查开关与风险提示。
    _env = dict(os.environ, PAPER2LAB_PREFILL_KEYS="0")
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-X", "utf8", "-m", "streamlit",
         "run", "app.py", "--server.headless", "true", "--server.address", "127.0.0.1",
         "--server.port", str(PORT), "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_env)
    try:
        up = False
        for _ in range(40):
            try:
                with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=3) as r:
                    if r.status == 200:
                        up = True
                        break
            except Exception:      # noqa: BLE001
                time.sleep(2)
        check(up, "界面服务已起来")
        if not up:
            return

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch(channel="msedge", headless=True)
            page = b.new_page(viewport={"width": 1500, "height": 1200})
            page.goto(URL, wait_until="domcontentloaded")
            # 等**目标文本**，不要等"body 变长"（分帧渲染会假通过）
            for _ in range(30):
                if "上传论文资料" in page.inner_text("body"):
                    break
                page.wait_for_timeout(1000)

            body = page.inner_text("body")
            check("模型 API" in body, "输入区已用通用名「模型 API」（不再写死 DeepSeek）")
            check("归因引擎：DeepSeek" not in body,
                  "主页面状态行不再写死 DeepSeek 品牌（改为实际模型名）")

            exp = page.get_by_text("高级设置", exact=False)
            if exp.count():
                exp.first.click()
                page.wait_for_timeout(1800)
            body2 = page.inner_text("body")
            check("模型 Base URL（必填）" in body2, "有 Base URL 输入（必填）")
            check("模型名（必填）" in body2, "有模型名输入（必填）")

            # 用户口径的核心：**只留一组**，第二组视觉密钥整块删掉
            check("视觉模型 API Key" not in body2 and "视觉 Base URL" not in body2,
                  "**没有第二组视觉密钥输入**（视觉 Key / 视觉 Base URL 已从页面删除）")
            check("该模型本身可以读图" not in body2,
                  "没有多模态勾选框（改由模型名自动识别）")

            # 安全：唯一的密钥框必须为空
            pws = page.locator("input[type=password]")
            vals = [pws.nth(i).input_value() for i in range(pws.count())]
            check(pws.count() == 1 and all(v == "" for v in vals),
                  f"密钥框共 {pws.count()} 个且**为空**（key 不进浏览器）")

            box = page.locator("div[data-testid='stExpander']").first
            info = box.locator("input").evaluate_all(
                "els => els.map(e => e.type + ' | ' + (e.getAttribute('aria-label') || ''))")
            print(f"  （展开区 input 共 {len(info)} 个）", flush=True)
            for one in info:
                print(f"      {one}", flush=True)
            check(len(info) == 3, f"展开区只有三个输入项（Key / Base URL / 模型名，实际 {len(info)}）")

            def field(prefix: str):
                loc = box.locator(f"input[aria-label^='{prefix}']")
                if loc.count() == 0:
                    raise AssertionError(f"找不到输入项：{prefix}")
                return loc.first

            f_llm_key = field("模型 API Key")
            f_llm_url = field("模型 Base URL")
            f_llm_name = field("模型名")

            def setv(loc, val: str):
                """填值并**提交**。

                ⚠️ 坑：Streamlit 的文本框只在**回车或失焦**时才把值发回服务端，
                只调 fill() 改的是 DOM，服务端根本不知道 —— 表现就是"填了但页面没反应"，
                很容易被误判成产品 bug。这里必须补一次 Enter。
                """
                loc.fill(val)
                loc.press("Enter")
                page.wait_for_timeout(900)

            def wait_until(pred, secs: float = 25) -> tuple[bool, str]:
                """等到正文满足条件（Streamlit 分帧渲染，必须等**目标文本**，不能只 sleep）。"""
                end = time.time() + secs
                t = ""
                while time.time() < end:
                    t = page.inner_text("body")
                    if pred(t):
                        return True, t
                    page.wait_for_timeout(500)
                return False, t

            # --- 多模态模型：自动识别 + 说明"文字 + 图片"
            setv(f_llm_key, "dummy-model-key-not-used")
            setv(f_llm_url, "https://api.example.invalid/v1")
            setv(f_llm_name, "gpt-4o")
            ok_mm, t = wait_until(lambda s: "已识别为多模态模型" in s and "gpt-4o" in s)
            check(ok_mm, "填多模态模型(gpt-4o) → **自动识别为多模态**（无需任何勾选）")
            check("文字 + 图片" in t, "提示写明**支持的输入类型**：文字 + 图片")
            check("依据" in t and "gpt-4o" in t, "提示写明**识别依据**（模型名含 gpt-4o）")
            check("要注意" in t, "提示写明**使用注意事项**")
            # 2026-10-30「通道状态汇总行」已整行删除（app.py 有注释），模型名并入
            # 「读图通道：已就绪 ✅ …· gpt-4o」这一块 —— 判据与 verify_keys_ui 同口径：
            # 「已就绪」之后 140 字内出现模型名。
            check(bool(re.search(r"读图通道：已就绪.{0,140}gpt-4o", t, re.S)),
                  "读图通道块同步显示 已就绪 + 模型名（gpt-4o）")
            ok_reuse, t = wait_until(lambda s: "读图通道：已就绪 ✅" in s and "自己读图" in s)
            check(ok_reuse, "读图通道：由这个模型**自己读图**（不需要第二组密钥）")

            # --- 单模态模型：自动识别 + 说明"只有文字"、图片类不参与审计
            setv(f_llm_name, "llama-3-70b")
            ok_sm, t = wait_until(lambda s: "已识别为单模态模型" in s and "llama-3-70b" in s)
            check(ok_sm, "换单模态模型(llama-3-70b) → **自动识别为单模态**")
            check("只有文字" in t, "提示写明单模态支持的输入：只有文字")
            check("不参与审计" in t, "提示写明使用注意：图片 / 扫描页不参与审计")
            check("自己读图" not in t, "不再说「由模型自己读图」（多模态提示已撤掉）")
            # 读图状态因机器而异：本机 .env 可能另配了视觉通道 → 两种如实文案都认
            ok_state, t = wait_until(
                lambda s: ("读图通道：已就绪" in s) or ("本次不读图" in s))
            check(ok_state, "单模态下读图状态如实给出（走服务端视觉 / 本次不读图）")

            b.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:      # noqa: BLE001
            proc.kill()


if __name__ == "__main__":
    layer1()
    layer2()
    print("\n" + "=" * 60, flush=True)
    if fails:
        print(f"结论：❌ {len(fails)} 项未通过", flush=True)
        for f in fails:
            print(f"   - {f}", flush=True)
        sys.exit(1)
    print("结论：✅ 全部通过", flush=True)
