"""ask_json 的两条硬口径守卫（离线，不联网、不花钱）。

守的是什么
----------
2026-10-31 用本机模型跑「独立功能」时抓到两个真 bug，都在 `llm.ask_json` 里：

  ① **参数收了不用**：函数签名有 `base_url` / `model`，函数体却只调
     `_call_deepseek(prompt, api_key, max_tokens=...)` —— 端点和模型一律回落到 .env。
     后果不是报错，而是**静默用错模型**：页面里填云端、排计划却打本机（或反过来），
     用户与日志都对不上。影响 9 个调用点（排计划 / 模型先判 / 角色 / 自主回头 /
     补丁提名 / 引证核验 / 实验站挑工具 / 抽取 Agent）。
  ② **超时写死 30s**：本机 8B 实测约 10 token/s，900~1400 token 的 JSON 要 90~140s，
     于是**必定超时** → `planner` 永远回落 `rule`、模型先判永远"没判出结果"，
     而界面只会说"模型没参与"，看不出是超时。

这里用替身把两条都钉住：不联网也能复现"转发没转发、超时算没算对"。
  A 源码层：ask_json 调用 `_call_deepseek` 时必须显式带 base_url / model / timeout
  B 行为层：给一个假端点，确认调用时**真的**用上了传进来的地址与模型
  C 超时折算：本机按 max_tokens 放大、远程维持 30s、且有上下限
  D 不越界：本机判定只认本机地址，云端地址不许被误判成本机
"""
from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import llm                                   # noqa: E402

N = 0
BAD: list[str] = []


def check(ok, label: str) -> None:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        BAD.append(label)


def main() -> int:
    src = inspect.getsource(llm.ask_json)
    # ⚠ 必须先把注释行剔掉：函数里那段说明就写着 `_call_deepseek(prompt, api_key, ...)`，
    #   直接搜会把**注释**当成真调用，判出"没转发"的假阴性（2026-10-31 本轮踩过）。
    code = "\n".join(x for x in src.splitlines() if not x.lstrip().startswith("#"))

    # ---------------------------------------------------------------- A 源码层
    print("A 源码层：转发有没有写")
    calls = re.findall(r"_call_deepseek\((.*?)\)", code, flags=re.S)
    call = max(calls, key=len) if calls else ""
    flat = call.replace(" ", "").replace("\n", "")
    check("base_url=base_url" in flat, "调用时传了 base_url（不再吃 .env 默认）")
    check("model=model" in flat, "调用时传了 model")
    check("timeout=_timeout" in flat, "调用时传了按端点算出的 timeout")
    check("is_local_endpoint(base_url)" in code.replace(" ", ""),
          "超时按端点区分（本机 / 远程）")
    check("min(300.0" in code or "min(300," in code, "本机超时有上限（不许无限等）")

    # ---------------------------------------------------------------- B 行为层
    print("\nB 行为层：转发真的生效吗")
    seen: dict = {}
    orig = llm._call_deepseek

    # ⚠ mode 必须每次都不一样：`ask_json` 自带结果缓存（键 = 提示词 + mode），
    #   用固定 mode 时第二次运行直接命中缓存、**根本不调** `_call_deepseek`，
    #   于是在替身里看到的全是空值，判出"没转发"的假阴性（2026-10-31 踩过）。
    import time as _t
    tag = _t.strftime("%H%M%S")

    def _fake(prompt, api_key, max_tokens=300, base_url="", model="", timeout=30.0,
              run_token=None):
        seen.update({"base_url": base_url, "model": model,
                     "timeout": timeout, "max_tokens": max_tokens})
        return '{"verdict":"一致","why":"替身"}'

    llm._call_deepseek = _fake
    try:
        obj, note = llm.ask_json("测试", "k", mode=f"guard-{tag}-1",
                                 base_url="https://api.deepseek.com",
                                 model="deepseek-chat")
        check(isinstance(obj, dict), "替身下能拿到 dict")
        check(seen.get("base_url") == "https://api.deepseek.com",
              "真的用了传进来的 base_url")
        check(seen.get("model") == "deepseek-chat", "真的用了传进来的 model")

        # ------------------------------------------------------------ C 超时折算
        print("\nC 超时折算")
        check(seen.get("timeout") == 30.0, "云端端点仍是 30s（老行为零变化）")

        llm.ask_json("测试2", "k", mode=f"guard-{tag}-2",
                     base_url="http://127.0.0.1:8080/v1",
                     model="intern-s1-mini", max_tokens=900)
        t_local = seen.get("timeout", 0)
        check(t_local >= 90, f"本机端点按 max_tokens 放大（900 token → {t_local}s）")
        check(t_local <= 300, "放大但有上限（不无限等）")

        llm.ask_json("测试3", "k", mode=f"guard-{tag}-3",
                     base_url="http://127.0.0.1:8080/v1",
                     model="intern-s1-mini", max_tokens=40)
        check(seen.get("timeout", 0) >= 90, "小额度也保底 90s（起手就要等模型加载）")
    finally:
        llm._call_deepseek = orig

    # ---------------------------------------------------------------- D 判定边界
    print("\nD 本机/远程判定")
    check(llm.is_local_endpoint("http://127.0.0.1:8080/v1"), "127.0.0.1 判成本机")
    check(llm.is_local_endpoint("http://localhost:8080/v1"), "localhost 判成本机")
    check(not llm.is_local_endpoint("https://api.deepseek.com"), "官方云端判成远程")
    check(not llm.is_local_endpoint("https://chat.intern-ai.org.cn/api/v1"),
          "书生云端判成远程（不是本机）")

    print("\n" + "=" * 58)
    print(f"共 {N} 项：通过 {N - len(BAD)}，未通过 {len(BAD)}")
    for x in BAD:
        print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
