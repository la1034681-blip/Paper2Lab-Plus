"""「点一次新的，就把上一次终止并清理干净」——离线验证。

背景（2026-10-30 晚，用户要求）：
    本地模型是单实例单槽，一问一答就要 2500+ token。上一次还在生成时又点一次，
    两个请求同时抢上下文 ⇒ 模型侧报 `Context size exceeded` / 429，
    用户看到「模型调用失败（RateLimitError）」而下面还挂着一屏审计事实 —— 很像程序坏了。
    现在的要求是：**每次点击都作废上一次，清理干净再跑新的**。

这一项只验不依赖浏览器的部分：
    A. 令牌机制本身：未受管辖的线程放行、同代通过、跨线程作废（含"不许拿自己跟自己比"）；
    B. 三层保护的源码守卫：本机不走代理 / 不重试 / 串行锁 / 返回前二次确认；
    C. 启动器参数守卫：8K 上下文 + 单槽 + 28 层（回退就会重现"调用失败"）。

用法：.venv\\Scripts\\python.exe -X utf8 tests\\verify_run_cancel.py
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.llm import (                     # noqa: E402
    RunCancelled, begin_run, check_run, current_run,
)

_ok = 0
_bad = 0


def check(cond, label: str, detail: str = "") -> None:
    global _ok, _bad
    ok = bool(cond)
    if ok:
        _ok += 1
        print(f"  OK   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}" + (f"  [{detail}]" if detail else ""))


print("===== A 令牌机制（行为，不看书）=====")

# A1 没调过 begin_run 的线程必须放行 —— 否则命令行调用、既有测试全被打挂
try:
    check_run()
    check(True, "未经 begin_run 的线程放行（命令行/测试不受管辖）")
except RunCancelled:
    check(False, "未经 begin_run 的线程放行", "被误判作废")

# A2 同线程同代通过
tok1 = begin_run()
try:
    check_run()
    check(True, f"本轮令牌通过（tok={tok1}）")
except RunCancelled:
    check(False, "本轮令牌通过", "误判作废")

check(current_run() == tok1, "本线程当前令牌 = 刚 begin_run 的那个")

# A3 另起线程 = 模拟"用户又点了一次" ⇒ 旧令牌作废
seen = {}


def _newer_click():
    seen["tok"] = begin_run()
    try:
        check_run()                 # 新线程自己这一轮必须通过
        seen["self_ok"] = True
    except RunCancelled:
        seen["self_ok"] = False


th = threading.Thread(target=_newer_click)
th.start()
th.join()
check(seen.get("self_ok") is True, "新线程自己这一轮通过（不会被自己作废）")

try:
    check_run(tok1)
    check(False, "旧令牌被作废", "没有抛 RunCancelled")
except RunCancelled:
    check(True, "旧令牌被作废（新的一次点击生效）")

# A4 回归守卫：不许写成"自己跟自己比"（那样恒等、永远不取消）
import paper2lab.llm as _llm          # noqa: E402

src = (ROOT / "paper2lab" / "llm.py").read_text(encoding="utf-8")
check("_RUN_LOCAL" in src and "threading.local()" in src,
      "令牌按线程分开存（threading.local），不是全局一个值")
check("if t is None:" in src or "if t is None：".replace("：", ":") in src,
      "无令牌线程走「放行」分支（不是当成 0 去比）")

print("\n===== B 三层保护的源码守卫 =====")
_call = src[src.find("def _call_deepseek"):src.find("def _call_deepseek") + 3000]
check("RunCancelled" in src and "class RunCancelled" in src, "llm 里定义了 RunCancelled")
check("trust_env=False" in _call, "本机端点不走代理（trust_env=False）")
check('kwargs["max_retries"] = 0' in _call, "本机端点不自动重试（max_retries=0）")
check("with _LOCAL_LOCK:" in _call, "本机端点串行（_LOCAL_LOCK）")
check("is_local_endpoint(url)" in _call, "按端点是否本机决定上面三条（远程不受影响）")

_qasrc = (ROOT / "paper2lab" / "qa.py").read_text(encoding="utf-8")
check("except RunCancelled:" in _qasrc, "问答把 RunCancelled 与「调用失败」分开处理")
check("check_run()" in _qasrc, "问答生成完再确认一次这一轮还算不算数")

_qisrc = (ROOT / "paper2lab" / "qaui.py").read_text(encoding="utf-8")
check("_begin_run()" in _qisrc, "点「发送」时开新一轮（作废上一次提问）")
check("except qa.RunCancelled" in _qisrc, "问答界面静默丢弃被作废的那一轮（不报错）")
check("hist.pop()" in _qisrc, "作废时撤掉那条提问，不留半截对话")

_appsrc = (ROOT / "app.py").read_text(encoding="utf-8")
check("_begin_run()" in _appsrc, "点「开始审计」时开新一轮（作废上一次运行）")
check("_check_run(_run_token)" in _appsrc, "审计结果落盘前再确认一次（旧结果不写进页面）")
check('st.session_state["audit"] = None' in _appsrc, "新的一次点击先清掉上一次的结论")

print("\n===== C 启动器参数守卫（回退就会重现「调用失败」）=====")
bat = Path(r"C:\Users\lijiahui\Desktop\书生本地模型\启动本地模型.bat")
if not bat.exists():
    print("  SKIP 启动器不在本机（换机器跑测试时会这样），跳过 C 组")
else:
    bsrc = bat.read_text(encoding="utf-8", errors="ignore")
    check("-c 8192" in bsrc, "上下文 8K（4K 时一问一答就占满，第二个请求必崩）")
    check("--parallel 1" in bsrc, "单推理槽（默认 4 槽共享上下文，多请求必崩）")
    check("-ngl 28" in bsrc, "28 层放显卡（32 层 + 8K 上下文会 OOM，实测过）")
    check("-c 4096" not in bsrc, "没有残留旧的 4096 参数")

print(f"\n===== 汇总：{_ok + _bad} 项，通过 {_ok}，未通过 {_bad} =====")
if _bad:
    sys.exit(1)
