"""云端端到端复现：真跑一轮审计，紧接着就问一问 —— 看会不会 429。

为什么要这个（2026-10-31 凌晨）：
    用户第三次说「dsh 密钥有钱，怎么可能不行」，界面却一直显示
    「模型调用失败（RateLimitError）」。命令行单独问一次是通的（7.4s），
    所以必须复现**他的真实时序**：审计刚跑完十几次模型调用，紧接着问答。

判读：
    · 审计 ok + 问答 ok   → 这条时序没问题，界面上那条是**历史消息**或当时的瞬时抖动；
    · 审计 ok + 问答 429  → 真撞上限流，看 `cache/qa_errors.log` 里的 status/body。

用法（默认读 .env，也可以临时压成云端）：
    set LLM_API_KEY=sk-xxx & set LLM_BASE_URL=https://api.deepseek.com & set LLM_MODEL=deepseek-chat
    .venv\\Scripts\\python.exe probes\\probe_cloud_e2e.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import envfile, llm, qa                        # noqa: E402
from paper2lab.graph import run_audit_multagent              # noqa: E402


def main() -> int:
    k, u, m = llm.resolve_endpoint()
    print(f"本次通道：{u}　模型：{m}　key：{(k or '')[:6]}…（{len(k)} 位）")
    if llm.is_local_endpoint(u):
        print("⚠ 现在指向本机。要复现云端请先用环境变量把 LLM_* 压成云端。")
    print()

    demo = ROOT / "demo"
    pdf, code = demo / "paper.pdf", demo / "project.zip"
    if not pdf.exists():
        print(f"缺样例：{pdf}")
        return 1

    print("① 跑一轮完整审计（多智能体档，和网页默认一致）…")
    t0 = time.time()
    try:
        audit, _report = run_audit_multagent(str(pdf), str(code))
    except Exception as ex:                        # noqa: BLE001
        print(f"   审计异常：{type(ex).__name__}: {ex}")
        return 1
    sec = time.time() - t0
    if audit is None:
        print(f"   审计返回空（{sec:.1f}s）")
        return 1
    n = len(getattr(audit, "findings", []) or [])
    print(f"   完成：{sec:.1f}s，{n} 条 finding")

    print("\n② 紧接着问一问（不等冷却，模拟用户真实操作）…")
    t1 = time.time()
    body, note = qa.answer(audit, "这次审计有没有漏掉什么？", [],
                           api_key=k, base_url=u, model=m)
    used = time.time() - t1
    print(f"   耗时 {used:.1f}s")
    print(f"   说明：{note}")
    print(f"   正文 {len(body)} 字，前 120 字：{body[:120]}")

    bad = "调用失败" in note
    print("\n—— 结论 ——")
    if bad:
        print("  问答真的失败了。看 cache/qa_errors.log 里的 status 与 body，那是原始凭据。")
        return 1
    print(f"  审计 {sec:.0f}s + 问答 {used:.0f}s 全程打通，没有 429。")
    print("  那么界面上那条「调用失败」是对话历史里的旧消息 —— 点「清空对话」即消失。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
