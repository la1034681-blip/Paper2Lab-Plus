"""问一问「云端 429 限流」的处理口径（2026-10-30 晚第二版）。

背景（用户原话）：「我就是故意用 dsh 的密钥进行提问，不是本地」——
    所以这里**不许**把请求偷偷改到本机模型。曾经写过一版「云端失败就换本机」，
    已撤回：那等于把他特意选定的云端降级成 8B 小模型，还慢一百秒，比报错更糟。
    该做的是另一件事 —— 429 多数是瞬时的（审计刚跑完几十次模型调用、紧接着就问，
    同一分钟密度太高），等几秒自己就恢复。于是改成退避重试 + 人话提示。

七组，全程替身调用，不联网、不花钱：
  A 429 一次后成功   —— 自动重试 1 次即成功，说明行写明「重试 1 次后成功」
  B 连续 429         —— 共试 3 次（原 1 + 重试 2）后放弃，提示"过半分钟再问"
  C 非限流错误        —— 不重试（400 超上下文那种，原样重发结果一样，白等）
  D 本机端点 429     —— 不重试（本机一次生成 100s，重试只是再等一遍）
  E 云端正常         —— 只调一次，说明行不许出现"重试"
  F 被新点击作废      —— RunCancelled 必须原样抛出，不许变成"调用失败"
  G 等待期间被作废    —— 睡醒后要先确认这一轮还算数，立即中止

用法:
    .venv\\Scripts\\python.exe tests\\verify_qa_fallback.py
"""
from __future__ import annotations

import sys
import time as _real_time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))      # 复用 verify_qa 里的假 audit（tests 不是包）

from paper2lab import llm, qa                               # noqa: E402
from verify_qa import _fake_audit                           # noqa: E402

CHECKS: list = []


def check(ok, label: str, detail: str = "") -> None:
    assert isinstance(ok, bool), f"断言值必须是 bool, 拿到 {type(ok).__name__}: {label}"
    CHECKS.append((ok, label, detail))
    print(f"  {'✅' if ok else '❌'} {label}" + (f" — {detail}" if detail else ""))


class _RateLimit(Exception):
    """假装 openai 的 RateLimitError（只看类名与文本，跟真库无关）。"""
    pass


_RateLimit.__name__ = "RateLimitError"


class _BadReq(Exception):
    pass


_BadReq.__name__ = "BadRequestError"


class _FakeTime:
    """替身 time：只记 sleep 了多久，真不睡（否则一套测试白等 9 秒 × 7 组）。"""

    def __init__(self) -> None:
        self.slept: list = []

    def sleep(self, sec: float) -> None:
        self.slept.append(float(sec))


def _run(seq, *, base_url="https://api.deepseek.com/v1", model="deepseek-chat",
         key="sk-test"):
    """按 `seq` 依次回放：元素是异常就抛，是字符串就当作模型输出。"""
    calls: list = []
    it = iter(seq)

    def _fake_call(prompt, k, max_tokens=0, timeout=0, base_url="", model=""):
        calls.append((base_url, model))
        item = next(it, "默认回答：学习率是 1e-4。")
        if isinstance(item, BaseException):
            raise item
        return item

    qa._call_deepseek = _fake_call          # noqa: SLF001 — 测试替身
    body, note = qa.answer(_fake_audit(), "学习率到底是多少？", [],
                           api_key=key, base_url=base_url, model=model)
    return body, note, calls


def main() -> int:
    _real_call = qa._call_deepseek          # noqa: SLF001
    _real_time = qa.time                    # noqa: SLF001
    try:
        print("\nA 429 一次后成功 → 自动重试一次")
        ft = _FakeTime()
        qa.time = ft                        # noqa: SLF001
        body, note, calls = _run([_RateLimit("Error code: 429 - rate limited"),
                                  "学习率 1e-4（答出来的）"])
        check(len(calls) == 2, "第一次失败、第二次成功（共两次）", f"calls={len(calls)}")
        check(ft.slept == [3.0], "等待 3 秒再试（不是立刻硬顶）", f"slept={ft.slept}")
        check("自动重试 1 次后成功" in note, "说明行写明重试过", note[:60])
        check("答出来的" in body, "回答正文来自重试成功的那一次")
        check("本机" not in note and "intern-s1" not in note,
              "没有偷偷改用本机模型（用户明确要用云端）")

        print("\nB 连续 429 → 重试两次后如实报错")
        ft = _FakeTime()
        qa.time = ft                        # noqa: SLF001
        _rl = _RateLimit("Error code: 429 - rate limited")
        body, note, calls = _run([_rl, _RateLimit("429 again"), _RateLimit("429 third")])
        check(len(calls) == 3, "原 1 次 + 重试 2 次（共三次）", f"calls={len(calls)}")
        check(ft.slept == [3.0, 6.0], "退避 3s → 6s", f"slept={ft.slept}")
        check("已自动重试 2 次仍未通过" in note, "说明行告诉他试过几轮")
        check("半分钟" in note, "告诉他等多久再问")
        check("钱和 key 都没问题" in note, "别让他以为是充值问题（实测余额充足）")
        check("本机" not in note and "intern-s1" not in note, "仍然没有偷偷换本机")

        print("\nC 非限流错误 → 不重试")
        ft = _FakeTime()
        qa.time = ft                        # noqa: SLF001
        body, note, calls = _run([_BadReq("context size has been exceeded")])
        check(len(calls) == 1, "只调一次（重发也一样，白等）", f"calls={len(calls)}")
        check(ft.slept == [], "没有无谓的等待", f"slept={ft.slept}")
        check("BadRequestError" in note, "如实报出异常类名")

        print("\nD 本机端点 429 → 不重试")
        ft = _FakeTime()
        qa.time = ft                        # noqa: SLF001
        body, note, calls = _run([_RateLimit("429")],
                                 base_url="http://127.0.0.1:8080/v1",
                                 model="intern-s1-mini", key="local")
        check(len(calls) == 1, "本机只调一次（再试只是再等 100 秒）", f"calls={len(calls)}")
        check(ft.slept == [], "本机不睡等", f"slept={ft.slept}")

        print("\nE 云端正常 → 一次就好，别多事")
        ft = _FakeTime()
        qa.time = ft                        # noqa: SLF001
        body, note, calls = _run(["学习率 1e-4。"])
        check(len(calls) == 1, "只调一次", f"calls={len(calls)}")
        check("重试" not in note, "说明行没有多余的重试提示")
        check("模型作答" in note, "说明行保留「模型作答」这个判据（真机探针靠它认答复）")
        check("deepseek-chat" not in note and "intern-s1" not in note,
              "说明行里不出现模型名（2026-10-31 用户要求去掉）")

        print("\nF 被新点击作废 → RunCancelled 原样抛出")

        def _fake_cancel(prompt, k, max_tokens=0, timeout=0, base_url="", model=""):
            raise llm.RunCancelled()

        qa._call_deepseek = _fake_cancel    # noqa: SLF001
        _raised = False
        try:
            qa.answer(_fake_audit(), "学习率到底是多少？", [],
                      api_key="sk-test", base_url="https://api.deepseek.com/v1",
                      model="deepseek-chat")
        except llm.RunCancelled:
            _raised = True
        except Exception:                   # noqa: BLE001
            _raised = False
        check(_raised, "RunCancelled 穿透重试逻辑（不会被当成「调用失败」）")

        print("\nG 重试等待期间被作废 → 睡醒立刻中止")
        # 做法：第一次抛 429，睡等时把全局令牌往后推一格（模拟用户又点了一次发送），
        # 睡醒后 `check_run()` 应当发现本线程的令牌已过期并抛 RunCancelled。
        llm.begin_run()                     # 给本线程发一个令牌
        _before = llm._run_seq               # noqa: SLF001

        class _CancelDuringSleep(_FakeTime):
            def sleep(self, sec: float) -> None:
                super().sleep(sec)
                llm._run_seq = _before + 1    # noqa: SLF001 — 别人开新的一轮了

        ft = _CancelDuringSleep()
        qa.time = ft                        # noqa: SLF001
        _raised = False
        try:
            _run([_RateLimit("429")])
        except llm.RunCancelled:
            _raised = True
        except Exception:                   # noqa: BLE001
            _raised = False
        finally:
            qa.time = _real_time            # noqa: SLF001
            llm._RUN_LOCAL.token = None      # noqa: SLF001 — 别污染后面的检查
        check(len(ft.slept) == 1, "确实等过一次（说明进了重试分支）", f"slept={ft.slept}")
        check(_raised, "睡醒后发现自己已作废，立即中止（不继续跑旧问题）")
    finally:
        qa._call_deepseek = _real_call      # noqa: SLF001
        qa.time = _real_time                # noqa: SLF001

    ok = sum(1 for c in CHECKS if c[0])
    print(f"\n{'=' * 58}\n  问答限流重试体检：{ok}/{len(CHECKS)} 项通过\n{'=' * 58}")
    for good, label, detail in CHECKS:
        if not good:
            print(f"  ❌ {label}" + (f" — {detail}" if detail else ""))
    if ok != len(CHECKS):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
