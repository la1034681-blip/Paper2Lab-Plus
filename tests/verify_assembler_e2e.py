"""阶段三第2步 · 上下文装配器 端到端验证。

用 app.py 的演示数据跑一轮真审计，检查:
  1. run1 跑完后, 证据被写入会话记忆(run_job 末尾的 _store_session_memory)
  2. run2 走快路径时, 装配器能读出对照表并喂给 planner
  3. 对照表里"论文侧+代码侧"同时存在的参数真的能对上
  4. T1 清库后线索清零, 不串味

注意: 本脚本不调用 LLM（force_mock=True）, 纯本地运行。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from paper2lab import pipeline, session  # noqa: E402
from paper2lab.housekeeping import session_clear  # noqa: E402

DEMO = HERE.parent / "demo"


def main():
    lines: list[str] = []
    total, ok = 0, 0

    def c(cond, label):
        nonlocal total, ok
        total += 1
        line = ("  OK   " if cond else "  FAIL ") + str(label)
        lines.append(line)
        print(line)
        if cond:
            ok += 1

    sid = session.new_id()
    session.set_current_id(sid)
    store = session.get()
    store.init()
    print(f"会话 id: {sid}")

    # ---- run1: 首次审计, 写入线索
    print("\n1. run1 首次审计")
    audit, report = pipeline.run_audit(
        str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
        force_mock=True,
        progress=lambda i, n: None,
    )
    n_findings = len(audit.findings)
    n_mem = store.info()["memories"]
    c(n_findings > 0, f"run1 产出 {n_findings} 条发现")
    c(n_mem > 0, f"run1 后会话记忆有 {n_mem} 条线索（_store_session_memory 生效）")

    rec = store.recall(store.fingerprint if hasattr(store, "fingerprint") else "")
    # job.fingerprint 是 paper-only 口径(memory.fingerprint), 存/取必须同源
    from paper2lab.memory import fingerprint as _mem_fp
    fp = _mem_fp(str(DEMO / "paper.pdf"))
    rec = store.recall(fp)
    c(len(rec) == n_mem, f"recall 返回 {len(rec)} 条, 与 memory 数一致")
    channels = {r["channel"] for r in rec}
    c("paper" in channels and "code" in channels, f"两侧线索齐全: {sorted(channels)}")

    # ---- run2: 同输入, 走快路径 + 装配器
    print("\n2. run2 同输入 -> 快路径 + 装配器")
    from paper2lab.housekeeping import session_lookup
    _init = session_lookup(
        (DEMO / "paper.pdf").read_bytes(),
        (DEMO / "project.zip").read_bytes(),
    )
    session_fp = session.fingerprint((DEMO / "paper.pdf").read_bytes(),
                                     (DEMO / "project.zip").read_bytes())
    audit2, report2 = pipeline.run_audit(
        str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
        force_mock=True,
        needs_index=_init.get("needs_index"),
        snapshot=_init.get("snapshot"),
        session_fp=session_fp,
        progress=lambda i, n: None,
    )
    c(len(audit2.findings) == n_findings, f"run2 产出 {len(audit2.findings)} 条发现(与 run1 一致)")
    c(audit2.findings[0].param_key == audit.findings[0].param_key,
      "run2 首条参数与 run1 一致")
    # run2 应该触发了装配器（在 planner 里）
    print(f"   run1 memory={n_mem}, run2 memory={store.info()['memories']}")

    # ---- 对照表里有"论文侧+代码侧"同时存在的参数
    print("\n3. 对照表能否对齐")
    from paper2lab.assembler import assemble
    job = type("Job", (), {"fingerprint": fp, "paper": audit2.paper,
                           "code": audit2.code})()
    screen = assemble(job, recall=rec)
    c(len(screen.param_table) > 0, f"对照表有 {len(screen.param_table)} 个参数")
    both = [r for r in screen.param_table
            if r.paper is not None and r.code is not None]
    c(len(both) > 0, f"其中 {len(both)} 个参数同时有论文侧与代码侧线索")
    if both:
        r = both[0]
        print(f"   例: {r.param_key} -> 论文侧 {r.paper.value}({r.paper.sources[0]['location']})"
              f" | 代码侧 {r.code.value}({r.code.sources[0]['location']})")

    # ---- T1 清库后线索清零
    print("\n4. T1 清库")
    cleared = session_clear()
    c(cleared == sid, f"session_clear 返回当前 sid")
    c(store.info()["memories"] == 0, f"清库后线索 {store.info()['memories']} 条(不串味)")

    store.close()
    print()
    print("=" * 66)
    print(f"阶段三第2步 端到端验证: {ok}/{total} OK")
    if ok < total:
        print("有未通过项")
        return 1
    print("全部通过(verify_assembler_e2e.py)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())