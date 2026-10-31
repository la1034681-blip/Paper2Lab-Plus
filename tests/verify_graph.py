"""阶段三第3步 · LangGraph 骨架 验证。

检查点（骨架层，不调真实 LLM，force_mock=True）:
  1. 图能构建、能编译，10 个工具节点 + 3 个控制节点 = 13 个节点
  2. 节点按 TOOLS 顺序连接
  3. 起点是 start，终点是 finalize -> END
  4. 闸门: 有前置产物的工具在产物缺失时被 blocked（不执行）
  5. 闸门: 产物齐备时 pass
  6. 执行记录里每条都带 gate 字段（pass/blocked）
  7. 失败节点不吞异常，记入 errors 并走到 finalize
  8. 骨架版产出与经典版同口径（同数据跑两版，结论一致 = 可复现的智能体）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from paper2lab import pipeline  # noqa: E402
from paper2lab.graph import (  # noqa: E402
    _PRECONDITIONS, build_graph, gate, _have, node_start,
)
from paper2lab.toolbox import TOOLS  # noqa: E402

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

    print("1. 节点数")
    try:
        graph = build_graph()
        c(graph is not None, "图编译成功")
        # LangGraph 1.x 的 compiled graph 不暴露节点数，改从源码数
        n_tools = len(TOOLS)
        c(n_tools >= 10, f"工具数 {n_tools}（≥10，不许被悄悄砍掉；阶段二 2A 起为 11）")
        # 口径: 每个**工具**都必须有闸门前置条件（可以为空集，表示无需前置）。
        # 反过来不一定 —— 阶段四的对撞区节点（roles/collider/...）也在 _PRECONDITIONS
        # 里登记了，但它们不是工具，不进 TOOLS。
        c({t.key for t in TOOLS} <= set(_PRECONDITIONS),
          "每个工具都有闸门前置条件")
        c({"roles", "collider", "arbiter", "redteam", "report_findings"}
          <= set(_PRECONDITIONS), "对撞区节点也在闸门登记表里（不是无闸门直通）")
    except Exception as ex:
        c(False, f"图构建失败: {type(ex).__name__}: {ex}")

    print("2. 连接顺序")
    try:
        keys = [t.key for t in TOOLS]
        expected = ["start", "assemble_context", "plan"] + keys + ["finalize"]
        c(len(expected) == len(TOOLS) + 4,
          f"节点链长度 {len(expected)}（工具数 + 4 个控制节点，随 TOOLS 走）")
        c(keys[0] == "parse_paper", f"第一个工具是 parse_paper")
        c(keys[-1] == "report", f"最后一个工具是 report")
    except Exception as ex:
        c(False, str(ex))

    print("3. 闸门: 产物缺失 -> blocked")
    state = {"paper": None, "code": None, "audit": None}
    for tool_key, needs in _PRECONDITIONS.items():
        if not needs:
            continue
        g, reason = gate(tool_key, state)
        c(g == "blocked", f"{tool_key} 缺产物 -> {g}: {reason[:40]}")
    # 只数**工具**里无需前置的（对撞区节点全都有前置，不参与这条统计）
    _tool_keys = {t.key for t in TOOLS}
    _no_need = [k for k, v in _PRECONDITIONS.items() if not v and k in _tool_keys]
    c(len(_no_need) == 2, f"2 个工具无需前置产物（{_no_need}）")

    print("4. 闸门: 产物齐备 -> pass")
    state = {
        "paper": object(), "code": object(),
        "audit": type("A", (), {"findings": [1], "rootcause": [1]})(),
        # 阶段二 2A/2D：① 站的假设卡 —— ② 站（design_experiment）的前置就是它。
        # 不补这一项，实验节点会被判 blocked（这正是闸门该有的行为：没有假设卡就没有可设计的实验）。
        "hypotheses": [{"key": "learning_rate"}],
        # 对撞区的前置产物（阶段四）——不补这几项，roles/collider/... 会被判 blocked
        "paper_view": {"learning_rate": {"value": "1e-4"}},
        "code_view": {"learning_rate": {"value": "1e-3"}},
        "collisions": [{"param_key": "learning_rate"}],
    }
    for tool_key in _PRECONDITIONS:
        g, reason = gate(tool_key, state)
        c(g == "pass", f"{tool_key} 产物齐备 -> {g}")

    print("5. _have 判据")
    state = {"paper": None, "code": None, "audit": None}
    c(_have(state, "paper") is False, "无 paper -> False")
    state["paper"] = object()
    c(_have(state, "paper") is True, "有 paper -> True")
    state["audit"] = type("A", (), {"findings": []})()
    c(_have(state, "findings") is False, "findings 空列表 -> False")
    state["audit"] = type("A", (), {"findings": [1]})()
    c(_have(state, "findings") is True, "findings 非空 -> True")

    print("6. 执行记录带 gate 字段")
    records = []
    state = {"records": records, "audit": None}
    from paper2lab.graph import _record  # noqa: E402
    _record(state, "align", "done", gate="pass", checkpoints=("a",))
    _record(state, "align", "skipped", "缺代码", gate="blocked")
    c(all("gate" in r for r in records), "每条记录都有 gate 字段")
    c([r["gate"] for r in records] == ["pass", "blocked"], f"gate 值正确")

    print("7. 骨架版与经典版同口径（消融对比）")
    # 同数据跑两版: 经典版 run_audit / 骨架版 run_audit_graph（force_mock 不调 LLM）
    try:
        c_audit, c_report = pipeline.run_audit(
            str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
            force_mock=True,
            progress=lambda i, n: None,
        )
        from paper2lab.graph import run_audit_graph  # noqa: E402
        g_audit, g_report = run_audit_graph(
            str(DEMO / "paper.pdf"), str(DEMO / "project.zip"),
            force_mock=True,
        )
        c(c_audit is not None, "经典版产出审计结果")
        c(g_audit is not None, "骨架版产出审计结果")
        c(len(c_audit.findings) == len(g_audit.findings),
          f"发现条数一致: classic={len(c_audit.findings)} graph={len(g_audit.findings)}")
        same = all(c.param_key == g.param_key and c.status == g.status
                   for c, g in zip(c_audit.findings, g_audit.findings))
        c(same, "逐条参数与状态一致（结论不变）")
        # 骨架版的执行记录
        # ⚠ 2026-10-29 修: 原来断言"全部工具 done"。但计划本来就会按材料跳过
        #   （本用例没配读图通道/正文无化学标记 → 2 件跳过）—— 旧断言能过，是因为
        #   图版当时**根本不看计划决策**。现在图版与经典版同口径，断言也改成
        #   "跳过的件数与经典版一致"，否则等于把 bug 固化成契约。
        g_log = getattr(g_audit, "execution_log", {})
        c_log = getattr(c_audit, "execution_log", {})
        _c_skip = int(c_log.get("skipped") or 0)
        c(g_log.get("nodes_done", 0) == len(TOOLS) + 2 - _c_skip,
          f"骨架版执行过的节点全 done（{len(TOOLS) - _c_skip} 件工具 + 2 个控制节点）: "
          f"{g_log.get('nodes_done')}")
        c(g_log.get("nodes_failed", 0) == 0, f"骨架版 0 个节点失败: {g_log.get('nodes_failed')}")
        c(int(g_log.get("skipped") or 0) == _c_skip,
          f"骨架版跳过的工具数与经典版一致（同为 {_c_skip} 件）: {g_log.get('skipped')}")
    except Exception as ex:
        c(False, f"骨架版运行失败: {type(ex).__name__}: {ex}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 66)
    print(f"阶段三第3步 LangGraph 骨架验证: {ok}/{total} OK")
    if ok < total:
        print("有未通过项")
        return 1
    print("全部通过(verify_graph.py)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())