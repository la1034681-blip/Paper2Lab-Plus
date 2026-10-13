"""阶段三第2步 · 上下文装配器 验证。

检查点:
  1. 无历史线索时返回空屏，history 说明"首次审计"
  2. 回忆结果按 channel 分两侧
  3. 同一参数在论文侧与代码侧对齐
  4. 陌生 channel 被丢弃（不污染）
  5. 每侧线索有上限，超预算给出裁剪说明
  6. 原文片段被截断，不占半屏
  7. 拼装出的 prompt 包含系统提示 + 任务卡 + 对照表
  8. 与真实 session 串起来（写入 -> recall -> assemble）
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from paper2lab import assembler  # noqa: E402
from paper2lab.session import SessionStore  # noqa: E402


def main() -> int:
    lines: list[str] = []
    total, ok = 0, 0

    def c(cond: bool, label: str) -> None:
        nonlocal total, ok
        total += 1
        line = ("  OK   " if cond else "  FAIL ") + str(label)
        lines.append(line)
        print(line)
        if cond:
            ok += 1

    print("1. 无历史线索")
    job = type("Job", (), {"fingerprint": "", "paper": None, "code": None})
    screen = assembler.assemble(job, recall=[])
    c(screen.param_table == [], "无线索 -> 对照表为空")
    c("首次审计" in screen.history, f"history 提示首次审计: {screen.history}")
    c(screen.system != "", "系统提示不为空")
    c(screen.overview == "", "无参数时 overview 为空")

    print("2. 按 channel 分两侧")
    recall = [
        {"param": "learning_rate", "value": "1e-4", "channel": "paper",
         "source": "Page 6 · Training Setup",
         "doc": "The learning rate is set to 1e-4"},
        {"param": "learning_rate", "value": "0.001", "channel": "code",
         "source": "train.py:42", "doc": "learning_rate = 0.001"},
    ]
    screen = assembler.assemble(job, recall=recall)
    p = screen.param_table[0] if screen.param_table else None
    c(len(screen.param_table) == 1, f"对齐到 1 个参数: {len(screen.param_table)}")
    c(p is not None and p.param_key == "learning_rate", f"参数名正确")
    c(p.paper is not None, "论文侧有线索")
    c(p.code is not None, "代码侧有线索")
    c(p.paper.value == "1e-4", f"论文值正确: {p.paper.value}")
    c(p.code.value == "0.001", f"代码值正确: {p.code.value}")
    c(p.paper.sources[0]["location"] == "Page 6 · Training Setup", "出处正确")
    c(p.paper.confidence == "已确认", f"置信度: {p.paper.confidence}")

    print("3. 陌生 channel 被丢弃")
    bad = [
        {"param": "x", "value": "1", "channel": "agent",
         "source": "s", "doc": "d"},
    ]
    screen = assembler.assemble(job, recall=bad)
    c(screen.param_table == [], "陌生 channel 不进入对照表")

    print("4. 每侧线索有上限")
    many = []
    for i in range(20):
        many.append({"param": f"p{i}", "value": f"v{i}", "channel": "paper",
                     "source": f"s{i}", "doc": f"doc{i}"})
    screen_budget = assembler.assemble(job, recall=many)
    c(len(screen_budget.param_table) <= assembler._MAX_HINTS_PER_SIDE,
      f"截断到 {len(screen_budget.param_table)} 条（上限 {assembler._MAX_HINTS_PER_SIDE}）")
    c(screen_budget.budget_note != "", "预算说明已给出")

    print("5. 原文片段截断")
    long_doc = {"param": "x", "value": "1", "channel": "paper", "source": "s",
                "doc": "a " * 200}
    screen_clip = assembler.assemble(job, recall=[long_doc])
    side = screen_clip.param_table[0].paper if screen_clip.param_table else None
    c(side is not None and len(side.sources[0]["doc"]) <= assembler._MAX_DOC_LEN,
      f"原文被截断（上限 {assembler._MAX_DOC_LEN}）")
    c(side is not None and side.sources[0]["doc"].endswith("…"), "以省略号结尾")

    print("6. 拼装出的 prompt")
    prompt = screen_budget.to_prompt()
    c("你是一名论文↔代码一致性审计员" in prompt, "系统提示在 prompt 里")
    c("参数 p0" in prompt, "对照表在 prompt 里")
    c("线索预算" in prompt, "预算说明在 prompt 里")

    print("7. 与真实 session 串起来")
    from paper2lab import session as _sess
    db = Path(tempfile.mktemp(suffix=".db"))
    store = SessionStore(db, ttl=3600)
    _sess.set_current_id(store.sid)
    store.init()
    fp = "test-fp"
    store.store_memory(fp, "learning_rate", "1e-4", "paper",
                       "Page 6", "The learning rate is 1e-4")
    store.store_memory(fp, "learning_rate", "0.001", "code",
                       "train.py:42", "learning_rate = 0.001")
    rec = store.recall(fp)
    c(len(rec) == 2, f"回忆出 {len(rec)} 条，带 channel")
    c({r["channel"] for r in rec} == {"paper", "code"},
      f"两侧齐全: {sorted(r['channel'] for r in rec)}")
    screen2 = assembler.assemble(
        type("Job", (), {"fingerprint": fp, "paper": None, "code": None})(),
        recall=rec)
    row = screen2.param_table[0] if screen2.param_table else None
    c(row is not None and len(screen2.param_table) == 1,
      f"串起来后对齐到 1 个参数")
    c(row is not None and row.paper.value == "1e-4", "论文值=1e-4")
    c(row is not None and row.code.value == "0.001", "代码值=0.001")
    store.close()
    db.unlink()

    print()
    print("=" * 66)
    print(f"阶段三第2步 装配器验证: {ok}/{total} OK")
    if fails := total - ok:
        print(f"有 {fails} 项未通过")
        return 1
    print("全部通过(verify_assembler.py)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())