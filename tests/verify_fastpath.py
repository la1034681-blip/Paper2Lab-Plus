"""阶段三第1步 · 快路径专项测试: 同会话内第二次跑, 跳过解析。

快档不依赖 fitz, 纯函数; 用项目 .venv 跑则需 fitz(本脚本不用 fitz)。
"""
from __future__ import annotations

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from paper2lab import housekeeping as hk, session  # noqa: E402

lines: list[str] = []
fails = []


def check(ok: bool, msg: str) -> None:
    ok = bool(ok)
    line = ("  OK   " if ok else "  FAIL ") + msg
    lines.append(line)
    print(line)
    if not ok:
        fails.append(msg)


def bar() -> None:
    lines.append("=" * 66)
    print("=" * 66)


def new_store() -> session.SessionStore:
    """换一个干净会话(避免跨节状态污染)。"""
    sid = session.new_id()
    session.set_current_id(sid)
    s = session.get()
    s.init()
    return s


bar()
lines.append("阶段三第1步 · 快路径专项测试")
print("阶段三第1步 · 快路径专项测试")
bar()

# --------------------------------------------------------------- 0. 环境
print()
lines.append("")
lines.append("0. 环境")
print("0. 环境")
check(session.get() is None, "未绑定时 get() 返回 None(不自动建库)")
print(f"   python: {sys.executable}")
lines.append(f"   python: {sys.executable}")

# --------------------------------------------------------------- 1. 快照存取
print()
lines.append("")
lines.append("1. 快照存取")
print("1. 快照存取")
s = new_store()
fp = session.fingerprint(b"fp-paper", b"fp-code")
paper = type("PaperInfo", (), {"title": "Test Paper", "params": {"lr": "1e-4"}})()
code = type("CodeInfo", (), {"params": {"lr": "0.001"}, "python_files": 3})()

s.store_snapshot(fp, paper, code)
snap = s.load_snapshot(fp)
check(snap is not None, "快照可读出")
check(snap[0].title == "Test Paper", f"paper 正确: {snap[0].title}")
check(snap[1].python_files == 3, f"code 正确: {snap[1].python_files}")
check(s.load_snapshot("no-such-fp") is None, "未存过的指纹返回 None")

# --------------------------------------------------------------- 2. T2 快照随 status 返回
print()
lines.append("")
lines.append("2. T2: 快照随 lookup 返回")
print("2. T2: 快照随 lookup 返回")

# 一个全新的指纹: 未索引 -> needs_index=True
fp_new = session.fingerprint(b"fresh-paper", b"fresh-code")
r = hk.session_lookup(b"fresh-paper", b"fresh-code")
check(r is not None, "lookup 返回非 None")
check(r["fingerprint"] == fp_new, f"指纹正确: {r['fingerprint']}")
check(r["needs_index"] is True, "新指纹 → needs_index=True")
check("snapshot" not in r, "新指纹无快照")
check(r["status"] is None, "新指纹 status=None")

# 同一个新指纹: 先存快照 -> 再查 -> 应返回快照
s.store_snapshot(fp_new, paper, code)
r2 = hk.session_lookup(b"fresh-paper", b"fresh-code")
check(r2["needs_index"] is False, "已索引 → needs_index=False")
snap2 = r2.get("snapshot")
check(snap2 is not None, "已缓存 → 返回 snapshot")
check(snap2[0].title == "Test Paper", "快照 paper 正确")
check(snap2[1].python_files == 3, "快照 code 正确")

# --------------------------------------------------------------- 3. T1 清快照
print()
lines.append("")
lines.append("3. T1 清库时快照一起清")
print("3. T1 清库时快照一起清")
check(s.load_snapshot(fp) is not None, "清库前快照在")
s.init()
check(s.load_snapshot(fp) is None, "init() 后快照被清")
check(s.load_snapshot(fp_new) is None, "同会话别的指纹快照也被清")

# --------------------------------------------------------------- 4. 多指纹隔离
print()
lines.append("")
lines.append("4. 多指纹隔离")
print("4. 多指纹隔离")
s2 = new_store()
fp_a = session.fingerprint(b"paper-A", b"code-A")
fp_b = session.fingerprint(b"paper-B", b"code-B")
b_paper = type("P", (), {"title": "B"})()
b_code = type("C", (), {"python_files": 7})()
s2.store_snapshot(fp_a, paper, code)
s2.store_snapshot(fp_b, b_paper, b_code)
check(s2.load_snapshot(fp_a)[0].title == "Test Paper", "fp-A 快照独立")
check(s2.load_snapshot(fp_b)[0].title == "B", "fp-B 快照独立")
check(s2.load_snapshot(fp_a)[1].python_files == 3, "fp-A 的 code 未被 fp-B 覆盖")

# --------------------------------------------------------------- 5. 多会话隔离
print()
lines.append("")
lines.append("5. 多会话隔离")
print("5. 多会话隔离")
s3 = new_store()
s3.store_snapshot(fp_a, paper, code)
s4 = new_store()          # 新会话, 不同的 sid
check(s4.sid != s3.sid, f"不同会话 id: {s3.sid} vs {s4.sid}")
check(s4.load_snapshot(fp_a) is None, "A 会话的快照在 B 会话不可见")

# --------------------------------------------------------------- 6. 无会话时安全
print()
lines.append("")
lines.append("6. 无会话时不报错")
print("6. 无会话时不报错")
session.set_current_id("")
check(hk.session_lookup(b"x", b"y") is None, "无会话返回 None")
try:
    session.get().store_snapshot(fp_a, paper, code)
    check(False, "无会话时 get() 返回 None(调用方应先判 None)")
except AttributeError:
    check(True, "无会话时 get() 返回 None, 不偷偷建库")

# --------------------------------------------------------------- 7. runner 接入点
print()
lines.append("")
lines.append("7. runner.py 把 snapshot 赋给 job")
print("7. runner.py 把 snapshot 赋给 job")
from paper2lab.toolbox import Job  # noqa: E402
job = Job(fingerprint=fp_a, needs_index=False, snapshot=(paper, code))
job.paper, job.code = job.snapshot
job.snapshot_reused = True
check(job.paper.title == "Test Paper", "job.paper 复用成功")
check(job.code.python_files == 3, "job.code 复用成功")
check(job.snapshot_reused is True, "snapshot_reused 标记正确")

# 未给 snapshot 时不崩
job2 = Job(fingerprint=fp_a, needs_index=True)
check(job2.snapshot is None, "无快照时 snapshot=None")
check(job2.needs_index is True, "needs_index=True 时不复用")

bar()
if fails:
    lines.append(f"未通过 {len(fails)} 项:")
    print(f"未通过 {len(fails)} 项:")
    for f in fails:
        lines.append("  -" + f)
        print("  -", f)
else:
    lines.append(f"全部通过({os.path.basename(__file__)})")
    print(f"全部通过({os.path.basename(__file__)})")
bar()

out = HERE / "cache" / "fastpath_probe.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")