"""阶段三第1步 · 端到端快路径验证(模拟 app.py 的完整代码路径)。

用项目 .venv 跑, 需 fitz。模拟 app.py 的 T1/T2 + run_audit:
  run1  -> needs_index=True, 解析并存快照
  run2  -> needs_index=False, 取快照跳过解析
  run3  -> 改一个字节 -> needs_index=True, 重新解析

输出: cache/fastpath_e2e.txt
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from paper2lab import housekeeping as hk, session  # noqa: E402
from paper2lab.envfile import load_env  # noqa: E402
from paper2lab.pipeline import run_audit  # noqa: E402

load_env()
DEMO = HERE / "demo"
PDF = str(DEMO / "paper.pdf")
ZIP = str(DEMO / "project.zip")
RES = json.loads((DEMO / "results.json").read_text(encoding="utf-8"))

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


bar()
lines.append("阶段三第1步 · 端到端快路径验证")
print("阶段三第1步 · 端到端快路径验证")
bar()

print()
lines.append("0. 环境")
print("0. 环境")
sid = session.new_id()
session.set_current_id(sid)
session.get()
check(session.current_id() == sid, f"会话 id: {sid}")
lines.append(f"会话 id: {sid}")
lines.append(f"python: {sys.executable}")
print(f"python: {sys.executable}")

pdf_bytes = pathlib.Path(PDF).read_bytes()
code_bytes = pathlib.Path(ZIP).read_bytes()
session_fp = session.fingerprint(pdf_bytes, code_bytes)
lines.append(f"session_fp(论文+代码): {session_fp}")
print(f"session_fp(论文+代码): {session_fp}")

# --------------------------------------------------------------- run1
print()
lines.append("")
lines.append("1. run1: 首次 -> 需要解析 + 存快照")
print("1. run1: 首次 -> 需要解析 + 存快照")
_init = hk.session_lookup(pdf_bytes, code_bytes)
check(_init["needs_index"] is True, "run1 前 needs_index=True")
check("snapshot" not in _init, "run1 前无快照")
t0 = time.time()
a1, _ = run_audit(PDF, ZIP, RES, session_fp=session_fp)
t1 = time.time()
lines.append(f"run1 耗时: {t1-t0:.2f}s")
print(f"run1 耗时: {t1-t0:.2f}s")
check(len(a1.findings) == 13, f"run1 产出 13 条发现: {len(a1.findings)}")
lines.append(f"run1 findings: {len(a1.findings)}")

# --------------------------------------------------------------- run2
print()
lines.append("")
lines.append("2. run2: 同输入 -> 命中快照, 跳过解析")
print("2. run2: 同输入 -> 命中快照, 跳过解析")
_init2 = hk.session_lookup(pdf_bytes, code_bytes)
check(_init2["needs_index"] is False, "run2 前 needs_index=False")
check(_init2.get("snapshot") is not None, "run2 前有快照")
t2 = time.time()
a2, _ = run_audit(PDF, ZIP, RES, session_fp=session_fp,
                  needs_index=_init2["needs_index"], snapshot=_init2.get("snapshot"))
t3 = time.time()
lines.append(f"run2 耗时: {t3-t2:.2f}s")
print(f"run2 耗时: {t3-t2:.2f}s")
check(len(a2.findings) == 13, f"run2 产出 13 条发现: {len(a2.findings)}")
el2 = a2.execution_log
parse2 = next(s for s in el2["steps"] if s["key"] == "parse_paper")
code2 = next(s for s in el2["steps"] if s["key"] == "analyze_code")
check(parse2["seconds"] < 0.1, f"run2 论文解析快跳({parse2['seconds']}s)")
check(code2["seconds"] < 0.1, f"run2 代码扫描快跳({code2['seconds']}s)")
check([(f.param_key, f.status.value) for f in a1.findings]
      == [(f.param_key, f.status.value) for f in a2.findings],
      "run1/run2 发现完全一致(结论不变)")
lines.append(f"run1 findings == run2 findings: True")

# --------------------------------------------------------------- run3
print()
lines.append("")
lines.append("3. run3: 改一个字节 -> 指纹变, 重新解析")
print("3. run3: 改一个字节 -> 指纹变, 重新解析")
_init3 = hk.session_lookup(pdf_bytes, code_bytes + b"x")
check(_init3["needs_index"] is True, "指纹变化后 needs_index=True")
t4 = time.time()
a3, _ = run_audit(PDF, ZIP, RES, session_fp=None)
t5 = time.time()
lines.append(f"run3 耗时: {t5-t4:.2f}s")
print(f"run3 耗时: {t5-t4:.2f}s")
check(len(a3.findings) == 13, f"run3 产出 13 条发现: {len(a3.findings)}")

# --------------------------------------------------------------- 4. 快照与库同步
print()
lines.append("")
lines.append("4. T1 清库时快照一起清")
print("4. T1 清库时快照一起清")
store = session.get()
check(store.load_snapshot(session_fp) is not None, "清库前快照在")
store.init()
check(store.load_snapshot(session_fp) is None, "init() 后快照被清")

bar()
if fails:
    lines.append(f"未通过 {len(fails)} 项:")
    print(f"未通过 {len(fails)} 项:")
    for f in fails:
        lines.append("  -" + f)
        print("  -", f)
else:
    lines.append(f"全部通过({pathlib.Path(__file__).name})")
    print(f"全部通过({pathlib.Path(__file__).name})")
bar()

out = HERE / "cache" / "fastpath_e2e.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")