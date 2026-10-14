#!/usr/bin/env python3
"""环境自检：跑审计之前先确认「缺什么」，避免拿着 ImportError 猜。

检查项:
  1. Python 版本（需要 3.10+）
  2. 能否找到并导入 paper2lab 包
  3. 核心模块是否齐全
  4. 可选能力是否可用（MCP 服务、多智能体链）
  5. 可选：用内置 demo 真跑一次（--demo）

用法:
    python check_env.py
    python check_env.py --demo
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import find_project_root  # noqa: E402  （同目录的小工具，避免重复实现）


OK = "✔"
BAD = "✘"
WARN = "!"


def _line(mark: str, text: str) -> None:
    print(f"  {mark} {text}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="check_env.py", description="环境自检")
    ap.add_argument("--demo", action="store_true", help="用内置 demo 真跑一次审计")
    args = ap.parse_args(argv)

    problems = 0

    print("=== 1. Python 版本 ===")
    v = sys.version_info
    if v >= (3, 10):
        _line(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        _line(BAD, f"Python {v.major}.{v.minor} —— 需要 3.10 及以上")
        problems += 1

    print("=== 2. 定位项目 ===")
    root = find_project_root()
    if root is None:
        _line(BAD, "找不到 paper2lab 包")
        _line(" ", "请设置 PAPER2LAB_ROOT 指向项目根目录后重试")
        problems += 1
        print(f"\n结论: 有 {problems} 项需要处理。")
        return 1
    _line(OK, f"项目根目录: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("=== 3. 核心模块 ===")
    mods = ("models", "toolbox", "pipeline", "aligner", "paper_parser",
            "code_analyzer", "selfcheck", "report")
    for name in mods:
        try:
            importlib.import_module(f"paper2lab.{name}")
            _line(OK, f"paper2lab.{name}")
        except Exception as exc:
            _line(BAD, f"paper2lab.{name} —— {type(exc).__name__}: {exc}")
            problems += 1

    print("=== 4. 阶段五能力（搜 / 算 / 做 / 迭代回边）===")
    for name in ("compute", "retrieval", "repro_plan", "loop"):
        try:
            importlib.import_module(f"paper2lab.{name}")
            _line(OK, f"paper2lab.{name}")
        except Exception as exc:
            _line(WARN, f"paper2lab.{name} 不可用 —— {type(exc).__name__}: {exc}")
    try:
        importlib.import_module("langgraph")
        _line(OK, "langgraph（多智能体链可用）")
    except Exception:
        _line(WARN, "langgraph 未安装 —— 只能用 classic 模式（不影响最小核对）")
    try:
        importlib.import_module("mcp")
        _line(OK, "mcp（MCP / SCP 通道可用）")
    except Exception:
        _line(WARN, "mcp 未安装 —— 命令行可用；要接 MCP 请 pip install \"mcp>=1.0,<2\"")

    if args.demo:
        print("=== 5. 用内置 demo 真跑一次 ===")
        demo = root / "demo"
        pdf, code = demo / "paper.pdf", demo / "project.zip"
        if not (pdf.is_file() and code.exists()):
            _line(WARN, f"内置 demo 不完整，跳过（{demo}）")
        else:
            try:
                from paper2lab.mcp_server import stdout_as_stderr, tool_audit_paper
                # 第三方库的提示统一压到 stderr，保证本脚本输出干净可读
                with stdout_as_stderr():
                    res = tool_audit_paper(str(pdf), str(code))
                if res.get("ok"):
                    _line(OK, f"审计跑通: {res.get('findings_total')} 条结论，"
                              f"耗时 {res.get('elapsed_s')}s")
                else:
                    _line(BAD, f"审计未通过: {res.get('error')}")
                    problems += 1
            except Exception as exc:
                _line(BAD, f"审计异常: {type(exc).__name__}: {exc}")
                problems += 1

    print(f"\n结论: {'全部就绪。' if problems == 0 else f'有 {problems} 项需要处理。'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
