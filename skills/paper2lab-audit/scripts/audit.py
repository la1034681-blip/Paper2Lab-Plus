#!/usr/bin/env python3
"""命令行跑一次「论文 x 代码」复现一致性审计，结果以 JSON 输出。

用法:
    python audit.py <论文.pdf> <代码包.zip|目录> [选项]

选项:
    --mode classic|multagent   审计链路（默认 classic，最快）
    --loop                     追加 ② 算 / ③ 核验 / ④ 方案与迭代回边（四站贯通）
    --llm                      允许调用大模型（默认关闭，成本为零）
    --report                   在结果里带上完整 Markdown 报告（默认只给预览）
    --pretty                   缩进输出，便于人读

设计要点（三条，与主程序同一条纪律）:
  1. **不重写业务逻辑** —— 直接调用 `paper2lab.mcp_server.tool_audit_paper`，
     与网页端、MCP 端共用同一份实现，结论天然一致；
  2. **只读输入，不写任何文件**；
  3. 找不到 `paper2lab` 包时给明确的补救提示，而不是抛一个裸 ImportError
     （技能包可能被单独下载到别处，这是常态而非异常）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def find_project_root() -> "Path | None":
    """定位 paper2lab 包所在的项目根目录。

    顺序: 环境变量 PAPER2LAB_ROOT -> 从本脚本所在目录逐级向上找。
    """
    env = (os.environ.get("PAPER2LAB_ROOT") or "").strip()
    if env:
        cand = Path(env).expanduser()
        if (cand / "paper2lab" / "__init__.py").is_file():
            return cand

    cur = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "paper2lab" / "__init__.py").is_file():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="audit.py",
        description="论文 x 代码 复现一致性审计（只读输入，不写文件）")
    ap.add_argument("pdf", help="论文 PDF 路径")
    ap.add_argument("code", help="代码包路径（.zip 或已解压目录）")
    ap.add_argument("--mode", default="classic", choices=("classic", "multagent"),
                    help="经典链（默认）或多智能体对撞链")
    ap.add_argument("--loop", action="store_true",
                    help="追加 ② 算 / ③ 核验 / ④ 方案与迭代回边（四站贯通）")
    ap.add_argument("--llm", action="store_true",
                    help="允许调用大模型（默认关闭，成本为零）")
    ap.add_argument("--report", action="store_true",
                    help="结果里带上完整 Markdown 报告")
    ap.add_argument("--pretty", action="store_true", help="缩进输出")
    args = ap.parse_args(argv)

    root = find_project_root()
    if root is None:
        print(
            "找不到 paper2lab 包。\n"
            "  技能包可能被下载到了别的目录 —— 请把环境变量指向项目根目录:\n"
            "    Linux/macOS:  export PAPER2LAB_ROOT=/path/to/paper2lab\n"
            "    Windows:      set PAPER2LAB_ROOT=D:\\path\\to\\paper2lab\n"
            "  项目根目录的特征是它下面有 paper2lab/__init__.py。",
            file=sys.stderr)
        return 2

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from paper2lab.mcp_server import stdout_as_stderr, tool_audit_paper
    except Exception as exc:
        print(f"导入 paper2lab 失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    # 把第三方库的提示挡在 stderr —— PyMuPDF 在 import 时和第一次解析页面时
    # 会各打一行提示到 stdout，否则本脚本的 JSON 输出会被污染，
    # `audit.py ... | jq` 这类用法会直接失败。
    with stdout_as_stderr():
        result = tool_audit_paper(
            args.pdf, args.code,
            mode=args.mode,
            closes_loop=bool(args.loop),
            use_llm=bool(args.llm),
            include_report=bool(args.report),
        )

    print(json.dumps(result, ensure_ascii=False,
                     indent=2 if args.pretty else None))
    # 约定: 0 = 审计跑通（哪怕结论里有不一致）；1 = 未能完成审计
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
