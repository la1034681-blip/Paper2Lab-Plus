#!/usr/bin/env python3
"""MCP 客户端调用示例 —— 演示怎么把本技能当服务用。

两种连接方式:

    # A. stdio（本地，把服务端当子进程拉起）
    python mcp_client_example.py --stdio /path/to/paper2lab/serve_mcp.py \
        --pdf /path/to/paper.pdf --code /path/to/code.zip

    # B. Streamable HTTP（平台 / 远端）
    python mcp_client_example.py --http http://127.0.0.1:8765/mcp \
        --pdf /path/to/paper.pdf --code /path/to/code.zip

依赖: pip install "mcp>=1.0,<2"
（与平台侧示例保持同一版本 —— mcp 2.x 把 FastMCP 改名为 MCPServer 且改了 API。）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _call_stdio(server_py: str, tool: str, arguments: dict):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=sys.executable, args=[server_py])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"[stdio] 已连接: {info.serverInfo.name} {info.serverInfo.version}")
            tools = await session.list_tools()
            print(f"[stdio] 可用工具: {[t.name for t in tools.tools]}")
            return await session.call_tool(tool, arguments)


async def _call_http(url: str, tool: str, arguments: dict):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"[http] 已连接: {info.serverInfo.name} {info.serverInfo.version}")
            return await session.call_tool(tool, arguments)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mcp_client_example.py",
                                 description="MCP 客户端调用示例")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdio", metavar="SERVE_MCP_PY",
                     help="stdio 模式：serve_mcp.py 的路径")
    src.add_argument("--http", metavar="URL",
                     help="Streamable HTTP 模式：如 http://127.0.0.1:8765/mcp")
    ap.add_argument("--tool", default="check_parameters", help="要调用的工具名")
    ap.add_argument("--pdf", required=True, help="论文 PDF 路径（服务端可读）")
    ap.add_argument("--code", required=True, help="代码包路径（服务端可读）")
    args = ap.parse_args(argv)

    arguments = {"pdf_path": args.pdf, "code_path": args.code}

    try:
        if args.stdio:
            result = asyncio.run(_call_stdio(args.stdio, args.tool, arguments))
        else:
            result = asyncio.run(_call_http(args.http, args.tool, arguments))
    except ImportError:
        print("缺少依赖，请先执行: pip install \"mcp>=1.0,<2\"", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"调用失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"isError: {result.isError}")
    for item in result.content:
        text = getattr(item, "text", None)
        if text is None:
            continue
        try:  # 服务端返回的是 JSON 文本，格式化后更好读
            print(json.dumps(json.loads(text), ensure_ascii=False, indent=2))
        except Exception:
            print(text)
    return 0 if not result.isError else 1


if __name__ == "__main__":
    raise SystemExit(main())
