"""Paper2Lab MCP / SCP 服务入口（阶段六 · 协议化①）。

用法:
    python serve_mcp.py                  # stdio，本地客户端直连（默认）
    python serve_mcp.py --http           # streamable-http，供平台 / 远端接入
    python serve_mcp.py --http --port 9000

客户端配置示例（stdio，本地）:
    {
      "mcpServers": {
        "paper2lab": {
          "command": "C:/Users/lijiahui/Desktop/兴智杯/.venv/Scripts/python.exe",
          "args": ["C:/Users/lijiahui/Desktop/兴智杯/serve_mcp.py"]
        }
      }
    }

远端接入（端砚 SCP 形态 = Streamable HTTP + 鉴权 Key）:
    URL:  http(s)://<主机>:<端口>/mcp
    注意: 本入口**不自带鉴权**，只监听 127.0.0.1。要对外发布，鉴权与
    出网策略由部署层负责（本轮按用户口径「不发布」，因此只做到本地可调通）。

依赖: pip install "mcp>=1.0,<2"   （见 requirements-mcp.txt）
      主程序 requirements.txt 里**故意不含 mcp** —— 网页端不需要它，
      少一个依赖就少一份构建时间与供应链面。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 根目录下的脚本: sys.path[0] 是脚本所在目录，显式补一次更稳（放进来看不出差别，
# 但将来若有人把本文件挪进子目录，这一行就是"能不能 import paper2lab"的分水岭）。
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper2lab.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
