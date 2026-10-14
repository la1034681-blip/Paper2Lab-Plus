# 接入方式

两种形态，按需选一种（也可以都接）：

| 形态 | 适合 | 本质 |
| --- | --- | --- |
| **技能包**（本目录） | 本地开发机 / 支持 Agent Skills 的客户端 | 一份可读的规程 + 几个脚本，按需加载 |
| **MCP 服务** | 平台、远端、任意 MCP 客户端 | 一个 JSON-RPC 服务，工具被你直接调用 |

## 一、本地直接跑（最省事）

```bash
# 环境自检
python scripts/check_env.py

# 最小核对
python scripts/audit.py 论文.pdf 代码.zip

# 完整闭环
python scripts/audit.py 论文.pdf 代码.zip --mode multagent --loop --pretty
```

脚本会自己向上寻找 `paper2lab` 包所在的项目根目录；
找不到时用环境变量指定：

```bash
export PAPER2LAB_ROOT=/path/to/paper2lab
```

## 二、接入 MCP（stdio，本地客户端）

先装可选依赖（只有 MCP 通道需要，网页端不需要）：

```bash
pip install "mcp>=1.0,<2"
```

客户端配置（以本地路径为例，按实际路径替换）：

```json
{
  "mcpServers": {
    "paper2lab": {
      "command": "<项目路径>/.venv/Scripts/python.exe",
      "args": ["<项目路径>/serve_mcp.py"]
    }
  }
}
```

> 版本注意：本项目钉 `mcp>=1.0,<2`。
> mcp 2.x 把 `FastMCP` 改名成了 `MCPServer`，与平台侧示例用到的 1.x API 不一致。

## 三、接入 MCP（Streamable HTTP，平台 / 远端）

```bash
python serve_mcp.py --http --host 127.0.0.1 --port 8765
```

端点：`http://127.0.0.1:8765/mcp`

调用示例（服务端不自带鉴权，请只在受信网络中使用）：

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def main():
    transport = streamablehttp_client(url="http://127.0.0.1:8765/mcp")
    read, write, _ = await transport.__aenter__()
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()

    result = await session.call_tool("check_parameters", arguments={
        "pdf_path": "/abs/path/paper.pdf",
        "code_path": "/abs/path/code.zip",
    })
    for item in result.content:
        if hasattr(item, "text"):
            print(item.text)


asyncio.run(main())
```

## 四、可用的工具

| 工具 | 作用 |
| --- | --- |
| `audit_paper` | 完整审计（可切 classic / multagent，可带闭环） |
| `check_parameters` | 最小竖切，只要参数一致性结论 |
| `compute_verify` | 「算」小规模复算与静态推演 |
| `citation_check` | 「搜」每条结论的引证核验 |
| `build_repro_plan` | 「做」七步复现方案 + 评级 |
| `iterate_fix` | 迭代回边（≤3 轮） |
| `list_pipeline_tools` | 列出内部流水线的 10 件工具 |

工具返回的路径参数都是**服务端所在机器**的路径 —— 远端接入时请传服务端能读到的路径。

## 五、成本

所有能力**规则优先**：默认不发起任何模型调用，成本严格为零。
需要模型参与时（角色规范化、AI 风险归因）需显式开启，并在调用后如实标注。
