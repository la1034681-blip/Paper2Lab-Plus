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

输入路径的两种口径（2026-10-30 起，别再混）：

| 你填的 | 行为 |
|---|---|
| 路径存在 | 正常审计 |
| 路径**不存在** | **当场报错**（`InputPathError`，继承 `FileNotFoundError`），不会跑出一份「代码里没有这些参数」的结论 |
| `code_path` **留空** | 合法，语义是「这次没有代码」，只做论文侧自查 |

第二条是踩出来的：以前路径填错会被当成「没提供代码包」，代码扫描整个跳过，
最后照样出一份两万字的报告 —— 用户会把「代码侧比对缺席」当成事实。

## 四、可用的工具（按四站排）

顺序就是**调用顺序**：① -> ② -> ③ -> ④。上一站的产出是下一站的入场券。

| 站 | 工具 | 作用 | 入 -> 出 |
| --- | --- | --- | --- |
| ① 假设 | `check_parameters` | 最小竖切，只要参数一致性结论 | 原始材料 -> 假设卡（可疑项） |
| ② 实验 | `compute_verify` | 「算」小规模复算与静态推演 | 可疑项 -> 实验卡（**只出数**） |
| ② 实验 | `science_calc` | 点名调一件具名科学工具（浮点精度 / 序列统计 / 直线拟合 / 峰位） | 你给的输入 -> 算出来的数（**不下判断**） |
| ③ 验证 | `citation_check` | 「搜」每条结论的引证核验 | 实际值 -> 判定卡（对 / 不对） |
| ③ 验证 | `literature_search` | 「搜」外部文献检索（6 个公开索引并发查，免费无 key） | 一个关键词 -> 可点开的候选文献（**不需论文/代码**） |
| ③ 验证 | `chem_check` | 「读」化学身份核验（分子式↔分子量、CAS 校验位、SMILES 语法） | 论文或一段原文 -> 逐条判定（**不猜结构、不命名**） |
| ④ 迭代 | `build_repro_plan` | 「做」七步复现方案 + 评级 | 不通过项 -> 方案与评分 |
| ④ 迭代 | `iterate_fix` | 迭代回边（≤3 轮） | 不通过项 -> 修正建议（草案） |
| 全流程 | `audit_paper` | 一次贯通四站（可切 classic / multagent，可带闭环） | 原始材料 -> 报告 |
| 清单 | `list_pipeline_tools` | 四站全景 ＋ 内部工序表（依赖 / 成本 / 归属站） | — -> 清单 |

两条容易用错的地方：

- **② 只交数，③ 才下判定。** 别把 `compute_verify` 返回的数字直接当结论讲；
- **③ 手上没有②的输入表。** 这是有意的设计 —— 它想编也编不出来，所以它的判定是可核验的。

工具返回的路径参数都是**服务端所在机器**的路径 —— 远端接入时请传服务端能读到的路径。

## 五、成本

所有能力**规则优先**：默认不发起任何模型调用，成本严格为零。
需要模型参与时（角色规范化、AI 风险归因）需显式开启，并在调用后如实标注。
