"""工具目录的**兼容壳**（2026-10-25 合并后只剩转发）。

用户口径：「把工具全部整合到一个代码文件，其余通过调用这个文件来实现 LLM 对工具的
调用以及程序化的确定」—— 登记部分已搬进 `toolbox.py`：
`TOOLS`（14 件审计工序）、`SCI_TOOLS`（4 件科学计算）、`CATALOG`（给模型看的同一张菜单）、
`LOOP_NODES`（闭环 4 个节点）。

本文件**不再定义任何东西**，只做转发，让老代码
`from .science_tools import CATALOG` 照样能跑。
新增工具请直接写在 `toolbox.py` —— 在这里另起一份必然漂成两个口径。
"""
from __future__ import annotations

from .toolbox import (CATALOG, SCIENCE_TOOLS, CatalogEntry, all_keys,
                      catalog_keys, menu, science_menu,
                      science_menu_for_model)

__all__ = ["CATALOG", "SCIENCE_TOOLS", "CatalogEntry", "all_keys",
           "catalog_keys", "menu", "science_menu", "science_menu_for_model"]
