"""阶段二 2E：多模态科学图表解析（子包）。

对外只暴露下面这几个名字 —— 调用方不该关心内部有几个文件。
拆成子包（而不是单文件）是为了让"多模态"这条线**有自己的边界**，
将来接谱图/分子式专用工具时加文件即可，不用动这个包的对外接口。
"""
from .figures import (CLUE_TYPES, FIGURE_KINDS, FIGURE_MAX, FIGURE_REGION_MAX,
                      FigureClue, engine_name, read_figures, summarize)

__all__ = ["CLUE_TYPES", "FIGURE_KINDS", "FIGURE_MAX", "FIGURE_REGION_MAX",
           "FigureClue", "engine_name", "read_figures", "summarize"]
