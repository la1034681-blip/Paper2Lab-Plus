"""输入路径的前置校验 —— 填错就当场报错，不许降级成「没有」。

为什么必须单独一层（2026-10-30）:
    下层其实有抛错（`paper_parser.parse_paper` / `code_analyzer.analyze_code`
    都对「路径不存在」抛 FileNotFoundError），但执行器 `runner.run_job` 的
    `except Exception` 会把它记成「某一步失败」然后**接着跑完整套**，
    最后照样产出结论。用户看到的是「论文有、代码没有」——
    他会当成事实，而那只是路径没指到东西。

口径（三条，别再合）:
    1. 路径为空          -> 合法，语义是「这次没有这份材料」，不报错。
    2. 路径非空但不存在  -> 用户填错了，**必须报错**，且要在跑之前报。
    3. 路径存在但类型不对（要文件给了目录）-> 同样是填错。
"""

from __future__ import annotations

from pathlib import Path


class InputPathError(FileNotFoundError):
    """用户填的输入路径无效。

    继承 FileNotFoundError 而不是自定义基类：调用方（app / MCP / CLI）
    按「文件找不到」处理即可，不必额外认一个新类型；命令行里也会得到
    与内置异常一致的退出行为。
    """


def check_paper(pdf_path: str) -> None:
    """论文路径：必须非空、存在、是文件。"""
    raw = (pdf_path or "").strip()
    if not raw:
        raise InputPathError("没有给论文文件：论文是审计的必需输入，不能为空")
    p = Path(raw)
    if not p.exists():
        raise InputPathError(f"论文文件不存在: {raw}")
    if p.is_dir():
        raise InputPathError(f"论文这一栏要的是 PDF 文件，给的是目录: {raw}")


def check_code(code_path: str) -> None:
    """代码包路径：可以为空（= 没有代码）；非空则必须存在。"""
    raw = (code_path or "").strip()
    if not raw:
        return                      # 空 = 这次没有代码包，合法
    p = Path(raw)
    if not p.exists():
        raise InputPathError(
            f"代码包路径不存在: {raw}　"
            "（若这次本来就没有代码，请把这一栏留空，而不是填一个路径）")


def check_inputs(pdf_path: str, code_path: str = "") -> None:
    """一次校验两个输入。任一不合规立刻抛 InputPathError。"""
    check_paper(pdf_path)
    check_code(code_path)
