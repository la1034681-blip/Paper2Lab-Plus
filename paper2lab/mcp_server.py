"""MCP / SCP 封装（阶段六 · 协议化①）——把审计能力做成标准 MCP server。

为什么单独一个模块、不进 app.py:
    MCP 是「给别的智能体调用」的入口（端砚平台把这形态叫 SCP），
    与 Streamlit 界面是**同一套能力的两种形态**。因此本模块只做
    「校验输入 -> 调用 -> 给摘要」, **不复制任何业务逻辑** ——
    审计仍然只由 pipeline / graph 实现, 两条路的结论天然一致
    （这与「单路径与多智能体对撞共用同一份实现」是同一条工程纪律）。

    端砚（书生科学发现平台 Intern InkStone）两个广场的对应关系:
      · Skills 广场 —— 技能包（文件夹 + SKILL.md）, 平台只分发不执行  -> 见 skills/
      · SCP    广场 —— 标准 MCP 协议的独立工具服务, Endpoint + Key 调用 -> 本模块

两种 transport:
    stdio            本地客户端直连（默认，最省事）
    streamable-http  平台 / 远端接入（python serve_mcp.py --http）

安全边界（与主线同一条，不是这里新加的）:
    · 只读输入, **绝不写任何文件**（迭代回边的修正只在内存里反事实推演）;
    · 「算」不执行用户代码（ast 白名单）, 迭代回边也不执行;
    · 输入路径必须真实存在、扩展名受限, 拒绝不存在的路径与目录穿越;
    · 返回值一律走 _shrink 裁剪, **不回传 page_texts 这类大字段** ——
      调用方（另一个智能体）的上下文很贵, 灌爆它等于把产品做坏。

版本说明（写在这里省得下次重踩）:
    mcp 2.x 已把 `FastMCP` 改名为 `MCPServer` 并改了若干 API
    （https://py.sdk.modelcontextprotocol.io/v2/migration/）。
    端砚 SCP 的官方示例用的是 1.x 的 `streamablehttp_client`,
    所以本项目钉 `mcp>=1.0,<2`（见 requirements-mcp.txt）。
    将来升 2.x, 要改的只有本文件的 import 与 build_server()。
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional


# ------------------------------------------------------- stdout 通道保护（重要）
#
# ★ 为什么这层必须定义在**所有重依赖导入之前**（3B 踩坑后提前）:
#   preload() 只在 build_server() 里调用，但"模块一被 import"就可能已经带进重依赖 ——
#   本模块要从 `toolbox` 取四站名，而 toolbox 一导入就顺着 `paper_parser` 把
#   PyMuPDF 拉进来。实测现象: 只 `import paper2lab.mcp_server`（不进 main），
#   stdout 就先多出一行 fitz 弃用提示 —— 这比"调工具才断"更致命:
#   客户端的**第一次握手**（initialize）就被这行脏数据打断。

def _flush_c_stdio() -> None:
    """把 **C 层** stdout 的缓冲刷掉（Python 的 sys.stdout.flush() 刷不到它）。

    为什么必须有这一步:
        PyMuPDF 的提示是 C 层写 stdout 的，会先落进 CRT 的 FILE* 缓冲。
        只做 dup2 重定向、却不在恢复 fd 之前刷 C 缓冲的话，那行内容会**攒着**，
        等 dup2 把 fd 1 换回真正的 stdout 之后才被刷出来。
    平台差异: Windows 上 Python 3.x 链接的是 **UCRT**，而 `msvcrt.dll` 是另一个运行时 ——
        刷 msvcrt 的缓冲刷不到 UCRT 写的 stdout，所以这里两个都试一遍。
        刷不动也不抛（拿不到 C 运行时是很边缘的情况，不该因此让服务起不来）。
    """
    try:
        import ctypes
    except Exception:
        return
    for _name in (("ucrtbase", "msvcrt") if os.name == "nt" else (None,)):
        try:
            ctypes.CDLL(_name).fflush(None)
        except Exception:
            continue


@contextlib.contextmanager
def stdout_as_stderr():
    """在导入重依赖期间, 把进程 stdout 临时让给 stderr。

    为什么必须用 fd 级 dup2, 而不是 contextlib.redirect_stdout:
        PyMuPDF 的 fitz 兼容层在 **import 时** 会往 stdout 打两行弃用提示 ——
            warning: The `fitz` API is deprecated and will be removed in future.
            Consider using the pymupdf_layout package for a greatly improved ...
        那是 C 层直接写 fd 1, Python 层的 redirect_stdout 拦不住。
        在 stdio 通道里 stdout 是 JSON-RPC 专线, 多一行脏数据就会让
        严格客户端抛 "Failed to parse JSONRPC message from server" ——
        这个坑实测踩过一次（现象是：tools/list 正常, 一调工具就断）。
    网页端不受影响, 所以这条保护只在本模块做, 主流程一行不动。
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass
    _flush_c_stdio()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        # 关键顺序: 先刷 C 缓冲（此时 fd1 还指向 stderr），再恢复 fd1
        _flush_c_stdio()
        try:
            os.dup2(saved, 1)
        finally:
            os.close(saved)


# 四站口径的唯一源头。★ 这个导入**会连带把 PyMuPDF 拉进来**（toolbox -> paper_parser），
# 所以必须包在与 preload 同一层的 fd 保护里 —— 否则脏行会出现在第一次握手之前。
with stdout_as_stderr():
    from .toolbox import STATION_DESC, STATIONS

try:  # 没装 mcp 时, 本模块仍可被导入(纯函数部分可离线测), 只是 build_server 会报错
    from mcp.server.fastmcp import FastMCP
except ImportError as _exc:  # pragma: no cover - 取决于环境
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = str(_exc)
else:
    _MCP_IMPORT_ERROR = ""

SERVER_NAME = "paper2lab-audit"
#: 1.1.0（3B）: 7 个工具按四站挂靠、清单多给 `stations` 全景。
#: **工具名一个没改** —— 名字是对外契约，只重排顺序与说明，老客户端不用改一行代码。
#: 1.2.0（3D）: 新增 `science_calc`，把 3A 的 4 件具名科学工具**单独对外露出**
#:（此前只能经 `compute_verify` 间接触达）。**原有 7 个工具名与参数一字未动**。
#: 1.3.0（4A）: 新增 `literature_search`，把「搜」从"论文内部检索"扩到
#: **6 个公开学术索引的联邦检索**（免费、无 key）。原有 8 个工具名与参数一字未动。
#: 1.4.0（2E-化学）: 新增 `chem_check`，把「读」里的化学身份核验单独露出
#:（分子式↔分子量复算 / CAS 校验位 / SMILES 语法）。**原有 9 个工具名与参数一字未动**。
SERVER_VERSION = "1.4.0"

#: 对外暴露的工具名（测试与文档都从这里取，别在别处硬编码字符串）
TOOL_NAMES: tuple = (
    "audit_paper",
    "check_parameters",
    "compute_verify",
    # 3D: 把 3A 的具名科学计算工具**单独露出来**，外部可以直接点名调用
    #（此前只能通过 compute_verify 跑固定流程，够不着「帮我算这串数的标准差」）。
    "science_calc",
    # 4A: 「搜」的对外入口 —— 直接拿关键词去 6 个公开学术索引里查真文献，
    # 不需要论文与代码也能用（外部智能体可以拿它去补背景知识）。
    "citation_check",
    "literature_search",
    # 2E-化学: 把「读」这一步的化学身份核验单独露出来（分子式↔分子量、CAS 校验位、SMILES 语法）。
    # 与 literature_search 一样**不需要代码包**，text 版连 PDF 都不用传。
    "chem_check",
    "build_repro_plan",
    "iterate_fix",
    "list_pipeline_tools",
)

# ------------------------------------------------- 四站口径（3B 对外接口重排）
#
# 对外叙事只有一套：**提 → 做 → 判 → 改**（四站）。对外工具不按"功能动词"平铺，
# 而是**各自挂到某一站**，调用方一眼能看出"我现在处于整条链的哪一段、手上应该有什么"。
#
# 站名一律从 `toolbox.STATIONS` 取，**不在这里另写一份字符串** —— 四站口径只能有一处
# 源头；界面进度条、科学工具菜单、本文件三处要能对上，靠的就是同一个元组。
S_ASSUME, S_EXPERIMENT, S_VERIFY, S_ITERATE = STATIONS
S_FLOW = "全流程（四站贯通）"
S_META = "（工具清单）"

#: 对外工具 -> 归属站。
#:
#: 划分依据是**这个工具主要产出哪一张卡**（看输入 / 输出，不看待办事项）：
#:   · `check_parameters` 出「可疑项」= 假设种子        -> ① 假设
#:   · `compute_verify`   出「算出来的数」              -> ② 实验（2C 后②只算数）
#:   · `citation_check`   出「出处支持不支持该结论」    -> ③ 验证（检索动作、核验产物）
#:   · `build_repro_plan` / `iterate_fix` 出方案与修正 -> ④ 迭代
#:   · `audit_paper` 一次贯通四站、`list_pipeline_tools` 是元信息，都不单属某一站。
TOOL_STATIONS: dict = {
    "check_parameters": S_ASSUME,
    "compute_verify": S_EXPERIMENT,
    "science_calc": S_EXPERIMENT,
    # 「搜」的两个工具都在③：一个证明"论文哪一页写了什么"（citation_check），
    # 一个给出"这个问题在人家的文献里长什么样"（literature_search）。
    # 两者都不下判定 —— 判定是③站读完之后才给的东西。
    "citation_check": S_VERIFY,
    "literature_search": S_VERIFY,
    # 化学身份核验也在③：它拿"论文声称的化学身份"去和"按原子量算出来的数"对账,
    # 产出的就是判定卡（对得上 / 对不上 / 本身不合规）—— 这是③站的正业。
    "chem_check": S_VERIFY,
    "build_repro_plan": S_ITERATE,
    "iterate_fix": S_ITERATE,
    "audit_paper": S_FLOW,
    "list_pipeline_tools": S_META,
}

#: 每站的交接物 —— 上游交给下游的**具体东西**，不是"做完了"这种虚话。
#: 外部智能体靠这张表判断"我现在手上有没有这一站的入场券"。
STATION_HANDOFF: dict = {
    S_ASSUME: {"入": "原始材料：论文 PDF ＋ 代码包",
               "出": "假设卡：可疑项 ＋ 每项可证伪的判据"},
    S_EXPERIMENT: {"入": "假设卡里推不掉的那些项",
                   "出": "实验卡：设计 ＋ 真跑的输入/输出/耗时；只出数，不下结论"},
    S_VERIFY: {"入": "实验卡里的实际值",
               "出": "判定卡：对得上 / 对不上 / 本次判不了（自己不造数）"},
    S_ITERATE: {"入": "判定卡里「对不上」的那些项",
                "出": "报告 ＋ 修正建议（草案，落不落地由人决定）"},
}

# 防呆: 归属表必须覆盖全部对外工具 —— 漏一个就意味着有工具在四站视图里凭空消失。
assert set(TOOL_STATIONS) == set(TOOL_NAMES), \
    "TOOL_STATIONS 必须与 TOOL_NAMES 一一对应"
assert set(TOOL_STATIONS.values()) <= (set(STATIONS) | {S_FLOW, S_META}), \
    "归属站必须是四站之一，或两个明确的非站点标记"

#: 状态 -> 中文标签。外部智能体读 JSON 时不必自己维护一份映射。
STATUS_LABELS: dict = {
    "consistent": "一致",
    "inconsistent": "不一致",
    "missing_in_code": "论文有·代码未找到",
    "missing_in_paper": "代码有·论文未声明",
    "not_fixed": "要求固定但未固定",
    "unverifiable": "动态值·无法静态验证",
    "internal_inconsistent": "内部自相矛盾",
}

#: 允许的代码包形态。目录（开发机解压后）与 zip（上传包）都收。
CODE_SUFFIXES: tuple = (".zip",)

#: 进程内小缓存: 同一个输入被连续问几件事（假设 / 算 / 核验 / 方案）时不重复跑审计。
#: 只缓存 4 条 —— MCP server 是长驻进程, 不设上限会缓慢吃内存。
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX = 4


# ------------------------------------------------------- 工具执行期保护
#
# 导入期的保护（`_flush_c_stdio` / `stdout_as_stderr`）已提到文件开头 ——
# 本模块 import 阶段就会顺着 toolbox 把 PyMuPDF 拉进来，必须在那之前就位。

def guarded(fn: Any) -> Any:
    """把工具执行期的 stdout 意外输出一律导向 stderr（通道保护）。

    为什么 import 期挡不住、运行期还要再挡一次:
        PyMuPDF 有两类提示, 发作时机不同 ——
          ① import 时: "The `fitz` API is deprecated ..."        （preload 挡）
          ② 第一次解析页面时: "Consider using the pymupdf_layout ..."（本装饰器挡）
        实测过: 只挡 ① 时, 第一个工具调用干净了, 第二个又开始报脏数据。

    将来新增工具**只要是经 build_server 注册的, 自动带这层保护**,
    不依赖"记得手动包一下"—— 防呆优先于省事。
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with stdout_as_stderr():
            return fn(*args, **kwargs)
    return wrapper


def preload() -> None:
    """提前导入审计链路（含 PyMuPDF），把导入期噪音挡在通道启用之前。

    一举两得: ① 脏输出发生在"还没开始收发 JSON-RPC"的时候, 天然无害;
    ② 第一次工具调用不再背负 import 抖动的耗时。

    任何一步导入失败都**不阻断服务启动** —— 缺 langgraph 就退单路径链,
    缺别的就在真正调用该能力时再如实报错（这与主程序"装不上就回落"一致）。
    """
    with stdout_as_stderr():
        for _mod in ("models", "toolbox", "pipeline", "compute",
                     "retrieval", "literature", "repro_plan", "loop", "graph"):
            try:
                __import__(f"{__package__}.{_mod}")
            except Exception:
                continue


# ------------------------------------------------------------------ 输入校验

def check_inputs(pdf_path: Any, code_path: Any) -> tuple[bool, str]:
    """校验输入路径。返回 (是否通过, 说明)。

    校验是**保守的**: 只放行真实存在的 pdf 文件 + 目录或 .zip 代码包。
    宁可拒绝并说清原因, 也不要把一个看不懂的路径丢进解析器里炸出栈。
    """
    if not pdf_path or not str(pdf_path).strip():
        return False, "缺少论文路径 pdf_path"
    if not code_path or not str(code_path).strip():
        return False, "缺少代码路径 code_path（.zip 或解压后的目录）"

    p = Path(str(pdf_path))
    c = Path(str(code_path))

    if not p.exists():
        return False, f"论文文件不存在: {p}"
    if not p.is_file():
        return False, f"论文路径不是文件: {p}"
    if p.suffix.lower() != ".pdf":
        return False, f"论文只支持 .pdf，收到: {p.suffix or '(无后缀)'}"

    if not c.exists():
        return False, f"代码路径不存在: {c}"
    if c.is_file() and c.suffix.lower() not in CODE_SUFFIXES:
        return False, f"代码包只支持目录或 {'/'.join(CODE_SUFFIXES)}，收到: {c.suffix or '(无后缀)'}"
    if not c.is_file() and not c.is_dir():
        return False, f"代码路径既不是文件也不是目录: {c}"

    return True, ""


# ------------------------------------------------------------------ 裁剪工具

def _shrink(obj: Any, *, depth: int = 3, max_items: int = 8,
            max_str: int = 300) -> Any:
    """把任意结构裁到「够读懂、不灌爆」的规模。

    调用方是另一个智能体, 它的上下文窗口是稀缺资源。这里统一限深、限项、限长,
    比"逐个字段手写白名单"更难漏 —— 新增字段不会因为忘了登记而全量漏出去。
    """
    if depth <= 0:
        return "…（已省略）"
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        out: dict = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["_more"] = f"…另有 {len(obj) - max_items} 项"
                break
            out[str(k)] = _shrink(v, depth=depth - 1, max_items=max_items,
                                  max_str=max_str)
        return out
    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        out = [_shrink(v, depth=depth - 1, max_items=max_items, max_str=max_str)
               for v in items[:max_items]]
        if len(items) > max_items:
            out.append(f"…另有 {len(items) - max_items} 项")
        return out
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _shrink(to_dict(), depth=depth, max_items=max_items,
                           max_str=max_str)
        except Exception:
            pass
    return str(obj)[:max_str]


def _ev_brief(ev: Any) -> Optional[dict]:
    """把一条 Evidence 压成「值 + 出处」。证据链的**出处处处带着走**，不回传整段原文。"""
    if ev is None:
        return None
    raw = getattr(ev, "raw_value", None)
    norm = getattr(ev, "normalized", None)
    value = raw if raw not in (None, "") else norm
    return {
        "value": _shrink(value, max_str=80),
        "location": _shrink(getattr(ev, "location", ""), max_str=120),
        "parser": _shrink(getattr(ev, "parser", ""), max_str=30),
    }


# ------------------------------------------------------------------ 跑一次审计

def _cache_key(pdf_path: str, code_path: str, mode: str, closes_loop: bool,
               use_llm: bool) -> tuple:
    return (str(Path(str(pdf_path))), str(Path(str(code_path))),
            mode, bool(closes_loop), bool(use_llm))


def run_audit_cached(pdf_path: str, code_path: str, *, mode: str = "classic",
                     closes_loop: bool = False, use_llm: bool = False,
                     user_results: Optional[dict] = None) -> tuple:
    """跑一次审计（带进程内缓存）。返回 (audit, report_md, elapsed_s, cache)。

    复用主线的两个入口, 不重写任何一步:
        mode="classic"    -> pipeline.run_audit        （经典单智能体链）
        mode="multagent"  -> graph.run_audit_multagent （多智能体对撞链，可带闭环）

    `use_llm=False`（默认）会走 force_mock: 不发起任何模型调用、成本严格为零,
    结论里凡涉及模型的部分都会如实标注「模拟」。要真调模型请显式开。
    """
    key = _cache_key(pdf_path, code_path, mode, closes_loop, use_llm)
    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return hit[0], hit[1], hit[2], "hit"

    t0 = time.time()
    if mode == "multagent":
        from .graph import run_audit_multagent as _run
        audit, report = _run(
            pdf_path, code_path,
            user_results=user_results,
            force_mock=not use_llm,
            loop=bool(closes_loop),
            loop_rounds=3,
        )
    else:
        from .pipeline import run_audit as _run
        audit, report = _run(
            pdf_path, code_path,
            user_results=user_results,
            force_mock=not use_llm,
        )
    elapsed = time.time() - t0

    _CACHE[key] = (audit, report, elapsed)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return audit, report, elapsed, "miss"


def clear_cache() -> None:
    """清空进程内缓存（测试用；长驻服务一般不需要手动清）。"""
    _CACHE.clear()


# ------------------------------------------------------------------ 摘要

def summarize_audit(audit: Any, *, findings_limit: int = 30) -> dict:
    """把 AuditResult 压成给外部智能体看的摘要。

    只给「能据以行动」的信息: 有几条、分别是什么、证据在哪一页/哪一行。
    **不含** page_texts / 报告全文 这类大字段 —— 需要全文请用主程序的导出入口。
    """
    findings = list(getattr(audit, "findings", None) or [])
    counts: dict = {}
    for f in findings:
        st = getattr(getattr(f, "status", None), "value", None) or str(
            getattr(f, "status", ""))
        counts[st] = counts.get(st, 0) + 1

    items = []
    for f in findings[:findings_limit]:
        paper_ev = _ev_brief(getattr(f, "paper", None))
        code_ev = _ev_brief(getattr(f, "code", None))
        st = getattr(getattr(f, "status", None), "value", None) or str(
            getattr(f, "status", ""))
        items.append({
            "param": getattr(f, "param_key", ""),
            "display_name": getattr(f, "display_name", ""),
            "status": st,
            "status_label": STATUS_LABELS.get(st, st),
            "risk": getattr(getattr(f, "risk", None), "value", None) or str(
                getattr(f, "risk", "")),
            "confidence": getattr(getattr(f, "confidence", None), "value", None)
            or str(getattr(f, "confidence", "")),
            "paper": paper_ev,
            "code": code_ev,
            "note": _shrink(getattr(f, "note", ""), max_str=200),
        })

    paper = getattr(audit, "paper", None)
    code = getattr(audit, "code", None)
    readiness = getattr(audit, "readiness", None) or {}
    coverage = getattr(audit, "coverage", None) or {}
    selfcheck = getattr(audit, "selfcheck", None) or {}

    summary = {
        "paper": {
            "title": _shrink(getattr(paper, "title", ""), max_str=160),
            "pages": getattr(paper, "page_count", 0),
            "params_declared": len(getattr(paper, "params", None) or {}),
        },
        "code": {
            "python_files": getattr(code, "python_files", 0),
            "params_found": len(getattr(code, "params", None) or {}),
            "seed_fixed": bool(getattr(code, "seed_fixed", False)),
            "requirements": len(getattr(code, "requirements", None) or {}),
        },
        "counts": counts,
        "findings_total": len(findings),
        "findings": items,
        "readiness": {
            "total": readiness.get("total"),
            "categories": _shrink(readiness.get("categories") or {}),
            "dominant_issues": _shrink(readiness.get("dominant_issues") or []),
            "audit_confidence": _shrink(readiness.get("audit_confidence") or {}),
        },
        "selfcheck": {
            "checked": selfcheck.get("checked"),
            "passed": selfcheck.get("passed"),
            "failed": selfcheck.get("failed"),
        },
        "coverage_confidence": {
            "level": coverage.get("confidence_level"),
            "score": coverage.get("confidence_score"),
        },
    }
    if len(findings) > findings_limit:
        summary["findings_truncated"] = (
            f"共 {len(findings)} 条, 只返回前 {findings_limit} 条")
    return summary


# ------------------------------------------------------------------ 工具实现（纯函数，不依赖 mcp 包）

def _fail(msg: str, **extra: Any) -> dict:
    out = {"ok": False, "error": msg}
    out.update(extra)
    return out


def tool_audit_paper(pdf_path: str, code_path: str, mode: str = "classic",
                     closes_loop: bool = False, use_llm: bool = False,
                     findings_limit: int = 30,
                     include_report: bool = False) -> dict:
    """完整审计: 论文 x 代码 -> 参数一致性结论 + 证据出处。

    `include_report=False`（默认）只给报告开头的预览 —— 完整报告动辄上万字符,
    默认塞进响应等于替调用方花掉一大块上下文预算。要全文就显式要。
    """
    ok, why = check_inputs(pdf_path, code_path)
    if not ok:
        return _fail(why)
    if mode not in ("classic", "multagent"):
        return _fail(f"mode 只支持 classic / multagent，收到: {mode}")
    if closes_loop and mode != "multagent":
        # 与界面同一套语义: 开闭环必然走多智能体链（闭环节点挂在多智能体图上）
        mode = "multagent"
    try:
        audit, report, elapsed, cache = run_audit_cached(
            pdf_path, code_path, mode=mode, closes_loop=closes_loop,
            use_llm=use_llm)
    except Exception as exc:  # 不抛给客户端: 结构化返回更容易被智能体处理
        return _fail(f"审计执行失败: {type(exc).__name__}: {exc}")

    out = {
        "ok": True,
        "mode": mode,
        "closes_loop": bool(closes_loop),
        "llm": "已启用" if use_llm else "未调用（force_mock，成本为零；涉及模型的部分标注为模拟）",
        "elapsed_s": round(elapsed, 3),
        "cache": cache,
    }
    if include_report:
        out["report_markdown"] = report or ""
    else:
        out["report_preview"] = _shrink(report or "", max_str=800)
        out["report_note"] = (
            f"完整报告共 {len(report or '')} 字符；需要全文请把 include_report 设为 true")
    out.update(summarize_audit(audit, findings_limit=findings_limit))
    return out


def tool_check_parameters(pdf_path: str, code_path: str, use_llm: bool = False,
                          findings_limit: int = 30,
                          include_report: bool = False) -> dict:
    """最小竖切（M1）: 只回答「哪几个关键参数对不上、证据在哪」，不跑闭环。"""
    return tool_audit_paper(pdf_path, code_path, mode="classic",
                            closes_loop=False, use_llm=use_llm,
                            findings_limit=findings_limit,
                            include_report=include_report)


def tool_compute_verify(pdf_path: str, code_path: str,
                        findings_limit: int = 30) -> dict:
    """「算」: 小规模复算与静态推演（关系式一致性 + 量纲提示）。"""
    ok, why = check_inputs(pdf_path, code_path)
    if not ok:
        return _fail(why)
    try:
        audit, _report, elapsed, cache = run_audit_cached(
            pdf_path, code_path, mode="multagent", closes_loop=True)
    except Exception as exc:
        return _fail(f"「算」执行失败: {type(exc).__name__}: {exc}")
    compute = getattr(audit, "compute", None) or {}
    if not compute:
        return _fail("本次没有得到「算」的产物（闭环链未产出）",
                     elapsed_s=round(elapsed, 3), cache=cache)
    out = {
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "cache": cache,
        "boundary": "只在纸面推演，不执行用户代码；训练级复现一律标注「需实机验证」",
    }
    out.update(_shrink(compute, depth=4, max_items=12, max_str=400))
    return out


def tool_science_calc(tool: str, inputs: Optional[dict] = None,
                      inputs_json: str = "") -> dict:
    """② 实验站 · 具名科学计算工具：**只算数，不下判断**。

    为什么单独开一个对外工具（而不是让调用方去猜 `compute_verify` 内部有什么）：
        `compute_verify` 跑的是**固定流程**（关系式复算 ＋ 量纲提示），
        外部智能体够不着「帮我算这串数的标准差」「这组点的线性度怎么样」。
        3A 已经在②站内部登记了 4 件具名工具，这里只是把它们**原样露出来**，
        不复制任何实现 —— 调用一律走 `scifuncs.run()` 这一个入口。

    三条对外承诺：
      · **清单外不存在**：只认 `SCI_TOOLS` 里登记过的键，其余一律拒；
      · **不造数**：输入由调用方给（它手上的材料），本工具不补、不猜、不平滑；
      · **不下判断**：返回里没有 `verdict` / `compare`，对得上还是对不上归③站。
    """
    import json

    from .scifuncs import SCI_TOOLS, menu, run

    keys = [t.key for t in SCI_TOOLS]

    raw: Any = inputs
    # `inputs` 给了东西却不是对象 —— 如实拒，**不当成空输入接着跑**
    #（静默当空会产出"算了个寂寞"的结果，比报错难查得多）。
    if raw is not None and not isinstance(raw, dict):
        return _fail(f"inputs 必须是对象（JSON object），收到的是 {type(raw).__name__}；"
                     f"只能发字符串的客户端请改用 inputs_json")
    if raw is None and str(inputs_json or "").strip():
        try:
            raw = json.loads(inputs_json)
        except Exception as ex:      # noqa: BLE001 —— 调用方给的串不合法, 如实说
            return _fail(f"inputs_json 不是合法 JSON: {type(ex).__name__}: {ex}")
    if not isinstance(raw, dict):
        raw = {}

    key = str(tool or "").strip()
    if key not in keys:
        return {
            "ok": False,
            "text": (f"没有名为 `{key}` 的科学计算工具 —— 菜单外的东西**本次不存在**"
                     f"（可用: {', '.join(keys)}）"),
            "available": menu(),
            "station": S_EXPERIMENT,
        }

    out = dict(run(key, raw))
    # 失败一律带 `error`（与本 server 其他对外工具同款），别让调用方记两套字段；
    # 成功时 `text` 是工具的人话结论。
    if out.get("ok") is not True and out.get("text") and not out.get("error"):
        out["error"] = out["text"]
    out["tool"] = key
    out["station"] = S_EXPERIMENT
    out["boundary"] = ("② 站只出数、不下判断：本返回里没有 verdict / compare，"
                       "「对得上还是对不上」归 ③ 验证站")
    # 兜底: 即使 run 那边以后被改成会带 compare, 也不让判定从②站漏出去
    out.pop("compare", None)
    out.pop("verdict", None)
    return out


def _external_note(citations: Any) -> str:
    """外部检索的一句话交代 —— 开没开、查了几个源、给了几条，全照实写。

    这段话会出现在 `citation_check` 的返回里，调用方常拿它判断"这份结果背后
    有没有外部依据"，所以既不能漏报（明明查了说没查），也不能虚报（没查到说查到了）。
    """
    ext = (citations or {}).get("external") or {}
    if not ext.get("enabled"):
        return "未启用；外部线索一律标记未核验，需人工确认后方可使用"
    n = len(ext.get("items") or [])
    return (f"已启用多源联邦检索（{ext.get('reached', 0)}/{ext.get('of', 0)} 个源返回，"
            f"{n} 条候选，均未核验）；文献真实存在，是否支持结论需人读确认")


def tool_literature_search(query: str, limit: int = 8, providers: str = "",
                           year_from: int = 0) -> dict:
    """③ 验证站 · 外部文献检索（真实多源，免费无 key）: 拿关键词去 6 个公开学术索引里查。

    为什么单独开这个工具（而不是只留在 `citation_check` 里）：
        `citation_check` 必须先有论文和代码才能跑，且它搜的是"这篇论文自己"；
        外部智能体常常只想问"这个方法别人怎么做的" —— 那是纯关键词检索，
        不该被迫先传一份 PDF。

    六个源各查各的、并发跑，一个挂了不影响别的（失败原因逐源回传）：
        openalex / crossref / europepmc / arxiv / datacite / doaj

    两条承诺：
      · **不编文献**：每条都来自真实 HTTP 响应，带 DOI 或落地页，点开就能核；
      · **不冒充核验**：`verified` 恒为 False —— "文献真实存在"与"它支持你的结论"
        是两件事，后者只能人读。
    """
    from .retrieval import literature_search

    q = str(query or "").strip()
    if not q:
        return _fail("query 不能为空 —— 没有检索词就不发请求（不拿空查询去试探人家的库）")

    provs = [p.strip() for p in str(providers or "").split(",") if p.strip()]
    kw: dict = {}
    if provs:
        kw["providers"] = provs
    if int(year_from or 0) > 0:
        kw["year_from"] = int(year_from)

    r = literature_search(query=q, limit=max(1, min(int(limit or 8), 30)), **kw)
    if not r.get("enabled"):
        return _fail(r.get("note") or "外部文献检索已关闭（PAPER2LAB_LIT=off）")

    items = []
    for it in (r.get("items") or []):
        d = dict(it)
        d["abstract"] = (d.get("abstract") or "")[:400]   # 对外别传整段摘要
        items.append(d)
    return {
        "ok": True,
        "query": q,
        "items": items,
        "n": len(items),
        "sources": r.get("sources") or [],
        "reached": r.get("reached"),
        "of": r.get("of"),
        "providers": r.get("providers") or {},
        "mode": r.get("mode"),
        "note": r.get("note"),
        "station": S_VERIFY,
        "boundary": ("文献记录真实存在（可点 DOI 核），但「是否支持你的结论」需人读确认；"
                     "本工具只给候选依据，不产出判定。"),
    }


def tool_chem_check(pdf_path: str = "", text: str = "") -> dict:
    """③ 验证站 · 化学身份核验（分子式 / CAS / SMILES / 分子量对账）。

    两条输入路径，给一个就够:
      · `pdf_path` —— 读整篇论文正文，抽出所有化学身份并逐条核对;
      · `text` —— 直接给一段原文（外部智能体常常只想核一个式子，不必先传 PDF）。

    核什么（全部确定性、本地零成本）:
      · **分子式 → 分子量**：按内置 IUPAC 原子量复算，再按论文自己给的小数位舍入后比较;
      · **CAS 号**：校验位算法（算错 = 抄错）;
      · **SMILES**：括号 / 环闭合 / 元素符号的语法体检（装了 RDKit 则做精确解析）。

    **不做什么**：不从图像像素猜分子结构、不给化合物命名。
    认结构靠猜（幻觉重灾区），对账靠算（可核验）—— 本工具只做后者。
    """
    from . import chem

    raw = str(text or "").strip()
    if not raw and not str(pdf_path or "").strip():
        return _fail("要么给 pdf_path（读整篇论文），要么给 text（一段原文）；两个都空就没法核")

    if raw:
        items = chem.measure_text(raw, page=0)
        stats = {"total": len(items), "pages": 1, "kinds": {},
                 "engine": chem.engine_name(), "rdkit": bool(chem.rdkit_analyze("CCO")),
                 "notes": []}
        for it in items:
            stats["kinds"][it.kind] = stats["kinds"].get(it.kind, 0) + 1
        source = "text"
    else:
        if not os.path.isfile(str(pdf_path)):
            return _fail(f"找不到论文文件: {pdf_path}")
        try:
            from .paper_parser import parse_paper
            paper = parse_paper(str(pdf_path))
        except Exception as exc:                              # noqa: BLE001
            return _fail(f"论文解析失败: {type(exc).__name__}: {exc}")
        items, stats = chem.measure(paper)
        source = "pdf"

    items = chem.judge(items)
    summary = chem.summarize(items, stats)
    return {
        "ok": True,
        "source": source,
        "items": [it.to_dict() for it in items],
        "n": len(items),
        "summary": {k: v for k, v in summary.items() if k != "engine"},
        "engine": stats.get("engine") or chem.engine_name(),
        "rdkit": bool(stats.get("rdkit")),
        "station": S_VERIFY,
        "boundary": ("只核对论文**自己写的**化学身份是否自洽（式子↔分子量、CAS 校验位、"
                     "SMILES 语法）；不猜分子结构、不给化合物命名、不判断该化合物是否存在。"),
    }


def tool_citation_check(pdf_path: str, code_path: str,
                        top_k: int = 2) -> dict:
    """「搜」: 每条结论回原文页码 / 代码行列的引证核验（附带外部文献候选）。"""
    ok, why = check_inputs(pdf_path, code_path)
    if not ok:
        return _fail(why)
    try:
        audit, _report, elapsed, cache = run_audit_cached(
            pdf_path, code_path, mode="multagent", closes_loop=True)
    except Exception as exc:
        return _fail(f"「搜」执行失败: {type(exc).__name__}: {exc}")
    citations = getattr(audit, "citations", None) or {}
    if not citations:
        return _fail("本次没有得到引证产物（闭环链未产出）",
                     elapsed_s=round(elapsed, 3), cache=cache)
    out = {
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "cache": cache,
        "search_mode": "字面检索（同义词扩展 + 取值命中），非向量/语义检索",
        "external_search": _external_note(citations),
    }
    out.update(_shrink(citations, depth=4, max_items=12, max_str=400))
    return out


def tool_build_repro_plan(pdf_path: str, code_path: str) -> dict:
    """「做」: 七步复现方案 + 可复现性评级（A/B/C/D）+ 扣分理由。"""
    ok, why = check_inputs(pdf_path, code_path)
    if not ok:
        return _fail(why)
    try:
        audit, _report, elapsed, cache = run_audit_cached(
            pdf_path, code_path, mode="multagent", closes_loop=True)
    except Exception as exc:
        return _fail(f"「做」执行失败: {type(exc).__name__}: {exc}")
    plan = getattr(audit, "repro_plan", None) or {}
    if not plan:
        return _fail("本次没有得到复现方案（闭环链未产出）",
                     elapsed_s=round(elapsed, 3), cache=cache)
    out = {
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "cache": cache,
        "grade_scale": "A >= 85 / B >= 70 / C >= 55 / D < 55",
    }
    out.update(_shrink(plan, depth=4, max_items=12, max_str=400))
    return out


def tool_iterate_fix(pdf_path: str, code_path: str, max_rounds: int = 3) -> dict:
    """迭代回边: 对不上的参数定位根因 -> 给修正草案 -> 复验，有界（<= 3 轮）。"""
    ok, why = check_inputs(pdf_path, code_path)
    if not ok:
        return _fail(why)
    try:
        audit, _report, elapsed, cache = run_audit_cached(
            pdf_path, code_path, mode="multagent", closes_loop=True)
    except Exception as exc:
        return _fail(f"迭代回边执行失败: {type(exc).__name__}: {exc}")
    loop = getattr(audit, "loop", None) or {}
    if not loop:
        return _fail("本次没有得到迭代留痕（闭环链未产出）",
                     elapsed_s=round(elapsed, 3), cache=cache)
    out = {
        "ok": True,
        "elapsed_s": round(elapsed, 3),
        "cache": cache,
        "requested_rounds": max_rounds,
        "boundary": ("修正只在内存里反事实推演，不写文件、不改用户代码；"
                     "收敛判据是「可落地」而非「把数字改成一样」"),
    }
    out.update(_shrink(loop, depth=4, max_items=12, max_str=400))
    return out


# ------------------------------------------------------------------ 四站视图

def station_view() -> dict:
    """按四站分组暴露本次能力 —— 3B 的对外口径。

    返回结构刻意做成「入 / 出」两栏: 调用方拿到的不是一个功能清单, 而是
    **整条链的交接顺序** —— 知道自己在哪一段、下游需要什么, 才不会把本该由
    ③ 下的判定当成 ② 的产物去用。

    ② 的计算工具**直接取自 `science_tools` 的同源菜单**, 不在此处手抄:
    3A 往 `scifuncs` 加一件工具, 这里会自动跟着出现; 手抄的版本必然漂移
    （加了工具忘了改, 对外就少报一件 —— 那是对外承诺与实物不符）。
    """
    from .science_tools import science_menu

    sci = science_menu()
    groups: dict = {}
    for st in STATIONS:
        groups[st] = {
            "入": STATION_HANDOFF[st]["入"],
            "出": STATION_HANDOFF[st]["出"],
            # 这一站在做什么: 去掉 ** 排版标记（只去标记, 不改一个字）
            "说明": STATION_DESC.get(st, "").replace("**", ""),
            "对外工具": [n for n in TOOL_NAMES if TOOL_STATIONS[n] == st],
            "计算工具": [c["key"] for c in sci if c["归属站"] == st],
        }
    return {
        "叙述": ("提 → 做 → 判 → 改: ① 假设站提出可疑项, ② 实验站真算, "
                 "③ 验证站下判定, ④ 迭代站出报告与修正建议"),
        "四站": groups,
        "非站点工具": [n for n in TOOL_NAMES if TOOL_STATIONS[n] in (S_FLOW, S_META)],
        "计算工具总数": len(sci),
    }


def tool_list_pipeline_tools() -> dict:
    """列出内部流水线的全部工具登记表（needs / gives / 成本 / 归属站）。

    这正是本 server 封装的那条流水线 —— 上面几个工具是「按用途切」的入口，
    这里是「按工序看」的全景；`stations` 再按四站把**对外工具与计算工具**
    归好类，调用方不必自己拿工序表推一遍。
    """
    from .science_tools import CATALOG
    from .toolbox import STEPS, TOOLS

    station_of = {c.key: c.station for c in CATALOG}
    return {
        "ok": True,
        "server": SERVER_NAME,
        "version": SERVER_VERSION,
        "pipeline_tools": [
            {
                "key": t.key,
                "title": t.title,
                "checkpoints": list(t.checkpoints),
                "needs": list(t.needs),
                "gives": list(t.gives),
                "cost": t.cost,
                "core": bool(t.core),
                # 归属站与科学工具菜单同源（同一张 CATALOG），不另写一份
                "station": station_of.get(t.key, ""),
            }
            for t in TOOLS
        ],
        "stations": station_view(),
        "steps": list(STEPS),
        "steps_count": len(STEPS),
        "exposed_tools": list(TOOL_NAMES),
        "tool_stations": dict(TOOL_STATIONS),
    }


# ------------------------------------------------------------------ FastMCP 绑定

def build_server(host: str = "127.0.0.1", port: int = 8765) -> Any:
    """构造 FastMCP server。

    transport 由调用方决定:
        server.run(transport="stdio")            本地客户端直连
        server.run(transport="streamable-http")  平台 / 远端接入
    """
    if FastMCP is None:  # pragma: no cover - 取决于环境
        raise RuntimeError(
            "未安装 mcp SDK，无法启动 MCP 服务。请先执行:\n"
            "    pip install \"mcp>=1.0,<2\"\n"
            f"（原始错误: {_MCP_IMPORT_ERROR}）"
        )

    # 先把重依赖导完（导入期的脏输出挡在通道启用之前），再挂工具
    preload()

    server = FastMCP(
        name=SERVER_NAME,
        instructions=(
            "Paper2Lab 科研复现一致性审计，按四站组织："
            "① 假设站（原始材料 -> 假设卡：哪几项可疑、各自怎么证伪）-> "
            "② 实验站（推不掉的项 -> 实验卡：真算出的数与执行记录，只出数、不下结论）-> "
            "③ 验证站（实际值 -> 判定卡：对得上 / 对不上，自己不造数）-> "
            "④ 迭代站（不通过项 -> 报告与修正建议）。"
            f"{len(TOOL_NAMES)} 个工具各挂其中一站，工具说明以站名开头；"
            "audit_paper 一次贯通四站，"
            "list_pipeline_tools 给出四站全景（每站入什么、出什么、谁负责）。"
            "所有结论都带出处（论文页码 / 代码 file:line），规则优先、模型可选、默认零成本。"
        ),
        host=host,
        port=port,
    )

    # 工具清单集中在一处声明: 新增能力只需加一行, 且**自动带上 guarded 保护**
    # （执行期的 stdout 意外输出会被导向 stderr, 见 guarded 的说明）。
    #
    # 3B: 顺序按四站排（① -> ② -> ③ -> ④ -> 全流程 -> 清单），说明以站名开头。
    # ★ 工具名**一个都没改** —— 它是对外契约（已有客户端与 skill 包都在用）；
    #   重排只动「顺序 + 说明」，让调用方看到的是流程顺序而不是功能平铺。
    #   站名一律取自 TOOL_STATIONS，与 station_view() 同一份口径。
    _specs: tuple = (
        ("check_parameters",
         "① 假设站 · 出假设卡: 只回答「哪几个关键参数对不上、证据在哪」，"
         "不跑闭环，最快。先用它筛出可疑项，再决定值不值得往下做。",
         tool_check_parameters),
        ("compute_verify",
         "② 实验站 · 出实验卡（算）: 对两侧数值做小规模复算与静态推演"
         "（关系式一致性、量纲疑似写错提示），只出数、不下结论。"
         "只在纸面推演，不执行用户代码；训练级复现一律标注「需实机验证」。",
         tool_compute_verify),
        ("science_calc",
         "② 实验站 · 具名科学计算工具（直接点名调用）: 在 "
         "fp16_precision（浮点精度）/ series_stats（序列统计）/ "
         "linear_fit（直线拟合 + R²）/ peak_pick（峰位）里挑一件真算。"
         "参数 tool 选工具、inputs 给输入（也可用 inputs_json 传 JSON 串）。"
         "**只出数、不下判断**：没有 verdict，对得上还是对不上归③站。",
         tool_science_calc),
        ("citation_check",
         "③ 验证站 · 出判定卡（搜转核验）: 为每条结论检索出处"
         "（论文页优先、其次代码 file:line），并判定这段出处是否真的支持该结论。"
         "字面检索，非语义检索。",
         tool_citation_check),
        ("literature_search",
         "③ 验证站 · 外部文献检索（真实多源，免费无 key）: 拿关键词并发查 6 个公开"
         "学术索引（openalex / crossref / europepmc / arxiv / datacite / doaj），"
         "按 DOI 去重、按相关度返回，每条都能点开核。**不需要论文和代码也能用**。"
         "返回的是候选依据，`verified` 恒为 False —— 是否支持你的结论需人读摘要确认。",
         tool_literature_search),
        ("chem_check",
         "③ 验证站 · 化学身份核验: 抽出论文里的分子式 / CAS / SMILES / 分子量声明，"
         "用化学规则**自己算一遍**再对账 —— 分子式按 IUPAC 原子量复算分子量"
         "（判据取论文自己给的小数位）、CAS 验校验位、SMILES 体检语法。"
         "**不需要代码包**；给 pdf_path 读整篇，或给 text 只核一段原文。"
         "不猜分子结构、不给化合物命名 —— 认结构靠猜，对账靠算。",
         tool_chem_check),
        ("build_repro_plan",
         "④ 迭代站 · 出方案: 七步复现方案（环境 / 数据 / 超参 / 随机性 / 试跑 / "
         "核对 / 留痕）与可复现性评级 A/B/C/D，并逐条给出扣分理由。",
         tool_build_repro_plan),
        ("iterate_fix",
         "④ 迭代站 · 出修正建议: 对不一致项定位根因、给修正草案、复验，最多 3 轮就停。"
         "修正只在内存中推演，绝不写文件、绝不改用户代码。",
         tool_iterate_fix),
        ("audit_paper",
         "全流程（四站贯通）: 一次跑完 ① -> ④，得到参数一致性结论 + 每条结论的证据"
         "（论文页码 / 代码文件:行号）。mode=classic 走单智能体链，"
         "mode=multagent 走多智能体对撞链；closes_loop=True 额外给出"
         "「② 算 / ③ 核验 / ④ 方案与迭代回边」。默认不调用大模型（成本为零）。",
         tool_audit_paper),
        ("list_pipeline_tools",
         "工具清单 · 四站全景: 每站入什么、出什么、哪个对外工具负责、"
         "② 站挂着哪些具名计算工具，外加内部工序表（依赖、成本、归属站）。",
         tool_list_pipeline_tools),
    )
    for _name, _desc, _fn in _specs:
        server.tool(name=_name, description=_desc)(guarded(_fn))

    return server


def main(argv: Optional[list] = None) -> int:
    """命令行入口（等价于根目录 serve_mcp.py）。"""
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        prog="paper2lab-mcp",
        description="Paper2Lab 审计能力的 MCP / SCP 服务")
    ap.add_argument("--http", action="store_true",
                    help="用 streamable-http 启动（平台/远端接入），默认 stdio")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址")
    ap.add_argument("--port", type=int, default=8765, help="HTTP 监听端口")
    args = ap.parse_args(argv)

    server = build_server(host=args.host, port=args.port)
    transport = "streamable-http" if args.http else "stdio"
    # stdio 模式下 stdout 是 JSON-RPC 通道，任何日志都必须走 stderr
    print(f"[{SERVER_NAME}] transport={transport}"
          + (f" http://{args.host}:{args.port}/mcp" if args.http else ""),
          file=sys.stderr)
    server.run(transport=transport)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
