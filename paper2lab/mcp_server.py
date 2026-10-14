"""MCP / SCP 封装（阶段六 · 协议化①）——把审计能力做成标准 MCP server。

为什么单独一个模块、不进 app.py:
    MCP 是「给别的智能体调用」的入口（端砚平台把这形态叫 SCP），
    与 Streamlit 界面是**同一套能力的两种形态**。因此本模块只做
    「校验输入 -> 调用 -> 给摘要」, **不复制任何业务逻辑** ——
    审计仍然只由 pipeline / graph 实现, 两条路的结论天然一致
    （这与「经典模式与计划模式共用同一份实现」是同一条工程纪律）。

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

try:  # 没装 mcp 时, 本模块仍可被导入(纯函数部分可离线测), 只是 build_server 会报错
    from mcp.server.fastmcp import FastMCP
except ImportError as _exc:  # pragma: no cover - 取决于环境
    FastMCP = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = str(_exc)
else:
    _MCP_IMPORT_ERROR = ""

SERVER_NAME = "paper2lab-audit"
SERVER_VERSION = "1.0.0"

#: 对外暴露的工具名（测试与文档都从这里取，别在别处硬编码字符串）
TOOL_NAMES: tuple = (
    "audit_paper",
    "check_parameters",
    "compute_verify",
    "citation_check",
    "build_repro_plan",
    "iterate_fix",
    "list_pipeline_tools",
)

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

#: 进程内小缓存: 同一个输入被连续问几件事（参数/算/搜/做）时不重复跑审计。
#: 只缓存 4 条 —— MCP server 是长驻进程, 不设上限会缓慢吃内存。
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX = 4


# ------------------------------------------------------- stdout 通道保护（重要）

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
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            os.dup2(saved, 1)
        finally:
            os.close(saved)


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

    任何一步导入失败都**不阻断服务启动** —— 缺 langgraph 就退经典链,
    缺别的就在真正调用该能力时再如实报错（这与主程序"装不上就回落"一致）。
    """
    with stdout_as_stderr():
        for _mod in ("models", "toolbox", "pipeline", "compute",
                     "retrieval", "repro_plan", "loop", "graph"):
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


def tool_citation_check(pdf_path: str, code_path: str,
                        top_k: int = 2) -> dict:
    """「搜」: 每条结论回原文页码 / 代码行列的引证核验。"""
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
        "external_search": "未启用；外部线索一律标记未核验，需人工确认后方可使用",
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


def tool_list_pipeline_tools() -> dict:
    """列出内部流水线的 10 件工具登记表（needs / gives / 成本 / 是否核心）。

    这正是本 server 封装的那条流水线 —— 上面几个工具是「按用途切」的入口，
    这里是「按工序看」的全景，方便调用方理解一次审计到底做了哪些事。
    """
    from .toolbox import STEPS, TOOLS
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
            }
            for t in TOOLS
        ],
        "steps": list(STEPS),
        "steps_count": len(STEPS),
        "exposed_tools": list(TOOL_NAMES),
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
            "Paper2Lab 科研复现一致性审计。输入一篇论文 PDF 与一份代码包"
            "（目录或 .zip），输出「哪几个关键参数对不上」以及每处冲突的证据出处，"
            "并可选给出小规模复算（算）、引证核验（搜）、复现方案与评级（做）、"
            "以及有界的迭代修正（迭代回边）。所有结论都带出处，规则优先、模型可选。"
        ),
        host=host,
        port=port,
    )

    # 工具清单集中在一处声明: 新增能力只需加一行, 且**自动带上 guarded 保护**
    # （执行期的 stdout 意外输出会被导向 stderr, 见 guarded 的说明）。
    _specs: tuple = (
        ("audit_paper",
         "完整审计: 论文 PDF x 代码包 -> 参数一致性结论 + 每条结论的证据"
         "（论文页码 / 代码文件:行号）。mode=classic 走单智能体链，"
         "mode=multagent 走多智能体对撞链；closes_loop=True 额外给出"
         "「算/搜/做 + 迭代回边」。默认不调用大模型（成本为零）。",
         tool_audit_paper),
        ("check_parameters",
         "最小竖切: 只回答「哪几个关键参数对不上、证据在哪」，"
         "不跑闭环，最快。适合先用它判断这份材料值不值得细看。",
         tool_check_parameters),
        ("compute_verify",
         "「算」: 对论文与代码两侧的数值做小规模复算与静态推演"
         "（关系式一致性、量纲疑似写错提示）。只在纸面推演，"
         "不执行用户代码；训练级复现一律标注「需实机验证」。",
         tool_compute_verify),
        ("citation_check",
         "「搜」: 为每条审计结论检索出处（论文页优先、其次代码 file:line），"
         "并判定出处片段是否真的支持该结论。字面检索，非语义检索。",
         tool_citation_check),
        ("build_repro_plan",
         "「做」: 生成七步复现方案（环境 / 数据 / 超参 / 随机性 / 试跑 / "
         "核对 / 留痕）与可复现性评级 A/B/C/D，并逐条给出扣分理由。",
         tool_build_repro_plan),
        ("iterate_fix",
         "迭代回边: 对不一致项定位根因、给修正草案、复验，最多 3 轮就停。"
         "修正只在内存中推演，绝不写文件、绝不改用户代码。",
         tool_iterate_fix),
        ("list_pipeline_tools",
         "列出内部审计流水线的 10 件工具（工序、依赖、成本、是否核心），"
         "用于理解一次审计到底做了哪些事。",
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
