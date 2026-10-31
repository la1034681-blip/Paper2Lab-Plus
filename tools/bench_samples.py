"""用真实论文样例批量验证 Paper2Lab：逐件模块 + 全链路 + 边界输入。

**为什么不用 demo/ 里那几份**：demo 是项目自带的，跑通只能说明"自测集没坏"。
这里用从 arXiv 与 GitHub 真实抓下来的论文与代码，覆盖视觉、语言、图、扩散、
分子势能、科学计算等不同写法 —— 只有换一批没见过的输入，才能看出抽取规则
是不是只对某一种排版有效。

**五段测什么**

    modules   14 件审计工序逐件单独调用（同一个 Job 顺序喂），记录 ok/empty/fail
    science   4 件科学计算工具，喂构造好的数值输入，核对算出来的数对不对
    chain     全链路 run_audit（模型未配 key → 规则排计划），记录四站产物计数
    multagent 多智能体档 run_audit_multagent（对撞/仲裁/红队），抽样跑
    extra     边界输入：截断 PDF、空包、伪 PDF、缺文件、重复跑（快路径）

三态口径（与 tests/verify_stages.py 一致）：
    ok    有产物
    empty 合法空集（例：非化学论文查不出化学式、没配视觉 key 读不到图）
    fail  抛异常，或该必有产物却是空

用法：
    python tools/bench_samples.py --stage all --out "<结果目录>"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from paper2lab.pipeline import run_audit                    # noqa: E402
from paper2lab.scifuncs import SCI_TOOLS                    # noqa: E402
from paper2lab.toolbox import TOOLS, Job                     # noqa: E402


def _noop(*_a, **_k) -> None:
    return None


# ---------------------------------------------------------------- 产物判据
# 每件工具：怎么算"有产物"，以及一句人话的计数。返回 (state, detail)
def _judge(job: Job, key: str) -> tuple[str, str]:
    p, c = job.paper, job.code
    if key == "parse_paper":
        if not p or not p.page_count:
            return "fail", "没解析出页"
        return "ok", f"{p.page_count}页/{len(p.params)}参数/{len(p.results)}结果"
    if key == "analyze_images":
        meta = (p.image_meta if p else {}) or {}
        kept = meta.get("kept") or 0
        # 只认「真读到图」：enabled=True 但 kept=0 说明开了通道却一处没读到，不能算通过
        if kept:
            return "ok", f"读图 {kept} 处（候选 {meta.get('regions', 0)}）"
        return "empty", (f"未读到图（{meta.get('mode') or 'off'}，"
                         f"候选 {meta.get('regions', 0)} 处）")
    if key == "read_figures":
        n = job.figure_total or len(job.figure_clues or [])
        note = (job.figure_note or {}).get("mode", "")
        if n:
            # 没配视觉 key 时这一路只读图注文本，不能算“看过图”
            return "ok", f"{n} 条图注线索（{note or '文本层'}）"
        return "empty", "无图注线索"
    if key == "chem_check":
        n = job.chem_total or len(job.chem_items or [])
        return ("ok", f"{n} 个化学身份") if n else ("empty", "非化学论文")
    if key == "fill_missing":
        # 只认「真补到参数」；trace 本身有长度不代表补到东西（无 key 时会记 skipped）
        tr = (p.agent_trace if p else {}) or {}
        filled = tr.get("filled") or {}
        steps = tr.get("steps") or []
        if filled:
            return "ok", f"补到 {len(filled)} 个参数"
        why = tr.get("stopped_reason") or ("无 key" if not tr.get("llm_available") else "无待补项")
        return "empty", f"没补到（试 {len(steps)} 步，{why}）"
    if key == "analyze_code":
        if not c:
            return "fail", "没产出代码分析"
        n = len(c.params or {})
        raw = c.python_files
        nf = raw if isinstance(raw, int) else len(raw or [])
        if not n and not nf:
            return "empty", "包里没有可分析源码"
        return "ok", f"{nf} 个 py/{n} 个参数"
    if key == "hypothesize":
        n = job.hypothesis_total or len(job.hypotheses or [])
        return ("ok", f"{n} 条假设") if n else ("fail", "0 条假设")
    if key == "design_experiment":
        n = job.experiment_total or len(job.experiments or [])
        return ("ok", f"{n} 张实验卡") if n else ("empty", "未设计出实验")
    if key == "align":
        if job.audit is None:
            return "fail", "没产出审计结果"
        return "ok", f"{len(job.audit.findings)} 条发现"
    if key == "llm_analyze":
        if job.llm_real_calls:
            return "ok", f"模型调用 {job.llm_real_calls} 次"
        if job.llm_cache_hits:
            # 命中缓存 = 前面跑过同一批论文，文字直接复用，不重复花钱。这是好状态。
            return "ok", f"缓存命中 {job.llm_cache_hits} 条（未重复调用）"
        why = job.llm_error or job.llm_mode or "无 key"
        return "empty", f"未真正调用（{why[:60]}）"
    if key == "selfcheck":
        sc = (job.audit.selfcheck if job.audit else {}) or {}
        n = sc.get("checked", 0)
        return ("ok", f"自检 {n} 条/过 {sc.get('passed', 0)}") if n else ("fail", "自检 0 条")
    if key == "rootcause":
        return ("ok", f"{len(job.rootcause)} 条根因") if job.rootcause else ("empty", "无需追根因")
    if key == "patch":
        return ("ok", f"{len(job.patches)} 个补丁") if job.patches else ("empty", "无需补丁")
    if key == "report":
        return ("ok", f"报告 {len(job.report)} 字") if len(job.report) > 500 else ("fail", "报告过短")
    return "empty", "未登记判据"


def stage_modules(item: Path) -> dict:
    pdf, code = item / "paper.pdf", item / "code.zip"
    job = Job(pdf_path=str(pdf), code_path=str(code))
    rows, t_all = [], time.time()
    for t in TOOLS:
        t0 = time.time()
        try:
            t.run(job, _noop)
            state, detail = _judge(job, t.key)
            err = ""
        except Exception as e:  # noqa: BLE001
            state, detail = "fail", f"{type(e).__name__}: {e}"
            err = traceback.format_exc(limit=3)
        rows.append({"tool": t.key, "title": t.title, "state": state,
                     "detail": detail, "seconds": round(time.time() - t0, 2),
                     "error": err})
    return {"id": item.name, "rows": rows,
            "seconds": round(time.time() - t_all, 1),
            "counts": {s: sum(1 for r in rows if r["state"] == s)
                       for s in ("ok", "empty", "fail")}}


SCI_CASES = [
    # 1e-8 比 fp16 的最小次正规数（5.96e-8）还小 → 在 fp16 里就是 0，这一项应被判为危险
    ("fp16_precision", {"raw": [1e-8, 5.96e-8, 1e-4], "precision": "fp16"}),
    ("series_stats", {"values": [0.81, 0.83, 0.82, 0.80, 0.84]}),
    ("linear_fit", {"xs": [1, 2, 3, 4], "ys": [2.0, 4.1, 5.9, 8.05]}),
    ("peak_pick", {"xs": [1, 2, 3, 4, 5], "ys": [0.1, 0.4, 0.9, 0.5, 0.2]}),
]


def stage_science() -> dict:
    """科学工具是纯计算：输入是构造的，但算得对不对是可核的。

    期望值是我手算的 —— 拟合那组斜率应约 2.02、R² 接近 1；
    峰位应落在 x=3（不是端点，说明采样罩住了峰）。
    """
    out = []
    for key, inp in SCI_CASES:
        tool = next((t for t in SCI_TOOLS if t.key == key), None)
        if tool is None:
            out.append({"tool": key, "state": "fail", "detail": "工具不存在"})
            continue
        try:
            r = tool.run(dict(inp))
            out.append({"tool": key, "state": "ok" if r.get("ok") else "empty",
                        "detail": str(r.get("text", ""))[:160],
                        "value": r.get("value"), "actual": r.get("actual")})
        except Exception as e:  # noqa: BLE001
            out.append({"tool": key, "state": "fail",
                        "detail": f"{type(e).__name__}: {e}"})
    return {"rows": out,
            "counts": {s: sum(1 for r in out if r["state"] == s)
                       for s in ("ok", "empty", "fail")}}


def stage_chain(item: Path, loop: bool = False) -> dict:
    pdf, code = item / "paper.pdf", item / "code.zip"
    t0 = time.time()
    try:
        audit, report = run_audit(str(pdf), str(code))
    except Exception as e:  # noqa: BLE001
        return {"id": item.name, "state": "fail",
                "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4)[-1200:],
                "seconds": round(time.time() - t0, 1)}
    if audit is None:
        return {"id": item.name, "state": "fail", "error": "返回空审计结果",
                "seconds": round(time.time() - t0, 1)}
    plan = (audit.execution_log or {}).get("plan") or {}
    decisions = plan.get("decisions") or []
    status: dict = {}
    for d in decisions:
        k = d.get("action", "?")
        status[k] = status.get(k, 0) + 1
    return {
        "id": item.name, "state": "ok",
        "seconds": round(time.time() - t0, 1),
        "pages": audit.paper.page_count,
        "paper_params": len(audit.paper.params),
        "paper_results": len(audit.paper.results),
        "code_params": len(audit.code.params or {}),
        "findings": len(audit.findings),
        "verified_ok": sum(1 for f in audit.findings if f.verified),
        "hypotheses": audit.hypothesis_total or len(audit.hypotheses or []),
        "experiments": audit.experiment_total or len(audit.experiments or []),
        "figures": audit.figure_total or len(audit.figure_clues or []),
        "chem": getattr(audit, "chem_total", 0) or len(getattr(audit, "chem_items", []) or []),
        "rootcause": len(audit.rootcause or []),
        "patches": len(audit.patches or []),
        "selfcheck_checked": (audit.selfcheck or {}).get("checked", 0),
        "selfcheck_passed": (audit.selfcheck or {}).get("passed", 0),
        "report_chars": len(report or ""),
        "planner": plan.get("planner", "") or plan.get("mode", ""),
        "tools_planned": len(decisions),
        "tool_status": status,
        "skipped_tools": [d.get("tool") for d in decisions if d.get("action") == "skip"],
    }


def stage_mockllm(item: Path) -> dict:
    """走一遍「模型通道」的代码路径（force_mock）。

    **只验管道连通，不验判断质量**：mock 返回的是固定假数据，能跑通说明
    「有 key 时那条路不会因为形状不对而炸」，但模型判得准不准它一句话都证明不了。
    """
    t0 = time.time()
    try:
        audit, report = run_audit(str(item / "paper.pdf"), str(item / "code.zip"),
                                  force_mock=True)
    except Exception as e:  # noqa: BLE001
        return {"id": item.name, "state": "fail",
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}
    if audit is None:
        return {"id": item.name, "state": "fail", "error": "返回空审计结果",
                "seconds": round(time.time() - t0, 1)}
    with_ai = sum(1 for f in audit.findings if (f.ai_analysis or "").strip())
    return {"id": item.name, "state": "ok",
            "seconds": round(time.time() - t0, 1),
            "findings": len(audit.findings), "with_ai": with_ai,
            "report_chars": len(report or ""),
            "note": "mock 通道：仅验证管道形状"}


def stage_mcp(item: Path) -> dict:
    """10 个对外工具（MCP 插座）逐个点名调用。

    这是"外部智能体能不能直接拿来用"的那一层 —— 与内部工序不是同一套接口，
    所以单独测：函数能调通、返回结构是 dict、不抛异常。
    """
    from paper2lab import mcp_server as M
    pdf, code = str(item / "paper.pdf"), str(item / "code.zip")
    rows: list[dict] = []

    def run(name: str, fn) -> None:
        t0 = time.time()
        try:
            r = fn()
            ok = isinstance(r, dict) and r.get("ok") is not False
            rows.append({"tool": name, "state": "ok" if ok else "fail",
                         "seconds": round(time.time() - t0, 1),
                         "keys": sorted(r.keys())[:7] if isinstance(r, dict) else str(type(r)),
                         "error": "" if ok else str(r)[:160]})
        except Exception as e:  # noqa: BLE001
            rows.append({"tool": name, "state": "fail",
                         "seconds": round(time.time() - t0, 1),
                         "error": f"{type(e).__name__}: {e}"})

    run("audit_paper", lambda: M.tool_audit_paper(pdf, code, findings_limit=5))
    run("check_parameters", lambda: M.tool_check_parameters(pdf, code, findings_limit=5))
    run("compute_verify", lambda: M.tool_compute_verify(pdf, code, findings_limit=5))
    run("science_calc", lambda: M.tool_science_calc(
        "series_stats", {"values": [1.0, 2.0, 3.0]}))
    run("chem_check", lambda: M.tool_chem_check(pdf_path=pdf))
    run("citation_check", lambda: M.tool_citation_check(pdf, code, top_k=2))
    run("build_repro_plan", lambda: M.tool_build_repro_plan(pdf, code))
    run("iterate_fix", lambda: M.tool_iterate_fix(pdf, code, max_rounds=1))
    run("list_pipeline_tools", lambda: M.tool_list_pipeline_tools())
    # 检索走外网，单独放最后：它慢，且失败原因通常是网络而不是程序
    run("literature_search", lambda: M.tool_literature_search(
        "denoising diffusion probabilistic model", limit=5))
    return {"id": item.name, "rows": rows,
            "counts": {s: sum(1 for r in rows if r["state"] == s)
                       for s in ("ok", "fail")}}


def stage_multagent(item: Path) -> dict:
    from paper2lab.graph import run_audit_multagent
    t0 = time.time()
    try:
        audit, report = run_audit_multagent(str(item / "paper.pdf"),
                                            str(item / "code.zip"))
    except Exception as e:  # noqa: BLE001
        return {"id": item.name, "state": "fail",
                "error": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 1)}
    if audit is None:
        return {"id": item.name, "state": "fail", "error": "返回空审计结果",
                "seconds": round(time.time() - t0, 1)}
    return {"id": item.name, "state": "ok",
            "seconds": round(time.time() - t0, 1),
            "findings": len(audit.findings),
            "hypotheses": audit.hypothesis_total or len(audit.hypotheses or []),
            "experiments": audit.experiment_total or len(audit.experiments or []),
            "report_chars": len(report or "")}


# ---------------------------------------------------------------- 边界输入
def _mutants(src: Path, work: Path) -> list[tuple[str, Path, Path]]:
    """造 6 种坏输入，期望都是「不崩、说人话、不给假结论」。"""
    work.mkdir(parents=True, exist_ok=True)
    pdf, code = src / "paper.pdf", src / "code.zip"
    cases: list[tuple[str, Path, Path]] = []

    trunc = work / "trunc.pdf"
    raw = pdf.read_bytes()
    trunc.write_bytes(raw[: len(raw) // 3])
    cases.append(("PDF 截断到 1/3", trunc, code))

    fake = work / "fake.pdf"
    fake.write_bytes(b"this is not a pdf at all\n" * 50)
    cases.append(("伪装成 PDF 的文本", fake, code))

    empty = work / "empty.zip"
    empty.write_bytes(b"")
    cases.append(("0 字节代码包", pdf, empty))

    nopy = work / "nopy.zip"
    with zipfile.ZipFile(nopy, "w") as z:
        z.writestr("readme.txt", "no source code here")
        z.writestr("data.csv", "a,b\n1,2\n")
    cases.append(("包里没有源码", pdf, nopy))

    cases.append(("代码包路径不存在", pdf, work / "nope.zip"))
    cases.append(("只给论文不给代码", pdf, None))
    return cases


def stage_extra(items: list[Path], work: Path) -> list[dict]:
    out: list[dict] = []
    for src in items:
        for label, pdf, code in _mutants(src, work / src.name):
            t0 = time.time()
            rec = {"sample": src.name, "case": label}
            code_arg = "" if code is None else str(code)
            try:
                audit, _ = run_audit(str(pdf), code_arg)
            except Exception as e:  # noqa: BLE001
                rec.update(state="raised", detail=f"{type(e).__name__}: {e}",
                           seconds=round(time.time() - t0, 1))
                out.append(rec)
                continue
            findings = len(audit.findings) if audit else -1
            # 合格线：不崩、结论里没有凭空编出来的「一致」
            rec.update(state="handled",
                       detail=(f"findings={findings} 页={audit.paper.page_count if audit else '-'}"),
                       findings=findings, seconds=round(time.time() - t0, 1))
            out.append(rec)
    return out


def stage_fastpath(item: Path) -> dict:
    """同一份连跑两次：第二次应走快路径（解析复用），结论必须与第一次一致。"""
    pdf, code = str(item / "paper.pdf"), str(item / "code.zip")
    from paper2lab.paper_parser import parse_paper
    from paper2lab.code_analyzer import analyze_code
    t0 = time.time()
    a1, _ = run_audit(pdf, code)
    t1 = time.time() - t0
    snap = (parse_paper(pdf), analyze_code(code))
    t0 = time.time()
    a2, _ = run_audit(pdf, code, needs_index=False, snapshot=snap)
    t2 = time.time() - t0
    same = (a1 is not None and a2 is not None
            and len(a1.findings) == len(a2.findings)
            and [f.status.value for f in a1.findings] == [f.status.value for f in a2.findings])
    return {"id": item.name, "state": "ok" if same else "fail",
            "first_seconds": round(t1, 1), "second_seconds": round(t2, 1),
            "findings": [len(a1.findings) if a1 else -1, len(a2.findings) if a2 else -1],
            "same": same}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(Path.home() / "Desktop" / "Paper2Lab测试集" / "50份样例"))
    ap.add_argument("--out", default="")
    ap.add_argument("--stage", default="all",
                    choices=["modules", "science", "chain", "multagent", "mockllm",
                             "mcp", "extra", "fastpath", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=6, help="多智能体/边界段抽样份数")
    ap.add_argument("--only", default="", help="只测这些编号，逗号分隔（例 S43,S44）")
    args = ap.parse_args()

    base = Path(args.dir)
    # 只认「真的是一份样例」的目录：paper.pdf 与 code.zip 同时在场
    items = sorted(p for p in base.iterdir()
                   if p.is_dir() and not p.name.startswith("_")
                   and (p / "paper.pdf").exists() and (p / "code.zip").exists())
    if args.only:
        keys = [x.strip() for x in args.only.split(",") if x.strip()]
        items = [p for p in items if any(p.name.startswith(k) for k in keys)]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"没有样例：{base}")
        return 1
    out = Path(args.out) if args.out else base / "_结果"
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, obj) -> None:
        (out / f"{name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    stages = ["modules", "science", "chain", "multagent", "mockllm", "mcp",
              "extra", "fastpath"] if args.stage == "all" else [args.stage]

    if "modules" in stages:
        print(f"== 逐件模块：{len(items)} 份 ==", flush=True)
        res = []
        for i, it in enumerate(items, 1):
            r = stage_modules(it)
            res.append(r)
            dump("modules", res)
            print(f"[{i:02d}/{len(items)}] {it.name:<40} {r['counts']} {r['seconds']}s",
                  flush=True)
        agg: dict[str, dict] = {}
        for r in res:
            for row in r["rows"]:
                d = agg.setdefault(row["tool"], {"ok": 0, "empty": 0, "fail": 0})
                d[row["state"]] += 1
        dump("modules_agg", agg)
        print("-- 逐件汇总 --")
        for k, v in agg.items():
            print(f"   {k:<20} ok={v['ok']:>2} empty={v['empty']:>2} fail={v['fail']:>2}")

    if "science" in stages:
        print("== 科学计算 4 件 ==", flush=True)
        r = stage_science()
        dump("science", r)
        for row in r["rows"]:
            print(f"   {row['tool']:<16} {row['state']:<6} {row['detail'][:70]}")

    if "chain" in stages:
        print(f"== 全链路：{len(items)} 份 ==", flush=True)
        res = []
        for i, it in enumerate(items, 1):
            r = stage_chain(it)
            res.append(r)
            dump("chain", res)
            if r["state"] == "ok":
                print(f"[{i:02d}/{len(items)}] {it.name:<38} 发现{r['findings']:>2} "
                      f"假设{r['hypotheses']:>2} 实验{r['experiments']:>2} "
                      f"自检{r['selfcheck_checked']:>3} {r['seconds']}s", flush=True)
            else:
                print(f"[{i:02d}/{len(items)}] {it.name:<38} FAIL {r.get('error')}", flush=True)
        ok = [r for r in res if r["state"] == "ok"]
        print(f"-- 全链路 {len(ok)}/{len(res)} 成功，"
              f"总耗时 {round(sum(r['seconds'] for r in res) / 60, 1)} 分钟 --")

    if "multagent" in stages:
        pick = items[: args.sample]
        print(f"== 多智能体档：{len(pick)} 份 ==", flush=True)
        res = []
        for i, it in enumerate(pick, 1):
            r = stage_multagent(it)
            res.append(r)
            dump("multagent", res)
            print(f"[{i}/{len(pick)}] {it.name:<38} {r['state']} "
                  f"发现{r.get('findings', '-')} 报告{r.get('report_chars', '-')} "
                  f"{r['seconds']}s", flush=True)

    if "mockllm" in stages:
        pick = items[: args.sample]
        print(f"== 模拟模型通道：{len(pick)} 份（只验管道，不验判断质量） ==", flush=True)
        res = []
        for it in pick:
            r = stage_mockllm(it)
            res.append(r)
            dump("mockllm", res)
            print(f"   {it.name:<38} {r['state']} 发现{r.get('findings', '-')} "
                  f"带AI归因{r.get('with_ai', '-')} {r['seconds']}s", flush=True)

    if "mcp" in stages:
        pick = items[: min(3, args.sample)]
        print(f"== 对外工具 10 个：{len(pick)} 份 ==", flush=True)
        agg: dict = {}
        for it in pick:
            r = stage_mcp(it)
            dump(f"mcp_{it.name}", r)
            for row in r["rows"]:
                d = agg.setdefault(row["tool"], {"ok": 0, "fail": 0})
                d[row["state"]] += 1
            print(f"   {it.name:<38} {r['counts']}", flush=True)
        dump("mcp_agg", agg)
        for k, v in agg.items():
            print(f"   {k:<22} ok={v['ok']} fail={v['fail']}")

    if "extra" in stages:
        pick = items[: args.sample]
        print(f"== 边界输入：{len(pick)} 份 × 6 种坏输入 ==", flush=True)
        res = stage_extra(pick, out / "_mutants")
        dump("extra", res)
        for r in res:
            print(f"   {r['sample'][:18]:<18} {r['case']:<14} {r['state']:<8} {r['detail'][:50]}")

    if "fastpath" in stages:
        pick = items[: args.sample]
        print(f"== 快路径一致性：{len(pick)} 份 ==", flush=True)
        res = []
        for it in pick:
            r = stage_fastpath(it)
            res.append(r)
            dump("fastpath", res)
            print(f"   {it.name:<38} {r['state']} {r['first_seconds']}s→"
                  f"{r['second_seconds']}s 一致={r['same']}", flush=True)

    print(f"\n结果已写入：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
