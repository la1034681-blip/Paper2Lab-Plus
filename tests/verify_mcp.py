"""协议化① 验证：MCP / SCP 封装（阶段六）。

关注点（每条都对应一个真实会错的地方）:
1. **stdio 通道纯净性** —— stdout 是 JSON-RPC 专线，多一行脏数据客户端就断。
   本轮实测踩过一次：PyMuPDF 有**两类**提示会写 stdout，一类在 import 时、
   一类在第一次解析页面时，只挡一类不够。这条测试把它钉死。
2. **封装不改结论** —— MCP 只是入口，审计仍由 pipeline / graph 实现。
   因此同一输入经 MCP 拿到的结论，必须与直接调 pipeline 逐条一致。
3. **失败要结构化** —— 路径不存在 / 模式非法 / 缺产物，都返回 ok=False 的 JSON，
   而不是抛异常让调用方拿到栈。
4. **只读** —— 跑一轮 MCP 调用前后，输入目录哈希一致。
5. **返回值不灌爆上下文** —— 摘要必须裁剪，不回传 page_texts 这类大字段。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 本项是**协议契约**测试，不测外网：把真实文献检索关掉，
# 免得「今天网通不通」影响 stdio/HTTP 握手的判定（真实检索由 verify_literature 专测）。
os.environ["PAPER2LAB_LIT"] = "off"

from paper2lab import mcp_server as M  # noqa: E402
from paper2lab.toolbox import STEPS, TOOLS  # noqa: E402

DEMO = ROOT / "demo"
PDF = str(DEMO / "paper.pdf")
CODE = str(DEMO / "project.zip")
SERVE = str(ROOT / "serve_mcp.py")

CHECKS = 0
FAILURES = []


def check(ok, label):
    global CHECKS
    CHECKS += 1
    # 防呆: 参数传反会让 ok 拿到非空字符串 -> 永远打印"通过"
    assert isinstance(ok, bool), "[%s] ok 不是布尔值: %r" % (label, ok)
    assert isinstance(label, str)
    if not ok:
        FAILURES.append(label)
        print("  *x* " + label)
    else:
        print("  [ok] " + label)


def sec(n):
    print("\n" + "─" * 62)
    print(n)
    print("─" * 62)


def snapshot(folder: Path) -> dict:
    """目录快照（相对路径 -> 大小+修改时间），用于证明"全程只读"。"""
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(folder))] = (st.st_size, st.st_mtime_ns)
    return out


def digest(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ═══════════════════════════════════════ 1. 输入校验

sec("1. check_inputs —— 保守校验，宁可拒绝并说清原因")

ok, why = M.check_inputs("", CODE)
check(ok is False and "pdf_path" in why, "空论文路径 -> 拒绝并指出字段名")
ok, why = M.check_inputs(PDF, "")
check(ok is False and "code_path" in why, "空代码路径 -> 拒绝并指出字段名")
ok, why = M.check_inputs(str(DEMO / "nope.pdf"), CODE)
check(ok is False and "不存在" in why, "论文不存在 -> 拒绝并说明")
ok, why = M.check_inputs(str(DEMO / "results.json"), CODE)
check(ok is False and ".pdf" in why, "论文后缀不对 -> 拒绝并说明只收 pdf")
ok, why = M.check_inputs(PDF, str(DEMO / "nope.zip"))
check(ok is False and "不存在" in why, "代码包不存在 -> 拒绝并说明")
ok, why = M.check_inputs(PDF, str(DEMO / "scan_paper.pdf"))
check(ok is False, "代码路径给了个 .pdf -> 拒绝（不是目录也不是 zip）")
ok, why = M.check_inputs(PDF, CODE)
check(ok is True, "正例: pdf + zip -> 通过")
ok, why = M.check_inputs(PDF, str(DEMO))
check(ok is True, "正例: pdf + 已解压目录 -> 通过（开发机形态）")

# ═══════════════════════════════════════ 2. 摘要裁剪

sec("2. _shrink / _ev_brief —— 返回值必须够读、又不灌爆调用方上下文")

long_s = "甲" * 500
check(len(M._shrink(long_s, max_str=100)) < 120, "超长字符串被截断")
check(M._shrink(42) == 42, "数字原样保留")
check(M._shrink(None) is None, "None 原样保留")
d = {str(i): i for i in range(20)}
shrunk = M._shrink(d, max_items=5)
check(len(shrunk) == 6 and "_more" in shrunk, "字典超项被裁剪并给出总数提示")
lst = M._shrink(list(range(20)), max_items=5)
check(len(lst) == 6 and isinstance(lst[-1], str), "列表超项被裁剪并给出总数提示")
check(M._shrink({"a": {"b": {"c": {"d": 1}}}}, depth=2) is not None, "深层结构不炸")
check(M._ev_brief(None) is None, "空证据 -> None")


class _FakeEv:
    raw_value = "1e-4"
    normalized = 0.0001
    location = "Page 3 · 4. Training Setup"
    parser = "text"


brief = M._ev_brief(_FakeEv())
check(brief["value"] == "1e-4" and "Page 3" in brief["location"],
      "证据压成「值 + 出处」且带解析通道")

# ═══════════════════════════════════════ 3. 工具实现（离线纯函数）

sec("3. 工具实现 —— 不依赖 mcp 包也能跑（纯函数层）")

M.clear_cache()
r = M.tool_list_pipeline_tools()
check(r["ok"] is True, "list_pipeline_tools 返回 ok")
check(len(r["pipeline_tools"]) == len(TOOLS),
      f"暴露 {len(TOOLS)} 件流水线工具（与主程序派生一致，不写死）")
check(r["steps_count"] == len(STEPS),
      f"{len(STEPS)} 个进度节点（与主程序派生一致，不写死）")
check(len(r["exposed_tools"]) == len(M.TOOL_NAMES) and "audit_paper" in r["exposed_tools"],
      f"对外 {len(M.TOOL_NAMES)} 个工具且含 audit_paper（与 TOOL_NAMES 派生一致，不写死）")

snap_before = snapshot(DEMO)
r = M.tool_audit_paper(PDF, CODE)
check(r["ok"] is True, "audit_paper 跑通")
check(r["findings_total"] == 11, "demo 得到 11 条结论")
check(r["counts"].get("inconsistent") == 2, "其中 2 条不一致（与主程序口径一致）")
check(r["cache"] == "miss", "首次调用 cache=miss")
first = {f["param"]: (f["status"], (f["paper"] or {}).get("value"),
                     (f["code"] or {}).get("value")) for f in r["findings"]}
check(first["learning_rate"][0] == "inconsistent"
      and first["learning_rate"][1] == "1e-4"
      and first["learning_rate"][2] == "0.001",
      "学习率 1e-4 vs 0.001 如实报出，两侧值都在")
check(bool(r["findings"][0]["paper"]) and bool(r["findings"][0]["paper"]["location"]),
      "每条结论带论文侧出处")
check(r["findings"][0]["code"]["location"].count(":") == 1,
      "代码侧出处是「文件:行号」")
check("page_texts" not in json.dumps(r, ensure_ascii=False),
      "摘要不含 page_texts 这类大字段")
check(isinstance(r.get("report_preview"), str) and "include_report" in r.get("report_note", ""),
      "默认只给报告预览并提示怎么拿全文")
check(r["elapsed_s"] > 0, "耗时如实记录")

r2 = M.tool_audit_paper(PDF, CODE)
check(r2["cache"] == "hit", "同输入第二次命中进程内缓存")
check(r2["findings_total"] == r["findings_total"], "缓存命中不改变结论")

snap_after = snapshot(DEMO)
check(digest(snap_before) == digest(snap_after),
      "全程只读: demo 目录快照哈希前后一致")

sec("3b. 六个入口的语义各不相同")

r = M.tool_check_parameters(PDF, CODE)
check(r["ok"] is True and r["mode"] == "classic", "check_parameters 走经典链")
p = M.tool_audit_paper(PDF, CODE, mode="classic")
check(r["findings_total"] == p["findings_total"], "check_parameters 与 audit_paper(classic) 同源")

r = M.tool_audit_paper(PDF, CODE, mode="classic", closes_loop=True)
check(r["ok"] is True and r["mode"] == "multagent",
      "开闭环时自动切到多智能体链（与界面同一套语义）")

r = M.tool_compute_verify(PDF, CODE)
check(r["ok"] is True and "relations" in r, "compute_verify 返回关系式")
check("不执行用户代码" in r.get("boundary", ""), "compute_verify 自带边界声明")

r = M.tool_citation_check(PDF, CODE)
check(r["ok"] is True and "coverage" in r, "citation_check 返回覆盖率")
check("字面检索" in r.get("search_mode", ""), "检索方式自述为字面检索（不吹成语义检索）")
check("未核验" in r.get("external_search", ""), "外部检索如实标注未启用/未核验")

r = M.tool_build_repro_plan(PDF, CODE)
check(r["ok"] is True and "steps" in r, "build_repro_plan 返回七步方案")
check("85" in r.get("grade_scale", ""), "评级刻度写清")

r = M.tool_iterate_fix(PDF, CODE)
check(r["ok"] is True and "rounds" in r, "iterate_fix 返回逐轮留痕")
check("不写文件" in r.get("boundary", ""), "迭代回边自带「不写文件」声明")

sec("3c. 失败必须结构化返回，不能抛异常")

r = M.tool_audit_paper("C:/definitely/not/here.pdf", CODE)
check(r["ok"] is False and "不存在" in r["error"], "路径不存在 -> ok=False + 说明")
r = M.tool_audit_paper(PDF, CODE, mode="turbo")
check(r["ok"] is False and "classic" in r["error"], "mode 非法 -> ok=False + 说明")
r = M.tool_audit_paper("", "")
check(r["ok"] is False, "空参数 -> ok=False（不是异常）")
for name in ("tool_compute_verify", "tool_citation_check",
             "tool_build_repro_plan", "tool_iterate_fix"):
    res = getattr(M, name)("C:/definitely/not/here.pdf", CODE)
    check(res["ok"] is False and isinstance(res.get("error"), str),
          f"{name} 坏输入 -> 结构化失败")

# ═══════════════════════════════════════ 4. 与主程序同源

sec("4. 与主程序同源 —— 封装不改结论（消融证据）")

from paper2lab.pipeline import run_audit  # noqa: E402

audit_direct, _report = run_audit(PDF, CODE, force_mock=True)
direct_sig = sorted(
    (f.param_key, f.status.value,
     f.paper.raw_value if f.paper else "", f.code.raw_value if f.code else "")
    for f in audit_direct.findings
)
M.clear_cache()
via_mcp = M.tool_audit_paper(PDF, CODE, findings_limit=100)
mcp_sig = sorted(
    (f["param"], f["status"], (f["paper"] or {}).get("value") or "",
     (f["code"] or {}).get("value") or "")
    for f in via_mcp["findings"]
)
check(len(direct_sig) == len(mcp_sig), "两路结论条数一致")
check(direct_sig == mcp_sig, "两路结论逐条一致（参数/状态/两侧值全同）")

# ═══════════════════════════════════════ 5. FastMCP 绑定

sec(f"5. FastMCP 绑定 —— {len(M.TOOL_NAMES)} 个工具、schema 正确")

server = M.build_server()
tools = asyncio.run(server.list_tools())
names = [t.name for t in tools]
check(len(names) == len(M.TOOL_NAMES), f"注册 {len(M.TOOL_NAMES)} 个工具（与 TOOL_NAMES 一致）")
check(sorted(names) == sorted(M.TOOL_NAMES), "工具名与 TOOL_NAMES 完全一致")
schema = {t.name: t.inputSchema for t in tools}
for t in tools:
    req = (t.inputSchema or {}).get("required", [])
    if t.name in ("list_pipeline_tools",):
        check(req == [], "list_pipeline_tools 无需入参")
    elif t.name == "science_calc":
        # 3D: 只要求「选哪件工具」；inputs / inputs_json 都可选（inputs_json 给只发字符串的客户端）
        check(req == ["tool"], "science_calc 只要求 tool（选工具），输入可选")
    elif t.name == "literature_search":
        # 4A: 只要检索词 —— 这是唯一**不需要论文与代码**的对外工具，
        # 外部智能体拿它当通用文献库用，不该被逼着先传一份 PDF。
        check(req == ["query"], "literature_search 只要求 query（不需要论文/代码）")
    elif t.name == "chem_check":
        # 2E-化学: pdf_path 与 text 二选一 —— 所以**两个都不是 required**,
        # 由函数内部校验"两个都空就拒绝"（测试里另有一条断言守这个）。
        check(req == [], "chem_check 不强制任何参数（pdf_path / text 二选一，函数内校验）")
    else:
        check("pdf_path" in req and "code_path" in req,
              f"{t.name} 的 schema 要求 pdf_path / code_path")
ap_props = schema["audit_paper"]["properties"]
check("closes_loop" in ap_props and "include_report" in ap_props,
      "audit_paper 暴露闭环与报告开关（Schema 未被装饰器打乱）")
check(ap_props["closes_loop"]["type"] == "boolean", "布尔参数类型正确")
check(ap_props["findings_limit"]["type"] == "integer", "整数参数类型正确")

# ─────────────────────────────── 5b. 四站口径（3B 对外接口重排）

sec(f"5b. 四站口径 —— {len(M.TOOL_NAMES)} 个工具挂到四站上，工具名一个都没改")

from paper2lab import science_tools as ST  # noqa: E402
from paper2lab.toolbox import STATIONS  # noqa: E402

check(set(M.TOOL_STATIONS) == set(M.TOOL_NAMES),
      "每个对外工具都有归属站（漏一个就说明有工具在四站视图里消失）")
check(set(M.TOOL_STATIONS.values()) <= (set(STATIONS) | {M.S_FLOW, M.S_META}),
      "归属站只用四站名与两个非站点标记（站名取自 toolbox，不另写一份）")
check(len(STATIONS) == 4, "四站就是四站（提 -> 做 -> 判 -> 改）")
check(sorted(M.TOOL_NAMES) == sorted([
    "audit_paper", "check_parameters", "compute_verify", "science_calc",
    "citation_check", "literature_search", "chem_check", "build_repro_plan",
    "iterate_fix", "list_pipeline_tools"]),
    "工具名单符合契约（3D 加 science_calc、4A 加 literature_search、"
    "2E-化学加 chem_check，老名字一字未动）")

view = M.station_view()
check(list(view["四站"]) == list(STATIONS), "四站按顺序全在视图里，一个不少")
for _st in STATIONS:
    _g = view["四站"][_st]
    check(bool(_g["入"]) and bool(_g["出"]) and isinstance(_g["对外工具"], list),
          f"{_st}: 入 / 出 / 对外工具三项齐备")
    check("**" not in _g["出"] and "**" not in _g["说明"],
          f"{_st}: 对外字段不带 Markdown 排版标记")
check(all(bool(view["四站"][s]["对外工具"]) for s in STATIONS),
      "每一站都至少挂着一个对外工具（不留空站）")

sci = ST.science_menu()
check(view["计算工具总数"] == len(sci),
      f"计算工具数与 science_tools 同源（{len(sci)} 件，不在此处手抄）")
check(sorted(view["四站"][M.S_EXPERIMENT]["计算工具"]) == sorted(c["key"] for c in sci),
      "② 站列出的计算工具 = science_tools 菜单全集")
check(view["四站"][M.S_ASSUME]["计算工具"] == []
      and view["四站"][M.S_VERIFY]["计算工具"] == [],
      "①/③ 站不挂计算工具（算数归 ②，2C 的结构不许破）")

r = M.tool_list_pipeline_tools()
check("stations" in r and "tool_stations" in r, "清单工具带四站视图与归属表")
check(all(bool(t["station"]) for t in r["pipeline_tools"]),
      "每件工序工具都带归属站（与 science_tools.CATALOG 同源）")
check(len(r["tool_stations"]) == len(M.TOOL_NAMES), "清单里回传的归属表覆盖全部对外工具")

desc = {t.name: (t.description or "") for t in tools}
check(desc["check_parameters"].startswith(M.S_ASSUME),
      "check_parameters 描述以「① 假设站」开头")
check(desc["compute_verify"].startswith(M.S_EXPERIMENT),
      "compute_verify 描述以「② 实验站」开头")
check(desc["citation_check"].startswith(M.S_VERIFY)
      and desc["literature_search"].startswith(M.S_VERIFY),
      "③ 验证站的两个「搜」工具描述都以站名开头")
check(desc["build_repro_plan"].startswith(M.S_ITERATE)
      and desc["iterate_fix"].startswith(M.S_ITERATE),
      "④ 迭代站的两个工具描述都以站名开头")
check("全流程" in desc["audit_paper"], "audit_paper 自述为四站贯通")
check(names.index("check_parameters") < names.index("compute_verify")
      < names.index("citation_check") < names.index("build_repro_plan"),
      "工具顺序按四站排（① -> ② -> ③ -> ④）")
check("四站" in (server.instructions or ""), "服务说明按四站口径讲，不是功能平铺")
check("不下结论" in desc["compute_verify"], "② 的工具自述「只出数、不下结论」")

# ─────────────────────────── 5c. 具名科学计算工具对外露出（3D）

sec("5c. science_calc —— 4 件具名科学工具可被外部直接点名调用")

check(desc["science_calc"].startswith(M.S_EXPERIMENT),
      "science_calc 描述以「② 实验站」开头")
check(M.TOOL_STATIONS["science_calc"] == M.S_EXPERIMENT,
      "science_calc 挂在 ② 实验站（算数不归①，也不归③）")
check(sorted(view["四站"][M.S_EXPERIMENT]["对外工具"])
      == sorted(["compute_verify", "science_calc"]),
      "② 站对外有两件：跑固定流程的 compute_verify ＋ 点名调用的 science_calc")

_sci_keys = [c["key"] for c in ST.science_menu()]
check(_sci_keys == [t.key for t in ST.SCIENCE_TOOLS],
      "菜单键与执行体同源（不在此处手抄一份名单）")
check("fp16_precision" in desc["science_calc"]
      and "series_stats" in desc["science_calc"]
      and "linear_fit" in desc["science_calc"]
      and "peak_pick" in desc["science_calc"],
      "描述里列出全部 4 件工具（调用方不用先猜）")

# 四件各真跑一次 —— 只认结果，不看实现
_r1 = M.tool_science_calc("fp16_precision", {"raw": ["1e-8"], "precision": "fp16"})
check(_r1.get("ok") is True, "fp16_precision 真跑成功")
check("0" in str(_r1.get("text")) and "fp16" in str(_r1.get("text")).lower(),
      "fp16_precision 如实报出「落成 0」（这是物理结论，不是套话）")

_r2 = M.tool_science_calc("series_stats", {"values": ["0.90", "0.92", "0.91", "0.93", "0.89"]})
check(_r2.get("ok") is True, "series_stats 真跑成功")
check("标准差" in str(_r2.get("text")), "series_stats 报出样本标准差")

_r3 = M.tool_science_calc(
    "linear_fit", inputs_json='{"xs": [1, 2, 3, 4], "ys": [0.50, 0.52, 0.51, 0.53]}')
check(_r3.get("ok") is True, "linear_fit 经 inputs_json 传入也能跑（只发字符串的客户端可用）")
check("R" in str(_r3.get("text")), "linear_fit 报出 R²")

_r4 = M.tool_science_calc("peak_pick", {"xs": [4000, 3500, 3000, 2500],
                                       "ys": [0.2, 0.8, 0.3, 0.1]})
check(_r4.get("ok") is True, "peak_pick 真跑成功")

for _r, _name in ((_r1, "fp16_precision"), (_r2, "series_stats"),
                  (_r3, "linear_fit"), (_r4, "peak_pick")):
    check("verdict" not in _r and "compare" not in _r,
          f"{_name}: 返回里没有 verdict / compare（② 只出数，判定归 ③）")
    check(_r.get("station") == M.S_EXPERIMENT and bool(_r.get("boundary")),
          f"{_name}: 标了归属站与边界说明")

# 清单外不存在 / 坏输入如实报错 —— 两条都**不许抛异常**
_r5 = M.tool_science_calc("drop_table", {})
check(_r5.get("ok") is False, "清单外的工具被拒（菜单外 = 本次不存在）")
check(len(_r5.get("available") or []) == len(_sci_keys),
      "被拒时回传可用菜单，调用方知道该挑哪件")
_r6 = M.tool_science_calc("series_stats", inputs_json="{not json")
check(_r6.get("ok") is False and "JSON" in str(_r6.get("error")),
      "inputs_json 不合法时如实报错，不抛异常")
_r6b = M.tool_science_calc("series_stats", "1,2,3")
check(_r6b.get("ok") is False and "inputs" in str(_r6b.get("error")),
      "inputs 传了非对象时如实拒（不静默当成空输入接着跑）")
_r7 = M.tool_science_calc("series_stats", {"values": ["1"]})
check(_r7.get("ok") is False or "标准差" not in str(_r7.get("text")),
      "取值不足 2 个时不给标准差（凑不齐就如实说，不硬算）")

# ═══════════════════════════════════════ 6. stdio 真机 + 通道纯净性

sec("6. stdio 真机 —— 逐行都必须是合法 JSON-RPC（脏行会断连）")


def probe_stdio():
    """手动发请求、逐行校验。返回 (行清单, stderr 文本, 原始字节清单)。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen(
        [sys.executable, "-u", SERVE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ROOT), env=env)
    err_buf = []

    def _drain():
        for raw in proc.stderr:
            err_buf.append(raw)
    threading.Thread(target=_drain, daemon=True).start()

    def send(obj):
        proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    raws = []

    def recv():
        raw = proc.stdout.readline()
        raws.append(raw)
        return raw

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "verify", "version": "1"}}})
        recv()
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        recv()
        # 关键: 第 3、4 次调用才触发 PDF 解析 —— 脏行历来出现在这里
        send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
              "params": {"name": "list_pipeline_tools", "arguments": {}}})
        recv()
        send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
              "params": {"name": "check_parameters",
                         "arguments": {"pdf_path": PDF, "code_path": CODE}}})
        recv()
        # 第 5 次: 3D 新增的 science_calc —— 证「外部真的能量名调用一件科学工具」，
        # 而不只是「它在清单里」。这一步不碰 PDF，很轻。
        send({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
              "params": {"name": "science_calc",
                         "arguments": {"tool": "series_stats",
                                       "inputs": {"values": ["0.90", "0.92", "0.91"]}}}})
        recv()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    return raws, b"".join(err_buf).decode("utf-8", "replace")


raws, err_text = probe_stdio()
check(len(raws) == 5, f"收到 5 行响应（实得 {len(raws)}）")

bad = []
for i, raw in enumerate(raws, 1):
    try:
        obj = json.loads(raw.decode("utf-8"))
        good = "jsonrpc" in obj and ("result" in obj or "error" in obj)
    except Exception as exc:
        good = False
        obj = str(exc)
    if not good:
        bad.append((i, raw.decode("utf-8", "replace")[:80]))
check(not bad, f"5 行全部是合法 JSON-RPC（脏行: {bad}）")

first_obj = json.loads(raws[0].decode("utf-8"))
check(first_obj["result"]["serverInfo"]["name"] == "paper2lab-audit",
      "initialize 返回正确的服务名")
tools_obj = json.loads(raws[1].decode("utf-8"))
remote_names = [t["name"] for t in tools_obj["result"]["tools"]]
check(sorted(remote_names) == sorted(M.TOOL_NAMES),
      f"tools/list 远端看到 {len(M.TOOL_NAMES)} 个工具（含 3D 的 science_calc、4A 的 literature_search）")
call3 = json.loads(raws[2].decode("utf-8"))
check(call3["result"]["isError"] is False, "第 3 次调用 isError=False")
payload = json.loads(call3["result"]["content"][0]["text"])
check(len(payload["pipeline_tools"]) == len(TOOLS),
      f"远端拿到的流水线清单是 {len(TOOLS)} 件（与本地派生一致）")
call4 = json.loads(raws[3].decode("utf-8"))
res4 = json.loads(call4["result"]["content"][0]["text"])
check(res4["ok"] is True and res4["findings_total"] == 11,
      "远端跑真实审计也得 11 条结论")
call5 = json.loads(raws[4].decode("utf-8"))
check(call5["result"]["isError"] is False, "第 5 次调用 isError=False")
res5 = json.loads(call5["result"]["content"][0]["text"])
check(res5.get("ok") is True and res5.get("tool") == "series_stats",
      "远端点名调用 science_calc 真跑成功（外部够得着单件科学工具）")
check("标准差" in str(res5.get("text")) and "verdict" not in res5,
      "远端拿到的只有数、没有判定（② 只出数这条在协议层也成立）")
check("Processing request" in err_text or "transport=stdio" in err_text,
      "服务端日志走 stderr（stdout 只留给 JSON-RPC）")

# ═══════════════════════════════════════ 7. Streamable HTTP 真机

sec("7. Streamable HTTP 真机 —— 端砚 SCP 的接入形态")

PORT = 8791
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

http_proc = subprocess.Popen(
    [sys.executable, "-u", SERVE, "--http", "--port", str(PORT)],
    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    cwd=str(ROOT), env=dict(os.environ))
http_err = []


def _drain_http():
    for raw in http_proc.stderr:
        http_err.append(raw)


threading.Thread(target=_drain_http, daemon=True).start()


async def call_http(tool, arguments):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    url = f"http://127.0.0.1:{PORT}/mcp"
    async with streamablehttp_client(url, timeout=10) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tl = await s.list_tools()
            res = await s.call_tool(tool, arguments)
            return [t.name for t in tl.tools], res


http_names, http_res = None, None
last_err = ""
for attempt in range(8):
    try:
        http_names, http_res = asyncio.run(
            call_http("check_parameters", {"pdf_path": PDF, "code_path": CODE}))
        break
    except Exception as exc:
        last_err = f"{type(exc).__name__}: {exc}"
        time.sleep(1.0)

if http_names is None:
    check(False, f"HTTP 通道可连接（最后错误: {last_err[:120]}）")
else:
    check(sorted(http_names) == sorted(M.TOOL_NAMES),
          f"HTTP 通道也看到 {len(M.TOOL_NAMES)} 个工具")
    body = json.loads(http_res.content[0].text)
    check(http_res.isError is False and body["ok"] is True,
          "HTTP 通道调用返回 ok=True")
    check(body["findings_total"] == 11, "HTTP 通道结论与 stdio 一致（11 条）")

try:
    http_proc.kill()
    http_proc.wait(timeout=5)
except Exception:
    pass

# ═══════════════════════════════════════ 8. 契约自述

sec("8. 契约自述与版本口径")


check(M.SERVER_NAME == "paper2lab-audit", "服务名固定")
check(bool(M.SERVER_VERSION), "服务版本非空")
check(len(M.TOOL_NAMES) == len(set(M.TOOL_NAMES)), "工具名无重复")
check(len(M.STATUS_LABELS) == 7, "七种状态都有中文标签")
check(all(k in M.STATUS_LABELS for k in
          ("consistent", "inconsistent", "missing_in_code", "missing_in_paper",
           "not_fixed", "unverifiable", "internal_inconsistent")),
      "状态标签覆盖 models.Status 全部取值")
check(M.CODE_SUFFIXES == (".zip",), "代码包只收 zip（目录走另一种分支）")
check(callable(M.guarded) and callable(M.stdout_as_stderr),
      "通道保护函数存在（这是本轮踩坑后加的防线）")

print("\n" + "═" * 62)
if FAILURES:
    print(f"失败 {len(FAILURES)} / {CHECKS}")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print(f"全部通过(verify_mcp.py): {CHECKS} 项")
