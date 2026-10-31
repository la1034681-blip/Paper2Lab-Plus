"""代表性样本测试 —— 不全量跑，用最少的样本覆盖最多的场景。

为什么要有这个脚本（2026-10-30 用户要求）：
    换成本机 Intern-S1-mini 后，模型速度从 100+ token/s 掉到约 10 token/s，
    全量 56 项（8 套论文 × 多次审计 × 批量读图）要跑一个多小时，而且**批量读图
    项必然超时**（单张图约 30s，一套论文 8 个区域就要 4 分钟）——
    那不是"程序坏了"，是本地 8B 量化模型的吞吐上限。
    所以改成：**检查项精选 + 样本精选**，20 分钟内拿到"核心链路还通不通"的结论。

样本选取标准（三条择一即入选，不重复占位）：
    1. **材料形态**：三材料齐全（论文+代码+实测结果）必须有 1 套 —— 只有它能走到
       第三站"拿实测值做判定"，否则第三站永远是"未核验"；
    2. **语言 / 版式**：英文真实论文 1 套（双栏、表格、公式）+ 中文案例 1 套；
    3. **侧枝**：无实测结果、仅论文无代码 —— 用最小代价点到即止。
    按此选：demo（三材料·中文）、01_DDPM（英文真实·完整代码）、D题（中文·有代码无实测），
    再加"demo 只给论文"验证无代码兜底。**不选** DETR / NeRF / PointNet / ViT：
    能力与 DDPM 高度重叠，而读图区域更多、纯属拖时间。

用法：
    .venv\\Scripts\\python.exe -X utf8 tools\\run_repr_checks.py both
    .venv\\Scripts\\python.exe -X utf8 tools\\run_repr_checks.py a     # 只跑独立功能
    .venv\\Scripts\\python.exe -X utf8 tools\\run_repr_checks.py b     # 只跑全链路
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from run_all_checks import CHECKS, run_one          # noqa: E402

OUT_DIR = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
SET = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")

# ---- 场景一：精选检查项 ----------------------------------------------------
# 选的是"单点能力、自带小样本、不依赖批量读图"的项。
# 刻意**不选**：图像通道三项（5/6/13，批量读图必超时）、五套审计(1)、
#   计划确定性(15，8 套×2)、就绪度(17，8 套)、执行记录(14，5 次审计)、
#   网页端全部（本次不要求）—— 它们在本地模型下的耗时由"读图张数 × 30s"决定。
A_IDS = [3, 24, 25, 27, 29, 30, 34, 35, 36, 37, 38,
         41, 42, 43, 46, 47, 48, 51, 52, 55]

# ---- 场景二：代表样本 ------------------------------------------------------
CASES = [
    # (标签, 论文, 代码, 实测结果)
    ("demo 三材料齐全", str(ROOT / "demo" / "paper.pdf"),
     str(ROOT / "demo" / "project.zip"), str(ROOT / "demo" / "results.json")),
    ("01_DDPM 英文真实论文", str(SET / "01_DDPM" / "paper.pdf"),
     str(SET / "01_DDPM" / "code.zip"), None),
    ("D题 中文案例", str(ROOT / "demo" / "D题论文.pdf"),
     str(ROOT / "demo" / "D题代码.zip"), None),
    ("demo 仅论文(无代码)", str(ROOT / "demo" / "paper.pdf"), None, None),
]


def _sig(audit) -> list[str]:
    """结论签名 —— 只取"哪个参数、判成什么"，用于两次跑一致性对比。"""
    out = []
    for f in audit.findings:
        st = getattr(f.status, "value", f.status)
        out.append(f"{getattr(f, 'param', '?')}|{st}")
    return sorted(out)


def _score(audit) -> str:
    r = audit.readiness or {}
    for k in ("total", "score", "overall"):
        if isinstance(r.get(k), (int, float)):
            return f"{r[k]}"
    return "-"


def stage_a() -> list[dict]:
    print("\n===== 阶段 A · 独立功能（精选 %d 项）=====" % len(A_IDS), flush=True)
    results = []
    for n, i in enumerate(A_IDS, 1):
        name, args, to = CHECKS[i - 1]
        print(f"[{n}/{len(A_IDS)}] #{i} {name} …", flush=True)
        r = run_one(name, args, to)
        r["id"] = i
        results.append(r)
        st = "TIMEOUT" if r["timeout"] else ("OK" if r["code"] == 0 else f"exit={r['code']}")
        print(f"    -> {st}  {r['elapsed']:.1f}s", flush=True)
        if r["timeout"] or r["code"] != 0:
            print("       " + "\n       ".join(r["tail"].splitlines()[-6:]), flush=True)
    bad = [r for r in results if r["timeout"] or r["code"] != 0]
    print(f"--- 阶段 A：{len(results)} 项，通过 {len(results)-len(bad)}，失败 {len(bad)} ---",
          flush=True)
    return results


def stage_b() -> list[dict]:
    print("\n===== 阶段 B · 全链路（%d 套代表样本）=====" % len(CASES), flush=True)
    from paper2lab.pipeline import run_audit

    rows = []
    for n, (label, pdf, code, res) in enumerate(CASES, 1):
        print(f"[{n}/{len(CASES)}] {label} …", flush=True)
        user = None
        if res and Path(res).exists():
            user = json.loads(Path(res).read_text(encoding="utf-8"))
        t0 = time.monotonic()
        row = {"label": label, "pdf": pdf, "code": code, "results": bool(user)}
        try:
            audit, report = run_audit(pdf, code, user, force_mock=False)
        except Exception as ex:                    # noqa: BLE001
            row.update({"ok": False, "error": f"{type(ex).__name__}: {ex}",
                        "elapsed": round(time.monotonic() - t0, 1)})
            print(f"    -> 抛错 {row['error']}  {row['elapsed']}s", flush=True)
            rows.append(row)
            continue
        row.update({
            "ok": audit is not None,
            "elapsed": round(time.monotonic() - t0, 1),
            "findings": len(audit.findings) if audit else 0,
            "readiness": _score(audit) if audit else "-",
            "selfcheck": (audit.selfcheck or {}).get("checked") if audit else None,
            "selfcheck_pass": (audit.selfcheck or {}).get("passed") if audit else None,
            "hypotheses": getattr(audit, "hypothesis_total", 0) if audit else 0,
            "rootcause": len(getattr(audit, "rootcause", []) or []) if audit else 0,
            "patches": len(getattr(audit, "patches", []) or []) if audit else 0,
            "report_chars": len(report or ""),
            "sig": _sig(audit) if audit else [],
        })
        rows.append(row)
        print(f"    -> 结论 {row['findings']} 条｜就绪度 {row['readiness']}｜"
              f"假设 {row['hypotheses']} 张｜根因 {row['rootcause']}｜补丁 {row['patches']}｜"
              f"报告 {row['report_chars']} 字｜{row['elapsed']}s", flush=True)

    # 一致性：demo 同一材料再跑一次，结论签名必须逐条相同
    label, pdf, code, res = CASES[0]
    print(f"[一致性] {label} 复跑一次 …", flush=True)
    user = None
    if res and Path(res).exists():
        user = json.loads(Path(res).read_text(encoding="utf-8"))
    t0 = time.monotonic()
    try:
        audit2, report2 = run_audit(pdf, code, user, force_mock=False)
        row2 = {"label": label + "（复跑）", "ok": audit2 is not None,
                "elapsed": round(time.monotonic() - t0, 1),
                "findings": len(audit2.findings),
                "readiness": _score(audit2), "sig": _sig(audit2),
                "report_chars": len(report2 or "")}
        same = row2["sig"] == rows[0]["sig"]
        row2["same_as_first"] = same
        print(f"    -> 结论 {row2['findings']} 条｜与首跑逐条一致 = {same}｜{row2['elapsed']}s",
              flush=True)
        rows.append(row2)
    except Exception as ex:                        # noqa: BLE001
        rows.append({"label": label + "（复跑）", "ok": False,
                     "error": f"{type(ex).__name__}: {ex}"})
        print(f"    -> 抛错 {type(ex).__name__}", flush=True)
    return rows


def main() -> int:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "both").strip().lower()
    t_all = time.monotonic()
    out = {"mode": mode, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if mode in ("a", "both"):
        out["stage_a"] = stage_a()
    if mode in ("b", "both"):
        out["stage_b"] = stage_b()
    out["total_seconds"] = round(time.monotonic() - t_all, 1)
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    stamp = time.strftime("%Y%m%d_%H%M")
    path = OUT_DIR / f"repr_checks_{mode}_{stamp}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    bad = [r for r in out.get("stage_a", []) if r["timeout"] or r["code"] != 0]
    print("\n=================== 汇总 ===================")
    if "stage_a" in out:
        print(f"阶段 A 独立功能：{len(out['stage_a'])} 项，失败 {len(bad)}")
    if "stage_b" in out:
        oks = [r for r in out["stage_b"] if r.get("ok")]
        print(f"阶段 B 全链路：{len(out['stage_b'])} 次运行，成功 {len(oks)}")
    print(f"总耗时 {out['total_seconds']}s")
    print(f"明细：{path}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
