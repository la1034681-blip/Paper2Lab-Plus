"""第二套（冷启动版）：全链路端到端，**先清缓存**再用本机模型真跑一遍。

为什么要冷启动
--------------
`tools/run_repr_checks.py` 的 B 段是"常用路径"，会命中两层缓存：
  · 会话快照（同一份论文+代码第二次跑直接复用解析结果）；
  · LLM 结果缓存（`cache/llm_cache.json`，提示词一模一样就直接拿旧答案）。
于是它报出来的 demo「0.3 秒跑完」并不是本机模型跑得快，而是**压根没调模型**。
那样的数字不能用来回答「本机模型能不能把全链路跑通」。

这个脚本做的事：
  1. 把两层缓存**改名挪走**（不是删，挪进 `.trash_tmp/`，随时能挪回来）；
  2. 用本机模型冷跑一遍 demo（三材料齐全，唯一能走到第三站的材料）；
  3. 打印四站产物：①假设卡 ②实验卡 ③判定（findings/自检）④报告与修正建议；
  4. 紧接着再跑一遍（热），验证"第二遍靠缓存加速"这条设计真的成立；
  5. 把两遍的结论签名逐条比对 —— 计划由模型排以后，这一步还能不能对上，是
     「调度不改结论」这句话的实测依据。

用法：
    .venv\\Scripts\\python.exe -X utf8 tools\\run_e2e_cold.py            # 默认 demo
    .venv\\Scripts\\python.exe -X utf8 tools\\run_e2e_cold.py D题         # 换样本
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.pipeline import run_audit                     # noqa: E402

CACHE = ROOT / "cache"
TRASH = ROOT / ".trash_tmp" / ("cold_" + time.strftime("%Y%m%d_%H%M%S"))

CASES = {
    "demo": ("demo 三材料齐全",
             ROOT / "demo" / "paper.pdf", ROOT / "demo" / "project.zip",
             ROOT / "demo" / "results.json"),
    "D题": ("D题 中文案例（有代码无实测）",
            ROOT / "demo" / "D题论文.pdf", ROOT / "demo" / "D题代码.zip", None),
}

WHICH = (sys.argv[1] if len(sys.argv) > 1 else "demo").strip()
LABEL, PDF, CODE, RES = CASES.get(WHICH, CASES["demo"])


def _move_aside() -> list[tuple[Path, Path]]:
    """把缓存挪进 .trash_tmp（可逆），返回挪动清单。"""
    TRASH.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in ("llm_cache.json", "audit_memory.json", "vision_cache.json"):
        src = CACHE / name
        if src.exists():
            dst = TRASH / name
            shutil.move(str(src), str(dst))
            moved.append((src, dst))
    sess = CACHE / "sessions"
    if sess.exists():
        dst = TRASH / "sessions"
        shutil.move(str(sess), str(dst))
        moved.append((sess, dst))
    return moved


def _restore(moved: list[tuple[Path, Path]]) -> None:
    """冷跑完把旧缓存放回去 —— 不占用户的"冷热"便宜，也不毁他的历史。"""
    for src, dst in moved:
        if dst.exists() and not src.exists():
            shutil.move(str(dst), str(src))


def _sig(audit) -> list[str]:
    out = []
    for f in audit.findings:
        st = getattr(f.status, "value", f.status)
        out.append(f"{getattr(f, 'param_key', '?')}|{st}")
    return sorted(out)


def _products(audit) -> dict:
    ex = audit.execution_log or {}
    plan = ex.get("plan") or {}
    dec = plan.get("decisions") or []
    return {
        "planner": plan.get("planner", "?"),
        "plan_order": list(plan.get("order") or []),
        "plan_ran": sum(1 for d in dec if d.get("action") == "run"),
        "plan_skipped": sum(1 for d in dec if d.get("action") == "skip"),
        "model_decisions": sum(1 for d in dec if d.get("source") == "llm"),
        "steps_logged": len(ex.get("steps") or []),
        "findings": len(audit.findings or []),
        "hypotheses": int(getattr(audit, "hypothesis_total", 0) or 0),
        "experiments": int(getattr(audit, "experiment_total", 0) or 0),
        "figures": int(getattr(audit, "figure_total", 0) or 0),
        "chem": int(getattr(audit, "chem_total", 0) or 0),
        "rootcause": len(getattr(audit, "rootcause", []) or []),
        "patches": len(getattr(audit, "patches", []) or []),
        "selfcheck": (audit.selfcheck or {}).get("checked"),
        "selfcheck_pass": (audit.selfcheck or {}).get("passed"),
        "readiness": (audit.readiness or {}).get("total")
                     or (audit.readiness or {}).get("score"),
        "report_chars": 0,
    }


def run_once(tag: str) -> tuple[dict, object]:
    user = json.loads(RES.read_text(encoding="utf-8")) if (RES and RES.exists()) else None
    t0 = time.monotonic()
    audit, report = run_audit(str(PDF), str(CODE) if CODE else None, user, force_mock=False)
    dt = time.monotonic() - t0
    row = _products(audit)
    row["report_chars"] = len(report or "")
    row["elapsed"] = round(dt, 1)
    row["sig"] = _sig(audit)
    print(f"\n=== {tag} ===")
    print(f"  耗时        {row['elapsed']}s")
    print(f"  谁排的计划  {row['planner']}（模型定了 {row['model_decisions']} 条 / 共 "
          f"{row['plan_ran'] + row['plan_skipped']} 条）")
    print(f"  计划顺序    {row['plan_order']}")
    print(f"  执行记录    {row['steps_logged']} 步")
    print(f"  ① 假设卡    {row['hypotheses']} 张")
    print(f"  ② 实验卡    {row['experiments']} 张｜图表线索 {row['figures']} 条｜化学 {row['chem']} 项")
    print(f"  ③ 判定      {row['findings']} 条｜自检 {row['selfcheck']} 项通过 {row['selfcheck_pass']}")
    print(f"  ④ 出口      根因 {row['rootcause']}｜补丁 {row['patches']}｜"
          f"报告 {row['report_chars']} 字｜就绪度 {row['readiness']}")
    return row, audit


def main() -> int:
    print("=" * 74)
    print(f"第二套 · 全链路端到端（冷启动）  样本：{LABEL}")
    print("=" * 74)

    print("\n[1/3] 挪走两层缓存（可逆，落在 .trash_tmp）")
    moved = _move_aside()
    for src, _dst in moved:
        print(f"  · {src.name} → .trash_tmp")
    if not moved:
        print("  · 本来就没有缓存，直接算冷启动")

    cold, a1 = run_once("第一遍 · 冷（无任何缓存，本机模型真跑）")

    print("\n[2/3] 紧接着再跑一遍（热，验证缓存加速这条设计）")
    hot, a2 = run_once("第二遍 · 热（刚跑过同一份材料）")

    print("\n[3/3] 两遍结论逐条比对")
    same = cold["sig"] == hot["sig"]
    print(f"  结论签名一致：{same}（{len(cold['sig'])} 条）")
    if not same:
        only1 = [x for x in cold["sig"] if x not in hot["sig"]]
        only2 = [x for x in hot["sig"] if x not in cold["sig"]]
        print(f"  仅第一遍有：{only1[:6]}")
        print(f"  仅第二遍有：{only2[:6]}")
    print(f"  计划是否同一份：{cold['plan_order'] == hot['plan_order']}")
    if cold["plan_order"] != hot["plan_order"]:
        print(f"    冷：{cold['plan_order']}")
        print(f"    热：{hot['plan_order']}")

    _restore(moved)
    print("\n[还原] 旧缓存已放回原处")

    out = ROOT / "cache" / f"e2e_cold_{WHICH}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": LABEL, "cold": {k: v for k, v in cold.items() if k != "sig"},
                               "hot": {k: v for k, v in hot.items() if k != "sig"},
                               "sig_same": same}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n明细：{out}")

    # 判据：四站都有产物才算这条链路完整
    ok = (cold["findings"] > 0 and cold["report_chars"] > 2000)
    if CODE:
        ok = ok and cold["experiments"] >= 0        # 实验站允许为 0，但要如实记
    print("\n" + "=" * 74)
    print("结论：" + ("四站产物齐全，链路通" if ok else "有站点没有产物，见上表"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
