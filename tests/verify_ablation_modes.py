"""计划确定性：**同一份材料跑两次，计划与结论必须一模一样**（八套样本逐项比对）。

2026-10-25 用户口径：删除「经典模式」（不看材料、固定全序全跑），只留"先排计划再执行"。
原先这一项是"经典 vs 计划"的对照，对照组没了，于是**改成守更要紧的一条**：

  审计平台最怕的不是"跳过了某个工具"，而是**同一份材料每次跑出来的结论不一样**。
  所以这里守两件事：
    ① **计划可复现** —— 两次跑的执行集合、顺序、跳过理由逐字一致；
    ② **结论可复现** —— 结论签名 / 就绪度核心字段 / 审计可信度一致。

  再加一条"跳过必须有话说"：每个被跳过的工具都得给出理由（不能空着跳过）。
  「调度不改变结论」的另一个证据在 verify_llm_planner.py：模型把 13 件工具全跳了，
  结论照样产出 —— 那是"人（模型）乱来也不改结论"，这里是"机器自己跑要稳"。

用法: .venv\\Scripts\\python.exe verify_ablation_modes.py
"""
from __future__ import annotations
# --- 目录整理（2026-09-30）后补的引导：脚本已下沉一层，需把项目根加回 sys.path ---
import sys as _sys
import pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))

import json
from pathlib import Path

from paper2lab.envfile import load_env
from paper2lab.pipeline import run_audit

load_env()
ROOT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
HERE = Path(__file__).resolve().parent.parent

CASES = [
    ("01_DDPM", ROOT / "01_DDPM/paper.pdf", ROOT / "01_DDPM/code.zip", None, True),
    ("02_DETR", ROOT / "02_DETR/paper.pdf", ROOT / "02_DETR/code.zip", None, True),
    ("03_NeRF", ROOT / "03_NeRF/paper.pdf", ROOT / "03_NeRF/code.zip", None, True),
    ("04_PointNet", ROOT / "04_PointNet/paper.pdf", ROOT / "04_PointNet/code.zip", None, True),
    ("05_ViT", ROOT / "05_ViT/paper.pdf", ROOT / "05_ViT/code.zip",
     ROOT / "05_ViT/results.json", True),
    ("demo", HERE / "demo/paper.pdf", HERE / "demo/project.zip", None, False),
    ("D题", HERE / "demo/D题论文.pdf", HERE / "demo/D题代码.zip", None, False),
    ("E题", Path(r"C:\Users\lijiahui\Desktop\E题论文.pdf"),
     Path(r"C:\Users\lijiahui\Desktop\E题作答.zip"), None, False),
]


def sig(a) -> list:
    """结论签名（**排序后**，可安全比较）。

    ⚠️ 必须排序（2026-09-29 修）：`llm_analyze` 走**分片并发**（4 路），findings 的返回
    顺序受线程调度影响 —— 不排序的话，"内容完全一样、只是顺序不同"也会被判成不一致，
    制造**假失败**（实测 03_NeRF 就这样失败过一次，而重跑立刻一致）。
    findings 是集合语义，顺序本身没有含义，所以排序才是正确口径。
    取值统一转字符串：避免 None 与 str 在排序时比较报错。
    """
    return sorted(
        (str(f.param_key), f.status.value, f.risk.value,
         str(f.paper.raw_value) if f.paper else "",
         str(f.code.raw_value) if f.code else "")
        for f in a.findings)


def rd_core(a) -> dict:
    """就绪度核心字段: 排除 llm_* / execution_log 这类运行态。"""
    return {k: v for k, v in (a.readiness or {}).items()
            if not k.startswith("llm_") and k != "execution_log"}


def plan_of(a) -> list:
    """执行计划序列：刻意**不排序** —— 顺序本身也是计划的一部分。

    每项 = (工具键, 状态, 理由)。两次跑必须逐项一致，否则就是"计划不可复现"。
    """
    steps = ((a.execution_log or {}).get("steps") or [])
    return [(s.get("key"), s.get("status"), s.get("reason") or "") for s in steps]


out: list[str] = []
out.append("# 计划确定性报告：同一材料跑两次，计划与结论必须一致")
out.append("")
out.append("用途：审计平台的第一条底线 —— **同一份材料，每次跑出来的东西要一样**。")
out.append("")
out.append("| 样本 | 结论数 | 结论签名 | 计划序列 | 跳过理由齐全 | 就绪度 | 可信度 | 判定 |")
out.append("|---|---|---|---|---|---|---|---|")

bad = 0
for name, pdf, code, res, real in CASES:
    ur = json.loads(Path(res).read_text(encoding="utf-8")) if res else None
    a_1, _ = run_audit(str(pdf), str(code), ur, force_mock=not real)
    a_2, _ = run_audit(str(pdf), str(code), ur, force_mock=not real)

    ok_sig = sig(a_1) == sig(a_2)
    ok_plan = plan_of(a_1) == plan_of(a_2)
    skipped = [s for s in plan_of(a_1) if s[1] == "skipped"]
    ok_reason = all(bool((s[2] or "").strip()) for s in skipped)
    ok_rd = rd_core(a_1) == rd_core(a_2)
    ok_conf = ((a_1.coverage or {}).get("confidence_score")
               == (a_2.coverage or {}).get("confidence_score"))
    ok = ok_sig and ok_plan and ok_reason and ok_rd and ok_conf
    if not ok:
        bad += 1

    out.append(
        f"| {name} | {len(a_1.findings)} | "
        f"{'一致 ✅' if ok_sig else '**不同 ⚠**'} | "
        f"{'一致 ✅' if ok_plan else '**不同 ⚠**'} | "
        f"{'齐全 ✅' if ok_reason else '**空理由 ⚠**'} | "
        f"{a_1.readiness.get('total')} | "
        f"{(a_1.coverage or {}).get('confidence_score')} | "
        f"{'通过' if ok else '**不通过 ⚠**'} |")

out.append("")
out.append(f"**结论：{len(CASES) - bad}/{len(CASES)} 套样本两次跑完全一致**"
           + ("（计划与结论都可复现）" if bad == 0 else "——有样本不一致，需排查！"))
out.append("")
if bad:
    out.append("## 不一致明细")
    out.append("")
    for name, pdf, code, res, real in CASES:
        ur = json.loads(Path(res).read_text(encoding="utf-8")) if res else None
        a_1, _ = run_audit(str(pdf), str(code), ur, force_mock=not real)
        a_2, _ = run_audit(str(pdf), str(code), ur, force_mock=not real)
        p1, p2 = plan_of(a_1), plan_of(a_2)
        if p1 != p2:
            out.append(f"### {name} · 计划序列不同")
            out.append("")
            for x, y in zip(p1, p2):
                if x != y:
                    out.append(f"- 第一次 `{x}` vs 第二次 `{y}`")
            out.append("")
        only_1 = [x for x in sig(a_1) if x not in sig(a_2)]
        only_2 = [x for x in sig(a_2) if x not in sig(a_1)]
        if only_1 or only_2:
            out.append(f"### {name} · 结论签名不同")
            out.append("")
            out.append(f"- 仅第一次有：{only_1}")
            out.append(f"- 仅第二次有：{only_2}")
            out.append("")
        # 就绪度**字段级** diff —— 只报"85 vs 87"没法定位，得指到具体字段上
        r1, r2 = rd_core(a_1), rd_core(a_2)
        if r1 != r2:
            out.append(f"### {name} · 就绪度字段不同")
            out.append("")
            for k in sorted(set(r1) | set(r2)):
                if r1.get(k) != r2.get(k):
                    out.append(f"- `{k}`：第一次={r1.get(k)} vs 第二次={r2.get(k)}")
            out.append("")

OUT = ROOT / "ablation_modes_report.md"
OUT.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
print(f"\n报告: {OUT}")
if bad:
    raise SystemExit(1)
