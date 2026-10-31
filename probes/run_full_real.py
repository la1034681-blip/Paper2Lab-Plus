"""全开 + 真实模型跑一次，并把结果落成 JSON —— 供界面方案用真实数据填充。

跑的东西（全部打开，一个不留）：
  · 模型排计划（llm_planner） / 模型提名补丁（llm_patch）
  · 模型先判程序后验（judge_llm，试点 ① 假设 ③ 配对）
  · 实验站让模型挑工具（experiment_llm）
  · 多智能体对撞，对撞区角色也用模型（role_llm）
  · 复现闭环 + 自主决定要不要回头（loop / auto_loop）
  · 多模态读图（复用 LLM 的 deepseek-chat 直接看图）

用法：
  .venv/Scripts/python.exe probes/run_full_real.py            # 用缓存（快）
  .venv/Scripts/python.exe probes/run_full_real.py --fresh    # 清模型/读图缓存后真跑

产出（桌面/Paper2Lab测试集/）：
  full_real.json         结构化结果（含执行记录、假设、实验、对撞、闭环）
  full_real_report.md    审计报告正文
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from paper2lab.envfile import get, load_env          # noqa: E402

load_env(force=True)

from paper2lab.graph import run_audit_multagent      # noqa: E402

OUT = Path.home() / "Desktop" / "Paper2Lab测试集"
OUT.mkdir(parents=True, exist_ok=True)

# 样本：默认 demo（受控样例）；可用 --case D / --case E 换带图的真实论文
_CASES = {
    "demo": (BASE / "demo" / "paper.pdf", BASE / "demo" / "project.zip",
             BASE / "demo" / "results.json"),
    "D": (BASE / "demo" / "D题论文.pdf", BASE / "demo" / "D题代码.zip", None),
    "E": (BASE / "demo" / "E题论文.pdf", BASE / "demo" / "E题代码.zip", None),
}
_case = "demo"
for _i, _a in enumerate(sys.argv):
    if _a == "--case" and _i + 1 < len(sys.argv):
        _case = sys.argv[_i + 1]
PDF, CODE, RES = _CASES.get(_case, _CASES["demo"])

CACHES = ("llm_cache.json", "vision_cache.json")


def clear_caches() -> list[str]:
    """把模型/读图缓存挪进 .trash_tmp —— 不删，可回滚。返回挪走的文件名。"""
    trash = BASE / ".trash_tmp" / "cache_backup_20261026"
    trash.mkdir(parents=True, exist_ok=True)
    moved = []
    for name in CACHES:
        src = BASE / "cache" / name
        if src.exists():
            shutil.move(str(src), str(trash / name))
            moved.append(name)
    return moved


def plain(obj):
    """把 dataclass / list 转成能 json 的纯结构（未知类型退成 str）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [plain(x) for x in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if hasattr(obj, "to_dict"):
        try:
            return plain(obj.to_dict())
        except Exception:      # noqa: BLE001
            pass
    if hasattr(obj, "value"):          # 枚举
        return obj.value
    return str(obj)


def main() -> int:
    fresh = "--fresh" in sys.argv
    if not PDF.exists():
        print(f"找不到样本: {PDF}")
        return 1

    if fresh:
        moved = clear_caches()
        print(f"已清缓存(移入 .trash_tmp): {moved or '无缓存文件'}", flush=True)

    user_results = (json.loads(RES.read_text(encoding="utf-8"))
                    if RES is not None and RES.exists() else None)

    t0 = time.monotonic()
    steps: list[dict] = []

    def on_progress(i: int, name: str) -> None:
        steps.append({"i": i, "name": name, "t": round(time.monotonic() - t0, 2)})
        print(f"  [{i:02d}] {name}  ({steps[-1]['t']}s)", flush=True)

    print("开跑：全功能 + 真实模型（deepseek-chat，含多模态读图）", flush=True)
    audit, report = run_audit_multagent(
        str(PDF), str(CODE), user_results,
        api_key=get("LLM_API_KEY"),
        llm_base_url=get("LLM_BASE_URL"),
        llm_model=get("LLM_MODEL"),
        vision_kwargs={
            "api_key": get("VISION_API_KEY"),
            "base_url": get("VISION_BASE_URL"),
            "model": get("VISION_MODEL"),
            "source_hint": "复用模型 API（多模态）",
            "budget_seconds": 0.0,
        },
        vision_max_regions=0,
        llm_planner=True,
        llm_patch=True,
        judge_llm=True,
        experiment_llm=True,
        role_llm=True,
        loop=True,
        loop_rounds=3,
        auto_loop=True,
        progress=on_progress,
        # 强制真解析，避免会话快照把上一轮产物伪装成本轮传下去的
        needs_index=False,
    )
    cost = round(time.monotonic() - t0, 1)

    if audit is None:
        print("审计返回空（(None, None)）—— 材料没解析出任何可审计项")
        return 1

    payload = {f.name: plain(getattr(audit, f.name))
               for f in dataclasses.fields(audit.__class__)}
    payload["_meta"] = {
        "cost_seconds": cost,
        "steps": steps,
        "model": get("LLM_MODEL"),
        "base_url": get("LLM_BASE_URL"),
        "vision_model": get("VISION_MODEL"),
        "fresh": fresh,
        "pdf": str(PDF.relative_to(BASE)),
        "code": str(CODE.relative_to(BASE)),
    }
    (OUT / "full_real.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "full_real_report.md").write_text(report or "", encoding="utf-8")

    ex = payload.get("execution_log") or {}
    plan = ex.get("plan") or {}
    print()
    print(f"完成，用时 {cost}s")
    print(f"计划器: {plan.get('planner') or ex.get('planner') or '(见 json)'}")
    print(f"发现 {len(audit.findings)} 条 / 假设 {audit.hypothesis_total} 张 "
          f"/ 实验 {audit.experiment_total} 张 / 化学 {audit.chem_total} 条")
    print(f"结果 -> {OUT / 'full_real.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
