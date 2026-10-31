"""按场景跑全套检查 —— 支持只跑「独立功能」或「全链路」，不必等网页端。

背景（2026-10-30）：56 项检查按**关注点**分三类场景，与测试报告口径一致：
  · 场景一 独立功能（38 项）：单项能力拿出来对不对，多数离线；
  · 场景二 全链路（8 项）  ：真跑完整审计，并比对两次一致性与产物数量；
  · 场景三 网页端（10 项）  ：真开浏览器点。
`tests/run_all_checks.py` 只能整体跑（约 20~40 分钟），而"换了模型想先确认
核心链路还能不能用"时不希望被网页端拖住 —— 所以这里复用它的 `CHECKS`
与 `run_one()`，只做**分组**，不复制任何执行逻辑（改一处就够）。

用法：
    .venv\\Scripts\\python.exe -X utf8 tools\\run_scene_checks.py scene1
    .venv\\Scripts\\python.exe -X utf8 tools\\run_scene_checks.py scene2
    .venv\\Scripts\\python.exe -X utf8 tools\\run_scene_checks.py both

输出：
    · 控制台逐项进度；
    · 桌面\\Paper2Lab测试集\\scene_checks_<场景>_<时间>.json（逐项明细）。
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

# 场景二 / 场景三的**编号集合**（1-based，与 CHECKS 顺序、与历史报告逐条对齐）。
# 其余自动归入场景一 —— 这样以后 CHECKS 加了新项，它默认进"独立功能"，
# 不会因为忘了登记就被静默漏掉。
SCENE2_IDS = {1, 2, 12, 14, 15, 16, 17, 31}
SCENE3_IDS = {8, 9, 10, 11, 23, 28, 39, 44, 45, 56}


def scene_items(scene: str) -> list[tuple[int, str, list[str], int]]:
    """返回 `[(编号, 名称, 参数, 超时)]`。"""
    out = []
    for i, (name, args, to) in enumerate(CHECKS, 1):
        if scene == "scene1" and (i in SCENE2_IDS or i in SCENE3_IDS):
            continue
        if scene == "scene2" and i not in SCENE2_IDS:
            continue
        if scene == "scene3" and i not in SCENE3_IDS:
            continue
        out.append((i, name, args, to))
    return out


def main() -> int:
    scene = (sys.argv[1] if len(sys.argv) > 1 else "both").strip().lower()
    scenes = ["scene1", "scene2", "scene3"] if scene == "both" else [scene]
    titles = {"scene1": "场景一 · 独立功能测试",
              "scene2": "场景二 · 全链路测试",
              "scene3": "场景三 · 网页端测试"}

    all_results = []
    for sc in scenes:
        items = scene_items(sc)
        if not items:
            print(f"[skip] {sc} 没有匹配项", flush=True)
            continue
        print(f"\n===== {titles[sc]}（{len(items)} 项）=====", flush=True)
        t_all = time.monotonic()
        results = []
        for n, (i, name, args, to) in enumerate(items, 1):
            print(f"[{n}/{len(items)}] #{i} {name} …", flush=True)
            r = run_one(name, args, to)
            r["id"] = i
            r["scene"] = sc
            results.append(r)
            st = "TIMEOUT" if r["timeout"] else ("OK" if r["code"] == 0 else f"exit={r['code']}")
            print(f"    -> {st}  {r['elapsed']:.1f}s", flush=True)
            # 每跑完一项就落盘一次：万一后面某项把进程拖死，前面结果不白丢
            json.dump({"scene": sc, "items": results,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(OUT_DIR / f"scene_checks_{sc}.json", "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        el = time.monotonic() - t_all
        bad = [r for r in results if r["timeout"] or r["code"] != 0]
        warn = [r for r in results if r not in bad and r["hits"]]
        print(f"\n--- {titles[sc]} 汇总：{len(results)} 项，"
              f"通过 {len(results) - len(bad)}，失败 {len(bad)}，"
              f"含警示 {len(warn)}，合计 {el:.0f}s ---", flush=True)
        for r in bad:
            print(f"    ❌ #{r['id']} {r['name']}  "
                  f"{'超时' if r['timeout'] else 'exit=' + str(r['code'])}", flush=True)
            print("       " + "\n       ".join(r["tail"].splitlines()[-6:]), flush=True)
        all_results.extend(results)

    stamp = time.strftime("%Y%m%d_%H%M")
    path = OUT_DIR / f"scene_checks_{scene}_{stamp}.json"
    json.dump({"mode": scene, "results": all_results,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细：{path}")
    bad = [r for r in all_results if r["timeout"] or r["code"] != 0]
    print(f"总计 {len(all_results)} 项，失败 {len(bad)} 项")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
