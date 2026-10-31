"""阶段三 3A：具名科学计算工具 —— 「这串数本身说明了什么」。

背景（为什么值得单独写一份验证）：
  2D 给 ② 实验站配了五件工具（单位展开 / 算术求值 / 区间检查 / 倍数探测 / 关系式复算），
  它们解决的是"**同一个数、两种写法**"——比大小、换单位、代公式。
  3A 补的是另一类问题："**这串数本身说明了什么**"——精度会不会丢、这批测量稳不稳、
  这条趋势是不是线性的、峰在哪里。这四件事关系式一条都算不出来。

本脚本最危险的失败模式不是报错，而是**"看起来做了"**：
  菜单上写着四件工具、面板上有调用记录，但那个数不是真算出来的，
  或者更坏——它是**编的**（凑一条假曲线再拟合出个漂亮的 R²）。
  所以断言的重点是三条红线：
    ① 真能跑，且物理结论对（1e-8 在 fp16 里必须落成 0，这是规格值，不是估计）；
    ② **不造数**——输入只能来自材料，装不上就如实说缺什么，不许合成坐标；
    ③ **② 只算数不下判断**——四件工具一律不给 compare（判定归 ③ 站，2C 顶的）。

退出码：0 = 全过，1 = 有没过（不给退出码 = 永远判 OK，等于白写）。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import scifuncs as S                       # noqa: E402
from paper2lab import science_tools as ST                 # noqa: E402
from paper2lab import experiment as E                      # noqa: E402

OK, BAD = [], []


def check(ok: bool, label: str) -> None:
    assert isinstance(ok, bool), "check 的第一个参数必须是 bool"
    (OK if ok else BAD).append(label)
    print(("  [ok] " if ok else "  [FAIL] ") + label)


# ======================================================================== A 登记与同源
print("\n== A. 登记与同源（菜单不许撒谎）==")

check(len(S.SCI_TOOLS) == 4,
      f"登记了 4 件具名科学计算工具: {len(S.SCI_TOOLS)}")
check(len({t.key for t in S.SCI_TOOLS}) == len(S.SCI_TOOLS),
      "科学工具的键互不重复")
check(all(t.name and t.desc and t.cost and t.needs and t.gives for t in S.SCI_TOOLS),
      "每件工具都写了名称 / 说明 / 成本 / 需要什么 / 产出什么（模型靠这些判断）")
check(all(callable(t.run) for t in S.SCI_TOOLS),
      "每件工具都有真能跑的 run（不是占位空壳）")

check(len(ST.SCIENCE_TOOLS) == len(S.SCI_TOOLS),
      "菜单条目数 == 执行体件数（少一件就等于悄悄砍了能力）")
check({c.key for c in ST.SCIENCE_TOOLS} == {t.key for t in S.SCI_TOOLS},
      "菜单与执行体的键一一对应")
check(all(c.station == "② 实验" for c in ST.SCIENCE_TOOLS),
      "四件都归 ② 实验站（③ 只判定、不算数 —— 2C 的口径）")
check(all(c.kind == "science" for c in ST.SCIENCE_TOOLS),
      "菜单条目如实标 kind=science（与审计工序区分开）")

check(not ({c.key for c in ST.CATALOG} & {c.key for c in ST.SCIENCE_TOOLS}),
      "审计工序与科学工具不重名（重名会让模型挑到哪件说不清）")
check(ST.all_keys() == frozenset(c.key for c in ST.CATALOG)
      | frozenset(c.key for c in ST.SCIENCE_TOOLS),
      f"统一白名单 all_keys() = 两边并集（{len(ST.all_keys())} 键）")
check(ST.catalog_keys() == frozenset(c.key for c in ST.CATALOG),
      "catalog_keys() 仍只含审计工序（调度员排的是工序，不该看到算术工具）")

check(len(ST.science_menu_for_model()) == len(S.SCI_TOOLS)
      and all("说明" in m for m in ST.science_menu_for_model()),
      "给模型看的那份菜单带「说明」字段（字段名与 experiment.menu() 同款）")

# ======================================================================== B 真能跑
print("\n== B. 真能跑，且结论对 ==")

# ---- 浮点精度：这一件是整箱里最容易出真问题的，结论必须对
r = S.run("fp16_precision", {"raw": ["1e-8"], "precision": "fp16"})
check(bool(r.get("ok")), "1e-8 在 fp16 下跑成了")
check("落成 0" in str(r.get("text")),
      "1e-8 被判为在 fp16 里落成 0（这是 IEEE-754 规格值, 不是估计）")
check(r.get("value") == 0.0, f"实际落值确实是 0: {r.get('value')}")

# 阈值两侧各点一个, 证明判据不是"凡小数都报"
r_ok = S.run("fp16_precision", {"raw": ["1e-4"], "precision": "fp16"})
check(bool(r_ok.get("ok")) and "正常可表示" in str(r_ok.get("text")),
      "1e-4 在 fp16 下判为正常可表示（阈值不过宽, 不变噪声）")
r_sub = S.run("fp16_precision", {"raw": ["1e-6"], "precision": "fp16"})
check("次正规区" in str(r_sub.get("text")),
      "1e-6 落在次正规区（进了危险区但没落成 0 —— 两档要分得开）")
r_over = S.run("fp16_precision", {"raw": ["1e5"], "precision": "fp16"})
check("溢出" in str(r_over.get("text")), "1e5 判为溢出（上限那一档也认）")

# 换档要真的换: 同一个值在 fp32 下就该没事
r32 = S.run("fp16_precision", {"raw": ["1e-8"], "precision": "fp32"})
check(bool(r32.get("ok")) and "正常可表示" in str(r32.get("text")),
      "同一个 1e-8 换到 fp32 就正常 —— 说明它认的是规格表, 不是硬写一句『太小』")
r_bad = S.run("fp16_precision", {"raw": ["1e-8"], "precision": "fp8"})
check(not r_bad.get("ok") and "没有" in str(r_bad.get("text")),
      "没登记的精度档位如实说不认（不替它编一档出来）")

# ---- 序列统计: 样本标准差（n-1）, 不是总体（n）
r = S.run("series_stats", {"values": ["0.90", "0.92", "0.91", "0.93", "0.89"]})
check(bool(r.get("ok")), "5 个取值跑成了")
check(abs(float(r["value"]) - 0.01581138830084191) < 1e-12,
      f"标准差用的是**样本**口径 n-1（不是 n）: {r['value']:.6g}")
check("1.74%" in str(r.get("text")), "给了离散度（无量纲, 可跨项比较）")
check("中位" in str(r.get("text")), "中位数也算出来了")
r1 = S.run("series_stats", {"values": ["0.9"]})
check(not r1.get("ok") and "至少要 2 个" in str(r1.get("text")),
      "只有 1 个取值时如实说算不了 —— 不硬凑一个 0 方差出来")
r_none = S.run("series_stats", {"values": []})
check(not r_none.get("ok"), "空输入不崩, 如实说算不出来")

# ---- 直线拟合: R² 与「点太少 / 横坐标相同」两条边界
r = S.run("linear_fit", {"xs": [1, 2, 3, 4], "ys": [0.5, 0.52, 0.51, 0.53]})
check(bool(r.get("ok")), "4 个点拟合跑成了")
check(abs(float(r["actual"]) - 0.64) < 1e-9,
      f"R² 算对了: {r['actual']:.4f}")
check("R² 偏低" in str(r.get("text")),
      "R²=0.64 时如实提示「若材料声称线性, 这几个点并不支持它」")
r_perfect = S.run("linear_fit", {"xs": [1, 2, 3], "ys": [2, 4, 6]})
check(abs(float(r_perfect["actual"]) - 1.0) < 1e-9, "完美线性时 R²=1")
r2 = S.run("linear_fit", {"xs": [1, 2], "ys": [2, 4]})
check(not r2.get("ok") and "至少要 3 个点" in str(r2.get("text")),
      "2 个点拒绝拟合（必然完美贴合, 那不是规律是巧合）")
r_flat = S.run("linear_fit", {"xs": [5, 5, 5], "ys": [1, 2, 3]})
check(not r_flat.get("ok") and "竖线" in str(r_flat.get("text")),
      "横坐标全同时如实说拟合不出斜率（不硬凑一个 0）")
r_mis = S.run("linear_fit", {"xs": [1, 2, 3], "ys": [1, 2]})
check(not r_mis.get("ok") and "对不上" in str(r_mis.get("text")),
      "横纵个数不一致时拒绝拟合（不硬配）")

# ---- 峰位: 端点要分得出来
r = S.run("peak_pick", {"xs": [4000, 3500, 3000, 2500], "ys": [0.2, 0.8, 0.3, 0.1]})
check(bool(r.get("ok")) and float(r["actual"]) == 3500.0,
      f"峰位定位到了横坐标 3500: {r['actual']}")
check("两侧邻点都更低" in str(r.get("text")), "内部峰如实说明两侧更低")
r_edge = S.run("peak_pick", {"xs": [1, 2, 3], "ys": [9, 1, 2]})
check("端点" in str(r_edge.get("text")),
      "最高点落在端点时如实提醒「采样范围还没罩住整个峰」")
r2 = S.run("peak_pick", {"xs": [1, 2], "ys": [1, 2]})
check(not r2.get("ok"), "2 个点找不出峰（如实说, 不硬报最高点）")

# ======================================================================== C ② 只算数
print("\n== C. ② 只算数、不下判断（判定归 ③ 站）==")

_res = [S.run(t.key, {"raw": ["1e-8"], "values": ["1", "2", "3"],
                      "xs": [1, 2, 3, 4], "ys": [1, 2, 3, 4.5]})
        for t in S.SCI_TOOLS]
check(all("compare" not in x for x in _res),
      "四件工具的产出里都没有 compare（② 站不下「对得上/对不上」的判断）")
check(all("expected_from" not in x for x in _res),
      "四件都不产 expected_from（判定所需的两侧对照不在 ② 这一层做）")

# 兜底路径: 即便实现里手滑写了 compare, 统一入口也不让它出去
_dirty = S.SCI_TOOLS[0].run({"raw": ["1e-8"], "compare": True,
                             "expected": 1.0, "expected_from": "x"})
check("compare" not in _dirty,
      "就算实现里手滑写了 compare, run() 也会把它剔掉（出口只有一道）")
check(bool(_dirty.get("text")), "剔 compare 不影响人话结论")

# ======================================================================== D 不造数
print("\n== D. 不造数（输入只能来自材料）==")


class _Card:
    key = "learning_rate"
    claim = "两侧学习率写法不同"
    paper_value = "1e-8"
    code_value = "1e-8"
    paper_values = None
    code_values = None
    kind = "取值冲突"
    falsifiable_by = ""
    verdict = ""


_pv = {"learning_rate": "1e-8"}
inp, note = S.assemble_inputs("fp16_precision", card=_Card(),
                              paper_vals=_pv, code_vals=_pv)
check(bool(inp.get("raw")), f"能装配出输入（两侧都抽到了值）: {inp}")
check(S.run("fp16_precision", inp).get("ok"),
      "装配出的输入能真跑（装配与执行是同一条链, 不是两套）")

# 一侧没抽到值 → 装配不出, 且理由写进 note（进「实验设计」而非执行记录）
inp, note = S.assemble_inputs("fp16_precision", card=_Card(),
                              paper_vals={}, code_vals={})
check(not inp and "没抽到" in str(note),
      "两侧都没值时如实说装不上（不用默认值补一个数出来）")

# 序列类工具: 只有一个取值就说算不了, 且明说「论文说稳定」没证据
inp, note = S.assemble_inputs("series_stats", card=_Card(),
                              paper_vals=_pv, code_vals={})
check("证据" in str(note) or "算不了" in str(note),
      "只出现 1 次时如实说「论文若声称这项很稳定, 本次拿不出证据」")

# 曲线类工具: 材料里凑不齐成对坐标 —— 绝不用等间隔坐标硬凑
for key in ("linear_fit", "peak_pick"):
    inp, note = S.assemble_inputs(key, card=_Card(),
                                  paper_vals=_pv, code_vals=_pv)
    check(not inp and "等间隔" in str(note),
          f"{key} 凑不齐成对坐标时**如实说装不上**, 不用合成坐标硬凑")

_bad_in, _bad_note = S.assemble_inputs("no_such_tool", card=_Card())
check(not _bad_in and "没有名为" in str(_bad_note),
      "装配器也只认注册表里的键（菜单外 = 不存在）")

# ======================================================================== E 接进 ② 站
print("\n== E. 接进 ② 实验站（能挑、能排、真跑）==")

check(len(E.EXPERIMENT_TOOLS) == 9,
      f"② 站的工具从 5 件扩到 9 件: {len(E.EXPERIMENT_TOOLS)}")
check(S.catalog_keys() <= {t.key for t in E.EXPERIMENT_TOOLS},
      "四件科学工具都进了 ② 站的执行清单")
check(len(E.menu()) == len(E.EXPERIMENT_TOOLS),
      "给模型看的菜单与执行体同源同数（少一件就等于菜单在骗模型）")

# 规则路径能排上: 极小值必然触发精度检查
from paper2lab.derivation import _expand_units              # noqa: E402
_p = _expand_units({"learning_rate": "1e-8"})
_c = _expand_units({"learning_rate": "1e-4"})
steps, _notes = E._plan_for(_Card(), {"learning_rate": "1e-8"},
                            {"learning_rate": "1e-4"}, _p, _c)
_keys = [s["tool"] for s in steps]
check("fp16_precision" in _keys,
      f"规则路径能排上精度检查（不是只能模型挑）: {_keys}")
check(_keys.index("fp16_precision") < len(_keys) - 1,
      "精度检查排在倍数/区间之前（结论更硬 —— 落成 0 比『差几倍』严重）")

# 判据不许太宽: 正常取值不该触发精度检查
steps2, _ = E._plan_for(_Card(), {"learning_rate": "1e-4"},
                         {"learning_rate": "1e-4"},
                         _expand_units({"learning_rate": "1e-4"}),
                         _expand_units({"learning_rate": "1e-4"}))
check("fp16_precision" not in [s["tool"] for s in steps2],
      "取值正常时不排精度检查（判据不宽, 不制造噪声步骤）")

# 端到端: 设计实验时精度检查真的跑了, 且结果落在记录里
from types import SimpleNamespace                           # noqa: E402
paper = SimpleNamespace(params={"learning_rate": "1e-8"})
code = SimpleNamespace(params={"learning_rate": "1e-4"})
cards = [SimpleNamespace(key="learning_rate", claim="两侧学习率差 4 个量级",
                         paper_value="1e-8", code_value="1e-4",
                         paper_values=None, code_values=None,
                         kind="取值冲突", falsifiable_by="", verdict="",
                         needs_rerun=False)]
exps = E.design_experiments(cards, paper, code)
check(bool(exps) and bool(exps[0].records), f"实验卡上真有执行记录: {len(exps[0].records)} 步")
_paths = list(exps[0].tool_path or ())
check("fp16_precision" in _paths, f"实验卡的工具路径里含科学工具: {_paths}")
_rec = [r for r in exps[0].records if r["tool"] == "fp16_precision"]
check(bool(_rec) and "落成 0" in _rec[0]["output"],
      f"记录里留下了真实结论: {_rec[0]['output'][:40] if _rec else '—'}")
check(_rec and _rec[0].get("ms") is not None,
      "每步都记了耗时（可核验 —— 面板上那个数不是编的）")
check(exps[0].plan_reason and "规则版" in exps[0].plan_reason,
      f"如实标了谁挑的工具路径: {exps[0].plan_reason[:40]}")

# ======================================================================== F 边界与摘要
print("\n== F. 边界与摘要 ==")

r = S.run("no_such_tool", {})
check(not r.get("ok") and "不存在" in str(r.get("text")),
      "调用菜单外的工具被直接拒（不是记一笔然后照跑）")
check(sorted(S.catalog_keys()) == ["fp16_precision", "linear_fit",
                                   "peak_pick", "series_stats"],
      "白名单就是登记的四件")

# 单件工具炸了不能拖垮整轮 —— run() 兜住异常
_orig = S.SCI_TOOLS[0].run
try:
    object.__setattr__(S.SCI_TOOLS[0], "run",
                        lambda inp: (_ for _ in ()).throw(ValueError("boom")))
    r = S.run("fp16_precision", {"raw": ["1e-8"]})
    check(not r.get("ok") and "执行出错" in str(r.get("text")),
          "工具内部抛异常时如实报错（不把异常甩给上层）")
finally:
    object.__setattr__(S.SCI_TOOLS[0], "run", _orig)

_sum = S.summarize([{"tool": "fp16_precision", "ok": True},
                    {"tool": "series_stats", "ok": False}])
check(_sum["calls"] == 2 and _sum["ok"] == 1, f"摘要统计调用数与成功数: {_sum}")
check("内置科学计算工具" in _sum["engine"],
      "摘要如实说用的是内置引擎（不引外部数值库）")

# 反向不 import（成环会直接崩 import）
for mod in ("compute.py", "derivation.py", "collider.py", "hypothesis.py"):
    src = (ROOT / "paper2lab" / mod).read_text(encoding="utf-8")
    check("scifuncs" not in src, f"{mod} 没有反向 import scifuncs（不成环）")

# ======================================================================== 汇总
print("\n" + "=" * 60)
print(f"通过 {len(OK)} 项，未过 {len(BAD)} 项")
if BAD:
    print("\n未过明细:")
    for b in BAD:
        print("  - " + b)
    sys.exit(1)
print("3A 全部通过")
sys.exit(0)
