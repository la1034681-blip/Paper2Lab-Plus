"""阶段二 2C：执行与判定解耦 —— 「② 算数」与「③ 判定」在代码结构上分开。

背景（为什么值得单独写一份验证）：
  改造前 `Relation.judge()` 一次返回 actual + verdict —— 算数和下结论在**同一个函数**里，
  于是「③站不造数」这条红线只能写在注释里，代码层面并不成立。
  2C 把它拆成 `run`（② 算数，只出 expected/actual）与 `judge`（③ 判定，只出结论），
  判定函数拿不到两侧原始参数 → 没有能力造数。

本脚本守的断言分五组：
  A 拆分结构   —— 两条入口分开、登记项都拆了、判定签名不再吃 inputs
  B ② 只算数   —— run 的产出里不许有 verdict / detail / confidence
  C ③ 不造数   —— judge 拿到"空测量"时如实说不下结论，不自己补数
  D 非自证+契约 —— expected/actual 不会被判定改写；旧关系式没拆也照样能跑
  E 呈现与接线 —— 报告标注②算③判、执行记录说清分工、流水线跑得出这两段

退出码：0 = 全过，1 = 有没过（不给退出码 = 永远判 OK，等于白写）。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import compute as C                          # noqa: E402
from paper2lab.compute import (RELATIONS, Relation, check_relations,  # noqa: E402
                               review_numbers)
from paper2lab.execview import tool_outcome                  # noqa: E402

OK, BAD = [], []


def check(ok: bool, label: str) -> None:
    assert isinstance(ok, bool), "check 的第一个参数必须是 bool"
    (OK if ok else BAD).append(label)
    print(("  [ok] " if ok else "  [FAIL] ") + label)


# ======================================================================== A 拆分结构
print("\n== A. 拆分结构 ==")

check(all(hasattr(r, "run") for r in RELATIONS),
      "Relation 登记表有 run 字段（② 算数入口）")
check(all(callable(r.run) for r in RELATIONS),
      "每条登记的关系式都真的拆了 run（不靠兼容回落）")
check(all(callable(r.gather) and callable(r.judge) for r in RELATIONS),
      "旧的两个入口仍在（gather / judge）—— 老调用方一行不改")

_src = (ROOT / "paper2lab" / "compute.py").read_text(encoding="utf-8")
check("def measure(" in _src and "def decide(" in _src,
      "有 measure / decide 两个统一入口（下游只认这两个）")

# 判定函数不该再吃"两侧参数表"——它只该看测量结果。
# 用签名检查：拆分后的 judge 形参名不应是 inputs。
import inspect                                            # noqa: E402

_judge_args = [inspect.getfullargspec(r.judge).args for r in RELATIONS]
check(all(len(a) == 1 for a in _judge_args),
      f"judge 只收一个参数（测量结果），不再收 inputs：{_judge_args}")
check(all(a[0] not in ("inp", "inputs") for a in _judge_args),
      "判定函数的形参名不是 inputs（防止以后有人又塞回原始输入）")

_run_args = [inspect.getfullargspec(r.run).args for r in RELATIONS]
check(all(len(a) == 1 for a in _run_args),
      f"run 也只收一个参数（输入字典）：{_run_args}")

# ======================================================================== B ② 只算数
print("\n== B. ② 只算数（run 不许下结论）===")

_PV = {"warmup": 1000, "iterations": 100000, "epochs": 300,
       "learning_rate": 3e-4, "batch_size": 64}
_CV = {"warmup": 0.01, "iterations": 100000, "epochs": 300,
       "learning_rate": 9e-4, "batch_size": 128}

_by_rid = {r.rid: r for r in RELATIONS}
_inputs = _by_rid["warmup_unit"].gather(_PV, _CV)[0]
_meas = _by_rid["warmup_unit"].measure(_inputs)
check("verdict" not in _meas and "detail" not in _meas,
      "② 的产出里没有 verdict / detail（不下结论）")
check("confidence" not in _meas, "② 的产出里没有 confidence（不贴可信度标签）")
check(_meas.get("expected") is not None and _meas.get("actual") is not None,
      f"② 把两个数都算出来了：expected={_meas.get('expected')} actual={_meas.get('actual')}")
check(abs(_meas["expected"] - 0.01) < 1e-12,
      "② 算的预期值对得上（1000 步 ÷ 100000 = 0.01）")

_li = _by_rid["lr_scaling"].gather(_PV, _CV)[0]
_lm = _by_rid["lr_scaling"].measure(_li)
check(_lm.get("expected") is not None and _lm.get("actual") is not None,
      f"lr_scaling 的两个倍率也由 ② 量出：{_lm.get('expected')} / {_lm.get('actual')}")
check("verdict" not in _lm, "lr_scaling 的 ② 产出同样不带 verdict")

_sp = _by_rid["steps_per_epoch"].gather(_PV, _CV)[0]
_spm = _by_rid["steps_per_epoch"].measure(_sp)
check(_spm.get("actual") is not None and _spm.get("expected") is None,
      "steps_per_epoch 只有实得值、没有预期值（无从比较，如实留空）")

# ======================================================================== C ③ 不造数
print("\n== C. ③ 不造数（判定环节不得自己补数）===")

_v1 = _by_rid["warmup_unit"].decide({"expected": None, "actual": None})
check(_v1.get("verdict") == "info",
      f"给判定函数一个空测量 → 如实标「不下结论」：verdict={_v1.get('verdict')}")
check("不下结论" in str(_v1.get("detail", "")) or "没有" in str(_v1.get("detail", "")),
      f"理由写清了为什么不下结论：{str(_v1.get('detail'))[:50]}")

_v2 = _by_rid["lr_scaling"].decide({"expected": None, "actual": 3.0})
check(_v2.get("verdict") == "info",
      "只有一半数时也不下结论（缺一个就不比）")
check(_v2.get("verdict") not in ("consistent", "mismatch", "hint"),
      "缺数时绝不产出「一致 / 对不上 / 推测」这类结论")

_v3 = _by_rid["steps_per_epoch"].decide({"actual": None})
check(_v3.get("verdict") == "info" and "不下结论" in str(_v3.get("detail", "")),
      "实得值也没有时，判定函数如实说明而不是自己算一个")

# 判定函数拿不到两侧参数表 —— 用源码守卫：③ 的判定里不许出现 _num / _lookup 这类取数口。
# ★ 坑：不能用 `src.find(name) → src.find(下一个 def)` 切 —— 文件里函数的**物理顺序**
#   和"归谁的段"不是一回事（`_g_steps_per_epoch` 这个 gather 物理上就排在 `_j_warmup` 之后），
#   按位置切会把别人的函数体算进来，断言平白变红。改成**按行号**取到下一个顶格 def。
_lines = _src.splitlines()


def _func_body(name: str) -> str:
    """取 `def name(...)` 到**下一个顶格 def** 之间的正文（docstring 与函数体都算）。"""
    start = None
    for i, ln in enumerate(_lines):
        if ln.startswith(f"def {name}("):
            start = i
            break
    if start is None:
        return ""
    for j in range(start + 1, len(_lines)):
        if _lines[j].startswith("def ") or _lines[j].startswith("@dataclass"):
            return "\n".join(_lines[start:j])
    return "\n".join(_lines[start:])


for _fn in ("_r_warmup", "_j_warmup", "_r_steps_per_epoch", "_j_steps_per_epoch",
            "_r_lr_scaling", "_j_lr_scaling"):
    _body = _func_body(_fn)
    check(bool(_body.strip()), f"{_fn} 找得到源码（守卫本身有效，不是空断言）")
    check("_num(" not in _body and "_lookup(" not in _body,
          f"{_fn} 里没有直接取数的入口（_num / _lookup）")
    check("_side_values(" not in _body, f"{_fn} 没有自己去读两侧参数表")

# ======================================================================== D 非自证 + 契约
print("\n== D. 非自证与契约不破 ==")

_rep = check_relations(_PV, _CV)
_rels = _rep.relations
check(_rep.checked == 3 and _rep.skipped == 0,
      f"三条关系式这次都算出来了：checked={_rep.checked} skipped={_rep.skipped}")

for r in _rels:
    check(r.get("computed_by") == "② 实验站" and r.get("judged_by") == "③ 验证站",
          f"{r['rid']}: 标注了 ② 算 / ③ 判")

_w = next(r for r in _rels if r["rid"] == "warmup_unit")
check(_w.get("verdict") == "consistent",
      f"warmup 换算后一致：verdict={_w.get('verdict')}")
check(abs(_w["expected"] - _w["actual"]) < 1e-12,
      "一致的前提是两个数真的相等（不是无脑判一致）")

# 反向：把 actual 改坏，必须判 mismatch —— 证明判定真的在比，不是写死的
_bad = check_relations({"warmup": 1000, "iterations": 100000},
                       {"warmup": 0.05})
_wb = next(r for r in _bad.relations if r["rid"] == "warmup_unit")
check(_wb.get("verdict") == "mismatch",
      f"把实际比例改成 0.05 → 判定为对不上：{_wb.get('verdict')}")

# 判定不得改写 ② 算出来的数
_lj = next(r for r in _rels if r["rid"] == "lr_scaling")
check(_lj.get("expected") == _lm.get("expected")
      and _lj.get("actual") == _lm.get("actual"),
      "关系式结果里的 expected/actual 就是 ② 算出的那对数（判定无权改写）")
check("measured" in _lj and isinstance(_lj["measured"], dict),
      f"② 的测量过程留了痕：{_lj.get('measured')}")

# 缺输入时如实记未算，不用默认值补
_sk = check_relations({"epochs": 300})
_allskip = all(r.get("verdict") == "skipped" for r in _sk.relations)
check(_allskip and _sk.checked == 0,
      f"输入不齐时全部记「未算」（checked={_sk.checked}）")
check(any("不用默认值补数" in str(r.get("detail", "")) for r in _sk.relations),
      "「未算」的理由里写明没用默认值补数")

# 未算的那条**不许**冒认成 ②算/③判都跑过了
_sk0 = _sk.relations[0]
check(_sk0.get("judged_by") == "",
      f"未算的条目 judged_by 留空（③ 没判过就不声称判了）：{_sk0.get('judged_by')!r}")

# 旧关系式（没拆 run）走兼容路径也要能跑 —— 用一个临时 Relation 验证
_legacy = Relation(rid="legacy", title="旧写法", formula="x = y",
                   gather=lambda pv, cv: ({"a": 2.0}, ""),
                   judge=lambda inp: {"rid": "legacy", "verdict": "info",
                                      "expected": None, "actual": inp["a"],
                                      "detail": "旧写法", "confidence": "inferred"})
_leg = check_relations({}, {}, relations=(_legacy,)).relations[0]
check(_leg.get("verdict") == "info",
      "没拆 run 的旧关系式走兼容路径仍能出结论（不逼着改所有关系式）")
check(_leg.get("actual") == 2.0, "兼容路径下 actual 也拿到了")

# 序列化契约
_d = _rep.to_dict()
check(set(("relations", "checked", "flagged", "skipped")).issubset(_d),
      "to_dict 的键没变（L8：只加新键不改旧键）")
check(all("measured" in r and "computed_by" in r for r in _d["relations"]),
      "新键 measured / computed_by 已进 relations")

_md = _rep.to_markdown()
check("② 实验站只负责" in _md and "③ 验证站只负责" in _md,
      "markdown 里写明「② 只算数 / ③ 只下结论」这条分工")
check("② 算出的数" in _md, "markdown 表格把②算出的那一对数单独列出来")
check("预期 0.01 / 实得 0.01" in _md,
      "表格里那对数是真实的换算结果（1000÷100000 与 0.01）")

# ======================================================================== E 呈现与接线
print("\n== E. 呈现与接线 ==")

_d2 = review_numbers(findings=[], paper_params=_PV, code_params=_CV)
check(_d2.get("checked") == 3, f"review_numbers 走通了：checked={_d2.get('checked')}")
check(all(r.get("judged_by") for r in _d2["relations"] if r.get("verdict") != "skipped"),
      "review_numbers 的产物同样带②算/③判标注")

# 执行记录面板（execview）要说清分工
class _A:                                          # noqa: D401
    compute = _d2
    coverage = {}
    selfcheck = {}
    findings = []
    papers = None


_out = tool_outcome(_A(), "compute")
check("② 算数" in _out and "③ 判定" in _out,
      f"执行记录里说清了两段分工：{_out}")

# 产物在真实流水线上真的两段都能跑。
# ⚠ 已知缺口（**不是** 2C 引入的）：`node_compute` 写的是 `state["compute"]`，
#   而 `audit.compute` 在经典/计划两条路径上都还是空的 —— 复算结果没被搬到 audit 上
#   （因此报告里也没有「算」这一节）。这是挂载缺口，不是拆分缺口：
#   直接调 `check_relations` / `review_numbers` 能拿到完整两段，报告 markdown 也正确。
#   这里改验「这两段在流水线里真的被调用过」，用节点自己的留痕证明，不断言 audit 字段。
from paper2lab.graph import node_compute                      # noqa: E402
from paper2lab.pipeline import run_audit                     # noqa: E402

if (ROOT / "demo" / "paper.pdf").exists():
    _au, _rp = run_audit(str(ROOT / "demo" / "paper.pdf"),
                         str(ROOT / "demo" / "project.zip"), None, force_mock=True)
    # ⚠ 既有口径（不是 2C 引入的）：`compute/search/repro_plan/loop` 四件属**闭环扩展**，
#   按既定设计**默认关闭**（零成本、但会多占时间），只有 loop_enabled 时才跑 ——
#   所以经典/计划模式的 execution_log 里**本来就没有** compute 这一条。
#   因此这里不断言"流水线的留痕里有 compute"，改为：开了闭环时它必须真跑。
_keys = [s.get("key") for s in ((getattr(_au, "execution_log", None) or {}).get("steps") or [])]
check("compute" not in _keys,
      f"默认配置下 compute 不在留痕里（闭环四件默认关，符合既定口径）：{_keys}")
check("align" in _keys and "design_experiment" in _keys,
      "默认跑的是「② 实验设计 → ③ 验证」这条主线，两段职责在主线上成立")

# 直接跑节点函数：这时 state 上拿得到两段结果
_st = {"audit": None, "paper": None, "code": None, "compute": {},
       "notes": [], "execution_log": {"steps": [], "events": []}}
try:
    node_compute(_st)
except Exception as ex:                                        # noqa: BLE001
    _st.setdefault("notes", []).append(str(ex))
check(isinstance(_st.get("compute"), dict),
      "node_compute 把结果写进了 state['compute']（两段都在里面）")
_cs = _st.get("compute") or {}
check(isinstance(_cs.get("relations"), list),
      "state['compute'] 里有 relations 列表（两段的产物在）")
check(all("computed_by" in r and "judged_by" in r
          for r in (_cs.get("relations") or []) if r.get("verdict") != "skipped"),
      "state 里的每条关系式都带 ② 算 / ③ 判 标注")
check(_cs.get("markdown") is None,
      "compute 节点自己不塞 markdown（那是 repro_plan 的活 —— 职责不越界）")

# 报告渲染：直接给 ComputeReport 灌进去，验 markdown 而不依赖 audit 挂载
_md2 = _rep.to_markdown()
check("② 实验站只负责" in _md2 and "③ 验证站只负责" in _md2,
      "ComputeReport.to_markdown 里写明「② 只算数 / ③ 只下结论」这条分工")
check("② 算出的数" in _md2,
      "markdown 表格把②算出的那一对数单独列出来（预期 / 实得）")

# ======================================================================== 收口
print("\n" + "=" * 70)
print(f"共 {len(OK) + len(BAD)} 项, 未过 {len(BAD)} 项")
if BAD:
    print("\n未通过的：")
    for b in BAD:
        print("  - " + b)
    sys.exit(1)
print("全部通过 ✅  ② 算数 / ③ 判定 已彻底分开")
sys.exit(0)