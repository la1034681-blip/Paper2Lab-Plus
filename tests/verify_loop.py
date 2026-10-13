"""迭代回边验证（阶段五 · M2 闭环）——离线、零成本。

这里最要紧的是三条**否定性**断言（做错了也"看起来能跑"）：
1. 第 1 轮**不是**"把代码改成论文值再看是否相等"（那是恒等成立、什么都没验）；
   真正的判据是**可落地性**：根因定得下来 + 补丁给得出来；
2. 第 3 轮（把论文改成代码）**绝不算收敛** —— 否则收敛率永远 100%；
3. 「修正」是内存里的反事实推演：**代码文件必须一字节不变**（硬自检）。
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.loop import (MAX_ROUNDS, STRATEGIES, _actionable, _round1_feasible,
                            _round2_candidate, _round3_observe, run_loop)
from paper2lab.models import (Category, CodeInfo, Confidence, Evidence, Finding,
                              Risk, Status)

CHECKS = 0
FAILURES = []


def check(ok, label):
    global CHECKS
    CHECKS += 1
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


def _tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


SRC = ("import torch\n"
       "lr = 0.001\n"
       "optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n")


def make_code_dir():
    d = Path(tempfile.mkdtemp())
    (d / "train.py").write_text(SRC, encoding="utf-8")
    return d


def mk_finding(key, disp, pv, cv, risk=Risk.HIGH, status=Status.INCONSISTENT,
               loc="train.py:2"):
    return Finding(
        param_key=key, display_name=disp, category=Category.PARAM, status=status,
        confidence=Confidence.CONFIRMED, risk=risk,
        paper=Evidence(source="paper", location="Page 3",
                       snippet=f"{key} is {pv}", raw_value=pv) if pv else None,
        code=Evidence(source="code", location=loc,
                      snippet=f"{key} = {cv}", raw_value=cv) if cv else None,
    )


class _Audit:
    def __init__(self, findings, code=None):
        self.findings = findings
        self.code = code if code is not None else CodeInfo()


# ═══════════════════════════════════════ 1. 前置与边界

sec("1. 可迭代项的挑选与边界")

f_inc = mk_finding("learning_rate", "Learning Rate", "1e-4", "0.001")
f_miss = mk_finding("input_size", "Input Size", "224", "", status=Status.MISSING_IN_CODE)
f_one = mk_finding("batch_size", "Batch Size", "256", "", status=Status.MISSING_IN_PAPER)
check(_actionable(_Audit([f_inc, f_miss, f_one])) == [f_inc],
      "只挑「值不一致且两侧都有值」的结论（缺失类无法谈修正）")

f_hi = mk_finding("a", "AA", "1", "2", risk=Risk.HIGH)
f_lo = mk_finding("b", "BB", "1", "2", risk=Risk.LOW)
acts = _actionable(_Audit([f_lo, f_hi]))
check([f.param_key for f in acts] == ["a", "b"],
      "高风险排前面（同样的三轮预算先用在影响最大的项上）")

empty = run_loop(_Audit([]))
check(empty.resolved == 0 and empty.rounds == [], "没有可迭代项 -> 不跑，如实记未执行")
check(any("无可迭代项" in n for n in empty.notes), "notes 里说明为什么没跑")

zero = run_loop(_Audit([f_inc]), max_rounds=0)
check(zero.rounds == [] and any("max_rounds=0" in n for n in zero.notes),
      "max_rounds=0 -> 一轮都不跑")

# ═══════════════════════════════════════ 2. 第 1 轮: 可落地性

sec("2. 第 1 轮 —— 判据是「可落地」而不是「值相等」")

d = make_code_dir()
before_hash = _tree_hash(d)
code = CodeInfo(params={"learning_rate": [Evidence(
    source="code", location="train.py:2", snippet="lr = 0.001", raw_value="0.001")]},
    file_lines={"train.py": 3})
a = _Audit([mk_finding("learning_rate", "Learning Rate", "1e-4", "0.001")], code)
rep = run_loop(a, str(d))
check(rep.resolved == 1, "根因定得下来 + 补丁给得出来 -> 判为可落地（收敛）")
check(rep.rounds[0]["improved"] == 1, "第 1 轮记 1 条改善")
r1 = rep.rounds[0]["items"][0]
check("train.py:2" in r1["reason"], "理由里写明改哪个位置（可核验）")
check("1e-4" in r1["reason"], "理由里写明改成什么值")
check("proposed" in r1["reason"], "理由里带补丁草案状态（不落盘但可查）")
check(r1["target_loc"] == "train.py:2", "target_loc 落到具体行")
check(r1["side"] == "code", "第 1 轮改的是代码侧（按论文对齐）")
check(_tree_hash(d) == before_hash, "**代码文件一字节不变**（反事实推演，不是真改代码）")

# 不可落地: 根因定不下来
a2 = _Audit([mk_finding("learning_rate", "Learning Rate", "1e-4", "0.001")], CodeInfo())
rep2 = run_loop(a2, str(d))
check(rep2.resolved == 0, "根因追查不可用 -> 第 1 轮不判收敛")
check(len(rep2.unresolved) == 1, "该条升级为待人工确认项")
check("不可落地" in rep2.rounds[0]["items"][0]["reason"], "理由开头写明不可落地")
check("根因" in rep2.rounds[0]["items"][0]["reason"], "并指向根因这个具体原因")

# 不可落地: 补丁被程序闸门拒
f_opt = mk_finding("optimizer", "Optimizer", "AdamW", "Adam", risk=Risk.MEDIUM)
code_opt = CodeInfo(params={"optimizer": [Evidence(
    source="code", location="train.py:3", snippet="Adam(...)", raw_value="Adam")]},
    file_lines={"train.py": 3})
rep3 = run_loop(_Audit([f_opt], code_opt), str(d))
check(rep3.resolved == 0, "实现选择类参数（优化器）不给补丁 -> 不判收敛")
check("补丁不可给" in rep3.rounds[0]["items"][0]["reason"]
      or "不可落地" in rep3.rounds[0]["items"][0]["reason"],
      "理由里说明是补丁这一环被拒")
check("实现选择" in rep3.rounds[0]["items"][0]["reason"],
      "并复述补丁模块的拒绝理由（不自己另编一个）")

# ═══════════════════════════════════════ 3. 第 2 轮: 换候选点

sec("3. 第 2 轮 —— 换根因候选点（是不是抽取取错了点）")

f = mk_finding("learning_rate", "Learning Rate", "1e-4", "0.001")
st = {"paper": "1e-4", "code": "0.001", "status": "inconsistent",
      "resolved": False, "history": []}
trace = SimpleNamespace(
    candidates=[SimpleNamespace(file="train.py", line=2, value="0.001"),
                SimpleNamespace(file="config.py", line=5, value="1e-4")],
    verdict_loc="", confidence="none")

item, improved = _round2_candidate(f, st, trace, "inconsistent")
check(improved is True, "候选里存在与论文一致的值 -> 收敛")
check(st["resolved"] is True, "状态账被更新")
check("config.py:5" == item.target_loc, "指向那个候选的位置")
check("抽取取错" in item.reason, "理由说明这是抽取问题而非真冲突")
check(item.strategy == "rootcause_candidate", "轮次标识正确")

# 没有可替代候选 -> 确认是真冲突
st2 = {"paper": "1e-4", "code": "0.001", "status": "inconsistent",
       "resolved": False, "history": []}
trace2 = SimpleNamespace(
    candidates=[SimpleNamespace(file="train.py", line=2, value="0.001")],
    verdict_loc="train.py:2", confidence="high")
item2, improved2 = _round2_candidate(f, st2, trace2, "inconsistent")
check(improved2 is False, "只此一处赋值 -> 不收敛")
check("确属真冲突" in item2.reason, "明确给出「不是抽取取错点」这个结论")
check(st2["resolved"] is False, "状态账未被错误置为已解决")

item3, improved3 = _round2_candidate(f, st2, None, "inconsistent")
check(improved3 is False and "没有其它赋值点" in item3.reason,
      "根因追查不可用时也如实说明（不静默）")

# ═══════════════════════════════════════ 4. 第 3 轮: 观察, 不计收敛

sec("4. 第 3 轮 —— 反向观察**不计收敛**（否则收敛率永远 100%）")

check(STRATEGIES[2]["observational"] is True, "第 3 轮在策略表里被标为观察轮")
check(STRATEGIES[0]["observational"] is False and STRATEGIES[1]["observational"] is False,
      "前两轮才是真正判收敛的轮次")

f_s = mk_finding("scheduler", "LR Scheduler", "cosine", "step", risk=Risk.MEDIUM)
st3 = {"paper": "cosine", "code": "step", "status": "inconsistent",
       "resolved": False, "history": []}
item4, improved4 = _round3_observe(f_s, st3, "inconsistent")
check(item4.after == "consistent", "按代码值对齐后字面确实一致")
check(item4.resolved is False, "**但不算收敛**（resolved 必须为 False）")
check(improved4 is False, "也不计作改善")
check(st3["resolved"] is False, "状态账保持未解决")
check("不计入收敛" in item4.reason, "理由里写明为什么不计（换目标 ≠ 解决问题）")
check("换复现目标" in item4.reason or "换目标" in item4.reason, "并点明这是换目标")
check(item4.side == "paper", "第 3 轮改的是论文侧")

# ═══════════════════════════════════════ 5. 有界性

sec("5. 有界自主 —— 上限 3 轮 + 无改善即停")

check(MAX_ROUNDS == 3, "硬上限 3 轮")
check(run_loop(_Audit([f_inc]), str(d), max_rounds=99).max_rounds == 3,
      "传入 99 也被夹到 3（防有人从上层放开）")

# 全程无改善的场景: 只会停在第 1 轮之后（不等跑满 3 轮）
stuck = mk_finding("optimizer", "Optimizer", "AdamW", "Adam", risk=Risk.HIGH)
rep_stuck = run_loop(_Audit([stuck], CodeInfo()), str(d))
check(len(rep_stuck.rounds) <= 2, "没有任何改善 -> 提前停止，轮数 %d ≤ 2" % len(rep_stuck.rounds))
check(any("提前停止" in n or "有界" in n for n in rep_stuck.notes), "notes 里说明是按有界自主停止")
check(rep_stuck.resolved == 0, "一条都没收敛")
check(len(rep_stuck.unresolved) == 1, "如实升级为待人工确认项")

# 多轮场景: 单条可落地 -> 只跑 1 轮就够
check(len(rep.rounds) == 1, "第 1 轮就全部收敛 -> 不再跑第 2/3 轮（省预算）")

# ═══════════════════════════════════════ 6. 留痕与交付

sec("6. 留痕与交付")

check(isinstance(rep.to_dict(), dict) and "rounds" in rep.to_dict(), "可序列化")
md = rep.to_markdown()
check("迭代回边" in md, "markdown 有标题")
check("不写任何文件" in md, "markdown 写明不写文件这条边界")
check("有界自主" in md, "markdown 写明有界自主")
check("第 1 轮" in md, "markdown 逐轮列出")

md_stuck = rep_stuck.to_markdown()
check("待人工确认项" in md_stuck, "未收敛的项在 markdown 里被显式升级为待人工确认")
check("仍" in md_stuck, "并说明仍未收敛（不粉饰）")

check(any("没有写任何文件" in n for n in rep.notes), "notes 里声明没有写文件")
check(any("补丁草案只生成不落盘" in n for n in rep.notes), "notes 里声明补丁不落盘")

# 每条未收敛项都带原两侧取值（升级给人工时要看得见原始证据）
u = rep_stuck.unresolved[0]
check(u["paper"] == "AdamW" and u["code"] == "Adam", "未收敛项带两侧原值")
check(bool(u["reason"]), "未收敛项带原因")

# 最终再确认一次：全程没动过用户的代码
check(_tree_hash(d) == before_hash, "跑完多轮后代码目录哈希仍与开跑前一致")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段五 迭代回边验证: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_loop.py)")
print("=" * 62)
