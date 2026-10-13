"""「算」科学计算工具验证（阶段五 · M2）——离线、零成本、纯规则。

关注点（每条都对应一个"容易做错"的地方）:
1. 安全求值：`eval_arith` 绝不能变成 `eval`（代码包是用户上传的）；
2. 关系式**非自证循环**：expected 与 actual 必须来自不同输入，否则恒等成立、什么都验不出；
3. 算不出来如实说"未算"，**不用默认值补数**；
4. 推演（inferred）与核验（confirmed）分开标注，不混进确定性结论。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.compute import (ComputeReport, RELATIONS, check_relations,
                               eval_arith, magnitude_hint, review_numbers,
                               to_number)

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


# ═══════════════════════════════════════ 1. 安全求值

sec("1. 安全求值 eval_arith —— 不是 eval")

check(abs(eval_arith("3*1e-4") - 3e-4) < 1e-15, "四则 + 科学计数法: 3*1e-4")
check(eval_arith("2**10") == 1024.0, "乘方: 2**10")
check(eval_arith("512*512") == 262144.0, "乘法: 512*512")
check(eval_arith("(1+2)*3") == 9.0, "括号优先级")
check(abs(eval_arith("10/4") - 2.5) < 1e-12, "除法")
check(eval_arith("10//4") == 2.0, "整除")
check(eval_arith("10%3") == 1.0, "取模")
check(eval_arith("-1e-4") == -1e-4, "一元负号含科学计数法")
check(eval_arith("1/0") is None, "除零 -> None（不抛异常）")
check(eval_arith("__import__('os').system('x')") is None, "拒绝函数调用（这是与 eval 的分水岭）")
check(eval_arith("open('a.txt')") is None, "拒绝任意函数名")
check(eval_arith("a + 1") is None, "拒绝变量名")
check(eval_arith("x") is None, "拒绝裸变量")
check(eval_arith("[1,2]") is None, "拒绝列表字面量")
check(eval_arith("1;2") is None, "拒绝多语句（eval 模式会拒）")
check(eval_arith("1" * 200) is None, "超长表达式直接拒（防解析炸弹）")
check(eval_arith(None) is None, "None 输入")
check(eval_arith("") is None, "空串")
check(eval_arith("True") is None, "布尔常量不算数")

# ═══════════════════════════════════════ 2. 数值归一

sec("2. to_number —— 字面量优先，表达式兜底")

check(to_number("1e-4") == 1e-4, "科学计数法")
check(to_number("0.001") == 0.001, "小数")
check(to_number("50%") == 0.5, "百分比 -> 小数")
check(to_number("1/2") == 0.5, "分数")
check(abs(to_number("3*1e-4") - 3e-4) < 1e-15, "表达式（对撞口不认，这里兜住）")
check(to_number("cosine") is None, "非数值参数返回 None（不猜）")
check(to_number(None) is None, "None")

# ═══════════════════════════════════════ 3. 量纲提示

sec("3. 量纲提示 magnitude_hint —— 阈值 10^2 起，避免噪声")

check(magnitude_hint("1e-4", "1e-1") is not None, "差 10^3 -> 提示")
check("10^-3" in magnitude_hint("1e-4", "1e-1"), "提示里写明倍率（可核验）")
check(magnitude_hint("1e-4", "1e-3") is None,
      "只差 10 倍 -> 不提示（学习率差 10 倍是常见合法差异，提示会变噪声）")
check(magnitude_hint("2", "1") is None, "2 倍 -> 不提示（正常差异）")
check(magnitude_hint("1.5", "1") is None, "1.5 倍 -> 不提示")
check(magnitude_hint("0", "0.1") is None, "含 0 -> 不提示（除零无意义）")
check(magnitude_hint("cosine", "0.1") is None, "非数 -> 不提示")
check(magnitude_hint("1e-12", "1e12") is None, "超过 10^12 倍 -> 超出可信范围不提示")

# ═══════════════════════════════════════ 4. 关系式登记表

sec("4. 关系式登记表")

rids = [r.rid for r in RELATIONS]
check(len(rids) == len(set(rids)), "rid 唯一")
check(all(callable(r.gather) and callable(r.judge) for r in RELATIONS),
      "每条关系式都有 gather 与 judge 两个入口")
check(all(r.formula and r.title for r in RELATIONS), "每条都写了可读的公式与标题")
check(len(RELATIONS) >= 3, "至少登记 3 条关系式: %d" % len(RELATIONS))

# ═══════════════════════════════════════ 5. warmup 步数↔比例（核心：非自证循环）

sec("5. warmup 步数 ↔ 比例 换算核验")

PV = {"learning_rate": "1e-4", "batch_size": "256", "warmup": "1000",
      "iterations": "20000", "epochs": "10"}
CV = {"learning_rate": "1e-4", "batch_size": "256", "warmup": "0.05"}

r = check_relations(PV, CV)
by = {x["rid"]: x for x in r.relations}
w = by["warmup_unit"]
check(w["verdict"] == "consistent", "论文 1000 步 / 代码 0.05 比例 -> 换算后一致")
check(w["expected"] is not None and w["actual"] is not None,
      "expected/actual 两个值都给了（换算过程可核验）")
check(abs(w["expected"] - w["actual"]) < 1e-12, "两者确实相等（不是靠字符串比出来的）")
check(abs(w["expected"] - 0.05) < 1e-12, "换算值 = 1000/20000 = 0.05")
check(w["confidence"] == "confirmed", "换算核验属确定性结论（confirmed）")

r2 = check_relations({"warmup": "0.05", "iterations": "20000"}, {"warmup": "1000"})
check(r2.relations[0]["verdict"] == "consistent", "反向（论文比例 / 代码步数）同样识别为一致")

r3 = check_relations({"warmup": "1000", "iterations": "20000"}, {"warmup": "0.5"})
check(r3.relations[0]["verdict"] == "mismatch", "代码写 0.5 -> 换算后仍对不上，报 mismatch")
check(r3.relations[0]["verdict"] != "consistent", "绝不把「换算对不上」说成一致")

r4 = check_relations({"warmup": "1000", "iterations": "20000"}, {"warmup": "800"})
check(r4.relations[0]["verdict"] == "info", "两侧同为步数 -> 不换算，如实说明无需换算")

r5 = check_relations({"warmup": "1000"}, {})
check(r5.relations[0]["verdict"] == "skipped", "缺 iterations -> 未算（不用默认值补）")
check("iterations" in r5.relations[0]["detail"], "detail 里写明缺哪个输入（可核验）")
check(r5.checked == 0 and r5.skipped >= 1, "未算计入 skipped 而不是 checked")

r6 = check_relations({"warmup": "1000", "iterations": "20000"}, {})
check(r6.relations[0]["verdict"] == "skipped", "只有一侧声明 warmup -> 未算（无从换算）")

# ═══════════════════════════════════════ 6. 推演类关系式

sec("6. 推演类关系式（标 inferred，不当结论用）")

s = by["steps_per_epoch"]
check(s["verdict"] == "info", "每轮步数只做推演（info），不给一致/不一致结论")
check(s["confidence"] == "inferred", "推演标注 inferred（不混进确定性结论）")
check(abs(s["actual"] - 2000) < 1e-9, "20000 / 10 = 2000 步/轮")

r7 = check_relations({"learning_rate": "1e-4", "batch_size": "256"},
                     {"learning_rate": "4e-4", "batch_size": "1024"})
ls = {x["rid"]: x for x in r7.relations}["lr_scaling"]
check(ls["verdict"] == "hint", "批大小×4 且 lr×4 -> 命中线性缩放推测")
check(ls["confidence"] == "inferred", "推测一律标 inferred")
check(abs(ls["expected"] - 4.0) < 1e-9 and abs(ls["actual"] - 4.0) < 1e-9, "两个比值都给出来")

r8 = check_relations({"learning_rate": "1e-4", "batch_size": "256"},
                     {"learning_rate": "9e-4", "batch_size": "1024"})
ls2 = {x["rid"]: x for x in r8.relations}["lr_scaling"]
check(ls2["verdict"] == "info", "比值不吻合 -> 明说线性缩放解释不成立，不硬套")
check(ls2["verdict"] != "hint", "不吻合就不给 hint")

r9 = check_relations({"learning_rate": "1e-4", "batch_size": "256"},
                     {"learning_rate": "1e-4", "batch_size": "256"})
ls3 = {x["rid"]: x for x in r9.relations}["lr_scaling"]
check(ls3["verdict"] == "skipped", "两侧相同 -> 无缩放可推演，如实未算")

# ═══════════════════════════════════════ 7. review_numbers 汇总

sec("7. review_numbers —— 逐条结论复核")

findings = [
    {"param_key": "learning_rate", "display_name": "Learning Rate",
     "paper": "1e-4", "code": "1e-1"},          # 差 10^3 -> 量纲提示
    {"param_key": "batch_size", "display_name": "Batch Size",
     "paper": "256", "code": "256"},            # 正常
]
d = review_numbers(findings=findings, paper_params=PV, code_params=CV)
check(isinstance(d, dict) and "relations" in d, "返回结构与 ComputeReport.to_dict 同构")
check(len(d["magnitudes"]) == 1, "只对差 10^3 的那条给量纲提示")
check(d["magnitudes"][0]["param_key"] == "learning_rate", "提示挂在正确的参数上")
check(d["flagged"] >= 1, "提示计入 flagged")

f_bad = [{"param_key": "learning_rate", "display_name": "Learning Rate",
          "paper": "5.0", "code": "1e-4"}]   # 5.0 超出 learning_rate 上界 1.0
d2 = review_numbers(findings=f_bad)
check(len(d2["plausibility"]) == 1, "超出合理区间的取值被抓进 plausibility")
check(d2["plausibility"][0]["side"] == "paper", "标明是哪一侧不合理")
check("合理区间" in d2["plausibility"][0]["note"], "理由里带区间（可核验）")

d3 = review_numbers(findings=[{"param_key": "optimizer", "display_name": "Optimizer",
                               "paper": "AdamW", "code": "Adam"}])
check(d3["magnitudes"] == [] and d3["plausibility"] == [],
      "非数值参数不产生任何提示（不制造噪声）")

d4 = review_numbers()
check(d4["checked"] == 0 and any("未算" in n for n in d4["notes"]),
      "空输入 -> 如实记未算并说明，不用默认值补数")

check(isinstance(ComputeReport().to_dict(), dict), "ComputeReport 可序列化")
md = check_relations(PV, CV).to_markdown()
check("不执行任何训练" in md and "不运行用户代码" in md,
      "markdown 写明「不执行训练 / 不运行用户代码」这条边界")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段五「算」验证: %d/%d OK" % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_compute.py)")
print("=" * 62)
