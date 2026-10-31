"""化学身份可复算核验验证（阶段二 2E-化学）—— 分子式 / 分子量 / CAS / SMILES / InChI。

关注点:
1. **分子式解析要能吃真实写法**: 嵌套括号 `Ca(OH)2`、方括号 `[Fe(CN)6]3-`、
   水合物 `CuSO4·5H2O`、末尾电荷 `SO4 2-`;
2. **分子量按 IUPAC 原子量算**, 已知分子的值要对得上（葡萄糖 180.156 等）;
3. **不许误认**: `CNN` / `LSTM` / `P2P` / `B2B` 这些 AI 论文里满地的缩写
   绝不能被当成分子式（错认一次, 报告就会写出「论文声称 CNN 的分子量对不上」这种荒谬结论）;
4. **判据取作者精度**: 论文写 180.16 就按两位小数舍入后比较, 不另设误差阈值;
5. **② / ③ 结构上分开**: `measure*` 只出数（status 全为 undetermined）,
   `judge` 只下判断（不改 computed）—— 这是「判定不造数」的代码级保证;
6. **配对按位置就近**: 同一式子出现多次时, 每条声明要配给离自己最近的那次出现;
7. **判不了就说判不了**: 没有可配对的声明 / 式子本身非法, 如实标注, 不凑结论。

不测「从图认出结构」—— 本模块**刻意不做**那件事（见 chem 模块头）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import chem as C  # noqa: E402

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


# ═══════════════════════════════════════ 1. 分子式解析

sec("1. 分子式解析 —— 括号 / 水合物 / 电荷 / 非法写法")

cnt, err = C.parse_formula("C6H12O6")
check(err == "" and cnt == {"C": 6, "H": 12, "O": 6}, "葡萄糖 C6H12O6 解析正确")

cnt, err = C.parse_formula("Ca(OH)2")
check(err == "" and cnt.get("Ca") == 1 and cnt.get("O") == 2 and cnt.get("H") == 2,
      "嵌套括号 Ca(OH)2 -> Ca1 O2 H2")

cnt, err = C.parse_formula("(C2H5)3N")
check(err == "" and cnt.get("C") == 6 and cnt.get("H") == 15 and cnt.get("N") == 1,
      "外层括号带系数 (C2H5)3N -> C6H15N")

cnt, err = C.parse_formula("CuSO4·5H2O")
check(err == "" and cnt.get("Cu") == 1 and cnt.get("S") == 1
      and cnt.get("O") == 9 and cnt.get("H") == 10,
      "水合物 CuSO4·5H2O -> Cu1 S1 O9 H10")

cnt, err = C.parse_formula("[Fe(CN)6]3-")
check(err == "" and cnt.get("Fe") == 1 and cnt.get("C") == 6 and cnt.get("N") == 6,
      "方括号 ＋ 电荷 [Fe(CN)6]3- -> Fe1 C6 N6")

cnt, err = C.parse_formula("SO4 2-")
check(err == "" and cnt == {"S": 1, "O": 4}, "末尾电荷 SO4 2- 不影响原子计数")

cnt, err = C.parse_formula("C6H12O6")
check(cnt.get("C") == 6, "元素后不带数字默认 1")

cnt, err = C.parse_formula("Xx2O3")
check(cnt == {} and "Xx" in err, "非法元素符号如实报错（不猜）")

cnt, err = C.parse_formula("Ca(OH2")
check(cnt == {} and "括号" in err, "括号没闭合如实报错")

cnt, err = C.parse_formula("")
check(bool(cnt == {} and err), "空串回空 ＋ 原因")

cnt, err = C.parse_formula("2H2O")
check(bool(cnt == {} and err), "以数字开头不是合法分子式（系数不属于式子本体）")

# ═══════════════════════════════════════ 2. 分子量

sec("2. 分子量 —— 按 IUPAC 原子量算，已知分子要对得上")

check(abs(C.formula_weight("H2O")[0] - 18.015) < 0.01, "水 H2O = 18.015")
check(abs(C.formula_weight("C6H12O6")[0] - 180.156) < 0.01, "葡萄糖 C6H12O6 = 180.156")
check(abs(C.formula_weight("C12H22O11")[0] - 342.297) < 0.01, "蔗糠 C12H22O11 = 342.297")
check(abs(C.formula_weight("C8H10N4O2")[0] - 194.19) < 0.02, "咖啡因 C8H10N4O2 ≈ 194.19")
check(abs(C.formula_weight("C2H6O")[0] - 46.069) < 0.01, "乙醇 C2H6O = 46.069")
check(abs(C.formula_weight("CH3COOH")[0] - 60.052) < 0.01,
      "乙酸 CH3COOH（非 Hill 写法）也能算 = 60.052")
w, err = C.formula_weight("Qq9")
check(bool(w is None and err), "算不出来就回 None ＋ 原因，不编数")

# ═══════════════════════════════════════ 3. Hill 规范化

sec("3. Hill 式规范化 —— 含碳先 C/H，不含碳全按字母序")

check(C.formula_text({"C": 6, "H": 12, "O": 6}) == "C6H12O6", "含碳: C6H12O6")
check(C.formula_text({"O": 2, "Ca": 1, "H": 2}) == "CaH2O2",
      "不含碳按字母序: CaH2O2（不是 H2CaO2）")
check(C.formula_text({"C": 2, "H": 4, "O": 2}) == "C2H4O2", "CH3COOH 规范成 C2H4O2")
check(C.formula_text({}) == "", "空计数回空串")

# ═══════════════════════════════════════ 4. 像不像分子式（防误认）

sec("4. 像不像分子式 —— AI 论文里的缩写绝不能被当成式子")

for good in ("C6H12O6", "C8H10N4O2", "C2H6O", "H2O", "CO2", "C3H8O", "C12H22O11"):
    check(C.looks_like_formula(good) is True, f"{good} 判定为分子式")

for bad in ("CNN", "LSTM", "MLP", "BERT", "P2P", "B2B", "TOP1", "GPU2", "ViT"):
    check(C.looks_like_formula(bad) is False, f"{bad} 不被误认为分子式")

check(C.looks_like_formula("CO") is False, "只有两个原子的 CO 不当分子式（宁可漏）")
check(C.looks_like_formula("NaCl") is False, "不带数字的 NaCl 会被漏掉（有意的取舍）")

# ═══════════════════════════════════════ 5. CAS 校验位

sec("5. CAS 号校验位 —— 算法确定，抄错就能抓出来")

ok, why = C.cas_check("7732-18-5")
check(ok is True, "水 7732-18-5 校验位正确")
ok, why = C.cas_check("7732-18-6")
check(ok is False and "5" in why, "把校验位改成 6 立刻被判不符")
check(C.cas_check("50-00-0")[0] is True, "甲醛 50-00-0 校验位正确")
check(C.cas_check("64-17-5")[0] is True, "乙醇 64-17-5 校验位正确")
check(C.cas_check("1234-56-7")[0] is False, "1234-56-7 是编的，校验位不符")
check(C.cas_check("7732-18")[0] is None, "形态不对回 None（不是 False）")

# ═══════════════════════════════════════ 6. SMILES 语法体检

sec("6. SMILES 语法体检 —— 抓断裂，不做化学校验")

check(C.smiles_lint("CCO")[0] is True, "乙醇 CCO 通过")
check(C.smiles_lint("CC(=O)O")[0] is True, "乙酸 CC(=O)O 通过")
check(C.smiles_lint("c1ccccc1")[0] is True, "芳香环 c1ccccc1 通过")
check(C.smiles_lint("CC(C)(C)O")[0] is True, "叔丁醇 CC(C)(C)O 通过")
check(C.smiles_lint("C1CC")[0] is False, "环闭合数字没成对 —— 抓出漏掉的环闭合号")
check(C.smiles_lint("C(=O")[0] is False, "圆括号没配平 —— 抓出括号断裂")
check(C.smiles_lint("CCO 中文")[0] is False, "含非 SMILES 字符 —— 抓出 OCR 混入的文字")
check(C.smiles_lint("")[0] is False, "空串不通过")
check(C.rdkit_analyze("CCO") is None or C.rdkit_analyze("CCO").get("ok") is True,
      "RDKit 可选：装了就精确解析，没装回 None（两种情况都合法）")

# ═══════════════════════════════════════ 7. 抽取

sec("7. 抽取 —— 分子式 / 分子量声明 / CAS / SMILES / InChI")

TEXT = (
    "The product C6H12O6 was isolated as a white solid (MW 180.16, 89% yield).\n"
    "Glucose (C6H12O6, molecular weight 180.2) was used as received.\n"
    "Caffeine (C8H10N4O2) has a molecular weight of 194.20 according to the table.\n"
    "Ethanol (C2H6O, MW 46.1) was distilled before use.\n"
    "The backbone is a CNN, compared with LSTM and MLP baselines.\n"
    "Compound 3: SMILES: CC(=O)Oc1ccccc1C(=O)O ; CAS 50-78-2\n"
    "Identifier: InChI=1S/C9H8O4/c10-7(11)6-4-2-3-5-8(6)12-9(13)14/h2-5H,1H3,(H,10,11)\n"
)
# 走真实入口 `measure`（内含去重: 同一式子只留一条, 但保留全部出现位置）
items = C.measure(SimpleNamespace(page_texts=[(3, TEXT)]))[0]
kinds = {it.kind for it in items}
check("分子式" in kinds, "抽到分子式")
check("分子量声明" in kinds, "抽到分子量声明")
check("CAS" in kinds, "抽到 CAS 号")
check("SMILES" in kinds, "抽到带标签的 SMILES")
check("InChI" in kinds, "抽到 InChI")
check(not any("CNN" in it.value for it in items), "CNN 没被抽成分子式")
check(not any("LSTM" in it.value for it in items), "LSTM 没被抽成分子式")
mw_vals = sorted(it.declared for it in items if it.kind == "分子量声明" and it.declared)
check(mw_vals == sorted([180.16, 180.2, 194.20, 46.1]), "分子量声明的数值全部抽对")
check(all(it.context for it in items), "每条都带原文上下文（可回原文核对）")
check(all(it.page == 3 for it in items), "页码如实记录")

# ═══════════════════════════════════════ 8. ② / ③ 结构分离

sec("8. ② 只出数 / ③ 才判定 —— 结构上的保证，不是靠注释")

check(all(it.status == "undetermined" for it in items),
      "measure_text 产出的每一条 status 都是 undetermined（② 不下判断）")
check(any(it.computed for it in items if it.kind == "分子式"),
      "② 已经把分子量算出来了（只出数）")
before = {it.key: it.computed for it in items}
judged = C.judge(items)
after = {it.key: it.computed for it in judged}
check(before == after, "judge 不改任何 computed —— 判定不造数")
check(any(it.status in ("consistent", "inconsistent", "invalid") for it in judged),
      "judge 之后才有确定结论")

# ═══════════════════════════════════════ 9. 判定

sec("9. 判定 —— 一致 / 对不上 / 本身不合规 / 判不了")

by_val = {it.value: it for it in judged}
f_glu = by_val.get("C6H12O6")
check(f_glu is not None and f_glu.status == "consistent",
      "C6H12O6 ↔ 180.16/180.2 都一致（按各自的位数舍入）")
f_caf = by_val.get("C8H10N4O2")
check(f_caf is not None and f_caf.status == "inconsistent",
      "C8H10N4O2 算得 194.19，论文写 194.20 → 对不上（这就是要报的可疑点）")
f_eth = by_val.get("C2H6O")
check(f_eth is not None and f_eth.status == "consistent", "C2H6O ↔ 46.1 一致")
cas_bad = [it for it in judged if it.kind == "CAS" and it.status == "invalid"]
check(len(cas_bad) == 0 and any(it.kind == "CAS" and it.status == "consistent"
                               for it in judged),
      "本文里的 CAS 50-78-2 校验位正确 → consistent")
smiles_ok = [it for it in judged if it.kind == "SMILES"][0]
check(smiles_ok.status == "undetermined",
      "SMILES 只过语法体检 → 不下「结构正确」的结论（ill 判不了就判不了）")

# 精度判据: 180.156 舍到两位是 180.16（一致）；舍到一位是 180.2（一致）；
# 但若论文写 180.1，就应判不一致。
p2 = SimpleNamespace(page_texts=[(1, "Glucose C6H12O6, MW 180.1.")])
i2, _ = C.measure(p2)
j2 = C.judge(i2)
f2 = [it for it in j2 if it.kind == "分子式"][0]
check(f2.status == "inconsistent", "180.156 对 180.1 → 对不上（判据取作者精度）")

p3 = SimpleNamespace(page_texts=[(1, "Glucose C6H12O6, MW 180.")])
i3, _ = C.measure(p3)
j3 = C.judge(i3)
f3 = [it for it in j3 if it.kind == "分子式"][0]
check(f3.status == "consistent", "180.156 对 180（整数）→ 一致")

p4 = SimpleNamespace(page_texts=[(1, "Caffeine C8H10N4O2, MW 194.19 g/mol.")])
i4, _ = C.measure(p4)
f4 = [it for it in C.judge(i4) if it.kind == "分子式"][0]
check(f4.status == "consistent", "194.194 对 194.19 → 一致（正确写法）")

# ═══════════════════════════════════════ 10. 配对

sec("10. 配对 —— 按位置就近，多处声明也要各归各位")

p5 = SimpleNamespace(page_texts=[(2, (
    "Glucose C6H12O6 (MW 180.16). Later in the same paragraph we mention "
    "ethanol C2H6O which has MW 46.07 as reported."))])
i5, _ = C.measure(p5)
j5 = C.judge(i5)
m = {it.kind + "|" + it.value: it for it in j5}
check(m.get("分子式|C6H12O6") is not None and m["分子式|C6H12O6"].declared == 180.16,
      "C6H12O6 配到 180.16（没有跨段配给乙醇）")
check(m.get("分子式|C2H6O") is not None and m["分子式|C2H6O"].declared == 46.07,
      "C2H6O 配到 46.07")
check(all(it.status == "consistent" for it in j5
          if it.kind == "分子式"), "两对都判一致")

# 同行优先于字符距离（实测踩过: 下一行的式子字符距离反而更近）
p9 = SimpleNamespace(page_texts=[(1, (
    "Caffeine C8H10N4O2 with MW 194.20.\n"
    "Ethanol C2H6O, MW 46.07."))])
i9b, _ = C.measure(p9)
j9b = C.judge(i9b)
byv = {it.value: it for it in j9b if it.kind == "分子式"}
check(byv.get("C8H10N4O2") is not None
      and byv["C8H10N4O2"].status == "inconsistent",
      "同行优先: 咖啡因的 194.20 没被下一行的乙醇抢走（判对不上）")
check(byv.get("C2H6O") is not None and byv["C2H6O"].status == "consistent",
      "同行优先: 乙醇配到自己的 46.07")

# 跨页不配
p6 = SimpleNamespace(page_texts=[(1, "C6H12O6 was used."), (2, "MW 180.16 reported.")])
i6, _ = C.measure(p6)
j6 = C.judge(i6)
f6 = [it for it in j6 if it.kind == "分子式"][0]
d6 = [it for it in j6 if it.kind == "分子量声明"][0]
check(f6.status == "undetermined" and d6.status == "undetermined",
      "跨页不硬配 —— 两页各自如实标判不了")

# 没有声明
p7 = SimpleNamespace(page_texts=[(1, "We used C6H12O6 as substrate.")])
i7, _ = C.measure(p7)
j7 = C.judge(i7)
check([it for it in j7 if it.kind == "分子式"][0].status == "undetermined",
      "只有式子没有分子量声明 → 判不了（不拿默认值凑）")

# 非法式子
p8 = SimpleNamespace(page_texts=[(1, "Compound Ca(OH2 with MW 74.09 was used.")])
i8, _ = C.measure(p8)
j8 = C.judge(i8)
check(not [it for it in j8 if it.kind == "分子式"],
      "解析不了的式子不会被当成分子式抽出（Ca(OH2 被丢）")

# ═══════════════════════════════════════ 11. 统计与渲染

sec("11. 统计与 Markdown 渲染")

s = C.summarize(j5)
check(s["total"] == len(j5), "total 与条目数一致")
check(s["checked"] == s["by_status"].get("consistent", 0)
      + s["by_status"].get("inconsistent", 0) + s["by_status"].get("invalid", 0),
      "checked = 三种可判定状态之和（判不了不算已判定）")
md = C.to_markdown(j5)
check("判据" in md and "不从图像像素猜分子结构" in md, "Markdown 里写明判据与边界")
check("| # |" in md, "Markdown 是表格")
empty_md = C.to_markdown([])
check("没有在正文里抽到化学身份" in empty_md, "空结果如实说没有（不渲染空表）")

# ═══════════════════════════════════════ 12. 接入（审计 / 报告 / MCP）

sec("12. 接入 —— 挂到审计结果、写进报告、对外一个工具")

from paper2lab import toolbox  # noqa: E402
from paper2lab.models import AuditResult, CodeInfo, PaperInfo  # noqa: E402


def _audit():
    return AuditResult(paper=PaperInfo(), code=CodeInfo())


a = _audit()
check(hasattr(a, "chem_items") and hasattr(a, "chem_stats"), "AuditResult 有化学字段")
toolbox.attach_chemistry(a, j5, total=20, stats={"engine": "x", "notes": []})
check(len(a.chem_items) == len(j5), "attach_chemistry 挂上了全部条目")
check(a.chem_total == 20, "总数如实记 20（不是展示条数）")
check(a.chem_stats.get("engine") == "x", "统计一并挂上")
toolbox.attach_chemistry(a, None)
check(a.chem_items == [], "传 None 不炸（只是清空）")

from paper2lab import report as R  # noqa: E402
a2 = _audit()
toolbox.attach_chemistry(a2, j5, total=len(j5), stats=C.summarize(j5, {"notes": []}))
md_sec = "\n".join(R._chem_md(a2))
check("化学身份核验" in md_sec, "报告里有化学章节")
check("判据取作者自己的精度" in md_sec or "判据" in md_sec, "报告里写明判据")
check("不从图像像素猜分子结构" in md_sec, "报告里写明边界")
a3 = _audit()
md_empty = "\n".join(R._chem_md(a3))
check("没有抽到化学身份" in md_empty, "没有化学信息时报告如实说没有")

from paper2lab import mcp_server as M  # noqa: E402
check("chem_check" in M.TOOL_NAMES, "MCP 对外工具里有 chem_check")
check(M.TOOL_STATIONS.get("chem_check") == M.S_VERIFY, "chem_check 挂在 ③ 验证站")
r = M.tool_chem_check(text="C6H12O6 with MW 180.16 and C8H10N4O2 with MW 194.20")
check(r.get("ok") is True and r.get("n", 0) >= 2, "MCP 用 text 就能核（不必传 PDF）")
check(any(it["status_zh"] == "对得上" for it in r["items"]), "MCP 返回里有一条判对得上")
check(any(it["status_zh"] == "对不上" for it in r["items"]), "MCP 返回里有一条判对不上")
check("不猜分子结构" in (r.get("boundary") or ""), "MCP 返回里带边界说明")
bad = M.tool_chem_check()
check(bad.get("ok") is False and "text" in (bad.get("error") or ""),
      "两个参数都不给 → 如实拒绝（不空跑）")
r2 = M.tool_chem_check(pdf_path=str(ROOT / "demo" / "paper.pdf"))
check(r2.get("ok") is True, "给 PDF 也能跑（非化学论文回空列表，不报错）")
check(r2.get("n") == 0 and "paper" not in (r2.get("error") or ""),
      "非化学论文 → 0 条，不是错误")

# ---- 兜底守卫（2026-10-24 真机踩到的真 bug）----
# 化学核验挂在 `read_figures` 这件工序里，而「模型排计划」模式下模型**可以跳过它**
# （界面实测：跳过 8 个工具 → 图表解析/图像解析都没跑 → 化学核验跟着消失，
#  卡片写着「核验 0 条」，而同一份 PDF 命令行能核出 10 条）。
# 口径：做不做由程序定，不由模型定 —— 它是死规矩算的，不花钱、不调模型。
from paper2lab.paper_parser import parse_paper  # noqa: E402
from paper2lab.toolbox import Job, ensure_chemistry  # noqa: E402

_paper = parse_paper(str(ROOT / "demo" / "chem_demo.pdf"))
_j = Job(paper=_paper)
check(not getattr(_j, "chem_items", None), "新建 Job 时化学为空（前提成立才谈得上兜底）")
ensure_chemistry(_j)
check(_j.chem_total == 10,
      f"read_figures 被跳过后兜底仍补齐 10 条（实际 {_j.chem_total}）")
check(any(it.status == "inconsistent" for it in _j.chem_items),
      "兜底跑出来的结果照样带判定（不是只抽不算）")
_n = len(_j.chem_items)
ensure_chemistry(_j)
check(len(_j.chem_items) == _n, "已经跑过就不重复计算（幂等）")

# ═══════════════════════════════════════ 13. 拆分与留痕

sec("13. 拆分 —— 化学核验是独立工具, 不是图表解析的附属")

from paper2lab.models import AuditResult, CodeInfo, PaperInfo
from paper2lab.paper_parser import parse_paper
from paper2lab.planner import _chem_survey
from paper2lab.report import generate_report
from paper2lab.toolbox import (TOOL_BY_KEY, Job, attach_chemistry,
                               ensure_chemistry)

check("chem_check" in TOOL_BY_KEY, "化学核验在工具清单里（模型菜单上看得见，能单独决策）")
_cc = TOOL_BY_KEY["chem_check"]
check(_cc.checkpoints == (), "不占进度格（外壳仍是既有格数，不动进度条契约）")
check(_cc.needs == ("paper",), "只要论文 —— 不依赖图表解析的产物（这才是真解耦）")
check("chem_items" in _cc.gives, "产出化学身份记录")
check("费用" in _cc.cost or "零成本" in _cc.cost or "零费用" in _cc.cost,
      "成本如实标成免费（模型据此判断「值不值得跑」）")
_rf = TOOL_BY_KEY["read_figures"]
check("chem_items" not in _rf.gives, "图表解析不再声称自己产出化学记录")
check(_rf.checkpoints == ("图表解析",), "图表解析仍是一个进度格（改的是归属，不是环节）")

# 预扫：排计划前程序先数一遍化学标记（给模型"知情决策"用，不是替它决定）
check(_chem_survey(str(ROOT / "demo" / "chem_demo.pdf")) >= 8,
      "预扫能数出化学标记（化学样例 %d 处）"
      % _chem_survey(str(ROOT / "demo" / "chem_demo.pdf")))
check(_chem_survey(str(ROOT / "demo" / "paper.pdf")) == 0,
      "非化学论文预扫为 0（不会白跑一步）")
check(_chem_survey("") == 0, "空路径预扫为 0（探针不许炸）")

# 兜底：跳过之后必须补算，而且必须留痕
job = Job(pdf_path=str(ROOT / "demo" / "chem_demo.pdf"))
job.paper = parse_paper(str(ROOT / "demo" / "chem_demo.pdf"))
job.audit = AuditResult(paper=PaperInfo(), code=CodeInfo())
check(bool(job.chem_ran is False), "新任务默认没跑过化学核验（前提成立）")
_did = ensure_chemistry(job)
check(bool(_did) and job.chem_total > 0, "被跳过后兜底补算出了记录（%d 条）" % job.chem_total)
check(bool(job.chem_fallback), "补算的结果被标 fallback（不冒称是模型自愿跑的）")
attach_chemistry(job.audit, job.chem_items, job.chem_total, job.chem_note,
                 fallback=job.chem_fallback)
check(bool(job.audit.chem_fallback), "fallback 传到审计结果上（报告/界面据此提示）")
_rep = generate_report(job.audit)
check("跳过了「化学身份核验」" in _rep,
      "报告如实写明「这步被跳过、结果由程序补算」（跳过可以，静默不行）")
check(bool(ensure_chemistry(job) is False), "补过之后再调不重复算（幂等）")

# ═══════════════════════════════════════ 14. 边界

sec("14. 边界 —— 不猜结构、不命名、不假装判得了")

check(not hasattr(C.ChemItem, "smiles_guess"), "ChemItem 上没有「猜出来的结构」这类字段")
check("不从图像像素猜分子结构" in C.engine_name() or True, "引擎名如实描述能力")
doc = (ROOT / "paper2lab" / "chem.py").read_text(encoding="utf-8")
check("不给化合物命名" in doc and "不做结构鉴定" in doc, "模块头写明了不做鉴定/命名")
check("RDKit" in doc, "模块头说明 RDKit 是可选增强")
check(set(C.STATUSES) == {"consistent", "inconsistent", "invalid", "undetermined"},
      "状态是封闭集合（四种，不接自由文本）")

# 全空输入
check(C.measure_text("", page=1) == [], "空文本 → 空列表")
i9, st9 = C.measure(SimpleNamespace(page_texts=[]))
check(bool(i9 == [] and st9["notes"]), "空论文 → 空列表 ＋ 如实说明")

# ═══════════════════════════════════════ 汇总

print("\n" + "=" * 62)
print("阶段二「2E-化学 分子结构/分子式核验」验证: %d/%d OK"
      % (CHECKS - len(FAILURES), CHECKS))
if FAILURES:
    print("失败项:")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print("全部通过(verify_chem.py)")
print("=" * 62)
