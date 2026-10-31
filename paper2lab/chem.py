"""化学身份的可复算核验（阶段二 2E-化学扩展）—— 分子式 / 分子量 / CAS / SMILES / InChI。

**为什么不做"看图认出这是什么分子"**:
从像素猜结构（原子、键、立体化学）是幻觉重灾区 —— 看错一个取代基, 整条结论就废,
而本系统的底线是"每个结论都能回原始材料核对"。所以这里的做法是另一条路:

    论文自己写了什么化学身份 → 用化学规则**自己算一遍** → 对不上就是真错。

四条**确定性**核验（全部可在本地零成本完成）:
1. **分子式配平** —— `C6H12O6` / `Ca(OH)2` / `CuSO4·5H2O` / `[Fe(CN)6]3-` 能否解析成元素计数;
   解析不了说明这个式子本身写坏了（OCR 断裂、括号不配对、元素符号非法）。
2. **分子量复算** —— 从分子式按 IUPAC 原子量加总, 与论文声明的分子量对账。
   判据刻意不是"误差小于 x%", 而是**按论文自己给的小数位四舍五入后是否相等**
   （论文写 180.16, 算得 180.156 → 舍到两位是 180.16 → 对得上;
   写 180.1 → 舍到一位是 180.2 → 对不上）。这是"作者声称的精度"决定的, 不是我们定的。
3. **CAS 号校验位** —— CAS 最后一位是校验位, 算法确定, 算错就是抄错。
4. **SMILES 语法** —— 括号/方括号配平、环闭合数字成对、元素符号合法。
   装了 RDKit 就用 RDKit 做精确解析（规范化 + 精确质量）；没装就用内置规则,
   **并如实说"这是语法体检, 不是完整化学校验"**。

三条边界（对外必须讲清）:
1. **不给化合物命名、不做结构鉴定**。只回答"这个式子/这个号/这个质量对不对得上"。
2. **每个结论都带原文上下文**（`context` 是论文里逐字的一段）, 便于人工回原文核对。
3. **判不了就说判不了**（`undetermined`）, 不拿"大概合理"凑结论。

② / ③ 仍然分开（与 `compute.Relation` 同款口径）:
- `measure(paper)` 是 **② 实验站的活** —— 只产出数（分子量、校验位、元素计数）;
- `judge(items)` 是 **③ 验证站的活** —— 只拿已经算出来的数下判断, 不重新算。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------- 原子量（IUPAC 2021 常用值）

#: 只收常用元素; 查不到的元素 -> 该分子式判"算不了", 不瞎猜。
ATOMIC_WEIGHTS: dict = {
    "H": 1.008, "He": 4.0026,
    "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922,
    "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Mo": 95.95,
    "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90,
    "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50,
    "Ho": 164.93, "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97,
    "Hf": 178.49, "Ta": 180.95, "W": 183.84, "Re": 186.21, "Os": 190.23,
    "Ir": 192.22, "Pt": 195.08, "Au": 196.97, "Hg": 200.59, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.98, "Th": 232.04, "U": 238.03,
}

_ELEMENTS = frozenset(ATOMIC_WEIGHTS)
#: 有机 / 生命科学场景里几乎必然出现的元素。用它挡"B2B / P2P"这类缩写误判（见 `looks_like_formula`）。
_ORGANIC_HINT = frozenset("CHONSP")

#: 单个元素下标的上限。真实分子式不会写 `P2005`，那多半是会议名缩写里的年份。
#: 留 200 的余量：已知最大的合法下标是蛋白质/糖类量级（C132H… 一类），远小于 200。
_MAX_SUBSCRIPT = 200

# ---------------------------------------------------------------- 形态常量

KINDS: tuple = ("分子式", "分子量声明", "CAS", "SMILES", "InChI")

#: 展示上限（与 FIGURE_MAX / HYPOTHESIS_MAX 同口径: 总数如实报, 界面只列前 N 条）
CHEM_MAX = 8

#: 判定状态（封闭集合）。`invalid` 是"自身就不合规", 与"两个数对不上"是两回事。
STATUSES: tuple = ("consistent", "inconsistent", "invalid", "undetermined")

STATUS_ZH: dict = {
    "consistent": "对得上",
    "inconsistent": "对不上",
    "invalid": "本身不合规",
    "undetermined": "判不了",
}

# ---------------------------------------------------------------- 抽取用的正则

#: 显式标记的 SMILES（只在有标签处抽 —— 裸 SMILES 和普通单词没法区分, 宁可漏）
_SMILES_LABEL = re.compile(
    r"(?i)\bSMILES?\b\s*[:：=]\s*[\"'\u201c\u2018]?([A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]{2,300})")
#: InChI（带分层, 以 / 分隔; 取到空白为止）
_INCHI = re.compile(r"(?i)\bInChI\s*=\s*1S?/([^\s\"'\u201d\u2019]{6,600})")
#: CAS 号: 2~7 位 - 2 位 - 1 位
_CAS = re.compile(r"(?<![\d-])(\d{2,7}-\d{2}-\d)(?![\d-])")
#: 分子量声明（英文 / 中文 / 常见缩写）
_MW_DECL = re.compile(
    r"(?i)(?:\bMW\b|\bM\.\s?W\.|\bmolecular\s+(?:weight|mass)\b|\bM[wW]\s*=|\bMr\b|"
    r"分子量|摩尔质量|相对分子质量)"
    r"\s*(?:of|is|为|是|=|:|：|≈|~)?\s*"
    r"(\d{1,4}(?:\.(\d{1,4}))?)\s*(?:g\s*/\s*mol|Da|kDa|道尔顿)?")
#: 分子式的"候选片段": 连续的元素符号/数字/括号/水合物点
_FORMULA_CAND = re.compile(
    r"(?<![A-Za-z0-9])"
    r"((?:\(|\[)?(?:[A-Z][a-z]?\d*(?:\(|\[|\)|\])*)+"
    r"(?:[·•]\d*(?:\(|\[)?(?:[A-Z][a-z]?\d*(?:\(|\[|\)|\])*)+)?)"
    r"(?:\d*[+-])?"
    r"(?![A-Za-z0-9])")

_HYDRATE_SEP = "\u00b7\u2022"      # · •

#: 上下文截取长度（两侧各取多少字符, 用于人工回原文核对）
_CTX_PAD = 90

#: 化学语境词 —— **用来挡 AI 论文里的误报**, 不是用来判断论文是不是化学论文。
#:
#: 为什么需要它（2026-10-29 实测 51 份 AI 论文）:
#:   · BERT 参考文献里的会议名 `IWP2005` 被解析成 I+W+P2005, 元素种类 3、含 P,
#:     词法判据全过 → 被当分子式;
#:   · MAE 正文里 `ImageNet-1K` 的缩写 `IN1K` → I+N1+K, 同样全过;
#:   · darknet 参考文献里的日期 `2015-10-2` 形态正好是 CAS 号, 校验位不符 →
#:     被判成「CAS 号无效」, 读起来像在报错。
#: 三者的共同点是**周围没有任何化学词**。所以判据放在"邻近上下文"而不是整篇:
#: 整篇只要出现一个 pH 就放开全部, 太松; 局部 90 字符内没有化学词就不认, 更准。
#: ⚠ 刻意**不**收的词（都是机器学习论文的常用词，收进来等于放开一切）：
#:   `synthesis`（image / speech synthesis）、`solid`（solid results）、
#:   `compound`（EfficientNet 的 compound scaling）、裸 `[uμ]M`（会和 "um" 撞）、
#:   `Mr`（会被当成 "Mr." / 缩写）、裸 `formula`（the formula of attention）。
#: ⚠ `mM` / `nM` 用 `(?-i:...)` **局部关掉大小写忽略** —— 否则会匹配到长度单位
#:   "mm"（毫米）。2026-10-29 实测：whisper 论文靠两个 "mm" 被判成化学语料，
#:   于是把 `FP16` 当成分子式抽了出来。
_CHEM_CTX = re.compile(
    r"(?i)(?:\bmol(?:e|es|ar|ecule|ecules)?\b|\bmmol\b|\b[μu]mol\b|(?-i:\bmM\b)|(?-i:\bnM\b)|"
    r"分子量|分子式|化学式|摩尔|溶液|溶剂|滴定|催化剂|化学合成|"
    r"\bsolvent\b|\baqueous\b|\bsynthesi[sz]ed\b|\bchemical\s+synthesis\b|\breagent\b|"
    r"\bNMR\b|\bHPLC\b|\bspectroscop\w*|\btitration\b|\bcatalyst\b|"
    r"\bchemical\s+compound\b|\bmolecule\b|\bcrystals?\b|\bdissolved\b|\bpH\b|"
    r"\bSMILES\b|\bInChI\b|\bCAS\b|\bwt\s*%\b|\bmolecular\s+(?:weight|mass)\b|"
    r"\bM\.?\s?W\.?(?=[\s,:=]|\d))")
#: ⚠ 刻意**不**收裸词 `formula` —— 机器学习论文会写 "the formula of attention",
#: 收进来等于给误报开后门。只认 `molecular formula` 这类明确搭配。

#: 判定「这篇论文算不算化学语料」的门槛：**整篇出现 ≥2 次**上面的化学词。
#:
#: 为什么非要整篇闸门（2026-10-29 第二次收紧）：
#:   词法规则只能挡住「形态不像」的串，挡不住「形态合法但语义无关」的串。实测 51 份
#:   AI 论文在加了词法规则后仍有 23 条误报，全是这类：
#:     `UCF101`（U 铀 + C 碳 + F 氟）、`CUB200`（C+U+B）、`VOC2012`、`FP16`（F+P）、
#:     `PPO2`（P+P+O）、`P2N`、`C2F` —— 每一个都"能解析成合法分子式"。
#:   它们的共同点是**整篇没有任何化学词**。所以判据必须放在语料层，而不是词法层。
#:
#: 门槛取 2 而不是 1：AI 论文偶尔会提一次 molecule（分子图神经网络那类），
#: 一次不足以说明是化学论文；而真化学论文里 mol / mM / solvent / 分子量 会出现很多次。
#: 代价是"只提一次分子的论文"会被漏掉 —— 与全模块「宁可漏，不要错认」一致。
_CHEM_CORPUS_MIN = 2

#: 短文本门槛。**整篇统计需要足够多的字**：真实论文正文都有几千字符，
#: 而单句 / 摘要片段的调用只有几十字，按"整篇次数"算没有统计意义。
#: 短文本一律**不做语料判断**（当作是化学语境），只影响片段级调用，不影响真实论文。
_SHORT_TEXT_CHARS = 1500

#: 显式标签 —— 出现这些时**不看语料闸门**（标签本身就是化学证据）。
_CHEM_LABELS = re.compile(r"(?i)\bSMILES?\s*[:：=]|\bInChI\s*=\s*1S?/")


def is_chem_corpus(text: str) -> bool:
    """这篇论文算不算化学语料。**只用于决定"要不要做分子式/CAS 核验"**。

    三种情况:
    - 带 SMILES / InChI 显式标签 -> 算（标签本身就是化学证据）;
    - 文本短于 `_SHORT_TEXT_CHARS` -> 算（片段级调用不做语料判断）;
    - 否则看化学词出现次数是否达到 `_CHEM_CORPUS_MIN`。
    """
    t = text or ""
    if _CHEM_LABELS.search(t):
        return True
    if len(t.strip()) < _SHORT_TEXT_CHARS:
        return True
    return len(_CHEM_CTX.findall(t)) >= _CHEM_CORPUS_MIN


# ---------------------------------------------------------------- 分子式解析

def parse_formula(text: str) -> tuple:
    """把分子式解析成 {元素: 原子数} —— 返回 (计数字典, 错误说明)。

    支持: 嵌套括号 `Ca(OH)2`、方括号 `[Fe(CN)6]`、水合物 `CuSO4·5H2O`、
    末尾电荷 `SO4 2-` / `SO4^2-` / `NH4+`（电荷**不计入**原子数, 只用来去掉尾巴）。

    解析不了（括号不配对 / 元素符号非法 / 系数缺失）一律回空字典 + 原因, 不抛异常。
    """
    s = (text or "").strip()
    if not s:
        return {}, "空式子"
    # 去掉电荷尾巴（2- / 2+ / + / - / ^2-）, 电荷不影响原子计数
    s = re.sub(r"\^?\d*[+-]\s*$", "", s).strip()
    if not s:
        return {}, "只有电荷"
    counts: dict = {}

    def _merge(dst: dict, src: dict, mult: int) -> None:
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v * mult

    def _parse(i: int, closer: str) -> tuple:
        local: dict = {}
        while i < len(s):
            ch = s[i]
            if closer and ch == closer:
                return local, i + 1
            if ch in _HYDRATE_SEP:
                i += 1
                # 水合物系数: ·5H2O —— 点号后可能先跟一个整数
                m = re.match(r"\d+", s[i:])
                coef = 1
                if m:
                    coef = int(m.group(0))
                    i += len(m.group(0))
                sub, i = _parse(i, "")
                _merge(local, sub, coef)
                continue
            if ch in "([":
                sub, i = _parse(i + 1, ")" if ch == "(" else "]")
                m = re.match(r"\d+", s[i:])
                mult = int(m.group(0)) if m else 1
                if m:
                    i += len(m.group(0))
                _merge(local, sub, mult)
                continue
            if ch in ")]":
                return local, i + 1
            m = re.match(r"[A-Z][a-z]?", s[i:])
            if not m:
                raise ValueError(f"位置 {i} 出现无法识别的字符 {ch!r}")
            sym = m.group(0)
            if sym not in _ELEMENTS:
                raise ValueError(f"元素符号 {sym} 不在常用元素表内")
            i += len(sym)
            m2 = re.match(r"\d+", s[i:])
            n = int(m2.group(0)) if m2 else 1
            if m2:
                i += len(m2.group(0))
            local[sym] = local.get(sym, 0) + n
        if closer:
            raise ValueError("括号没有闭合")
        return local, i

    try:
        counts, _ = _parse(0, "")
    except ValueError as ex:
        return {}, str(ex)
    if not counts:
        return {}, "没解析出任何元素"
    if sum(counts.values()) <= 0:
        return {}, "原子总数为 0"
    return counts, ""


def formula_weight(formula: str) -> tuple:
    """分子式 -> 相对分子质量。返回 (质量 or None, 说明)。"""
    counts, err = parse_formula(formula)
    if err:
        return None, err
    total = 0.0
    for sym, n in counts.items():
        total += ATOMIC_WEIGHTS[sym] * n
    return round(total, 4), ""


def formula_text(counts: dict) -> str:
    """把元素计数还原成 Hill 式分子式 —— 用于"规范化后对比"。

    Hill 规则: 含碳时先 C 后 H 再按字母序; **不含碳时全部按字母序**
    （`Ca(OH)2` 应写 CaH2O2, 不是 H2CaO2）。
    """
    if not counts:
        return ""
    if "C" in counts:
        order = [s for s in ("C", "H") if counts.get(s)]
        order += sorted(k for k in counts if k not in ("C", "H"))
    else:
        order = sorted(counts)
    out = []
    for sym in order:
        n = counts[sym]
        out.append(sym + ("" if n == 1 else str(n)))
    return "".join(out)


def looks_like_formula(text: str) -> bool:
    """判断一个片段**像不像分子式** —— 这是整个模块最容易误伤的地方。

    五条判据（全过才算）:
    1. 能解析出元素计数;
    2. 元素种类 ≥ 2（挡 `P2P` / `B2B` 这种同元素重复的缩写）;
    3. 原子总数 ≥ 3（挡 `CO` / `NO` 这种两个原子、更像缩写的东西 —— 宁可漏）;
    4. 至少含 C/H/O/N/S/P 之一（化学与生命科学场景的绝大多数分子式都满足）;
    5. **原串里必须有数字**（挡 `CNN` / `LSTM` / `MLP` 这类全大写缩写 ——
       AI 论文里这些词满地都是, 而真实分子式几乎都带下标数字）。

    外加两条**防 AI 论文误报**的硬规则（2026-10-29 加, 由 51 份真实论文逼出来的）:
    6. **单元素下标 ≤ 200**：`IWP2005` 会解析成 I·W·P₂₀₀₅（P 下标 2005）,
       分子式里不可能有这种下标（聚合物写 `(C2H4)n`, 不写具体大数）→ 挡掉。
    7. **不能是「三种以上元素各一个」的全大写短串**：`IN1K`（ImageNet-1K 的缩写）
       会解析成 I·N·K 各 1 个 —— 真实分子式里三元素各一个且带数字的几乎没有
       （`HBrO`/`HNO` 既不带数字也不全大写）→ 挡掉。

    代价: `NaCl` / `HCl` 这类不带数字的式子会被漏掉。**宁可漏, 不要错认** ——
    错认一次就会在报告里出现"论文声称 NaCl 的分子量对不上"这种荒谬结论。
    """
    s = (text or "").strip()
    if not any(ch.isdigit() for ch in s):
        return False
    counts, err = parse_formula(s)
    if err or not counts:
        return False
    if len(counts) < 2:
        return False
    if sum(counts.values()) < 3:
        return False
    if not (set(counts) & _ORGANIC_HINT):
        return False
    # 规则 6: 下标过大 —— 缩写里常带年份/规模数（IWP2005 / CVPR2020）
    if max(counts.values()) > _MAX_SUBSCRIPT:
        return False
    # 规则 7: 三种以上元素、每种都恰好 1 个、且全大写短串 —— 是缩写的形态
    if (len(counts) >= 3 and all(n == 1 for n in counts.values())
            and s.isupper() and len(s) <= 6):
        return False
    # 规则 8: **只有两种单字母元素、其中一个下标 ≥ 10** —— 是精度/型号缩写。
    #   `FP16`（F+P16）、`FP32`、`BF16` 都长这样；真实双元素分子式的下标都是个位数
    #   （`CO2` / `NO2` / `SO2` / `P2O5`），所以不会误伤。
    #   代价：`B10H14`（硼烷）这类会被漏掉 —— 与全模块「宁可漏，不要错认」一致。
    if (len(counts) == 2 and all(len(sym) == 1 for sym in counts)
            and max(counts.values()) >= 10):
        return False
    return True


# ---------------------------------------------------------------- CAS 校验位

def cas_check(cas: str) -> tuple:
    """CAS 号校验位。返回 (是否正确 or None, 说明)。

    算法（公开确定）: 去掉两个连字符后, 从**右往左**依次乘 1,2,3…, 求和后 mod 10,
    结果应等于校验位。例: 7732-18-5 -> 8*1+1*2+2*3+3*4+7*5+7*6 = 105 -> 5 ✓
    """
    s = (cas or "").strip()
    m = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", s)
    if not m:
        return None, "不是 CAS 号的标准形态（应为 2~7 位-2 位-1 位）"
    head, mid, check = m.group(1), m.group(2), m.group(3)
    digits = (head + mid)[::-1]
    total = sum((i + 1) * int(d) for i, d in enumerate(digits))
    expect = total % 10
    if expect == int(check):
        return True, f"校验位正确（按算法应得 {expect}）"
    return False, f"校验位不符：算法应得 {expect}，原文写的是 {check}"


# ---------------------------------------------------------------- SMILES 语法体检

_SMILES_OK_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]+$")
_BRACKET_ATOM = re.compile(r"\[([^\]]+)\]")


def smiles_lint(smiles: str) -> tuple:
    """SMILES 语法体检（**规则版, 不是完整化学校验**）。返回 (是否通过, 说明)。

    检查四件事: 字符集、圆括号/方括号配平、环闭合数字成对、方括号外的元素符号合法。
    这些能抓住"OCR 把一个键读丢""括号断了"这类最常见的坏数据;
    但它**不校验价态与芳香性** —— 那是化学引擎的活, 没装 RDKit 就如实说不校验。
    """
    s = (smiles or "").strip()
    if not s:
        return False, "空串"
    if len(s) > 300:
        return False, "过长（>300 字符），疑似把整段文字当成 SMILES"
    if not _SMILES_OK_CHARS.match(s):
        bad = sorted(set(ch for ch in s if not re.match(r"[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.%]", ch)))
        return False, f"含 SMILES 不允许的字符: {''.join(bad)[:12]}"

    # 圆括号配平（方括号单独配平）
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False, "圆括号右括号多了"
    if depth != 0:
        return False, f"圆括号没配平（差 {abs(depth)} 个）"
    if s.count("[") != s.count("]"):
        return False, "方括号没配平"

    # 环闭合: 数字必须成对出现（每个数字出现偶数次）
    ring = {}
    for ch in s:
        if ch.isdigit():
            ring[ch] = ring.get(ch, 0) + 1
    odd = [d for d, n in ring.items() if n % 2]
    if odd:
        return False, f"环闭合数字没成对（{'、'.join(sorted(odd))}），可能是漏了一个环闭合号"

    # 方括号外的元素符号: 大写 + 可选小写; 不在元素表内 -> 提醒（芳香小写 a/c/n 等也允许）
    arom = set("acnsope")
    for i, ch in enumerate(s):
        if not ch.isupper():
            continue
        sym = ch
        if i + 1 < len(s) and s[i + 1].islower():
            sym += s[i + 1]
        if sym in _ELEMENTS:
            continue
        if ch.lower() in arom and sym not in _ELEMENTS:
            # `Cl` / `Br` 已在上一步命中; 这里只剩单个大写字母
            if ch not in arom.union(set("COSNPFIBK")):
                return False, f"出现不像元素符号的组合 {sym!r}"
            continue
        if ch.lower() in arom:
            continue
        return False, f"元素符号 {sym!r} 不在常用元素表内"
    return True, "语法体检通过（括号/环闭合/元素符号）"


def rdkit_analyze(smiles: str) -> Optional[dict]:
    """RDKit 可用时给出精确结果（规范化 SMILES / 分子式 / 精确质量）; 不可用回 None。

    **刻意做成可选**: 本系统线上是零成本部署, 不装 RDKit 也必须能跑完整条链路,
    只是精度降一档（规则版语法体检 + 不算 SMILES 的分子量）, 并在结果里如实标注。
    """
    try:
        from rdkit import Chem, RDLogger                   # noqa: F401
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except Exception:
        return None
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"ok": False, "reason": "RDKit 无法解析（化学语法不成立）"}
        return {
            "ok": True,
            "canonical": Chem.MolToSmiles(mol),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "exact_mass": round(Descriptors.ExactMolWt(mol), 4),
            "mol_weight": round(Descriptors.MolWt(mol), 4),
        }
    except Exception as ex:                                  # noqa: BLE001
        return {"ok": False, "reason": f"RDKit 解析出错: {type(ex).__name__}"}


def engine_name() -> str:
    """本模块用的引擎 —— 如实写出来（装了 RDKit 就说明, 没装也说明）。"""
    return ("化学规则引擎（内置 IUPAC 原子量 + CAS 校验位 + SMILES 语法体检）"
            + ("＋RDKit 精确解析" if rdkit_analyze("CCO") else "（未安装 RDKit，SMILES 只做语法体检）"))


# ---------------------------------------------------------------- 数据结构

@dataclass
class ChemItem:
    """一条**化学身份记录** —— 本模块的产物（界面 / 报告 / 留痕都只读这里）。

    `computed` 是 ② 站算出来的数（分子量 / 校验位结论 / 元素计数）;
    `status` 是 ③ 站下的判断 —— **两个字段分开存**, 结构上保证"判定不造数"。
    """
    key: str
    kind: str                       # KINDS 之一
    value: str                      # 原文逐字
    page: int = 0
    context: str = ""               # 原文上下文（人工回原文核对用）
    computed: str = ""               # ② 算出来的数（分子量等）
    computed_value: Optional[float] = None   # 数值形态（便于排序/比较）
    declared: Optional[float] = None         # 论文声明的数（分子量声明才有）
    declared_digits: int = 0                 # 声明给了几位小数（判据要用）
    status: str = "undetermined"             # STATUSES 之一
    detail: str = ""
    note: str = ""
    rank: int = 0
    pos_hint: int = 0                        # 本页内的字符位置（配对用, 不对外展示）
    #: 去重后保留**全部出现位置** —— 同一个分子式在论文里出现多次时,
    #: 每条分子量声明都该和"离自己最近的那次出现"配对, 否则会跨段误配（实测踩过）。
    pos_hints: list = field(default_factory=list)
    #: 所在**行号**（页内换行计数）。配对时"同一行"优先于"字符距离近" ——
    #: 实测: `... Table 1 with MW 194.20.` 里, 下一行的 `C2H6O` 字符距离反而更近,
    #: 只按距离会把咖啡因的分子量配给乙醇。行是比距离更强的归属信号。
    line_no: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key, "rank": self.rank, "kind": self.kind,
            "value": self.value, "page": self.page, "context": self.context,
            "computed": self.computed, "computed_value": self.computed_value,
            "declared": self.declared, "declared_digits": self.declared_digits,
            "status": self.status, "status_zh": STATUS_ZH.get(self.status, self.status),
            "detail": self.detail, "note": self.note,
        }


# ---------------------------------------------------------------- ② 站：只出数

def _line_of(text: str, pos: int) -> int:
    """算出某个偏移在第几行（页内换行计数）—— 配对时用来判"同一行"。"""
    return (text or "").count("\n", 0, max(0, int(pos)))


def _ctx(text: str, start: int, end: int) -> str:
    a = max(0, start - _CTX_PAD)
    b = min(len(text), end + _CTX_PAD)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def _declared_hits(text: str) -> list:
    """抽出所有「分子量声明」—— (位置, 数值, 小数位数, 原文)。"""
    out = []
    for m in _MW_DECL.finditer(text or ""):
        raw = m.group(1)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        # 声明的"精度": 小数点后给了几位（180.16 -> 2 位; 180 -> 0 位）
        digits = len(raw.split(".")[1]) if "." in raw else 0
        out.append((m.start(), val, digits, m.group(0).strip()))
    return out


def _strip_outer_parens(s: str) -> str:
    """去掉分子式最外层的**排版圆括号**: `(C8H10N4O2)` -> `C8H10N4O2`。

    为什么要这一步: 论文里常见 `Glucose (C6H12O6, MW 180.2)` 这种写法, 正则会把
    左边那个排版括号吃进候选串, 于是出现 `(C2H6O` 这种"括号没闭合"的假故障。
    只在**剥掉之后还能解析**时才剥 —— 剥了反而坏掉的, 说明括号本来就是式子的一部分。
    **不剥方括号**: `[Fe(CN)6]3-` 里的方括号是配位化学标记, 不是排版符号。
    """
    while s.startswith("("):
        cand = s[1:-1] if s.endswith(")") else s[1:]
        if not cand or parse_formula(cand)[1]:
            break
        s = cand
    return s


def _formula_hits(text: str) -> list:
    """抽出所有「像分子式」的片段 —— (位置, 原文)。"""
    out = []
    for m in _FORMULA_CAND.finditer(text or ""):
        raw = _strip_outer_parens(m.group(1))
        if looks_like_formula(raw):
            out.append((m.start(), raw))
    return out


def measure_text(text: str, page: int = 0, corpus_ok: bool = True) -> list:
    """② 实验站的活: 从一页文本里抽出化学身份, **并算出该算的数**（不下判断）。

    三类产物:
    - 分子式 -> 解析出的元素计数 + 按原子量算出的分子量（`computed`）;
    - CAS -> 校验位算法应得的数（`computed`）;
    - SMILES -> 语法体检结论（`computed` 写"通过/不通过"）, 装了 RDKit 再补分子量;
    - 分子量声明 -> 只登记数（`declared`）, 等 ③ 去和某个分子式配对。

    `corpus_ok` 是**整篇语料闸门**（由 `measure` 用 `is_chem_corpus` 判定后传入）。
    为 False 时只抽带显式标签的 SMILES / InChI, 分子式 / 分子量声明 / CAS 一律不抽 ——
    因为它们在非化学论文里几乎全是误报（UCF101 / FP16 / PPO2 等, 见 `_CHEM_CORPUS_MIN`）。
    """
    text = text or ""
    items: list = []

    # ---- SMILES（只在有显式标签处抽）
    for m in _SMILES_LABEL.finditer(text):
        smi = m.group(1).strip(".,;")
        ok, why = smiles_lint(smi)
        extra = rdkit_analyze(smi) if ok else None
        computed = "语法体检通过" if ok else f"语法体检不通过（{why}）"
        item = ChemItem(
            key=f"smiles:{page}:{smi[:40]}", kind="SMILES", value=smi, page=page,
            context=_ctx(text, m.start(), m.end()), pos_hint=m.start(), pos_hints=[m.start()], line_no=_line_of(text, m.start()),
            computed=computed,
            status="undetermined", detail=why,
            note="SMILES 语法不等于化学正确性校验" if not extra else "RDKit 精确解析")
        if extra and extra.get("ok"):
            item.computed_value = extra.get("mol_weight")
            item.computed = (f"规范化 {extra.get('canonical')}；分子式 "
                             f"{extra.get('formula')}；分子量 {extra.get('mol_weight')}")
        items.append(item)

    # ---- InChI（只做形态登记, 不解析）
    for m in _INCHI.finditer(text):
        raw = "InChI=1S/" + m.group(1).rstrip(".,;")
        items.append(ChemItem(
            key=f"inchi:{page}:{raw[:40]}", kind="InChI", value=raw, page=page,
            context=_ctx(text, m.start(), m.end()), pos_hint=m.start(), pos_hints=[m.start()], line_no=_line_of(text, m.start()),
            computed="已登记（本系统不解析 InChI 分层）",
            status="undetermined",
            detail="InChI 是标准标识符，本系统只核对形态是否存在，不做结构还原",
            note="需要结构还原请用 RDKit / InChI 官方工具"))

    if not corpus_ok:
        # 非化学语料：到此为止。只留下带显式标签的 SMILES / InChI ——
        # 标签本身是化学证据；而裸分子式 / 分子量声明 / CAS 在这类论文里
        # 几乎全是误报（UCF101 / CUB200 / FP16 / PPO2 / 日期 …）。
        return items

    # ---- CAS（算校验位）
    for m in _CAS.finditer(text):
        cas = m.group(1)
        ok, why = cas_check(cas)
        # 校验位**不符**时, 只有邻近上下文明确是化学才登记。
        # 理由: 日期 `2015-10-2`、页码区间 `12-34-5` 形态都合法但校验位几乎必错,
        # 登记它们只会凭空造一条「CAS 号无效」（2026-10-29 darknet 参考文献实测）。
        # 校验位**正确**的一律登记 —— 10 位数字串恰好过了校验位的概率只有 1/10。
        # 判断"邻近是不是化学"时要把自身去掉, 否则 `CAS 50-78-2` 会靠自己的标签过闸。
        if ok is not True:
            if not _CHEM_CTX.search(_ctx(text, m.start(), m.end()).replace(cas, "")):
                continue
        items.append(ChemItem(
            key=f"cas:{page}:{cas}", kind="CAS", value=cas, page=page,
            context=_ctx(text, m.start(), m.end()), pos_hint=m.start(), pos_hints=[m.start()], line_no=_line_of(text, m.start()),
            computed=("校验位正确" if ok else ("校验位不符" if ok is False else "无法校验")),
            status="undetermined", detail=why,
            note="CAS 校验位由算法算出，不是查库比对"))

    # ---- 分子式（算分子量）
    for pos, raw in _formula_hits(text):
        counts, err = parse_formula(raw)
        weight, werr = formula_weight(raw)
        item = ChemItem(
            key=f"formula:{page}:{raw}", kind="分子式", value=raw, page=page,
            context=_ctx(text, pos, pos + len(raw)), pos_hint=pos, pos_hints=[pos], line_no=_line_of(text, pos),
            computed=(f"元素 {formula_text(counts)}；分子量 {weight:.3f}" if weight else ""),
            computed_value=weight,
            status="undetermined",
            detail=(f"解析失败：{err or werr}" if (err or werr) else
                    "、".join(f"{k}×{v}" for k, v in sorted(counts.items()))))
        items.append(item)

    # ---- 分子量声明（只登记, 等配对）
    for pos, val, digits, raw in _declared_hits(text):
        items.append(ChemItem(
            key=f"mw:{page}:{pos}:{val}", kind="分子量声明", value=raw, page=page,
            context=_ctx(text, pos, pos + len(raw)), pos_hint=pos, pos_hints=[pos], line_no=_line_of(text, pos),
            declared=val, declared_digits=digits,
            computed="", status="undetermined",
            detail=f"论文声明分子量 {val}（给了 {digits} 位小数）"))
    return items


def measure(paper) -> tuple:
    """对整篇论文跑一遍 ② 站: 返回 (记录列表, 统计)。**只出数, 不下判断**。"""
    items: list = []
    pages = list(getattr(paper, "page_texts", None) or [])
    full_text = "\n".join(t or "" for _, t in pages)
    corpus_ok = is_chem_corpus(full_text)
    for page_no, text in pages:
        items += measure_text(text or "", int(page_no), corpus_ok=corpus_ok)
    # 去重（同页同类型同原文只留一条）—— 但**位置全部留下**:
    # 同一个分子式在论文里出现多次, 每条分子量声明要跟"离自己最近的那次出现"配对,
    # 只留第一个位置会让后面的声明跨段误配到别的分子式上（实测踩过）。
    seen: dict = {}
    uniq: list = []
    for it in items:
        sig = (it.page, it.kind, it.value)
        hit = seen.get(sig)
        if hit is not None:
            hit.pos_hints.extend(it.pos_hints)
            continue
        seen[sig] = it
        uniq.append(it)
    stats = {
        "total": len(uniq),
        "pages": len(pages),
        "kinds": {},
        "engine": engine_name(),
        "rdkit": bool(rdkit_analyze("CCO")),
        "notes": [],
        # 新键（只加不改）：本篇是否被判为化学语料。界面 / 报告据此说明
        # 「没抽到」到底是「论文里没有」还是「本篇压根不是化学论文」。
        "chem_corpus": corpus_ok,
    }
    for it in uniq:
        stats["kinds"][it.kind] = stats["kinds"].get(it.kind, 0) + 1
    if not uniq:
        if corpus_ok:
            stats["notes"].append(
                "本论文正文里没有抽到化学身份（分子式 / CAS / SMILES / 分子量声明）—— "
                "这可能是**非化学领域论文**，也可能化学信息只存在图里（本系统不从像素猜结构）")
        else:
            stats["notes"].append(
                "本篇不是化学语料（正文没有分子量 / 摩尔 / 溶剂 / NMR 这类词），"
                "按口径**不做化学核验** —— 强行核验只会把 UCF101、FP16 这类串认成分子式")
    return uniq, stats


# ---------------------------------------------------------------- ③ 站：只下判断

def _round_like(value: float, digits: int) -> float:
    """按论文自己给的小数位四舍五入 —— 判据的基准就取作者的精度。"""
    return round(float(value), int(digits))


def judge(items: list, *, near: int = 260) -> list:
    """③ 验证站的活: 拿 ② 已经算出来的数下判断, **不重新算任何数**。

    两类配对:
    - **分子式 ↔ 分子量声明**: 同一页、位置最近（±`near` 字符内）的一对,
      按"声明的小数位"四舍五入后比较;
    - **自带结论的项**: CAS（校验位）、SMILES（语法）、分子式（能否解析）。
    """
    items = list(items or [])

    # ---- 1. 自身合规性（用 ② 已经写好的 computed / detail, 不重算）
    for it in items:
        if it.kind == "CAS":
            if it.computed == "校验位正确":
                it.status, it.detail = "consistent", it.detail
            elif it.computed == "校验位不符":
                it.status = "invalid"
            else:
                it.status = "undetermined"
        elif it.kind == "SMILES":
            if it.computed.startswith("语法体检不通过"):
                it.status = "invalid"
            elif it.computed.startswith("语法体检通过"):
                # 装了 RDKit 且给得出分子量 -> 仍等"有无分子量声明"再判一致性
                it.status = "consistent" if it.computed_value else "undetermined"
            else:
                it.status = "undetermined"
        elif it.kind == "分子式":
            if not it.computed:
                it.status = "invalid"          # 式子本身解析不了
            else:
                it.status = "undetermined"     # 要先找到配对的分子量声明

    # ---- 2. 分子量声明 ↔ 分子式配对（**以声明为主**: 每条声明各找最近的分子式）
    #
    # 为什么不是"以分子式为主": 同一分子式在论文里常被提好几次（正文一次、表里一次），
    # 每次都可能给不同精度的分子量。以分子式为主会发现"第二条声明没人配" —— 实测踩过。
    declared = [it for it in items if it.kind == "分子量声明" and it.declared is not None]
    formulas = [it for it in items if it.kind == "分子式" and it.computed_value]
    paired: dict = {}                      # id(分子式) -> [(声明, 是否一致, 舍入后, 声明值)]
    for d in declared:
        best, best_key = None, None
        for f in formulas:
            if f.page != d.page:
                continue
            # 距离取"声明位置 → 该式子**任一次**出现位置"的最近值
            spots = f.pos_hints or [f.pos_hint]
            dist = min(abs(d.pos_hint - p) for p in spots)
            # **同行优先**: 行是比字符距离更强的归属信号 ——
            # 实测 `... with MW 194.20.` 下一行的式子字符距离更近, 只按距离会配错。
            key = (0 if f.line_no == d.line_no else 1, dist)
            if best is None or key < best_key:
                best, best_key = f, key
        if best is None or best_key[1] > near:
            d.status = "undetermined"
            d.detail = (d.detail + "；同页没找到可配对的分子式"
                        "（本系统不从图里猜分子式）").strip("；")
            continue
        want = _round_like(best.computed_value, d.declared_digits)
        got = _round_like(d.declared or 0.0, d.declared_digits)
        ok = abs(want - got) < 1e-9
        d.status = "consistent" if ok else "inconsistent"
        d.detail = (f"与同页 {best.value} 配对：算得 {best.computed_value:.4f}，"
                    f"按 {d.declared_digits} 位小数舍入 = {want}，声明 {got}"
                    + ("" if ok else " ← 对不上"))
        paired.setdefault(id(best), []).append((d, ok, want, got))

    # ---- 3. 分子式侧回填（同一式子被多处声明引用时: **逐条核对, 取最差的那条**）
    #
    # 为什么不排他配对: 同一个分子式在论文里被提多次, 每次给的精度可能不同
    # （180.16 与 180.2）。排他配对会让其中一条声明"无主"而假装判不了 ——
    # 而它其实完全可判。这里改成"一处对不上就算对不上"。
    for f in formulas:
        hits = paired.get(id(f))
        if not hits:
            f.detail = (f.detail + "；同页没有可配对的分子量声明").strip("；")
            continue
        bad = [h for h in hits if not h[1]]
        d0, _, want0, got0 = (bad or hits)[0]
        f.declared, f.declared_digits = d0.declared, d0.declared_digits
        f.status = "inconsistent" if bad else "consistent"
        f.detail = (f"{f.value} 按原子量算得 {f.computed_value:.4f}，"
                    f"按论文的 {d0.declared_digits} 位小数舍入 = {want0}，"
                    + (f"但论文声明的是 {got0}" if bad else f"与论文声明的 {got0} 一致"))
        f.note = "判据 = 按论文自己给的小数位舍入后是否相等（阈值取作者的精度，不另设）"
        if len(hits) > 1:
            f.note += (f"；本文里这个式子被 {len(hits)} 处声明引用，"
                       f"其中 {len(bad)} 处对不上")

    # ---- 排序：先硬结论（不合规 / 对不上 / 对得上），再判不了
    order = {"invalid": 0, "inconsistent": 1, "consistent": 2, "undetermined": 3}
    items.sort(key=lambda x: (order.get(x.status, 9), x.kind, x.page, x.value))
    for i, it in enumerate(items, 1):
        it.rank = i
    return items


def summarize(items: list, stats: Optional[dict] = None) -> dict:
    """一句话说清这批记录分成了几堆（界面 / 报告共用）。"""
    items = list(items or [])
    by_status: dict = {}
    by_kind: dict = {}
    for it in items:
        by_status[it.status] = by_status.get(it.status, 0) + 1
        by_kind[it.kind] = by_kind.get(it.kind, 0) + 1
    out = {
        "total": len(items),
        "by_status": by_status,
        "by_kind": by_kind,
        "checked": sum(1 for it in items if it.status in ("consistent", "inconsistent", "invalid")),
        "bad": sum(1 for it in items if it.status in ("inconsistent", "invalid")),
        "engine": engine_name(),
    }
    if stats:
        out["stats"] = stats
    return out


def to_markdown(items: list, stats: Optional[dict] = None, *, limit: int = 8) -> str:
    """把化学核验结果渲染成一段 Markdown（报告用）。**没有结论就如实说没有。**"""
    items = list(items or [])
    if not items:
        return "本次没有在正文里抽到化学身份（分子式 / CAS / SMILES / 分子量声明）。"
    s = summarize(items, stats)
    lines = [f"共 **{s['total']}** 条化学身份记录，其中可判定 **{s['checked']}** 条"
             f"（对不上 / 不合规 **{s['bad']}** 条）。"]
    lines.append("")
    lines.append("| # | 类型 | 原文 | 页 | 判定 | 依据 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for it in items[:limit]:
        detail = (it.detail or "").replace("|", "\\|")
        val = (it.value or "").replace("|", "\\|")
        lines.append(f"| {it.rank} | {it.kind} | `{val}` | {it.page} | "
                     f"{STATUS_ZH.get(it.status, it.status)} | {detail[:120]} |")
    if len(items) > limit:
        lines.append(f"| … | | 其余 {len(items) - limit} 条见执行记录 | | | |")
    lines.append("")
    lines.append("> 判据: 分子量按内置 IUPAC 原子量复算，再**按论文自己给的小数位**舍入后比较；"
                 "CAS 校验位由算法算出；SMILES 只做语法体检（不校验价态），"
                 "本系统**不从图像像素猜分子结构**。")
    if stats and stats.get("notes"):
        lines.append("")
        for n in stats["notes"][:3]:
            lines.append(f"> {n}")
    return "\n".join(lines)
