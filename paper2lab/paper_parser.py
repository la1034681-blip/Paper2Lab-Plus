"""论文 PDF 解析器——PyMuPDF 按页提取, 正则定位参数与实验结果。

证据链要求: 每条参数声明都记录"Page N · Section"。
"""

from __future__ import annotations

import re
from typing import Optional

import fitz  # PyMuPDF

from .models import Evidence, PaperInfo
from .synonyms import (PARAM_SYNONYMS, canonical_key, is_plausible,
                       normalize_number)

# ---------------------------------------------------------------- 数值/结果模式

# 数值 token: 支持 0.0001 / 1e-4 / 500K / 1M / 10^-4 / 10⁻⁴ / 1×10⁻⁴ / 1,500
#
# 两处刻意的写法(都由 2026-09-28 新样本实测逼出来, 见测试集 v2 发现报告):
#  ① `\d+(?:\.\d+)?` 而非 `\d+\.?\d*` —— 后者会把**句末句点**一起吞进值里。
#     实测: ConvNeXt "resolution is 2242." 抽成 `2242.`（再归一化成 2242.0, 并被写进补丁
#     `default=224 → 2242.0`）; ImprovedDDPM "K = 5. We trained" 抽成 `5.`。
#  ② 千分位单列一支 —— 否则 "1,500" 只匹配到 "1"。
#     实测: Swin "linear warmup of 1,500 iterations" 的 warmup 抽成 1,
#     并生成了错补丁 `WARMUP_EPOCHS 20 → 1`。
_THOUSANDS = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"
_NUM = (rf"({_THOUSANDS}\s*[×xX*]\s*10\s*(?:\^|⁻|−|-)\s*\d+"   # 1,000×10^-4
        rf"|10\s*(?:\^|⁻|−|-)\s*\d+"                            # 10^-4 / 10⁻⁴
        rf"|{_THOUSANDS}(?![,.\d])"                             # 1,500 / 2,000,000
        rf"|\d+(?:\.\d+)?\s*(?:[eE][+-]?\d+)"                   # 1e-4
        rf"|\d+(?:\.\d+)?\s*[kKMGB](?![A-Za-z])"                # 500K / 1.5M(不收小写 m/b)
        rf"|\.\d+|\d+(?:\.\d+)?)")                              # 0.0001 / 32
# 带边界保护的数字: 前面不能是字母/数字/点, 避免 "model.7" 里的小数尾巴被当成参数值
# (中文不受影响: 只排除 ASCII 词字符与小数点)
_NUM_B = r"(?<![A-Za-z0-9_.])" + _NUM

def _has_cjk(s: str) -> bool:
    return any('一' <= ch <= '鿿' for ch in s)


# 论文中参数声明的典型句式, 如:
#   "learning rate of 1e-4" / "learning rate = 1e-4" / "learning rate is set to 0.0001"
#   中文: "学习率为0.01" / "聚类数K=4" / "迭代次数取100"
def _param_patterns(aliases: list[str]) -> list[re.Pattern]:
    pats = []
    connectors = (r"(?:of|=|:|is set to|is fixed to|is fixed at|fixed to|set to|is|to|"
                  r"was set to|为|取|设定为|设置为|设为|等于|是|确定为|选取|选择)")
    for a in aliases:
        a_re = re.escape(a).replace(r"\ ", r"[\s_\-]")
        if _has_cjk(a):
            # 中文别名: \b 对 CJK 无效, 直接匹配词本身;
            # 允许别名与数值之间出现符号位, 如 "聚类数K=4" / "杀伤半径R=20m" / "深度定位值h₀=150m"
            symbol = "(?:[A-Za-zσ]{1,3}[₀-₉]?\\s*)?"
            pats.append(re.compile(rf"{a_re}\s*{symbol}{connectors}?\s*{_NUM_B}"))
        else:
            pats.append(re.compile(
                rf"\b{a_re}\b\s*{connectors}?\s*{_NUM_B}",
                re.IGNORECASE,
            ))
    return pats


# ---------------------------------------------------------------- 兜底句式
# 别名与数值之间夹着短语的写法, 常规 "别名+连接词+数字" 抓不到, 例如:
#   "We set the dropout rate on CIFAR10 to 0.1 by sweeping over ..."
#   "We set dropout rate on the other datasets to zero"
# 策略(保守): 在**同一句**内, 取别名之后最近的一个数字(或数量级词),
# 且别名与数字之间必须出现连接词(to/as/of/is/为/是...)——否则不采信。
# 句子切分: 句号/分号断句, 但**小数点与换行都不能当句末**。
#   · 分支顺序很关键: 必须把 \d\.\d 放在前面, 否则 [^.;] 先吃掉数字、轮到 '.' 就断句
#     (实测 "to 0.1" 会被切成 "to 0", dropout 取到 0 而非 0.1);
#   · 换行不能断句: PDF 按行排版, "learning rate that begins at\n5 × 10−4" 会被硬切开
#     (实测 NeRF 的学习率因此漏抽)。
_SENTENCE_RE = re.compile(r"(?:\d\.\d|[^.;。；]){15,400}")
# 连接词必须**紧邻数值**(gap 的结尾), 否则不采信
# ⚠️ 连接词前必须加词边界 `(?<![A-Za-z])` —— 否则 "that"/"what"/"format" 里的 `at`
#    会被当成连接词 `at` 命中。实证(2026-09-28, DeiT p12):
#    text-l2 把 "…the average time over 30 runs to process that batch" 里的 30 当成批大小,
#    而"数字与参数名之间必须由连接词挂钩"这道收紧闸门**因为 `that` 尾巴的 `at` 而误判通过**。
_FALLBACK_CONNECTOR_END = re.compile(
    r"(?<![A-Za-z])(?:to|as|of|is|are|was|were|equals?|at|=|:|为|是|取|达到|等于|设定为)\s*$",
    re.IGNORECASE)
# 这些短语说明上下文不是"参数取值", 出现即放弃(如 "a factor of two" / "stride 4")
_BAD_GAP = re.compile(
    r"\b(factor of|a factor|stride|times|out of|divided by|per|rather than|instead of)\b",
    re.IGNORECASE)
_GAP_LIMIT = 50          # 别名与数值的最大字符间隔(超出视为不相关)
# 仅收"zero"这类语义唯一的数量词("by a factor of two" 里的 two 不是取值, 故不收)
_NUM_WORDS = {"zero": "0", "none": "0"}

# 「gap 里夹着别的参数名」用的正则(懒编译)。
# 只收**足够特异**的别名(多词或 ≥6 字符), 避免 "data"/"net" 这类泛词误伤。
_OTHER_ALIAS_RE: Optional[re.Pattern] = None


def _other_alias_re() -> re.Pattern:
    global _OTHER_ALIAS_RE
    if _OTHER_ALIAS_RE is None:
        names: set[str] = set()
        for _k, (_disp, aliases) in PARAM_SYNONYMS.items():
            for a in aliases:
                if len(a) >= 6 or " " in a or "_" in a:
                    names.add(a)
        parts = sorted((re.escape(a).replace(r"\ ", r"[\s_\-]") for a in names),
                       key=len, reverse=True)
        _OTHER_ALIAS_RE = re.compile(
            r"(?<![A-Za-z0-9_])(?:" + "|".join(parts) + r")(?![A-Za-z0-9_])",
            re.IGNORECASE)
    return _OTHER_ALIAS_RE


def _alias_re(alias: str) -> re.Pattern:
    """ASCII 别名要求词边界(否则 'lr' 会命中 'already'); 中文别名直接找。"""
    if alias.isascii():
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])",
                          re.IGNORECASE)
    return re.compile(re.escape(alias))


def gap_value_after_alias(text: str, aliases: list[str]
                          ) -> Optional[tuple[str, int, int]]:
    """在"别名 + 短语 + 数值"的句式里取值。

    返回 (原始值, 句子起点, 数值终点) 或 None。仅作**兜底**(前面句式没取到值时才用)。
    三道收紧(实测过松会把 "a factor of two" 当成取值 2):
      ① 连接词必须紧邻数值; ② gap 里出现 "factor of/stride" 等短语即放弃;
      ③ 数量词只收 "zero/none" 这类语义唯一的。
    """
    for sm in _SENTENCE_RE.finditer(text):
        sent = sm.group(0)
        for a in aliases:
            am = _alias_re(a).search(sent)
            if not am:
                continue
            tail = sent[am.end(): am.end() + _GAP_LIMIT]

            def _ok(gap: str) -> bool:
                if _BAD_GAP.search(gap) or not _FALLBACK_CONNECTOR_END.search(gap):
                    return False
                # gap 里夹着**别的参数名** -> 这个数字其实属于那个参数。
                # 实测(2026-09-28, ConvNeXt): "All model variants are trained for 160K
                # iterations with a batch size of 16" —— iterations 的兜底取值跨过了
                # "with a batch size of", 把 **batch size 的 16** 当成了迭代次数。
                if _other_alias_re().search(gap):
                    return False
                return True

            nm = re.compile(_NUM_B).search(tail)
            if nm and _ok(tail[: nm.start()]):
                return nm.group(0), sm.start(), sm.start() + am.end() + nm.end()
            wm = re.search(r"\b(" + "|".join(_NUM_WORDS) + r")\b", tail)
            if wm and _ok(tail[: wm.start()]):
                return (_NUM_WORDS[wm.group(1).lower()], sm.start(),
                        sm.start() + am.end() + wm.end())
    return None


# 数字在前的句式: "train for 100 epochs" / "100 epochs"
_REVERSE_PATTERNS: dict[str, list[re.Pattern]] = {
    "input_size": [
        # "Our 32 x 32 models use four feature map resolutions"
        re.compile(rf"\b{_NUM}\s*[×x*]\s*{_NUM}\s*(?:pixel\s+)?(?:images?|models?|inputs?)\b",
                   re.IGNORECASE),
        # "images of size 32 x 32" / "images of 224 x 224"
        re.compile(rf"\bimages?\s+(?:of\s+)?(?:size\s+|resolution\s+)?{_NUM}\s*[×x*]\s*{_NUM}",
                   re.IGNORECASE),
        # "input size of 224 x 224" / "resolution of 32 x 32"
        re.compile(rf"\b(?:input\s+size|image\s+size|resolution|img_size)\s+(?:of\s+|is\s+)?"
                   rf"{_NUM}\s*[×x*]\s*{_NUM}", re.IGNORECASE),
    ],
    "epochs": [
        re.compile(rf"(?:train(?:ed|ing)?|run(?:ning)?)\s+(?:for\s+)?{_NUM}\s*epochs?\b",
                   re.IGNORECASE),
        re.compile(rf"\bfor\s+{_NUM}\s*epochs?\b", re.IGNORECASE),
        re.compile(rf"迭代\s*{_NUM}\s*次"),
    ],
    "batch_size": [
        re.compile(rf"\b{_NUM}\s*(?:samples?\s+)?per\s+batch\b", re.IGNORECASE),
    ],
    "num_clusters": [
        re.compile(rf"\b[Kk]\s*=\s*{_NUM}"),           # "K=4"
        re.compile(rf"聚成\s*{_NUM}\s*(?:类|簇)"),        # "聚成4类"
        re.compile(rf"{_NUM}\s*个(?:聚类|簇)"),           # "4个聚类"
    ],
    "num_bombs": [
        re.compile(rf"{_NUM}\s*枚深弹"),                 # "9 枚深弹"
        re.compile(rf"深弹\s*{_NUM}\s*枚"),              # "深弹 9 枚"
        re.compile(rf"携带\s*{_NUM}\s*枚"),              # "可携带9枚"
    ],
}

# 取值为文本的参数(与 code_analyzer 保持一致的口径)
_TEXT_VALUED_KEYS = {"optimizer", "scheduler", "dataset", "model"}

# 文本型参数(optimizer / dataset / scheduler / model)的模式
_TEXT_PARAM_PATTERNS: dict[str, list[re.Pattern]] = {
    "optimizer": [
        re.compile(r"\b(Adam|AdamW|SGD|RMSprop|Adagrad|AdaDelta|Lamb)\b"
                   r"(?:\s+optimizer)?", re.IGNORECASE),
        re.compile(r"optimizer\s*(?:of|=|:|is)?\s*(Adam|AdamW|SGD|RMSprop|Adagrad|AdaDelta|Lamb)",
                   re.IGNORECASE),
    ],
    "scheduler": [
        re.compile(r"\b(cosine(?:\s+annealing)?|step\s*lr|multistep|exponential|"
                   r"plateau|warmup\s+cosine)\b(?:\s+(?:learning\s+rate\s+)?schedul\w+)?",
                   re.IGNORECASE),
    ],
    # 模型架构: 只认明确的架构名, 防止把正文里任意数字当成模型规模。
    # 具体架构名(ResNet-50/ViT/...)优先于泛称(Transformer/GPT), 避免抓到泛称。
    "model": [
        re.compile(r"\b(ResNet-?\d+|ResNeXt-?\d+|WideResNet-?\d+|VGG-?\d+|"
                   r"DenseNet-?\d+|MobileNet-?V?\d*|EfficientNet-?[Bb]?\d|"
                   r"ViT-?[HLB]?\d*|DeiT-?\w*|Swin-?\w*|BERT-?\w+|RoBERTa-?\w*|"
                   r"U-?Net|CycleGAN|pix2pix|TabNet|SimCLR|SimSiam|MoCo|BYOL|"
                   r"PointNet\+*|NeRF|DETR|GPT-?\d*|Transformer-?\w*)\b"),
        re.compile(r"\b(model|architecture|backbone)\s*(?:是|为|is|:|=)\s*"
                   r"([A-Za-z][\w\-\.]{1,30})"),
    ],
}

# 实验结果句式: "accuracy of 94.3%" / "achieves 94.3% accuracy" / "Top-1 accuracy 94.3"
# 中文: "准确率达到95.2%" / "精度为0.87"
_RESULT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("accuracy", re.compile(rf"\b(?:top-?1\s+)?accuracy\b\s*(?:of|=|:|is|reaches?|achieves?)?\s*{_NUM}\s*%?", re.IGNORECASE)),
    ("accuracy", re.compile(rf"\bachieves?\s+{_NUM}\s*%?\s*(?:top-?1\s+)?accuracy", re.IGNORECASE)),
    ("accuracy", re.compile(rf"(?:准确率|正确率|精度)\s*(?:达到|为|=|:|是)?\s*{_NUM}\s*%?")),
    ("f1", re.compile(rf"\bf1(?:\s*score)?\b\s*(?:of|=|:|is)?\s*{_NUM}\s*%?", re.IGNORECASE)),
    ("f1", re.compile(rf"f1\s*(?:值|分数|score)?\s*(?:达到|为|=|:)?\s*{_NUM}\s*%?", re.IGNORECASE)),
    ("precision", re.compile(rf"\bprecision\b\s*(?:of|=|:|is)?\s*{_NUM}\s*%?", re.IGNORECASE)),
    ("precision", re.compile(rf"精确率\s*(?:达到|为|=|:)?\s*{_NUM}\s*%?")),
    ("recall", re.compile(rf"\brecall\b\s*(?:of|=|:|is)?\s*{_NUM}\s*%?", re.IGNORECASE)),
    ("recall", re.compile(rf"召回率\s*(?:达到|为|=|:)?\s*{_NUM}\s*%?")),
    ("loss", re.compile(rf"\b(?:final\s+|test\s+)?loss\b\s*(?:of|=|:|is)?\s*{_NUM}", re.IGNORECASE)),
    ("silhouette", re.compile(rf"(?:silhouette|轮廓系数)\s*(?:coefficient|系数|score)?\s*(?:of|=|:|为|是)?\s*{_NUM}", re.IGNORECASE)),
]

#: 「**数值在前**」的写法 —— 2026-10-29 由 51 份真实论文逼出来的补充。
#:
#: 上面那组模式只认「指标词 → 数字」的顺序（`accuracy of 76.5%`），而真实论文更常写
#: 数字在前：DETR 摘要写 `42 AP`、CenterNet 写 `45.1% COCO AP`、Glow 写 `3.35 bits/dim`。
#: 实测 51 份里 28 份因此一条结果都抽不到（不是论文没写，是写法没覆盖）。
#:
#: ⚠ 代价是**误抽风险变高**：`Fig. 1 AP` / `Table 2 AP` 这种图注编号也会命中。
#: 所以每条都过 `_result_veto()`（图注编号、引用方括号、数值范围三道否决），
#: 且**只在正文/章节位置成立时**登记。宁可少抽，也不能把图号当精度写进证据。
_NUMFIRST: list[tuple[str, re.Pattern]] = [
    ("accuracy", re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,2})?)\s*%\s*(?:top-?[15]\s+)?(?:accuracy|acc)\b", re.IGNORECASE)),
    ("accuracy", re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,2})?)\s*(?:%|\s)\s*top-?[15]\s+(?:accuracy|acc)\b", re.IGNORECASE)),
    ("ap", re.compile(r"(?<![\w.])(\d{1,2}(?:\.\d{1,2})?)\s*(?:%|\s)\s*(?:box\s+|mask\s+)?AP(?:50|75|s|m|l)?\b")),
    ("f1", re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,2})?)\s*(?:%|\s)\s*F1\b", re.IGNORECASE)),
    ("psnr", re.compile(r"(?<![\w.])(\d{1,2}\.\d{1,2})\s*dB\b", re.IGNORECASE)),
    ("ssim", re.compile(r"(?i)\bSSIM\b\s*(?:of|=|:|is)?\s*(\d\.\d{2,4})")),
    ("bleu", re.compile(r"(?<![\w.])(\d{1,2}\.\d{1,2})\s*BLEU\b", re.IGNORECASE)),
    ("miou", re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,2})?)\s*(?:%|\s)\s*(?:mIoU|IoU)\b", re.IGNORECASE)),
    ("bits/dim", re.compile(r"(?<![\w.])(\d{1,2}\.\d{2})\s*bits\s*/\s*dim", re.IGNORECASE)),
]

#: 每个指标的**合法数值区间**。超出一律不登记 —— 这是挡「把图号/年份当精度」的最后一道。
#: `bits/dim` 是负对数似然，ImageNet 上约 1–4；`fid` 能到几百。
_RESULT_RANGE: dict = {
    "accuracy": (0.0, 100.0), "f1": (0.0, 100.0), "precision": (0.0, 100.0),
    "recall": (0.0, 100.0), "ap": (0.0, 100.0), "miou": (0.0, 100.0),
    "bleu": (0.0, 100.0), "psnr": (5.0, 60.0), "ssim": (0.0, 1.0),
    "loss": (0.0, 100.0), "silhouette": (-1.0, 1.0), "bits/dim": (0.0, 20.0),
    "fid": (0.0, 500.0), "inception": (0.0, 100.0),
    # Inception Score 在 ImageNet 上几十到几百
    "is": (0.0, 1000.0),
    # 表格通道用到的：困惑度在语言模型上几十到几百，上限给到 10000 足够
    "ppl": (0.0, 10000.0),
}

#: 数字前面出现这些, 说明它是**图注/表号/式号/章节号**, 不是实验结果。
#: 窗口取 16 字符: 实测 `Table 2 shows ... 42 AP` 里 42 离 Table 太远, 不会被误判。
_RESULT_REF_BEFORE = re.compile(
    r"(?i)(?:fig(?:ure)?\.?|tab(?:le)?\.?|eq(?:uation)?\.?|sec(?:tion)?\.?|alg(?:orithm)?\.?|"
    r"appendix|补充材料)\s*\.?\s*\d{0,2}\s*[.:]?\s*$")


#: 强结果动词 —— 只有含这些词的句子才被当成「结果句」，句内才允许「数值在前」匹配。
#:
#: 为什么必须有这道: 不加时实测会把 DETR 的 `approximately 1 AP`（讲随机裁剪比例）、
#: CenterNet 的 `0.5 IOU threshold`、SENet 表头 `AP@IoU=0.5` 全当成精度登记 ——
#: 这些句子都在讲**参数/阈值**, 没有一个在报结果。加了动词闸门, 它们全部落选。
_RESULT_VERB = re.compile(
    r"(?i)\b(?:achiev\w*|obtain\w*|reach\w*|report\w*|outperform\w*|surpass\w*|exceed\w*|"
    r"state-of-the-art|sota|yield\w*|deliver\w*|improv\w*|reduc\w*\s+(?:error|loss))\b|"
    r"达到|取得|实现|优于|提升")

#: 数字**前面**是这些 = 它是阈值/约数/超参, 不是结果。
_NUM_PREFIX_BAD = re.compile(
    r"(?i)(?:[=@~≈<>]\s*|approximately\s+|approx\.?\s*|about\s+|around\s+|roughly\s+|"
    r"up to\s+|at most\s+|at least\s+|less than\s+|more than\s+|"
    r"(?:threshold|ratio|probability|prob\.?|margin|alpha|beta|gamma|k|p)\s*[=:of]{0,2}\s*)\s*$")
#: 数字**后面**是这些 = 它在解释阈值/比率, 不是结果。
_NUM_SUFFIX_BAD = re.compile(r"(?i)^\s*(?:threshold|ratio|probability|of the|interval)")


def _sentences_with_pos(text: str) -> list[tuple[int, str]]:
    """按句切成 (起始偏移, 句子)。

    ⚠ **不按换行切**。PDF 的换行是排版需要，不是句子边界 —— 实测 CenterNet 摘要：
        `Center-\\nNet achieves the best speed-accuracy trade-off on the MS\\nCOCO dataset,
         with 28.1% AP at 142 FPS, 37.4% AP at 52\\nFPS`
    按换行切会把 `achieves` 和 `28.1% AP` 分进两段，于是「结果句」判据永远不成立。
    只按 `[.;!?]` 后面跟空白切，跨行的一句话仍然是同一句。
    小数点和图号不会被切断（`3.14` 后面没有空白）。
    """
    out, start = [], 0
    for m in re.finditer(r"(?<=[.;!?])\s+", text or ""):
        seg = text[start:m.start()]
        if len(seg.strip()) >= 8:
            out.append((start, seg))
        start = m.end()
    seg = (text or "")[start:]
    if len(seg.strip()) >= 8:
        out.append((start, seg))
    return out


#: 以百分数报结果的指标。这类指标**小于 10 且不带 `%`** 时几乎都不是结果
#: （实测误抽：`AP 2.7`、`F1 01`、`accuracy 2`、`mIoU 5.4`、`precision 0.0`）。
_PCT_METRICS = frozenset({"accuracy", "ap", "miou", "f1", "precision", "recall", "bleu"})


def _result_veto(metric: str, text: str, start: int, end: int, raw: str) -> str:
    """实验结果的三道否决。返回非空字符串 = 否决原因（用于排查, 不写进证据）。"""
    # ① 数值区间
    lo, hi = _RESULT_RANGE.get(metric, (0.0, 1e9))
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return "数值无法解析"
    if not (lo <= val <= hi):
        return f"{val} 不在 {metric} 的合法区间 [{lo}, {hi}]"
    # ①b 百分数类指标：数值 < 10 又没带百分号 → 多半是阈值/编号/图号
    if metric in _PCT_METRICS and val < 10.0 and "%" not in text[end:end + 3]:
        return f"{metric}={val} 偏小且没带百分号, 疑似阈值/编号"
    # ② 前面是图注/表号/式号
    before = text[max(0, start - 16):start]
    if _RESULT_REF_BEFORE.search(before):
        return "紧跟图注/表号, 不是实验结果"
    # ③ 前面是阈值/约数/超参（`=0.5`、`approximately 1`、`threshold 0.5`）
    if _NUM_PREFIX_BAD.search(text[max(0, start - 24):start]):
        return "数字前面是阈值/约数/超参"
    # ④ 后面是阈值/比率说明
    if _NUM_SUFFIX_BAD.match(text[end:end + 14]):
        return "数字后面是阈值/比率说明"
    # ⑤ 后面紧跟方括号引用号（`42 [58]` —— CenterNet 就是这么写的, 那是别人论文的数）
    after = text[end:end + 4]
    if re.match(r"\s*\[\d", after):
        return "数字后跟引用号, 属于引用他人的结果"
    # ⑥ 数字在括号/方括号里 —— `ResNet [30] (40.6% AP in 4.1 FPS)` 这种是**引述他人**的
    #    结果, 括号本身就是"补充说明别人"的信号（CenterNet 实测因此抽错成 40.6）。
    if text[max(0, start - 2):start].rstrip().endswith(("(", "[")):
        return "数字在括号内, 属于引述他人的结果"
    return ""

# 数据集声明: "CIFAR-10 v2.1" / "on CIFAR-10" / "ImageNet dataset"
# 同页同行必须出现数据集语境词, 否则视为参考文献里的顺带提及(防误抽)
_DATASET_RE = re.compile(
    r"\b(CIFAR-?10|CIFAR-?100|ImageNet-?21k|ImageNet-?1k|ImageNet|MNIST|FashionMNIST|"
    r"SVHN|STL-?10|TinyImageNet|COCO|MS-?COCO|VOC-?20\d\d|PASCAL-?VOC|ADE20K|Cityscapes|"
    r"KITTI|LLFF|DeepVoxels|ShapeNet|ModelNet-?\d*|SUN360|OpenImages|Places-?365|"
    r"LAION-?\w*|CC-?3M|CC-?12M|SQuAD|GLUE|CoNLL|WikiText|MultiNLI)"
    r"(?:\s*(?:v|version)\s*(\d+(?:\.\d+)*))?",
    re.IGNORECASE,
)
_DATASET_CONTEXT = re.compile(
    r"(dataset|数据集|corpus|语料|trained on|train on|evaluat\w+ on|evaluate on|"
    r"benchmark|benchmarks|on the|使用|采用|在\s*\S{0,12}\s*上|分类|任务|test set|training set)",
    re.IGNORECASE,
)
# 「训练语境」——比上面的通用语境更强: 出现在这里的才是论文**主实验**用的数据集。
# 实测(2026-09-28, Swin/MAE): 摘要里先出现 "58.7 box AP on COCO test-dev",
# 于是 dataset 被抽成 COCO; 但两篇的主实验数据集都是 ImageNet(只在检测/迁移章节提 COCO)。
_DATASET_TRAIN_CTX = re.compile(
    r"trained?\s+on|train(?:ing)?\s+on|we\s+train\b|pre-?train\w*\s+on|"
    r"dataset|数据集|训练", re.IGNORECASE)


# ---------------------------------------------------------------- 文本型参数语境否决
# optimizer / scheduler / model 是"认名不认数"的抽取, 极易抓到同形词。实测(2026-09-28):
#   · ImprovedDDPM "where the model is dealing with imperceptible details"
#     → model 抽成 **dealing**(动词);
#   · MAE "…the sine-cosine version … positional embedding"
#     / DeiT "average cosine similarity between these tokens"
#     → scheduler 抽成 **cosine**(一个是位置编码, 一个是余弦相似度);
#   · ConvNeXt "A vanilla ViT, on the other hand…" → model 抽成 ViT(另一篇的名字)。
_TEXT_VETO = {
    # scheduler: 只否决**语义明确无关**的搭配。
    # ⚠️ 刻意**不**收裸词 "embedding"/"positional embedding" —— 实测(2026-09-28)
    # DETR/PointNet 通篇讲位置编码, 会把它们真正的调度器一并误杀
    # （表现为「LR Scheduler — 不一致」变成「论文未声明」）。
    # 已知的两个真误抓("the sine-cosine version … positional embedding" /
    # "average cosine similarity between these tokens")由下面两条覆盖。
    "scheduler": re.compile(
        # 2026-10-29 补 `sine ... cosine`: DETR 原文「2 sine and cosine / d channel
        # positional encoding」(正弦位置编码)，原来只认连字符写法 sine-cosine，
        # 于是把位置编码读成了 LR 调度器 —— 与代码里的 StepLR 撞成一条假冲突。
        r"similarity|sine\s*(?:-|–|—|and|&)\s*cosine|cosine\s+(?:distance|similarity)",
        re.IGNORECASE),
    "optimizer": re.compile(
        r"minibatch\s+SGD|SGD:\s*\w|Proceedings|Conference|arXiv|doi:", re.IGNORECASE),
}
# 模型名绝不能是这些普通词/动词(英文常见虚词与动词)
_MODEL_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "with", "for", "from", "to", "in", "of",
    "on", "at", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "can", "could", "will", "would", "should", "may", "might",
    "must", "does", "do", "did", "not", "no", "also", "more", "most", "less", "least",
    "other", "others", "same", "such", "when", "where", "while", "if", "then", "than",
    "it", "its", "their", "our", "we", "they", "he", "she", "you", "this", "that",
    "these", "those", "each", "both", "all", "any", "some", "only", "very", "well",
    "used", "using", "based", "proposed", "trained", "training", "shown", "given",
    "presented", "described", "applied", "defined", "called", "named", "known",
    "able", "similar", "different", "dealing", "consists", "contains", "achieves",
}
# 小写且以 ing/ed/ly 结尾的: 动词/副词, 不是架构名(除非在白名单里)
_MODEL_WHITELIST = {
    "bert", "gpt", "vit", "resnet", "unet", "moco", "byol", "simclr", "tabnet",
    "densenet", "vgg", "mobilenet", "efficientnet", "swin", "deit", "nerf", "detr",
    "pointnet", "roberta", "alexnet", "inception", "lenet", "squeezenet",
}


def _text_param_veto(key: str, val: str, ctx: str) -> str:
    """文本型参数(optimizer/scheduler/model)的语境否决。返回空串 = 通过。"""
    pat = _TEXT_VETO.get(key)
    if pat is not None and pat.search(ctx):
        return f"语境里出现与『{key}』无关的表述(同形词), 已丢弃"
    v = val.strip()
    # 以连字符结尾 = PDF 断行连字符, 不是完整词。
    # 实测(2026-09-28): MAE 抽成 `ViT-`(原文 "ViT-Huge"), ImprovedDDPM 抽成 `compet-`
    # (原文 "our model is compet- itive with ...")。
    if v.endswith(("-", "–", "—")) or v in ("-", "--"):
        return f"取值 `{v}` 以连字符结尾, 是 PDF 断行残片, 已丢弃"
    if key == "model":
        w = v.lower()
        if w in _MODEL_STOPWORDS:
            return f"取值 `{v}` 是普通虚词/动词, 不是架构名, 已丢弃"
        if w.islower() and re.search(r"(?:ing|ed|ly)$", w) and w not in _MODEL_WHITELIST:
            return f"取值 `{v}` 是动词/副词形态, 不是架构名, 已丢弃"
    return ""


def _num_value_veto(key: str, raw: str, text: str, start: int, end: int) -> str:
    """按**取值本身**或它的**紧邻形态**否决。返回空串 = 通过。

    实测(2026-09-28):
      · ConvNeXt "a larger dataset of 21841 classes" —— 数据集被抽成纯数字 `21841`(类别数);
      · DeiT 的表格里有公式单元格 `lr_scaled = lr × batchsize` / `0.0005 × batchsize 512`
        —— 那个 512 是**公式里的因子**, 不是论文设的批大小。
    """
    v = (raw or "").strip()
    if key == "dataset" and re.fullmatch(r"[\d.,%]+", v):
        return f"取值 `{v}` 是纯数字, 不是数据集名, 已丢弃"
    # 方括号编号: "cosine decay [38]" / "AdamW [39]" 里的数字是**参考文献编号**(与 tools.gate 同口径)。
    # 实证(2026-09-28, MAE): Table 8 压平后 "… cosine decay [38] ⏎ warmup epochs [20] ⏎ 40 …",
    # 真值 40 在下一行, 而 45 字窗口里抓到的是 `[38]` 的 38。
    if 0 < start < end < len(text) \
            and text[start - 1] in "[(（【" and text[end] in "])）】":
        return f"取值 `{v}` 被方括号包着, 是参考文献/图表编号, 已丢弃"
    # 乘法公式: 取值紧邻乘号时说明它在公式里(input_size 除外 —— "224×224" 是分辨率写法)
    if key != "input_size":
        left = text[max(0, start - 8): start]
        right = text[end: end + 8]
        if re.search(r"[×xX*]\s*$", left) or re.match(r"\s*[×xX*]", right):
            return f"取值 `{v}` 处在乘法公式里(两侧有 ×), 不是独立设置值, 已丢弃"
    return ""

# ---------------------------------------------------------------- 表格压平检测
# PDF 文本层会把表格**压平**成"表头 + 一串裸数字", 且这些数字之间**只隔空白**:
#   "Weight decay ⏎ 0.3 ⏎ 0.05 ⏎ Warmup epochs ⏎ 3.4 ⏎ 5"
#   "# Epochs ⏎ 30 ⏎ 60 ⏎ 30 ⏎ 80 ⏎ 25 ⏎ 25 ⏎ 80 ⏎ 40"
#   "Batch Size ⏎ 32 ⏎ 16 ⏎ 1"
# 这类位置的表头后面是**多列取值**(多个实验配置), 取第一个数等于**任选一列** —— 实测
# (2026-09-28) 因此把 DeiT 的 Weight decay 抽成 0.3(实际 0.05 才是它的列)、
# LoRA 的 Batch Size 抽成 32 / Epochs 抽成 30。这类值**不可信**, 一律不进结论与补丁链。
#
# 判据取"两个及以上裸数字只隔空白" —— 正文里几乎不会这样写
# ("8 GPUs with 2 images" / "batch size of 4096, a learning rate of 0.001" 都有词隔着),
# 因此精度较高。
_TABLE_NUM = (r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|"
              r"\.\d+|\d+(?:[kKMGB])?)")


def _in_table_run(text: str, start: int, end: int, span: int = 40) -> bool:
    """**匹配处本身**是否与相邻数值只隔空白(即被压平的表格行)。

    ⚠️ 必须只看匹配值的**紧邻两侧**, 不能"在窗口里找连续数字" —— 后者会连带误杀:
    实测(2026-09-28, E题) "最大迭代次数设为300，收敛阈值设为 4 10" —— 匹配值 300 是真值,
    却因为窗口里存在 "4 10"(其实是 4×10⁻⁴ 的 × 与上标丢失)而被判成表格串列。
    """
    left = text[max(0, start - span): start]
    if re.search(_TABLE_NUM + r"\s*$", left, re.IGNORECASE):
        return True
    right = text[end: end + span]
    return bool(re.match(r"\s*" + _TABLE_NUM, right, re.IGNORECASE))


# ---------------------------------------------------------------- 数值型参数语境否决
# 数值型参数会抓到"单位/语义不对但形态一样"的数字。实测(2026-09-28):
#   · Swin "by a multiple of 2×2 = 4 (2× downsampling of resolution)" → input_size 抽成 2;
#   · DeiT "decomposed into a batch of N patches of a fixed size of 16 × 16 pixels"
#     → batch_size 抽成 16(patch 尺寸)。
#   · ConvNeXt "resolution is 2242."(上标 224² 被文本层压平) → 靠**取值范围**拦不住,
#     改由 range 上限 2048 兜住。
_NUM_CTX_VETO: dict[str, list[tuple[re.Pattern, str]]] = {
    "input_size": [
        # ⚠️ 刻意**不**收裸词 "patch"/"stride" —— 实测(2026-09-28) 会误伤点云论文
        # (PointNet 的 input_size 因上下文含 stride 被错杀)。只收"乘法/下采样倍数"这类
        # 语义明确的语境。
        (re.compile(r"downsampl\w*|multiple\s+of|reduces\s+the\s+number\s+of\s+tokens",
                    re.IGNORECASE),
         "语境是下采样倍数 / 乘法, 不是输入分辨率"),
        # "patch(es) of a fixed size of 16 × 16" —— 这是 patch 尺寸, 不是输入尺寸
        (re.compile(r"patch(?:es)?\s+(?:of\s+)?(?:a\s+)?(?:fixed[-\s])?size\s+of\s*\d+",
                    re.IGNORECASE),
         "语境是 patch 尺寸, 不是输入分辨率"),
    ],
    "batch_size": [
        (re.compile(r"(?:size\s+of|of)\s*\d+\s*[×xX]\s*\d+", re.IGNORECASE),
         "语境是一个尺寸(如 patch 16×16), 不是批大小"),
        (re.compile(r"\bpatch(?:es)?\b", re.IGNORECASE),
         "语境在讲 patch, 不是批大小"),
        (re.compile(r"minibatch\s+SGD", re.IGNORECASE), "语境是文献标题"),
    ],
    "epochs": [
        # "divided by 2 every 20 epochs" 讲的是调度周期, 不是总轮数
        (re.compile(r"every\s+\d+\s+epochs?|per\s+\d+\s+epochs?", re.IGNORECASE),
         "语境是调度周期(every/per N epochs), 不是总轮数"),
    ],
    "iterations": [
        (re.compile(r"design|development|experimental\s+iterations?", re.IGNORECASE),
         "语境是『设计迭代』, 不是训练迭代次数"),
    ],
}


def _num_ctx_veto(key: str, ctx: str) -> str:
    """数值型参数的语境否决。返回空串 = 通过。"""
    for pat, why in _NUM_CTX_VETO.get(key, ()):
        if pat.search(ctx):
            return why
    return ""

# 章节标题探测: 全大写/编号开头的短行
_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\d*\s+)?([A-Z][A-Za-z \-]{2,40}|[A-Z][A-Z \-]{3,40})\s*$"
)


# ---------------------------------------------------------------- 主解析流程

# 疑似参数声明句式(覆盖率分母): 左侧必须是"像参数名"的标识或中文词,
# 避免把正文里所有 "= 数字" 都算进来导致分母虚高。
_DECL_RE = re.compile(
    r"(?:^|[\s,，;；。、()（）])"
    r"(?:[A-Za-z_][A-Za-z0-9_]{1,24}"          # lr / batch_size / K
    r"|[\u4e00-\u9fff]{2,10})"                 # 学习率 / 杀伤半径
    rf"\s*(?:=|＝|为|是|取|设为|设定为|设置为|of|is set to|is)\s*{_NUM}",
    re.MULTILINE,
)

# 灵敏度/分组/扩展语境: 这类语境里的不同取值是有意为之, 不算内部不一致
_CONTEXT_EXCLUDE = re.compile(
    r"(灵敏度|敏感性|扩展|扩为|增至|降至|上升至|分别取|取值区间|扫描|网格|"
    r"假设|当[^，。]{0,12}时|情形|对比|变体|方案比较|敏感性分析|"
    r"sensitivit|vary|varying|when\s|extension)",
    re.IGNORECASE,
)

# 文本层字符数低于该值的页视为"薄页"(图片/公式/扫描区, 可能未完整解析)
_THIN_PAGE_CHARS = 120


def _extract_pages(pdf_path: str) -> tuple[list[tuple[int, str]], list[dict]]:
    """返回 ([(页码, 页面文本)], 页级覆盖图)。
    覆盖图每页记录: 字符数/图片数/状态(ok|thin)。"""
    pages, coverage = [], []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            chars = len(text.strip())
            try:
                images = len(page.get_images(full=True))
            except Exception:
                images = 0
            status = "ok" if chars >= _THIN_PAGE_CHARS else "thin"
            coverage.append({"page": i + 1, "chars": chars,
                             "images": images, "status": status})
            pages.append((i + 1, text))
    return pages, coverage


# 别名同形异义否决: 别名命中处若处于下列搭配中, 说明该词在此处不是本参数。
# 实证来源(真实论文回读): DETR p2 "many design iterations" —— 指"反复改设计",
# 不是训练迭代次数; 该词全文仅此一处出现, 剔除后 DETR 不再虚报 iterations。
#
# 注意: 只否决"论文根本没在讲该参数"的情形。像 DETR 的 input_size
# （论文只给了 480–800 的范围, 没给单值）属于**真实的取不到**, 必须保留在
# 覆盖率分母里 —— 剔除它就成了粉饰指标。
_ALIAS_VETO: dict[str, list[re.Pattern]] = {
    "iterations": [
        re.compile(r"\b(design|development|experimental|manuscript|proposal|"
                   r"planning|review|code)\s+iterations?\b"),
    ],
}
_ALIAS_VETO_WINDOW = 60        # 判定同形异义的上下文窗口(字符)


def _alias_kinds_in_text(text: str) -> set[str]:
    """正文里出现过(未必带数值)的参数种类: 用于计算"参数种类覆盖率"。

    英文别名按词边界匹配, 中文别名直接子串匹配(避免中文无词边界的问题)。
    """
    low = text.lower()
    kinds: set[str] = set()
    for key, (_disp, aliases) in PARAM_SYNONYMS.items():
        vetoes = _ALIAS_VETO.get(key, [])
        for a in aliases:
            al = a.lower()
            if _has_cjk(al):
                if al in low:
                    kinds.add(key)
                    break
            else:
                pattern = r"(?<![a-z0-9_])" + re.escape(al).replace(r"\ ", r"[\s_\-]") + r"(?![a-z0-9_])"
                for m in re.finditer(pattern, low):
                    w = _ALIAS_VETO_WINDOW
                    ctx = low[max(0, m.start() - w): m.end() + w]
                    if vetoes and any(v.search(ctx) for v in vetoes):
                        continue        # 该处属同形异义搭配 -> 不算"提到本参数"
                    kinds.add(key)
                    break
    return kinds


def _extract_tables(pdf_path: str) -> list[tuple[int, list[list[str]]]]:
    """提取 PDF 中的表格(三线表等): 返回 [(页码, [[单元格文本...], ...]), ...]。

    论文的超参数常整张表给出, 只靠正则抓正文会漏; 表格是文字层, 无需 OCR。
    """
    out: list[tuple[int, list[list[str]]]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            try:
                tables = page.find_tables()
            except Exception:
                continue
            for t in getattr(tables, "tables", []):
                try:
                    rows = [[("" if c is None else str(c).strip()) for c in row]
                            for row in t.extract()]
                except Exception:
                    continue
                rows = [r for r in rows if any(r)]
                if len(rows) >= 2:
                    out.append((i + 1, rows))
    return out


# ============================================================ 按坐标重组表格行（2026-10-30）
# 为什么必须自己算坐标（这是「25 份论文抽不到结果指标」的真正病根）:
#   ① arXiv 双栏 PDF 的文本层是**按列**输出的。DETR 第 13 页的 Table 3，
#      get_text("text") 给出的是「spatial pos. enc. / output pos. enc. / encoder /
#      decoder / decoder / AP / ∆ / AP50 / ∆ / none / none / learned at input /
#      32.8 / -7.8 / 55.2 / -6.5 / ...」—— 每个单元格单独一行，指标名和它对应的
#      数字隔了几十行，按句子切分永远对不上。
#   ② pymupdf 的 find_tables 在**无边框三线表**上完全失效: 实测 DETR 只抓到 4 张
#      「表」，全是插图/公式区的碎片，真正的 Table 3/4 一张没抓到。
#   唯一可靠的定位信息是每个词的 (x, y): 同一行的词 y 相近，同列的词 x 相近。
#   实测同一套坐标规则把上面那张表完整还原成
#      「encoder || decoder || decoder || AP || ∆ || AP50 || ∆」
#      「none || none || learned at input || 32.8 || -7.8 || 55.2 || -6.5」

#: 同一行: 词与行基准 y 的差不超过这个值（约半个字高，正文约 10pt）
_ROW_TOL = 3.5
#: 同一行内: 相邻词的 x 间隙超过这个值就算换了一列（同列单词间距通常 < 3pt）
_CELL_GAP = 6.0
#: 一个纯数值单元格（允许负号、百分号；不含单位，单位会在同一单元格里跟着数字）
_CELL_NUM = re.compile(r"^[-−–+]?\d{1,4}(?:\.\d{1,4})?%?$")


def _has_gutter(words, width: float, height: float) -> bool:
    """页面中间是否存在一条**贯穿版面的空白竖带**（双栏排版的 gutter）。

    踩过的两个坑，决定了判据长这样:
      ① 按「中间带里有几个词」判 —— 跨栏的宽表格（DETR 第 13 页的 Table 3 就横跨
         整页）会把词摆在中线附近，于是被误判成双栏，表格被拦腰切成左右两半，
         指标名和数字分到不同栏里，比不判栏还糟。
      ② 只看某一处有没有空隙 —— 表格列与列之间的空隙也有 50pt，会被当成 gutter。
    所以改成**纵向分段统计**: 把页面按高度切成 20 段，只有「有内容的分段足够多
    （≥8）」且「其中 60% 以上中间带是空的」才算双栏。稀疏页（表格只占几行）
    一律按单栏处理 —— 宁可少判一次栏，也不能把一张好表切坏。
    """
    lo, hi = width * 0.42, width * 0.58
    seg_n = 20
    filled = 0
    empty = 0
    for i in range(seg_n):
        top, bottom = height * i / seg_n, height * (i + 1) / seg_n
        seg = [w for w in words if w[1] < bottom and w[3] > top]
        if not seg:
            continue
        filled += 1
        if not any(w[2] > lo and w[0] < hi for w in seg):
            empty += 1
    return filled >= 8 and empty >= filled * 0.6


def _page_rows(page) -> list[list[str]]:
    """把一页的文字按坐标重组成「行 × 单元格」。

    返回 [[cell, cell, ...], ...]，行自上而下，行内自左而右。
    双栏页先按中间那条空白竖带判栏，再在每栏内部聚行 —— 不分栏的话，
    左右两栏恰好同高的行会被拼成一行，一个指标名配上另一栏的数字。
    """
    words = page.get_text("words")          # (x0,y0,x1,y1,word,block,line,wno)
    if not words:
        return []
    width = page.rect.width
    mid = width / 2.0
    two_col = _has_gutter(words, width, page.rect.height)
    columns = [(0.0, mid), (mid, width)] if two_col else [(0.0, width)]

    rows: list[list[str]] = []
    for lo, hi in columns:
        col_words = [w for w in words if lo <= (w[0] + w[2]) / 2 < hi]
        if not col_words:
            continue
        col_words.sort(key=lambda w: (round(w[1], 1), w[0]))
        lines: list[list] = []
        cur: list = []
        base = None
        for w in col_words:
            if base is None or abs(w[1] - base) <= _ROW_TOL:
                cur.append(w)
                if base is None:
                    base = w[1]
            else:
                lines.append(cur)
                cur, base = [w], w[1]
        if cur:
            lines.append(cur)
        for ln in lines:
            ln.sort(key=lambda w: w[0])
            cells: list[str] = []
            acc: list[str] = []
            prev_x1 = None
            for w in ln:
                if prev_x1 is not None and w[0] - prev_x1 > _CELL_GAP:
                    cells.append(" ".join(acc))
                    acc = []
                acc.append(w[4])
                prev_x1 = w[2]
            if acc:
                cells.append(" ".join(acc))
            if any(c.strip() for c in cells):
                rows.append(cells)
    return rows


#: 表头单元格 -> 规范指标名。**只认主指标**，子指标各自归到主指标名下
#: （Top-1/Top-5 都算 accuracy、FID/sFID 都算 fid）—— 否则「论文声明的结果」里
#: 会冒出一堆近义词，判定层还要再写一遍同义映射。
#: 差值列（∆ / Δ）不是结果本身，明确排除。
#: ⚠ 词表放宽的代价由「表头至少两个指标列」那道闸兜底，别把数据集名加进来
#:   （GPT-2 那张词相似度表的列名里就有 "AP" = AP news，加进来就误抽了）。
_HEADER_METRIC: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(?:mAP|AP|AP\d{2})$"), "ap"),
    (re.compile(r"^(?:top-?\d{1,3}|top\d|acc(?:uracy)?)$", re.IGNORECASE), "accuracy"),
    (re.compile(r"^precision$", re.IGNORECASE), "precision"),
    (re.compile(r"^recall$", re.IGNORECASE), "recall"),
    (re.compile(r"^F1(?:-?score)?$", re.IGNORECASE), "f1"),
    (re.compile(r"^(?:bleu(?:\d+)?)$", re.IGNORECASE), "bleu"),
    (re.compile(r"^(?:ppl|perplexity)$", re.IGNORECASE), "ppl"),
    (re.compile(r"^(?:m?iou)$", re.IGNORECASE), "miou"),
    (re.compile(r"^psnr$", re.IGNORECASE), "psnr"),
    (re.compile(r"^ssim$", re.IGNORECASE), "ssim"),
    (re.compile(r"^s?FID$", re.IGNORECASE), "fid"),
    # 这里**不加 IGNORECASE**: 表头里的 "is" 是普通英文词，加了会把
    # "this is a table" 这类碎片认成 Inception Score。
    (re.compile(r"^(?:IS|Inception Score)$"), "is"),
]

#: 这些指标**越小越好** —— 取表内代表值时要取最小而不是最大。
_LOWER_IS_BETTER = {"loss", "ppl", "fid", "bits/dim", "wer"}

#: 表格里一个指标的「代表值」怎么取 —— 这一层只做登记，不改任何判定
_TABLE_ROW_CAP = 16          # 一张表最多往下看这么多行


def _header_metric(cell: str):
    """表头单元格 -> 规范指标名（认不出返回 None）。"""
    c = cell.strip().replace("−", "-")
    if not c or len(c) > 12:
        return None
    for pat, name in _HEADER_METRIC:
        if pat.match(c):
            return name
    return None


def _rows_to_results(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    """从坐标行里认出「指标表」并取代表值。

    返回 [(metric, raw_value, snippet), ...]。
    防误报的四道闸（缺一不可）:
      1. 表头行必须含**可识别的指标名**，且该行本身不是数值行；
      2. 数据行与表头的**列数必须一致** —— 这是「按坐标对齐」的核心，
         也是最强的一道: 正文段落切成同样列数的概率极低；
      3. 值必须落在该指标的合法区间（`_RESULT_RANGE`）；
      4. 至少要凑够 2 个有效数据行，单行不算表。
    代表值取法: 越大越好的取表内最大值、越小越好的取最小值 —— 论文主张的
    通常是最优配置，而不是表格第一行的 ablation 变体。snippet 里同时给出表头、
    用到的行数与取值口径，人可以回原文核。
    """
    out: list[tuple[str, str, str]] = []
    n = len(rows)
    for i, cells in enumerate(rows):
        if len(cells) < 3:
            continue
        nums_in_header = sum(1 for c in cells if _CELL_NUM.match(c.strip()))
        if nums_in_header > len(cells) // 2:
            continue                       # 这一行本身是数据行，不是表头
        # 表头里至少要认出**两个**指标列，才认它是指标表。
        # 只认出一个的时候，绝大多数是「列名恰好撞上指标缩写」：
        # 实测 GPT-2 第 8 页那张词相似度表，最后一列的列名就是数据集名 "AP"
        # （AP news），于是被当成平均精度，抽出了 ap=0.321。
        # 正常的指标表都有并列列（AP/AP50、Top-1/Top-5、PPL/BLEU），代价很小。
        # ⚠ 必须**先数命中个数、再去重**：DETR 的表头是 AP/∆/AP50/∆，
        # 两个指标都归一到 ap，先去重就只剩一列，会被这条闸误杀。
        raw_hits = []
        for ci, c in enumerate(cells):
            name = _header_metric(c)
            if not name:
                continue
            if re.search(r"\s\d", c):
                # 列名里夹着空格+数字（"Accuracy 60"）—— 真表头不长这样，
                # 这是正文句子被坐标切成几段后的碎片。实测 DINOv2 第 3 页
                # 的「Accuracy | Accuracy 60 | Accuracy」就是这么被误认成表头的。
                continue
            raw_hits.append((ci, name, c.strip()))
        # 真表的并列列写的是**不同**名字（AP/AP50、Top-1/Top-5）。
        # 认出来的指标全是同一个词，说明这张"表"其实是正文句子被切成的几段
        # —— DINOv2 p3 只剩两个 "Accuracy"，就是这种情况。
        if len({t for _ci, _n, t in raw_hits}) < 2:
            continue
        if len(raw_hits) < 2:
            continue
        col_map: dict[int, str] = {}
        for ci, name, _t in raw_hits:
            if name not in col_map.values():
                col_map[ci] = name         # 同一指标只认第一列（AP 在前、AP50 在后）
        width = len(cells)
        # 收集时就带上**这一行的原文** —— 最后取的是最大值/最小值，可能出自第 3 行
        # 而不是首行；只给首行的话，人回原文核对时会对不上，等于证据不可核。
        vals: dict[str, list[tuple[float, str]]] = {m: [] for m in col_map.values()}
        used: list[str] = []
        for j in range(i + 1, min(i + 1 + _TABLE_ROW_CAP, n)):
            nxt = rows[j]
            if len(nxt) != width:
                break                      # 列数一变，这张表就结束了
            row_txt = " | ".join(nxt)
            got = False
            for ci, name in col_map.items():
                raw = nxt[ci].strip() if ci < len(nxt) else ""
                if not _CELL_NUM.match(raw):
                    continue
                num = normalize_number(raw)
                if num is None:
                    continue
                lo, hi = _RESULT_RANGE.get(name, (0.0, 100.0))
                if not (lo <= num <= hi):
                    continue
                vals[name].append((num, row_txt))
                got = True
            if got:
                used.append(row_txt)
            elif used:
                break                      # 连续两行都取不到值 -> 表结束
        if len(used) < 2:
            continue
        header_txt = " | ".join(cells)
        for name, vs in vals.items():
            if not vs:
                continue
            how = "最小" if name in _LOWER_IS_BETTER else "最大"
            pick, src_row = (min(vs) if name in _LOWER_IS_BETTER else max(vs))
            snip = (f"表头: {header_txt} ｜ 表内 {len(vs)} 个有效行，"
                    f"取{how}值 {pick:g} ｜ 取值行: {src_row}")
            out.append((name, f"{pick:g}", snip[:240]))
        if out:
            break                          # 一张表只认第一个表头
    return out


def _extract_rows(pdf_path: str) -> dict[int, list[list[str]]]:
    """整份 PDF 的「坐标行」表：{页码: [[cell, ...], ...]}。"""
    out: dict[int, list[list[str]]] = {}
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            try:
                rows = _page_rows(page)
            except Exception:      # noqa: BLE001
                continue
            if rows:
                out[i + 1] = rows
    return out


# 表头里出现这些词才认为该表与超参数有关(否则多为插图区被误检为表格)
_TABLE_HEADER_HINT = re.compile(
    r"(learning rate|batch|epoch|optimizer|dropout|weight decay|momentum|"
    r"hyper-?parameter|parameter|setting|configuration|train|"
    r"学习率|批次|批量|迭代|优化器|超参|参数|配置|设置)",
    re.IGNORECASE,
)
_PARAM_CELL = re.compile(r"^[A-Za-z_][A-Za-z0-9_ \-]{1,24}$")
_SWEEP_TABLE = re.compile(
    r"(灵敏度|敏感性|对比|比较|扫描|网格|sensitivit|compar|sweep|ablation|search)",
    re.IGNORECASE,
)


def _rows_to_findings(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    """把可能含超参数的表格行转成 (canonical_key, 值, 行文本) 三元组。

    收紧判据(arXiv 双栏 PDF 里 find_tables 常把插图区误判为表格):
    - 表头必须出现参数语境词;
    - 参数名单元格必须是"纯参数名"(不含数字/长句)。
    """
    header = " ".join(rows[0])[:300]
    if not _TABLE_HEADER_HINT.search(header):
        return []
    hits: list[tuple[str, str, str]] = []
    for row in rows:
        cells = list(row)
        for ci, cell in enumerate(cells):
            if not cell or len(cell) > 30 or not _PARAM_CELL.match(cell.strip()):
                continue
            key = canonical_key(cell.strip()) or canonical_key(cell.strip().replace(" ", "_"))
            if key is None:
                continue
            for c in cells[ci + 1:] + cells[:ci]:
                c = c.strip()
                if not c or len(c) > 40:
                    continue
                if key in _TEXT_VALUED_KEYS:
                    if re.match(r"^[A-Za-z_][\w \-\.]{1,40}$", c):
                        hits.append((key, c, " | ".join(cells)))
                        break
                elif normalize_number(c) is not None:
                    hits.append((key, c, " | ".join(cells)))
                    break
    return hits


def _find_section(lines_before: list[str]) -> str:
    """从当前行向上找最近的章节标题。"""
    for line in reversed(lines_before):
        line = line.strip()
        if len(line) < 3 or len(line) > 45:
            continue
        m = _HEADING_RE.match(line)
        if m and not line.endswith("."):
            return line
    return ""


def _snippet_around(text: str, start: int, end: int, width: int = 80) -> str:
    """截取匹配点前后的原文片段作为证据展示。"""
    s = max(0, start - 10)
    e = min(len(text), end + width)
    snip = " ".join(text[s:e].split())
    return snip


# ---------------------------------------------------------------- 参考文献否决
# 参考文献条目里会大量出现与参数同形的词, 必须整段排除。实测(2026-09-28, MAE 论文):
#   "[20] ... Kaiming He. Accurate, large minibatch SGD: Training ImageNet
#    in 1 hour. arXiv:1706.02677, 2017."
# 其中 **SGD** 被当成了"论文选的优化器", 直接污染「优化器」结论。
#
# 两道判据都取**高精度**特征, 避免误伤正文与附录(注意: ICLR/NeurIPS 论文的附录常在
# 参考文献**之后**, 所以**不能**用"References 标题之后一律排除"这种粗规则 —— 会连带
# 砍掉附录里的超参数表):
#   ① 邻近窗口出现 arXiv:xxxx.xxxxx / doi:10.xxxx / "pp. 123–130" / "vol. 1, no. 2"
#      —— 这些**只在参考文献里**出现;
#   ② 该行以 "[12] " 编号条目开头。
#
# ⚠️ 刻意**不**收 "Proceedings of / Conference on" 这类会场名 —— 实测(2026-09-28):
# CVPR/ICCV 论文的**页脚**就写着 "Proceedings of the IEEE/CVF Conference on ...",
# 把它们当引用标识会误伤正文(NeRF/PointNet 的**模型名**因此被错杀)。
_BIB_MARKS = re.compile(
    r"arXiv:\s*\d{4}\.\d{4,5}|doi:\s*10\.\d{4,}|"
    r"\bpp\.\s*\d+\s*[-–—]\s*\d+|\bvol\.\s*\d+\s*,\s*no\.\s*\d+",
    re.IGNORECASE)
_BIB_ENTRY_LINE = re.compile(r"^\s*\[\d{1,3}\]\s*\S")


def _in_bibliography(text: str, start: int, end: int, lines_around: int = 2) -> bool:
    """start..end 处是否落在参考文献条目里。

    ⚠️ 判定窗口按**行**取(匹配行 ± lines_around 行), 不用固定字符窗口 ——
    实测(2026-09-28): arXiv 论文**首页页边**会印 "arXiv:2003.08934v2 [cs.CV] ...",
    用字符窗口会把它算进判定范围, 于是 NeRF / PointNet 的**模型名**被误判成引用内容。
    参考文献标识只与它同一段/邻行共现, 按行取既准又不误伤。
    """
    p = start
    for _ in range(lines_around):
        cut = text.rfind("\n", 0, max(0, p - 1))
        if cut < 0:
            p = 0
            break
        p = cut
    q = end
    for _ in range(lines_around + 1):
        nxt = text.find("\n", q)
        if nxt < 0:
            q = len(text)
            break
        q = nxt + 1
    if _BIB_MARKS.search(text[p:q]):
        return True
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start: line_end if line_end >= 0 else len(text)]
    return bool(_BIB_ENTRY_LINE.match(line))


#: 首页页眉 / 版权行：排在正文最上面，但**不是标题**。
# 实证 2026-10-29：ICLR 投稿模板的首页第一行是「Published as a conference paper at ICLR 2018」，
# 被当成标题写进记忆库 → 好几篇不同论文的"标题"完全一样，记忆索引像"同一篇存了多条"。
_BANNER_LINE = re.compile(
    r"^(published\s+as\s+a\s+conference\s+paper"
    r"|under\s+review\s+as\s+a\s+conference\s+paper"
    r"|proceedings\s+of|to\s+appear\s+in|accepted\s+(at|by)|preprint\b"
    r"|arxiv:|copyright\b|©|all\s+rights\s+reserved|doi:|https?://)",
    re.IGNORECASE)


def parse_paper(pdf_path: str) -> PaperInfo:
    """解析论文 PDF, 输出 PaperInfo(参数与结果均带页码证据)。"""
    info = PaperInfo()
    try:
        pages, info.page_coverage = _extract_pages(pdf_path)
    except FileNotFoundError:
        # 路径本身不存在 —— 这是**用户填错了**，必须明确报错。
        # 静默出一份空报告比报错更糟: 用户会以为"审计做完了, 论文确实没参数"。
        raise
    except Exception as e:  # noqa: BLE001
        # ⚠ pymupdf 抛的是它自己的 FileNotFoundError（继承 RuntimeError），
        # 不是内置的 FileNotFoundError —— 上面那个 except 拦不住，这里再按类型名认一次。
        if type(e).__name__ == "FileNotFoundError":
            raise
        # 打不开的 PDF(伪装成 .pdf 的文本、损坏文件、加密文件)不该让整次审计炸掉:
        # 返回空页集, 让下游按"没读到论文"处理(可信度低、结论标注材料不足),
        # 并把原因如实写进页覆盖表 —— 用户能看见"为什么什么都没读出来"。
        pages = []
        info.page_coverage = [{"page": 0, "chars": 0, "images": 0,
                               "status": "unreadable",
                               "note": f"论文读不出来（{type(e).__name__}）: {e}"}]
    info.page_count = len(pages)
    info.page_texts = pages          # 供抽取 Agent 回原文核对

    # ---- 表格通道: 三线表里的超参数(纯文字层, 无需 OCR)
    tables_by_page: dict[int, list[list[list[str]]]] = {}
    try:
        for page_no, rows in _extract_tables(pdf_path):
            tables_by_page.setdefault(page_no, []).append(rows)
    except Exception:  # noqa: BLE001
        # 与上面的页解析同款: 文件打不开时表格通道也读不了, 跳过即可,
        # 不能因为「同一份坏文件被第二次打开」再抛一次。
        tables_by_page = {}

    # ---- 坐标行通道: 结果指标表（无边框三线表，find_tables 抓不到的那种）
    # 与上面的表格参数通道分开取: 那个走 pymupdf 的自动识别（适合有框线的参数表），
    # 这个走自己的坐标重建（适合无框线的指标表）。两者互补，不互相替代。
    rows_by_page: dict[int, list[list[str]]] = {}
    try:
        rows_by_page = _extract_rows(pdf_path)
    except Exception:  # noqa: BLE001
        # 同上: 坏文件已经在上面的页解析里报过一次，这里不再抛
        rows_by_page = {}

    numeric_pats: list[tuple[str, re.Pattern]] = []
    for key, (_disp, aliases) in PARAM_SYNONYMS.items():
        # 文本型参数交给 _TEXT_PARAM_PATTERNS 处理, 不做数值匹配
        # (避免把 "model 25" 这类正文数字当成模型规模)
        if key in ("optimizer", "scheduler", "model"):
            continue
        for p in _param_patterns(aliases):
            numeric_pats.append((key, p))

    # 疑似声明计数(参考指标): 左侧像参数名的 "xx = 1.5" / "学习率为0.4" 句式总数
    info.decl_total = sum(len(_DECL_RE.findall(text)) for _p, text in pages)

    # 参数种类覆盖(覆盖率分母): 正文里"提到过"的参数种类
    # 例如论文提到 batch size 但没给出数值 -> 计入分母不计入分子
    full_text = "\n".join(t for _p, t in pages)
    info.alias_kinds = sorted(_alias_kinds_in_text(full_text))

    def _accept(  # type: ignore[no-untyped-def]
            key: str, m, page_no: int, lines: list[str], text: str):
        """收集一处声明: 合理性校验 -> params_all 全量 -> params 首个有效值。"""
        _accept_span(key, m.group(1), m.start(), m.end(), page_no, lines, text)

    def _accept_span(  # type: ignore[no-untyped-def]
            key: str, raw: str, start: int, end: int,
            page_no: int, lines: list[str], text: str,
            guard_text: Optional[str] = None):
        # guard_text: 拼接了**下一页开头**的文本, 只用于"参考文献/表格串列"两类上下文判据。
        # 为什么需要: 实测(2026-09-28, DeiT) 附表跨页, 匹配落在页尾,
        # 右邻的多列取值在下一页 —— 只看本页会漏判, 于是 `learning rate 0.003`（另一列的
        # 值, 属于 ViT-B）被当成本文的值, 还生成了错补丁。
        guard_text = guard_text if guard_text is not None else _guard[0]
        norm = normalize_number(raw)
        section = _find_section(lines[: text[:start].count("\n") + 1])
        loc = f"Page {page_no}" + (f" · {section}" if section else "")
        if _in_bibliography(guard_text, start, end):
            info.rejected.append({
                "key": key, "value": raw, "location": loc,
                "reason": "命中处位于参考文献条目, 疑似引用内容而非论文设置",
            })
            return
        _veto = _num_ctx_veto(key, text[max(0, start - 90): end + 90])
        if _veto:
            info.rejected.append({
                "key": key, "value": raw, "location": loc, "reason": _veto,
            })
            return
        _veto = _num_value_veto(key, raw, guard_text, start, end)
        if _veto:
            info.rejected.append({
                "key": key, "value": raw, "location": loc, "reason": _veto,
            })
            return
        if _in_table_run(guard_text, start, end):
            info.rejected.append({
                "key": key, "value": raw, "location": loc,
                "reason": "位于表格的多列取值中（表头后跟一串裸数字），"
                          "无法确定哪一列对应本实验配置，已丢弃",
            })
            return
        plausible = is_plausible(key, raw)
        if plausible is False:
            info.rejected.append({
                "key": key, "value": raw, "location": loc,
                "reason": "超出领域先验范围, 疑似错位抽取",
            })
            return
        existing = info.params_all.setdefault(key, [])
        if any(e.location == loc and e.normalized == norm for e in existing):
            return
        snippet = _snippet_around(text, start, end)
        ev = Evidence(
            source="paper", location=loc, snippet=snippet,
            raw_value=raw, normalized=norm,
            # 灵敏度/分组/扩展语境中出现的取值: 不计入内部一致性判定
            context_excluded=bool(_CONTEXT_EXCLUDE.search(snippet)),
            parser="text",
        )
        existing.append(ev)
        if key not in info.params:
            info.params[key] = ev

    # 数据集候选: 逐页收集, 循环结束后**全篇择优**(见下方"数据集择优")
    dataset_cands: list[tuple[int, str, Evidence]] = []

    # 当前页 + 下一页开头, 供"参考文献/表格串列"判据使用(见 _accept_span 注释)。
    # 用可变容器是为了不改动所有 _accept 调用点的签名。
    _guard: list[str] = [""]

    for _pi, (page_no, text) in enumerate(pages):
        _nxt = pages[_pi + 1][1] if _pi + 1 < len(pages) else ""
        _guard[0] = text + "\n" + _nxt
        lines = text.splitlines()

        # ---- 标题/摘要(首页启发式, 跳过纯数字页码行)
        if page_no == 1:
            non_empty = [l.strip() for l in lines
                         if l.strip() and not l.strip().isdigit() and len(l.strip()) > 4]
            # 页眉/版权行不是标题。实证 2026-10-29：ICLR 投稿模板首页第一行是
            # 「Published as a conference paper at ICLR 2018」，被当成标题写进记忆库，
            # 于是**好几篇不同论文的"标题"一模一样**，记忆索引看着像"同一篇存了多条"。
            _t = [l for l in non_empty if not _BANNER_LINE.search(l)]
            if _t:
                info.title = _t[0]
            abs_m = re.search(r"\babstract\b[\s\-—:]*(.{50,1200}?)(?:\n\s*\n|\b1\.?\s*Introduction\b|\bI\.\s*INTRODUCTION\b)",
                              text, re.IGNORECASE | re.DOTALL)
            if abs_m:
                info.abstract = " ".join(abs_m.group(1).split())
            elif "摘要" in text:
                abs_zh = re.search(r"摘要[\s:：]*(.{50,1500}?)(?:关键词|关键字)", text, re.DOTALL)
                if abs_zh:
                    info.abstract = " ".join(abs_zh.group(1).split())

        # ---- 数值型参数(全量收集, 支持内部交叉核对)
        for key, pat in numeric_pats:
            for m in pat.finditer(text):
                _accept(key, m, page_no, lines, text)

        # ---- 数字在前的句式兜底(如 "train for 100 epochs" / "9 枚深弹")
        for key, pats in _REVERSE_PATTERNS.items():
            for pat in pats:
                for m in pat.finditer(text):
                    _accept(key, m, page_no, lines, text)

        # ---- 别名与数值被短语隔开的兜底(如 "set the dropout rate on CIFAR10 to 0.1")
        #      仅在该参数**本页之前还没取到值**时启用, 控制误报面
        for key, (_disp, aliases) in PARAM_SYNONYMS.items():
            if key in info.params or key in _TEXT_VALUED_KEYS:
                continue
            got = gap_value_after_alias(text, aliases)
            if got:
                _accept_span(key, got[0], got[1], got[2], page_no, lines, text)

        # ---- 文本型参数(optimizer / scheduler / model)
        for key, pats in _TEXT_PARAM_PATTERNS.items():
            if key in info.params:
                continue
            # 收集全部命中, 记录 (是否泛称, 取值, 匹配对象) 后再择优——
            # 片段必须取自"被选中的那一处", 否则证据与取值对不上(自证会失败)
            hits: list[tuple[int, str, "re.Match"]] = []
            for pat in pats:
                for m in pat.finditer(text):
                    val = m.group(m.lastindex) if m.lastindex else m.group(0)
                    val_s = str(val).strip()
                    _sec0 = _find_section(lines[: text[: m.start()].count("\n") + 1])
                    _loc0 = f"Page {page_no}" + (f" · {_sec0}" if _sec0 else "")
                    # 参考文献条目里的同形词不算(实测: MAE 从引用里抓到 SGD 当优化器)
                    if _in_bibliography(text, m.start(), m.end()):
                        info.rejected.append({
                            "key": key, "value": val_s, "location": _loc0,
                            "reason": "命中处位于参考文献条目, 疑似引用内容",
                        })
                        continue
                    _why = _text_param_veto(
                        key, val_s, text[max(0, m.start() - 90): m.end() + 90])
                    if _why:
                        info.rejected.append({
                            "key": key, "value": val_s, "location": _loc0,
                            "reason": _why,
                        })
                        continue
                    generic = val_s.lower() in (
                        "transformer", "transformers", "gpt", "model",
                        "architecture", "backbone")
                    hits.append((1 if generic else 0, val_s, m))
            if not hits:
                continue
            hits.sort(key=lambda h: h[0])           # 具体架构名优先于泛称
            _score, raw, m = hits[0]
            section = _find_section(lines[: text[: m.start()].count("\n") + 1])
            _ev = Evidence(
                source="paper",
                location=f"Page {page_no}" + (f" · {section}" if section else ""),
                snippet=_snippet_around(text, m.start(), m.end()),
                raw_value=raw,
                normalized=str(raw).strip().lower(),
                parser="text",
            )
            info.params[key] = _ev
            # 同步登记到 params_all(否则覆盖率会把"已取到值"的种类误判为未取到)
            info.params_all.setdefault(key, []).append(_ev)

        # ---- 数据集(候选; 全篇择优见循环之后)
        for m in _DATASET_RE.finditer(text):
            if _in_bibliography(text, m.start(), m.end()):
                continue
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start: line_end if line_end != -1 else len(text)]
            if not _DATASET_CONTEXT.search(line):
                continue
            name = m.group(1)
            ver = m.group(2)
            raw = name + (f" v{ver}" if ver else "")
            section = _find_section(lines[: text[: m.start()].count("\n") + 1])
            dataset_cands.append((
                0 if _DATASET_TRAIN_CTX.search(line) else 1,
                name.lower(),
                Evidence(
                    source="paper",
                    location=f"Page {page_no}" + (f" · {section}" if section else ""),
                    snippet=_snippet_around(text, m.start(), m.end()),
                    raw_value=raw,
                    normalized=re.sub(r"\s+", "", raw.lower()),
                    parser="text",
                ),
            ))

        # ---- 实验结果
        # 两组模式, 走两条不同的路（2026-10-29 由 28 份抽不到结果的论文逼出来的）:
        #   ① 指标词在前（`accuracy of 76.5%`）—— 模式本身够严, 全页扫;
        #   ② 数值在前（`42 AP` / `45.1% COCO AP`）—— **只在含结果动词的句子内扫**,
        #      否则会把 `Fig. 1 AP`、`AP@IoU=0.5`、`approximately 1 AP` 全当精度。
        for metric, pat in _RESULT_PATTERNS:
            if metric in info.results:
                continue
            for m in pat.finditer(text):
                if _in_bibliography(text, m.start(), m.end()):
                    continue
                raw = m.group(1)
                if _result_veto(metric, text, m.start(1), m.end(1), raw):
                    continue
                section = _find_section(lines[: text[: m.start()].count("\n") + 1])
                info.results[metric] = Evidence(
                    source="paper",
                    location=f"Page {page_no}" + (f" · {section}" if section else ""),
                    snippet=_snippet_around(text, m.start(), m.end()),
                    raw_value=raw,
                    normalized=normalize_number(raw),
                    parser="text",
                )
                break

        for s_off, sent in _sentences_with_pos(text):
            if len(info.results) >= 4:
                break
            if not _RESULT_VERB.search(sent):
                continue
            for metric, pat in _NUMFIRST:
                if metric in info.results:
                    continue
                for m in pat.finditer(sent):
                    a, b = s_off + m.start(), s_off + m.end()
                    if _in_bibliography(text, a, b):
                        continue
                    raw = m.group(1)
                    if _result_veto(metric, text, s_off + m.start(1), s_off + m.end(1), raw):
                        continue
                    section = _find_section(lines[: text[:a].count("\n") + 1])
                    info.results[metric] = Evidence(
                        source="paper",
                        location=f"Page {page_no}" + (f" · {section}" if section else ""),
                        snippet=_snippet_around(text, a, b),
                        raw_value=raw,
                        normalized=normalize_number(raw),
                        parser="text",
                    )
                    break

        # ---- 表格里的结果指标(第三条通道, 2026-10-30)
        # 为什么必须单开一条: 上面两条都要求「指标词与数字在同一句里」，
        # 而表格的文本层是按列输出的 —— 两者相隔几十行，永远凑不到一个句子里。
        # 实测 26 份论文一条结果都抽不到，根因就在这里（不是论文没写）。
        # 放在两条文本通道**之后**: 文本有句子可读，证据强度高于表格单元格。
        if len(info.results) < 4:
            for metric, raw_v, snip in _rows_to_results(rows_by_page.get(page_no, [])):
                if metric in info.results:
                    continue
                norm = normalize_number(raw_v)
                if norm is None:
                    continue
                info.results[metric] = Evidence(
                    source="paper",
                    location=f"Page {page_no} · 表格",
                    snippet=snip,
                    raw_value=raw_v,
                    normalized=norm,
                    parser="table",
                )

        # ---- 表格参数(三线表/参数表): 论文超参数常整表给出, 正文正则抓不到
        for ti, rows in enumerate(tables_by_page.get(page_no, []), 1):
            header = " ".join(rows[0])[:200]
            is_sweep = bool(_SWEEP_TABLE.search(header))
            for key, value, row_text in _rows_to_findings(rows):
                norm = normalize_number(value)
                if key in _TEXT_VALUED_KEYS and norm is None:
                    norm = re.sub(r"\s+", "", value.lower())
                if norm is None:
                    continue
                loc = f"Page {page_no} · 表格{ti}"
                existing = info.params_all.setdefault(key, [])
                if any(e.location == loc and e.raw_value == value for e in existing):
                    continue
                # 灵敏度/对比表里的多个取值是有意为之, 不计入内部不一致判定
                multi_col = sum(1 for c in row_text.split("|") if normalize_number(c.strip()))
                ev = Evidence(
                    source="paper", location=loc,
                    snippet=row_text[:200], raw_value=value, normalized=norm,
                    context_excluded=is_sweep or multi_col > 2,
                    parser="table",
                )
                existing.append(ev)
                if key not in info.params:
                    info.params[key] = ev

    # ---- 数据集: 全篇择优
    # 为什么不"第一个命中就用": 摘要里常先出现**检测/迁移**用的数据集。实测(2026-09-28):
    #   · Swin 摘要先写 "58.7 box AP on COCO test-dev" → dataset 抽成 COCO(主实验是 ImageNet);
    #   · MAE 先写 "results on COCO validation images"(迁移实验) → 同样抽成 COCO(主实验是 ImageNet)。
    # 改为全篇收集后择优:
    #   ① 训练语境(trained on / dataset / 训练)优先于泛语境;
    #   ② 同前提下取**在全文出现次数最多**的(主实验数据集必然反复出现);
    #   ③ 再同则取最靠前的(位置字符串天然有序)。
    # 表格通道给出的数据集保持优先(表格比正文更明确)。
    if dataset_cands:
        _cur = info.params.get("dataset")
        if _cur is None or _cur.parser != "table":
            dataset_cands.sort(key=lambda c: (c[0], -full_text.lower().count(c[1]),
                                              c[2].location))
            _best = dataset_cands[0][2]
            info.params["dataset"] = _best
            # 只把**选中**的那一处记进 params_all。
            # 为什么不全登记: 论文会在不同任务里提不同数据集(如 NeRF 的 LLMR/DeepVoxels/
            # Blender), 全登记会让"内部不一致"判定凭空多出一条噪音结论 ——
            # 实测(2026-09-28) 03_NeRF 因此多了一条 Dataset 内部不一致。
            _lst = info.params_all.setdefault("dataset", [])
            if not any(e.location == _best.location and e.raw_value == _best.raw_value
                       for e in _lst):
                _lst.append(_best)

    return info
