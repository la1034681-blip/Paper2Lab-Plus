"""审计金标准集 —— 评测的"标准答案"。

两条来源，都**不是**取自本工具的输出（否则就是自己给自己判卷）：
  1. **真实样本**：`Paper2Lab测试集\\01_DDPM ... 05_ViT` 五套 arXiv 论文 + 官方代码包。
     每套的 `expect`（期望抓到的参数）依据该论文正文与官方开源实现的**公开训练超参**
     人工整理 —— 例如 DDPM 的 batch 128 / lr 2e-4 / Adam；ViT 的 warmup 10k steps。
  2. **诱饵样本**：`demo/edge_code.zip` 刻意写入的"看起来像不一致、其实合法"的值，
     用来度量**精度**（不该报的有没有被报出来）。

判分口径说明（重要）
--------------------
`expect` 只用于算**召回**（这些参数里抓到了几成），**不做"必须 100%"的硬断言** ——
因为论文版本不同，个别超参可能只在附录或压根没写。评测的意义是**给出一个可比的数**，
不是当测试通过与否的开关。

样本类型 `kind`：
    real     五套真实 arXiv 论文（主考卷）
    modeling 数学建模题（参数域与 ML 论文不同，单独标注，避免拿错模板）
    demo     项目自带演示样本（回归）
    edge     诱饵样本（考精度）
    external 仓库外的样本（换机器可能不存在 → 自动跳过）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

#: 项目根（`paper2lab/evalsuite/goldset.py` → 上溯两级）
ROOT = Path(__file__).resolve().parent.parent.parent

#: 真实测试集目录（与 tests/monitor_samples.py 同一处）
REAL_ROOT = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")

#: 论文/实现的公开训练超参依据（写进报告，供答辩核对）
EVIDENCE_NOTE = "依据各论文正文 + 官方开源实现的公开训练超参人工整理，非取自本工具输出"


@dataclass
class GoldSample:
    """一套考卷。"""

    sid: str                        # 样本 id
    pdf: Path                       # 论文路径
    code: Path                      # 代码包路径
    expect: List[str] = field(default_factory=list)   # 期望抓到的规范化参数键（召回分母）
    decoy: List[str] = field(default_factory=list)    # 诱饵参数键（精度抓手）
    kind: str = "real"              # real / demo / edge / external
    results: Optional[Path] = None  # 用户实验结果 json（可选）
    note: str = ""

    def resolve(self) -> Optional["ResolvedSample"]:
        """解析为可跑样本；文件缺失返回 None（调用方据此跳过）。"""
        if not self.pdf.exists() or not self.code.exists():
            return None
        return ResolvedSample(
            sid=self.sid, pdf=str(self.pdf), code=str(self.code),
            results=str(self.results) if (self.results and self.results.exists()) else None,
            expect=list(self.expect), decoy=list(self.decoy),
            kind=self.kind, note=self.note,
        )


@dataclass
class ResolvedSample:
    """已落地的考卷（路径已展开、文件已确认存在）。"""

    sid: str
    pdf: str
    code: str
    expect: List[str]
    decoy: List[str]
    kind: str
    results: Optional[str] = None
    note: str = ""

    def to_meta(self) -> dict:
        """样本元数据（值统一为字符串，与 SciEvalKit 的 EvalSample.metadata 类型一致）。"""
        return {
            "sid": self.sid, "kind": self.kind,
            "expect": ",".join(self.expect), "decoy": ",".join(self.decoy),
            "note": self.note,
        }


# ---------------------------------------------------------------- 五套真实论文
# expect 的依据写在 note 里；不做硬断言，只算召回。
_REAL = [
    GoldSample(
        sid="01_DDPM",
        pdf=REAL_ROOT / "01_DDPM" / "paper.pdf",
        code=REAL_ROOT / "01_DDPM" / "code.zip",
        expect=["learning_rate", "batch_size", "optimizer", "warmup"],
        note="DDPM: Adam, batch 128, lr 2e-4(线性 warmup), T=1000 timesteps",
    ),
    GoldSample(
        sid="02_DETR",
        pdf=REAL_ROOT / "02_DETR" / "paper.pdf",
        code=REAL_ROOT / "02_DETR" / "code.zip",
        expect=["learning_rate", "batch_size", "epochs", "optimizer", "weight_decay", "dropout"],
        note="DETR: AdamW, batch 64, lr 1e-4(backbone 1e-5), 300 epochs, dropout",
    ),
    GoldSample(
        sid="03_NeRF",
        pdf=REAL_ROOT / "03_NeRF" / "paper.pdf",
        code=REAL_ROOT / "03_NeRF" / "code.zip",
        expect=["learning_rate", "batch_size", "iterations", "optimizer"],
        note="NeRF: Adam, batch 4096 rays, lr 5e-4, 100-300k iterations",
    ),
    GoldSample(
        sid="04_PointNet",
        pdf=REAL_ROOT / "04_PointNet" / "paper.pdf",
        code=REAL_ROOT / "04_PointNet" / "code.zip",
        expect=["learning_rate", "batch_size", "epochs", "optimizer"],
        note="PointNet: Adam, batch 32, lr 0.001, 250 epochs",
    ),
    GoldSample(
        sid="05_ViT",
        pdf=REAL_ROOT / "05_ViT" / "paper.pdf",
        code=REAL_ROOT / "05_ViT" / "code.zip",
        results=REAL_ROOT / "05_ViT" / "results.json",
        expect=["learning_rate", "batch_size", "optimizer", "warmup", "weight_decay"],
        note="ViT: AdamW, batch 4096, lr 3e-3, warmup 10k steps, wd 0.1-0.3",
    ),
]

# ---------------------------------------------------------------- 自带样本
_SELF = [
    GoldSample(
        sid="demo",
        pdf=ROOT / "demo" / "paper.pdf",
        code=ROOT / "demo" / "project.zip",
        results=ROOT / "demo" / "results.json",
        expect=["learning_rate", "batch_size", "epochs", "optimizer", "dataset", "seed"],
        kind="demo",
        note="项目演示样本（小论文 + 小代码），期望项覆盖主要训练超参与随机性",
    ),
    GoldSample(
        sid="D题",
        pdf=ROOT / "demo" / "D题论文.pdf",
        code=ROOT / "demo" / "D题代码.zip",
        expect=["kill_radius", "num_bombs", "sigma_xy", "sigma_z", "depth_est", "min_depth"],
        kind="modeling",
        note="比赛 D 题「反潜航空深弹命中概率」—— **数学建模题，不是 ML 论文**；"
             "期望项按该题物理参数域整理（杀伤半径 / 弹数 / 投放误差 / 深度估计）",
    ),
    GoldSample(
        sid="E题",
        pdf=Path(r"C:\Users\lijiahui\Desktop\E题论文.pdf"),
        code=Path(r"C:\Users\lijiahui\Desktop\E题作答.zip"),
        expect=["num_clusters", "threshold", "epochs"],
        kind="modeling",
        note="比赛 E 题「K-means 聚类 + Webster 配时 + 阈值判别」—— **数学建模题**；"
             "期望项按该题算法参数域整理（聚类数 / 判别阈值 / 迭代）（仓库外，缺失自动跳过）",
    ),
]

# ---------------------------------------------------------------- 诱饵样本
# edge_code.zip 是刻意构造的"看起来像不一致、其实合法"的取值 ——
# 审计**不该**把它们升级成高风险问题（这类误报最能毁掉信任）。
_EDGE = [
    GoldSample(
        sid="edge_诱饵",
        pdf=ROOT / "demo" / "paper.pdf",
        code=ROOT / "demo" / "edge_code.zip",
        expect=["learning_rate"],
        decoy=[],
        kind="edge",
        note="边缘集：该报的是 learning_rate 跨文件不一致（train.py 0.001 vs config.json 0.0001）；"
             "另含错位值 learning_rate=2024（显然不是学习率），应被**拒收**而非参与比对，"
             "该防线由 rejected 指标度量",
    ),
]

#: 全套考卷（按此顺序跑）
GOLD_SAMPLES: List[GoldSample] = _REAL + _SELF + _EDGE


def list_resolved() -> List[ResolvedSample]:
    """返回当前机器上实际可跑的考卷。"""
    out: List[ResolvedSample] = []
    for g in GOLD_SAMPLES:
        r = g.resolve()
        if r is not None:
            out.append(r)
    return out


def missing() -> List[str]:
    """返回本机缺失的考卷 id（换机器时会少样本，如实列出）。"""
    return [g.sid for g in GOLD_SAMPLES if g.resolve() is None]


def describe() -> dict:
    """给报告用的一张总表。"""
    ok = list_resolved()
    return {
        "total": len(GOLD_SAMPLES),
        "available": len(ok),
        "missing": missing(),
        "by_kind": {
            k: sum(1 for s in ok if s.kind == k)
            for k in sorted({s.kind for s in ok})
        },
        "evidence_note": EVIDENCE_NOTE,
        "samples": [g.sid for g in GOLD_SAMPLES],
    }
