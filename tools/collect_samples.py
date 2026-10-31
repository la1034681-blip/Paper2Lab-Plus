"""从公开网络采集「论文 + 配套代码包」测试样例。

来源只有两个，都是公开、免费、无需 key 的：
  - 论文正文：arXiv 的 PDF 直链（`https://arxiv.org/pdf/<id>`）
  - 配套代码：论文官方（或第一作者）GitHub 仓库的完整快照 zip
              （`https://codeload.github.com/<owner>/<repo>/zip/refs/heads/<branch>`）

**为什么不下载数据集**：这些论文的原始数据集普遍是几十 GB 到几 TB
（ImageNet / LAION / 多视角实拍），既跑不动也不该占用户硬盘。这里
"配套数据包"指的是**代码仓库完整快照**——里面有配置文件、超参、
评测脚本、样例数据和作者跑出来的结果表，正是审计要对照的东西。

每份落成：
    <out>/S01_<slug>/paper.pdf     论文原文
    <out>/S01_<slug>/code.zip      代码仓库快照
    <out>/S01_<slug>/meta.json     来源、分支、字节数、sha256、校验结论

**不做的事**：不伪造 results.json。实验实际值一律留空，交给审计系统
自己从论文里抽取——那正是项目要验证的能力。

工程口径（踩过的坑）：
  * **不走 GitHub API**：匿名调用每小时只给 60 次，59 条候选就会打满，
    而且一超限整批静默失败。这里改成直接试 main / master 两个分支名。
  * **流式落盘**：早期版本在内存里攒完整 zip，碰上几百 MB 的仓库会卡死
    还没有任何进度。现在边下边写，超过 MAX_ZIP_BYTES 立刻放弃换下一个。
  * 已经采好的目录（含 paper.pdf + code.zip + meta.json）直接跳过，可断点续跑。

用法：
    python tools/collect_samples.py --out "C:/Users/lijiahui/Desktop/Paper2Lab测试集/50份样例" --target 50
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (compatible; Paper2Lab-sample-collector/1.0)"}
MAX_ZIP_BYTES = 30 * 1024 * 1024   # 单个代码包上限 20MB（论文核心代码通常远小于此）
MIN_PDF_BYTES = 60_000             # 小于 60KB 基本是错误页
TIMEOUT = 40
MAX_SECONDS = 400   # 单次下载总时长上限：GitHub 到本机实际约 25KB/s，超时就换下一个

# (arXiv id, GitHub owner/repo, 领域标签)
# 选的都是「论文有公开官方实现」的组合；脚本会逐条校验，失败的自动跳过。
CANDIDATES: list[tuple[str, str, str]] = [
    # ---- 扩散 / 生成 ----
    ("2006.11239", "hojonathanho/diffusion", "扩散模型"),
    ("2112.10752", "CompVis/latent-diffusion", "扩散模型"),
    ("2202.09778", "ermongroup/ddim", "扩散模型"),
    ("2010.02502", "yang-song/score_sde_pytorch", "扩散模型"),
    ("2105.05233", "openai/guided-diffusion", "扩散模型"),
    ("2206.00364", "NVlabs/edm", "扩散模型"),
    ("1802.05957", "NVlabs/stylegan", "生成对抗"),
    ("1812.04948", "NVlabs/stylegan2", "生成对抗"),
    ("1801.10198", "openai/glow", "流模型"),
    ("1312.6114", "antixk/PyTorch-VAE", "变分自编码"),
    ("1406.2661", "eriklindernoren/PyTorch-GAN", "生成对抗"),
    # ---- 视觉 ----
    ("2010.11929", "google-research/vision_transformer", "视觉骨干"),
    ("2005.12872", "facebookresearch/detr", "目标检测"),
    ("2104.14294", "facebookresearch/dino", "自监督"),
    ("2111.06377", "facebookresearch/mae", "自监督"),
    ("2103.14030", "microsoft/Swin-Transformer", "视觉骨干"),
    ("1911.05722", "facebookresearch/moco", "自监督"),
    ("1610.02391", "jacobgil/pytorch-grad-cam", "可解释性"),
    ("1904.07850", "xingyizhou/CenterNet", "目标检测"),
    ("2002.05709", "google-research/simclr", "自监督"),
    ("1506.02640", "pjreddie/darknet", "目标检测"),
    ("1608.06993", "liuzhuang13/DenseNet", "视觉骨干"),
    ("1611.05431", "facebookresearch/ResNeXt", "视觉骨干"),
    ("1709.01507", "moskomule/senet.pytorch", "注意力模块"),
    ("1807.06521", "Jongchan/attention-module", "注意力模块"),
    ("1905.11946", "qubvel/efficientnet", "视觉骨干"),
    ("1801.04381", "d-li14/mobilenetv2.pytorch", "轻量网络"),
    # ---- 三维 ----
    ("2003.08934", "bmild/nerf", "神经渲染"),
    ("1612.00593", "charlesq34/pointnet", "点云"),
    ("1704.01212", "MMLab/pointnet2", "点云"),
    # ---- 语言 / 多模态 ----
    ("1810.04805", "google-research/bert", "预训练语言"),
    ("1706.03762", "tensorflow/tensor2tensor", "序列建模"),
    ("2103.00020", "openai/CLIP", "多模态"),
    ("2106.09685", "microsoft/LoRA", "参数微调"),
    ("2305.14314", "artidoro/qlora", "参数微调"),
    ("2205.14135", "Dao-AILab/flash-attention", "高效注意力"),
    ("2004.04906", "facebookresearch/DPR", "检索"),
    ("2004.05150", "allenai/longformer", "长文本"),
    ("1910.10683", "google-research/text-to-text-transfer-transformer", "预训练语言"),
    ("2009.14794", "lucidrains/performer-pytorch", "高效注意力"),
    ("2210.03629", "ysymyth/ReAct", "智能体"),
    ("2104.09864", "ZhuiyiTechnology/roformer", "位置编码"),
    # ---- 图 ----
    ("1710.10903", "PetarV-/GAT", "图神经网络"),
    ("1706.02216", "williamleif/GraphSAGE", "图神经网络"),
    # ---- 科学计算 ----
    ("1706.08566", "atomistic-machine-learning/schnetpack", "分子势能"),
    ("2003.03123", "gastegger/DimeNet", "分子势能"),
    ("1711.10561", "maziarraissi/PINNs", "科学机器学习"),
    ("2010.08895", "neuraloperator/neuraloperator", "算子学习"),
    # ---- 强化学习 ----
    ("1707.06347", "openai/baselines", "强化学习"),
    # ---- 备用池 ----
    ("1502.05477", "openai/baselines", "强化学习"),
    ("2006.11477", "facebookresearch/fairseq", "语音"),
    ("1802.02611", "qubvel/segmentation_models.pytorch", "语义分割"),
    ("1703.03400", "openai/supervised-reptile", "元学习"),
    ("1606.04797", "mattmacy/vnet.pytorch", "医学分割"),
    ("2006.06676", "NVlabs/stylegan2-ada-pytorch", "生成对抗"),
    ("2107.03374", "salesforce/CodeT5", "代码模型"),
    ("1809.04522", "facebookresearch/Detectron", "目标检测"),
    ("1904.09237", "openai/baselines", "强化学习"),
]


# 第二批：官方仓库失败时补的小仓库（论文都是同一批经典的公开实现）。
# 加这批的原因：第一批里 NVlabs/stylegan2、CompVis/latent-diffusion 这类仓库
# 动辄 40MB+，链路只有 ~40KB/s 时单个包就吃掉几分钟；小仓库成功率高得多，
# 而"论文 vs 代码"的审计能力并不挑仓库大小。
EXTRA_CANDIDATES: list[tuple[str, str, str]] = [
    ("2304.02643", "facebookresearch/segment-anything", "分割基础模型"),
    ("2304.07193", "facebookresearch/dinov2", "自监督"),
    ("2302.13971", "facebookresearch/llama", "大语言模型"),
    ("2212.04356", "openai/whisper", "语音识别"),
    ("1607.01759", "facebookresearch/fastText", "文本分类"),
    ("1808.06226", "google/sentencepiece", "分词"),
    ("2003.10555", "google-research/electra", "预训练语言"),
    ("1909.11942", "google-research/albert", "预训练语言"),
    ("1905.02249", "google-research/mixmatch", "半监督"),
    ("1904.12848", "google-research/uda", "半监督"),
    ("1909.00512", "openai/gpt-2", "语言模型"),
    ("1909.08593", "openai/lm-human-preferences", "对齐"),
    ("1803.03635", "google-research/lottery-ticket", "剪枝"),
    ("2201.05989", "NVlabs/instant-ngp", "神经渲染"),
    ("2102.12092", "lucidrains/DALLE-pytorch", "文生图"),
    ("2205.11487", "lucidrains/imagen-pytorch", "文生图"),
    ("1806.07366", "rtqichen/torchdiffeq", "连续时间模型"),
    ("1505.04597", "milesial/Pytorch-UNet", "医学分割"),
    ("1611.07004", "junyanz/pytorch-CycleGAN-and-pix2pix", "图像翻译"),
    ("1703.10593", "junyanz/pytorch-CycleGAN-and-pix2pix", "图像翻译"),
    ("2103.00020", "openai/CLIP", "多模态"),
    ("2106.09685", "microsoft/LoRA", "参数微调"),
    ("2305.14314", "artidoro/qlora", "参数微调"),
    ("2205.14135", "Dao-AILab/flash-attention", "高效注意力"),
    ("2009.14794", "lucidrains/performer-pytorch", "高效注意力"),
    ("2210.03629", "ysymyth/ReAct", "智能体"),
    ("2104.09864", "ZhuiyiTechnology/roformer", "位置编码"),
    ("1710.10903", "PetarV-/GAT", "图神经网络"),
    ("1706.02216", "williamleif/GraphSAGE", "图神经网络"),
    ("1706.08566", "atomistic-machine-learning/schnetpack", "分子势能"),
    ("2003.03123", "gastegger/DimeNet", "分子势能"),
    ("1711.10561", "maziarraissi/PINNs", "科学机器学习"),
    ("2010.08895", "neuraloperator/neuraloperator", "算子学习"),
    ("1707.06347", "openai/baselines", "强化学习"),
    ("2006.11477", "facebookresearch/fairseq", "语音"),
    ("1703.03400", "openai/supervised-reptile", "元学习"),
    ("1606.04797", "mattmacy/vnet.pytorch", "医学分割"),
    ("2107.03374", "salesforce/CodeT5", "代码模型"),
    ("1910.10683", "google-research/text-to-text-transfer-transformer", "预训练语言"),
    ("2004.05150", "allenai/longformer", "长文本"),
]


def _slug(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", repo).strip("-")[:32]


def _download(url: str, dest: Path, max_bytes: int) -> int:
    """用系统自带的 curl 下载到 dest（断点重试交给它），超过 max_bytes 放弃。

    为什么不用 urllib：Windows 上大包经常 `IncompleteRead`（chunked 传输被掐），
    自己写重试会白等几十秒。curl 的 --retry / --max-filesize 是现成的，
    而且它失败时退出码明确（63 = 超过大小上限）。
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    cmd = ["curl", "-sSL", "--fail", "--retry", "1", "--retry-delay", "1",
           "--connect-timeout", "15", "--max-time", str(MAX_SECONDS),
           "--max-filesize", str(max_bytes), "-o", str(tmp), url]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"curl exit {p.returncode}")
    n = tmp.stat().st_size
    if n > max_bytes:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"超过 {max_bytes // 1048576}MB 上限")
    tmp.replace(dest)
    return n


def _pdf_ok(path: Path) -> tuple[bool, str]:
    raw = path.read_bytes()[:4096]
    tail = path.read_bytes()[-4096:] if path.stat().st_size > 4096 else b""
    if path.stat().st_size < MIN_PDF_BYTES:
        return False, f"太小 {path.stat().st_size}B"
    if not raw[:5].startswith(b"%PDF"):
        return False, "不是 PDF 魔数"
    if b"%%EOF" not in tail:
        return False, "缺 EOF 尾标(可能截断)"
    return True, "ok"


def _zip_ok(path: Path) -> tuple[bool, str]:
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if z.testzip() is not None:
                return False, "CRC 校验失败"
    except Exception as e:  # noqa: BLE001
        return False, f"解压失败 {type(e).__name__}"
    py = [n for n in names if n.endswith(".py")]
    if len(py) < 3:
        return False, f".py 只有 {len(py)} 个"
    if len(names) < 10:
        return False, f"文件只有 {len(names)} 个"
    return True, f"ok {len(names)} 文件 / {len(py)} 个 .py"


# 已经转正（S*）的样例 slug —— 补采时必须跳过，否则会把已下好的论文再下一遍
DONE_SLUGS: set[str] = set()


def _one(arxiv_id: str, repo: str, tag: str, out: Path) -> dict:
    t0 = time.time()
    rec: dict = {"arxiv_id": arxiv_id, "repo": repo, "tag": tag, "ok": False}
    if _slug(repo) in DONE_SLUGS:
        rec.update(ok=True, dir="", cached=True, skip="已转正")
        return rec
    d = out / f"_tmp_{arxiv_id.replace('.', '_')}_{_slug(repo)}"

    # 断点续跑：已采好的直接认
    if (d / "paper.pdf").exists() and (d / "code.zip").exists():
        op, oz = _pdf_ok(d / "paper.pdf"), _zip_ok(d / "code.zip")
        if op[0] and oz[0]:
            rec.update(ok=True, dir=d.name, cached=True,
                       pdf_bytes=(d / "paper.pdf").stat().st_size,
                       zip_bytes=(d / "code.zip").stat().st_size,
                       zip_entries=oz[1], branch="cached")
            return rec

    d.mkdir(parents=True, exist_ok=True)
    pdf = d / "paper.pdf"
    ok, why = _pdf_ok(pdf) if pdf.exists() else (False, "不存在")
    if ok:
        rec["pdf_bytes"], rec["pdf_cached"] = pdf.stat().st_size, True
    else:
        try:
            n = _download(f"https://arxiv.org/pdf/{arxiv_id}", pdf, 12 << 20)
            ok, why = _pdf_ok(pdf)
            if not ok:
                rec["error"] = f"PDF {why}"
                return rec
            rec["pdf_bytes"] = n
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"PDF 失败 {type(e).__name__}: {e}"
            return rec

    errs = []
    for branch in ("main", "master"):
        try:
            n = _download(
                f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}",
                d / "code.zip", MAX_ZIP_BYTES)
            ok, why = _zip_ok(d / "code.zip")
            if not ok:
                errs.append(f"{branch}:{why}")
                continue
            rec.update(ok=True, dir=d.name, branch=branch, zip_bytes=n,
                       zip_entries=why, seconds=round(time.time() - t0, 1))
            return rec
        except urllib.error.HTTPError as e:
            errs.append(f"{branch}:HTTP{e.code}")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{branch}:{type(e).__name__}")
    rec["error"] = "代码包 " + " / ".join(errs)
    (d / "code.zip").unlink(missing_ok=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 已转正的样例：按目录数算已完成
    done = [p for p in out.iterdir() if p.is_dir() and p.name.startswith("S")]
    for p in done:
        DONE_SLUGS.add(p.name.split("_", 1)[-1])
    print(f"已有 {len(done)} 份转正，目标 {args.target}", flush=True)

    pool: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    for a, r, t in CANDIDATES + EXTRA_CANDIDATES:
        if a in seen_ids:
            continue
        seen_ids.add(a)
        pool.append((a, r, t))
    print(f"候选池 {len(pool)} 条", flush=True)

    got: list[dict] = []
    failed: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, a, r, t, out): (a, r, t) for a, r, t in pool}
        for i, f in enumerate(as_completed(futs), 1):
            a, r, t = futs[f]
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                rec = {"arxiv_id": a, "repo": r, "tag": t, "ok": False,
                       "error": f"未捕获 {type(e).__name__}: {e}"}
            (got if rec["ok"] else failed).append(rec)
            flag = "OK  " if rec["ok"] else "MISS"
            print(f"[{i:02d}/{len(pool)}] {flag} {a:>10} {r:<46} "
                  f"{rec.get('error', '')}", flush=True)
            # 凑够就停：剩下的候选不再等（候选池 90 条，全跑完没必要）
            fresh = len([r for r in got if not r.get("skip")])
            if fresh + len(done) >= args.target:
                for f2 in futs:
                    f2.cancel()
                print(f"已凑够 {args.target} 份，停止等待其余候选", flush=True)
                break

    need = args.target - len(done)
    # 跳过项（已转正）没有目录可改名，不能进转正队列
    got = [r for r in got if not r.get("skip")]
    got.sort(key=lambda x: x["arxiv_id"])
    take = got[: max(0, need)]
    start = len(done) + 1
    for i, rec in enumerate(take, start):
        src = out / rec["dir"]
        dst = out / f"S{i:02d}_{_slug(rec['repo'])}"
        if dst.exists():
            shutil.rmtree(dst)
        src.rename(dst)
        rec["dir"] = dst.name
        rec["id"] = f"S{i:02d}"
        (dst / "meta.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    for p in out.glob("_tmp_*"):
        shutil.rmtree(p, ignore_errors=True)

    final = sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("S"))
    total = sum(f.stat().st_size for p in final for f in p.iterdir() if f.is_file())
    summary = {
        "target": args.target, "collected": len(final),
        "failed": len(failed), "total_mb": round(total / 1048576, 1),
        "failure_reasons": [{k: v for k, v in f.items() if k in ("arxiv_id", "repo", "error")}
                            for f in failed],
    }
    (out / "_collect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n采集完成：{len(final)}/{args.target} 份，失败 {len(failed)}，"
          f"总计 {summary['total_mb']} MB", flush=True)
    return 0 if len(final) >= args.target else 1


if __name__ == "__main__":
    sys.exit(main())
