"""生成「多模态图表」演示数据（阶段二 2E）。

为什么需要单独一份 demo: 主演示数据 `demo/paper.pdf` 是纯文本论文（无位图），
它恰好**测不到** 2E 的那条通道。验收要求是"给定带谱图 / 曲线的 demo,
能抽出至少一类结构化线索并给出出处"—— 所以这里造一份**带真图**的论文:

  · 第 1 页: 实验设置正文（含几个超参数）＋ 训练损失曲线（位图）＋ 图注
  · 第 2 页: 红外光谱图（位图）＋ 图注 ＋ 超参数表

图形用 PIL 画成位图再嵌进 PDF（PyMuPDF 的矢量绘制不会出现在 `get_images()` 里,
那样"读图通道"根本看不到它 —— 这正是当初踩过的坑: 矢量图 = 文本层已覆盖）。
图片角落**烧进一行防伪标记**（`P2L-FIG-DEMO`）, 用来验证"读图通道真的看了这张图",
而不是靠猜。

用法: .venv\\Scripts\\python.exe tools\\make_figure_demo.py
"""
from __future__ import annotations

import math
import sys
import pathlib

# --- 目录整理后补的引导: 脚本在 tools/ 下, 项目根要加回 sys.path ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PDF = ROOT / "demo" / "figure_demo.pdf"
TMP = ROOT / ".trash_tmp"
MARK = "P2L-FIG-DEMO"          # 烧进图里的防伪标记（测试据此断言"图上确实有字"）


# ---------------------------------------------------------------- 画图（PIL）

def _axes(draw, box, xlabel, ylabel, xr, yr, xfmt="{:.0f}", yfmt="{:.2f}"):
    """画坐标轴 + 刻度 + 轴标签。box = (l, t, r, b)。"""
    l, t, r, b = box
    draw.line([(l, b), (r, b)], fill=(40, 40, 40), width=2)     # x 轴
    draw.line([(l, t), (l, b)], fill=(40, 40, 40), width=2)     # y 轴
    for i in range(6):
        x = l + (r - l) * i / 5
        draw.line([(x, b), (x, b + 5)], fill=(40, 40, 40), width=1)
        draw.text((x - 12, b + 8), xfmt.format(xr[0] + (xr[1] - xr[0]) * i / 5),
                  fill=(60, 60, 60))
    for i in range(5):
        y = b - (b - t) * i / 4
        draw.line([(l - 5, y), (l, y)], fill=(40, 40, 40), width=1)
        draw.text((6, y - 6), yfmt.format(yr[0] + (yr[1] - yr[0]) * i / 4),
                  fill=(60, 60, 60))
    draw.text(((l + r) / 2 - len(xlabel) * 3, b + 26), xlabel, fill=(20, 20, 20))
    draw.text((l - 30, t - 20), ylabel, fill=(20, 20, 20))


def _to_px(box, xr, yr, x, y):
    l, t, r, b = box
    px = l + (r - l) * (x - xr[0]) / (xr[1] - xr[0])
    py = b - (b - t) * (y - yr[0]) / (yr[1] - yr[0])
    return px, py


def _loss_curve_png(path: pathlib.Path) -> None:
    """训练/验证损失曲线: 两条下降曲线（趋势清晰, 不解释原因）。"""
    W, H = 880, 760
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (110, 70, 820, 660)
    xr, yr = (0, 300), (0.0, 2.5)
    _axes(d, box, "epoch", "loss", xr, yr)
    train = [(x, 2.3 * math.exp(-x / 70) + 0.25) for x in range(0, 301, 5)]
    val = [(x, 2.55 * math.exp(-x / 60) + 0.55) for x in range(0, 301, 5)]
    d.line([_to_px(box, xr, yr, *p) for p in train], fill=(210, 60, 60), width=3)
    d.line([_to_px(box, xr, yr, *p) for p in val], fill=(60, 110, 210), width=3)
    d.text((640, 80), "train", fill=(210, 60, 60))
    d.text((640, 100), "val", fill=(60, 110, 210))
    d.text((640, 130), MARK, fill=(120, 120, 120))
    img.save(path)


def _spectrum_png(path: pathlib.Path) -> None:
    """红外光谱示意: 一个主峰（1730）+ 一个次峰（3450），峰位是"看得见的"硬线索。"""
    W, H = 880, 760
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    box = (110, 70, 820, 660)
    xr, yr = (4000, 400), (0.0, 1.0)
    _axes(d, box, "wavenumber (cm-1)", "transmittance", xr, yr, xfmt="{:.0f}")

    def peak(x, c, w, dep):
        return 1.0 - dep * math.exp(-((x - c) ** 2) / (2 * w * w))

    xs = list(range(400, 4001, 10))
    ys = [peak(x, 1730, 60, 0.82) * peak(x, 3450, 180, 0.55) for x in xs]
    d.line([_to_px(box, xr, yr, x, y) for x, y in zip(xs, ys)],
           fill=(40, 40, 40), width=2)
    # 峰位标注（图上直接印出来的数值 —— 读图线索的 anchor 就是这些字）
    for c, label in ((1730, "1730 cm-1"), (3450, "3450 cm-1")):
        px, py = _to_px(box, xr, yr, c, peak(c, c, 60 if c == 1730 else 180,
                                            0.82 if c == 1730 else 0.55))
        d.line([(px, py), (px, 55)], fill=(200, 120, 40), width=1)
        d.text((px - 30, 36), label, fill=(200, 120, 40))
    d.text((640, 100), MARK, fill=(120, 120, 120))
    img.save(path)


# ---------------------------------------------------------------- 组 PDF

BODY = """4.1 Experimental setup
We train a ResNet-18 on CIFAR-10 for 300 epochs with batch size 128.
The learning rate is set to 3e-4 with a cosine decay schedule, and the weight
decay is 0.05. Dropout is 0.1. All models are trained on 8 GPUs. The optimizer
is AdamW. The random seed is not fixed in our released code.

4.2 Results
Figure 1 shows the training and validation curves over 300 epochs. Figure 2
reports the infrared spectrum of the sample, with the main absorption band at
1730 cm-1. Table 1 lists the hyperparameters used in all experiments.
"""


def _write_page(doc, paragraphs: list, images: list) -> None:
    """paragraphs = [(text, size, font)]; images = [(png, caption)]（图注走中文字体）。

    图放在**固定位置**（正文之下、图注之上）, 并且**足够大**:
    读图通道认"大块位图"的阈值是页面面积的 20%（`vision._MIN_IMG_AREA_RATIO`,
    与图像通道同一口径）—— 图太小的话读图通道根本看不到它, demo 也就测不到那条路径。
    """
    page = doc.new_page(width=595, height=842)
    y = 56.0
    for text, size, font in paragraphs:
        lines = text.count("\n") + 1
        page.insert_textbox(fitz.Rect(56, y, 539, y + 20 * lines + 8), text,
                            fontsize=size, fontname=font, align=0)
        y += 20 * lines + 8
    for png, caption in images:
        rect = fitz.Rect(96, 280, 499, 620)          # 403×340 ≈ 页面 27%（> 20%）
        page.insert_image(rect, filename=str(png))
        page.insert_textbox(fitz.Rect(96, 626, 499, 700), caption,
                            fontsize=9, fontname="china-s")


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    loss_png = TMP / "fig_demo_loss.png"
    spec_png = TMP / "fig_demo_spectrum.png"
    _loss_curve_png(loss_png)
    _spectrum_png(spec_png)

    doc = fitz.open()
    # 第 1 页: 实验设置正文（含超参数）＋ 损失曲线 ＋ **中文图注**
    _write_page(doc,
                [(BODY, 10, "helv")],
                [(loss_png, "图 1　训练与验证损失曲线（300 epochs）")])
    # 第 2 页: 红外光谱 ＋ 中文图注 ＋ 超参数表（英文表注，覆盖两条通道）
    _write_page(doc,
                [("Table 1: Hyperparameters used in all experiments.\n"
                  "  learning rate   3e-4\n  batch size      128\n"
                  "  epochs          300\n  optimizer       AdamW\n"
                  "  weight decay    0.05\n  dropout         0.1\n"
                  "  lr scheduler    cosine\n", 10, "helv")],
                [(spec_png, "图 2　样品的红外光谱（KBr 压片），主吸收带位于 1730 cm-1")])
    doc.save(str(OUT_PDF))
    doc.close()
    print(f"OK -> {OUT_PDF}")
    print(f"（图中防伪标记: {MARK}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
