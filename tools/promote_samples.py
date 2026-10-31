"""把已经采好的 `_tmp_*` 目录立刻转成正式样例（S01、S02…）。

为什么单独写一个：采集脚本是"全部下完再统一改名"，碰上慢网络要等很久。
测试不必等 —— 已完整的先转正先测，剩下的采集完再补一次。

判定"完整"的口径与采集脚本一致：paper.pdf 能开（%PDF 魔数 + EOF 尾标 + 够大）、
code.zip 是有效 zip 且含 ≥3 个 .py。差一样就不转，留给下一轮。

幂等：已经有 S* 目录时从下一个号继续，不会覆盖。
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path.home() / "Desktop" / "Paper2Lab测试集" / "50份样例"


def pdf_ok(p: Path) -> bool:
    if not p.exists() or p.stat().st_size < 60_000:
        return False
    head = p.open("rb").read(5)
    tail = p.open("rb").read()[-4096:]
    return head.startswith(b"%PDF") and b"%%EOF" in tail


def zip_ok(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, "缺 code.zip"
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            if z.testzip() is not None:
                return False, "CRC 坏"
    except Exception as e:  # noqa: BLE001
        return False, f"打不开 {type(e).__name__}"
    py = sum(1 for n in names if n.endswith(".py"))
    if py < 3:
        return False, f".py 只有 {py} 个"
    return True, f"{len(names)} 文件/{py} py"


def main() -> int:
    existing = sorted(p for p in BASE.iterdir() if p.is_dir() and p.name.startswith("S"))
    nxt = len(existing) + 1
    rows = []
    for d in sorted(BASE.iterdir()):
        if not d.is_dir() or not d.name.startswith("_tmp_"):
            continue
        if not pdf_ok(d / "paper.pdf"):
            rows.append((d.name, "跳过", "PDF 不合格"))
            continue
        ok, why = zip_ok(d / "code.zip")
        if not ok:
            rows.append((d.name, "跳过", why))
            continue
        slug = d.name[len("_tmp_"):]
        slug = "_".join(slug.split("_")[2:]) or "sample"
        dst = BASE / f"S{nxt:02d}_{slug[:36]}"
        if dst.exists():
            shutil.rmtree(dst)
        d.rename(dst)
        meta = {"id": f"S{nxt:02d}", "dir": dst.name, "source": d.name,
                "pdf_bytes": (dst / "paper.pdf").stat().st_size,
                "zip_bytes": (dst / "code.zip").stat().st_size,
                "zip_entries": why, "promoted_from": d.name}
        (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        rows.append((d.name, "转正", dst.name))
        nxt += 1

    for src, act, note in rows:
        print(f"{act}  {src[:52]:<52} {note}")
    total = len([p for p in BASE.iterdir() if p.is_dir() and p.name.startswith("S")])
    print(f"\n现有正式样例：{total} 份")
    return 0


if __name__ == "__main__":
    sys.exit(main())
