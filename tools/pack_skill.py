#!/usr/bin/env python3
"""把 Agent Skills 技能包打成可上传的 zip（阶段六 · 协议化②）。

用法:
    python tools/pack_skill.py                    # 打包 skills/ 下所有技能
    python tools/pack_skill.py skills/paper2lab-audit
    python tools/pack_skill.py --check            # 只校验，不产出 zip

打包前会做三道检查，任何一道不过就**拒绝出包**（宁可不出，也不出个坏包）:

  1. **规范检查**: SKILL.md 必须存在、有 YAML frontmatter、name/description
     齐全且合规（name 与目录名一致、小写连字符、≤64 字符；description ≤1024）;
  2. **密钥检查**: 扫描包内全部文本文件，出现疑似密钥/令牌就拦下 ——
     技能包会被分发到别人的机器上，带密钥出去是不可逆的泄露;
  3. **结构检查**: zip 根目录必须是技能文件夹本身（含 SKILL.md），
     不允许把脚本散落在包根 —— 平台按"文件夹即技能"识别。

产出的 zip 落在 `dist/`（该目录不进版本库）。
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("缺少 pyyaml，请先 pip install pyyaml", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
DIST_DIR = ROOT / "dist"

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TEXT_SUFFIXES = {".md", ".py", ".json", ".txt", ".yaml", ".yml", ".toml",
                  ".cfg", ".ini", ".sh", ".bat", ".ps1", ".env"}
# 疑似密钥：只匹配"看起来就是一把钥匙"的形态，避免把普通示例文本误杀
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\btp-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret[_-]?key|password)"
               r"\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']"),
)


def validate_skill(skill_dir: Path) -> list:
    """返回问题清单（空列表 = 通过）。"""
    problems = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [f"缺少 SKILL.md: {skill_md}"]

    raw = skill_md.read_text(encoding="utf-8")
    if not raw.lstrip().startswith("---"):
        return ["SKILL.md 开头必须是 YAML frontmatter（以 --- 起止）"]

    parts = raw.split("---", 2)
    if len(parts) < 3:
        return ["SKILL.md 的 frontmatter 没有正确闭合（需要成对的 ---）"]
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except Exception as exc:
        return [f"frontmatter 不是合法 YAML: {exc}"]
    if not isinstance(meta, dict):
        return ["frontmatter 解析结果不是键值对"]

    name = str(meta.get("name", "")).strip()
    desc = str(meta.get("description", "")).strip()

    if not name:
        problems.append("frontmatter 缺少 name")
    else:
        if len(name) > 64:
            problems.append(f"name 超过 64 字符（{len(name)}）")
        if not _NAME_RE.match(name):
            problems.append(f"name 不合规（只允许小写字母/数字/单个连字符）: {name}")
        if name != skill_dir.name:
            problems.append(
                f"name 与目录名必须一致: name={name} vs 目录={skill_dir.name}")

    if not desc:
        problems.append("frontmatter 缺少 description")
    elif len(desc) > 1024:
        problems.append(f"description 超过 1024 字符（{len(desc)}）")

    compat = str(meta.get("compatibility", "")).strip()
    if len(compat) > 500:
        problems.append(f"compatibility 超过 500 字符（{len(compat)}）")

    body = parts[2].strip()
    if len(body) < 50:
        problems.append("SKILL.md 正文太短（至少要说明做什么、什么时候用）")

    return problems


def scan_secrets(files) -> list:
    """扫描疑似密钥，返回 [(相对路径, 命中片段)]。"""
    hits = []
    for path in files:
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in _SECRET_PATTERNS:
            m = pat.search(text)
            if m:
                frag = m.group(0)
                hits.append((str(path), frag[:24] + "..."))
                break
    return hits


def collect_files(skill_dir: Path) -> list:
    return sorted(p for p in skill_dir.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix.lower() not in (".pyc", ".pyo"))


def pack(skill_dir: Path, out_dir: Path = DIST_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    version = "0.0.0"
    skill_md = skill_dir / "SKILL.md"
    if skill_md.is_file():
        try:
            meta = yaml.safe_load(skill_md.read_text(encoding="utf-8")
                                  .split("---", 2)[1]) or {}
            version = str((meta.get("metadata") or {}).get("version", version))
        except Exception:
            pass
    out = out_dir / f"{skill_dir.name}-{version}.zip"
    files = collect_files(skill_dir)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            # arcname 带一级目录名: 平台按"根目录下含 SKILL.md"识别
            zf.write(p, arcname=str(Path(skill_dir.name) / p.relative_to(skill_dir)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pack_skill.py",
                                 description="打包 Agent Skills 技能包")
    ap.add_argument("skill", nargs="?", help="技能目录（缺省则打包 skills/ 下全部）")
    ap.add_argument("--check", action="store_true", help="只校验，不产出 zip")
    ap.add_argument("--out", default=str(DIST_DIR), help="输出目录（默认 dist/）")
    args = ap.parse_args(argv)

    if args.skill:
        targets = [Path(args.skill).resolve()]
    else:
        if not SKILLS_DIR.is_dir():
            print(f"没有找到 {SKILLS_DIR}", file=sys.stderr)
            return 1
        targets = sorted(p for p in SKILLS_DIR.iterdir()
                         if p.is_dir() and (p / "SKILL.md").is_file())

    if not targets:
        print("没有可打包的技能包", file=sys.stderr)
        return 1

    failed = 0
    for skill_dir in targets:
        print(f"=== {skill_dir.name} ===")
        problems = validate_skill(skill_dir)
        if problems:
            failed += 1
            print("  [拒绝出包] 规范检查未通过:")
            for p in problems:
                print(f"    - {p}")
            continue
        print("  ✔ 规范检查通过")

        files = collect_files(skill_dir)
        hits = scan_secrets(files)
        if hits:
            failed += 1
            print("  [拒绝出包] 疑似密钥:")
            for path, frag in hits:
                print(f"    - {path}: {frag}")
            continue
        print(f"  ✔ 密钥检查通过（{len(files)} 个文件）")

        if args.check:
            print("  （--check：不产出 zip）")
            continue

        out = pack(skill_dir, Path(args.out))
        names = zipfile.ZipFile(out).namelist()
        has_root_skill = f"{skill_dir.name}/SKILL.md" in names
        print(f"  ✔ 已打包: {out}  ({out.stat().st_size} 字节, {len(names)} 个条目)")
        print(f"  {'✔' if has_root_skill else '✘'} 包根含 SKILL.md")
        if not has_root_skill:
            failed += 1

    print(f"\n结论: {'全部通过。' if failed == 0 else f'{failed} 个技能包未通过。'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
