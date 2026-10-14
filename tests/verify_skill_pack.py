"""协议化② 验证：Agent Skills 技能包（阶段六）。

关注点（每条都对应一个真实会错的地方）:
1. **规范硬指标** —— SKILL.md 的 name 必须与目录名一致、只含小写字母数字连字符、
   ≤64 字符；description 非空且 ≤1024。这些是平台能否识别的硬门槛，不是风格问题。
2. **检查器本身要有效** —— 用一个"故意做坏"的临时技能包反向验证：
   含假密钥必须被拦、name 不合规必须被拦。没这条，前面那些"通过"就只是摆设。
3. **脚本真能跑** —— 技能包是要分发到别人机器上的，脚本语法错误等于发了个坏包。
4. **zip 结构对** —— 包根必须是技能文件夹本身且含 SKILL.md；路径必须是正斜杠。
5. **包里不能有密钥** —— 分发出去不可逆。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "skills" / "paper2lab-audit" / "scripts"))

import pack_skill as P  # noqa: E402
from audit import find_project_root  # noqa: E402

SKILL_DIR = ROOT / "skills" / "paper2lab-audit"
SCRIPTS = SKILL_DIR / "scripts"
DEMO_PDF = str(ROOT / "demo" / "paper.pdf")
DEMO_CODE = str(ROOT / "demo" / "project.zip")

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


def run(cmd, timeout=180):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT),
                       env=env, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# ═══════════════════════════════════════ 1. 目录结构

sec("1. 技能包结构 —— 「一个文件夹就是一个技能」")

check(SKILL_DIR.is_dir(), "技能目录存在")
check((SKILL_DIR / "SKILL.md").is_file(), "SKILL.md 在包根（平台据此识别）")
check((SKILL_DIR / "scripts").is_dir(), "有 scripts/ 目录")
check((SKILL_DIR / "references").is_dir(), "有 references/ 目录")
for name in ("audit.py", "check_env.py", "mcp_client_example.py"):
    check((SCRIPTS / name).is_file(), f"脚本存在: scripts/{name}")
for name in ("parameters.md", "boundaries.md", "client-setup.md"):
    check((SKILL_DIR / "references" / name).is_file(), f"参考文档存在: references/{name}")

# ═══════════════════════════════════════ 2. frontmatter 硬指标

sec("2. SKILL.md frontmatter —— 平台识别的硬门槛")

text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
check(text.lstrip().startswith("---"), "文件以 frontmatter 开头")

import yaml  # noqa: E402
fm = yaml.safe_load(text.split("---", 2)[1])
check(isinstance(fm, dict), "frontmatter 是键值对")
name = str(fm.get("name", ""))
desc = str(fm.get("description", ""))

check(name == SKILL_DIR.name, f"name 与目录名一致（{name}）")
check(len(name) <= 64, "name ≤ 64 字符")
check(bool(re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name)),
      "name 只含小写字母/数字/单连字符")
check(0 < len(desc) <= 1024, f"description 长度合规（{len(desc)}）")
check("复现" in desc and "参数" in desc, "description 说清了做什么")
check(("当" in desc) or ("时使用" in desc), "description 说了什么时候用（触发条件）")
check("论文" in desc and "代码" in desc, "description 含关键实体（论文 / 代码）")
check(bool((fm.get("metadata") or {}).get("version")), "metadata.version 存在")
check(len(str(fm.get("compatibility", ""))) <= 500, "compatibility ≤ 500 字符")

body = text.split("---", 2)[2]
check(len(body.splitlines()) <= 500, "正文 ≤ 500 行（激活时全量进上下文）")
check("什么时候用" in body, "正文有「什么时候用」小节")
check("| 参数 | 类型 | 必填 | 说明 |" in body, "正文有标准输入参数表")
check("不该用" in body, "正文写了「不该用」的场景（防误触发）")
check(body.count("只读") >= 1 and "不执行用户代码" in body,
      "正文写明只读与不执行用户代码两条边界")
check("最多 3 轮" in body, "正文写明迭代有界（≤3 轮）")

# ═══════════════════════════════════════ 3. 检查器有效性（反向验证）

sec("3. 检查器反向验证 —— 故意做坏的包必须被拦下")

tmp = Path(tempfile.mkdtemp(prefix="p2l_skill_bad_"))
try:
    bad = tmp / "bad-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\nname: WrongName\ndescription: 太短\n---\n\n正文。\n",
        encoding="utf-8")
    problems = P.validate_skill(bad)
    check(any("name" in p for p in problems), "name 大写/不合规 -> 被拦")
    check(any("目录名" in p for p in problems), "name 与目录名不一致 -> 被拦")
    check(any("正文太短" in p for p in problems), "正文过短 -> 被拦")

    no_fm = tmp / "no-frontmatter"
    no_fm.mkdir()
    (no_fm / "SKILL.md").write_text("# 没有 frontmatter\n", encoding="utf-8")
    problems = P.validate_skill(no_fm)
    check(any("frontmatter" in p for p in problems), "缺 frontmatter -> 被拦")

    leak = tmp / "leak-skill"
    leak.mkdir()
    (leak / "SKILL.md").write_text(
        "---\nname: leak-skill\ndescription: 这是一个用于测试密钥检查的技能包描述文本。\n"
        "---\n\n" + "正文" * 40 + "\n", encoding="utf-8")
    (leak / "config.txt").write_text(
        'LLM_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz012345"\n', encoding="utf-8")
    hits = P.scan_secrets(P.collect_files(leak))
    check(len(hits) == 1, "假密钥 -> 被扫出来（密钥检查不是摆设）")

    (leak / "config.txt").write_text("只是普通说明文字，没有钥匙。\n", encoding="utf-8")
    hits = P.scan_secrets(P.collect_files(leak))
    check(len(hits) == 0, "普通文本不误杀（检查器不制造噪声）")

    rc, out = run([sys.executable, str(ROOT / "tools" / "pack_skill.py"),
                   str(bad)])
    check(rc != 0 and "拒绝出包" in out, "做坏的包 -> 打包器返回非 0 且拒绝")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ═══════════════════════════════════════ 4. 真实技能包通过校验

sec("4. 真实技能包 —— 规范 + 密钥双检通过")

problems = P.validate_skill(SKILL_DIR)
check(problems == [], f"规范检查通过（问题: {problems}）")
files = P.collect_files(SKILL_DIR)
check(len(files) >= 7, f"技能包文件数合理（{len(files)}）")
check(all("__pycache__" not in str(f) for f in files), "不含 __pycache__ 等编译产物")
check(not [f for f in files if f.suffix in (".pyc", ".pyo")], "不含 .pyc/.pyo")
hits = P.scan_secrets(files)
check(hits == [], f"没有疑似密钥（命中: {hits}）")

# ═══════════════════════════════════════ 5. 脚本真能跑

sec("5. 脚本真机 —— 发出去的包不能是坏的")

for name in ("audit.py", "check_env.py", "mcp_client_example.py"):
    rc, out = run([sys.executable, "-m", "py_compile", str(SCRIPTS / name)])
    check(rc == 0, f"scripts/{name} 语法可编译")

rc, out = run([sys.executable, str(SCRIPTS / "audit.py"), "--help"])
check(rc == 0 and "只读输入" in out, "audit.py --help 可用")

rc, out = run([sys.executable, str(SCRIPTS / "mcp_client_example.py"), "--help"])
check(rc == 0 and "mcp" in out.lower(), "mcp_client_example.py --help 可用")

root_found = find_project_root()
check(root_found == ROOT, f"脚本能自己找到项目根（{root_found}）")

rc, out = run([sys.executable, str(SCRIPTS / "check_env.py")], timeout=240)
check(rc == 0 and "全部就绪" in out, "check_env.py 自检通过")

rc, out = run([sys.executable, str(SCRIPTS / "audit.py"),
               DEMO_PDF, DEMO_CODE], timeout=300)
check(rc == 0, "audit.py 真机跑通（退出码 0）")
try:
    # 输出里可能混有第三方提示，取最后一个 JSON 对象
    start = out.index("{")
    data = json.loads(out[start:out.rindex("}") + 1])
    check(data.get("ok") is True, "audit.py 输出 ok=true")
    check(data.get("findings_total") == 11, "audit.py 得 11 条结论（与主程序一致）")
except Exception as exc:
    check(False, f"audit.py 输出不是可解析 JSON: {type(exc).__name__}: {exc}")

rc, out = run([sys.executable, str(SCRIPTS / "audit.py"),
               "C:/nope/x.pdf", DEMO_CODE], timeout=120)
check(rc == 1 and "不存在" in out, "audit.py 坏输入 -> 退出码 1 且说明原因")

# ═══════════════════════════════════════ 6. 打包产物

sec("6. 打包 zip —— 结构规范、可上传")

out_dir = Path(tempfile.mkdtemp(prefix="p2l_skill_dist_"))
try:
    zip_path = P.pack(SKILL_DIR, out_dir)
    check(zip_path.is_file() and zip_path.suffix == ".zip", "zip 已产出")
    check("1.0.0" in zip_path.name, "文件名带版本号（来自 metadata.version）")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        check(f"{SKILL_DIR.name}/SKILL.md" in names, "包根含 技能目录/SKILL.md")
        check(all(not n.startswith("/") for n in names), "没有绝对路径条目")
        check(all("\\" not in n for n in names), "zip 内路径用正斜杠（zip 规范要求）")
        check(not any("__pycache__" in n for n in names), "不含 __pycache__")
        check(any(n.endswith("scripts/audit.py") for n in names), "含 audit.py")
        check(len(names) == len(files), f"条目数与文件数一致（{len(names)}）")
        # zip 里不能有密钥
        leaked = []
        for n in names:
            data = zf.read(n)
            try:
                txt = data.decode("utf-8")
            except Exception:
                continue
            for pat in P._SECRET_PATTERNS:
                if pat.search(txt):
                    leaked.append(n)
        check(not leaked, f"zip 内无密钥（命中: {leaked}）")

    rc, out = run([sys.executable, str(ROOT / "tools" / "pack_skill.py"), "--check"])
    check(rc == 0 and "全部通过" in out, "pack_skill.py --check 通过")
    # 全量打包走临时输出目录，避免污染工作区
    rc, out = run([sys.executable, str(ROOT / "tools" / "pack_skill.py"),
                   "--out", str(out_dir)])
    check(rc == 0 and "已打包" in out, "pack_skill.py 全量打包通过")
    check("拒绝出包" not in out, "正常包不会被误拦")
finally:
    shutil.rmtree(out_dir, ignore_errors=True)

# ═══════════════════════════════════════ 7. 契约自述

sec("7. 契约自述 —— 包里的说法与程序行为一致")

check("字面检索" in (SKILL_DIR / "references" / "boundaries.md").read_text("utf-8"),
      "boundaries.md 如实自述检索方式（不吹成语义检索）")
check("最多 3 轮" in (SKILL_DIR / "references" / "boundaries.md").read_text("utf-8"),
      "boundaries.md 写明迭代有界")
check("不写文件" in (SKILL_DIR / "references" / "boundaries.md").read_text("utf-8"),
      "boundaries.md 写明只读")
setup = (SKILL_DIR / "references" / "client-setup.md").read_text("utf-8")
check("mcp>=1.0,<2" in setup, "client-setup.md 写明版本约束（与代码一致）")
check("streamablehttp_client" in setup, "client-setup.md 给了 HTTP 接法")
check("不发布" not in setup and "端砚" not in setup, "客户端文档不掺杂平台运营口径")

print("\n" + "═" * 62)
if FAILURES:
    print(f"失败 {len(FAILURES)} / {CHECKS}")
    for f in FAILURES:
        print("  - " + f)
    sys.exit(1)
print(f"全部通过(verify_skill_pack.py): {CHECKS} 项")
