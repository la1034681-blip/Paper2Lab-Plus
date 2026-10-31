"""弹窗名额守卫（离线源码体检，不起服务、不开浏览器）。

守的是什么
----------
Streamlit 1.64 有一条硬限制：**一次脚本运行只允许打开一个 `st.dialog`**。
开第二个不是「弹窗不出现」，而是抛
`StreamlitInvalidLayoutContextError: Only one dialog is allowed to be opened
at the same time`，然后**整页从那一行往下全都不渲染**，进程还会退出。

2026-10-31 00:40 就是这么崩的：用户先在指挥台点过某一块（焦点留在
session 里），之后点「问一问」—— 问答先开、详情再开，两者同一轮撞车。

所以「名额」这件事必须被固化：谁先开谁占，后来的那一轮不开。
这里逐条盯六件事：
  A 名额变量存在，且**在问答弹窗之前**就初始化（否则详情分支读到未定义）；
  B 问答分支一进门就占名额（写在 try 之前，异常也不能漏占）；
  C 详情调用点受名额保护，且整个调用包在 try/except 里（再保一层）；
  D 名额被占时**不清焦点** —— 等问答关掉，下一页自然摊开；
  E 全仓只有两个 `@st.dialog`，且都不在 app.py 里（app.py 只负责编排）；
  F 两个弹窗入口在 app.py 里各只有一处调用（多一处就多一次撞车机会）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
DECKUI = ROOT / "paper2lab" / "deckui.py"
QAUI = ROOT / "paper2lab" / "qaui.py"

N = 0
BAD: list[str] = []


def check(ok: bool, label: str) -> None:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        BAD.append(label)


def _lines() -> list[str]:
    return APP.read_text(encoding="utf-8").splitlines()


def _idx(lines: list[str], needle: str, start: int = 0) -> int:
    """返回 needle 首次出现的行号（0 基），找不到返回 -1。"""
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i
    return -1


def _block(lines: list[str], at: int) -> list[str]:
    """从 at 行起，取到下一个顶格语句为止（粗略的缩进块切片）。"""
    out = [lines[at]]
    for line in lines[at + 1:]:
        if line.strip() and not line.startswith((" ", "\t")):
            break
        out.append(line)
    return out


def main() -> int:
    lines = _lines()
    app = "\n".join(lines)

    # ---------------------------------------------------------------- A 名额存在且够早
    print("A 名额变量")
    i_init = _idx(lines, '_DLG_TAKEN = ""')
    i_qa = _idx(lines, "if st.session_state.get(QA_OPEN):")
    i_detail = _idx(lines, "_render_detail(_dk_audit, _focus)")
    check(i_init >= 0, "app.py 里有名额变量的初始化（_DLG_TAKEN）")
    check(0 <= i_init < i_qa, "名额初始化在问答弹窗**之前**（否则详情分支读到未定义）")
    check(0 <= i_qa < i_detail, "问答弹窗在详情弹窗之前（顺序固定，才谈得上先到先得）")

    # ---------------------------------------------------------------- B 问答一进门就占
    print("\nB 问答占名额")
    qa_blk = _block(lines, i_qa) if i_qa >= 0 else []
    qa_txt = "\n".join(qa_blk)
    i_take = _idx(qa_blk, '_DLG_TAKEN = "qa"')
    i_try = _idx(qa_blk, "try:")
    check(i_take >= 0, "问答分支里占了名额（_DLG_TAKEN = qa）")
    check(0 <= i_take < i_try or i_try < 0,
          "占名额写在 try **之前**（问答自己抛异常也不许把名额漏掉）")
    check("_qa_dialog(" in qa_txt, "问答分支里确实调用了弹窗")

    # ---------------------------------------------------------------- C 详情受保护
    print("\nC 详情让位")
    det = _block(lines, i_detail - 1) if i_detail > 0 else []
    # 往上找 if 条件那一行
    guard = ""
    for k in range(max(0, i_detail - 6), i_detail):
        if "_render_detail(" in lines[k]:
            continue
        if lines[k].strip().startswith("if "):
            guard = lines[k]
    check("_DLG_TAKEN" in guard, "详情调用受名额保护（if ... and not _DLG_TAKEN）")
    check("not _DLG_TAKEN" in guard, "条件是「名额没被占」才开")
    try_ok = any("try:" in x for x in det) and any("except Exception" in x for x in det)
    check(try_ok, "详情调用整个包在 try/except 里（第三层保险，崩了也只降级成一行说明）")

    # ---------------------------------------------------------------- D 被占时不清焦点
    print("\nD 名额被占时不清焦点")
    i_elif = _idx(lines, "elif _focus and _DLG_TAKEN:", max(0, i_detail - 4))
    if i_elif >= 0:
        elif_blk = _block(lines, i_elif)
        elif_txt = "\n".join(elif_blk)
        check("pop(" not in elif_txt and "_DK_FOCUS] =" not in elif_txt,
              "被占分支不清焦点（等问答关掉，下一页自然摊开）")
        check("_render_detail(" not in elif_txt,
              "被占分支不再尝试开详情（那正是崩掉的那一行）")
    else:
        check(False, "被占分支存在（elif _focus and _DLG_TAKEN:）")

    # ---------------------------------------------------------------- E 弹窗总数
    print("\nE 弹窗定义只在两个模块")
    dialogs = []
    for p in ROOT.rglob("*.py"):
        low = str(p).lower()
        if any(x in low for x in (".venv", "site-packages", ".trash_tmp", "\\tests\\")):
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if re.search(r"@st\.dialog\(", line):
                dialogs.append((str(p.relative_to(ROOT)), i))
    # ⚠ Windows 上 `relative_to` 给的是反斜杠，断言里写的是正斜杠 ——
    #   不统一就会被判「不在 paper2lab 里」这种假阴性（本轮已踩一次）。
    names = sorted({d[0].replace("\\", "/") for d in dialogs})
    check(len(dialogs) == 2, f"全仓恰好两个弹窗定义（现有 {len(dialogs)} 个）")
    check(names == ["paper2lab/deckui.py", "paper2lab/qaui.py"],
          f"两个都在 paper2lab 里（实际：{names}）")
    check("app.py" not in names, "app.py 自己不定义弹窗（它只负责编排名额）")

    # ---------------------------------------------------------------- F 调用点唯一
    print("\nF 调用点各一处")
    n_det = len(re.findall(r"^\s+_render_detail\(", app, flags=re.M))
    n_qa = len(re.findall(r"^\s+_qa_dialog\(", app, flags=re.M))
    check(n_det == 1, f"详情渲染在 app.py 只有一处调用（现有 {n_det}）")
    check(n_qa == 1, f"问答弹窗在 app.py 只有一处调用（现有 {n_qa}）")

    # ---------------------------------------------------------------- 收尾
    print("\n" + "=" * 58)
    ok = N - len(BAD)
    print(f"共 {N} 项：通过 {ok}，未通过 {len(BAD)}")
    if BAD:
        for x in BAD:
            print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
