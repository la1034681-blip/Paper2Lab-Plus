"""坏输入的**语义边界**守卫（2026-10-29 新建）。

这个脚本守的不是「能不能跑」，而是**三种情况的界线有没有串**：

    ┌ 文件在、但打不开      → 降级成「没读到」，把原因写进页覆盖表，不抛异常
    ├ 路径根本不存在        → 必须抛错。静默出一份空报告比崩溃更危险
    └ 压根没给（空字符串）  → 按「没有代码包」走，不抛异常

为什么单独建一个脚本守它：上一轮修「伪 PDF 崩溃」时，第一版兜底把
「文件在但打不开」和「路径根本不存在」一起吞了 —— 填错路径不再报错，
而是安静地出一份「论文没参数」的空报告。用户会以为审计做完了。
当时是 `verify_stages.py` 里一条旧断言把它抓住的，说明这条边界值钱；
现在固化成独立脚本，改 `paper_parser.py` / `code_analyzer.py` 时先跑它。

跑法：  .venv/Scripts/python.exe tests/verify_bad_inputs.py
离线、不联网、不写项目目录（坏文件全在系统临时目录里造）。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.paper_parser import parse_paper          # noqa: E402
from paper2lab.code_analyzer import analyze_code        # noqa: E402
from paper2lab.pipeline import run_audit                # noqa: E402

OK = 0
BAD = 0
FAILS: list[str] = []


def check(ok, label: str, extra: str = "") -> None:
    global OK, BAD
    assert isinstance(ok, bool), f"断言值必须是 bool，拿到 {type(ok).__name__}"
    if ok:
        OK += 1
        print(f"  [ok] {label}" + (f"  —— {extra}" if extra else ""))
    else:
        BAD += 1
        FAILS.append(label)
        print(f"  [!!] {label}" + (f"  —— {extra}" if extra else ""))


def sec(title: str) -> None:
    print(f"\n{'─' * 66}\n{title}\n{'─' * 66}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="p2l_bad_"))

    # 材料：四种坏文件
    fake_pdf = tmp / "fake.pdf"
    fake_pdf.write_text("this is plain text, not a pdf\n" * 30, encoding="utf-8")

    good_pdf = ROOT / "demo" / "paper.pdf"
    trunc = tmp / "truncated.pdf"
    if good_pdf.exists():
        blob = good_pdf.read_bytes()
        trunc.write_bytes(blob[: max(1, len(blob) // 3)])

    empty_zip = tmp / "empty.zip"
    empty_zip.write_bytes(b"")

    not_here = tmp / "nope.pdf"

    sec("一、文件在、但打不开 —— 必须降级，不许抛异常")

    try:
        info = parse_paper(str(fake_pdf))
        check(True, "伪 PDF（文本改名）不抛异常")
        check(info.page_count == 0, "页数如实记为 0", f"page_count={info.page_count}")
        cov = info.page_coverage or []
        check(bool(cov) and any(str(c.get("status")) == "unreadable" for c in cov),
              "页覆盖表里写明了「读不出来」及原因",
              f"{cov[0].get('note', '')[:52] if cov else '空'}")
    except Exception as e:  # noqa: BLE001
        check(False, "伪 PDF（文本改名）不抛异常", f"{type(e).__name__}: {e}")

    if trunc.exists():
        try:
            t = parse_paper(str(trunc))
            check(True, "截断的 PDF（只留前 1/3）不抛异常", f"读到 {t.page_count} 页")
        except Exception as e:  # noqa: BLE001
            check(False, "截断的 PDF（只留前 1/3）不抛异常", f"{type(e).__name__}: {e}")

    sec("二、路径根本不存在 —— 必须抛错（这条是上一轮差点踩穿的）")

    raised, kind = False, ""
    try:
        parse_paper(str(not_here))
    except Exception as e:  # noqa: BLE001
        raised, kind = True, type(e).__name__
    check(raised, "论文路径不存在时明确抛错，不静默出空报告", f"抛的是 {kind}")
    # pymupdf 自带一个同名 FileNotFoundError（继承 RuntimeError），不能按内置类型认
    check("FileNotFound" in kind or "Runtime" in kind,
          "抛的是「文件找不到」这一类，不是别的错", f"{kind}")

    sec("三、压根没给 / 给的包是坏的 —— 按「没有」处理，不抛异常")

    try:
        empty = analyze_code("")
        check(True, "空路径（= 没给代码包）不抛异常")
        check(getattr(empty, "python_files", 1) == 0, "空路径返回空结果，不去扫当前目录",
              f"python_files={getattr(empty, 'python_files', '?')}")
    except Exception as e:  # noqa: BLE001
        check(False, "空路径（= 没给代码包）不抛异常", f"{type(e).__name__}: {e}")

    try:
        z = analyze_code(str(empty_zip))
        check(True, "0 字节 zip 不抛异常（先判是不是有效压缩包）")
        check(getattr(z, "python_files", 1) == 0, "0 字节 zip 按「无源码」处理",
              f"python_files={getattr(z, 'python_files', '?')}")
    except Exception as e:  # noqa: BLE001
        check(False, "0 字节 zip 不抛异常（先判是不是有效压缩包）", f"{type(e).__name__}: {e}")

    # 2026-10-29 修：代码包**填错路径**必须与论文同口径 —— 明确抛错，
    # 不能静默返回空 CodeInfo（那会被读成"代码里确实没有这些参数"）。
    raised_c, kind_c = False, ""
    try:
        analyze_code(str(tmp / "no_such_code.zip"))
    except Exception as e:  # noqa: BLE001
        raised_c, kind_c = True, type(e).__name__
    check(raised_c, "代码包路径不存在时明确抛错，不静默出空结果", f"抛的是 {kind_c}")

    sec("二之二、入口层 —— 真实用户走的是 run_audit，不是底层函数（这层以前是假绿）")
    # 2026-10-30 补：上面那条只测了 `analyze_code` 自己抛不抛，而真实用户走的是
    # run_audit → run_job → 工具。实测更糟的地方在于**根本没走到抛错**：
    # 计划器看到 `Path(code_path).exists() is False`，直接判定「未提供代码包」，
    # 把代码扫描整个跳过，最后照样出一份 18221 字的报告，里面写着
    # 「代码侧比对整体缺席」。用户会把这句话当成事实 —— 实际只是路径没指到东西。
    # 所以校验必须放在**入口的最外侧**，三条入口都要拦。
    from paper2lab.preflight import InputPathError

    good = str(ROOT / "demo" / "D题论文.pdf")
    if not Path(good).exists():
        good = str(ROOT / "demo" / "paper.pdf")

    def _entry_raises(fn, label: str, *args) -> None:
        try:
            fn(*args)
            check(False, label, "没有抛错")
        except InputPathError:
            check(True, label)
        except Exception as e:      # noqa: BLE001
            check(False, label, f"抛的是 {type(e).__name__}: {e}")

    _entry_raises(run_audit, "run_audit：论文路径不存在 → 抛错",
                  str(tmp / "no_such.pdf"), str(Path(good)))
    _entry_raises(run_audit, "run_audit：代码包路径不存在 → 抛错",
                  good, str(tmp / "no_such_code.zip"))
    _entry_raises(run_audit, "run_audit：论文一栏给了目录 → 抛错",
                  str(ROOT / "demo"), str(Path(good)))

    try:
        from paper2lab.graph import run_audit_graph, run_audit_multagent
        _entry_raises(run_audit_graph, "图版入口：代码包路径不存在 → 抛错",
                      good, str(tmp / "no_such_code.zip"))
        _entry_raises(run_audit_multagent, "多智能体入口：代码包路径不存在 → 抛错",
                      good, str(tmp / "no_such_code.zip"))
    except ImportError as e:        # langgraph 不在时如实跳过，不伪装成通过
        print(f"  [--] langgraph 不可用，图版两条入口未测：{e}")

    # 空代码包是**合法**输入（= 这次没有代码），绝不能跟着一起报错
    from paper2lab.preflight import check_inputs
    try:
        check_inputs(good, "")
        check(True, "空代码包仍然合法（= 没有代码，不报错）")
    except Exception as e:          # noqa: BLE001
        check(False, "空代码包仍然合法（= 没有代码，不报错）", f"{type(e).__name__}: {e}")

    # 兜底措辞：万一有人绕过入口直接构造 Job，也不许把「填错了」说成「没给」
    try:
        from paper2lab.planner import _rule_plan
        from paper2lab.toolbox import Job
        _job = Job(fingerprint="", pdf_path=good,
                   code_path=str(tmp / "no_such_code.zip"))
        _plan = _rule_plan(_job)
        _reason = next((d.reason for d in _plan.decisions
                        if d.tool == "analyze_code"), "")
        check("路径不存在" in _reason,
              "计划器把「路径填错」与「没提供代码包」分开说", _reason[:60])
    except Exception as e:          # noqa: BLE001
        check(False, "计划器把「路径填错」与「没提供代码包」分开说",
              f"{type(e).__name__}: {e}")

    sec("四、论文一页没读到时的结论口径 —— 不许假装审计有效")

    real_zip = None
    for cand in sorted((ROOT / "demo").glob("*.zip")):
        real_zip = cand
        break
    sample_code = str(real_zip) if real_zip else ""

    try:
        audit, report = run_audit(str(fake_pdf), sample_code)
        check(audit is not None, "端到端：伪 PDF 仍返回一份审计结果（不崩）")
        if audit is not None:
            conf = (audit.readiness or {}).get("audit_confidence") or {}
            level = str(conf.get("level", "")) if isinstance(conf, dict) else str(conf)
            check(level in ("低", "low"), "一页都没读到 → 可信度判「低」，不给「中」",
                  f"level={level}")
            faked = [f for f in (audit.findings or [])
                     if str(getattr(f.status, "value", f.status)) == "consistent"]
            check(not faked, "没有任何一条发现被伪造成「一致」", f"{len(faked)} 条")
            check(bool(report), "仍然给得出一份报告（写明读到 0 页）",
                  f"{len(report or '')} 字")
    except Exception as e:  # noqa: BLE001
        check(False, "端到端：伪 PDF 仍返回一份审计结果（不崩）", f"{type(e).__name__}: {e}")

    print(f"\n{'=' * 66}")
    print(f"坏输入语义边界: {OK + BAD} 项，通过 {OK}，未通过 {BAD}")
    if FAILS:
        print("未通过项：")
        for f in FAILS:
            print(f"  · {f}")
    print("=" * 66)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
