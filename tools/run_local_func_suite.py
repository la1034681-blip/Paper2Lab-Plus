"""第一套：本地模型在本项目中的「独立功能」验证（真调本机 8080，不用替身）。

为什么要单独一套
----------------
全链路跑通 ≠ 每个功能都能用。全链路里模型只要有一处成功，链路就"过得去"，
某个能力（例如读图、排计划、模型先判）可能一直是空转的，从外面看不出来。
所以这一套**逐个能力单独叫一遍**，各自计时、各自判据，谁的锅谁背。

比的不是模型跑分，而是「这个项目里的这个功能，喂进去能不能出来东西」：
  01 端点与模型清单        本机服务活着、模型名对得上
  02 基础对话              一句话问答能出中文，不是空串
  03 结构化输出            要求返回 JSON 时真能返回可解析的 JSON（项目大量用它）
  04 归因解释              LLMAnalyzer 能给一条不一致项写出人话解释
  05 问答                  qa.answer 按提示词模板答满，且没落到「调用失败」
  06 排计划                planner 让模型排，planner 标记必须是 llm 而非 rule
  07 模型先判              judge_llm 给参数配对下判断，三态之一，且不覆盖规则结论
  08 视觉读图              vision 从论文里读 1 个区域，能出候选参数
  09 长上下文              接近 3k token 的输入不报 400
  10 预算口径              本机与远程两档数值正确（超时/额度写死会必挂）
  11 连续两次              单槽下第二个请求也成功（并发会撞爆上下文）
  12 中文输出              回答以中文为主，不把额度全花在英文推理上

用法：
    .venv\\Scripts\\python.exe -X utf8 tools\\run_local_func_suite.py
    （可选）--skip-vision   跳过读图项（本机读图约 30s/区域）
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from paper2lab import envfile, llm, qa, planner, vision      # noqa: E402
from paper2lab import judge_llm                              # noqa: E402
from paper2lab.toolbox import TOOLS, Job                      # noqa: E402
from verify_qa import _fake_audit                             # noqa: E402

# ⚠ 开关参数必须先剔掉再当"只跑哪几项"的过滤器 ——
#   否则只传 `--skip-vision` 时，过滤器会去匹配这个开关名，结果一项都不跑、还报"全过"。
ONLY = [a for a in sys.argv[1:] if not a.startswith("-")]
SKIP_VISION = "--skip-vision" in sys.argv

KEY = envfile.get("LLM_API_KEY") or ""
URL = envfile.get("LLM_BASE_URL") or ""
MODEL = envfile.get("LLM_MODEL") or ""
VKEY = envfile.get("VISION_API_KEY") or KEY
VURL = envfile.get("VISION_BASE_URL") or URL
VMODEL = envfile.get("VISION_MODEL") or MODEL

N = 0
BAD: list[str] = []
ROWS: list[tuple[str, bool, str]] = []      # (名称, 是否通过, 实测备注)


def check(ok, label: str, note: str = "") -> bool:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label + (f"   [{note}]" if note else ""))
    ROWS.append((label, ok, note))
    if not ok:
        BAD.append(label)
    return ok


def _cn_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")
    return n / max(1, len(s))


def item(tag: str) -> bool:
    """按名字筛单项：命令行给了名字就只跑这些。"""
    return (not ONLY) or any(tag in x for x in ONLY)


def main() -> int:
    print("=" * 74)
    print("第一套 · 本地模型独立功能验证")
    print(f"端点 {URL or '(未配置)'}   模型 {MODEL or '(未配置)'}   Key {'已填' if KEY else '(空)'}")
    print("=" * 74)

    # ---------------------------------------------------------------- 01 端点
    if item("01"):
        print("\n01 端点与模型清单")
        t = time.time()
        try:
            import httpx
            c = httpx.Client(trust_env=False, timeout=10)
            h = c.get(URL.replace("/v1", "") + "/health").json()
            r = c.get(URL.rstrip("/") + "/models").json()
            names = [m.get("id") for m in (r.get("data") or [])]
            check(h.get("status") == "ok", "本机服务 health 返回 ok", f"{time.time() - t:.1f}s")
            check(any(MODEL == n or (MODEL or "") in str(n) for n in names),
                  "模型清单里有配置的模型名", f"清单 {names}")
        except Exception as ex:            # noqa: BLE001
            check(False, "端点探测失败", f"{type(ex).__name__}: {ex}")
            print(traceback.format_exc(limit=2))

    # ---------------------------------------------------------------- 02 基础对话
    txt2 = ""
    if item("02"):
        print("\n02 基础对话")
        t = time.time()
        try:
            txt2 = llm._call_deepseek("用一句中文回答：一加一等于几？只给答案。",
                                      KEY, max_tokens=60, timeout=180,
                                      base_url=URL, model=MODEL)
            dt = time.time() - t
            check(bool((txt2 or "").strip()), "返回非空", f"{dt:.1f}s，{len(txt2 or '')} 字")
            check("2" in (txt2 or "") or "二" in (txt2 or ""), "答案里有一加一的结果",
                  (txt2 or "")[:30].replace("\n", " "))
        except Exception as ex:            # noqa: BLE001
            check(False, "基础对话失败", f"{type(ex).__name__}: {ex}")

    # ---------------------------------------------------------------- 03 结构化输出
    if item("03"):
        print("\n03 结构化输出（JSON）")
        t = time.time()
        try:
            # ⚠ ask_json 返回的是 (dict|None, 说明) 两元组，不是 dict —— 别当 dict 判
            obj, note3 = llm.ask_json(
                '给你两个参数：论文写 1e-4，代码写 0.001。请只输出 JSON：'
                '{"verdict":"一致"或"不一致","why":"一句话理由"}',
                KEY, mode="local-suite", base_url=URL, model=MODEL)
            dt = time.time() - t
            ok = isinstance(obj, dict) and ("verdict" in obj or "why" in obj)
            check(ok, "返回可解析的 JSON 且带预期键",
                  f"{dt:.1f}s，{str(obj)[:60]}｜{note3[:24]}")
        except Exception as ex:            # noqa: BLE001
            check(False, "结构化输出失败", f"{type(ex).__name__}: {ex}")

    # ---------------------------------------------------------------- 04 归因解释
    if item("04"):
        print("\n04 归因解释（LLMAnalyzer）")
        t = time.time()
        try:
            audit = _fake_audit()
            f = audit.findings[0]           # learning_rate，不一致
            an = llm.LLMAnalyzer(KEY, base_url=URL, model=MODEL)
            text = an.analyze(f)
            dt = time.time() - t
            check(bool((text or "").strip()), "写出归因文字", f"{dt:.1f}s，{len(text or '')} 字")
            # ⚠ 判据别写成「必须 real_calls>=1」：`LLMAnalyzer` 自带结果缓存，
            #   同一份替身材料第二次跑必然命中缓存（real=0, cache=1）——
            #   那不是没调模型，是缓存层按设计生效了（2026-10-31 本轮误判过一次）。
            #   要证明的是「没落到 mock」，命中缓存同样说明模型通道是通的。
            check(an.mode != "mock", "没落到 mock（模型通道是通的）",
                  f"real={an.real_calls} cache={an.cache_hits} mode={an.mode}")
            if an.cache_hits and not an.real_calls:
                print("     备注：本次命中归因缓存（同一提示词之前算过），不算失败")
            if text:
                check(_cn_ratio(text) > 0.25, "归因以中文为主",
                      f"中文占比 {_cn_ratio(text):.0%}")
        except Exception as ex:            # noqa: BLE001
            check(False, "归因解释失败", f"{type(ex).__name__}: {ex}")

    # ---------------------------------------------------------------- 05 问答
    body5 = ""
    if item("05"):
        print("\n05 问答（qa.answer）")
        t = time.time()
        try:
            body5, note = qa.answer(_fake_audit(), "学习率到底是多少？复现该按哪个改？",
                                    [], api_key=KEY, base_url=URL, model=MODEL)
            dt = time.time() - t
            check(len((body5 or "").strip()) >= 120, "正文有实质内容（≥120 字）",
                  f"{dt:.1f}s，{len(body5 or '')} 字")
            check("失败" not in (note or ""), "说明行不是「调用失败」", (note or "")[:40])
            check(("1e-4" in (body5 or "")) or ("0.001" in (body5 or "")),
                  "答到了具体数值（没跑题）", "")
            check(_cn_ratio(body5 or "") > 0.25, "回答以中文为主",
                  f"中文占比 {_cn_ratio(body5 or ''):.0%}")
        except Exception as ex:            # noqa: BLE001
            check(False, "问答失败", f"{type(ex).__name__}: {ex}")
            print(traceback.format_exc(limit=3))

    # ---------------------------------------------------------------- 06 排计划
    if item("06"):
        print("\n06 排计划（planner + 模型）")
        t = time.time()
        try:
            job = Job(fingerprint="local-suite", pdf_path=str(ROOT / "demo" / "paper.pdf"),
                      code_path=str(ROOT / "demo" / "project.zip"),
                      api_key=KEY, llm_base_url=URL, llm_model=MODEL,
                      llm_planner=True)
            plan = planner.make_plan(job, use_llm=True)
            dt = time.time() - t
            who = getattr(plan, "planner", "")
            # ⚠ 工具顺序存在 `Plan.order`（不是 `tools`）—— 读错字段会得到空列表，
            #   看起来像"模型一件工具都没排"，实际是断言写错了（本轮踩过）。
            keys = list(getattr(plan, "order", None) or ())
            # ⚠ 工具箱里存的是**工具对象**（带 .key），不是字符串 ——
            #   直接 `"align" in TOOLS` 恒为 False，会报成"全部越界"（本轮踩过）。
            _all = {getattr(t, "key", t) for t in TOOLS}
            legal = [k for k in keys if k in _all]
            check(who == "llm", "计划由模型排出（不是规则兜底）",
                  f"planner={who}，{dt:.1f}s")
            check(len(keys) >= 3, "排出的工具数合理（≥3）", f"{len(keys)} 件：{keys[:8]}")
            check(len(legal) == len(keys), "计划里的工具键全部在工具箱清单内",
                  f"越界 {[k for k in keys if k not in _all]}")
            if who != "llm":
                # 回落时 planner 会把原因写在 notes[0] 里，直接打出来，别让人猜
                print("     回落原因：", str((getattr(plan, 'notes', None) or [''])[0])[:160])
        except Exception as ex:            # noqa: BLE001
            check(False, "排计划失败", f"{type(ex).__name__}: {ex}")
            print(traceback.format_exc(limit=2))

    # ---------------------------------------------------------------- 07 模型先判
    if item("07"):
        print("\n07 模型先判（judge_llm，只写 llm_* 不覆盖规则结论）")
        t = time.time()
        try:
            audit = _fake_audit()
            before = [(f.param_key, getattr(f.status, "value", f.status))
                      for f in audit.findings]
            got, note = judge_llm.judge_findings(audit.findings, api_key=KEY,
                                                 base_url=URL, model=MODEL)
            stat = judge_llm.apply_findings(audit.findings, got)
            after = [(f.param_key, getattr(f.status, "value", f.status))
                     for f in audit.findings]
            dt = time.time() - t
            check(bool(got), "模型给出了判断", f"{note[:40]}，{dt:.1f}s")
            check(stat["judged"] >= 1, "至少有一条进入比对", str(stat))
            check(before == after, "规则结论未被模型覆盖（status 原样）", "")
            three = all(getattr(f, "llm_agree", None) in (True, False, None)
                        for f in audit.findings)
            check(three, "一致性标记只有三态", "")
        except Exception as ex:            # noqa: BLE001
            check(False, "模型先判失败", f"{type(ex).__name__}: {ex}")
            print(traceback.format_exc(limit=2))

    # ---------------------------------------------------------------- 08 视觉读图
    if item("08"):
        print("\n08 视觉读图（vision，取 1 个区域）")
        if SKIP_VISION:
            print("  ⏭ 按要求跳过")
        else:
            t = time.time()
            try:
                # ⚠ 样本必须选**有图**的：`demo/paper.pdf` 只有 3 页、图注 0 条，
                #   拿它测读图只会得到「未发现需要读图的区域」——
                #   那不是读图坏了，是没东西可读（2026-10-31 本轮误判过一次）。
                #   `demo/figure_demo.pdf` 是专门造的图表样本（2 页 / 6 条图注）。
                _pdf = ROOT / "demo" / "figure_demo.pdf"
                if not _pdf.exists():
                    _pdf = ROOT / "demo" / "paper.pdf"
                ev, meta = vision.analyze_images(
                    str(_pdf), max_regions=1,
                    api_key=VKEY, base_url=VURL, model=VMODEL, budget_seconds=240)
                dt = time.time() - t
                check(meta.get("enabled") is True, "视觉通道已启用（多模态真的被用上）",
                      f"model={meta.get('model')}，{dt:.1f}s，样本 {_pdf.name}")
                check(meta.get("regions_read", 0) >= 1, "至少读了 1 个区域",
                      f"regions_read={meta.get('regions_read')}")
                check(isinstance(ev, dict), "返回参数证据字典", f"{len(ev)} 个参数")
                if meta.get("notes"):
                    print("     备注：", " / ".join(str(x) for x in meta["notes"])[:150])
            except Exception as ex:        # noqa: BLE001
                check(False, "视觉读图失败", f"{type(ex).__name__}: {ex}")
                print(traceback.format_exc(limit=2))

    # ---------------------------------------------------------------- 09 长上下文
    if item("09"):
        print("\n09 长上下文（约 3k token）")
        t = time.time()
        try:
            # ⚠ 长上下文的判据必须**只有唯一答案**：原来问「冒号前的四个字」，
            #   填充文本里根本没有冒号，模型只能猜，答什么都不能算错（本轮踩过）。
            #   改成在文末埋一个唯一的词，问它是什么。
            filler = "以下是一段背景描述，仅作占位，不含任何结论。" * 80   # ≈2500 字
            prompt = (filler + "\n\n上面这段背景的最后一个词是「青花瓷」。"
                      "请问最后一个词是什么？只回答那个词。")
            out = llm._call_deepseek(prompt, KEY, max_tokens=50, timeout=300,
                                     base_url=URL, model=MODEL)
            dt = time.time() - t
            check(bool((out or "").strip()), "长输入下模型仍有输出",
                  f"{dt:.1f}s，{len(prompt)} 字输入")
            check("青花瓷" in (out or ""), "答到了长文末尾的内容（没有中途断掉）",
                  (out or "")[:30].replace("\n", " "))
        except Exception as ex:            # noqa: BLE001
            check(False, "长上下文失败", f"{type(ex).__name__}: {str(ex)[:120]}")

    # ---------------------------------------------------------------- 10 预算口径
    if item("10"):
        print("\n10 预算口径（一处真相）")
        mt_l, to_l = llm.qa_budget(URL)
        mt_r, to_r = llm.qa_budget("https://api.deepseek.com")
        check((mt_l, to_l) == (1000, 360), "本机端点预算 = 1000 token / 360s", f"{mt_l}/{to_l}s")
        check((mt_r, to_r) == (1800, 75), "远程端点预算 = 1800 token / 75s", f"{mt_r}/{to_r}s")
        check(llm.is_local_endpoint(URL) and not llm.is_local_endpoint("https://api.deepseek.com"),
              "本机/远程判定正确", "")

    # ---------------------------------------------------------------- 11 连续两次
    if item("11"):
        print("\n11 连续两次（单槽不撞车）")
        try:
            t1 = time.time()
            b1, n1 = qa.answer(_fake_audit(), "这次审计有几条不一致？只答数量。", [],
                               api_key=KEY, base_url=URL, model=MODEL)
            d1 = time.time() - t1
            t2 = time.time()
            b2, n2 = qa.answer(_fake_audit(), "再问一次：有几条不一致？只答数量。", [],
                               api_key=KEY, base_url=URL, model=MODEL)
            d2 = time.time() - t2
            check(bool(b1.strip()) and "失败" not in n1, "第一次成功", f"{d1:.1f}s")
            check(bool(b2.strip()) and "失败" not in n2, "紧接着第二次也成功",
                  f"{d2:.1f}s")
        except Exception as ex:            # noqa: BLE001
            check(False, "连续两次失败", f"{type(ex).__name__}: {ex}")

    # ---------------------------------------------------------------- 12 中文输出
    if item("12"):
        print("\n12 中文输出")
        s = (txt2 or "") + (body5 or "")
        if s.strip():
            check(_cn_ratio(s) > 0.25, "本机模型输出以中文为主",
                  f"中文 {_cn_ratio(s):.0%}（{len(s)} 字样本）")
        else:
            print("  ⏭ 没有可分析的样本（02/05 未跑或为空）")

    # ---------------------------------------------------------------- 汇总
    print("\n" + "=" * 74)
    ok = N - len(BAD)
    print(f"共 {N} 项：通过 {ok}，未通过 {len(BAD)}")
    # ⚠ 一项都没跑必须算不合格：「0 项 / 0 未通过」打印出来太像全绿了
    #   （2026-10-31 踩过：命令行开关被当成项名过滤，静默空跑还退出码 0）。
    if N == 0:
        print("  ✗ 一项都没执行 —— 检查命令行过滤条件")
        return 1
    if BAD:
        for x in BAD:
            print("  ✗", x)
    # 落一份机器可读的结果，便于汇总时引用
    out = ROOT / "cache" / "local_func_suite.json"
    try:
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(
            {"items": [{"name": a, "ok": b, "note": c} for a, b, c in ROWS],
             "total": N, "passed": ok}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"明细已写入 {out}")
    except Exception:                      # noqa: BLE001
        pass
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
