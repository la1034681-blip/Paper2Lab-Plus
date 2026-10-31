"""问答窗口的离线体检（2026-10-27）。

用户口径：「加一个问答窗口入口，点开能多轮问、由 LLM 答疑」。
这里只测**口径**（问答到底依据什么、拿不到模型时会不会编），不测像素；
像素与点击由 probes/shot_qa.py 真机点。

为什么全程不联网：一问一答要花钱，而且"今天有没有额度"不该变成
测试结果的一部分（与 verify_stages 同口径）。所以：
  · 正常路径用 `api_key=""`（未配置）走证据检索分支；
  · 失败路径用一个本地必然拒绝连接的地址（毫秒级失败，不可能是网络慢）。

五组：
  A 空材料  —— 没有结论时不许崩、不许瞎答
  B 事实摘要 —— 数字/出处必须来自 audit，一致项不进"问题清单"
  C 按问题展开 —— 命中什么补什么，没命中不硬凑
  D 降级与多轮 —— 没 key 只列证据；调用失败不抛；历史只带最近 8 轮
  E 只读保证 —— 问答前后 findings / readiness 一个字节都不变

用法:
    .venv\\Scripts\\python.exe tests\\verify_qa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import qa                                        # noqa: E402
from paper2lab.models import (AuditResult, Category, CodeInfo,  # noqa: E402
                              Confidence, Evidence, Finding, PaperInfo, Risk,
                              Status)

CHECKS: list = []


def check(ok, label: str, detail: str = "") -> None:
    """断言防呆：ok 必须是真 bool（and 会返回操作数，不许直接塞进来）。"""
    assert isinstance(ok, bool), f"断言值必须是 bool, 拿到 {type(ok).__name__}: {label}"
    CHECKS.append((ok, label, detail))
    print(f"  {'✅' if ok else '❌'} {label}" + (f" — {detail}" if detail else ""))


def _fake_audit() -> AuditResult:
    """一份结构真实、内容少量的 audit（不是从 PDF 跑的，所以快且确定）。"""
    paper = PaperInfo(title="Reproducing Sparse Attention", page_count=9)
    code = CodeInfo(python_files=4, requirements={"torch": "2.1"}, seed_fixed=False)
    findings = [
        Finding(
            param_key="learning_rate", display_name="Learning Rate",
            category=Category.PARAM, status=Status.INCONSISTENT,
            confidence=Confidence.CONFIRMED, risk=Risk.HIGH,
            paper=Evidence(source="paper", location="Page 3 · 4. Training Setup",
                           snippet="lr = 1e-4", raw_value="1e-4", normalized=0.0001),
            code=Evidence(source="code", location="train.py:11",
                          snippet="lr=0.001", raw_value="0.001", normalized=0.001),
            note="两侧相差 10 倍",
            risk_reason="学习率决定能否复现同等精度，量级错会直接跑飞"),
        Finding(
            param_key="batch_size", display_name="Batch Size",
            category=Category.PARAM, status=Status.CONSISTENT,
            confidence=Confidence.CONFIRMED, risk=Risk.NONE,
            paper=Evidence(source="paper", location="Page 3", snippet="bs = 64",
                           raw_value="64", normalized=64),
            code=Evidence(source="code", location="train.py:12", snippet="bs=64",
                          raw_value="64", normalized=64),
        ),
        Finding(
            param_key="seed", display_name="Random Seed",
            category=Category.RECORD, status=Status.NOT_FIXED,
            confidence=Confidence.CONFIRMED, risk=Risk.MEDIUM,
            paper=Evidence(source="paper", location="Page 4", snippet="seed=42",
                           raw_value="42", normalized=42),
            code=None,
            note="代码里没有固定种子",
        ),
    ]
    audit = AuditResult(paper=paper, code=code, findings=findings)
    audit.readiness = {
        "total": 62,
        "categories": {"环境": 100, "数据集": 90, "参数": 40, "实验记录": 60, "结果": None},
        "unverified": {"结果": "本次未提供实测结果，结果一致性未核验"},
        "dominant_issues": ["Learning Rate"],
    }
    audit.execution_log = {
        "plan": {"planner": "llm", "steps": [
            {"key": "parse_paper", "action": "run", "reason": "必须先读论文"},
            {"key": "llm_analyze", "action": "skip", "reason": "多智能体档不重复归因"},
        ]},
        "tools": [{"key": "parse_paper", "status": "ok"},
                  {"key": "align", "status": "ok"},
                  {"key": "llm_analyze", "action": "skip"}],
    }
    audit.selfcheck = {"checked": 42, "passed": 42, "failed": 0}
    # 真实审计里 coverage 一定存在（问「有没有漏查」就是问它），假材料也要给一份，
    # 否则会测出一个现实里不会出现的分支。
    audit.coverage = {"pages_parsed": 8, "pages_total": 8, "page_ratio": 1.0,
                      "decl_ratio": 0.75, "confidence_level": "中",
                      "confidence_score": 68, "kinds_valued": 9, "kinds_seen": 9}
    audit.hypotheses = [type("H", (), {
        "claim": "论文的学习率 1e-4 与代码 0.001 不是同一个量级",
        "falsifiable_by": "在代码里找到 1e-4 的赋值",
        "verdict": "待验", "score": 9, "why": "硬冲突"})()]
    audit.hypothesis_total = 1
    audit.repro_plan = {"summary": "先统一学习率再跑 3 个种子"}
    return audit


# ================================================================ A 空材料
def part_empty() -> None:
    print("\n===== A 空材料（还没跑审计） =====")
    txt = qa.facts(None)
    check(bool("没有审计结论" in txt), "没结论时 facts 如实说明，不编数据")
    body, note = qa.answer(None, "", [])
    check((body == "" and note == "空问题，未调用"), "空问题不调模型",
          f"note={note}")
    body, note = qa.answer(None, "这个平台怎么用", [], api_key="")
    check(bool(body and "还没有审计结论" in body), "没结论 + 没模型：请他先跑一轮，不瞎答")
    check(bool("未配置模型" in note or "不调用模型" in note),
          "没 key / 没结论时说明里写明原因（不假装答过）", note)
    # 没结论 + **有** key：也不许调模型 —— 实测它会编出「论文 3.2 节 / train.py:L45」这种出处
    body, note = qa.answer(None, "学习率在论文和代码里各是多少？", [],
                           api_key="sk-fake-but-present", model="whatever")
    check(bool(body == qa._NO_AUDIT and "3.2" not in body and "L45" not in body),
          "没结论 + 有 key：不调模型，也不编页码/行号")


# ================================================================ B 事实摘要
def part_facts() -> None:
    print("\n===== B 事实摘要（数字与出处只能来自 audit） =====")
    audit = _fake_audit()
    txt = qa.facts(audit, {"source": "用户上传", "files": "a.pdf + b.zip"})
    for want, label in (
        ("62", "就绪度取的是 readiness['total']"),
        ("Learning Rate", "不一致项在清单里"),
        ("Page 3 · 4. Training Setup", "带论文页码·章节"),
        ("train.py:11", "带代码文件:行号"),
        ("1e-4", "带论文原值"),
        ("0.001", "带代码原值"),
        ("相差 10 倍", "带说明 note"),
        ("42 项", "带自检项数"),
        ("计划由模型排定", "带计划来源（planner）"),
    ):
        check(bool(want in txt), label, want)
    # 一致项不该混进"其余 N 条"的清单里 —— 否则用户会以为那条也有问题
    body = txt.split("【逐条对表】")[1].split("【核验自证】")[0]
    check(bool("Batch Size" not in body), "对得上的项不进问题清单")
    check(bool("共 3 条" in body), "对表条数现场数出来", "3")
    check(bool("对得上 1 条" in body), "一致条数单独报")


# ================================================================ C 按问题展开
def part_focus() -> None:
    print("\n===== C 按问题展开（命中才补） =====")
    audit = _fake_audit()
    p = qa.focus(audit, "这几个数为什么对不上？")
    check(bool("【参数细节" in p), "问「为什么对不上」-> 展开参数原文")
    check(bool("lr = 1e-4" in p or "1e-4" in p), "展开里有论文原值")
    check(bool("风险理由" in p), "展开里有风险理由")
    rp = qa.focus(audit, "我要复现这篇论文，最先该改哪里？")
    check(bool("【复现方案】" in rp), "问「复现先改哪」-> 展开复现方案")
    hp = qa.focus(audit, "假设卡说了什么")
    check(bool("【假设卡" in hp), "问「假设」-> 展开假设卡")
    pl = qa.focus(audit, "为什么有的工序被跳过")
    check(bool("【本次计划" in pl), "问「跳过」-> 展开计划（含跳过理由）")
    none = qa.focus(audit, "今天天气怎么样")
    check(bool(none == ""), "问无关话题不硬凑内容")


# ================================================================ C2 命中率（中文问法）
def part_hits() -> None:
    print("\n===== C2 按问题挑事实（中文问法也要能对上） =====")
    audit = _fake_audit()
    # 用户不会说 learning_rate，只会说「学习率」——别名表要和提问用词对得上
    h = qa.top_hits(audit, "学习率差多少")
    check(bool(h and "Learning Rate" in h[0]), "问「学习率」能命中 Learning Rate", h[:1])
    h2 = qa.top_hits(audit, "批大小呢")
    check(bool(h2 and "Batch Size" in h2[0]), "问「批大小」能命中 Batch Size")
    # 词边界：eta(=学习率) 不能被 beta 命中，bs(=批大小) 不能被 subsample 命中
    check(bool(qa.top_hits(audit, "beta1 怎么设") == []), "英文别名不被 beta 误命中")
    check(bool(qa.top_hits(audit, "subsample 呢") == []), "英文别名不被 subsample 误命中")
    # 每条命中都必须逐字来自 findings 里的原值/位置，不许自己拼
    for q in ("学习率差多少", "seed 有没有固定"):
        for line in qa.top_hits(audit, q):
            okk = any((str((f.paper.raw_value if f.paper else "") or "") in line
                       and str((f.paper.location if f.paper else "") or "") in line)
                      or (str((f.code.raw_value if f.code else "") or "") in line
                          and str((f.code.location if f.code else "") or "") in line)
                      for f in audit.findings)
            check(bool(okk), f"命中条目逐字来自原结论（{q}）", line[:40])
    # 「有没有漏查」不该答成「没有相关条目」——覆盖情况本身就是答案
    body, _ = qa.answer(audit, "这次审计有没有漏掉什么", [], api_key="")
    check(bool("没有与这一问直接相关的条目" not in body), "问「有没有漏」不会答成「没有相关条目」")


# ================================================================ D 降级与多轮
def part_answer() -> None:
    print("\n===== D 降级与多轮 =====")
    audit = _fake_audit()
    body, note = qa.answer(audit, "Learning Rate 为什么对不上", [], api_key="")
    check(bool("未配置模型" in note), "没 key：说明里写明未配置")
    check(bool("Learning Rate" in body), "没 key：正文给出命中的证据条目")
    check(bool("模型作答" not in note), "没 key：绝不假装是模型答的")
    check(bool("未做改写" in body), "没 key：明说这是原样列出、不是生成的话")

    # 调用失败：指向本机一个必被拒绝的端口，毫秒级失败（不是"网络慢"）
    # ⚠ 故意制造的失败不许写进 cache/qa_errors.log —— 那是**运维凭据**，
    #   混进测试噪音后，用户报"又失败了"时就没法只靠日志分辨真伪（2026-10-31）。
    _orig_log = qa._log_qa_error
    qa._log_qa_error = lambda *a, **kw: None
    try:
        body, note = qa.answer(audit, "Learning Rate 为什么对不上", [],
                               api_key="sk-not-real", base_url="http://127.0.0.1:9/v1",
                               model="whatever")
    finally:
        qa._log_qa_error = _orig_log
    check(bool("调用失败" in note), "调用失败：如实说是失败，不静默", note)
    check(bool(body and "Learning Rate" in body), "调用失败：仍然给出证据条目")

    # ---- 2026-10-31：「控制台地址当成 API」的误诊防线（凭据 cache/qa_errors.log）----
    # 实锤：Base URL 填成 https://platform.deepseek.com/ 时，WAF 回 HTML + 429，
    # SDK 包成 RateLimitError —— 旧代码退避重试两次（白等 9s）再提示「云端在限流，
    # 钱和 key 都没问题」，两条都是误诊。现在：不重试 + 直说地址错。
    from paper2lab.llm import endpoint_url_hint, looks_non_api_reply
    check(bool("api.deepseek.com" in endpoint_url_hint("https://platform.deepseek.com/")),
          "控制台地址被认出并给出正确 API 地址")
    check(bool("api.deepseek.com" in endpoint_url_hint("https://platform.deepseek.com")),
          "控制台地址（无尾斜杠）同样被认出")
    check(bool(endpoint_url_hint("https://api.deepseek.com") == ""), "正常 API 地址不误报")
    check(bool(endpoint_url_hint("") == ""), "空地址不误报")
    check(bool(looks_non_api_reply(Exception("429 <!DOCTYPE html><title>Request Blocked</title>"))),
          "HTML 拦截页被认出「对端不是 API」")
    check(bool(not looks_non_api_reply(Exception("429 too many requests"))),
          "真 429（JSON 错误体）不误判成非 API")

    _calls = {"n": 0}
    _orig_call = qa._call_deepseek

    def _boom(*a, **kw):
        _calls["n"] += 1
        raise Exception("429 Error - Request Blocked <!DOCTYPE html><html>")

    qa._call_deepseek, qa._log_qa_error = _boom, (lambda *a, **kw: None)
    try:
        body, note = qa.answer(audit, "学习率为什么对不上", [],
                               api_key="sk-x", base_url="https://platform.deepseek.com/",
                               model="deepseek-chat")
    finally:
        qa._call_deepseek, qa._log_qa_error = _orig_call, _orig_log
    check(bool(_calls["n"] == 1), f"控制台地址不重试（白等 9s），实调 {_calls['n']} 次")
    check(bool("api.deepseek.com" in note), "说明行直给正确 API 地址", note[:70])
    # 误诊词是「云端在限流」这个判断句；提示里的「这不是限流」是**澄清**，不许拦
    check(bool("云端在限流" not in note), "不再误诊成「云端在限流」")
    check(bool("调用失败" in note), "仍然如实说是失败，不静默")

    # 真 429（正常 API 地址 + JSON 错误体）仍走退避重试 —— 防线不能把对的也拦了
    _calls2 = {"n": 0}

    def _boom2(*a, **kw):
        _calls2["n"] += 1
        raise Exception("429 too many requests")

    qa._call_deepseek, qa._log_qa_error = _boom2, (lambda *a, **kw: None)
    _orig_waits = qa._RETRY_WAITS
    qa._RETRY_WAITS = (0.01, 0.01)      # 测试不等真秒数，口径不变
    try:
        _, note2 = qa.answer(audit, "学习率为什么对不上", [],
                             api_key="sk-x", base_url="https://api.deepseek.com",
                             model="deepseek-chat")
    finally:
        qa._call_deepseek, qa._log_qa_error, qa._RETRY_WAITS = \
            _orig_call, _orig_log, _orig_waits
    check(bool(_calls2["n"] == 3), f"真 429 仍重试 2 次（实调 {_calls2['n']} 次）")
    check(bool("限流" in note2), "真 429 的提示仍是「云端在限流」", note2[:70])

    hist = [{"role": "user", "content": f"问题{i}"} if i % 2 == 0
            else {"role": "assistant", "content": f"回答{i}"} for i in range(20)]
    prompt = qa.build_prompt(audit, "最后一个问题", hist)
    check(bool("问题18" in prompt and "问题0" not in prompt), "多轮只带最近 8 条，不无限长")
    # 2026-10-30 用户定版：从「只许复述事实」放开到「允许解释 + 允许给方案」，
    # 但必须保留两条：数字/出处不许编、推断必须显式标注。这三条断言就是那个口径。
    check(bool("允许你做成因解释、给修复方案" in prompt), "提示词放开「解释成因 + 给修复方案」")
    check(bool("最可能 / 我判断" in prompt), "提示词要求推断显式标注（不能写成既成事实）")
    check(bool("「没有」只允许用在两种情况" in prompt),
          "「没有」只留两种情形（不再拿「没覆盖」回避可推断的问题）")
    check(bool("不许估、不许编" in prompt), "数字与出处仍然不许编（死线没松）")
    # 2026-10-30 第二次补充：材料内给推断、材料外给常识，两者必须能分家
    check(bool("两样东西" in prompt and "领域知识" in prompt),
          "总纲：默认同时用「本材料 + 自身领域知识」回答")
    check(bool("材料是**边界**" in prompt), "写明材料是边界（通用知识不得覆盖/改写结论）")
    check(bool(qa.GENERAL_MARK in prompt), "超出材料的内容必须冠以〔通用知识〕")
    check(bool("最后一个问题" in prompt), "这一轮的问题在提示词末尾")
    check(bool(len(prompt) < 12000), "上下文有上限，不会把模型撑爆", f"{len(prompt)} 字符")


# ================================================================ E 只读保证
def part_readonly() -> None:
    print("\n===== E 只读保证（解释层不许动判定） =====")
    audit = _fake_audit()
    before = [(f.param_key, f.status, f.risk) for f in audit.findings]
    ready_before = dict(audit.readiness)
    qa.answer(audit, "Learning Rate 为什么对不上", [], api_key="")
    qa.focus(audit, "复现先改哪")
    qa.top_hits(audit, "seed")
    after = [(f.param_key, f.status, f.risk) for f in audit.findings]
    check(bool(before == after), "问答前后 findings 的判定一个没变")
    check(bool(ready_before == audit.readiness), "问答前后 readiness 一个没变")


# ================================================================ F 思考过程
def part_thinking() -> None:
    """书生 intern-s2 会把推理写进正文 —— 不许原样丢给用户（2026-10-30 实测）。"""
    print("\n===== F 思考过程（别把模型的草稿当答案给用户） =====")
    raw = ROOT / ".trash_tmp" / "internlm_raw.txt"
    if raw.exists():
        body, cut = qa.strip_thinking(raw.read_text(encoding="utf-8", errors="replace"))
        check(bool(cut), "能检出「思考过程」痕迹")
        check(body == "", "整段都是推理 → 不给用户看（退回证据检索），不给半截草稿")

    good = "论文学习率 1e-4（Page 3），代码 0.001（train.py:11），差 10 倍。"
    body, cut = qa.strip_thinking(good)
    check(bool(body == good and not cut), "没有思考痕迹的正常答案**原样**返回（一个字不动）")

    anchored = ("Thinking Process: 我先看看数据。\n\n最终答案：论文 1e-4，代码 0.001，差 10 倍。"
                "依据：Page 3 · 4. Training Setup 与 train.py:11，两侧判定为不一致。"
                "复现时把 train.py 第 11 行的默认值改成 1e-4 即可与论文对齐；"
                "改完重跑一轮，逐条对表里这一条就会从「不一致」变成「一致」。")
    body, cut = qa.strip_thinking(anchored)
    check(bool(cut and body.startswith("最终答案")), "有「最终答案」强锚点 → 从锚点开始给")

    weak = ("Thinking Process: 用户问「回答：」是什么意思。\n\n" + "草稿" * 40)
    body, cut = qa.strip_thinking(weak)
    check(bool(cut and body == ""), "只有「回答：」这种弱词不算锚点（会切在草稿中段）")

    # 源码守卫：这几条是把上面那个坑堵住的具体手段，改回去就会重现「没有答案」
    src = (ROOT / "paper2lab" / "qa.py").read_text(encoding="utf-8")
    check(bool("不要输出思考过程" in src), "提示词里明确禁止输出思考过程")
    # 2026-10-30 改口径：预算不再写死在 qa.py，统一由 llm.qa_budget() 按端点定。
    #   守两件事：① qa.py 必须真的去取预算（否则又回退成写死）；
    #            ② 两档数值本身（远程 1800/75s、本机 1000/360s）不许被改小。
    lsrc = (ROOT / "paper2lab" / "llm.py").read_text(encoding="utf-8")
    check(bool("qa_budget(" in src), "问答预算由 qa_budget() 统一派生（不在调用处写死）")
    check(bool("return 1800, 75.0" in lsrc), "远程档 1800 token / 75s（四段式写得开、慢回答不误杀）")
    check(bool("return 1000, 360.0" in lsrc), "本机档 1000 token / 360s（本地约 10 token/s，1800 必然超时）")
    check(bool("is_local_endpoint(" in lsrc), "本机端点由地址自动识别（不靠模型名猜）")

    # 「问为什么 / 问怎么改」必须真的补上能动手的素材（2026-10-30 加）
    fake = _fake_audit()
    fx = qa.focus(fake, "这个参数为什么对不上，我该怎么改？")
    check(bool("可动手的对象" in fx), "问原因/修法：补上「可动手的对象」这一段")
    check(bool("论文 " in fx and "代码 " in fx), "修法素材给出两侧原值（改成什么有参照）")
    check(bool("可动手的对象" in qa.build_prompt(fake, "怎么解决")), "修法素材进得了提示词")

    # 通用知识切分：放开「补常识」之后，数字校验要靠它避免误伤常识里的数字
    a1, g1 = qa.split_general(
        "论文写 1e-4（Page 3）。〔通用知识〕这类任务常用 1e-3 到 1e-2。改完重跑再看。")
    check(bool("1e-4" in a1 and "1e-3" not in a1),
          "标注〔通用知识〕的句子被切出去（不参与材料数字校验）")
    check(bool("1e-3" in g1), "切出去的正是那条通用知识")
    a2, g2 = qa.split_general("没有标注的回答，整段都属于材料侧。")
    check(bool(g2 == "" and a2 == "没有标注的回答，整段都属于材料侧。"),
          "没有标注：整段按材料对待（不给逃逸口）")


def main() -> int:
    print("问答窗口 · 口径体检（不联网，不花钱）")
    part_empty()
    part_facts()
    part_focus()
    part_hits()
    part_answer()
    part_readonly()
    part_thinking()

    bad = [c for c in CHECKS if not c[0]]
    print(f"\n共 {len(CHECKS)} 项：通过 {len(CHECKS) - len(bad)} 项，未通过 {len(bad)} 项")
    if bad:
        print("未通过清单：")
        for _, label, detail in bad:
            print(f"  - {label}" + (f"（{detail}）" if detail else ""))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
