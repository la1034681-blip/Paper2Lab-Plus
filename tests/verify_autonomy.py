"""自主回头（L3 回边）验证 —— 2026-10-25。

守的契约:
  A 默认关: 没配 Key / 强制离线时**一次都不问模型**, 直接按规则收尾（零成本）
  B 模型能定: 说 stop 就收尾; 说 retry 且合法, 就真的把 next_step 改成 compute
  C 闸门齐全: 每一次判定都留七道闸门的通过情况（不是"看不见就等于没做"）
  D G1 轮数: 到顶强制收尾, 不管模型多想继续
  E G2 有事可做: 没有未收敛项时, "回头"无处可回 → 拒
  F G3 防幻觉: 指名的参数必须真的在未收敛清单里
  G G4 目标合法: assume.key 必须落在 focus 内; side 只能是 paper/code
  H G5 值可解析: 非正数 / NaN / 文字 → 拒
  I G6 不空转: 值等于现有值或已试过的值 → 拒（换了等于没换）
  J 被拒不丢: 模型的原话（llm_action / llm_why）保留, rejected=True
  K 结果由程序算: record_outcome 回填的收敛数来自 loop 重算, 没多收敛也照实写
  L 端到端: `_autonomy_next` 关着时返回 selfcheck（与改造前逐字一致）;
     开着且模型要回头时返回 compute, 并记账 auto_round / auto_assume
  M 人话总结: summary() 必须说清「回头几次、几次真换到新结论、几次被拦」

为什么要有这一项:
  「给模型最大权限」最坏的结局是**它说什么就是什么** —— 说"再算一遍"就一直算,
  指名一个不存在的参数, 或者把两侧现有值当"新值"提上来空转。
  所以这里守的不是"能不能回头", 而是"每一次回头都必须换到新信息, 换不到就停"。

用法: .venv\\Scripts\\python.exe tests/verify_autonomy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab import autonomy as A                                        # noqa: E402
from paper2lab import graph as G                                           # noqa: E402

CHECKS = 0
FAILURES = []


def check(ok, label: str) -> None:
    global CHECKS
    CHECKS += 1
    assert isinstance(ok, bool), f"断言返回值必须是 bool: {label} → {type(ok)}"
    if ok:
        print(f"  OK  {label}")
    else:
        print(f"  ** FAIL **  {label}")
        FAILURES.append(label)


def sec(name: str) -> None:
    print(f"\n== {name} ==")


# ---------------------------------------------------------------- 测试数据

def _lr(**kw) -> dict:
    """一份「有 1 条推不动」的闭环结果 —— 反复用，避免每个用例各写一份漂移。"""
    d = {
        "resolved": 1,
        "max_rounds": 3,
        "unresolved": [{
            "param_key": "learning_rate", "display": "Learning Rate",
            "paper": "1e-4", "code": "0.03",
            "reason": "根因候选里没有与论文一致的取值",
        }],
        "items": [{"param_key": "learning_rate", "attempt": "1e-4", "side": "code"}],
        "rounds": [{"round": 1, "improved": 0}],
    }
    d.update(kw)
    return d


def _ask_retry(value="0.05", key="learning_rate", side="code", focus=None):
    return lambda _p: {
        "action": "retry",
        "focus": list(focus if focus is not None else [key]),
        "assume": {"key": key, "side": side, "value": value},
        "why": "换一个没试过的值再算",
    }


def _ask_stop():
    return lambda _p: {"action": "stop", "why": "证据已经够了"}


# ---------------------------------------------------------------- A 默认关

def test_default_off():
    sec("A. 默认关（零成本）")
    d = A.decide(_lr())                      # 没 key、没注入
    check(d.action == "stop" and d.target == "selfcheck",
          "A1 没配 Key → 直接收尾，不留半截决定")
    check(d.source == "rule" and d.llm_action == "",
          "A2 来源标 rule、模型原话为空 —— 不假装它判过")
    check(d.gates == [], "A3 一次都没问模型 → 闸门清单为空（没调用就没有判定）")


# ---------------------------------------------------------------- B 模型能定

def test_model_decides():
    sec("B. 模型真的能定（自主权是实的）")
    d = A.decide(_lr(), ask=_ask_stop())
    check(d.action == "stop" and d.source == "llm" and d.llm_action == "stop",
          "B1 模型说收尾 → 采纳，且留痕说这是它定的")

    d = A.decide(_lr(), api_key="x", ask=_ask_retry())
    check(d.action == "retry" and d.target == "compute",
          "B2 模型说再算一遍且合法 → 真的回头（target=compute）")
    check(d.assume == {"key": "learning_rate", "side": "code", "value": "0.05"},
          "B3 回头带的具体假设值原样记下（回边靠它才能算出不同结果）")
    check(d.before_resolved == 1, "B4 回头前先记下收敛数，供事后比「这一遍值不值」")
    check(d.rejected is False, "B5 采纳的决定不算被拒")


# ---------------------------------------------------------------- C 闸门齐全

def test_gates_complete():
    sec("C. 七道闸门都要留痕")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry())
    names = [g["name"] for g in d.gates]
    check(len(names) == 7, f"C1 合法 retry 也要过满 7 道（实际 {len(names)} 道）")
    check(all(g["ok"] for g in d.gates), "C2 合法的决定七道全过")
    check(names[0].startswith("G1") and names[-1].startswith("G7"),
          "C3 闸门按 G1→G7 顺序排（报告要能按号引用）")


# ---------------------------------------------------------------- D~I 逐道闸门

def test_gate_g1_rounds():
    sec("D. G1 轮数到顶就停")
    d = A.decide(_lr(), round_no=3, max_rounds=2, api_key="x", ask=_ask_retry())
    check(d.action == "stop" and d.source == "gate",
          "D1 第 3 次（上限 2）→ 强制收尾，不听模型")
    check(d.llm_action == "retry" and d.rejected,
          "D2 它想继续这件事要留着 —— 报告里写「想继续，没让」")


def test_gate_g2_nothing():
    sec("E. G2 没有推不动的条目就不许回头")
    d = A.decide(_lr(unresolved=[], resolved=4), api_key="x", ask=_ask_retry())
    check(d.action == "stop", "E1 全收敛了还说回头 → 拒（回头无事可做）")
    check(any(g["name"].startswith("G2") and not g["ok"] for g in d.gates),
          "E2 拒的理由落在 G2 上")


def test_gate_g3_hallucination():
    sec("F. G3 不许编造参数名")
    d = A.decide(_lr(), api_key="x",
                 ask=_ask_retry(key="batch_size", focus=["batch_size"]))
    check(d.action == "stop", "F1 指名一个不在清单里的参数 → 拒")
    check(any(g["name"].startswith("G3") and not g["ok"] for g in d.gates),
          "F2 拒的理由落在 G3 上")


def test_gate_g4_target():
    sec("G. G4 回头目标必须合法")
    d = A.decide(_lr(), api_key="x",
                 ask=_ask_retry(key="learning_rate", focus=["batch_size"]))
    check(d.action == "stop", "G1 assume.key 不在 focus 内 → 拒")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry(side="result"))
    check(d.action == "stop", "G2 side='result'（只能碰论文/代码两侧）→ 拒")
    check(any(g["name"].startswith("G4") and not g["ok"] for g in d.gates),
          "G3 拒的理由落在 G4 上")


def test_gate_g5_value():
    sec("H. G5 值必须是能用的正数")
    for bad in ("约0.05", "见论文", -0.05, 0, "nan"):
        d = A.decide(_lr(), api_key="x", ask=_ask_retry(value=bad))
        check(d.action == "stop", f"H 值 {bad!r} → 拒")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry(value="1e-4"))
    check(d.action == "stop", "H5 1e-4 与论文侧同值 → 归到 G6（空转）而不是 G5")


def test_gate_g6_noop():
    sec("I. G6 换不出新信息的一律拒（防空转）")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry(value="0.03"))
    check(d.action == "stop", "I1 值等于代码侧现有值 → 拒")
    check(any(g["name"].startswith("G6") and not g["ok"] for g in d.gates),
          "I2 拒的理由落在 G6 上")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry(value="1e-04"))
    check(d.action == "stop", "I3 写法不同但数值等于已试过的值（1e-04 vs 1e-4）→ 照样拒")


# ---------------------------------------------------------------- J 被拒不丢

def test_rejected_kept():
    sec("J. 被拦下的决定照样留痕")
    d = A.decide(_lr(), api_key="x",
                 ask=lambda _p: {"action": "retry", "focus": ["不存在的键"],
                                 "assume": {"key": "不存在的键", "side": "code",
                                            "value": "9"},
                                 "why": "我觉得该再试一次"})
    check(d.action == "stop" and d.llm_action == "retry",
          "J1 动作被改判，但模型的原意留着")
    check(d.llm_why == "我觉得该再试一次", "J2 理由原话留着（报告要写「它想…，程序没让」）")
    check(d.source == "gate", "J3 来源标 gate —— 一眼看出这是被程序按下的")
    check(bool(d.note), "J4 有人话结论，不是空 string")


# ---------------------------------------------------------------- K 结果回填

def test_outcome():
    sec("K. 回头之后「值不值」由程序算")
    d = A.decide(_lr(), api_key="x", ask=_ask_retry())
    rows = [d.to_dict()]
    msg = A.record_outcome(rows, 2)
    check("多收敛 1 条" in msg, f"K1 收敛数涨了要如实说涨了几条（{msg}）")
    check(rows[0]["after_resolved"] == 2 and rows[0]["before_resolved"] == 1,
          "K2 前后两个数都留底")

    d2 = A.decide(_lr(), api_key="x", ask=_ask_retry())
    rows2 = [d2.to_dict()]
    msg2 = A.record_outcome(rows2, 1)
    check("没换来新证据" in msg2, f"K3 白跑一遍也要如实写，不许粉饰（{msg2}）")

    check(A.record_outcome([{"action": "stop"}], 3) == "",
          "K4 没回头就没有回填（不产生空话）")


def test_summary():
    sec("L. 人话总结")
    d1 = A.decide(_lr(), api_key="x", ask=_ask_retry()).to_dict()
    d1["after_resolved"] = 2
    s = A.summary([d1])
    check("回头 1 次" in s and "1 次换到了新结论" in s, f"L1 说清几次回头、几次有用（{s}）")

    d2 = A.decide(_lr(), api_key="x", ask=_ask_stop()).to_dict()
    s2 = A.summary([d2])
    check("没有回头" in s2, f"L2 模型说收尾时说没回头，不硬凑（{s2}）")

    d3 = A.decide(_lr(unresolved=[], resolved=3), api_key="x",
                  ask=_ask_retry()).to_dict()
    s3 = A.summary([d3])
    check("被程序挡下" in s3, f"L3 想回头但无事可做时，如实写被挡下（{s3}）")


# ---------------------------------------------------------------- M 端到端（图）

def test_graph_wire():
    sec("M. 图接线：关着不变、开着能回头")
    st = {"loop_result": _lr(), "auto_loop": False, "api_key": "", "errors": []}
    # 收尾节点是 `llm_analyze`（归因）而非 `selfcheck`：归因与自检的先后
    # 必须与经典版一致，否则多智能体档会漏跑归因（见 graph.node_loop 注释）。
    check(G._autonomy_next(st) == "llm_analyze",
          "M1 auto_loop 关着 → 收尾走 llm_analyze（与经典版同序，selfcheck 在其后）")
    check("auto_decisions" not in st or st["auto_decisions"] == [],
          "M2 关着时一条决策都不记（零成本、零副作用）")

    st = {"loop_result": _lr(), "auto_loop": True, "api_key": "",
          "auto_round": 0, "auto_decisions": [], "errors": []}
    nxt = G._autonomy_next(st)
    check(nxt == "llm_analyze" and len(st["auto_decisions"]) == 1,
          "M3 开着但没 Key → 收尾走 llm_analyze，且如实记一条「模型未启用」")
    check(st["auto_decisions"][0]["source"] == "rule",
          "M4 那条记录的来源是 rule，不冒充模型判断")

    # 注入模型意见：让它真的回头
    st = {"loop_result": _lr(), "auto_loop": True, "api_key": "x",
          "auto_round": 0, "auto_decisions": [], "auto_assume": {}, "errors": []}
    _orig = A._ask
    try:
        A._ask = lambda *a, **k: (_ask_retry()(""), "测试注入")
        nxt = G._autonomy_next(st)
        check(nxt == "compute", "M5 模型要回头 → next_step 真的指回 compute（回边形了）")
        check(st["auto_round"] == 1, "M6 回头次数记账 +1（闸门 G1 靠它）")
        check(st["auto_assume"].get("value") == "0.05",
              "M7 假设值传下去了 —— compute 会用它换输入重算，回边才有新结果")

        st["loop_result"] = _lr(resolved=2)
        A._ask = lambda *a, **k: ({"action": "stop", "why": "够了"}, "测试注入")
        check(G._autonomy_next(st) == "llm_analyze",
              "M8 模型改口收尾 → 交回 llm_analyze，正常往 selfcheck 走")
        check(st["auto_decisions"][0]["after_resolved"] == 2,
              "M9 上一轮回头的结果被回填（2 条收敛写进第一条决策）")
        check(st["auto_assume"] == {}, "M10 收尾时清掉假设值，不许污染后续")
    finally:
        A._ask = _orig


def test_contract():
    sec("N. 契约：默认关 + 硬上限")
    check(A.MAX_AUTO_ROUNDS == 2, "N1 回头硬上限是 2（常量在模块级，改不靠印象）")
    check(A.ACTIONS == ("retry", "stop"), "N2 动作是封闭集合，越界丢弃")
    check(A.SIDES == ("paper", "code"), "N3 只能动论文侧或代码侧（代码只改内存副本）")
    import inspect
    sig = inspect.signature(G.run_audit_multagent)
    check(sig.parameters["auto_loop"].default is False,
          "N4 图入口的 auto_loop 默认 False —— 不传就是老行为")
    d = A.decide(_lr(), api_key="x", ask=lambda _p: {"action": "乱写", "why": "x"})
    check(d.action == "stop" and d.source == "rule",
          "N5 动作不在集合里 → 当成没判，按规则收尾（不猜最近的一个）")


def test_auto_mode_wiring():
    sec("O. 自主模式总闸：六处开关必须一路通到后端（断一处 = 界面上点不到）")
    import inspect
    import re
    from paper2lab import pipeline as P
    from paper2lab import toolbox as T

    # O1/O2 两条后端入口都认这个参数。少一条就会出现「某条调用方式静默丢参数」，
    #      而且不会报错 —— 正是本轮查出来的那个真缺口。
    check("experiment_llm" in inspect.signature(P.run_audit).parameters,
          "O1 单链入口认 experiment_llm")
    check("experiment_llm" in inspect.signature(G.run_audit_multagent).parameters,
          "O2 图入口认 experiment_llm")

    # O3 工单真的收到它 —— 支持不等于接通。这条是本轮的核心修复，必须钉住。
    check("experiment_llm=" in inspect.getsource(G._job_from_state),
          "O3 工单构造把开关传进去了（不传 = 代码支持但界面开不了）")

    # O4 实验站读的是工单字段，不是别处的全局变量（否则开关形同虚设）
    check("job.experiment_llm" in inspect.getsource(T._experiment_llm),
          "O4 实验站读的是工单上的开关")

    # O5 默认关：不传参就是 False，老行为逐字不变
    check(inspect.signature(P.run_audit).parameters["experiment_llm"].default is False,
          "O5 默认 False —— 不传参时一行模型都不调")

    # O6/O7 2026-10-27 用户定版：那一排开关**从页面上整块删除**了，六项恒开。
    # 原来的断言是「总闸按 key 带开六处分项」—— 开关都没了，这条不再适用；
    # 换成守新口径：六处必须是模块级常量 True（不是读 env、不是读 session）。
    app_src = (ROOT / "app.py").read_text(encoding="utf-8")
    _six = ("use_multagent", "use_closeloop", "use_auto_loop",
            "use_role_llm", "judge_llm", "use_exp_llm")
    _off = [n for n in _six
            if not re.search(rf"^{n} = _AUTO_ON$", app_src, re.M)]
    check(not _off, f"O6 六处分项恒为开（没接上的: {_off}）")
    _opt = app_src.split("# ---- 选项区")[-1].split("# ---- 状态提示")[0]
    check("st.toggle(" not in _opt,
          "O7 选项区一个开关都没有（含旧的「使用内置演示数据」）")

    # O8 分项自己的值要真的传给后端 —— 否则界面上是"开"，后端仍是关
    check("experiment_llm=bool(use_exp_llm)" in app_src,
          "O8 界面把实验站开关传给了后端（不传就是假开关）")
    check('_mk["role_llm"] = bool(use_role_llm)' in app_src,
          "O9 对撞区角色开关也真的传了")

    # O10/O11 上一站产物必须带进工单 —— 2026-10-26 查出的真缺口：
    #   ② 实验站每次拿到的都是空工单，`design_experiments([])` 返回 0 张，
    #   而节点状态仍是 done（进度条走完、报告有章节、内容是零）。
    #   这类"跑了但什么都没算"只能靠接线守卫拦住。
    _jfs = inspect.getsource(G._job_from_state)
    check("all_hypotheses=" in _jfs and "hypotheses=" in _jfs,
          "O10 工单带上了 ① 站的假设卡（不带 = ② 站空转、还显示成功）")
    check('"all_hypotheses": []' in inspect.getsource(G.run_audit_multagent),
          "O11 状态里有 all_hypotheses 键（② 站要全部卡，不是界面截断那份）")


def main():
    print("=" * 62)
    print("自主回头（L3 回边）—— 验证")
    print("=" * 62)
    test_default_off()
    test_model_decides()
    test_gates_complete()
    test_gate_g1_rounds()
    test_gate_g2_nothing()
    test_gate_g3_hallucination()
    test_gate_g4_target()
    test_gate_g5_value()
    test_gate_g6_noop()
    test_rejected_kept()
    test_outcome()
    test_summary()
    test_graph_wire()
    test_contract()
    test_auto_mode_wiring()

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"未通过 {len(FAILURES)} / {CHECKS} 项：")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 62)
        sys.exit(1)
    print(f"全部通过：{CHECKS} 项")
    print("=" * 62)


if __name__ == "__main__":
    main()
