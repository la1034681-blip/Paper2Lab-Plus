"""对撞区角色接模型（阶段四补强）——两层验证。

口径（2026-10-14 用户定）:
  对撞区（解析员 / 代码验证员 / 数值验证员）接模型，先拿已知分歧
  scheduler / seed 试；规则版永远留着当兜底。

这个改造的生死线只有一条: **隔离不能破**。
  解析员只喂论文侧材料、代码验证员只喂代码侧材料，另一侧一个字符都不给。
  所以本脚本第 1 组断言的重点不是「模型调没调通」，而是
  **「另一侧的内容绝不出现在本侧的 prompt 里」**。

第 1 层 · 纯离线（不联网、不花钱）:
  · 隔离: 两个角色的 prompt 里都搜不到「对面那一侧」的特征串；
  · 规范化: 模型给的标记被写进视图（normalized / normalized_by）；
  · 省调用: 数值型参数不送模型（1e-4 与 0.0001 的关系程序算得更准）；
  · 防呆: 超长标记 / 带换行 / 不合规一律丢弃；
  · 失败回落: 拿不到 JSON -> 视图原样返回 + note 如实说明「回退规则版」；
  · 对撞: 两侧标记等价 -> 判一致且 note 写明「经规范化后等价」；
         只有一侧有标记 -> **不**判一致（不许单方面定案）；
  · 契约: graph 的 run_audit_multagent 确实接了 role_llm（默认 False）、
         app.py 确实接了这个开关（防「加了开关没接线」）。

第 2 层 · 真机（可选，--offline 跳过）:
  用 .env 里的模型 API 真跑一次 canonicalize。没有 key 就**如实跳过**，
  不假装跑过。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS = 0
FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    # 防呆（这个项目踩过）: check(ok, label) 传反会让 ok 拿到非空字符串，
    # 于是**永远打印「通过」**，用退出码判成败会静默放行真问题。
    assert isinstance(ok, bool), f"check() 第 1 个参数必须是 bool，收到 {type(ok)}"
    assert isinstance(label, str), f"check() 第 2 个参数必须是 str，收到 {type(label)}"
    global CHECKS
    CHECKS += 1
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}", flush=True)
    if not ok:
        FAILURES.append(label)


def skip(label: str) -> None:
    print(f"  [SKIP] {label}", flush=True)


# ---------------------------------------------------------------- 测试替身

class _Ev:
    def __init__(self, raw, loc, snip):
        self.raw_value = raw
        self.location = loc
        self.snippet = snip
        self.normalized = None


class _P:
    def __init__(self):
        self.params = {"scheduler": _Ev("余弦退火调度", "Page 4", "采用 PAPERONLY 余弦退火")}
        self.params_all = {}
        self.image_params = {}


class _C:
    def __init__(self):
        self.params = {"scheduler": [_Ev("CosineAnnealingLR", "train.py:42",
                                         "scheduler = CODEONLY_CosineAnnealingLR(opt)")]}
        self.aux_only_keys = set()


class _J:
    def __init__(self, p, c):
        self.paper = p
        self.code = c
        self.paper_view = {}
        self.code_view = {}


class _Fake:
    """假模型: 记下 prompt，按需返回。"""

    def __init__(self, obj=None, note="AI（role_x）· 真实调用", calls=None):
        self.obj = obj
        self.note = note
        self.prompts: list[str] = []
        self.calls = calls

    def __call__(self, prompt, api_key, **kw):
        self.prompts.append(prompt)
        if self.calls is not None:
            self.calls.append(prompt)
        return self.obj, self.note


def _patch(monkey) -> None:
    import paper2lab.llm as llm
    llm.ask_json = monkey      # canonicalize 在函数内 from .llm import ask_json


def _restore() -> None:
    import importlib
    import paper2lab.llm as llm
    importlib.reload(llm)


# ================================================================ 第 1 组：隔离

def group_isolation() -> None:
    print("\n=== 1. 隔离（生死线：另一侧的内容不得进入本侧 prompt）===", flush=True)
    from paper2lab.agent_roles import Analyst, CodeVerifier, RoleLLM

    fake = _Fake({"normalized": {"scheduler": "cosine"}})
    _patch(fake)
    try:
        cfg = RoleLLM(api_key="fake-key")
        job = _J(_P(), _C())
        pv = Analyst(cfg)(job)
        cv = CodeVerifier(cfg)(job)
    finally:
        _restore()

    joined = "\n\n".join(fake.prompts)
    paper_prompt = next((p for p in fake.prompts if "PAPERONLY" in p), "")
    code_prompt = next((p for p in fake.prompts if "CODEONLY" in p), "")

    check(len(fake.prompts) == 2, f"两个角色各调一次模型: {len(fake.prompts)} 次")
    check(bool(paper_prompt), "解析员的 prompt 里能搜到论文侧特征串")
    check(bool(code_prompt), "代码验证员的 prompt 里能搜到代码侧特征串")
    check("CODEONLY" not in paper_prompt,
          "★隔离: 论文侧 prompt 里搜不到代码侧内容（CODEONLY）")
    check("PAPERONLY" not in code_prompt,
          "★隔离: 代码侧 prompt 里搜不到论文侧内容（PAPERONLY）")
    check("只能看到" in joined, "prompt 里明确告知「只能看到本侧」")
    check(pv.get("scheduler", {}).get("normalized") == "cosine",
          f"论文侧拿到规范化标记: {pv.get('scheduler', {}).get('normalized')}")
    check(cv.get("scheduler", {}).get("normalized") == "cosine",
          f"代码侧拿到规范化标记: {cv.get('scheduler', {}).get('normalized')}")
    check(pv["scheduler"].get("normalized_by") == "llm", "标记标注了来源=llm")
    check(pv["scheduler"]["value"] == "余弦退火调度", "原值仍保留（可审计）")


# ================================================================ 第 2 组：省调用 / 防呆 / 回落

def group_guard() -> None:
    print("\n=== 2. 省调用 · 防呆 · 失败回落 ===", flush=True)
    from paper2lab.agent_roles import Analyst, RoleLLM

    class _PNum:
        params = {"learning_rate": _Ev("1e-4", "Page 3", "lr = 1e-4")}
        params_all = {}
        image_params = {}

    calls: list[str] = []
    _patch(_Fake({"normalized": {}}, calls=calls))
    try:
        a = Analyst(RoleLLM(api_key="k"))
        pv = a(_J(_PNum(), _C()))
        note = a.last_note
    finally:
        _restore()
    check(len(calls) == 0, f"数值型参数不送模型（实际调用 {len(calls)} 次）")
    check(pv["learning_rate"]["value"] == "1e-4", "数值型参数仍是规则版抽到的值")
    check("未调用模型" in note, f"如实说明「未调用模型」: {note}")

    # 超长 / 带换行的标记必须丢弃
    _patch(_Fake({"normalized": {"scheduler": "x" * 60 + "\nbad"}}))
    try:
        v = Analyst(RoleLLM(api_key="k"))(_J(_P(), _C()))
    finally:
        _restore()
    check("normalized" not in v.get("scheduler", {}), "不合规标记被丢弃（不写进视图）")

    # 调用失败 -> 原样返回 + 如实说明
    _patch(_Fake(None, "AI（role_analyst）· 调用失败（APIError），回退确定性路径"))
    try:
        b = Analyst(RoleLLM(api_key="k"))
        v2 = b(_J(_P(), _C()))
    finally:
        _restore()
    check(v2["scheduler"]["value"] == "余弦退火调度", "失败时视图原样返回（内容不丢）")
    check("回退规则版" in b.last_note, f"失败时如实说明: {b.last_note[:40]}")

    # 没 key -> 角色不 ready -> 一次都不调
    calls2: list[str] = []
    _patch(_Fake({"normalized": {"scheduler": "cosine"}}, calls=calls2))
    try:
        cfg = RoleLLM(api_key="")
        c = Analyst(cfg)
        v3 = c(_J(_P(), _C()))
    finally:
        _restore()
    check(len(calls2) == 0, "没配 key 时一次都不调模型")
    check(cfg.ready is False, "RoleLLM.ready 在没 key 时为 False")
    check(v3 == {"scheduler": {"value": "余弦退火调度", "source": "Page 4",
                               "doc": "采用 PAPERONLY 余弦退火", "confidence": "已确认"}},
          "没 key 时视图与规则版逐字段一致")


# ================================================================ 第 3 组：对撞用标记

def group_collide() -> None:
    print("\n=== 3. 对撞：标记只在「两侧都有」时才参与判定 ===", flush=True)
    import paper2lab.collider as col

    pv = {"scheduler": {"value": "余弦退火", "source": "P4", "doc": "",
                        "confidence": "已确认", "normalized": "cosine"}}
    cv = {"scheduler": {"value": "CosineAnnealingLR", "source": "train.py:42", "doc": "",
                        "confidence": "已确认", "normalized": "cosine"}}
    r = col.collide(pv, cv)
    check(len(r) == 1, f"对撞出 1 条: {len(r)}")
    check(r[0].status == col.Status.CONSISTENT,
          f"两侧标记等价 -> 判一致: {r[0].status.value}")
    check("规范化" in r[0].note, f"note 写明经规范化后等价: {r[0].note}")
    check(r[0].paper == "余弦退火" and r[0].code == "CosineAnnealingLR",
          "原值仍如实保留（没有抹掉写法差异）")

    # 只有一侧有标记 -> 不许判一致
    cv2 = {"scheduler": {k: v for k, v in cv["scheduler"].items() if k != "normalized"}}
    r2 = col.collide(pv, cv2)
    check(r2[0].status == col.Status.INCONSISTENT,
          f"只有一侧有标记 -> 不判一致（不许单方面定案）: {r2[0].status.value}")

    # 标记不等价 -> 维持冲突，不能因为「有标记」就放宽
    cv3 = {"scheduler": {"value": "CosineAnnealingLR", "source": "train.py:42",
                         "doc": "", "confidence": "已确认", "normalized": "exponential"}}
    r3 = col.collide(pv, cv3)
    check(r3[0].status == col.Status.INCONSISTENT, "标记不等价 -> 维持冲突")

    # 规则版（两侧都没标记）行为不变
    pv4 = {"scheduler": {"value": "余弦退火", "source": "P4", "doc": "", "confidence": "已确认"}}
    cv4 = {"scheduler": {"value": "余弦退火", "source": "train.py:42", "doc": "",
                         "confidence": "已确认"}}
    r4 = col.collide(pv4, cv4)
    check(r4[0].status == col.Status.CONSISTENT, "规则版（无标记）行为不变")

    # 序列化回绕
    d = r[0].to_dict()
    check(d["paper_normalized"] == "cosine" and d["code_normalized"] == "cosine",
          f"to_dict 带标记: {d['paper_normalized']}/{d['code_normalized']}")
    check(col.Collision.from_dict(d).paper_normalized == "cosine", "from_dict 回绕标记")
    # 老结构（没有这两个键）必须还能反序列化 —— 向后兼容
    old = {k: v for k, v in d.items() if not k.endswith("_normalized")}
    check(col.Collision.from_dict(old).paper_normalized == "",
          "老结构缺这两个键时能正常反序列化（向后兼容）")


# ================================================================ 第 4 组：契约（防「没接线」）

def group_contract() -> None:
    print("\n=== 4. 契约：开关确实接到了图与界面 ===", flush=True)
    import inspect

    from paper2lab.graph import run_audit_multagent
    sig = inspect.signature(run_audit_multagent)
    check("role_llm" in sig.parameters, "run_audit_multagent 有 role_llm 参数")
    check(sig.parameters["role_llm"].default is False,
          f"role_llm 默认 False（线上零成本）: {sig.parameters['role_llm'].default}")

    src = (ROOT / "app.py").read_text(encoding="utf-8")
    check("use_role_llm" in src, "app.py 里有 use_role_llm 开关")
    check('_mk["role_llm"]' in src, "app.py 把开关真的传给了多智能体版")
    shared = src.split("_audit_kwargs = dict(")[1].split(")")[0]
    check('"role_llm"' not in shared,
          "role_llm 没被塞进经典版共用的 _audit_kwargs（否则经典版会 TypeError）")

    gsrc = (ROOT / "paper2lab" / "graph.py").read_text(encoding="utf-8")
    check("role_llm_note" in gsrc, "graph.py 把「模型参没参与」写进留痕")

    asrc = (ROOT / "paper2lab" / "agent_roles.py").read_text(encoding="utf-8")
    check("只能看到" in asrc, "agent_roles.py 里写明隔离约束（只喂本侧材料）")


# ================================================================ 第 5 组：真机（可选）

def group_live() -> None:
    print("\n=== 5. 真机：真调一次模型做规范化 ===", flush=True)
    from paper2lab.envfile import get as env_get
    key = env_get("LLM_API_KEY") or env_get("DEEPSEEK_API_KEY")
    if not key:
        skip("未配置 LLM_API_KEY —— 真机层跳过（不假装跑过）")
        return
    from paper2lab.agent_roles import Analyst, RoleLLM

    cfg = RoleLLM(api_key=key, base_url=env_get("LLM_BASE_URL") or "",
                  model=env_get("LLM_MODEL") or "")
    a = Analyst(cfg)
    v = a(_J(_P(), _C()))
    ent = v.get("scheduler", {})
    print(f"    note: {a.last_note}")
    print(f"    论文侧 scheduler: 值={ent.get('value')} 标记={ent.get('normalized')}")
    check(bool(a.last_note), "真机跑完有留痕说明")
    check(isinstance(ent.get("value"), str) and bool(ent.get("value")),
          "真机返回的视图里原值仍在（不因调模型丢内容）")
    if ent.get("normalized"):
        check(isinstance(ent["normalized"], str) and len(ent["normalized"]) <= 40,
              f"真机标记合规: {ent['normalized']}")
    else:
        skip("模型这次没给标记（材料很短，属预期内的保守行为）")


def main() -> int:
    offline = "--offline" in sys.argv
    print("=" * 60)
    print("对撞区角色接模型 · 验证" + ("（仅离线层）" if offline else ""))
    print("=" * 60)

    group_isolation()
    group_guard()
    group_collide()
    group_contract()
    if not offline:
        group_live()

    print("\n" + "=" * 50)
    print(f"通过 {CHECKS - len(FAILURES)}/{CHECKS}")
    if FAILURES:
        print(f"失败 {len(FAILURES)}: {'；'.join(FAILURES)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
