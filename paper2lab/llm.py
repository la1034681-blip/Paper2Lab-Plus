"""LLM 归因层——DeepSeek API + 缓存兜底 + Mock 模式。

架构原则(评审定稿): "确定性问题交给程序验证, 复杂归因交给 AI"。
LLM 只负责对已确认的 Finding 生成风险解释与归因, 不参与判断。

可靠性三保险:
1. temperature=0 + 超时 + 重试
2. 结果按 prompt 哈希缓存到 cache/llm_cache.json, API 故障直接用缓存
3. 无 API key 或调用失败 -> Mock 模板生成, 演示永不中断
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from .models import Finding, Status
from .shards import DEFAULT_MAX_WORKERS, run_sharded

CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "llm_cache.json"

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_DEEPSEEK_MODEL = "deepseek-chat"

# ---- 通用「模型 API」解析（2026-09-30）--------------------------------------
# 原先页面只有一个 "DeepSeek API Key"，端点与模型名写死在 .env —— 用户换不了别的模型。
# 现在页面可直接填 Base URL / 模型名，解析顺序（**页面输入 > 通用名 > DeepSeek 名 > 默认**）：
#   key : 页面输入 → LLM_API_KEY    → DEEPSEEK_API_KEY
#   url : 页面输入 → LLM_BASE_URL   → DEEPSEEK_BASE_URL → 官方
#   name: 页面输入 → LLM_MODEL      → DEEPSEEK_MODEL    → deepseek-chat
# 保留 DEEPSEEK_* 是为了**向后兼容**：老配置（.env / Secrets）不改也照旧生效。

# 模型名里出现这些片段 = 该模型自带视觉能力（可读图）
_MULTIMODAL_HINTS = (
    "-vl", "vl-", "vision", "-4v", "4v-", "glm-4v", "qwen-vl", "qwen2-vl", "qwen2.5-vl",
    "gpt-4o", "gpt-4.1", "gpt-5", "o4-mini", "gemini", "claude-3", "claude-4",
    "claude-sonnet", "claude-opus", "internvl", "llava", "minicpm-v",
    "step-1v", "ernie-4", "yi-vl", "pixtral", "phi-3.5-vision", "llama-3.2-vision",
    # 书生 InternLM 系列（2026-10-11 接入）：S1 / S2 均自带视觉编码器、可读图；
    # intern-latest 当前指向 intern-s2-preview-397b，同样多模态。
    # 不写这两组会误判"模型不能读图"，导致读图通道被白白关掉。
    "intern-s2", "intern-s1", "intern-latest",
    # DeepSeek 多模态（2026-10-24 实测，端点 api.deepseek.com/v1）：
    # `deepseek-chat` 带 image_url 能正确识图（返回图上的文字，prompt_tokens 含图）；
    # 而 `deepseek-v4-pro` **不能** —— 它不报错，回 content="" 并在 reasoning_content
    # 里说 "Unsupported Image"。所以这里**只精确匹配 chat**，**绝不写成泛词 "deepseek"**，
    # 否则会把 v4-pro 误判成多模态，读图全静默失败。
    "deepseek-chat",
)


def modality_hit(model: str) -> str:
    """返回模型名里命中的那个**视觉标识片段**（没命中返回空串）。

    为什么要把"命中了哪个词"单独抛出来：页面上要如实写「依据：模型名含 `-vl`」。
    只报结论「已识别为多模态」是不够的 —— 用户改了模型名却是另一种结果时，
    看不到依据就只能猜程序是不是坏了。
    """
    m = (model or "").strip().lower()
    if not m:
        return ""
    for h in _MULTIMODAL_HINTS:
        if h in m:
            return h
    return ""


def looks_multimodal(model: str, base_url: str = "") -> bool:
    """按模型名**判**它是否自带读图能力（多模态）。

    ⚠️ 2026-10-30 起页面上**不再有勾选框**：识别结果直接用于判定，页面如实写出
    「识别成什么 + 依据是什么 + 能读什么 / 要注意什么」。要改判定就改模型名
    （或改这里的词表），不靠用户手动覆盖 —— 少一个开关，就少一处「勾了却没生效」。
    猜错的代价不对称 —— 猜成"不能读图"最多是少读图（并如实说明）；
    猜成"能读图"而实际不能，会导致读图调用全部失败。所以规则**偏保守**：
    只认明确的视觉模型命名特征，认不出就返回 False。
    """
    return bool(modality_hit(model))


def resolve_endpoint(api_key: str = "", base_url: str = "",
                     model: str = "") -> tuple[str, str, str]:
    """解析出本次调用真正使用的 (api_key, base_url, model)。"""
    from .envfile import get
    k = (api_key or "").strip() or get("LLM_API_KEY") or get("DEEPSEEK_API_KEY") or ""
    u = ((base_url or "").strip() or get("LLM_BASE_URL") or get("DEEPSEEK_BASE_URL")
         or _DEEPSEEK_BASE_URL)
    m = ((model or "").strip() or get("LLM_MODEL") or get("DEEPSEEK_MODEL")
         or _DEEPSEEK_MODEL)
    return k, u, m


# ---- 调用预算：按端点自动定 token/超时（2026-10-30 本地模型接入）----------
# 背景：本地 llama.cpp 服务（Intern-S1-mini，8B / Q8 / 8GB 显卡）实测约
#   **10 token/s**，而云端 deepseek-chat 是 100+ token/s —— 相差一个数量级。
#   同一套 1800 token / 75s 的预算，在本地意味着"必然超时"（1800/10 = 180s）。
# 口径：**一个判定只有一处真相** —— 是否本地、给多少额度，都只在这里定，
#   调用方（qa.py）不许再写 1800 / 75 这类字面量。
_LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")


def is_local_endpoint(base_url: str = "") -> bool:
    """本次调用是否打向**本机**服务（本地推理慢，预算必须放宽）。"""
    from urllib.parse import urlparse

    from .envfile import get
    u = ((base_url or "").strip() or get("LLM_BASE_URL")
         or get("DEEPSEEK_BASE_URL") or _DEEPSEEK_BASE_URL)
    try:
        host = (urlparse(u).hostname or "").strip().lower()
    except Exception:          # noqa: BLE001 — 地址不合规就当远程，交给调用层报错
        host = ""
    return host in _LOCAL_HOSTS


def qa_budget(base_url: str = "") -> tuple[int, float]:
    """问答用的 `(max_tokens, timeout)`。

    · 远程（deepseek / 书生云端）：1800 / 75s —— 原口径不动，四段式写得开；
    · 本机（llama.cpp）：1000 / 360s —— 1000 token 约 100s 生成完，
      给 360s 是留足 prompt 处理与显卡抖动；额度压到 1000 是为了让用户
      **两分钟内看到答案**，而不是"答得更全但等三分钟"。
    """
    if is_local_endpoint(base_url):
        return 1000, 360.0
    return 1800, 75.0


# ---- 非 API 地址识别（2026-10-31 实测踩坑，凭据 cache/qa_errors.log）----------
# 把 Base URL 填成「网页控制台」地址（platform.deepseek.com）时，请求根本到不了
# 模型：WAF 直接拦下，回一张 HTML「Error - Request Blocked」页 + 状态码 429，
# SDK 把它包成 RateLimitError。若上层按「限流」处理（退避重试 + 提示"稍后再问"）
# 就是误诊 —— 重试一万次也不会通，还白等 9 秒。
# 这里只列**确认过**的对照，不猜；拿不准的一律返回空串，交给调用层如实报错。
_NON_API_HOSTS = {
    "platform.deepseek.com": "https://api.deepseek.com",
    "www.deepseek.com": "https://api.deepseek.com",
    "deepseek.com": "https://api.deepseek.com",
}


def endpoint_url_hint(base_url: str = "") -> str:
    """`base_url` 指向已知的**网页控制台/门户**时，返回一句人话纠正；否则返回空串。"""
    from urllib.parse import urlparse
    try:
        host = (urlparse((base_url or "").strip()).hostname or "").strip().lower()
    except Exception:          # noqa: BLE001 — 不合规地址交给调用层如实报错
        return ""
    api = _NON_API_HOSTS.get(host)
    if api:
        return f"{host} 是网页控制台，不是 API 地址；API 地址是 {api}"
    return ""


def looks_non_api_reply(ex: BaseException) -> bool:
    """异常内容显示**对端根本不是 API**（回的是 HTML 拦截页，而不是 JSON 错误体）。

    正常 OpenAI 兼容服务的错误体是 JSON（`{"error": {...}}`）；
    出现 `<!DOCTYPE` / `<html` / `Request Blocked` 说明打到了网页/WAF。
    """
    low = str(ex).lower()
    return ("<!doctype" in low or "<html" in low or "request blocked" in low)


def probe_local(*, port: int = 0, timeout: float = 0.9) -> dict:
    """探测本机是否已有部署好的 OpenAI 兼容模型服务（页面三级优先级的第二层）。

    为什么需要（2026-10-30 用户定）：用户没在页面填 API 时，先看看本机有没有
    跑着模型（桌面『书生本地模型』那个 llama.cpp，默认 `127.0.0.1:8080`）。
    有就直接用 —— 不让用户为了"本机明明已经有"再去申请云端 Key。

    返回 `{ok, url, model, port, note}`，**绝不抛异常**：
    探测不到是正常状态（大多数机器都没跑本地模型），不能把页面带崩。

    ⚠ **必须绕开 HTTP 代理**：本机常设 `http_proxy`，urllib 会照着走，
      对 127.0.0.1 的请求会被代理劫持成 502（2026-10-29 已踩过同类坑）。
      `ProxyHandler({})` 空表 = 一个代理都不认。
    ⚠ 端口可用 `PAPER2LAB_LOCAL_PORT` 覆盖（默认 8080），换端口不用改代码。
    """
    import json as _json
    import urllib.request as _rq

    from .envfile import get
    try:
        p = int(port or (get("PAPER2LAB_LOCAL_PORT") or "8080").strip() or 8080)
    except (TypeError, ValueError):
        p = 8080
    base = f"http://127.0.0.1:{p}/v1"
    try:
        opener = _rq.build_opener(_rq.ProxyHandler({}))
        with opener.open(base + "/models", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace") or "{}"
        data = _json.loads(raw)
        items = data.get("data") or []
        mid = ""
        if items and isinstance(items[0], dict):
            mid = (items[0].get("id") or "").strip()
        return {"ok": True, "url": base, "model": mid, "port": p, "note": ""}
    except Exception as ex:        # noqa: BLE001 — 探测失败就是"没有"，不是错误
        return {"ok": False, "url": base, "model": "", "port": p,
                "note": type(ex).__name__}


def resolve_vision(*, page_key: str = "", page_url: str = "", page_model: str = "",
                   env_key: str = "", env_url: str = "", env_model: str = "",
                   env_key_has_known_url: bool = False,
                   llm_key: str = "", llm_url: str = "", llm_model: str = "",
                   llm_multimodal: bool = False) -> dict:
    """判定"读图通道"到底能不能用、用的是谁 —— **纯函数，便于逐分支验证**。

    口径（2026-09-30 用户定）：
      · 模型 API 与视觉 API 都必须 **Key + Base URL 成对填写**，程序**不按 key 前缀猜端点**
        （猜错会打到错误地址，每个区域都要重试等待，表现为极慢甚至像卡死）；
      · 多模态模型（自带读图）→ 无需视觉 API，直接复用模型 API；
      · 单模态模型 → 必须单独配视觉 API，否则**如实提示"无法读图"**。

    `env_key_has_known_url`：服务端用的是 DASHSCOPE_API_KEY / ZHIPUAI_API_KEY 这类
    **厂商专用变量**时，端点可由程序确知，因此允许只填 key（向后兼容老配置）。

    返回 dict：{ok, key, url, model, source, from_llm, reason, warning}
      · ok=True  → 三项可用，source 说明"谁提供的"；warning 非空时表示**还有一处配置不完整**
      · ok=False → reason 是**给用户看的**人话原因（不能是空串）

    优先级（**页面意图 > 服务端默认**，与本项目一贯口径一致）：
      1. 页面②成对填了 Key+URL          → 用它（2026-10-30 起页面已无这组输入，
                                            该分支只为 CLI / 老调用方保留）
      2. ①的模型被判为多模态且成对       → 复用①的模型 API
         （判定来源：`looks_multimodal(model)` 按模型名自动识别，页面不再有勾选框）
      3. 页面②只填了 Key 没填 URL       → 失败（提示补 URL）
      4. 服务端①成对 / 厂商专用变量      → 用它
      5. 服务端只填了 Key 没填 URL      → 失败
      6. 其余                          → 失败（说清是"没配"还是"模型不支持读图"）
    """
    pk, pu, pm = (page_key or "").strip(), (page_url or "").strip(), \
        (page_model or "").strip()
    ek, eu, em = (env_key or "").strip(), (env_url or "").strip(), \
        (env_model or "").strip()
    lk, lu, lm = (llm_key or "").strip(), (llm_url or "").strip(), \
        (llm_model or "").strip()

    # 页面②填了 key 却没填 url：无论最终走哪条路，这件事都要说出来（不能静默忽略）
    _incomplete = "已填视觉模型 Key，但**没有填视觉 Base URL**（成对填写才生效，程序不猜端点）"

    # ① 页面②成对填写 —— 意图最明确
    if pk and pu:
        return {"ok": True, "key": pk, "url": pu, "model": pm or em,
                "source": "页面输入", "from_llm": False, "reason": "",
                "warning": ""}

    # ② ①的模型被判为多模态且成对 → 复用①的模型 API
    #    （刻意排在服务端之前：页面填的模型是**明确的当前意图**，不该被 .env 里遗留的
    #      视觉 key 架空 —— 否则用户换了多模态模型，读图却还在偷偷走旧的视觉端点。）
    if llm_multimodal and lk and lu:
        return {"ok": True, "key": lk, "url": lu, "model": lm,
                "source": "复用模型 API（多模态）", "from_llm": True, "reason": "",
                "warning": (_incomplete + "——该配置已被忽略，当前走①的模型 API"
                            if (pk and not pu) else "")}

    # ③ 页面②填了 key 却缺 URL → 失败（说清缺什么）
    if pk and not pu:
        return {"ok": False, "key": "", "url": "", "model": "", "source": "",
                "from_llm": False, "reason": _incomplete, "warning": ""}

    # ④ 服务端（.env / Secrets）：成对，或厂商专用变量（端点已知）
    if ek and (eu or env_key_has_known_url):
        return {"ok": True, "key": ek, "url": eu, "model": pm or em,
                "source": ".env / 环境变量 / Secrets", "from_llm": False,
                "reason": "", "warning": ""}
    if ek and not eu and not env_key_has_known_url:
        return {"ok": False, "key": "", "url": "", "model": "", "source": "",
                "from_llm": False,
                "reason": "服务端配了视觉 key 但**没有配 Base URL**"
                          "（成对配置才生效，程序不猜端点）", "warning": ""}

    # ⑤ 都不满足 —— 如实说明"为什么读不了图"
    if not lk:
        why = "未配置任何模型 API（① 为空，服务端也没配）"
    elif llm_multimodal and not lu:
        why = "①自称多模态但**没有填 Base URL**，无法复用它读图（成对填写才生效）"
    else:
        why = (f"①的模型 `{lm}` 按模型名判定为单模态（不含 `-vl` / `vision` / `4v` 这类标识），"
               "服务端也没有另配视觉模型 API")
    return {"ok": False, "key": "", "url": "", "model": "", "source": "",
            "from_llm": False, "reason": why, "warning": ""}

_STATUS_ZH = {
    Status.INCONSISTENT: "不一致",
    Status.MISSING_IN_CODE: "代码中未找到",
    Status.MISSING_IN_PAPER: "论文中未声明",
    Status.NOT_FIXED: "未固定",
    Status.UNVERIFIABLE: "代码中以变量动态确定, 静态分析无法验证",
    Status.INTERNAL_INCONSISTENT: "同一载体内部多处声明互不一致",
    Status.CONSISTENT: "一致",
}


# ---------------------------------------------------------------- 缓存

def _load_cache() -> dict:
    try:
        if CACHE_PATH.exists():
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: dict):
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    except Exception:
        pass


def _cache_key(prompt: str, mode: str) -> str:
    """缓存键带模式前缀: Mock 与真实 API 的结果不互相污染。

    (早期版本用同一个键, 导致先跑 Mock 后配 key 时命中 Mock 文本,
     看起来像"DeepSeek 没生效"。)
    """
    return hashlib.sha256(f"{mode}\x00{prompt}".encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------- Prompt

def _build_prompt(f: Finding) -> str:
    paper_side = "论文中未找到该参数声明" if f.paper is None else (
        f"论文声明: {f.paper.raw_value}(出处: {f.paper.location}, 原文: \"{f.paper.snippet}\")")
    code_side = "代码中未找到该参数" if f.code is None else (
        f"代码实现: {f.code.raw_value}(位置: {f.code.location}, 代码: \"{f.code.snippet}\")")
    return (
        "你是科研实验复现审计助手。以下审计发现已由程序化规则验证确认, "
        "不需要你判断真伪, 请只做风险解释与归因。\n\n"
        f"参数: {f.display_name}\n"
        f"审计状态: {_STATUS_ZH.get(f.status, f.status.value)}\n"
        f"风险等级: {f.risk.value}\n"
        f"{paper_side}\n"
        f"{code_side}\n\n"
        "请用 2-3 句中文回答:\n"
        "1) 该差异对实验复现的具体影响;\n"
        "2) 最可能的成因;\n"
        "3) 一句可执行的修复建议。\n"
        "只输出纯文本, 不要标题、不要 markdown、不要客套话。"
    )


# ---------------------------------------------------------------- Mock 模板

_MOCK_TEMPLATES = {
    "learning_rate": (
        "学习率存在{magnitude}差异, 会直接改变梯度下降步长与收敛轨迹, "
        "通常导致最终精度与论文结果出现可观测偏差, 是最常见的复现失败诱因。"
        "成因多为代码默认值未按论文更新或沿用了其他实验配置。"
        "建议将训练脚本中的学习率修正为论文声明值后重新实验。"
    ),
    "seed": (
        "论文要求固定随机种子而代码未固定, 数据打乱顺序与参数初始化每次运行都不同, "
        "实验结果将无法稳定复现。成因多为开发阶段图省事省略了 seed 设置。"
        "建议在训练入口显式调用 torch.manual_seed 等固定种子。"
    ),
    "dataset": (
        "数据集版本不一致意味着训练/评测样本分布可能不同, 指标不具备可比性, "
        "会直接削弱复现结论的可信度。成因多为沿用了旧版数据或未记录版本。"
        "建议统一下载论文声明的数据集版本并核对校验信息。"
    ),
    "_default": (
        "该参数与论文声明不一致, 可能使实际训练配置偏离论文设定, "
        "从而影响复现结果的可比性。成因多为代码默认值未同步论文更新。"
        "建议按论文声明值修正后重新运行实验验证。"
    ),
}

# 结果类 finding 的模板
_MOCK_RESULT = (
    "实际实验结果与论文声明值存在明显差距, 结合上方参数审计发现, "
    "差异很可能源于训练配置不一致而非方法本身失效。"
    "建议先消除所有高风险参数差异, 再重新执行实验对比。"
)

# 动态确定(变量引用)类 finding 的模板
_MOCK_UNVERIFIABLE = (
    "该参数在代码中以变量形式动态确定(非硬编码字面量), 静态分析无法验证其运行时取值。"
    "这本身不是错误, 但意味着论文声明值与代码实际值的一致性需要运行时核对。"
    "建议在代码中打印该参数的运行日志, 或将其值写入配置文件以便追溯。"
)


# 内部不一致类 finding 的模板
_MOCK_INTERNAL = (
    "同一载体内部对该参数的表述/实现存在多个不同值, 这是一条由交叉审查(多处声明互查)"
    "自动发现的问题。若两处差异并非出自灵敏度分析或分组实验的刻意设置, "
    "则说明该载体的参数表述不自洽, 会直接影响复现时的取值判断。"
    "建议先明确唯一口径, 再统一正文、附录与代码中的取值。"
)


# 代码未声明/找不到时的模板
_MOCK_MISSING_IN_CODE = (
    "论文明确声明了该参数, 但静态扫描未在代码中找到对应实现(既非命名赋值也非内联判定)。"
    "可能原因: 代码使用了框架默认值、参数由外部配置注入, 或该设置确实被遗漏。"
    "建议在代码中显式声明并与论文对齐, 以保证复现时取值可追溯。"
)

# 论文未声明时的模板
_MOCK_MISSING_IN_PAPER = (
    "代码中给出了该参数的实现值, 但论文未作声明(或仅在正文中泛泛提及而未给出取值)。"
    "这属于实验记录完整度问题: 读者仅凭论文无法还原该设置, 必须翻阅代码。"
    "建议在实验设置章节补齐声明, 提升论文自身的可复现性。"
)


def _mock_analysis(f: Finding) -> str:
    if f.param_key.startswith("result_"):
        return _MOCK_RESULT
    if f.status == Status.UNVERIFIABLE:
        return _MOCK_UNVERIFIABLE
    if f.status == Status.INTERNAL_INCONSISTENT:
        return _MOCK_INTERNAL
    if f.status == Status.MISSING_IN_CODE:
        return _MOCK_MISSING_IN_CODE
    if f.status == Status.MISSING_IN_PAPER:
        return _MOCK_MISSING_IN_PAPER
    tpl = _MOCK_TEMPLATES.get(f.param_key, _MOCK_TEMPLATES["_default"])
    magnitude = ""
    try:
        pv, cv = float(f.paper.normalized), float(f.code.normalized)
        if pv and cv and pv != cv:
            ratio = max(pv, cv) / min(pv, cv)
            magnitude = f"约 {ratio:g} 倍数量级" if ratio >= 5 else ""
    except (TypeError, ValueError, AttributeError):
        pass
    return tpl.format(magnitude=magnitude)


# ---------------------------------------------------------------- DeepSeek 调用

def _deepseek_model() -> str:
    """模型名优先取 .env 的 DEEPSEEK_MODEL(调用时读取, 不受导入顺序影响)。"""
    from .envfile import get
    return get("LLM_MODEL") or get("DEEPSEEK_MODEL") or _DEEPSEEK_MODEL


# ---------------------------------------------------------------- 运行代次 + 本机串行
# 2026-10-30 晚（用户要求）：「点一次新的，就把上一次终止、清理干净再跑」。
#
# 为什么必须有：本地 llama.cpp 是**单实例单槽**，上下文总量有限；
# 一问一答就要 2500+ token，两个请求同时进来必然把共享上下文挤爆
# （实测报 500 `Context size has been exceeded`）或触发限流（429）。
# 用户看到的就是「模型调用失败」，而下面还挂着一屏审计事实 —— 很像程序坏了。
#
# 三层保护，缺一不可：
#   ① `begin_run()`：每次点击生成新令牌，**旧令牌立刻作废**。旧任务在
#      **下一次模型调用前**发现自己过期就抛 `RunCancelled` 停手，结果丢弃。
#      ⚠ Python 无法硬中断在途的 HTTP 请求，所以"终止"的粒度 = 这一次模型调用。
#   ② `_LOCAL_LOCK`：本机端点**串行**，一个时刻只有一个请求打到 llama.cpp。
#      所以旧请求还在途时，新任务只是排队，绝不会并发挤爆上下文。
#   ③ 本机端点 `trust_env=False` + `max_retries=0`：
#      · 不走系统/环境代理 —— 本机地址经过代理既慢又可能被代理限流（429 的常见来源）；
#      · 不自动重试 —— 重试只会把并发叠得更高，还让用户多等一倍时间。
class RunCancelled(Exception):
    """本次运行已被新的一次点击作废。**不是错误**，不要当失败报给用户。"""


_run_seq = 0
_run_seq_lock = threading.Lock()
_RUN_LOCAL = threading.local()      # 本线程"这一轮"的令牌（Streamlit 每次 rerun 是新线程）
_LOCAL_LOCK = threading.Lock()


def begin_run() -> int:
    """开始一次新运行并作废上一次，返回本次令牌。"""
    global _run_seq
    with _run_seq_lock:
        _run_seq += 1
        tok = _run_seq
    _RUN_LOCAL.token = tok
    return tok


def current_run() -> Optional[int]:
    """**本线程**这一轮的令牌；`None` = 本线程不受管辖。"""
    return getattr(_RUN_LOCAL, "token", None)


def check_run(token: Optional[int] = None) -> None:
    """令牌过期就抛 `RunCancelled`。

    ⚠ 两个最容易写错的地方（10-30 晚都踩过）：
      ① 不能拿「全局最新令牌」自己跟自己比 —— 那样恒等、永远不取消。
         必须比 **本线程开跑时的令牌** vs **当前全局最新令牌**；
      ② 没调过 `begin_run()` 的线程（脚本、测试、一次性调用）**必须放行**。
         否则任何命令行调用都会被判成"已作废"，把既有测试全打挂。
    """
    t = current_run() if token is None else token
    if t is None:                   # 本线程没有令牌 ⇒ 不受管辖，一律放行
        return
    with _run_seq_lock:
        latest = _run_seq
    if t != latest:
        raise RunCancelled("已被更新的一次点击作废")


def _call_deepseek(prompt: str, api_key: str, max_tokens: int = 300,
                   base_url: str = "", model: str = "",
                   timeout: float = 30.0, run_token: Optional[int] = None) -> str:
    """调用 OpenAI 兼容的 chat.completions。

    `base_url` / `model` 为空时回落到 .env / 环境变量 / 官方默认 —— 因此
    **不传参的老调用点行为完全不变**。

    `timeout`：2026-10-30 加。默认仍是 30s（老调用点不变），但**问答**必须给更长 ——
    书生 intern-s2 会先把推理写进正文，1500 token 要 40s 以上，30s 一定超时，
    用户看到的就是"模型调用失败"（实测：问答 92s 后超时、退回证据检索）。

    `run_token`：2026-10-30 晚加。「点击即终止上一次」靠它 —— 传了就只认这个令牌，
    不传就用当前全局令牌（这样流水线里几十个老调用点不用逐个改）。
    """
    import httpx
    from openai import OpenAI

    from .envfile import get
    url = (base_url or "").strip() or get("LLM_BASE_URL") or get("DEEPSEEK_BASE_URL") \
        or _DEEPSEEK_BASE_URL
    name = (model or "").strip() or _deepseek_model()
    local = is_local_endpoint(url)
    # 本机：不走代理（trust_env=False）+ 不自动重试；远程：保持 SDK 默认（要吃代理、要重试）
    kwargs: dict = {"api_key": api_key, "base_url": url, "timeout": timeout}
    if local:
        kwargs["max_retries"] = 0
        kwargs["http_client"] = httpx.Client(trust_env=False, timeout=timeout)
    client = OpenAI(**kwargs)

    def _fire() -> str:
        resp = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    if not local:
        check_run(run_token)
        return _fire()
    # 本机串行：排队后**先看自己是否已作废**，作废就别白跑一次生成
    with _LOCAL_LOCK:
        check_run(run_token)
        return _fire()


# ---------------------------------------------------------------- 一次性 JSON 问答
# 给"不是归因"的两处用（2026-09-28）:
#   · 规划器 `planner`（让 AI 排计划）—— mode="plan"
#   · 补丁的「LLM 提议 + 程序盖章」通道 —— mode="patch"
# 与归因共用**同一份缓存**（省调用、可复现），但**不共用**降级模板：
# 这里拿不到 JSON 就返回 None，由调用方**回退确定性路径**（规则版计划 / 拒绝），
# 绝不编一个假计划出来。
_aux_lock = threading.Lock()


def _extract_json(text: str) -> Optional[dict]:
    """从模型回复里抠出 JSON（容忍 ```json 围栏与前后废话）。"""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if "```" in t[3:] else t[3:]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
    except Exception:      # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def ask_json(prompt: str, api_key: str, *, mode: str = "aux",
             max_tokens: int = 800, force_mock: bool = False,
             base_url: str = "", model: str = "") -> tuple:
    """一次性的 JSON 问答，返回 `(dict 或 None, 说明)`。**绝不抛异常。**

    调用方口径：拿不到（None）就回退确定性路径，并把说明如实写进日志/报告 ——
    "AI 这次没参与"本身是要告诉用户的信息，不是静默失败。
    """
    # ⚠️ force_mock 必须**优先于缓存**（2026-09-29 修）:
    # 它的语义是"完全确定性、绝不调外部"。原来把缓存检查放在它前面,
    # 于是"离线测试会不会回退"取决于**之前有没有真跑过一次**（缓存里有 = AI 参与过）
    # —— 测试因此时好时坏，是典型的 flaky。强制离线就该直接回退，不看缓存。
    if force_mock:
        return None, f"AI（{mode}）· 强制离线（force_mock），回退确定性路径"
    key = _cache_key(prompt, mode)
    with _aux_lock:
        cache = _load_cache()
    if key in cache:
        obj = _extract_json(cache[key]) if isinstance(cache[key], str) else None
        if obj is not None:
            return obj, f"AI（{mode}）· 缓存命中"
    if not api_key:
        return None, f"AI（{mode}）· 未配置 LLM，回退确定性路径"

    # 为什么允许**一次**重试（2026-09-29 实测）:
    #   真实调用会偶发返回非 JSON —— 那多半不是"模型不会写 JSON"，而是服务端回了
    #   限流/异常文案（实测：同一个 prompt 连跑 3 次全成功，但此前有一次就是这种失败）。
    #   这类抖动重试一次就能过去。**API 层异常（鉴权/网络）不重试** —— 那种重试也是白费。
    #   重试必须**如实说出来**（"第 2 次尝试才成功"），不能假装一次就成。
    text, last_note = "", "未调用"
    # ---- 2026-10-31 本机模型实测暴露的两个真 bug，都在这一行调用上 --------------
    # ① `base_url` / `model` 两个参数**收了却没用**：ask_json 从头到尾只调
    #    `_call_deepseek(prompt, api_key, max_tokens=...)`，端点/模型一律回落到 .env。
    #    后果不是报错，而是**静默用错模型** —— 页面里填云端、排计划却打本机（或反过来）。
    #    这条影响全部 9 个调用点：排计划 / 模型先判 / 角色 / 自主回头 / 补丁提名 /
    #    引证核验 / 实验站挑工具 / 抽取 Agent。这里补上转发。
    # ② 超时写死 30s（`_call_deepseek` 的默认值）。本机 8B 实测约 10 token/s，
    #    900~1400 token 的 JSON 计划要 90~140s ⇒ **必定超时** ⇒ `planner` 永远是 `rule`、
    #    模型先判永远"没判出结果"，而界面只显示"模型没参与"，看不出是超时。
    #    改成按端点折算：远程维持 30s 不变（老行为零变化），本机按 max_tokens 估时。
    _timeout = 30.0
    if is_local_endpoint(base_url):
        _timeout = min(300.0, max(90.0, max_tokens / 8.0 + 30.0))
    for attempt in (1, 2):
        try:
            text = _call_deepseek(prompt, api_key, max_tokens=max_tokens,
                                  base_url=base_url, model=model,
                                  timeout=_timeout)
        except Exception as ex:      # noqa: BLE001
            return None, (f"AI（{mode}）· 调用失败（{type(ex).__name__}）"
                          f"，回退确定性路径")
        obj = _extract_json(text)
        if obj is not None:
            with _aux_lock:
                cache = _load_cache()
                cache[key] = text
                _save_cache(cache)
            return obj, (f"AI（{mode}）· 真实调用"
                         + ("" if attempt == 1 else "（第 2 次尝试才成功）"))
        last_note = "返回不是合法 JSON"

    # 失败必须带上**返回片段**：只报"不是合法 JSON"会把排查方向带偏
    # —— 真实原因可能是限流 / 余额 / 服务异常，那些文案本来就不是 JSON，看片段一眼能分清。
    head = re.sub(r"\s+", " ", text or "").strip()[:140]
    detail = f"（返回片段：{head}）" if head else "（返回为空）"
    return None, f"AI（{mode}）· {last_note}{detail}，回退确定性路径"



# ---------------------------------------------------------------- 对外接口

class LLMAnalyzer:
    """归因分析器: 自动选择 真实API -> 缓存 -> Mock 三级降级。"""

    def __init__(self, api_key: Optional[str] = None, force_mock: bool = False, *,
                 base_url: str = "", model: str = ""):
        from .envfile import get
        # 走 envfile: 兼容 .env 与系统环境变量(CLI 入口不必自己 load_env)
        self.api_key = api_key or get("LLM_API_KEY") or get("DEEPSEEK_API_KEY")
        # 端点与模型名：调用方没传就回落 .env / 环境变量 / 官方默认
        self.base_url = (base_url or "").strip() or get("LLM_BASE_URL") \
            or get("DEEPSEEK_BASE_URL") or _DEEPSEEK_BASE_URL
        self.model = (model or "").strip() or _deepseek_model()
        self.force_mock = force_mock
        self.cache = _load_cache()
        self.mode = "mock"
        self.last_error = ""      # 真实 API 失败原因(界面如实提示)
        self.failed_calls = 0
        self.real_calls = 0       # 成功调用真实 API 的次数
        self.cache_hits = 0       # 命中缓存的条数(不产生 API 费用)
        # 缓存命名空间（2026-09-30）：换模型必须换缓存 —— 否则从 A 模型切到 B 模型时
        # 会命中 A 模型写下的文本，看起来像"新模型没生效"。
        # 默认模型仍用 "deepseek" 前缀，**与历史缓存完全兼容**（老缓存不作废）。
        self.cache_ns = "deepseek" if self.model == _DEEPSEEK_MODEL else f"llm:{self.model}"
        # 分片并发(第二期第四批)下的共享资源保护: cache / 计数 / mode 都会被多线程碰
        self._lock = threading.Lock()
        self.shard_report: dict = {}
        if not force_mock and self.api_key:
            self.mode = "deepseek"

    def analyze(self, f: Finding) -> str:
        """为单条 Finding 生成归因文字(只处理非一致项)。"""
        if f.status == Status.CONSISTENT:
            f.ai_analysis = ""
            f.confidence = f.confidence  # 保持 Confirmed
            return ""

        prompt = _build_prompt(f)
        # 缓存命名空间含模型名 —— 换模型即换缓存，避免命中上一个模型写的文本。
        key = _cache_key(prompt, self.cache_ns)

        # 1) 缓存命中(加锁: 分片并发下 cache 会被多个线程同时读写)
        with self._lock:
            if key in self.cache:
                f.ai_analysis = self.cache[key]
                self.cache_hits += 1
                return f.ai_analysis

        # 2) 真实 API
        if self.mode == "deepseek":
            last_err: Optional[Exception] = None
            for _ in range(3):  # 重试 3 次
                try:
                    text = _call_deepseek(prompt, self.api_key,
                                          base_url=self.base_url, model=self.model)
                    if text:
                        with self._lock:      # 写缓存+落盘必须互斥, 否则并发下会互相覆盖
                            self.cache[key] = text
                            _save_cache(self.cache)
                            self.real_calls += 1
                        f.ai_analysis = text
                        return text
                except Exception as e:  # noqa: BLE001
                    last_err = e
            # API 彻底失败, 落 Mock 保底(演示不中断)。失败原因记在实例上供界面提示。
            with self._lock:
                self.last_error = (f"{type(last_err).__name__}: {last_err}"
                                   if last_err else "未知错误")
                self.failed_calls += 1
                if self.failed_calls == 1:
                    self.mode = "fallback"    # 真实 API 不可用, 后续走模板

        # 3) Mock 保底(模板确定性输出, 不写缓存以免污染真实 API 的缓存)
        text = _mock_analysis(f)
        f.ai_analysis = text
        return text

    def analyze_all(self, findings: list[Finding], *,
                    max_workers: int = DEFAULT_MAX_WORKERS) -> int:
        """批量归因, 返回生成条数。

        第二期第四批: 由**串行**改为**分片并发**。每片仍是一条独立调用
        (prompt 里只有这一条的证据, 上下文互不干扰), 因此"拆分 - 各自处理 - 汇总"
        只改变调度方式, **不改变任何结论**; 结果**保序**, 与串行逐字一致。
        """
        todo = [f for f in findings if f.status != Status.CONSISTENT]
        if not todo:
            self.shard_report = {"label": "AI 归因", "mode": "serial", "total": 0,
                                 "done": 0, "failed": 0, "workers": 1,
                                 "seconds": 0.0, "failures": []}
            return 0
        _, report = run_sharded(todo, lambda i, f: self.analyze(f),
                                max_workers=max_workers, label="AI 归因")
        self.shard_report = report.to_dict()
        return len(todo)
