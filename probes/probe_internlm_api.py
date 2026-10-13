"""能力探针：书生（InternLM）API 是否支持智能体所需的三个关键能力。

为什么先跑它再动手改造：整个"智能体化"方案的地基是"模型能自己调工具"。
如果不先钉死这三件事，后面所有设计都可能建在错误前提上：
  T1 连通性   —— key / base_url 是否可用，能看到哪些模型名；
  T2 函数调用 —— 支不支持 OpenAI 标准的 tools / tool_calls（决定走"标准 function
                 calling"还是"模型输出 JSON 指令 + 程序解析"的退路）；
  T3 读图     —— 能不能直接喂图片（S2 自称多模态；决定能不能简化现有 vision 通道）。

本探针**只用标准库**（urllib），不依赖 openai 包，因此不装任何东西就能跑。
key 一律从环境变量读，**绝不写进本文件**。

用法（Git Bash / PowerShell）：
    INTERNLM_API_KEY=sk-xxxx python probes/probe_internlm_api.py
可选环境变量：
    INTERNLM_BASE_URL  默认 https://chat.intern-ai.org.cn/api/v1
    INTERNLM_MODEL     默认 intern-s2-preview-397b
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:      # noqa: BLE001
    pass

BASE = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1").rstrip("/")
MODEL = os.environ.get("INTERNLM_MODEL", "intern-s2-preview-397b")
KEY = os.environ.get("INTERNLM_API_KEY") or os.environ.get("LLM_API_KEY") or ""

TIMEOUT = 60


def _req(method: str, path: str, payload: dict | None = None) -> tuple[int, dict | str]:
    """发一个请求，返回 (状态码, 解析后的 dict 或原始文本)。"""
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {KEY}")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            body = r.read().decode("utf-8", "replace")
            try:
                return r.status, json.loads(body)
            except ValueError:
                return r.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except ValueError:
            return e.code, body
    except Exception as e:                       # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}"


def _tiny_png(w: int = 48, h: int = 48, rgb=(220, 30, 30)) -> str:
    """生成一张纯色 PNG 的 data URI（给读图测试用，不依赖 Pillow）。"""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    hdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", hdr)
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _head(x, n: int = 220) -> str:
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


def main() -> int:
    print("=" * 68)
    print("书生 API 能力探针")
    print("=" * 68)
    print(f"base_url : {BASE}")
    print(f"model    : {MODEL}")
    print(f"key      : {'已配置（' + KEY[:7] + '…，长度 ' + str(len(KEY)) + '）' if KEY else '❌ 未配置'}")
    if not KEY:
        print("\n请先设置环境变量 INTERNLM_API_KEY 再运行。")
        return 2

    verdict: dict[str, str] = {}

    # ---------------------------------------------------------- T1 连通性 / 模型清单
    print("\n" + "-" * 68)
    print("T1 · 连通性与模型清单  GET /models")
    print("-" * 68)
    status, body = _req("GET", "/models")
    print(f"HTTP {status}")
    ids: list[str] = []
    if isinstance(body, dict) and isinstance(body.get("data"), list):
        ids = [str(m.get("id")) for m in body["data"] if isinstance(m, dict)]
        print(f"可见模型 {len(ids)} 个：")
        for i in ids:
            mark = "  ← 本次选用" if i == MODEL else ""
            print(f"    - {i}{mark}")
        if MODEL not in ids:
            near = [i for i in ids if "s2" in i.lower() or "intern-latest" in i.lower()]
            print(f"  ⚠️ 选用模型 {MODEL} 不在清单里；相近的：{near or '无'}")
        verdict["T1 连通性"] = "✅ 可用"
    else:
        print(f"响应：{_head(body)}")
        verdict["T1 连通性"] = "❌ 失败（检查 key / base_url / 网络）"
    print(f"判定：{verdict['T1 连通性']}")

    # ---------------------------------------------------------- T2 函数调用
    print("\n" + "-" * 68)
    print("T2 · 函数调用（tools / tool_calls）  POST /chat/completions")
    print("-" * 68)
    tools = [{
        "type": "function",
        "function": {
            "name": "record_param",
            "description": "记录论文中的一个训练参数及其取值",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "参数名"},
                    "value": {"type": "string", "description": "参数取值"},
                },
                "required": ["name", "value"],
            },
        },
    }]
    payload = {
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": "论文里学习率是 1e-4。请调用工具把它记录下来，不要用文字回答。"}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    status, body = _req("POST", "/chat/completions", payload)
    print(f"HTTP {status}")
    if isinstance(body, dict) and body.get("choices"):
        msg = body["choices"][0].get("message", {}) or {}
        tc = msg.get("tool_calls")
        if tc:
            fn = (tc[0].get("function") or {})
            print(f"✅ 返回了 tool_calls：name={fn.get('name')} args={_head(fn.get('arguments'), 120)}")
            verdict["T2 函数调用"] = "✅ 支持标准 function calling（最优路线）"
        else:
            print(f"未返回 tool_calls，模型直接回了文字：{_head(msg.get('content'), 160)}")
            verdict["T2 函数调用"] = "⚠️ 不返回 tool_calls（需退到 JSON 指令路线）"
    else:
        print(f"响应：{_head(body)}")
        verdict["T2 函数调用"] = "❌ 调用失败"
    print(f"判定：{verdict['T2 函数调用']}")

    # ---------------------------------------------------------- T3 读图
    print("\n" + "-" * 68)
    print("T3 · 读图（多模态）  POST /chat/completions（带 image_url）")
    print("-" * 68)
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片的主色调是什么颜色？只回一个词。"},
                {"type": "image_url", "image_url": {"url": _tiny_png()}},
            ],
        }],
        "max_tokens": 64,
    }
    status, body = _req("POST", "/chat/completions", payload)
    print(f"HTTP {status}")
    if isinstance(body, dict) and body.get("choices"):
        txt = (body["choices"][0].get("message", {}) or {}).get("content")
        print(f"模型回复：{_head(txt, 120)}")
        verdict["T3 读图"] = "✅ 支持读图（可直接复用 S2 的多模态）"
    else:
        print(f"响应：{_head(body)}")
        verdict["T3 读图"] = "❌ 不支持或调用失败（保留独立视觉通道）"
    print(f"判定：{verdict['T3 读图']}")

    # ---------------------------------------------------------- 汇总
    print("\n" + "=" * 68)
    print("汇总")
    print("=" * 68)
    for k, v in verdict.items():
        print(f"  {k} : {v}")
    print("\n下一步（按 T2 结论二选一）：")
    print("  · T2 支持 → 走 LangGraph 标准 function calling：把工具箱做成 schema 给模型")
    print("  · T2 不支持 → 走 Lagent 式：让模型输出 JSON 指令，程序解析后执行")
    return 0 if verdict.get("T1 连通性", "").startswith("✅") else 1


if __name__ == "__main__":
    raise SystemExit(main())
