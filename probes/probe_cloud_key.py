"""云端 key 自查：这个 DeepSeek 密钥到底还能不能用、还剩多少钱。

用途（2026-10-30 晚）：
    用户问「怎么可能用云端 dsh 会不行？」——那就别猜，直接测。
    这个脚本只做三件事，全部打真实接口，几秒钟出结果：
      ① 查余额（花 0 元）
      ② 发一句「hi」（花几厘钱）
      ③ 连发 8 次短请求，看会不会被限流（429）

判读方法（脚本会直接把结论写成人话）：
    · 余额 > 0 且单次调用 200 → key 是好的，"调用失败"只可能是**瞬时**限流；
    · 余额 = 0 / 401 → 是真的没钱或 key 失效，去充值/换 key；
    · 连发里出现 429 → 这一分钟内请求太密，等半分钟就好。

⚠ 不走系统代理也测一遍：本机常设 http_proxy（用户机器上有 127.0.0.1:63504），
  代理对同一个出口 IP 的请求也可能回 429 —— 那样的话问题在代理，不在 key。
  两种都通才算 key 真好。

用法:
    .venv\\Scripts\\python.exe probes\\probe_cloud_key.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx                                          # noqa: E402

from paper2lab import envfile                         # noqa: E402

API = "https://api.deepseek.com"


def _key() -> str:
    """优先取 DEEPSEEK_API_KEY（云端专用），没有再看 LLM_* 是不是云端配置。"""
    k = (envfile.get("DEEPSEEK_API_KEY") or "").strip()
    if k:
        return k
    u = (envfile.get("LLM_BASE_URL") or "").strip()
    if u and "127.0.0.1" not in u and "localhost" not in u:
        return (envfile.get("LLM_API_KEY") or "").strip()
    return ""


def main() -> int:
    k = _key()
    if not k:
        print("没找到云端 key：.env 里既没有 DEEPSEEK_API_KEY，"
              "LLM_* 也不是云端地址。")
        return 1
    print(f"用这个 key 测：{k[:8]}…{k[-4:]}（共 {len(k)} 位）\n")

    ok_any = False
    for trust, label in ((False, "直连（不走代理）"), (True, "走系统代理")):
        c = httpx.Client(timeout=30, trust_env=trust)
        print(f"—— {label} ——")
        # ① 余额
        try:
            r = c.get(f"{API}/user/balance",
                      headers={"Authorization": f"Bearer {k}"})
            if r.status_code == 200:
                d = r.json()
                infos = d.get("balance_infos") or [{}]
                b = infos[0]
                print(f"  余额：{b.get('total_balance', '?')} {b.get('currency', '')}"
                      f"（可用：{d.get('is_available')}）")
            else:
                print(f"  余额查询 HTTP {r.status_code}：{r.text[:120]}")
        except Exception as ex:                        # noqa: BLE001
            print(f"  余额查询失败：{type(ex).__name__} {str(ex)[:100]}")

        # ② 单次调用
        t = time.time()
        try:
            r = c.post(f"{API}/chat/completions",
                       headers={"Authorization": f"Bearer {k}"},
                       json={"model": "deepseek-chat",
                             "messages": [{"role": "user", "content": "hi"}],
                             "max_tokens": 3})
            used = time.time() - t
            if r.status_code == 200:
                ok_any = True
                print(f"  单次调用：成功（{used:.1f}s）")
            else:
                print(f"  单次调用 HTTP {r.status_code}（{used:.1f}s）：{r.text[:160]}")
        except Exception as ex:                        # noqa: BLE001
            print(f"  单次调用失败：{type(ex).__name__} {str(ex)[:100]}")

        # ③ 连发 8 次，专门抓限流
        codes = []
        for _ in range(8):
            try:
                rr = c.post(f"{API}/chat/completions",
                            headers={"Authorization": f"Bearer {k}"},
                            json={"model": "deepseek-chat",
                                  "messages": [{"role": "user", "content": "hi"}],
                                  "max_tokens": 3})
                codes.append(rr.status_code)
            except Exception as ex:                    # noqa: BLE001
                codes.append(type(ex).__name__)
        n429 = sum(1 for x in codes if x == 429)
        print(f"  连发 8 次：{codes}" + (f"　← 有 {n429} 次被限流" if n429 else "　← 全部通过"))
        print()

    print("—— 结论 ——")
    if ok_any:
        print("  key 是好的，云端服务此刻正常。")
        print("  如果之前看到「模型调用失败（RateLimitError）」，那是**瞬时**限流：")
        print("  同一分钟内请求太密（例如审计刚跑完十几次模型调用、紧接着就问）。")
        print("  处理办法：等半分钟再问。程序现在会自己先等 3 秒、6 秒各重试一次。")
    else:
        print("  key 打不通。看上面的状态码：")
        print("  401 → key 失效/填错；402 或余额 0 → 没钱了去充值；")
        print("  429 → 被限流（等半小时）；连不上 → 网络或代理问题。")
    return 0 if ok_any else 1


if __name__ == "__main__":
    sys.exit(main())
