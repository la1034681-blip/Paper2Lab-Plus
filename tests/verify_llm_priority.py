"""LLM 三级优先级 —— 离线验证（探针行为 + 页面源码守卫）。

背景（2026-10-30 用户定）：
    页面的模型选择改成三级优先级：
      ① 页面三个框填**完整** → 用页面这套；
      ② 页面没填（或没填全）→ 探测本机有没有已部署的模型，有就用本地；
      ③ 本机也没有 → 服务端 .env 配了就用它；再没有 → 明确提示"无可用 LLM"，
        并自动降级跑全套链路 + 说明哪些环节准确度会下降。

这一项只验**不依赖浏览器**的部分：
    A. `probe_local()` 在"有服务 / 无服务 / 坏端口"三种情况下都不抛异常、判定正确；
    B. app.py 里三级的**顺序**没被写反（页面判断必须在探本机之前）；
    C. 三种回落状态的提示文案都存在（文案丢了等于用户看不到状态，与逻辑写错一样糟）；
       而「页面填全」这一档**刻意不弹状态卡**（2026-10-30 用户要求），用反向断言守住；
    D. 状态提示渲染在输入区之后（用户要求"放在 API 输入框下方"）。

用法：.venv\\Scripts\\python.exe -X utf8 tests\\verify_llm_priority.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper2lab.llm import probe_local          # noqa: E402

_ok = 0
_bad = 0


def check(cond, label: str, detail: str = "") -> None:
    global _ok, _bad
    ok = bool(cond)
    assert isinstance(ok, bool)
    if ok:
        _ok += 1
        print(f"  OK   {label}")
    else:
        _bad += 1
        print(f"  FAIL {label}" + (f"  —— {detail}" if detail else ""))


print("===== A 探针行为：三种情况都不能抛异常 =====")

# A1 明确没人监听的端口 —— 连接被拒是**立刻**返回，不该等超时
t0 = time.monotonic()
r = probe_local(port=59999, timeout=1.0)
el = time.monotonic() - t0
check(r["ok"] is False, "空闲端口 → 判定为「没有本地模型」", str(r))
check(bool(r["url"].endswith("/v1")), "空闲端口仍返回规范的 v1 地址（便于提示用户）", r["url"])
check(el < 3.0, "空闲端口快速返回（<3s，不会卡住页面）", f"{el:.1f}s")

# A2 非法端口（0 / 越界）也不能炸 —— 用户可能配了怪值
for bad in (0, 70000):
    try:
        r2 = probe_local(port=bad, timeout=0.6)
        check(isinstance(r2, dict) and "ok" in r2, f"端口 {bad} → 仍返回 dict（不抛异常）", str(r2)[:80])
    except Exception as ex:            # noqa: BLE001
        check(False, f"端口 {bad} → 仍返回 dict（不抛异常）", f"{type(ex).__name__}: {ex}")

# A3 真实 8080（本机装没装本地模型都可能，自适应判定，不写死）
r3 = probe_local(timeout=1.0)
if r3["ok"]:
    check(bool(r3["model"]), "8080 有服务 → 拿到模型名", str(r3))
    check(r3["url"] == "http://127.0.0.1:8080/v1", "8080 有服务 → 地址规范", r3["url"])
else:
    check(True, f"8080 当前没服务 → 判定为「没有」（{r3['note']}）")

print("\n===== B 三级顺序：页面 > 本机 > 服务端 > 无 =====")
src = (ROOT / "app.py").read_text(encoding="utf-8")

i_page = src.find("_page_full = bool(")
i_probe = src.find("_local = _probe_local()")
i_branch_page = src.find('if _page_full:')
i_branch_local = src.find('elif _local["ok"]:')
i_branch_env = src.find('elif (_lk or "").strip() and (_lu or "").strip():')
i_branch_none = src.find('_llm_src = "none"')

check(i_page > 0 and i_probe > 0, "页面输入与本机探测都真的存在")
check(0 < i_page < i_probe, "先读页面输入、再探本机（顺序不能反）",
      f"page={i_page} probe={i_probe}")
check(0 < i_branch_page < i_branch_local < i_branch_env < i_branch_none,
      "四个分支的判定顺序：页面 → 本机 → 服务端 → 无",
      f"{i_branch_page}/{i_branch_local}/{i_branch_env}/{i_branch_none}")

# 页面没填全时不能静默采用
check("_page_part" in src and "没填全" in src,
      "只填一半会当场提示（不静默采用半套配置）")
check("_lk_eff, _lu_eff, _lm_eff = \"local\", _local[\"url\"], _local[\"model\"]" in src,
      "本机分支真的把 key/url/model 三项都指到本地（不是只改地址）")

print("\n===== C 三种回落状态的文案（丢了文案=用户看不到状态）=====")
for label, kw in (
    ("② 本机已部署", "已检测到本机已部署的模型"),
    ("③ 服务端配置", "使用服务端已配置的模型"),
    ("④ 无可用 LLM", "当前没有可用的 LLM"),
):
    check(kw in src, f"{label} 有明确提示：{kw}")

# ① 页面填写：**刻意不弹状态卡**（2026-10-30 用户要求去掉那一行）。
#   三个框就在正上方，再复述"模型来源：你在上面填写的 + 模型名 + 地址"是噪音。
#   这条反向断言守的是"别再被加回来"。
check("模型来源：你在上面填写的" not in src,
      "① 页面填写不重复弹状态卡（那一行已删除）")

print("\n===== D 降级说明必须讲清后果 =====")
i_warn = src.find("当前没有可用的 LLM")
seg = src[i_warn:i_warn + 1800] if i_warn > 0 else ""
check("自动降级" in seg and "完整套链路" in seg, "说明「仍会跑完整套链路」（不中断）")
check("调度决策" in seg, "点明「调度决策」会退步")
check("归因解释" in seg, "点明「归因解释」会退步")
check("读图" in seg, "点明「读图」会退步")
check("准确度" in seg or "准确" in seg, "明确说准确度会下降（用户要求的原话）")
check("启动" in seg and "书生本地模型" in seg, "给出恢复办法①：启动本地模型")
check("Base URL" in seg or "Base" in seg, "给出恢复办法②：填云端 API")

print("\n===== E 状态提示渲染在输入区之后（即「输入框下方」）=====")
i_input = src.find("api_key = st.text_input(")
i_status = src.find("LLM 可用性：三级优先级的结论")
check(i_input > 0 and i_status > 0, "输入框与状态提示都在页面里")
check(i_status > i_input, "状态提示在输入框之后渲染（用户要求放在下方）",
      f"input={i_input} status={i_status}")

print("\n===== F 无 LLM 时不留空壳 =====")
check('if _llm_src == "none":' in src, "无 LLM 时走独立分支（不硬套模态判定）")
check("_vc = {\"ok\": False" in src, "无 LLM 时读图通道判定为不可用（不是空 dict）")
check("vision_max_regions" in src and "vision_budget" in src, "读图预算变量在两个分支外仍有定义")

print(f"\n===== 汇总：{_ok + _bad} 项，通过 {_ok}，未通过 {_bad} =====")
if _bad:
    print("有未通过项，见上面 FAIL 行")
    sys.exit(1)
sys.exit(0)
