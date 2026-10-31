"""守「那一排开关已经整块删掉」这件事（离线，秒级）。

背景（2026-10-27 用户定版）
--------------------------
页面上原本有一排开关：使用内置演示数据 / 自主模式总闸 + 六处分项 /
读图区域上限 / 读图耗时预算。用户要求**全部从页面移除**，改成「默认全开，
页面上只留填 API 的那一栏」。

于是原来的三条真机测试失去了对象：
  · tests/verify_autonomy_ui.py      —— 点总闸带开六个分项
  · tests/verify_multagent_ui.py     —— 关掉总闸再只开多智能体，比对撞区在不在
  · tests/verify_closeloop_ui.py     —— 打开闭环开关，看闭环面板在不在
开关没了，「关掉再看看」这条路就不存在了。它们已退役（见 .trash_tmp/），
由本文件守新口径。

守什么
------
  A 页面上没有开关：源码里选项区一段不许出现 `st.toggle` / `st.checkbox`
    （2026-10-30 起连「该模型本身可以读图」那个勾选框也删了 —— 改由模型名自动识别，
    见 A2）；
  A2 密钥区**只有一组**：Key / Base URL / 模型名三个框，没有第二组视觉密钥；
    多模态结论必须来自 `looks_multimodal` 的自动判定，且两种结论都配有
    「支持的输入类型 + 使用注意事项」的提示文案；
  B 六处分项恒为开：必须是模块级常量，不许读 env、不许读 session_state；
  C 六个值真的传到了后端：`_audit_kwargs` / `_mk` 里逐个出现；
  D 后端默认值仍是 False：CLI 与验证脚本的老行为不受影响（界面全开 ≠ 库里全开）；
  E 演示数据不再有开关，但「没上传时用内置样例」这条兜底还在。

用法: python tests/verify_switches_removed.py
退出码: 0 全过 / 1 有异常
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "app.py").read_text(encoding="utf-8")

N = 0
BAD: list[str] = []


def check(ok: bool, label: str) -> None:
    global N
    N += 1
    ok = bool(ok)
    print(("  ✅ " if ok else "  ❌ ") + label)
    if not ok:
        BAD.append(label)


#: 选项区 = 从「选项区」注释到「状态提示与高级设置」之间那一段
_OPT = APP.split("# ---- 选项区")[-1].split("# ---- 状态提示")[0]
_ADV = APP.split("高级设置 · 模型 API")[-1]

_SIX = ("use_multagent", "use_closeloop", "use_auto_loop",
        "use_role_llm", "judge_llm", "use_exp_llm")


def main() -> int:
    print("===== A 页面上不再有那一排开关 =====")
    check("st.toggle(" not in _OPT, "选项区里一个 toggle 都没有")
    check("st.checkbox(" not in _OPT, "选项区里没有 checkbox")
    check("使用内置演示数据" not in _OPT, "「使用内置演示数据」字样已从选项区删除")
    check("自主模式" not in _OPT, "「自主模式」总闸已删除")
    check("读图区域上限" not in APP, "读图上限输入框已删除")
    check("读图耗时预算" not in APP, "读图耗时预算输入框已删除")
    check("高级设置 · 模型 API" in APP, "还留着填 API 的那一栏")
    check("模型 API Key" in _ADV and "模型 Base URL" in _ADV,
          "填写 API 的三件（Key / Base URL / 模型名）都在")
    check("st.checkbox(" not in _ADV, "高级设置里连多模态勾选框也没有（全靠自动识别）")

    print("\n===== A2 密钥区只有一组 + 多模态按模型名自动识别 =====")
    check("视觉模型 API Key" not in APP, "第二组视觉密钥输入框已删除")
    check("视觉 Base URL" not in APP, "第二组视觉 Base URL 输入框已删除")
    _n_pw = _ADV.count('type="password"')
    check(_n_pw == 1, f"高级设置里只有 1 个密码框（实际 {_n_pw} 个）")
    check("llm_multimodal = _mm_auto" in _ADV,
          "多模态结论直接取自动识别结果（没有第二个真相）")
    check("_looks_multimodal(_lm_eff" in _ADV, "判定入口是 looks_multimodal（按模型名）")
    check("_modality_hit(_lm_eff)" in _ADV, "页面上如实给出识别依据（命中的视觉标识）")
    check("已识别为多模态模型" in _ADV and "已识别为单模态模型" in _ADV,
          "多模态 / 单模态两种结论各有对应提示")
    check("文字 + 图片" in _ADV and "只有文字" in _ADV,
          "提示里写明各自**支持的输入类型**")
    check("要注意" in _ADV, "提示里写明**使用注意事项**")

    print("\n===== B 六处分项恒为开（不再看 env / session） =====")
    for name in _SIX:
        check(re.search(rf"^{name} = _AUTO_ON$", APP, re.M) is not None,
              f"{name} 写成模块级常量（恒开）")
    check(re.search(r"^_AUTO_ON = True$", APP, re.M) is not None, "_AUTO_ON 就是 True")
    check("PAPER2LAB_AUTO_MODE" not in _OPT,
          "选项区不再读 PAPER2LAB_AUTO_MODE（开关没了，env 也不该再管它）")

    print("\n===== C 六个值真的传到了后端 =====")
    for frag, label in (("judge_llm=bool(judge_llm)", "模型先判"),
                        ("experiment_llm=bool(use_exp_llm)", "实验站挑工具"),
                        ('_mk["role_llm"] = bool(use_role_llm)', "角色用模型"),
                        ('_mk["loop"] = bool(use_closeloop)', "复现闭环"),
                        ('_mk["auto_loop"] = bool(use_auto_loop)', "自主回头")):
        check(frag in APP, f"{label}这个开关真的传给了后端")

    print("\n===== D 后端默认值不变（CLI / 测试的老行为不受影响） =====")
    sys.path.insert(0, str(ROOT))
    import inspect

    from paper2lab.pipeline import run_audit          # noqa: E402

    sig = inspect.signature(run_audit)
    for pname in ("judge_llm", "experiment_llm", "loop", "auto_loop"):
        if pname in sig.parameters:
            check(sig.parameters[pname].default in (False, None),
                  f"run_audit 的 {pname} 默认仍是关的")
    try:
        from paper2lab.graph import run_audit_multagent as _ram
        sig2 = inspect.signature(_ram)
        for pname in ("role_llm", "loop", "auto_loop", "judge_llm"):
            if pname in sig2.parameters:
                check(sig2.parameters[pname].default in (False, None),
                      f"图入口的 {pname} 默认仍是关的")
    except Exception as exc:                        # noqa: BLE001
        check(False, f"图入口导入失败：{type(exc).__name__}")

    print("\n===== E 演示材料：开关没了，兜底还在 =====")
    check("use_demo = True" in APP, "没上传时仍用内置样例（不给勾，但不至于空白）")
    check("elif use_demo:" in APP, "演示分支的代码路径还在（只是不再由开关决定）")
    check("内置样例" in APP, "页面上如实写着「本次用内置样例」")

    print("\n" + "=" * 58)
    print(f"共 {N} 项：通过 {N - len(BAD)}，未通过 {len(BAD)}")
    for x in BAD:
        print("  ✗", x)
    return 0 if not BAD else 1


if __name__ == "__main__":
    sys.exit(main())
