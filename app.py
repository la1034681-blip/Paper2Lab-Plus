"""Paper2Lab —— 科研实验智能一致性审计平台 (产品网站风格 UI · 横向布局)。

运行: .venv/Scripts/python -m streamlit run app.py
"""

from __future__ import annotations

import hashlib
import html as _html
import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st

from paper2lab.envfile import get as env_get
from paper2lab.envfile import load_env
from paper2lab.llm import RunCancelled as _RunCancelled
from paper2lab.llm import begin_run as _begin_run
from paper2lab.llm import check_run as _check_run
from paper2lab.llm import endpoint_url_hint as _endpoint_url_hint
from paper2lab.llm import looks_multimodal as _looks_multimodal
from paper2lab.llm import modality_hit as _modality_hit
from paper2lab.llm import probe_local as _probe_local
from paper2lab.llm import resolve_endpoint as _resolve_endpoint
from paper2lab.llm import resolve_vision as _resolve_vision
from paper2lab.execview import agent_panel_title as _agent_panel_title
from paper2lab.execview import tool_outcome as _tool_outcome
from paper2lab.housekeeping import format_report as _purge_report
from paper2lab.housekeeping import purge_run_state as _purge_run_state
from paper2lab.housekeeping import session_clear as _session_clear
from paper2lab.housekeeping import session_lookup as _session_lookup
from paper2lab.models import Confidence, Finding, Status
# 2026-10-26：实验室指挥台（三栏大屏）。渲染器是纯函数、在 paper2lab/deck.py，
# 这里只负责注入一次样式 + 在「有结果」和「正在跑」两种时机各调一次。
from paper2lab.deck import DECK_CSS, deck_html
# 2026-10-27：指挥台「可点击版」—— 每一行 HTML 上盖一个透明按钮，
# 点了就能回到 Python 摊开该环节的详情（st.html 本身点不动）。
from paper2lab.deckui import FOCUS_KEY as _DK_FOCUS
from paper2lab.deckui import render_deck as _render_deck
from paper2lab.deckui import render_detail as _render_detail
from paper2lab.deckui import shell_css as _deck_shell_css
# 问答窗口（2026-10-27）: 入口 + 弹窗都在 paper2lab/qaui，口径在 paper2lab/qa。
# 单独成模块的理由：问答的「问什么、拿哪些事实去问」是口径，界面换了口径不该跟着改。
from paper2lab.qaui import HIST_KEY as QA_HIST
from paper2lab.qaui import OPEN_KEY as QA_OPEN
from paper2lab.qaui import render_dialog as _qa_dialog
from paper2lab.qaui import render_entry as _qa_entry
from paper2lab.agent_roles import POSITIONS
from paper2lab.pipeline import STEPS, TOOLS, run_audit
# 阶段二 2D：② 实验站的观察态常量 —— 界面 / 报告 / 测试都读这一份, 不写字面量
from paper2lab.experiment import (EXP_AGREE, EXP_DISAGREE, EXP_MACHINE,
                                  EXP_MEASURED, EXP_NO_NEED, EXP_UNDECIDED)
from paper2lab.toolbox import STATION_DESC, STATION_OF_STEP, STATIONS

load_env()   # 读取项目根目录 .env(API key 等本地私有配置)

BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / "demo"

st.set_page_config(page_title="Paper2Lab · 科研一致性审计平台",
                   page_icon="🔬", layout="wide",
                   # 2026-10-26 随 B 方案改版从 collapsed 改成 expanded：
                   # 左侧导航现在是主要入口（总览 / 参数对表 / 假设卡 / 实验卡 / …），
                   # 默认收起的话用户根本找不到「去哪看」—— 那不是省空间，是把门关上。
                   initial_sidebar_state="expanded")

# ================================================================ 访问口令门
# 部署到公网时用环境变量 PAPER2LAB_ACCESS_CODE 设一道统一口令: 拿到链接 ≠ 能跑审计
# （审计会消耗 API 额度）。**不设这个变量就完全不生效** —— 本地照旧一字不差。
from paper2lab.access import configured_code as _gate_code   # noqa: E402
from paper2lab.access import verify_code as _gate_verify     # noqa: E402

if _gate_code() and not st.session_state.get("_access_ok"):
    st.markdown(
        "<div style='max-width:420px;margin:14vh auto 0;text-align:center'>"
        "<div style='font-size:34px'>🔒</div>"
        "<div style='font-size:20px;font-weight:800;color:#0f172a;margin:.4rem 0'>"
        "Paper2Lab · 访问口令</div>"
        "<div style='color:#5a6b88;font-size:13px'>本演示链接仅向评审开放，请输入邀请码</div>"
        "</div>", unsafe_allow_html=True)
    _c1, _c2, _c3 = st.columns([1, 1.2, 1])
    with _c2:
        _entered = st.text_input("访问口令", type="password", key="_gate_input",
                                 placeholder="请输入邀请码", label_visibility="collapsed")
        if st.button("进 入", type="primary", use_container_width=True):
            if _gate_verify(_entered):
                st.session_state["_access_ok"] = True
                st.rerun()
            else:
                st.error("口令不正确。")
    st.caption("　")
    st.stop()

# ================================================================ 会话绑定（多用户隔离）
# 每次脚本执行**都要**把本会话 id 绑到当前线程 —— 位置必须在任何业务代码之前。
#
# 为什么: 公开链接是**一个进程服务所有访问者**。会话 id 的真源是 st.session_state
# （Streamlit 保证每个浏览器会话各一份, 不会串）; 而 session 模块（缓存/记忆/轨迹）
# 刻意不依赖 streamlit, 靠一个"当前是谁"的入口取值。
#   旧实现把这个入口放在**进程级全局变量**里 —— 单人单机看不出问题, 多用户就串:
#   A 点过运行之后全局 id 是 A, B 刷新时若没走到绑定分支就会读到 A 的会话。
#   现在 session 侧改成**线程局部**（见 paper2lab/session.py）, 这里在每次执行开头写一次。
# 没绑定过的线程 → 读到空串 → session.get() 返回 None → 走"无缓存"降级路径:
# 宁可这次不缓存, 也绝不读到别人的会话（记忆与轨迹是隐私级数据）。
if st.session_state.get("_p2l_sid"):
    from paper2lab.session import set_current_id as _set_current_id

    _set_current_id(st.session_state["_p2l_sid"])

# ================================================================ 全局样式
st.markdown("""
<style>
/* ============================================================
   视觉基调（2026-10-26 UI 改版 · B「证据审阅台」）
   口径：这是一个**校对工具**，不是营销落地页。
   所以整页浅底、白卡、细线、圆角收敛；只有「就绪度」那一块是深色，
   让眼睛第一眼落在分数上，其余全是可核对的内容。
   —— 所有 class 名保持不变，只换观感，不动任何渲染逻辑与文案。
   ============================================================ */
.stApp {background: #f6f8fb;}
.block-container {padding: 0 3rem 2.4rem 3rem; max-width: 1320px;}
#MainMenu, footer, header {visibility: hidden;}
div[data-testid="stVerticalBlock"] {gap: .6rem;}
html, body, .stApp {color: #1b2436;
    font-family: "Microsoft YaHei","PingFang SC","Hiragino Sans GB",-apple-system,"Segoe UI",sans-serif;}

/* ---- 顶部导航栏：白底细线，像工具不���官网 ---- */
.navbar {
    position: sticky; top: 0; z-index: 999;
    margin: 0 -3rem; padding: 0 3rem; height: 54px;
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,.94); backdrop-filter: blur(10px);
    border-bottom: 1px solid #e8edf5;
}
.nav-logo {color: #12203a; font-size: 16px; font-weight: 800; letter-spacing: .2px;}
.nav-logo b {color: #2563eb;}
.nav-links {display: flex; align-items: center;}
.nav-links span {color: #6b7a95; font-size: 12.5px; margin-left: 20px; font-weight: 500;}
.nav-cta {
    color: #fff !important; font-size: 12.5px; font-weight: 700; margin-left: 20px;
    background: #12203a; border-radius: 7px; padding: 6px 14px;
}

/* ---- Hero：浅色、留白、一句话说清是什么 ---- */
.hero {
    margin: 0 -3rem; padding: 40px 3rem 30px 3rem;
    background: #fff;
    border-bottom: 1px solid #e8edf5;
}
.hero-kicker {color:#2563eb; font-size:11px; font-weight:800; letter-spacing:2.6px;}
.hero-title {color:#12203a; font-size:38px; font-weight:800; margin:8px 0 10px 0; letter-spacing:-.4px;}
.hero-title em {font-style:normal; color:#2563eb;}
.hero-sub {color:#5a6b88; font-size:14px; max-width:760px; line-height:1.8;}
.hero-stats {display:flex; gap:10px; margin-top:26px; flex-wrap:wrap;}
.hero-stat {
    flex:1; min-width:150px; background:#fafbfe; border:1px solid #e8edf5;
    border-radius:10px; padding:12px 15px;
}
.hero-stat b {color:#12203a; font-size:20px; font-weight:800; font-variant-numeric:tabular-nums;}
.hero-stat span {color:#7c8aa3; font-size:12px; display:block; margin-top:3px; line-height:1.5;}

/* ---- 区块标题 ---- */
.sec {display:flex; align-items:baseline; gap:11px; margin:30px 0 12px 0;
      padding-bottom:9px; border-bottom:1px solid #e8edf5;}
.sec-h {font-size:18px; font-weight:800; color:#12203a; letter-spacing:-.2px;}
.sec-n {font-size:11px; font-weight:800; color:#2563eb; letter-spacing:1.6px;
        background:#eef4ff; border-radius:5px; padding:2px 7px;}
.sec-sub {color:#8b9ab2; font-size:12px;}

/* ---- 三个上传入口 ---- */
.entry {
    background:#fff; border:1px solid #e8edf5; border-radius:11px;
    padding:15px 16px 6px 16px; transition: border-color .18s, box-shadow .18s;
}
.entry:hover {border-color:#c9d8ef; box-shadow:0 6px 18px rgba(24,38,68,.07);}
.entry-head {display:flex; align-items:center; gap:10px;}
.entry-ic {width:34px;height:34px;border-radius:9px;background:var(--c);color:#fff;
           display:flex;align-items:center;justify-content:center;font-size:17px;}
.entry-t {font-size:15px;font-weight:800;color:#12203a;}
.entry-tag {font-size:10px;font-weight:700;color:var(--c);background:var(--cl);
            border-radius:99px;padding:2px 8px;margin-left:auto;}
.entry-d {font-size:12px;color:#7c8aa3;line-height:1.65;margin:8px 0 6px 0;}

/* ---- 主按钮 / 下载按钮 ---- */
div.stButton > button[kind="primary"] {
    background: #12203a; border:none;
    font-size:16px; font-weight:800; padding:.62rem 0; border-radius:9px;
    box-shadow: none; letter-spacing:.5px; transition: background .18s;
}
div.stButton > button[kind="primary"]:hover {background:#1e3358;}
div.stDownloadButton > button {
    background: #1f8a54; color:#fff; border:none;
    font-weight:800; border-radius:9px; padding:.55rem 1.2rem;
}

/* ---- 四站分区带 ---- */
.station-row {display:flex; align-items:stretch; margin:4px 0 2px 0;
              background:#fff; border:1px solid #e8edf5; border-radius:11px; padding:12px 14px 10px;}
.station {display:flex; flex-direction:column; min-width:88px; padding:0 5px;}
.station-head {font-size:11px; font-weight:800; letter-spacing:.2px; text-align:left;
               padding:0 0 8px 0; border-top:none; color:#8b9ab2;
               display:flex; align-items:center; gap:5px; white-space:nowrap;}
.station-head::before {content:""; width:6px; height:6px; border-radius:2px; background:#c8d3e3;}
.station.s1 .station-head {color:#2563eb;} .station.s1 .station-head::before {background:#2563eb;}
.station.s2 .station-head {color:#b45309;} .station.s2 .station-head::before {background:#d97706;}
.station.s3 .station-head {color:#6d28d9;} .station.s3 .station-head::before {background:#7c3aed;}
.station.s4 .station-head {color:#0e7490;} .station.s4 .station-head::before {background:#0891b2;}
.station-n {font-size:10px; font-weight:700; color:#a8b4c8; background:#f2f5fa;
            border-radius:4px; padding:0 5px; margin-left:0; flex:none;}
.station-body {display:flex; align-items:flex-start;}
.station-empty {font-size:10px; line-height:1.5; color:#b45309; text-align:center;
                border:1px dashed #f0c48a; background:#fffaf2; border-radius:7px;
                padding:6px 4px; margin:0 2px; flex:1;}

/* ---- 横向流程步骤条 ---- */
.stepper {display:flex; align-items:flex-start; margin:10px 0 2px 0;}
.step {flex:1; text-align:center; position:relative;}
.step::before {content:""; position:absolute; top:12px; left:-50%; width:100%;
               height:2px; background:#e8edf5; z-index:0;}
.step:first-child::before {display:none;}
.step.done::before {background:#c3d6f5;}
.step-dot {width:24px;height:24px;border-radius:50%;margin:0 auto;position:relative;z-index:1;
           background:#fff;border:2px solid #dde5f0;color:#a8b4c8;font-size:11px;
           font-weight:800;display:flex;align-items:center;justify-content:center;}
.step.done .step-dot {background:#12203a;border-color:#12203a;color:#fff;}
.step-t {font-size:11px;color:#8b9ab2;margin-top:6px;font-weight:600;line-height:1.35;}
.step.done .step-t {color:#3b4d6b;}
.step.skipped::before {background:repeating-linear-gradient(90deg,#e6e9f0 0 5px,transparent 5px 9px);}
.step.skipped .step-dot {background:#fff;border-color:#e2b98a;color:#b87333;border-style:dashed;}
.step.skipped .step-t {color:#a9855f;}
.step-tag {font-size:10px;color:#9a5b28;background:#fdf3e7;border:1px solid #f2dcc0;
           border-radius:4px;padding:1px 5px;display:inline-block;margin-top:3px;font-weight:700;}
.step.failed .step-dot {background:#fdeceb;border-color:#f0a6a6;color:#c02b2b;}
.step.failed .step-t {color:#c02b2b;}

/* ---- 就绪度横向面板（全页唯一深色块 = 视觉落点） ---- */
.ready-band {
    background:#12203a;
    border-radius:13px; padding:22px 26px; display:flex; align-items:center; gap:30px;
}
.ready-score {text-align:center; min-width:148px;}
.ready-num {font-size:50px; font-weight:800; line-height:1; font-variant-numeric:tabular-nums;}
.ready-num small {font-size:17px; color:rgba(255,255,255,.4); font-weight:600;}
.ready-lab {color:rgba(255,255,255,.6); font-size:12px; font-weight:700; letter-spacing:1.6px; margin-top:5px;}
.ready-conf {display:inline-block;margin-top:9px;font-size:11px;font-weight:700;
             border:1px solid;border-radius:99px;padding:3px 11px;background:rgba(255,255,255,.05);}
.ready-dims {flex:1; display:grid; grid-template-columns:repeat(5,1fr); gap:11px;}
.rdim {background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.09);
       border-radius:10px; padding:11px 12px 10px 12px;}
.rdim-n {color:rgba(255,255,255,.62); font-size:11px; font-weight:700;}
.rdim-v {font-size:21px; font-weight:800; margin:3px 0 7px 0; font-variant-numeric:tabular-nums;}
.rdim-bar {height:4px;border-radius:3px;background:rgba(255,255,255,.13);overflow:hidden;}
.rdim-bar > div {height:100%;border-radius:3px;}
.rdim-v.rdim-na {color:rgba(255,255,255,.4); font-size:14px; font-weight:700;}
.rdim-na-bar {width:100%;height:100%;background:transparent;
              border:1px dashed rgba(255,255,255,.3); box-sizing:border-box;}

/* ---- 审计发现 ---- */
.finding {
    background:#fff; border:1px solid #e8edf5; border-radius:11px;
    padding:14px 18px 16px 18px; margin-bottom:10px;
}
.risk-bar {height:3px;border-radius:11px 11px 0 0;margin:-14px -18px 12px -18px;
           background:var(--rc);}
.f-head {display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:11px;}
.f-name {font-size:18px;font-weight:800;color:#12203a;margin-right:2px;letter-spacing:-.2px;}
.chip {display:inline-block;font-size:12px;font-weight:700;border-radius:6px;
       padding:3px 10px;color:#fff;background:var(--cc,#888);}
.chip-ghost {background:#eef2f9;color:#46587a;}
.f-evi {display:grid;grid-template-columns:1.15fr 1.15fr 0.9fr;gap:9px;}
.evi {background:#fafbfe;border:1px solid #eaeff7;border-radius:8px;
      padding:10px 13px;font-size:14px;line-height:1.55;color:#26334d;}
.evi-loc {font-size:12px;font-weight:800;color:#2563eb;margin-bottom:5px;}
.evi-val {margin-top:6px;font-size:13px;color:#3a4a6b;}
.evi-none {color:#94a2bd;}
.evi b {font-weight:800;}
.ai-box {background:#f4f7fe;
         border:1px solid #dde7fa;border-radius:8px;padding:10px 14px;
         font-size:14px;color:#2c3a63;margin-top:10px;line-height:1.65;}
.vfy {margin-top:10px;font-size:13px;font-weight:700;border-radius:7px;padding:8px 14px;}
.vfy-ok {background:#eef8f2;border:1px solid #c6e6d4;color:#1f7a45;}
.vfy-bad {background:#fdf0ef;border:1px solid #f5cccc;color:#a32d2d;}
.mc-item {background:#fff;border:1px solid #e8edf5;border-left:3px solid var(--mc,#f76b15);
          border-radius:8px;padding:10px 14px;margin-bottom:8px;}
.mc-head {font-size:14px;font-weight:800;color:#12203a;}
.mc-line {font-size:13px;color:#4a5b7a;margin-top:4px;line-height:1.6;}
.why {margin-top:10px;background:#fffbf0;border:1px solid #f2e4c4;border-radius:7px;
      padding:8px 13px;font-size:13px;color:#6b5320;line-height:1.6;}
@media (max-width:1150px){ .f-evi{grid-template-columns:1fr;} }

/* ---- ① 假设卡 ---- */
.hyp {background:#fafbfe;border:1px solid #eaeff7;border-radius:9px;
      padding:11px 14px;margin-bottom:8px;}
.hyp-h {display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:7px;}
.hyp-r {font-size:12px;font-weight:800;color:#fff;background:#12203a;
        border-radius:5px;padding:1px 8px;}
.hyp-k {font-size:15px;font-weight:800;color:#12203a;}
.hyp-c {font-size:14px;line-height:1.6;color:#26334d;}
.hyp-f {margin-top:7px;background:#fffbf0;border:1px solid #f2e4c4;border-radius:7px;
        padding:7px 12px;font-size:13px;color:#6b5320;line-height:1.6;}
.hyp-w {font-size:12px;color:#8b9ab2;margin-top:6px;line-height:1.5;}
.hyp-d {margin-top:8px;background:#f4f7fe;border:1px solid #dde7fa;border-radius:7px;
padding:8px 12px;font-size:13px;color:#28406e;line-height:1.65;}
.hyp-d b {color:#12203a;}
.hyp-d-ex {background:#f4f6f9;border-color:#e2e8f0;color:#6b7a92;}
.hyp-d-fg {background:#fdf1f0;border-color:#ffd6d3;color:#9b2c2c;}

/* ---- ② 实验卡 ---- */
.exp-d {margin-top:7px;font-size:13px;color:#4a5a78;line-height:1.6;}
.exp-r {margin-top:6px;background:#f4f8fe;border:1px solid #dde8f9;border-radius:7px;
        padding:6px 11px;font-size:13px;color:#2b3a5c;line-height:1.55;}
.exp-r b {color:#12203a;}
.exp-ms {float:right;font-size:11px;color:#8b9ab2;}
.exp-v {margin-top:6px;background:#f1fbf5;border:1px solid #cfeedd;border-radius:7px;
        padding:7px 12px;font-size:13px;color:#245c42;line-height:1.65;}
.exp-v b {color:#123a2a;}

/* ---- 自主运行面板（新增 · D 方案的信息形态）---- */
.auto-panel {background:#fff;border:1px solid #e8edf5;border-radius:11px;padding:16px 18px;}
.auto-hd {display:flex;align-items:center;gap:9px;margin-bottom:13px;}
.auto-hd .t {font-size:14px;font-weight:800;color:#12203a;}
.auto-hd .pill {font-size:11px;font-weight:800;border-radius:5px;padding:2px 8px;}
.auto-row {display:flex;gap:10px;flex-wrap:wrap;}
.auto-cell {flex:1;min-width:132px;background:#fafbfe;border:1px solid #eaeff7;
            border-radius:9px;padding:10px 12px;}
.auto-cell .l {font-size:11px;color:#8b9ab2;font-weight:700;}
.auto-cell .v {font-size:15px;font-weight:800;color:#12203a;margin-top:3px;
               font-variant-numeric:tabular-nums;}
.auto-cell .s {font-size:11px;color:#7c8aa3;margin-top:2px;line-height:1.5;}
.auto-dec {margin-top:11px;background:#f4f7fe;border:1px solid #dde7fa;border-radius:9px;
           padding:10px 13px;font-size:13px;color:#28406e;line-height:1.7;}
.auto-dec b {color:#12203a;}
.auto-gate {display:inline-block;font-size:11px;font-weight:800;border-radius:4px;
            padding:1px 7px;margin-right:5px;}

/* ---- B 方案「证据审阅台」：KPI 大数卡 ---- */
.kpis {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:2px 0 16px 0;}
.kpi-c {background:#fff;border:1px solid #e8ebf0;border-radius:10px;padding:15px 17px 14px 17px;}
.kpi-c .v {font-size:30px;font-weight:800;letter-spacing:-.8px;color:#1a1f2b;
           line-height:1.05;font-variant-numeric:tabular-nums;}
.kpi-c .l {font-size:12px;color:#6b7280;margin-top:5px;font-weight:500;}
.kpi-c.g .v {color:#12a150;}
.kpi-c.r .v {color:#e5484d;}
.kpi-c.o .v {color:#f5a524;}
.kpi-c.b .v {color:#2563eb;}
@media (max-width:1150px){ .kpis{grid-template-columns:repeat(2,minmax(0,1fr));} }

/* ---- 左侧导航（B 方案）：计数徽标 + 分组小标题 ---- */
section[data-testid="stSidebar"] {background:#fbfcfe;border-right:1px solid #e8ebf0;}
section[data-testid="stSidebar"] .nav-grp {font-size:11px;font-weight:800;color:#a3a8b4;
    letter-spacing:.6px;margin:16px 0 4px 4px;text-transform:uppercase;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap:2px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding:7px 10px;border-radius:8px;transition:background .12s;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background:#eef2f9;}
section[data-testid="stSidebar"] label p {font-size:14px !important;color:#26334d;}
/* 选中项：浅蓝底 + 左侧蓝条 + 加粗（B 方案里导航项的"当前在这里"感） */
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background:#eaf1ff; box-shadow: inset 2px 0 0 #2563eb;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color:#1d4ed8 !important; font-weight:700 !important;}

/* ---- 对表抽屉：右栏「论文原文 / 代码 / 结论」三段 ---- */
.drw-t {font-size:15px;font-weight:800;color:#12203a;line-height:1.45;}
.drw-s {font-size:12px;color:#8b9ab2;margin-top:4px;line-height:1.5;}
.drw-p {margin-top:11px;border:1px solid #e8ebf0;border-radius:9px;overflow:hidden;}
.drw-ph {background:#f6f8fc;font-size:11px;font-weight:800;color:#5a6b8a;
         letter-spacing:.8px;padding:6px 12px;border-bottom:1px solid #e8ebf0;}
.drw-pb {padding:11px 13px;font-size:13px;line-height:1.72;color:#26334d;
         word-break:break-word;}
.drw-hl {background:#fff1b8;border-radius:3px;padding:0 3px;font-weight:800;color:#6b4a00;}
.drw-hl2 {background:#ffe0de;border-radius:3px;padding:0 3px;font-weight:800;color:#9b2c2c;}
.drw-v {margin-top:12px;background:#f4f7fe;border:1px solid #dde7fa;border-radius:9px;
        padding:11px 14px;font-size:13px;line-height:1.75;color:#28406e;}
.drw-v b {color:#12203a;}
.drw-empty {border:1px dashed #d8dfea;border-radius:9px;padding:26px 18px;text-align:center;
            color:#95a2b8;font-size:13px;line-height:1.8;background:#fcfdff;}
</style>
""", unsafe_allow_html=True)

# ================================================================ 常量
STATUS_ICON = {
    Status.CONSISTENT: "🟢", Status.INCONSISTENT: "🔴",
    Status.MISSING_IN_CODE: "🟡", Status.MISSING_IN_PAPER: "🟡",
    Status.NOT_FIXED: "🟠", Status.UNVERIFIABLE: "🔵",
    Status.INTERNAL_INCONSISTENT: "🟣",
}
STATUS_ZH = {
    Status.CONSISTENT: "一致", Status.INCONSISTENT: "不一致",
    Status.MISSING_IN_CODE: "代码未找到", Status.MISSING_IN_PAPER: "论文未声明",
    Status.NOT_FIXED: "未固定", Status.UNVERIFIABLE: "动态确定·无法静态验证",
    Status.INTERNAL_INCONSISTENT: "内部不一致",
}
RISK_META = {"high": ("高风险", "#e5484d"), "medium": ("中风险", "#f76b15"),
             "low": ("低风险", "#3d63dd"), "none": ("通过", "#30a46c")}
CONF_ZH = {Confidence.CONFIRMED: "🟢 Confirmed · 规则验证",
           Confidence.INFERRED: "🔵 Inferred · AI 推断"}


# ================================================================ 工具
def _save_upload(uploaded, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


def _cell(v, dash: str = "—") -> str:
    """把一个单元格值转成**纯字符串**（空/None → dash）。

    为什么必须有这个（2026-10-15 修的真实崩溃）:
      st.dataframe 底层是 pyarrow, 一列里只要**混了两种类型**（比如 int 页码 +
      字符串占位符）就直接抛
        ArrowTypeError: Expected bytes, got a 'int' object
      而 Streamlit 里单个组件抛异常 → **整页往下的内容全部渲染中断**,
      用户看到的就是「点了按钮一直转、结果区不出来」, 日志里只有一串 pyarrow 栈。

      典型的触发写法是 `s.get("page") or "——"`：page 是 int 12 时填 int,
      page 缺失时才填字符串 —— 同一列两种类型, 必崩。

    所以：**凡是喂给 st.dataframe 的值, 一律先过这个函数**。
    """
    if v is None or v == "":
        return dash
    if isinstance(v, bool):
        return "是" if v else "否"
    return str(v)


def _table(rows, **kw):
    """渲染一张表 —— **所有列强制转字符串**, 从根上堵住 pyarrow 的类型崩溃。

    背景（2026-10-15 真实故障）:
      上传 01_DDPM 后, 服务日志里出现
        pyarrow.lib.ArrowTypeError: Expected bytes, got a 'int' object
        Conversion failed for column 页 with type object
      来源是 `s.get("page") or "——"` 这种写法: page 是 int 时填 int、缺失时填 str,
      同一列两种类型 → pyarrow 直接抛。而 Streamlit 里**单个组件抛异常会让
      整页往下全部渲染中断**, 用户看到的就是「点了按钮一直不出结果」。

    为什么用「全转字符串」而不是「只转有问题的那列」:
      混合类型只在上游某个字段换了类型时才暴露(页码有时 int、有时缺失),
      等它暴露就已经是一次线上故障。这里代价是 0 —— 表格本来就是给人读的,
      排序/对齐由 Streamlit 自己按字符串做, 不影响任何结论。

    rows 为空时也照样渲染(给一句空话), 免得界面少一块、看不出"是没数据还是崩了"。
    """
    if not rows:
        st.caption(kw.pop("empty_text", "（本次没有可展示的条目）"))
        return
    safe = [{k: _cell(v) for k, v in r.items()} for r in rows]
    st.dataframe(safe, **kw)          # ⚠ 这里必须是 st.dataframe, 不能写成 _table
                                       #   —— 批量替换时误改过一次, 直接无限递归到
                                       #   RecursionError（实测崩在 app.py:323）。


def _score_color(v: int) -> str:
    return "#34d399" if v >= 85 else ("#fbbf24" if v >= 70 else "#f87171")


def _step_status_map(ex: dict) -> dict:
    """环节名 → (状态, 说明)。一个工具可能覆盖多个环节, 逐个摊开。"""
    out: dict = {}
    for s in (ex or {}).get("steps", []):
        for cp in (s.get("checkpoints") or []):
            out[cp] = (s.get("status") or "", s.get("reason") or "")
    return out


def _attr(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") \
                       .replace('"', "&quot;").replace("'", "&#39;")


def _stepper_html(execution_log: dict | None = None) -> str:
    """流程进度条（节点数由 `STEPS` 决定，**不写死数字**）。

    **按实际执行渲染**：做过的打 ✓；计划跳过的**标「未执行」并给出理由**（悬停可见）。
    跳过的工具对应环节不勾、并在面板上标「未执行」——计划跑什么，面板就显示什么。

    **按四站分区**（阶段一 1A · 2026-10-15）：进度条不再是一条平铺的链，而是按
    「① 假设 / ② 实验 / ③ 验证 / ④ 迭代」分组显示，一眼看得出此刻在哪一站。
    四站现在都有真实环节（② 实验站的「实验设计」在阶段二 2D 落地后补上）。

    注：早先这里写死"十环节"，2026-09-28 把「根因追查 / 补丁草案」也登记成工具后
    （节点 10 → 12）就地对不上了。现一律改用 `len(STEPS)`/`len(TOOLS)` 派生，
    数字随工具箱自动走，杜绝再次漂移。
    """
    st_map = _step_status_map(execution_log or {})

    def _cell(s: str) -> str:
        status, reason = st_map.get(s, ("", ""))
        if not st_map:
            cls, mark, tag, tip = "done", "✓", "", ""
        elif status == "skipped":
            cls, mark, tag, tip = "skipped", "—", "未执行", reason or "计划跳过"
        elif status == "failed":
            cls, mark, tag, tip = "failed", "!", "执行失败", reason or "执行失败"
        else:
            cls, mark, tag, tip = "done", "✓", "", ""
        title = f" title='{_attr(tip)}'" if tip else ""
        tag_html = f"<div class='step-tag'>{tag}</div>" if tag else ""
        return (f"<div class='step {cls}'{title}>"
                f"<div class='step-dot'>{mark}</div>"
                f"<div class='step-t'>{s}</div>{tag_html}</div>")

    # 沿 STEPS 原序按四站分组（只分组，不重排、不增删）
    grouped: dict = {st: [] for st in STATIONS}
    for s in STEPS:
        grouped.setdefault(STATION_OF_STEP.get(s, ""), []).append(s)

    blocks: list[str] = []
    for idx, st in enumerate(STATIONS, start=1):
        cells = grouped.get(st) or []
        desc = _attr(STATION_DESC.get(st, ""))
        if cells:
            body = "".join(_cell(s) for s in cells)
            flex = len(cells)      # 宽度按站内环节数分配 → 与下方格子天然对齐
        else:
            # 该站暂无环节：如实标「规划中」，给最小宽度免得被挤没
            body = ("<div class='station-empty' title='%s'>规划中<br>"
                    "（2D 落地后出现）</div>" % desc)
            flex = 1.3
        # 站名右侧的「N 格」由 cells 派生，不写死 —— 加环节时自动跟着走。
        n_tag = f"<span class='station-n'>{len(cells)} 格</span>" if cells else ""
        blocks.append(
            f"<div class='station s{idx}' style='flex:{flex} 1 0%' title='{desc}'>"
            f"<div class='station-head'>{st}{n_tag}</div>"
            f"<div class='station-body'>{body}</div></div>")
    return f"<div class='station-row'>{''.join(blocks)}</div>"


def _auto_panel(audit) -> str:
    """自主运行面板：把「谁定的、做了什么、被谁拦下」摊开成一排计数。

    ⚠ 这里**一个数都不许编**：每一格都从 execution_log / hypotheses / findings /
    loop 里现场数出来；没有发生的事显示「未触发」，不显示 0 分或「全部通过」。
    """
    ex = getattr(audit, "execution_log", None) or {}
    steps = list(ex.get("steps") or [])
    ran = [s for s in steps if (s.get("status") or "") != "skipped"]
    skipped = [s for s in steps if (s.get("status") or "") == "skipped"]
    failed = [s for s in steps if (s.get("status") or "") == "failed"]
    planner = ((ex.get("plan") or {}).get("planner") or "")

    # 模型先判（两类判定目标合并统计）
    hyps = list(getattr(audit, "hypotheses", None) or [])
    finds = list(getattr(audit, "findings", None) or [])
    j_total = sum(1 for h in hyps if getattr(h, "llm_verdict", None)) \
        + sum(1 for f in finds if getattr(f, "llm_status", None))
    j_agree = sum(1 for h in hyps if getattr(h, "llm_agree", None) is True) \
        + sum(1 for f in finds if getattr(f, "llm_agree", None) is True)
    j_diff = sum(1 for h in hyps if getattr(h, "llm_agree", None) is False) \
        + sum(1 for f in finds if getattr(f, "llm_agree", None) is False)

    # 自主回头：模型提了几次、采纳几次、被闸门拦几次
    ad = (getattr(audit, "loop", None) or {}).get("auto_decisions") or []
    ad_ok = sum(1 for d in ad if (d.get("action") or "") == "retry")
    ad_gate = sum(1 for d in ad if (d.get("source") or "") == "gate")

    # ---- 四格计数
    if planner == "llm":
        c1_v, c1_s = "模型排的计划", "跑哪些环节由模型决定"
    elif planner:
        c1_v, c1_s = "规则兜底", "模型不可用，程序按全序兜底"
    else:
        c1_v, c1_s = "强制全跑", "本轮未走排计划"
    c2_v = f"{len(ran)} / {len(steps)}" if steps else "—"
    c2_s = (f"跳过 {len(skipped)} 件 · 失败 {len(failed)} 件" if steps
            else "无执行记录")
    if ad:
        c3_v = f"{len(ad)} 次决策"
        c3_s = f"采纳 {ad_ok} 次 · 闸门拦下 {ad_gate} 次"
    else:
        # 2026-10-28：开关已整块删除（六项恒开），所以不再说「处于关闭」——
        # 那句话会让用户以为有个关着的开关在那儿。
        c3_v, c3_s = "未触发", "本轮没有需要回头的条目"
    if j_total:
        c4_v = f"{j_total} 条先判"
        c4_s = f"与程序一致 {j_agree} 条 · 不一致 {j_diff} 条"
    else:
        # 同上：能力默认开，这里的「未启用」只可能是「本轮没有可判条目」。
        c4_v, c4_s = "未启用", "本轮没有可判条目（能力默认开）"

    cells = ""
    for lab, val, sub in (("计划主导方", c1_v, c1_s),
                          ("工序执行", c2_v, c2_s),
                          ("自主回头", c3_v, c3_s),
                          ("模型先判", c4_v, c4_s)):
        cells += (f"<div class='auto-cell'><div class='l'>{lab}</div>"
                  f"<div class='v'>{val}</div><div class='s'>{sub}</div></div>")

    # 闸门明细：只列真被拦下的，一条都没拦就不出现这一行（不凑数）
    gates = [d for d in ad if (d.get("source") or "") == "gate"]
    gline = ""
    if gates:
        bits = "".join(
            f"<span class='auto-gate' style='color:#a32d2d;background:#fdeceb'>"
            f"{(d.get('llm_action') or 'retry')} 被拒</span>"
            f"{(d.get('note') or '')[:60]}　" for d in gates[:3])
        gline = f"<div class='auto-dec'>闸门拦下的动作：{bits}</div>"

    return (f"<div class='auto-panel'>"
            f"<div class='auto-hd'><span class='t'>自主运行</span>"
            f"<span class='pill' style='color:#1f8a54;background:#eef8f2'>"
            f"模型定怎么查 · 程序定查到什么</span></div>"
            f"<div class='auto-row'>{cells}</div>{gline}</div>")


def _ready_band(total, cats: dict, conf: dict | None = None) -> str:
    """就绪度横条。`total=None` 表示**一个维度都没核验**(显示「—」, 不凭空给分)。
    `cats` 里值为 None 的维度同样表示「未核验」——不给分、不给彩色进度条。
    """
    dims = ""
    for name, v in cats.items():
        if v is None:
            # 未核验: 不给分、不给彩色进度条(避免被读成"核验过且通过")
            dims += (f"<div class='rdim'><div class='rdim-n'>{name}</div>"
                     f"<div class='rdim-v rdim-na'>未核验</div>"
                     f"<div class='rdim-bar'><div class='rdim-na-bar'></div></div></div>")
            continue
        c = _score_color(v)
        dims += (f"<div class='rdim'><div class='rdim-n'>{name}</div>"
                 f"<div class='rdim-v' style='color:{c}'>{v}</div>"
                 f"<div class='rdim-bar'><div style='width:{v}%;background:{c}'></div></div></div>")
    conf = conf or {}
    level = conf.get("level", "—")
    conf_color = {"高": "#34d399", "中": "#fbbf24", "低": "#f87171"}.get(level, "#94a3b8")
    if total is None:
        num_html = "<div class='ready-num' style='color:#94a3b8'>—<small> /100</small></div>"
    else:
        num_html = (f"<div class='ready-num' style='color:{_score_color(total)}'>"
                    f"{total}<small> /100</small></div>")
    return f"""
    <div class="ready-band">
      <div class="ready-score">
        {num_html}
        <div class="ready-lab">综合复现就绪度</div>
        <div class="ready-conf" style="border-color:{conf_color};color:{conf_color}">
          审计可信度 {level}　{conf.get('score', '—')}/100</div>
      </div>
      <div class="ready-dims">{dims}</div>
    </div>"""


def _render_finding(f: Finding):
    zh, color = RISK_META[f.risk.value]
    paper_html = (f"<div class='evi-loc'>📄 {f.paper.location}</div>{f.paper.snippet}"
                  f"<div class='evi-val'>声明值 <b>{f.paper.raw_value}</b>"
                  + (f"　·　归一化 <b>{f.paper.normalized}</b>" if f.paper.normalized is not None else "")
                  + "</div>"
                  if f.paper else
                  "<div class='evi-loc'>📄 论文侧</div><span class='evi-none'>论文中未声明该参数</span>")
    code_html = (f"<div class='evi-loc'>💻 {f.code.location}</div>{f.code.snippet}"
                 f"<div class='evi-val'>实现值 <b>{f.code.raw_value}</b>"
                 + (f"　·　归一化 <b>{f.code.normalized}</b>" if f.code.normalized is not None else "")
                 + "</div>"
                 if f.code else
                 "<div class='evi-loc'>💻 代码侧</div><span class='evi-none'>静态扫描未命中该参数</span>")
    # 第三列: 比对过程(让卡片右侧不留空白)
    if f.paper and f.code:
        if f.status == Status.INCONSISTENT:
            norm_html = (f"<div class='evi-loc'>🔍 比对结论</div>"
                         f"<b>{f.paper.normalized}</b> ≠ <b>{f.code.normalized}</b>"
                         f"<div class='evi-val'>判定 <b>不一致</b> · 差异可直接复算</div>")
        else:
            norm_html = (f"<div class='evi-loc'>🔍 比对结论</div>"
                         f"<b>{f.paper.normalized}</b> = <b>{f.code.normalized}</b>"
                         f"<div class='evi-val'>判定 <b>一致</b> · 规则验证通过</div>")
    elif f.status == Status.UNVERIFIABLE:
        norm_html = ("<div class='evi-loc'>🔍 比对结论</div>代码为运行时变量"
                     "<div class='evi-val'>判定 <b>动态确定</b> · 无法静态验证</div>")
    elif f.status == Status.INTERNAL_INCONSISTENT:
        norm_html = ("<div class='evi-loc'>🔍 比对结论</div>同一载体内多个取值"
                     "<div class='evi-val'>判定 <b>内部不一致</b></div>")
    elif f.status == Status.MISSING_IN_CODE:
        norm_html = ("<div class='evi-loc'>🔍 比对结论</div>论文有值 / 代码无证据"
                     "<div class='evi-val'>判定 <b>代码未找到</b></div>")
    elif f.status == Status.MISSING_IN_PAPER:
        norm_html = ("<div class='evi-loc'>🔍 比对结论</div>代码有值 / 论文未声明"
                     "<div class='evi-val'>判定 <b>论文未声明</b></div>")
    elif f.status == Status.NOT_FIXED:
        norm_html = ("<div class='evi-loc'>🔍 比对结论</div>论文要求固定 / 代码无固定动作"
                     "<div class='evi-val'>判定 <b>未固定</b></div>")
    else:
        norm_html = "<div class='evi-loc'>🔍 比对结论</div><span class='evi-none'>单侧证据</span>"
    ai_html = f"<div class='ai-box'>🤖 <b>AI 归因</b>　{f.ai_analysis}</div>" if f.ai_analysis else ""
    extra_html = ""
    for i, ev in enumerate(f.extra, 1):
        extra_html += (f"<div class='evi' style='margin-top:8px'><div class='evi-loc'>"
                       f"➕ 附加证据 {i} · {ev.location}</div>{ev.snippet}"
                       f"<div class='evi-val'>值 <b>{ev.raw_value}</b></div></div>")
    verify_html = ("<div class='vfy vfy-ok'>🛡️ 证据自证通过 · 原文片段确含该值，结论可复算</div>"
                   if f.verified else
                   f"<div class='vfy vfy-bad'>🛡️ 证据自证未通过 · {f.verify_note}　"
                   f"→ 请人工核对原位置后再采信</div>")
    evidence = [ev for ev in ([f.paper, f.code] + list(f.extra)) if ev]
    parsers = {ev.parser for ev in evidence} - {"ast"}
    chan_html = ""
    if parsers:
        _pname = {"regex": "🔧 通用通道", "table": "📊 表格通道",
                  "ocr": "🖼️ 图像 OCR", "vlm": "👁️ 视觉读图"}
        chan_html = "".join(f"<span class='chip chip-ghost'>{_pname.get(pp, pp)}</span>"
                            for pp in sorted(parsers))
    note_html = (f"<div class='evi-val' style='margin-top:8px'>ℹ️ {f.note}</div>"
                 if f.note else "")
    why_html = (f"<div class='why'>📐 <b>风险判定依据</b>：{f.risk_reason}</div>"
                if f.risk_reason else "")
    vfy_chip = "🛡️ 已自证" if f.verified else "⚠️ 待人工复核"
    # 用 st.html 渲染(纯 HTML, 不经过 Markdown 解析, 避免缩进/空行把标签截断成文本)
    st.html(f"""<div class="finding" style="--rc:{color}">
      <div class="risk-bar"></div>
      <div class="f-head"><span class="f-name">{STATUS_ICON[f.status]} {f.display_name}</span>
        <span class="chip" style="--cc:{color}">{zh}</span>
        <span class="chip chip-ghost">{STATUS_ZH[f.status]}</span>
        <span class="chip chip-ghost">{CONF_ZH[f.confidence]}</span>
        <span class="chip chip-ghost">{vfy_chip}</span>{chan_html}</div>
      <div class="f-evi"><div class="evi">{paper_html}</div><div class="evi">{code_html}</div><div class="evi">{norm_html}</div></div>
      {why_html}{extra_html}{ai_html}{verify_html}{note_html}
    </div>""")


def _hyp_where(h) -> str:
    """假设卡的出处一行（论文页码 / 代码文件行号）+ 它为什么排在这个位置。"""
    bits = []
    if h.paper_value:
        bits.append(f"📄 论文 {h.paper_value}"
                    + (f"（{h.paper_where}）" if h.paper_where else ""))
    if h.code_value:
        bits.append(f"💻 代码 {h.code_value}"
                    + (f"（{h.code_where}）" if h.code_where else ""))
    line = "　|　".join(bits)
    if h.why:
        line += ("　|　" if line else "") + f"排这个位置的理由：{h.why}"
    return line


def _hyp_verdict_chip(h) -> str:
    """推演判决的小标签（阶段二 2B）。

    三色各自对应一件事，别混: 红=一侧取值非法（比冲突更硬）、灰=推演证明不是真分歧、
    蓝=推演没结论照常待验。带颜色的出现在卡片右上，用户扫一眼就知道该先看哪几张。
    """
    v = getattr(h, "verdict", "")
    if not v:
        return ""
    color = {"已定位": "#e5484d", "已排除": "#8a99b5"}.get(v, "#3d63dd")
    return f"<span class='chip' style='--cc:{color}'>{_attr(v)}</span>"


def _hyp_derivation(h) -> str:
    """推演那一行 —— 没推演就不占版面（宁可少一行, 不要一行废话）。"""
    if not getattr(h, "verdict", ""):
        return ""
    cls = {"已排除": " hyp-d-ex", "已定位": " hyp-d-fg"}.get(h.verdict, "")
    kind = f"（{h.derivation_kind}）" if h.derivation_kind else ""
    body = h.derivation or "推演没给出结论，按原样交给 ② 实验站"
    return (f"<div class='hyp-d{cls}'>🧮 <b>{_attr(h.verdict)}</b>{_attr(kind)}　"
            f"{_attr(body)}</div>")


# 图表线索的来源 → 颜色（读图=蓝, 图注原文=灰 —— 图注是论文自己的话, 不加戏）。
_FIG_SRC_COLOR = {"vlm": "#3d63dd", "caption": "#8a99b5"}


def _fig_src_chip(c) -> str:
    src = getattr(c, "source", "")
    if not src:
        return ""
    label = "读图" if src == "vlm" else "图注原文"
    return (f"<span class='chip' style='--cc:{_FIG_SRC_COLOR.get(src, '#8a99b5')}'>"
            f"{label}</span>")


def _render_figures(audit) -> None:
    """① 假设站「读」这一步的**图表线索**（阶段二 2E）。

    与「图像通道」的分工: 那边读的是**超参数值**, 这边读的是**图本身**
    （图注原文 ＋ 图上看得见的坐标轴 / 趋势 / 峰位 / 图例）。

    边界写进副标题里: 只读看得见的东西, **不做鉴定、不推机理、不写结论**。
    两条通道: 图注（零成本, 无 key 也能跑）＋ 读图（需视觉 key, 两遍取交集防幻觉）。
    """
    clues = list(getattr(audit, "figure_clues", None) or [])
    total = int(getattr(audit, "figure_total", 0) or len(clues))
    stats = dict(getattr(audit, "figure_stats", None) or {})
    sub = f"读到 <b>{total}</b> 条图表线索（图注原文 ＋ 图上可见信息）"
    if stats.get("vision"):
        sub += f"　·　读图 <b>{stats['vision']}</b> 条（两遍取交集）"
    elif stats.get("pages"):
        sub += "　·　读图未启用（未配置视觉 key）"
    st.markdown("""<div class="sec"><span class="sec-h">🖼️ ① 假设站 · 读（图表）</span>
    <span class="sec-sub">%s</span></div>""" % sub, unsafe_allow_html=True)
    if not clues:
        st.info("本次没有读到图表线索 —— 论文正文里没有独立成行的图注"
                "（也可能本就不含图注/图）。这只是「没找到图注」，不等于论文里没有图。")
        return
    if total > len(clues):
        st.caption(f"共读到 {total} 条，按「越硬越前」排序后展示前 {len(clues)} 条。")
    if stats.get("engine"):
        st.caption(f"🖼️ 引擎：{stats['engine']}　·　只读图上看得见的东西，"
                   "不做化合物鉴定、不推断机理、不写结论（判定仍归 ③ 验证站）")
    for c in clues:
        st.html(
            f"<div class='hyp'>"
            f"<div class='hyp-h'><span class='hyp-r'>#{c.rank}</span>"
            f"<span class='hyp-k'>{_attr(c.figure)}</span>"
            f"<span class='chip chip-ghost'>{_attr(c.clue_type)}</span>"
            f"<span class='chip chip-ghost'>{_attr(c.kind)}</span>"
            f"{_fig_src_chip(c)}</div>"
            f"<div class='hyp-c'>{_attr(c.statement)}</div>"
            f"<div class='hyp-w'>图上依据：{_attr(c.anchor)}　·　{_attr(c.page_ref)}</div>"
            f"</div>")


def _render_chemistry(audit) -> None:
    """① 假设站「读」这一步的**化学身份核验**（阶段二 2E-化学）。

    与上面「图表线索」的分工: 那边读的是"图上看得见什么", 这边核的是
    "论文写的化学身份自不自洽"—— 分子式按原子量复算分子量、CAS 验校验位、
    SMILES 体检语法, 再与论文声明的数对账。

    **边界写进副标题**: 不从图像像素猜分子结构、不给化合物命名。
    认结构靠猜, 对账靠算 —— 这一块只做后者。
    """
    from paper2lab import chem as _chem

    items = list(getattr(audit, "chem_items", None) or [])
    total = int(getattr(audit, "chem_total", 0) or len(items))
    stats = dict(getattr(audit, "chem_stats", None) or {})
    summary = _chem.summarize(items, stats)
    sub = f"核验 <b>{total}</b> 条化学身份（分子式 / CAS / SMILES / 分子量）"
    if summary.get("checked"):
        sub += f"　·　可判定 <b>{summary['checked']}</b> 条"
    if summary.get("bad"):
        sub += f"，其中 <b>{summary['bad']}</b> 条对不上 / 不合规"
    st.markdown("""<div class="sec"><span class="sec-h">🧪 ① 假设站 · 化学身份核验</span>
    <span class="sec-sub">%s</span></div>""" % sub, unsafe_allow_html=True)
    if not items:
        st.info("本次没有抽到化学身份记录 —— 正文里没有分子式 / CAS / SMILES / 分子量声明"
                "（非化学领域论文属正常）。这只说明「没抽到」，不等于「化学信息没问题」。")
        return
    # 如实标注：这一步是模型跳过之后由程序补算的。跳过允许（调度权在模型），
    # 但"跳过"不能被读成"论文里本来就没有" —— 这是审计平台最容易糊过去的地方。
    if bool(getattr(audit, "chem_fallback", False)):
        st.warning("本次规划**跳过了「化学身份核验」**这一步，以下结果是程序在生成报告前"
                   "按「免费且必做」口径补算的 —— 结论有效，跳过本身已如实记录。")
    if total > len(items):
        st.caption(f"共抽到 {total} 条，按「先看对不上的」排序后展示前 {len(items)} 条。")
    if stats.get("engine"):
        st.caption(f"🧪 引擎：{stats['engine']}"
                   "　·　判据 = 按论文自己给的小数位舍入后比较；"
                   "不从像素猜结构、不给化合物命名")
    for it in items:
        st.html(
            f"<div class='hyp'>"
            f"<div class='hyp-h'><span class='hyp-r'>#{it.rank}</span>"
            f"<span class='hyp-k'>{_attr(it.value)}</span>"
            f"<span class='chip chip-ghost'>{_attr(it.kind)}</span>"
            f"<span class='chip' style='--cc:{_CHEM_COLOR.get(it.status, '#8a99b5')}'>"
            f"{_attr(_chem.STATUS_ZH.get(it.status, it.status))}</span>"
            f"<span class='chip chip-ghost'>第 {it.page} 页</span></div>"
            f"<div class='hyp-c'>{_attr(it.detail)}</div>"
            f"<div class='hyp-w'>原文：{_attr(it.context)}</div>"
            f"</div>")


# 化学身份的判定态 → 颜色（红=对不上 / 不合规, 绿=对得上, 灰=判不了）
_CHEM_COLOR = {
    "inconsistent": "#d94f4f",
    "invalid": "#d94f4f",
    "consistent": "#2f9e63",
    "undetermined": "#8a99b5",
}


def _hyp_judge(h) -> str:
    """模型判断那一行（2026-10-25）—— **只在模型真的判过时出现**。

    口径: 与程序结论**并列**, 不覆盖。程序说"待验"、模型说"已排除", 卡上的判决
    仍然是"待验"（那是可重算的），这里如实写"与程序不一致"并给出模型的理由 ——
    两边都留着, 不替读者做取舍。
    """
    v = getattr(h, "llm_verdict", "") or ""
    if not v:
        return ""
    agree = getattr(h, "llm_agree", None)
    if agree is True:
        tag, color = "与程序一致", "#30a46c"
    elif agree is False:
        tag, color = "与程序不一致", "#f76b15"
    else:
        # 规则这次没算出结论 —— 模型的话只能当线索, 不能当确证
        tag, color = "程序没结论，仅作线索", "#8a99b5"
    why = _attr(getattr(h, "llm_why", "") or "")
    return (f"<div class='hyp-f' style='color:{color}'>"
            f"🧠 模型判为「{_attr(v)}」· {tag}"
            + (f"：{why}" if why else "") + "</div>")


def _render_hypotheses(audit) -> None:
    """① 假设站的产出（阶段二 2A 提出 ＋ 2B 推演筛排）—— 一组**可以被实验推翻**的假设卡。

    这块是「从查错转向发现」的分界：上面的进度条讲的是"跑了哪些工序"，
    这里讲的是"因此我们**假设**什么、以及看到什么会推翻它"。

    阶段二 2B 之后每张卡多一个**推演判决**: 计算员在动手之前先算了一遍 ——
    一侧取值非法的提到最前（不用跑实验就能定）、推演证明不是真分歧的排到最后。
    判决与推演文案都只读卡片上的字段（由 derivation.py 回填），本函数不自己判断。
    """
    hyps = list(getattr(audit, "hypotheses", None) or [])
    total = int(getattr(audit, "hypothesis_total", 0) or len(hyps))
    stats = dict(getattr(audit, "hypothesis_stats", None) or {})
    sub = f"提出 <b>{total}</b> 条可证伪假设（按「最该先验」排序）"
    if stats.get("total"):
        bits = []
        if stats.get("flagged"):
            bits.append(f"<b>{stats['flagged']}</b> 条已定位为取值非法")
        if stats.get("excluded"):
            bits.append(f"<b>{stats['excluded']}</b> 条推演已排除")
        bits.append(f"<b>{stats['pending']}</b> 条待验")
        sub += "　·　" + "、".join(bits)
    st.markdown("""<div class="sec"><span class="sec-h">🔬 ① 假设站</span>
    <span class="sec-sub">%s</span></div>""" % sub, unsafe_allow_html=True)
    if not hyps:
        st.info("本次没有提出假设 —— 已抽取到的参数在两侧没有值得追问的分歧。"
                "（这不是「全部通过」的另一种说法：判定在下面的风险项里。）")
        return
    if total > len(hyps):
        st.caption(f"共提出 {total} 条，按可验证性排序后展示前 {len(hyps)} 条。")
    if stats.get("total"):
        st.caption(f"🧮 推演引擎：{stats.get('engine', '—')}"
                   "　（提完假设先算一遍：能算掉的先剔、一侧写错的先标，省下实验的力气）")
    for h in hyps:
        st.html(
            f"<div class='hyp'>"
            f"<div class='hyp-h'><span class='hyp-r'>#{h.rank}</span>"
            f"<span class='hyp-k'>{_attr(h.key)}</span>"
            f"<span class='chip chip-ghost'>{_attr(h.kind)}</span>"
            f"{_hyp_verdict_chip(h)}"
            f"<span class='chip chip-ghost'>"
            f"{'必须实机验证' if h.needs_rerun else '读代码即可定论'}</span></div>"
            f"<div class='hyp-c'>{_attr(h.claim)}</div>"
            f"<div class='hyp-f'>怎么推翻它：{_attr(h.falsifiable_by)}</div>"
            f"<div class='hyp-w'>{_attr(_hyp_where(h))}</div>"
            f"{_hyp_derivation(h)}"
            f"{_hyp_judge(h)}"
            f"</div>")


# 影响方向 → 颜色。橙=高低偏了（该留意）、蓝=批大小解释不掉（要另找原因）、
# 绿=已经对齐（不用查）、灰=算过但没影响。
_IMPACT_DIR_COLOR = {
    "偏高": "#f76b15", "偏低": "#f76b15", "偏少": "#f76b15", "偏多": "#f76b15",
    "已对齐": "#30a46c", "无影响": "#8a99b5", "无法用批大小解释": "#3d63dd",
}


def _impact_card(g: dict, idx: int) -> str:
    """一张「影响量级」卡 —— 只读 impact 里已有的字段, 本函数不自己判断。"""
    from paper2lab.report import _KIND_IMPACT_ZH

    im = g.get("impact") or {}
    d = str(im.get("direction") or "")
    color = _IMPACT_DIR_COLOR.get(d, "#8a99b5")
    kind_zh = _KIND_IMPACT_ZH.get(str(im.get("kind", "")), str(im.get("kind", "")))
    head = (
        f"<div class='hyp-h'><span class='hyp-r'>#{idx}</span>"
        f"<span class='hyp-k'>{_attr('、'.join(g.get('keys') or []))}</span>"
        f"<span class='chip chip-ghost'>{_attr(kind_zh)}</span>"
        + (f"<span class='chip' style='--cc:{color}'>{_attr(d)}</span>" if d else "")
        + "</div>"
    )
    body = (f"<div class='hyp-c'>比的是 <b>{_attr(str(im.get('compares') or '—'))}</b>："
            f"倍数 <b>{_attr(str(im.get('magnitude')))}</b>"
            + (f"，方向 <b>{_attr(d)}</b>" if d else "，落在两派区间内，不标方向")
            + "</div>")
    fm = "".join(f"<div class='hyp-f'>· {_attr(str(f))}</div>"
                 for f in (im.get("formulas") or []))
    cands = list(im.get("candidates") or [])
    cd = ""
    if cands:
        lines = []
        for c in cands:
            tag = "证据最硬" if c.get("rank") == 1 else f"第 {c.get('rank')} 顺位"
            lines.append(
                f"<div class='hyp-f'>· {_attr(tag)}（{c.get('score')} 分）："
                f"{_attr(str(c.get('reading')))} → 应有 "
                f"{_attr(str(c.get('quantity') or ''))} {_attr(str(c.get('implied')))} 倍"
                f"（冲突{'消失' if c.get('resolves') else '仍在'}）</div>")
        cd = ("<div class='hyp-w'>候选解释（按证据硬度排序，<b>不是「最优」</b>）：</div>"
              + "".join(lines))
    fl = "".join(
        f"<div class='hyp-w'>翻盘 · {_attr(str(f.get('input')))}："
        f"{_attr(str(f.get('now')))} → {_attr(str(f.get('if_changed')))}，"
        f"{_attr(str(f.get('effect')))}</div>" for f in (im.get("flip") or []))
    if not fl:
        # 空着不写会让人以为"没做这一步"。四项核算是承诺, 哪一项落地为空就**说清为什么空**,
        # 这比省一行字重要（报告那份同款口径, 不许一边写一边不写）。
        fl = ("<div class='hyp-w'>翻盘 · 本次没有会让结论反转的输入"
              "（口径无歧义，梯度累积也没有改变倍数）</div>")
    zw = "".join(f"<div class='hyp-w'>边界 · {_attr(str(z))}</div>"
                 for z in (im.get("zero_when") or []))
    ag = im.get("against") or {}
    if ag.get("available"):
        where = f"（{_attr(str(ag.get('where')))}）" if ag.get("where") else ""
        at = (f"<div class='hyp-w'>对照 · 拿论文报告的 {_attr(str(ag.get('metric')))} "
              f"= {_attr(str(ag.get('paper_value')))}{where} 作基准："
              f"{_attr(str(ag.get('predict')))}　⚠ {_attr(str(ag.get('note')))}</div>")
    else:
        at = f"<div class='hyp-w'>对照 · 本次做不了：{_attr(str(ag.get('why') or ''))}</div>"
    return f"<div class='hyp'>{head}{body}{fm}{cd}{fl}{zw}{at}</div>"


def _render_impact(audit) -> None:
    """① 假设站「算」这一步的**影响量级**（阶段二 2E）。

    与「推演判决」的分工: 那边说「这条是不是真分歧」, 这边说「如果是, 后果有多大」。
    三条边界写在副标题里 —— 提示层 / 给区间不给点值 / 「证据最硬」不是「最优」,
    免得看的人把它当成结论。
    """
    from paper2lab.report import _impact_groups

    groups = _impact_groups(audit)
    sub = f"算出 <b>{len(groups)}</b> 条可核算的影响（折成倍数 ＋ 方向）"
    st.markdown("""<div class="sec"><span class="sec-h">🧪 ① 假设站 · 算（影响量级）</span>
    <span class="sec-sub">%s</span></div>""" % sub, unsafe_allow_html=True)
    if not groups:
        st.info("本次没算出影响量级 —— 要么这条差异涉及的是分类型参数"
                "（dataset / optimizer / model 这类只有「一样 / 不一样」，没有「影响多大」），"
                "要么两侧缺量（例如一侧没写批大小、又查不到卡数）。"
                "**算不出就如实说算不出，不拿空话充数。**")
        return
    st.caption("提示层：不进判定、不进评分、不改就绪度　·　给区间不给点值（缩放有两派）"
               "　·　「证据最硬」不是「最优」—— 优不优要跑训练比准确率，本项目不跑训练")
    for i, g in enumerate(groups, start=1):
        st.html(_impact_card(g, i))


# ② 实验站的观察态 → 颜色。红=复算对不上（最该看）、橙=要动手做、蓝=量到了待判、
# 绿=已抹平、灰=算不出来 / 无需实验（没信息量的两类, 不抢眼）。
_EXP_COLOR = {
    EXP_DISAGREE: "#e5484d",
    EXP_MACHINE: "#f76b15",
    EXP_MEASURED: "#3d63dd",
    EXP_AGREE: "#30a46c",
    EXP_UNDECIDED: "#8a99b5",
    EXP_NO_NEED: "#8a99b5",
}


def _exp_chip(status: str) -> str:
    if not status:
        return ""
    return (f"<span class='chip' style='--cc:{_EXP_COLOR.get(status, '#8a99b5')}'>"
            f"{_attr(status)}</span>")


def _exp_rerun(e) -> str:
    """需实机时的可执行方案 —— 没有就不占版面（宁可少一行, 不要一行废话）。"""
    plan = getattr(e, "rerun_plan", "")
    return f"<div class='hyp-f'>实机方案：{_attr(plan)}</div>" if plan else ""


def _render_experiments(audit) -> None:
    """② 实验站的产出（阶段二 2D）—— 实验设计 ＋ 真跑记录 ＋ 实际值。

    与 ① 站的分界: ① 说"我们怀疑什么"（并把推得掉的先推掉）、
    ② 说"那怎么验"—— 而且**真的动手算了**: 每一步都留下调用的工具、喂进去的输入、
    算出来的结果与耗时。没有这些记录, "实验站"三个字就只是又一个面板。

    ⚠ ② 站只负责**测**: 卡片上写的是"观察结果"（复算一致 / 对不上 / 量到了 / 算不出来）,
    不写"假设成立 / 不成立" —— 判定是 ③ 站的事, 这里越位就等于自己给自己盖章。
    """
    exps = list(getattr(audit, "experiments", None) or [])
    total = int(getattr(audit, "experiment_total", 0) or len(exps))
    stats = dict(getattr(audit, "experiment_stats", None) or {})
    sub = f"设计并真跑 <b>{total}</b> 个实验（② 站只负责测，判定归 ③ 站）"
    if stats.get("calls"):
        sub += f"　·　调用计算工具 <b>{stats['calls']}</b> 次"
    bits = []
    for label, k in (("条复算对不上", "disagree"), ("条需实机", "machine"),
                     ("条实测待判", "measured"), ("条复算一致", "agree")):
        if stats.get(k):
            bits.append(f"<b>{stats[k]}</b> {label}")
    if bits:
        sub += "　·　" + "、".join(bits)
    st.markdown("""<div class="sec"><span class="sec-h">🧪 ② 实验站</span>
    <span class="sec-sub">%s</span></div>""" % sub, unsafe_allow_html=True)
    if not exps:
        st.info("本次没有做实验 —— 没有可设计的假设卡（① 站没提出假设），或实验站点被跳过。")
        return
    if total > len(exps):
        st.caption(f"共设计 {total} 个实验，按「最值得先看」排序后展示前 {len(exps)} 个。")
    if stats.get("engine"):
        _tools = "、".join(stats.get("tools") or []) or "—"
        st.caption(f"🔧 调用的计算工具：{_tools}　·　引擎：{stats['engine']}"
                   "　·　不跑训练、不执行用户代码")
    for e in exps:
        rows = ""
        for r in (e.records or []):
            ok = "✓" if r.get("ok") else "×"
            rows += (f"<div class='exp-r'>{ok} <b>{_attr(r.get('name'))}</b>　"
                     f"{_attr(r.get('output'))}"
                     f"<span class='exp-ms'>{r.get('ms')} ms</span></div>")
        vals = ""
        if e.expected is not None or e.actual is not None:
            vals = (f"<div class='exp-v'>预期 <b>{_attr(str(e.expected))}</b>"
                    f"　←　{_attr(e.expected_from)}<br>"
                    f"实际 <b>{_attr(str(e.actual))}</b>　←　{_attr(e.actual_from)}</div>")
        st.html(
            f"<div class='hyp'>"
            f"<div class='hyp-h'><span class='hyp-r'>#{e.rank}</span>"
            f"<span class='hyp-k'>{_attr(e.key)}</span>"
            f"{_exp_chip(e.status)}"
            f"<span class='chip chip-ghost'>"
            f"{'　'.join(e.tool_path) or '未调用工具'}</span></div>"
            f"<div class='hyp-c'>{_attr(e.claim)}</div>"
            f"<div class='exp-d'>实验设计：{_attr(e.design)}</div>"
            f"{rows}"
            f"{vals}"
            f"{_exp_rerun(e)}"
            f"<div class='hyp-w'>{_attr(e.plan_reason)}</div>"
            f"</div>")


def _render_trust_panel(audit):
    """审计可信度面板: 解析覆盖 + 合理性校验拒收 + 证据自证 + 比较日志。"""
    cov = audit.coverage or {}
    sc = audit.selfcheck or {}
    level = cov.get("confidence_level", "—")
    lv_color = {"高": "#30a46c", "中": "#f76b15", "低": "#e5484d"}.get(level, "#888")

    st.html(
        f"<div style='background:#fff;border:1px solid #e7ecf4;border-radius:12px;"
        f"padding:14px 20px;margin-bottom:12px'>"
        f"<div style='font-size:15px;font-weight:800;color:#16223a'>"
        f"本次审计可信度：<span style='color:{lv_color}'>{level}</span> "
        f"<span style='color:#8a99b5;font-size:12px'>({cov.get('confidence_score', '—')}/100)</span></div>"
        f"<div style='font-size:13px;color:#5a6b88;margin-top:6px'>"
        f"就绪度＝被审对象；可信度＝本次审计过程。</div></div>")

    c1, c2, c3 = st.columns(3)
    with c1:
        _pvb = cov.get("pages_covered_by_vision") or 0
        st.metric("论文内容覆盖（全通道）",
                  f"{cov.get('pages_covered', 0)} / {cov.get('pages_total', 0)}",
                  delta=(f"含读图覆盖 {_pvb} 页" if _pvb else None),
                  delta_color="off")
        st.metric("其中文本层可解析",
                  f"{cov.get('pages_parsed', 0)} / {cov.get('pages_total', 0)}")
        st.metric("论文参数种类覆盖",
                  f"{cov.get('kinds_valued', 0)} / {cov.get('kinds_seen', 0)} 类")
    with c2:
        st.metric("代码文件成功解析", f"{cov.get('code_ok', 0)} / {cov.get('code_files', 0)}")
        st.metric("多语言通道 / 完全无法处理",
                  f"{len(cov.get('multilang') or [])} / {len(cov.get('unsupported') or [])}")
    with c3:
        st.metric("证据自证通过", f"{sc.get('passed', 0)} / {sc.get('checked', 0)}")
        st.metric("被合理性校验拒收", len(cov.get("rejected") or []))

    img = cov.get("image") or {}
    if img.get("regions"):
        if img.get("enabled"):
            st.info(f"👁️ **图像通道已启用**（{img.get('model', '—')}）："
                    f"区域 {img.get('regions')} 个 · 读图 {img.get('regions_read')} 个 · "
                    f"采信 {img.get('kept')} 条 · 丢弃 {img.get('dropped')} 条"
                    "（读图结论标 🔵 Inferred，不计入就绪度）")
            for n in img.get("notes", []):
                st.caption(f"提示：{n}")
            with st.expander(f"🖼️ 读图的区域清单（{len(img.get('region_list') or [])} 个）"):
                _table([{"页码": r["page"],
                               "类型": "整页(扫描/图片页)" if r["kind"] == "page" else "内嵌图片",
                               "判定依据": r["reason"]}
                              for r in (img.get("region_list") or [])],
                             use_container_width=True, hide_index=True)
        else:
            st.warning(f"👁️ **图像通道未启用**：{img.get('regions')} 个图片/扫描区域"
                       "未参与本次审计（未配置视觉模型 key）")
            with st.expander(f"🖼️ 未读图的区域清单（{len(img.get('region_list') or [])} 个）"):
                _table([{"页码": r["page"],
                               "类型": "整页(扫描/图片页)" if r["kind"] == "page" else "内嵌图片",
                               "判定依据": r["reason"]}
                              for r in (img.get("region_list") or [])],
                             use_container_width=True, hide_index=True)

    if cov.get("multilang"):
        with st.expander(f"🔧 多语言通用通道解析的文件（{len(cov['multilang'])} 个："
                         f"MATLAB/R/C++/Julia/Fortran/Shell/INI/TOML 等）"):
            st.caption("语言无关的赋值抽取通道（带 文件:行号 证据）；"
                       "结论标 `🔧 通用通道`，置信度低于 Python AST。")
            _table([{"文件": m["file"], "语言": m["lang"], "抽取参数": m["params"]}
                          for m in cov["multilang"]],
                         use_container_width=True, hide_index=True)

    if cov.get("confidence_notes"):
        st.warning("**可信度扣分原因**\n\n" + "\n".join(f"- {n}" for n in cov["confidence_notes"]))

    if cov.get("kinds_unmatched"):
        st.info("📌 **论文提到但未取到的参数种类**：" + "、".join(cov["kinds_unmatched"])
                + "（正文、表格、图片三条通道都没取到）")

    _kbc = cov.get("kinds_by_channel") or {}
    if any(_kbc.values()):
        _seg = []
        if _kbc.get("text"):
            _seg.append(f"📄 正文等文本通道 {len(_kbc['text'])} 类（{'、'.join(_kbc['text'])}）")
        if _kbc.get("table"):
            _seg.append(f"📊 PDF 表格通道 {len(_kbc['table'])} 类（{'、'.join(_kbc['table'])}）")
        if _kbc.get("vision"):
            _seg.append(f"👁️ 视觉读图通道 {len(_kbc['vision'])} 类（{'、'.join(_kbc['vision'])}）")
        if _kbc.get("agent"):
            _seg.append(f"🤖 抽取 Agent 补漏 {len(_kbc['agent'])} 类（{'、'.join(_kbc['agent'])}）")
        st.info("🔎 取值来源分布：\n\n- " + "\n- ".join(_seg))

    _ag = cov.get("agent") or {}
    if _ag:
        _ag_steps = _ag.get("steps") or []
        _ag_filled = _ag.get("filled") or {}
        _ag_budget = _ag.get("budget") or {}
        _ag_targets = _ag.get("targets") or []
        # 标题里必须放**会随材料变**的数字（待补 N 类）；生成逻辑在 execview 里，
        # 便于用数据层断言守住（见 verify_execview.py）。
        with st.expander(
                _agent_panel_title(cov),
                expanded=bool(_ag_filled) or bool(_ag_targets)):
            st.caption("阶梯：文本紧邻 → 文本近邻 → 否定式 → LLM 提议 → 聚焦读图；"
                       "任一候选都要过闸门才入库，否则丢弃并留痕。")

            if _ag_targets:
                _tl = "、".join(str(t) for t in _ag_targets[:8])
                if len(_ag_targets) > 8:
                    _tl += " 等"
                st.caption(
                    f"待补目标 {len(_ag_targets)} 类（{_tl}）｜ "
                    f"补回 {len(_ag_filled)} 类 ｜ "
                    f"消耗 LLM {_ag_budget.get('llm_calls', 0)}/"
                    f"{_ag_budget.get('max_llm_calls', '—')} 次 · "
                    f"耗时 {_ag_budget.get('seconds', 0)}s ｜ "
                    f"结束：{_ag.get('stopped_reason', '—')}")
            else:
                # 没待补目标时不能只显示一排 0 —— 要说清"内核到底覆盖了什么",
                # 否则不同材料看到的是同一句"0 类"，等于没信息。
                _kbc = cov.get("kinds_by_channel") or {}
                _got = sorted({k for ch in ("text", "table", "vision", "agent")
                               for k in (_kbc.get(ch) or [])})
                st.info(
                    f"**本次无需补漏** —— 决定性内核已抽到 "
                    f"{cov.get('kinds_valued', 0)}/{cov.get('kinds_seen', 0)} 类参数，"
                    "没有「论文提到但没取到值」的参数。"
                    + (f"\n\n已取到：{'、'.join(_got)}" if _got else "")
                    + f"\n\n结束原因：{_ag.get('stopped_reason', '—')}")
            if _ag_steps:
                _table([{
                    "#": s.get("seq"), "参数": s.get("display"), "工具": s.get("tool"),
                    "结果": s.get("status"), "取值": s.get("value") or "—",
                    "页": s.get("page") or "—", "说明": s.get("note") or "",
                } for s in _ag_steps], use_container_width=True, hide_index=True)
            if _ag_filled:
                st.success("Agent 补回（均已过闸门）："
                           + "、".join(f"{k} = {v}" for k, v in _ag_filled.items()))

    if cov.get("thin_page_list"):
        _tread = cov.get("thin_pages_read") or []
        _tunread = cov.get("thin_pages_unread") or []
        _all = ", ".join(map(str, cov["thin_page_list"]))
        _msg = (f"📄 **文本稀薄页：第 {_all} 页**（文字层极少，无法走文本解析）")
        if _tread:
            _msg += f"\n\n- ✅ 已由**视觉读图**覆盖：第 {', '.join(map(str, _tread))} 页"
        if _tunread:
            _msg += (f"\n- ⚠️ **未读图**：第 {', '.join(map(str, _tunread))} 页 —— "
                     + (cov.get("thin_unread_reason") or "未参与审计"))
        st.warning(_msg)
        _img_on = bool((cov.get("image") or {}).get("enabled"))
        if _tunread and _img_on:
            st.caption("👉 想覆盖更多页面：在 `.env` 里把 `PAPER2LAB_VISION_MAX_REGIONS` "
                       "设为 0（全部读取）后重跑 —— 页面上不再有这个输入框；"
                       "读图结果按图缓存，已读过的不会重复计费。")
    if cov.get("code_failed_list"):
        with st.expander(f"❌ 代码解析失败文件（{len(cov['code_failed_list'])} 个，内容未参与审计）"):
            _table([{"文件": x["file"], "原因": x["reason"], "行数": x["lines"]}
                          for x in cov["code_failed_list"]],
                         use_container_width=True, hide_index=True)
    if cov.get("unsupported"):
        with st.expander(f"⚠️ 已识别但暂不支持分析的格式（{len(cov['unsupported'])} 个）"):
            _table([{"文件": x["file"], "语言": x["lang"]} for x in cov["unsupported"]],
                         use_container_width=True, hide_index=True)
    if cov.get("rejected"):
        with st.expander(f"🚫 被合理性校验拒收的抽取（{len(cov['rejected'])} 处，超领域先验范围）"):
            st.caption("以下数值被判定为错位抽取（如把年份当学习率），已拒收且不参与审计：")
            _table([{"参数": x["key"], "抽取值": x["value"],
                           "位置": x["location"], "原因": x["reason"]}
                          for x in cov["rejected"]],
                         use_container_width=True, hide_index=True)

    with st.expander(f"🛡️ 审计自检（证据自证 + 结论复算）："
                     f"{sc.get('passed', 0)}/{sc.get('checked', 0)} 通过",
                     expanded=bool(sc.get("failures"))):
        if sc.get("failures"):
            for fl in sc["failures"]:
                st.error(f"**{fl['param']}** — " + "；".join(fl["reasons"]))
        else:
            st.success(f"全部结论通过自检（{sc.get('passed', 0)}/{sc.get('checked', 0)}）")
        st.caption("自检两项：① 证据自证 —— 证据片段中确实出现所声明的值；"
                   "② 结论复算 —— 用同一套归一化逻辑重比一遍，与结论一致。")

    # ---- 未参与本次审计的部分(覆盖边界, 如实交代)
    not_cov: list[str] = []
    if img.get("regions") and not img.get("enabled"):
        not_cov.append(f"图像/扫描区域 {img.get('regions', 0)} 个"
                       f"（共发现候选 {img.get('candidates', img.get('regions', 0))} 个）："
                       "未配置视觉模型 key，内容未参与审计")
    if img.get("enabled") and img.get("regions_read", 0) < img.get("regions", 0):
        not_cov.append(f"图像区域 {img.get('regions', 0) - img.get('regions_read', 0)} 个："
                       "超出耗时预算未读图")
    if img.get("enabled") and img.get("skipped_by_cap"):
        not_cov.append(f"图像候选区域 {img['skipped_by_cap']} 个：超出上限 "
                       f"{img.get('max_regions')}（共发现候选 {img.get('candidates')} 个，"
                       "按「更可能含参数」排序择优读取；"
                       "上限在 .env 的 PAPER2LAB_VISION_MAX_REGIONS）")
    for x in cov.get("code_failed_list") or []:
        not_cov.append(f"代码文件 `{x['file']}`：{x['reason']}")
    for x in cov.get("unsupported") or []:
        not_cov.append(f"格式暂不支持 `{x['file']}`（{x['lang']}）")
    if cov.get("thin_page_list"):
        _tread = cov.get("thin_pages_read") or []
        _tunread = cov.get("thin_pages_unread") or []
        not_cov.append(
            f"论文文本稀薄页：第 {', '.join(map(str, cov['thin_page_list']))} 页"
            + (f"（其中第 {', '.join(map(str, _tread))} 页已由读图覆盖）" if _tread else "")
            + (f"；**未读图**：第 {', '.join(map(str, _tunread))} 页" if _tunread else ""))
    if cov.get("kinds_unmatched"):
        not_cov.append("论文提到但未取到数值的参数种类（所有通道均未取到）："
                       + "、".join(cov["kinds_unmatched"]))
    if cov.get("rejected"):
        not_cov.append(f"被合理性校验拒收的抽取 {len(cov['rejected'])} 处（判定为错位抽取）")
    st.markdown(f"**🚧 未参与本次审计的部分（{len(not_cov)} 项）**：")
    if not_cov:
        for x in not_cov:
            st.markdown(f"- {x}")
    else:
        st.success("本次审计的载体内容已全部纳入解析，无遗漏部分。")

    # ---- 参数覆盖清单(解析透明度: 抽到什么 / 没抽到什么 / 为什么)
    pm = cov.get("param_matrix") or []
    hit = sum(1 for r in pm if str(r["是否进入比对"]).startswith("✓"))
    st.markdown(f"**📋 参数覆盖清单（{len(pm)} 项已知参数：{hit} 项已进入比对，"
                f"{len(pm) - hit} 项未进入）**：")
    if pm:
        _table([{"参数": r["参数"], "论文侧": r["论文侧"], "代码侧": r["代码侧"],
                       "是否进入比对": r["是否进入比对"], "状态": r["状态"], "风险": r["风险"]}
                      for r in pm],
                     use_container_width=True, hide_index=True)

    # ---- 需人工核查清单(统一数据源, 与报告一致)
    mc = cov.get("manual_check") or []
    lv_color_mc = {"高": "#e5484d", "中": "#f76b15", "低": "#8a99b5"}
    st.markdown(f"**🙋 需人工核查清单（{len(mc)} 项）**：")
    if mc:
        for m in mc:
            st.html(
                f"<div class='mc-item' style='--mc:{lv_color_mc.get(m['level'], '#8a99b5')}'>"
                f"<div class='mc-head'>[{m['level']}] {m['kind']}　·　{m['item']}</div>"
                f"<div class='mc-line'>原因：{m['why']}</div>"
                f"<div class='mc-line'>建议：{m['action']}</div></div>")
    else:
        st.success("无：本次审计未产生需人工核查的条目。")

    with st.expander(f"📜 程序比较日志（全部 {len(audit.compare_log)} 条，逐条可复核）",
                     expanded=True):
        st.caption("全部判定由程序完成，以下为完整比较记录：")
        if audit.compare_log:
            st.code("\n".join(f"{i:>3}. {line}"
                              for i, line in enumerate(audit.compare_log, 1)),
                    language="text")
        else:
            st.info("本次审计没有产生比较日志")


# ================================================================ 主界面 · 深色指挥台
# ⚠ 2026-10-27 用户定版（第三次强调「要深色、要一进来就是它」）：
#   **一进入就是深色三栏指挥台，上传入口收在它左上角的第一行。**
#   旧版式是「浅色 hero + 上传区」在上、三栏大屏在下面 —— 首屏看到的还是浅色那一块，
#   得往下滚才看得见指挥台，等于没换。
#   现在：指挥台页**不画**浅色导航栏与 hero，整页底色转深，三栏大屏就是首屏；
#   上传三件与运行按钮压在这一屏左上角，其余每一块（材料 / 七位置 / 四站 / 工序 /
#   判定计数 / 就绪度）都点得开。
#   其余各页（审计总览 / 参数对表 / 假设卡 / …）照旧浅色，一行没动。
#
# 页面 key 必须在这里就算出来 —— 左侧导航栏在文件后半段才渲染。
#   ① URL 带 `?view=`（分享链接 / 探针直达）② session 里已选过 ③ 缺省 = 指挥台
_qv_early = str(st.query_params.get("view") or "").strip()
_page_now = (st.session_state.get("p2l_page") or _qv_early or "deck")
_DECK_ENTRY = (_page_now == "deck")
_on_deck_page = _DECK_ENTRY          # 旧变量名保留：下面「顶部常显区」还在用它
# 详情弹窗只在指挥台页存在。切到别的页时把「上次点开的是哪一块」清掉 ——
# 否则回指挥台时会莫名其妙自己弹出上次那块（2026-10-27 加弹窗时顺手处理）。
if not _DECK_ENTRY and st.session_state.get("dk_focus"):
    st.session_state.pop("dk_focus", None)
# 2026-10-28 修：只判 key 在不在会把「跑了但没产出」也算成有结果 ——
# 坏文件/缺代码时 run 会写入 audit=None，于是顶栏照样显示「● 已完成」，
# 页面下半段却是一句「这一轮没有产出审计结论」，自相矛盾。
_HAS_RESULT = st.session_state.get("audit") is not None
# 2026-10-29 修：**坏 PDF 不会再返回 None 了**（`paper_parser` 改成"读不出就出一份 0 页
# 的结果、把原因写进页覆盖表"），于是顶栏照样写「● 已完成」、页面上也没有一句"读不出来"——
# 用户会以为审计做完了、只是没查出问题。这里把"论文一页都没读到"单独判成材料异常：
# 顶栏改口径、上传区回到可重试状态、"没有产出审计结论"的提示照旧给出。
# 真机实证：上传一个内容不是 PDF 的 .pdf，页面显示「已解析 · 假设 6/6 格 · 已完成」。
_audit_now = st.session_state.get("audit")
_paper_now = getattr(_audit_now, "paper", None)
_page_cov = getattr(_paper_now, "page_coverage", None) or []
_PAPER_DEAD = bool(_HAS_RESULT) and int(getattr(_paper_now, "page_count", 0) or 0) <= 0 \
    and not any(str((c or {}).get("status")) in ("ok", "thin") for c in _page_cov)
_META_NOW = st.session_state.get("meta") or {}

# ---- 深色壳：整页底色 + 上传控件/开关/按钮/提示条一并换深色
# 只挂 `!important` 在「指挥台页」用到的元素上，且只在 `_DECK_ENTRY` 时输出 ——
# 其余页面连这段 <style> 都不会出现在 DOM 里，不存在串色。
_ENTRY_CSS = """<style>
.stApp{background:#05090f !important;}
[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stSidebar"]{background:#070c15 !important;border-right:1px solid #16203a;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label p{color:#8b9ab8 !important;}
.stApp .block-container{padding-top:1.1rem;max-width:1600px;}
/* 深色壳本体 */
.st-key-dk_entry{background:#080d17;border:1px solid #1b2540;border-radius:16px;
  padding:0 0 14px;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.55);}
.st-key-dk_entry [data-testid="stVerticalBlock"]{gap:.35rem;}
/* 序号标题：上传区那一行（保留「上传论文资料」原字，测试按它找首屏） */
.st-key-dk_entry .sec{background:transparent;border:0;padding:14px 18px 2px;margin:0;}
.st-key-dk_entry .sec-n{background:#141d33;color:#8ab0ff;}
.st-key-dk_entry .sec-h{color:#dbe4f5;}
.st-key-dk_entry .sec-sub{color:#4f5f80;}
/* 三张材料卡：换成深色底、浅色字（统一高度，别三高一低） */
.st-key-dk_entry .entry{background:#0b1424 !important;border:1px solid #1a2540 !important;
  box-shadow:none !important;min-height:112px;}
.st-key-dk_entry .entry-t{color:#dbe4f5 !important;}
.st-key-dk_entry .entry-d{color:#5b6b8c !important;}
.st-key-dk_entry .entry-tag{background:#141d33 !important;color:#8ab0ff !important;}
/* 上传控件 */
.st-key-dk_entry [data-testid="stFileUploaderDropzone"],
.st-key-dk_entry [data-testid="stFileUploader"] section{background:#070d19 !important;
  border:1px dashed #22304d !important;border-radius:9px !important;}
.st-key-dk_entry [data-testid="stFileUploaderDropzoneInstructions"] div,
.st-key-dk_entry [data-testid="stFileUploaderDropzoneInstructions"] span,
.st-key-dk_entry [data-testid="stFileUploader"] small,
.st-key-dk_entry [data-testid="stFileUploader"] span{color:#5b6b8c !important;}
.st-key-dk_entry [data-testid="stFileUploader"] button{background:#16223a !important;
  color:#c3d1e8 !important;border:1px solid #24314f !important;box-shadow:none !important;}
.st-key-dk_entry [data-testid="stFileUploaderFile"]{background:#0b1424 !important;
  border:1px solid #1a2540 !important;}
.st-key-dk_entry [data-testid="stFileUploaderFileName"]{color:#c3d1e8 !important;}
/* 运行按钮：主色保留蓝色，但深色下更醒目一点 */
.st-key-dk_entry .stButton>button{background:#16223a;color:#c3d1e8;
  border:1px solid #24314f;box-shadow:none;}
.st-key-dk_entry .stButton>button[kind="primary"],
.st-key-dk_entry [data-testid="stBaseButton-primary"]{background:linear-gradient(180deg,#2f5bd0,#2447a8) !important;
  color:#fff !important;border:1px solid #3d6ae0 !important;}
/* 开关：2026-10-27 起页面上不再有那一排开关（口径改成默认全开），
   这里的选择器留空不写，避免以后有人照旧样式加回去。 */
.st-key-dk_entry [data-testid="stWidgetLabel"] p{color:#8b9ab8 !important;font-size:13px !important;}
.st-key-dk_entry [data-testid="stTooltipIcon"] svg{fill:#3f4c69;}
/* 折叠面板 / 提示条 / 说明文字 */
.st-key-dk_entry [data-testid="stExpander"] details{background:#0b1424 !important;
  border:1px solid #1a2540 !important;border-radius:9px !important;}
.st-key-dk_entry [data-testid="stExpander"] summary,
.st-key-dk_entry [data-testid="stExpander"] summary p{color:#8b9ab8 !important;}
.st-key-dk_entry [data-testid="stAlert"]{background:#0e1728 !important;
  border:1px solid #1a2540 !important;color:#8b9ab8 !important;}
.st-key-dk_entry [data-testid="stAlert"] p{color:#8b9ab8 !important;}
/* ⚠ Streamlit 给 caption 挂了 margin-bottom:-16px（原本用来抵消默认的块间距）。
   深色壳把 [stVerticalBlock] 的 gap 调小了，这个负边距就把**下一块内容往上拉 16px**，
   正好盖住折叠面板里最后一行说明的字的下半截 —— 用户 2026-10-30 截图报的
   「读图通道 / 归因引擎 / 决策引擎」那行像被挡住，根因就是它，不是字号也不是行高。
   归零 + 给折叠面板内的说明补一点底部留白，字不再贴边。 */
.stApp [data-testid="stCaptionContainer"]{margin-bottom:0 !important;}
.st-key-dk_entry [data-testid="stExpander"] [data-testid="stCaptionContainer"]{padding-bottom:6px;}
/* 这行颜色原来 #4f5f80，在 #0b1424 底上对比度只有 3.2:1，缩到 14px 就"像被埋住"。
   提亮一档到 #7a8cb2（约 5.6:1），不抢正文但仍能读。 */
.st-key-dk_entry [data-testid="stCaptionContainer"],
.st-key-dk_entry [data-testid="stCaptionContainer"] p{color:#7a8cb2 !important;}
/* 窄窗口下这行会折成两行，而「(agent 模式)」单独掉到第二行很难看。
   balance 让两行长度接近，不会再把半截括号留在下一行。 */
.st-key-dk_entry [data-testid="stCaptionContainer"] p{text-wrap:balance;}
.st-key-dk_entry [data-testid="stMarkdownContainer"] p{color:#8b9ab8;}
/* 文本输入 / 数字输入（高级设置里） */
.st-key-dk_entry [data-testid="stTextInput"] input,
.st-key-dk_entry [data-testid="stNumberInput"] input{background:#070d19 !important;
  color:#c3d1e8 !important;border:1px solid #22304d !important;}
/* 壳内的三栏大屏（独立使用时自带圆角边框，嵌进来时抹平，看起来是一整块） */
.st-key-dk_entry .p2l-deck{border:0 !important;border-radius:0 !important;
  box-shadow:none !important;margin:0 !important;background:transparent !important;}
.st-key-dk_entry .st-key-dk_shell{border:0 !important;border-radius:0 !important;
  background:transparent !important;}
/* 指挥台页底部那三颗直达按钮与说明 */
.stApp .stButton>button{border-color:#22304d;color:#c3d1e8;background:#0e1728;}
.stApp [data-testid="stCaptionContainer"] p{color:#5b6b8c;}
</style>"""


def _deck_head_html(status: str, meta: dict) -> str:
    """指挥台顶栏（首屏最上面那一条）。

    为什么状态要由外面传：按钮点下去那一轮，顶栏在这之前就画好了 ——
    跑的过程中由 `_paint_deck` 用 status="run" 重画一次，跑完再按"已完成"画。
    中间那行口径里保留了「论文—代码—实验结果」这句标题文案：
    十几个真机测试拿它判断「首屏渲染成功」，换版式不能把这句一起换掉。
    """
    cls, txt = {"run": ("run", "● 审计中"), "idle": ("warn", "● 待运行"),
                "bad": ("warn", "● 材料读不出来")}.get(status, ("ok", "● 已完成"))
    files = str((meta or {}).get("files") or "")
    right = (files[:56] + "…") if len(files) > 56 else files
    if not right:
        right = "尚未选择材料"
    return (f"<div class='dk-top'><div class='dk-lg'>Paper<b>2</b>Lab</div>"
            f"<div class='dk-pill {cls}'>{txt}</div>"
            f"<div class='dk-hd'>论文—代码—实验结果 <b>智能一致性审计</b>"
            f"<span>{len(STEPS)} 格 · {len(TOOLS)} 件工序 · {len(POSITIONS)} 位置 · 3 道核验</span></div>"
            f"<div class='dk-sid'>{_html.escape(right)}</div></div>")


if not _DECK_ENTRY:
    # ---- 层级导航：这一行保证「点了能出去、也能回来」 ----------------------
    # 为什么必须有（2026-10-27 用户报的 bug）：指挥台底下那三个直达按钮
    # （逐条对表 / 审计发现 / 审计报告）会把页面切到二级视图，而侧栏在窗口不够宽时
    # 会被 Streamlit 自己收起 —— 于是二级视图看起来就是一扇没有出口的门。
    # 所以每一张非指挥台页面的最上面，固定放一颗「返回指挥台」。
    _bk_l, _bk_r = st.columns([1.05, 3.95], vertical_alignment="center")
    with _bk_l:
        if st.button("← 返回指挥台", key="back_to_deck", use_container_width=True):
            st.session_state["p2l_page"] = "deck"
            st.rerun()
    with _bk_r:
        st.caption("二级页面 · 上面这颗按钮随时能回指挥台（侧栏也能切）")

    # 注意这个 f 前缀是必需的：下方 hero-stat 里有 {len(STEPS)}。漏了它页面会
    # 原样显示「{len(STEPS)} 步」给用户看（线上实测抓到过，audit_placeholders.py 可复查）。
    st.markdown(f"""
<div class="navbar">
  <div class="nav-logo">🔬 Paper<b>2</b>Lab</div>
  <div class="nav-links">
    <span>审计输入</span><span>就绪度</span><span>风险项</span><span>审计报告</span>
    <a class="nav-cta" href="#" style="text-decoration:none">开始审计</a>
  </div>
</div>
""", unsafe_allow_html=True)

    # 有结果时不再摆那块大横幅 —— 结果才是主舞台（2026-10-26 B 方案改版）。
    # 横幅只在「还没开始」时承担引导作用；跑完之后它只是把结果顶出屏幕第一屏。
    if not _HAS_RESULT:
        st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">REPRODUCIBILITY READINESS AUDIT</div>
  <div class="hero-title">论文—代码—实验结果 <em>智能一致性审计</em></div>
  <div class="hero-sub">
    面向科研复现场景的审计流水线:AST 全量静态分析 × 数值归一化比对 × 证据链追溯。
    确定性问题交给程序验证,复杂归因交给 AI——让每一篇论文,更快从「读懂」走向「复现」。
  </div>
  <div class="hero-stats">
    <div class="hero-stat"><b>{len(STEPS)} 步</b><span>审计流水线(含 Agent 补漏与自检)</span></div>
    <div class="hero-stat"><b>{len(POSITIONS)} 智能体</b><span>解析 · 检索 · 主持 · 实验 · 计算 · 审查 · 撰写</span></div>
    <div class="hero-stat"><b>100%</b><span>发现均可追溯至页码与行号</span></div>
    <div class="hero-stat"><b>3 道</b><span>合理性校验 · 证据自证 · 结论复算</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# 跑完一轮之后，输入区折叠起来 —— 否则每次回看结果都要先滚过一整个上传界面。
# with 不产生作用域，所以里面的变量（run / use_demo / 上传路径…）照旧能往下用。
if _DECK_ENTRY:
    # 指挥台页：整块深色壳。顶栏一个占位，跑的过程里会被"● 审计中"重画一次。
    st.html(_ENTRY_CSS)
    # 三栏大屏那套 `.dk-*` 样式，这里换成挂到外层壳上 —— 与独立使用时**同一份** CSS，
    # 不另写一套，避免两处漂移。
    st.html(DECK_CSS.replace(".p2l-deck", ".st-key-dk_entry"))
    st.html(_deck_shell_css())
    _dk_entry = st.container(key="dk_entry")
    with _dk_entry:
        # 顶栏一行 = 左边「状态 + 标题 + 会话」（占位，跑的过程中被重画成 ● 审计中），
        # 右边是**问答入口**。为什么钉在首屏右上角：
        #   ① 不用滚动就能看见 —— 这是"醒目"的最低要求；
        #   ② 全页只有它是亮紫蓝渐变 + 呼吸光，靠颜色对比而不是靠位置去猜；
        #   ③ 不占额外高度（原来那一行本来就空着右半边），1440 宽也不会把大屏挤下去。
        _hd_l, _hd_r = st.columns([5.5, 1.0], gap="small", vertical_alignment="center")
        with _hd_l:
            _dk_top = st.empty()
        with _hd_r:
            if _qa_entry("qa_open_head"):
                st.session_state[QA_OPEN] = True
    _dk_top.html(_deck_head_html(
        "bad" if _PAPER_DEAD else ("done" if _HAS_RESULT else "idle"), _META_NOW))
    if _HAS_RESULT and not _PAPER_DEAD:
        # 跑完之后上传与开关只占一行，把首屏整个让给指挥台
        with _dk_entry:
            _input_host = st.expander("✏️ 换材料 / 调整选项后重新审计", expanded=False)
    else:
        # 材料读不出来时上传区保持展开 —— 用户要能立刻换一份重跑
        _input_host = _dk_entry
elif _HAS_RESULT and not _PAPER_DEAD:
    _input_host = st.expander("✏️ 换一份材料 / 调整设置后重新审计", expanded=False)
else:
    _input_host = st.container()

with _input_host:
    # ================================================================ ① 上传论文资料
    # 2026-10-27 用户定的口径：**三个上传入口统一收到页面左上角**，紧跟一个运行按钮。
    # 理由（人话）：进来第一眼该看见「材料往哪儿放、放完点哪儿」，
    #   而不是先滚过一排开关才知道这页是干什么的。
    # 顺序上先把上传与按钮摆出来，再往下是演示开关 / 决策权限 / 密钥 ——
    #   `with` 不产生作用域，所以 `run` 在这里定义、在下面 `if run:` 照旧能用。
    st.markdown("""<div class="sec"><span class="sec-n">01</span>
    <span class="sec-h">上传论文资料</span>
    <span class="sec-sub">三个载体统一从这里进,系统自动交叉核验</span></div>""",
                unsafe_allow_html=True)

    _up1, _up2, _up3, _up4 = st.columns([3, 3, 3, 2.15])
    with _up1:
        st.markdown("""<div class="entry" style="--c:#3d63dd;--cl:#e8efff">
          <div class="entry-head"><div class="entry-ic">📄</div>
          <div class="entry-t">论文 PDF</div><span class="entry-tag">必需</span></div>
          <div class="entry-d">提取方法、超参数与实验结果声明,证据定位精确到页码与章节</div>
        </div>""", unsafe_allow_html=True)
        pdf_file = st.file_uploader("论文", type=["pdf"], label_visibility="collapsed")
    with _up2:
        st.markdown("""<div class="entry" style="--c:#30a46c;--cl:#e6f7ef">
          <div class="entry-head"><div class="entry-ic">💻</div>
          <div class="entry-t">代码 ZIP</div><span class="entry-tag">可选</span></div>
          <div class="entry-d">AST 全量扫描训练配置与依赖环境,参数定位精确到文件与行号；不传则只做论文侧自查</div>
        </div>""", unsafe_allow_html=True)
        code_file = st.file_uploader("代码", type=["zip"], label_visibility="collapsed")
    with _up3:
        st.markdown("""<div class="entry" style="--c:#f76b15;--cl:#fff0e5">
          <div class="entry-head"><div class="entry-ic">📊</div>
          <div class="entry-t">实验结果 JSON</div><span class="entry-tag">可选</span></div>
          <div class="entry-d">回传实际跑出的指标,用于结果一致性核验与差异归因</div>
        </div>""", unsafe_allow_html=True)
        result_file = st.file_uploader("结果", type=["json"], label_visibility="collapsed")
    with _up4:
        st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
        # 文案保持「开 始 科 研 审 计」带字距的写法：十几个真机测试都在按这串字找按钮
        # （get_by_text 去掉空格后比对），改字会让它们集体点空 —— 位置换了，字别动。
        run = st.button("🚀 开 始 科 研 审 计", type="primary", use_container_width=True,
                        key="run_audit_main")
        # 说明压成一行：按钮右边这一列只有 200 来像素，写长了会折成三行、撑高首屏。
        st.caption(f"{len(TOOLS)} 件工序 · {len(STEPS)} 格 · 每格都能点开")

# ---- 选项区 --------------------------------------------------------------
# 2026-10-27 用户定版：**这一排开关全部从页面上移除**。
# 理由（人话）：这一页只需要回答两件事 ——「材料放哪儿」「点哪儿开始」。
#   多一排开关只会让人以为要先学会配置才能用；而这些开关**都不改判定口径**，
#   关掉只是少一层独立核验，没有任何一种组合能让结论更可信。
# 现在口径改成「默认全开，不给勾」：
#   · 演示数据开关也删掉了 —— 不给勾，改为**缺省材料**：
#     没上传就用内置样例（demo/ 下那三件）跑一遍流程，一上传就自动让位给你的文件。
#     这样新用户点一下就能看到完整流程，也不用先学「演示开关在哪」。
_AUTO_ON = True
use_demo = True
use_multagent = _AUTO_ON
use_closeloop = _AUTO_ON
use_auto_loop = _AUTO_ON
use_role_llm = _AUTO_ON
judge_llm = _AUTO_ON
use_exp_llm = _AUTO_ON

# 🧠 AI 参与决策（排计划 + 给补丁提名目标）—— **不设开关，恒开**。
# 口径（2026-09-29 用户定）：网页端就是 agent 入口 —— 输入材料、点启动，
# AI 一开始就参与决定**跑哪些环节**；它判定不值得跑的环节，
# 在流程面板标「未执行」并逐条给出理由。
# 若 AI 不可用（未配 key / 调用失败），自动回退规则版**并如实说明**。
_AI_ON = True


# ---- 状态提示与高级设置：整行宽，不进上面那个两栏栅格 ----------------------
with _input_host:
    # 材料：上传优先；没上传就用内置样例（页面上不再有「演示数据」开关）
    has_pdf = bool(pdf_file)
    has_upload = has_pdf
    if has_upload:
        st.caption("📌 本次用你上传的材料跑（内置样例自动让位）")
    else:
        st.caption("📌 还没上传材料 → 本次用内置样例 demo/paper.pdf ＋ project.zip ＋ "
                   "results.json 走一遍完整流程；上传自己的文件后自动改用你的")

    with st.expander("⚙️ 高级设置 · 模型 API"):
        # ---- 服务端（.env / Secrets）里配了什么：如实说明，但**绝不预填进页面**
        _lk = env_get("LLM_API_KEY") or env_get("DEEPSEEK_API_KEY")
        _lu = env_get("LLM_BASE_URL") or env_get("DEEPSEEK_BASE_URL")
        _lm_env = env_get("LLM_MODEL") or env_get("DEEPSEEK_MODEL")
        _vk = env_get("VISION_API_KEY") or env_get("DASHSCOPE_API_KEY") or env_get("ZHIPUAI_API_KEY")
        _vu = env_get("VISION_BASE_URL")
        _vm = env_get("VISION_MODEL")
        if not _vm and _vu:
            # 服务端只配了 base_url 时按地址猜模型名（2026-10-24：小米 MiMo 下线，
            # 首选改为 DeepSeek 的 deepseek-chat —— 实测带图能正确识图）
            _vm = {"deepseek": "deepseek-chat", "bigmodel": "glm-4v-plus",
                   "dashscope": "qwen-vl-max"}.get(
                next((s for s in ("deepseek", "bigmodel", "dashscope") if s in _vu), ""), "")
        # ============ 模型 API：页面上**只有这一组** ============
        # 2026-10-30 用户定：只留 Key / Base URL / 模型名三个框 —— 多模态与否由程序按
        # 模型名**自动识别**，并在下方写明「能读什么、要注意什么」。
        # 不再有第二组视觉密钥（会被读成"必须填两个"），也不再有多模态勾选框
        # （勾选与自动判定会打架，出现"勾了却没生效"这种最难解释的状态）。
        # 口径（2026-09-30 用户定）：支持任意 OpenAI 兼容模型，**并且必须填对应的 Base URL**
        # —— 不按 key 前缀猜端点，猜错会打到错误地址，表现为极慢甚至像卡死。
        st.markdown("**模型 API**　一个 Key 管全部：归因解释 / 排计划 / 读图（模型支持时）"
                    "　填完自动判断它能不能读图；它只输出文字解释与读图线索，**不影响结论**")
        # 2026-10-26 用户要求：页面上的输入框**从 .env 预填**，不用每次手填。
        # 安全边界（改成预填之后必须写在明面上）:
        #   `value=<真 key>` 会把 key 随页面送到浏览器 —— 点「眼睛」图标就能看见。
        #   所以这是**本机自用**的便利，不是可以公开的配置。
        #   开关: .env 里 `PAPER2LAB_PREFILL_KEYS=1` 打开（本机自用），
        #   **默认关**（键缺省 = 0）—— 线上没有 .env，所以线上一定是留空态。
        #   不预填时行为与以前完全一致: 留空 → `api_key or None` 回落服务端环境变量。
        _PREFILL = (env_get("PAPER2LAB_PREFILL_KEYS") or "0").strip().lower() \
            not in ("0", "false", "no", "off")
        api_key = st.text_input(
            "模型 API Key（已从 .env 预填，可直接改）" if _PREFILL else "模型 API Key",
            type="password", value=_lk if _PREFILL else "")
        mc1, mc2 = st.columns(2)
        with mc1:
            llm_base_url = st.text_input("模型 Base URL（必填）",
                                         value=_lu if _PREFILL else "",
                                         placeholder=_lu or "例如 https://api.deepseek.com")
        with mc2:
            llm_model = st.text_input("模型名（必填）", value=_lm_env if _PREFILL else "",
                                      placeholder=_lm_env or "例如 deepseek-chat")
        if _PREFILL and _lk:
            st.caption("📌 以上三项已按 `.env` 预填（模型 Key / Base URL / 模型名成对来自同一份配置）。"
                       "⚠️ 预填的值会随页面发到浏览器 —— **仅限本机自用**；"
                       "上线前在 `.env` 设 `PAPER2LAB_PREFILL_KEYS=0` 即可恢复留空。")
        else:
            st.caption("服务端模型 key："
                       + ("已配置 ✅（未预填到页面；留空即用服务端配置）"
                          if _lk else "未配置 ⚠️（留空即用内置模板，演示不中断）"))

        # ---- LLM 三级优先级（2026-10-30 用户定）------------------------------
        # ① 页面三个框**都填了** → 直接用页面这一套；
        # ② 页面没填（或没填全）→ 探测本机有没有已部署好的模型（默认 127.0.0.1:8080），
        #    有就用本地的 —— 不让用户为了"本机明明已经有"再去申请云端 Key；
        # ③ 本机也没有 → 服务端 .env / Secrets 配了就用它；再没有 → **无可用 LLM**，
        #    自动降级跑全套链路，并在页面上把"哪些环节会退步"讲清楚。
        # 为什么判定写在这里、而不是塞进 resolve_endpoint：页面必须把"本次用了谁、
        # 为什么是它"当场讲给用户看；resolve_endpoint 只负责给值，不负责解释。
        _page_k = (api_key or "").strip()
        _page_u = (llm_base_url or "").strip()
        _page_m = (llm_model or "").strip()
        _page_full = bool(_page_k and _page_u and _page_m)
        _page_part = bool((_page_k or _page_u or _page_m) and not _page_full)
        _local = _probe_local()

        if _page_full:
            _lk_eff, _lu_eff, _lm_eff = _page_k, _page_u, _page_m
            _llm_src = "page"
        elif _local["ok"]:
            _lk_eff, _lu_eff, _lm_eff = "local", _local["url"], _local["model"]
            _llm_src = "local"
        elif (_lk or "").strip() and (_lu or "").strip():
            _lk_eff, _lu_eff, _lm_eff = _resolve_endpoint("", "", "")
            _llm_src = "env"
        else:
            _lk_eff, _lu_eff, _lm_eff = "", "", ""
            _llm_src = "none"

        # 「填了一半」必须当场说清 —— 半套配置想用也用不了，静默忽略最伤人
        if _page_part:
            _miss = "、".join([n for n, v in (("Key", _page_k), ("Base URL", _page_u),
                                             ("模型名", _page_m)) if not v])
            st.warning(f"页面的模型 API **没填全**（缺 {_miss}），本次不采用这里的输入。"
                       "三个框要么都填、要么都留空 —— 只填一半会连错地址或认证失败。",
                       icon="⚠️")

        # 把「控制台地址」当成 API 填进来，要当场拦下（2026-10-31 实锤：
        # platform.deepseek.com 会被 WAF 拦成 429 + HTML 拦截页，问答那里显示成
        # 「模型调用失败（RateLimitError）· 云端在限流」完全是误诊，见 cache/qa_errors.log）。
        _url_hint = _endpoint_url_hint(_page_u)
        if _url_hint:
            st.warning(f"Base URL 可能填错了：**{_url_hint}**。", icon="🚫")

        _mm_auto = False
        _mm_hit = ""
        llm_multimodal = False
        _vision_from_llm = False
        _vk_eff = _vu_eff = _vm_eff = ""
        if _llm_src == "none":
            # 一个模型都没有：模态/读图都无从谈起，直接给一个"不可用"的判定，
            # 免得页面出现「已识别为单模态模型：``」这种看不懂的空壳。
            _vc = {"ok": False, "key": "", "url": "", "model": "", "source": "无",
                   "from_llm": False,
                   "reason": "当前没有任何可用的模型 API（页面没填、本机没检测到、"
                             "服务端也没配）",
                   "warning": ""}
        else:
            _mm_auto = _looks_multimodal(_lm_eff, _lu_eff)
            _mm_hit = _modality_hit(_lm_eff)
            llm_multimodal = _mm_auto

            # ---- 单模态 / 多模态：**按模型名自动判定**，结论 + 依据 + 注意事项写在下面 ----
            # 为什么不做成勾选框：判定只有一处真相（模型名），勾选框会造出第二个真相，
            # 两者不一致时用户看到的和程序实际用的就不是一回事。识别错了就改模型名。
            if not _lm_eff:
                st.caption("这个通道没报出模型名 —— 按「不能读图」处理；"
                           "填上模型名或换一个带视觉标识的模型名就会自动重新判断。")
            if _mm_auto:
                st.success(
                    f"**已识别为多模态模型：`{_lm_eff}`**　（依据：模型名含 `{_mm_hit}`）\n\n"
                    "- **能读什么**：文字 + 图片 —— 论文里的插图、表格截图、扫描页都能直接读进审计，"
                    "不需要另配视觉模型\n"
                    "- **要注意**：图片按区域逐张送模型，页数多时第一次会慢一些（之后走本地缓存）；"
                    "读图只补「图上写了什么」，**不参与判定**，判定仍由可复算的规则给出；"
                    "若这个模型其实读不了图，调用会如实标失败、未读区域列在「审计可信度」页，"
                    "不会编造内容", icon="🖼️")
            else:
                st.info(
                    f"**已识别为单模态模型：`{_lm_eff}`**"
                    "　（依据：模型名里没有视觉标识，如 `-vl` / `vision` / `4v`）\n\n"
                    "- **能读什么**：只有文字 —— 论文正文、代码、公式、表格里的文字层照常审计\n"
                    "- **要注意**：图片 / 表格截图 / 扫描页**不参与审计**，未读区域会列在"
                    "「审计可信度」页；关键数值只出现在图里时可能抽不到，建议换成带视觉标识的模型"
                    "（如 `gpt-4o` / `qwen-vl-max` / `intern-s1-mini`）；"
                    "认不出就按「不能读图」处理 —— 宁可不读图，也不假装读过", icon="📄")

            # ---- 读图通道判定：交给 paper2lab.llm.resolve_vision（纯函数，分支都可独立验证）。
            #      页面**不再传第二组视觉输入**（2026-10-30）：多模态 → 直接用①的模型读图；
            #      单模态 → 只看服务端有没有另配视觉通道（.env / Secrets），配了就自动接上，
            #      没配就如实说"本次不读图"。**Key 与 Base URL 成对才算配置**（不猜端点）。
            _vc = _resolve_vision(
                page_key="", page_url="", page_model="",
                env_key=_vk, env_url=_vu, env_model=_vm,
                env_key_has_known_url=bool(env_get("DASHSCOPE_API_KEY")
                                           or env_get("ZHIPUAI_API_KEY")),
                llm_key=_lk_eff, llm_url=_lu_eff, llm_model=_lm_eff,
                llm_multimodal=llm_multimodal)
            _vk_eff = _vc["key"] if _vc["ok"] else ""
            _vu_eff = _vc["url"] if _vc["ok"] else ""
            _vm_eff = _vc["model"] if _vc["ok"] else ""
            _vision_from_llm = bool(_vc["ok"] and _vc["from_llm"])

            if _vc["ok"] and _vision_from_llm:
                st.success(f"**读图通道：已就绪 ✅**　由上面这个模型自己读图（多模态）· `{_vm_eff}`"
                           "　不需要第二组密钥", icon="✅")
            elif _vc["ok"]:
                st.success(f"**读图通道：已就绪 ✅**　`{_vm_eff or '自动识别'}`"
                           f"　来源：{_vc['source']}（服务端配置，页面不用再填）", icon="✅")
            else:
                st.warning(f"**读图通道：本次不读图 ⚠️**　{_vc['reason']}。"
                           "图片 / 表格 / 扫描页不参与审计，未读区域会列在「审计可信度」页；"
                           "换一个带视觉标识的模型（如 `gpt-4o`）就会自动开启。", icon="⚠️")
            if _vc.get("warning"):
                # 不留死角：配置里有"填了一半"的东西，必须说清最终到底用了谁
                st.caption(f"ℹ️ {_vc['warning']}")
        # 读图上限与耗时预算（2026-10-27：页面上不再给输入框 —— 用户只填 API）。
        # 默认 0 = 全部读取、不限时；要改就在 .env 里写
        # PAPER2LAB_VISION_MAX_REGIONS / PAPER2LAB_VISION_BUDGET_SECONDS。
        try:
            vision_max_regions = max(0, int(env_get("PAPER2LAB_VISION_MAX_REGIONS") or "0"))
        except ValueError:
            vision_max_regions = 0
        try:
            vision_budget = max(0, int(float(env_get("PAPER2LAB_VISION_BUDGET_SECONDS") or "0")))
        except ValueError:
            vision_budget = 0

    # ---- LLM 可用性：三级优先级的结论，**紧跟 API 输入区下方**、始终可见 ----------
    # 为什么放在展开区外面：高级设置默认收起，状态要是也收在里面，用户就永远看不见
    # "这次到底用了谁"。这里只讲结论与后果，不解释机制。
    if _llm_src == "page":
        # 页面三个框都填了 ⇒ 直接用你填的，**这里刻意什么都不画**（2026-10-30 用户要求）。
        # 理由：三个输入框就在正上方，再复述一遍「用了哪个模型、连的哪个地址」是纯噪音，
        # 还白占掉首屏一行；你填完本来就知道自己填了什么。
        # 只有「用户没填、程序替他做了决定」时才需要解释 —— 那三种情况下面各自都有卡片。
        # 测试里有一条反向守卫盯着这里：别把复述卡片再加回来。
        pass
    elif _llm_src == "local":
        st.success(
            f"**页面没填 → 已检测到本机已部署的模型，本次直接用它** ✅\n\n"
            f"- 地址：`{_lu_eff}`（本机 {_local['port']} 端口，"
            f"来自桌面「书生本地模型」那个启动器）\n"
            f"- 模型：`{_lm_eff}`\n"
            "- 断网可用、数据不出本机；速度比云端慢（约 10 字/秒）", icon="🖥️")
    elif _llm_src == "env":
        st.info(
            f"**页面没填、本机也没检测到模型 → 使用服务端已配置的模型** ℹ️\n\n"
            f"- 地址：`{_lu_eff}`\n- 模型：`{_lm_eff}`\n"
            "- 想换成自己的：在上面三个框里填全（Key + Base URL + 模型名），页面填的优先",
            icon="ℹ️")
    else:
        st.warning(
            "**当前没有可用的 LLM ⚠️**　（页面三个框没填、本机 "
            f"{_local['port']} 端口没检测到模型、服务端也没配置）\n\n"
            "**会自动降级：仍按默认顺序跑完整套链路**（解析论文 → 抽取参数 → 两侧对表 → "
            "判定 → 自检 → 报告），不会中断，也不会编造内容。\n\n"
            "**但下面这些环节的准确度与详细程度会下降**（它们需要模型判断，现在退回规则版）：\n"
            "- **调度决策**：跑哪些环节由固定规则决定，不再按材料特点挑重点，可能多跑无关环节"
            "或漏掉该追问的地方\n"
            "- **归因解释**：只能给出内置模板式说明，讲不清「为什么会出现这个不一致」\n"
            "- **读图**：图片 / 表格截图 / 扫描页**完全不参与审计**，只出现在图里的数值抽不到\n"
            "- **问答**：无法回答追问，只能原样列出已检索到的审计事实\n\n"
            "**要恢复完整能力，二选一**：① 双击桌面「书生本地模型」里的启动器（本机模型，免费、"
            "离线）；② 在上面三个框里填入任一 OpenAI 兼容服务的 Key / Base URL / 模型名。",
            icon="⚠️")

    # 通道状态汇总行（读图通道｜归因引擎｜决策引擎）**已整行删除**（2026-10-30 用户要求）。
    # 为什么该删：上面两块（「已识别为多模态/单模态」+「读图通道：已就绪 ✅」）已经把
    # 读图通道和模型名说全了，这行只是复述，还多占一行、窄窗口会折成两行。
    # 若将来要恢复，记住当年的坑：这行用 `text-wrap:balance` 调过折行位置，
    # 也曾往分隔符里塞零宽空格引导断点 —— Chromium 不买账（中文本就允许任意字间断），
    # 要控断点只能把每段包进 `white-space:nowrap` 的 span（那要改渲染方式）。
    # 变量本身（_vk_eff / _vm_eff / _lm_eff / _lk_eff）**不能删**：下面问答弹窗要用。

    # 运行按钮已经挪到页面左上角（紧跟三个上传入口），这里只留通道状态说明。
    # 为什么不留第二个按钮：两个同名按钮会让人以为「上面那个不算」，
    # 而且 Streamlit 里两个按钮各自独立的 key，点哪个都只生效一次，容易误以为没反应。

# ---- 指挥台的三栏大屏占位 ------------------------------------------------
# 为什么占位要**先**建、内容**后**填：整页是自上而下逐块渲染的，而三栏里要显示的
# 执行记录得等下面 `if run:` 跑完才有。先在这里留一个空位，跑完再往里画 ——
# 这样大屏永远在「上传区下面、开关上面之后」，不会跑到页面最底下去。
if _DECK_ENTRY:
    with _dk_entry:
        _dk_body = st.empty()

# ================================================================ 问答窗口
# 为什么开在这里（执行段之前、所有 `st.stop()` 之前）：
#   ① 弹窗要能被「顶栏那颗按钮」在同一次运行里打开 —— 按钮在 1520 行附近画的，
#      状态在那时就写好了，这里正好读得到；
#   ② 这里 `_lk_eff / _lu_eff / _lm_eff` 已经算出来（高级设置那一段），
#      问答用的是**和审计完全相同**的模型通道，不另开一条；
#   ③ 后面的三个 `st.stop()`（没结果 / 结果为空 / 早退分支）都不会把它跳过去，
#      否则「点了按钮没反应」这种最难受的 bug 就会出现。
# 口径：问答**只读** audit，不写回任何字段 —— 解释层不许动判定。
#: 本轮脚本里「弹窗名额」被谁占了（"" = 还没人占）。
#: ⚠ Streamlit 1.64 的硬限制：**一次脚本运行只允许打开一个 st.dialog**，开第二个直接抛
#:   `StreamlitInvalidLayoutContextError: Only one dialog is allowed to be opened at the same time`，
#:   而它的后果不是"第二个弹窗不出现"，而是**整页从那一行往下全都不渲染**、进程还退出。
#:   2026-10-31 00:40 实测就栽在这：先在指挥台点过某一块（焦点留在 session 里），
#:   之后又点「问一问」→ 问答先开、详情再开 → 服务直接崩掉。
#:   所以两个入口必须互相让位：先到先得，后来的那一轮不开。
_DLG_TAKEN = ""
if st.session_state.get(QA_OPEN):
    _DLG_TAKEN = "qa"
    try:
        _qa_dialog(st.session_state.get("audit"),
                   st.session_state.get("meta") or {},
                   api_key=(_lk_eff or ""), base_url=(_lu_eff or ""),
                   model=(_lm_eff or ""),
                   report=st.session_state.get("report") or "")
    except Exception as _qa_ex:      # noqa: BLE001
        # 问答坏了绝不能把整页带下水：关掉弹窗、留一句原因，页面照常。
        st.session_state[QA_OPEN] = False
        st.warning(f"问答窗口这次没能打开（{type(_qa_ex).__name__}），"
                   "审计结果不受影响。可以关掉重开一次。", icon="💬")

# ================================================================ 执行
if run:
    # ---- 「点一次新的，就把上一次终止并清干净」（2026-10-30 用户要求）------------
    # 为什么必须做：本地模型是单实例，一问一答就要 2500+ token；上一次还在生成时
    # 又点一次，两个请求同时抢上下文 ⇒ 模型侧报 Context size exceeded / 429，
    # 用户看到「模型调用失败」还以为程序坏了。
    # 做法分两步，缺一不可：
    #   ① 令牌换代：上一次运行在**下一次模型调用前**收到 RunCancelled 立刻停手，
    #      （在途的那一次 HTTP 请求无法硬中断，所以是"最多再多等一次生成"）；
    #   ② 抹掉上一次的结果：页面不留旧结论，避免新旧两份混在一起被误读。
    _run_token = _begin_run()
    st.session_state["audit"] = None
    st.session_state["report"] = ""
    st.session_state["meta"] = {}
    st.session_state.pop("_p2l_prev_run", None)

    # ---- 会话级生命周期(阶段二): 一个会话一个库, 进入先清, 超时自动清。
    #      浏览器"关闭页面"事件不保证到达服务器, 因此"关页面即清"不可靠;
    #      等价且可靠的替代 = T1(进入先清) + T3(超时清扫)。
    #
    #      会话 id 由服务端首次签发后写进 session_state, 前后端一致,
    #      不靠客户端随机值(可伪造 / 碰撞)。
    if not st.session_state.get("_p2l_sid"):
        from paper2lab.session import new_id

        st.session_state["_p2l_sid"] = new_id()
    # 首次签发发生在本轮执行的**中途**, 而页面顶部的线程绑定早在签发之前就跑过了,
    # 所以这里必须**再绑一次**（多用户隔离的落点见文件上方"会话绑定"段）。
    from paper2lab.session import set_current_id

    set_current_id(st.session_state["_p2l_sid"])

    _new_uploads: list = []

    if has_upload:
        # 上传优先: 用用户自己的数据(代码 ZIP 可选——不传时只做论文侧自查)
        pdf_bytes = pdf_file.getvalue()
        pdf_path = _save_upload(pdf_file, ".pdf")
        _new_uploads.append(pdf_path)
        if code_file:
            code_bytes = code_file.getvalue()
            code_path = _save_upload(code_file, ".zip")
            _new_uploads.append(code_path)
        else:
            code_bytes = b""
            code_path = ""
        user_results = (json.loads(result_file.getvalue().decode("utf-8"))
                        if result_file else None)
        fingerprint = hashlib.sha1(pdf_bytes + code_bytes).hexdigest()[:10]
        # 会话级复用键(阶段三第1步): 论文+代码, 与 session.fingerprint 同源,
        #   交给 run_audit 当快照键; job.fingerprint 仍用 memory.fingerprint(只论文)
        #   做跨任务记忆, 两条线口径不同, 不能混用。
        session_fp = fingerprint
        meta = {"source": "用户上传",
                "files": f"{pdf_file.name}"
                         + (f" + {code_file.name}" if code_file else "（未提供代码）")
                         + (f" + {result_file.name}" if result_file else ""),
                "fingerprint": fingerprint}
    elif use_demo:
        if not (DEMO_DIR / "paper.pdf").exists():
            st.error("演示数据不存在，请先运行: python make_demo.py")
            st.stop()
        pdf_path = str(DEMO_DIR / "paper.pdf")
        code_path = str(DEMO_DIR / "project.zip")
        # 演示数据也要给出**输入字节**：T2 的会话指纹（_session_lookup）只认字节，
        # 用 pdf_bytes/code_bytes 两个名字。这一段曾漏掉，导致"默认开关＝演示数据"
        # 一点按钮就 NameError（上传路径有这两个变量、演示分支没有）。
        pdf_bytes = (DEMO_DIR / "paper.pdf").read_bytes()
        code_bytes = (DEMO_DIR / "project.zip").read_bytes()
        user_results = json.loads((DEMO_DIR / "results.json").read_text(encoding="utf-8"))
        fingerprint = hashlib.sha1(pdf_bytes + code_bytes).hexdigest()[:10]
        session_fp = fingerprint
        meta = {"source": "内置演示数据(受控样例)",
                "files": "demo/paper.pdf + demo/project.zip + demo/results.json",
                "fingerprint": fingerprint}
    else:
        # 2026-10-27 起这条基本走不到（没上传就用内置样例），留着是兜底：
        # 万一 demo/ 被删了，也得把话说明白，而不是静默什么都不做。
        st.warning("请上传论文 PDF（代码 ZIP 与结果 JSON 可选）后再开始")
        st.stop()

    # ---- T1: 新会话进入先清(只做一次——首次点击时)。
    #      后续同会话点击不再清库, 否则 T2 的索引会被自己清掉。
    _cleared = _session_clear()
    if _cleared is None:
        # 会话存储不可用(降级路径): 退回逐轮清理, 行为与改造前一致
        _purge = _purge_run_state(st.session_state.pop("_tmp_uploads", None) or [])
    else:
        # 会话内: 不清运行缓存(要让指纹命中生效), 只留痕上传临时文件
        _purge = _purge_run_state(st.session_state.pop("_tmp_uploads", None) or [],
                                  force=False)

    # ---- T2: 查输入指纹的状态(每次运行前都查, 不清库)。
    #      返回 needs_index 告诉后面"要不要重解析";
    #      若同会话内已解析过, snapshot 里给出复用产物, 跳过 PDF 解析与代码扫描。
    _init = _session_lookup(pdf_bytes, code_bytes)

    for _stale in ("audit", "report", "meta", "clar_done"):
        st.session_state.pop(_stale, None)

    # ---- 进度条：指挥台页**不画**。
    # 为什么：指挥台页的进度条会落在大屏下面的页面底部（超出首屏），而大屏右栏本来
    # 就有「实时事件 + N 格进度」，两条进度并排只会互相打架。所以指挥台页换成一个
    # 空壳对象，`bar.progress()/empty()` 照旧调用，什么也不画。
    if _DECK_ENTRY:
        class _NullBar:                     # noqa: D401 —— 只为吃掉两个方法调用
            def progress(self, *_a, **_k):
                return None

            def empty(self):
                return None

        bar = _NullBar()
    else:
        bar = st.progress(0, text="准备中…")

    # ---- 实时指挥台（2026-10-26）----------------------------------------
    # 跑的过程中就把三栏大屏画出来、逐条往上长，而不是"转圈等到最后才看见东西"。
    # 用的是 run_audit 本来就有的 progress 回调（**不新增任何执行侧接口**），
    # 所以 CLI / MCP / 测试的行为零变化，只有网页端多了一次重绘。
    # 指挥台页：直接画进上面那个占位里 —— 大屏就在首屏，不用滚。
    _deck_slot = _dk_body if _DECK_ENTRY else st.empty()
    _seen: list = []                       # 按出现顺序攒"走到过哪些工具"
    _t_deck = time.monotonic()             # 单调时钟：休眠/改系统时间都不会跳变
    _live_files = [x.strip() for x in (meta.get("files") or "").split("+")]

    def _tool_of(cp: str) -> str:
        """进度回调给的是**环节名**（如「论文解析」），这里反查它属于哪件工具。

        反查而不是让 runner 直接给 key —— 因为回调签名是既有契约，
        为了界面好看去改它，会让 CLI 与 MCP 一起受影响。
        """
        for _t in TOOLS:
            for _c in (_t.checkpoints or ()):
                if cp.startswith(_c):
                    return _t.key
        return ""

    def _paint_deck() -> None:
        try:
            # 指挥台页：顶栏的状态胶囊也要跟着变「● 审计中」
            if _DECK_ENTRY:
                _dk_top.html(_deck_head_html("run", meta))
            _deck_slot.html(DECK_CSS + deck_html(
                None,
                paper_name=(_live_files[0] if _live_files else ""),
                code_name=(_live_files[1] if len(_live_files) > 1
                           and "未提供" not in _live_files[1] else ""),
                phase="run",
                live_key=(_seen[-1] if _seen else ""),
                done_keys=tuple(_seen[:-1]),
                # 顶栏已经由外层画了，这里不再来一条
                show_top=not _DECK_ENTRY,
                elapsed=time.monotonic() - _t_deck))
        except Exception:      # noqa: BLE001
            # 指挥台画不出来**绝不允许影响审计本身** —— 静默跳过，
            # 进度条照旧，跑完照样出结论。
            pass

    _paint_deck()

    def on_progress(i: int, name: str):
        bar.progress(int((i + 1) / len(STEPS) * 100), text=f"✅ {name}")
        _k = _tool_of(name)
        if _k and _k not in _seen:
            _seen.append(_k)
        _paint_deck()

    # 2026-10-30：**跑之前先校验输入**。以前路径填错会一路跑到底、出一份看着正常的
    # 报告（18221 字），里面写着「代码侧比对整体缺席」—— 用户会当成「代码里确实没有」。
    # 校验放在这里（两条分支都已确定 pdf_path / code_path、进度条也已建好）：
    # 能把这轮明确停住，并在页面上用红字说清是哪一个输入有问题。
    from paper2lab.preflight import InputPathError, check_inputs
    try:
        check_inputs(pdf_path, code_path)
    except InputPathError as _pe:
        bar.empty()
        st.error(f"输入有问题，本次审计没有开始：{_pe}")
        st.stop()

    _audit_kwargs = dict(
        api_key=api_key or None, progress=on_progress,
        # 模型 API 的端点与模型名（页面可填，任意 OpenAI 兼容模型）
        llm_base_url=(llm_base_url or "").strip(),
        llm_model=(llm_model or "").strip(),
        # 读图用**已判定过的生效值**：单模态时是②里成对填的视觉配置，
        # 多模态时是①的模型 API 本身（_vision_from_llm）。
        # 直接用原始输入框会让"只填了 key 没填 URL"也偷偷生效。
        vision_key=_vk_eff,
        vision_base_url=_vu_eff,
        vision_model=_vm_eff,
        vision_source=("复用模型 API（多模态）" if _vision_from_llm else ""),
        vision_max_regions=int(vision_max_regions),
        vision_budget_seconds=float(vision_budget),
        # 2026-10-25 删除经典模式：不再有 plan_mode 传参，一律"先排计划再执行"。
        llm_planner=_AI_ON, llm_patch=_AI_ON,
        # 会话级复用(阶段三第1步): 同会话内第二次跑, 跳过解析
        needs_index=_init.get("needs_index") if _init else None,
        snapshot=_init.get("snapshot") if _init else None,
        # 快照键: 与 session.fingerprint 同源(论文+代码)
        session_fp=session_fp,
        # 2026-10-25: 模型先判、程序后验。放这里而不是只加在多智能体分支 ——
        # 两条入口（run_audit / run_audit_multagent）都认这个参数，默认 False 行为不变。
        judge_llm=bool(judge_llm),
        # 2026-10-26 自主模式：② 实验站模型挑工具。两条入口都认，默认 False 行为不变。
        experiment_llm=bool(use_exp_llm))

    if use_multagent:
        # 多智能体版（阶段四）: 与经典版同源签名，差别只在"参数比对"那段换成对撞区。
        # 依赖 langgraph —— **装不上/调用失败就回落经典版并如实说明**，
        # 绝不让整个页面因为一个可选能力崩掉。
        try:
            from paper2lab.graph import run_audit_multagent
            # 图版把视觉参数打包成 vision_kwargs（与 pipeline 内部构造同源）
            _mk = dict(_audit_kwargs)
            _mk["vision_kwargs"] = {
                "api_key": _vk_eff, "base_url": _vu_eff, "model": _vm_eff,
                "source_hint": ("复用模型 API（多模态）" if _vision_from_llm else ""),
                "budget_seconds": float(vision_budget)}
            _mk["vision_max_regions"] = int(vision_max_regions)
            # 对撞区角色是否用模型（默认关）。经典版 run_audit 不认识这个参数，
            # 所以只能加在这条分支上，不能放进上面共用的 _audit_kwargs。
            _mk["role_llm"] = bool(use_role_llm)
            # 阶段五（M2 闭环扩展）: 搜 / 算 / 做 + 迭代回边（默认关，纯规则零成本）。
            # 同样不能放进 _audit_kwargs —— run_audit 不认识这两个参数会 TypeError。
            _mk["loop"] = bool(use_closeloop)
            _mk["loop_rounds"] = 3
            # 2026-10-25 自主回头：把「要不要再走一遍」交给模型（默认关）。
            # 只加在这一条分支上 —— run_audit 不认识 auto_loop，会 TypeError。
            _mk["auto_loop"] = bool(use_auto_loop)
            for _drop in ("vision_key", "vision_base_url", "vision_model",
                          "vision_source", "vision_budget_seconds"):
                _mk.pop(_drop, None)
            audit, report_md = run_audit_multagent(pdf_path, code_path,
                                                   user_results, **_mk)
        except _RunCancelled:
            # 2026-10-30 晚：被新的一次点击作废 —— **不是故障**，更不该落回单路径
            # 再跑一遍（那等于把刚作废的任务又启动一次）。`st.rerun()` 是干净的
            # 退出方式：不弹红框、不留半成品，页面立刻进入新一轮。
            st.rerun()
        except InputPathError:
            # 输入路径问题不是「对撞区不可用」—— 让它原样上抛，别回落成单路径
            # 再跑一遍（那样只会把同一个错重复一遍，还多一条误导性的警告）。
            raise
        except Exception as _mex:      # noqa: BLE001
            st.warning(f"多智能体对撞区不可用（{type(_mex).__name__}: {_mex}），"
                       "已自动回落到单路径审计 —— 结论口径不变，只是没有对撞区留痕。")
            audit, report_md = run_audit(pdf_path, code_path, user_results,
                                         **_audit_kwargs)
    else:
        try:
            audit, report_md = run_audit(pdf_path, code_path, user_results,
                                         **_audit_kwargs)
        except _RunCancelled:
            st.rerun()
    bar.empty()
    # 落盘前再确认一次这一轮还算不算数（2026-10-30 晚）：
    # 用户在我们跑到一半时又点了一次「开始审计」，这一份就是**已作废的旧结果**，
    # 绝不能写进页面（否则他会以为新的一次已经跑完）。`st.rerun()` 干净退出。
    try:
        _check_run(_run_token)
    except _RunCancelled:
        st.rerun()
    st.session_state["audit"] = audit
    st.session_state["report"] = report_md
    st.session_state["meta"] = meta
    # 记住本轮上传落盘的位置 + 本轮开始前清掉了什么 —— 下一轮开始时会用到/会显示
    st.session_state["_tmp_uploads"] = _new_uploads
    st.session_state["_purge"] = _purge
    # 2026-10-26（B 方案改版）新增：跑完主动重跑一次脚本。
    # 为什么必须重跑：输入区是否折叠取决于「本轮开始渲染时 session_state 里有没有结果」。
    # 第一次点按钮时结果还没写进 session_state，于是那一轮输入区仍是展开的 ——
    # 用户看到的是"跑完了，但屏幕上还是上传界面"，得手动点一下才折叠。
    # 重跑后 run 已变 False（不会重复审计），结果从 session_state 读，页面直接是结果视图。
    st.rerun()

# ================================================================ 指挥台（首屏就是它）
# 2026-10-27：**从这里开始画主界面**，位置在「结果」判空之前 ——
# 因为没跑过的时候也要能看见它（空态三栏：材料未载入 / 七个位置待命 / 工序未执行），
# 每一块照样点得开。放在 `st.stop()` 之后的话，首屏就只剩上传区了，等于没换版式。
# 口径：数字全部现场派生（格数取 len(STEPS)、工序取 len(TOOLS)）；
#      没做的事标「未执行」并给理由，绝不填 0 分糊过去。
_DK_DRAWN = False          # 只画一次；页面下半段的兜底分支靠它判断
if _DECK_ENTRY:
    _dk_audit = st.session_state.get("audit")
    _dk_meta = st.session_state.get("meta") or {}
    _files = [x.strip() for x in (_dk_meta.get("files") or "").split("+")]
    _paper_n = _files[0] if _files else ""
    _code_n = (_files[1] if len(_files) > 1 and "未提供" not in _files[1] else "")
    _res_n = _files[2] if len(_files) > 2 else ""
    _exd = (getattr(_dk_audit, "execution_log", None) or {}) if _dk_audit else {}
    with _dk_body.container():
        _DK_DRAWN = True
        _render_deck(
            _dk_audit, paper_name=_paper_n, code_name=_code_n, results_name=_res_n,
            elapsed=float(_exd.get("seconds") or 0.0),
            planner=str((_exd.get("plan") or {}).get("planner") or ""),
            hosted=True)

        # 全景看完想深挖 -> 三个直达入口（值仍是固定 key，不用带数字的 label 当值）
        _jump = st.columns([1, 1, 1, 2.2])
        with _jump[0]:
            if st.button(f"逐条对表 · {len(getattr(_dk_audit, 'findings', None) or [])} 条",
                         use_container_width=True, key="dk_jump_table"):
                st.session_state["p2l_page"] = "table"
                st.rerun()
        with _jump[1]:
            _n_issue = len([f for f in (getattr(_dk_audit, "findings", None) or [])
                            if f.status != Status.CONSISTENT])
            if st.button(f"审计发现 · {_n_issue} 项", use_container_width=True,
                         key="dk_jump_find"):
                st.session_state["p2l_page"] = "find"
                st.rerun()
        with _jump[2]:
            if st.button("审计报告", use_container_width=True, key="dk_jump_report"):
                st.session_state["p2l_page"] = "report"
                st.rerun()
        with _jump[3]:
            st.caption("大屏上七块区域都能点：材料 / 七位置 / 四站 / 工序 / "
                       "实时事件 / 判定计数 / 就绪度 —— 点了就在弹窗里摊开，"
                       "关掉回到大屏，点过的那一行会一直亮着。")

    # ---- 详情弹窗（2026-10-27 第二版）
    # 一定要在容器**外面**调用：`st.dialog` 是挂在 body 下的门户，
    # 从 `with container():` 里调会把上下文串错，症状是"点了没反应"。
    # 也不能放在任何 `st.stop()` 之后，否则空态点不开。
    _focus = st.session_state.get(_DK_FOCUS) or ""
    if _focus and not _DLG_TAKEN:
        # 再保一层：即便哪天又冒出第三个弹窗入口，也不许它把整页带下水。
        # 详情画不出来只降级成一行说明，大屏和其它区域照常（铁律同 deckui.render_detail）。
        try:
            _render_detail(_dk_audit, _focus)
        except Exception as _dk_ex:      # noqa: BLE001
            st.caption(f"这一块详情这次没能弹出（{type(_dk_ex).__name__}），"
                       "大屏和其它区域不受影响；点一下别处再点回来可重开。")
    elif _focus and _DLG_TAKEN:
        # 问答弹窗正开着，名额被它占了 —— 这一轮先不开详情，但**不清焦点**：
        # 等你关掉问答，下一页自然把详情摊开。这比"两个弹窗互挤"或"点了没反应"都好。
        pass

# ================================================================ 结果
# 2026-10-26 改版（B 方案「证据审阅台」）：原来是一条长滚动页，现在拆成
# 「左侧导航 + 主区分页」。**判据没变、结论没变**，只是不再让用户一路滚到底。
# 保留的两条铁律：
#   ① 判空先于一切（只上传论文时 run_audit 会返回 (None, None)，
#      直接读 audit.readiness 会抛异常 → 整页往下的内容全部不渲染）；
#   ② 顶部四站进度带 + 自主面板**每一页都在** —— 「跑到哪了、谁定的」永远看得见。
if "audit" not in st.session_state:
    # 指挥台页不摆这两条 —— 它的空态三栏本身就写着「材料尚未载入 / 待命 / 未执行」，
    # 再叠一个浅色提示框和折叠面板，等于把深色大屏顶下去、又回到旧版的观感。
    if not _DECK_ENTRY:
        st.info("👆 选择输入后点击「开始科研审计」", icon="💡")
        with st.expander("ℹ️ 平台工作原理"):
            # 流水线用 STEPS 派生，不写死 —— 之前这里硬编码了 7 步（早过期了）
            st.markdown(f"**审计流水线（{len(STEPS)} 环节 / {len(TOOLS)} 工具）**："
                        + " → ".join(STEPS) + "\n\n"
                        "- 🔧 一致性判断由**程序化规则验证**完成"
                        "（AST 静态分析 + 数值归一化比对，如 `1e-4 = 0.0001`）\n"
                        "- 🤖 AI 负责**风险解释与归因**，并**全权规划与调度工具**"
                        "（调不调、跳不跳由它定；清单外的工具不存在，且每次调用都留痕）\n"
                        "- 🔗 每条审计发现均附**证据链**：论文页码·章节 + 代码文件:行号，可追溯、可复核\n")
    st.stop()

audit = st.session_state["audit"]
report_md = st.session_state.get("report") or ""
meta = st.session_state.get("meta", {})
# ⚠ 必须判空（2026-10-15 修）：只上传论文、不传代码时, run_audit 会走到
#   "没有代码材料"的降级分支并**返回 (None, None)**。而这里原来只判 key 在不在,
#   于是 `audit.readiness` 抛 AttributeError: 'NoneType' object has no attribute
#   'readiness' —— Streamlit 里单个组件抛异常会让**整页往下的内容全部不再渲染**,
#   用户看到的就是「点了开始科研审计, 进度条不动, 结果一直不出来」。
if audit is None or _PAPER_DEAD:
    # 2026-10-28 改文案：三件事必须说准 ——
    #   ① 坏文件（不是 PDF / 已损坏）也会走到这里，不能只归因于「缺代码」；
    #   ② 「演示数据开关」已在 10-27 删除，不能再让用户去点一个不存在的东西；
    #   ③ 给出可操作的三步，而不是一句结论。
    # 2026-10-29 补：`_PAPER_DEAD`（论文一页都没读到）也归到这一支 —— 坏 PDF 现在
    #   返回的是一份"0 页"的结果而不是 None，不判它会一路画成「已完成 · 0 条发现」。
    st.error(
        "⚠️ **这一轮没有产出审计结论** —— 不是「全部通过」。\n\n"
        "常见的三种原因：\n"
        "① 上传的论文**不是有效的 PDF**（文件损坏 / 只有扩展名是 .pdf）；\n"
        "② 只上传了论文、**没有代码 ZIP** —— 本平台的核心是论文与代码逐参数对照，"
        "缺一侧就无从比对；\n"
        "③ 文件路径不可读（权限或文件已被移动）。\n\n"
        "请这样处理：换个能正常打开的 PDF 重新上传；需要逐参数对照时补上传配套代码 ZIP；"
        "什么都不传直接点开始时，会用内置样例跑一遍完整流程。")
    st.stop()

r = audit.readiness
issues = [f for f in audit.findings if f.status != Status.CONSISTENT]
ok_count = len(audit.findings) - len(issues)
consistent = [f for f in audit.findings if f.status == Status.CONSISTENT]
_ex = getattr(audit, "execution_log", None) or {}

# 一条审计项都没有 ≠ 全部通过: 说明提取失败, 必须明确警示。
# ⚠ 2026-10-26 修正：这句警示**不能放在这里**。
#   放顶上会把后面的「假设卡」「化学核验」一起停掉 —— 而这两块本来就不依赖
#   参数比对结果（化学样本 chem_demo.pdf 就是这种情况：没有可审计参数，
#   但有 10 条化学身份可核）。所以改成在「参数对表」「审计发现」这两页各自拦。
_FINDINGS_EMPTY_MSG = (
    "⚠️ **未能从论文/代码中提取到任何可审计的参数**——这不是「全部通过」。\n\n"
    "可能原因：① 论文为扫描版/图片型 PDF，文字无法提取；"
    "② 参数以表格或图片形式存在；③ 参数表述句式超出当前规则库覆盖范围；"
    "④ 代码中未找到训练配置（如在 notebook 或二进制文件中）。\n\n"
    "建议：换用文字型 PDF 重试，或将该样本反馈给开发组补充解析规则。")


def _count(obj) -> int:
    """「有几条」的两种存法都要认（dict / list / 对象属性），别为了排版改数据合同。"""
    if not obj:
        return 0
    if isinstance(obj, dict):
        return len(obj)
    try:
        return len(obj)
    except TypeError:
        return 0


def _kpi_html(items) -> str:
    """四张大数卡。数字全部现场数，不写死。"""
    cells = "".join(
        f"<div class='kpi-c {cls}'><div class='v'>{val}</div><div class='l'>{lab}</div></div>"
        for cls, val, lab in items)
    return f"<div class='kpis'>{cells}</div>"


def _drawer(f) -> None:
    """右栏「抽屉」：论文原文 / 代码原文 / 比对结论 —— 三段并排看。
    三段全部取自 Evidence.snippet（抽取时留存的原文），不是拿结论反推的措辞。"""
    zh, color = RISK_META[f.risk.value]
    if f.paper:
        p_body = (f"{(f.paper.snippet or '（原文片段未留存）')}"
                  f"<div style='margin-top:9px;color:#3a4a6b'>声明值 "
                  f"<b>{f.paper.raw_value}</b></div>")
        p_head = f"论 文 原 文 · {f.paper.location}"
    else:
        p_body = "<span style='color:#94a2bd'>论文中未声明该参数</span>"
        p_head = "论 文 原 文"
    if f.code:
        c_body = (f"{(f.code.snippet or '（原文片段未留存）')}"
                  f"<div style='margin-top:9px;color:#3a4a6b'>实现值 "
                  f"<b>{f.code.raw_value}</b></div>")
        c_head = f"代 码 · {f.code.location}"
    else:
        c_body = "<span style='color:#94a2bd'>静态扫描未命中该参数</span>"
        c_head = "代 码"
    if f.paper and f.code:
        rel = "=" if f.status == Status.CONSISTENT else "≠"
        # normalized 可能是 None（这条抽取没做成功归一化）⇒ 退回原始值，
        # 别在结论里显示字面的 None。
        _pv = f.paper.normalized if f.paper.normalized is not None else f.paper.raw_value
        _cv = f.code.normalized if f.code.normalized is not None else f.code.raw_value
        concl = (f"论文写 <b>{_pv}</b>，代码实现 "
                 f"<b>{_cv}</b>（{rel}）。判定 "
                 f"<b>{STATUS_ZH[f.status]}</b>。")
    else:
        concl = f"单侧证据。判定 <b>{STATUS_ZH[f.status]}</b>。"
    if f.paper and f.paper.location:
        concl += f"<div style='margin-top:6px;color:#5a6b8a'>出处：{f.paper.location}"
        concl += (f" ｜ {f.code.location}" if f.code else "") + "</div>"
    st.html(
        f"<div class='drw-t'>{STATUS_ICON[f.status]} {f.display_name}</div>"
        f"<div class='drw-s'>判定 <b style='color:{color}'>{zh}</b> · "
        f"{STATUS_ZH[f.status]} · {CONF_ZH[f.confidence]} · "
        f"{'证据已自证' if f.verified else '待人工复核'}</div>"
        f"<div class='drw-p'><div class='drw-ph'>{p_head}</div>"
        f"<div class='drw-pb'>{p_body}</div></div>"
        f"<div class='drw-p'><div class='drw-ph'>{c_head}</div>"
        f"<div class='drw-pb'>{c_body}</div></div>"
        f"<div class='drw-v'><b>结 论</b><br>{concl}</div>")


# ================================================================ 顶部常显区（每一页都有）
# ---- 当前审计对象(防止"结果对不上数据"的困惑)
# 2026-10-27：指挥台页这一条改成深色 —— 整页已经是深色底，还摆一块浅蓝色的条，
# 看起来就像从别的网站抠下来贴上去的。内容一个字没改（数据指纹还在这里）。
if _on_deck_page:
    _obj_style = ("background:#0b1424;border:1px solid #1a2540;border-radius:10px;"
                  "padding:10px 16px;font-size:13px;color:#8b9ab8;margin-top:12px")
    _obj_code = "color:#a9c6ff"
else:
    _obj_style = ("background:#eef4ff;border:1px solid #d9e4ff;border-radius:10px;"
                  "padding:10px 16px;font-size:13px;color:#2c3a63;margin-top:14px")
    _obj_code = ""
st.html(
    f"<div style='{_obj_style}'>"
    f"🗂️ <b>当前审计对象</b>：{meta.get('source', '—')}　|　"
    f"📁 {meta.get('files', '—')}　|　"
    f"🔑 数据指纹 <code style='{_obj_code}'>{meta.get('fingerprint', '—')}</code>　|　"
    f"📄 论文《{audit.paper.title or '未识别标题'}》</div>")

# ---- 清理回执（一行功能名 + 事实；说理不占界面，见 housekeeping.format_report）
_pr = st.session_state.get("_purge")
if _pr:
    st.caption("🧹 清理：" + _purge_report(_pr))

# ---- ② 横向流程步骤条 + 就绪度横向面板
# ⚠ 2026-10-26：**指挥台页不画这两块**。原因是它们和指挥台重复了 ——
#   指挥台自己的中栏就是四站带、右栏就是 15 格进度与判定计数。
#   两套「跑到哪了」叠在一起，第一屏全被占满，真正的三栏大屏得往下滚才看得见，
#   那就违背了「一屏看清全貌」的初衷。其余各页照旧全画。
#   判断用的是 session_state 里的当前页（导航在下面才渲染，此处先读一次）——
#   第一次进来还没有这个键，就按 URL 上的 `view` 参数；两者都没有才按默认页（指挥台）。
# ⚠ 2026-10-26 真 bug（不是测试问题）：`?view=总览` 这类直达链接**第一次渲染时**
#   导航还没画出来，`p2l_page` 还不存在 ⇒ 原先误判成"在指挥台页" ⇒ 这一页的头带
#   （审计流水线 + 自主面板）凭空消失，分享链接一打开就是残缺页。
# ⚠ 2026-10-27：`_qv_early` / `_on_deck_page` 已经挪到文件前半段（主界面那一段），
#   因为指挥台要在这里之前就画出来 —— 变量名保持不变，这里直接用，不重算。
_ex0 = getattr(audit, "execution_log", None) or {}
_n_skip0 = _ex0.get("skipped", 0)
if not _on_deck_page:
    if _ex0.get("mode") == "planned":
        # 谁排的计划必须写清（1C 起: 归 LLM 全权排, 程序只在模型不可用时兜底）。
        # 写"调度员排计划"是阶段二的旧口径 —— 调度员是规则程序, 不是决策者。
        _pk0 = (_ex0.get("plan") or {}).get("planner", "rule")
        _who0 = "模型排计划" if _pk0 == "llm" else "规则兜底排计划"
        _step_sub = (f"{len(STEPS)} 环节 · {_who0}（跳过 {_n_skip0} 个工具）" if _n_skip0
                     else f"{len(STEPS)} 环节 · {_who0}（本次无跳过）")
    else:
        _step_sub = f"{len(STEPS)} 环节 · 强制全跑（兜底模式）"
    st.markdown(f"""<div class="sec"><span class="sec-n">02</span>
<span class="sec-h">审计流水线</span>
<span class="sec-sub">{_step_sub}</span></div>""", unsafe_allow_html=True)
    st.html(_stepper_html(_ex0))

    # ---- 自主运行面板（2026-10-26 UI 改版新增）：紧跟在进度条下面 ——
    #      进度条回答「跑到哪了」，这一排回答「谁定的、跑了多少、被谁拦下」。
    #      数字全部现场数出来，没发生的事显示「未触发」。
    st.html(_auto_panel(audit))

# ================================================================ 左侧导航（B 方案）
# 为什么把「一页长滚动」拆开：这个平台一次的产出是几十条发现 + 若干张假设/实验卡，
# 排成一列没人滚得完，而且「哪里是结论、哪里是过程」在长页里分不清。
# 计数全部现场数出来 —— 导航上的数字必须等于里面真实条数，不能写死。
_hyp_n = int(getattr(audit, "hypothesis_total", 0)
             or _count(getattr(audit, "hypotheses", None)))
_exp_n = int(getattr(audit, "experiment_total", 0)
             or _count(getattr(audit, "experiments", None)))
_fig_n = int(getattr(audit, "figure_total", 0)
             or _count(getattr(audit, "figure_clues", None)))
_chem_n = int(getattr(audit, "chem_total", 0)
              or _count(getattr(audit, "chem_items", None)))
_ask_n = 0
try:
    from paper2lab.inquiry import build_questions as _bq
    _ask_n = len(_bq(audit))
except Exception:      # noqa: BLE001
    _ask_n = 0

_PAGES = [
    # 2026-10-26：指挥台放在第一位、且是默认页 —— 一屏就能看完
    # 「材料 / 七位置 / 四站 / 工序 / 判定」，不用先学会翻十页。
    # 其余各页保持原样：指挥台是总入口，不是替代品。
    ("deck", "指挥台"),
    ("总览", "审计总览"),
    ("table", f"参数对表 · {len(audit.findings)}"),
    ("hyp", f"假设卡 · {_hyp_n}"),
    ("exp", f"实验卡 · {_exp_n}"),
    ("find", f"审计发现 · {len(issues)}"),
    ("multi", "多智能体对撞"),
    ("loop", "复现闭环"),
    ("ask", f"待确认 · {_ask_n}"),
    ("exec", "执行记录"),
    ("report", "审计报告"),
]
_KEYS = [k for k, _ in _PAGES]
_LABEL_BY_KEY = dict(_PAGES)
with st.sidebar:
    st.html("<div class='nav-grp'>Paper2Lab · 审计视图</div>")
    # 换到别的页以后，首屏那颗按钮就看不见了 —— 侧栏最顶再给一颗。
    # 为什么必须 `st.rerun()`：弹窗的「开不开」是在页面更靠上的位置判断的，
    # 而这里已经是页面底部，不重跑一轮这一下点了会像没反应。
    if _qa_entry("qa_open_side", "💬 问 一 问"):
        st.session_state[QA_OPEN] = True
        st.rerun()
    st.html("<div style='height:10px'></div>")
    # ⚠ 必须「值是固定 key、显示才带计数」（format_func）。
    #   2026-10-26 踩过：直接把「审计发现 · 7」这种带数字的字符串当选项值，
    #   换一份材料重跑后条数变了（7→12），旧值对不上新选项 ⇒ 导航被重置回总览页，
    #   用户明明在看某一页，跑完却被弹回第一页。值稳定就再也不会跳。
    # ⚠ 2026-10-26：支持 `?view=<key>` 直达某一页。
    #   两个用途：① 分享链接（"你看这一页"）；② 自动化探针直接落到它关心的页。
    #   只在**首次**渲染时起作用 —— radio 一旦有了 session 值，index 就不再参与，
    #   所以点按钮重跑不会把用户从当前页弹走。
    _qv = str(st.query_params.get("view") or "").strip()
    _idx0 = _KEYS.index(_qv) if _qv in _KEYS else 0
    _pick = st.radio("视图", _KEYS, index=_idx0, key="p2l_page",
                     format_func=lambda k: _LABEL_BY_KEY.get(k, k),
                     label_visibility="collapsed")
    st.caption("每一页顶上都带同一条四站进度带 —— 换页不换的是「跑到哪了」。")
_page = _pick if _pick in _KEYS else "总览"


if _page == "deck":
    # ---- 指挥台已经在上面的「指挥台（首屏就是它）」那一段画完了 ——
    #   它要在 `st.stop()` 之前渲染，首屏才对；这里留一个兜底：
    #   万一有人从别的页切回来而上面那一段没画（例如 URL 参数与 session 打架），
    #   就在这儿补画一次，绝不让「点了导航却没内容」发生。
    if not _DECK_ENTRY or not _DK_DRAWN:
        _render_deck(audit, hosted=False)


elif _page == "总览":
    # ---- KPI 四张卡：先把「一共比了多少、对上多少、对不上多少、还有多少别的问题」
    #      摆在第一屏。数字全部现场数；顺序固定（不按大小重排，否则每次跑都在跳）。
    _n_incon = sum(1 for f in issues if f.status == Status.INCONSISTENT)
    st.html(_kpi_html([
        ("b", len(audit.findings), "提取到可比对项"),
        ("g", ok_count, "两侧对得上"),
        ("r", _n_incon, "两数不一致"),
        ("o", len(issues) - _n_incon, "其他风险（缺失 / 矛盾 / 未固定）"),
    ]))
    st.caption("一眼看结论：「两数不一致」才是硬冲突；「其他风险」多半是论文写了、"
               "代码里没找到，属于**缺证据**而不是**写错**。两者在报告里分开记，"
               "不合并成一个「风险数」。")

    _lm = r.get("llm_mode")
    # 模型名如实显示：换过模型就不能还写 DeepSeek（readiness["llm_model"] 由 toolbox 写入）
    _lm_name = r.get("llm_model") or "模型 API"
    if _lm == "deepseek":
        _rc, _ch = r.get("llm_real_calls", 0), r.get("llm_cache_hits", 0)
        _llm_label = f"{_lm_name}（真实调用 {_rc} 条" + (
            f" · 缓存命中 {_ch} 条）" if _ch else "）")
    elif _lm == "fallback":
        _llm_label = (f"内置模板（{_lm_name} 调用失败已降级："
                      f"{r.get('llm_error') or '未知原因'}）")
    elif _lm == "skipped":
        # 2026-10-26：配了 key，但这一步本次没执行。
        # 多智能体档用对撞区替代了「AI 风险归因」，所以这里空着是设计，不是没配。
        _llm_label = "未启用（本次计划未包含「AI 风险归因」，由对撞区承担）"
    else:
        _llm_label = "内置模板（未配置 API key）"
    st.html(f"""<div class="sec"><span class="sec-n">03</span>
    <span class="sec-h">复现就绪度</span>
    <span class="sec-sub">发现 <b style="color:#e5484d">{len(issues)}</b> 项复现风险 ·
    归因引擎 {_llm_label}</span></div>""")
    st.html(_ready_band(r.get("total", 0), r.get("categories", {}),
                        r.get("audit_confidence")))
    st.caption("就绪度＝被审对象；审计可信度＝本次审计过程。")
    _unv = r.get("unverified") or {}
    if _unv:
        st.info("未核验维度（不计入总分）：\n\n"
                + "\n".join(f"- **{k}**：{v}" for k, v in _unv.items()), icon="⚪")
    if r.get("dominant_issues"):
        st.warning("⚠️ 当前阻碍复现的主要因素：**" + "、".join(r["dominant_issues"]) + "**")

elif _page == "table":
    # ======== B 方案核心页：筛选 → 表格 → 点一行看原文对照 ========
    # 结论列表最怕「只能信、不能查」。这一页把每一行做成可点选的：
    # 点哪一行，右边就摆出**论文原文片段**与**代码原文片段**
    # （都是抽取时留存的原文，不是按结论重新造句），外加比对结论与两侧出处。
    if not audit.findings:
        st.error(_FINDINGS_EMPTY_MSG)
        st.stop()
    _GROUPS = [
        ("全部", None),
        ("两数不一致", Status.INCONSISTENT),
        ("论文有代码无", Status.MISSING_IN_CODE),
        ("代码有论文无", Status.MISSING_IN_PAPER),
        ("内部矛盾", Status.INTERNAL_INCONSISTENT),
        ("未固定", Status.NOT_FIXED),
        ("无法静态核验", Status.UNVERIFIABLE),
        ("一致", Status.CONSISTENT),
    ]
    _opts, _map = [], {}
    for _lab, _st in _GROUPS:
        _n = len(audit.findings) if _st is None else sum(
            1 for f in audit.findings if f.status == _st)
        if _n == 0 and _st is not None:
            continue                      # 0 条的筛选留着只会让人白点一次
        _t = f"{_lab} {_n}"
        _opts.append(_t)
        _map[_t] = _st
    _pick2 = st.pills("筛选", _opts, default=_opts[0], key="p2l_filter",
                      label_visibility="collapsed")
    _st_sel = _map.get(_pick2 or _opts[0])
    _view = [f for f in audit.findings
             if (_st_sel is None or f.status == _st_sel)]
    st.caption(f"当前列出 **{len(_view)}** 项。点任意一行，右侧显示这一项的论文原文、"
               "代码原文与比对结论 —— 三条都能独立复核。")

    _cA, _cB = st.columns([2.65, 0.85], gap="medium")
    with _cA:
        # 列数刻意压到 5 列：在 1500px 宽的笔电上（侧栏占 300px、主区只剩 ~750px），
        # 6 列会把「出处」和「置信」挤成省略号。置信度不丢 —— 点行后右栏第一行就是它。
        _rows = [{
            "判定": f"{STATUS_ICON[f.status]} {STATUS_ZH[f.status]}",
            "比对项": f.display_name,
            "论文": (str(f.paper.raw_value) if f.paper else "未声明"),
            "代码": (str(f.code.raw_value) if f.code else "未找到"),
            "出处": ((f.paper.location if f.paper else "—")
                     + " ｜ " + (f.code.location if f.code else "—")),
        } for f in _view]
        _ev = st.dataframe(
            _rows, hide_index=True, height=560, width="stretch", on_select="rerun",
            selection_mode="single-row", key="p2l_tbl",
            column_config={
                "判定": st.column_config.TextColumn("判定", width=96),
                "比对项": st.column_config.TextColumn("比对项", width=150),
                "论文": st.column_config.TextColumn("论文", width=78),
                "代码": st.column_config.TextColumn("代码", width=78),
                "出处": st.column_config.TextColumn("出处", width=190),
            })
    with _cB:
        _sel = []
        try:
            _sel = list(getattr(getattr(_ev, "selection", None), "rows", []) or [])
        except Exception:      # noqa: BLE001
            _sel = []
        if _sel and 0 <= _sel[0] < len(_view):
            _drawer(_view[_sel[0]])
        else:
            st.html("<div class='drw-empty'>👈 在左边点一行<br>"
                    "这里显示它的论文原文 / 代码原文 / 比对结论</div>")
    st.caption("表格是画布渲染（读屏与复制取不到文字），所以「共多少项、几项对不上」"
               "已用文字写清；逐项详情点行看右栏，或展开下面明细核对。")

    with st.expander(f"📋 逐条明细（{len(_view)} 项，可展开复制）", expanded=False):
        for _f in _view:
            _render_finding(_f)


elif _page == "hyp":
    st.markdown('<div class="sec"><span class="sec-h">🔬 ① 假设站</span><span class="sec-sub">读材料 → 读图 → 核化学身份 → 补漏 → 提假设 → 算影响量级</span></div>', unsafe_allow_html=True)
    # ---- ① 假设站「读」（阶段二 2E）：图表线索排在假设卡之前 —— 先看图, 再提假设。
    _render_figures(audit)

    # ---- ① 假设站「读」的另一半（2E-化学）：化学身份核验（分子式 / CAS / SMILES / 分子量）。
    #      与图表线索同一件工序、同一步产出，所以紧挨着排 —— 但**分开渲染**:
    #      一个是"图上看得见什么", 一个是"化学身份算不算得对", 混在一张卡里两边都说不清。
    _render_chemistry(audit)

    # ---- ① 假设站（阶段二 2A）：流水线跑完，先说"我们假设什么、看到什么会推翻它"，
    #      再谈判定。**紧跟进度条** —— 与四站分区一一对上（① 的产出就在 ① 的进度条下面）。
    _render_hypotheses(audit)

    # ---- ① 假设站「算」的另一半（2E）：影响量级 / 候选解释 / 翻盘 / 边界 / 对照。
    #      紧跟在假设卡之后 —— 卡片说"这里不一样", 这里说"不一样有多大后果"。
    _render_impact(audit)

elif _page == "exp":
    st.markdown('<div class="sec"><span class="sec-h">🧪 ② 实验站</span><span class="sec-sub">把假设卡变成「怎么验」，并真的动手算</span></div>', unsafe_allow_html=True)
    # ---- ② 实验站（阶段二 2D）：紧接着 ① 站 —— 提完假设就说"那怎么验、算了什么"。
    #      四站的顺序就是这么来的（① 提 → ② 做 → ③ 判 → ④ 改），界面上不能乱。
    _render_experiments(audit)

elif _page == "find":
    st.markdown('<div class="sec"><span class="sec-h">📑 ③ 验证站 · 审计发现</span><span class="sec-sub">无论是否发现风险，每项结论均附证据链：论文页码·章节 + 代码文件:行号</span></div>', unsafe_allow_html=True)
    if not audit.findings:
        st.error(_FINDINGS_EMPTY_MSG)
        st.stop()
    # ---- ④ 风险与报告
    # ---- 证据链总览表(一致项同样可追溯)
    overview_rows = []
    for f in audit.findings:
        zh, _c = RISK_META[f.risk.value]
        overview_rows.append({
            "参数 / 指标": f.display_name,
            "论文声明值": f.paper.raw_value if f.paper else "—",
            "📄 论文出处": f.paper.location if f.paper else "未声明",
            "代码实现值": f.code.raw_value if f.code else "—",
            "💻 代码位置": f.code.location if f.code else "未找到",
            "状态": f"{STATUS_ICON[f.status]} {STATUS_ZH[f.status]}",
            "风险": zh,
            "证据自证": "✅" if f.verified else "❌",
        })
    with st.expander(f"🔗 证据链总览（{len(audit.findings)} 项：{len(issues)} 项风险 + {ok_count} 项一致，全部可追溯）",
                     expanded=True):
        _table(overview_rows, use_container_width=True, hide_index=True)
        env = audit.code.requirements
        if env:
            env_str = "　".join(f"`{k}{'==' + v if v else ''}`" for k, v in list(env.items())[:8])
            st.caption(f"🧰 环境依赖证据（出处: requirements.txt）: {env_str}")

    _t_risk, _t_ok, _t_trust = st.tabs(
        [f"⚠️ 风险项 ({len(issues)})", f"✅ 一致性证据 ({ok_count})", "🔎 审计可信度"])
    with _t_trust:
        _render_trust_panel(audit)
    with _t_risk:
        if not issues:
            st.success(f"✅ 未发现复现风险：{ok_count} 项审计项全部通过规则验证")
            st.markdown("👉 请切换到 **「✅ 一致性证据」** 标签页查看所有通过项的完整证据链"
                        "（论文页码·章节 + 代码文件:行号 + 归一化验证）")
        for f in issues:
            _render_finding(f)
    with _t_ok:
        if not consistent:
            st.info("本期没有一致性项")
        st.caption("以下审计项经程序规则验证一致，证据链同样完整可追溯：")
        for f in consistent:
            _render_finding(f)


elif _page == "multi":
    st.markdown('<div class="sec"><span class="sec-h">🧬 ③ 验证站 · 多智能体对撞</span><span class="sec-sub">三方独立抽取 → 对撞 → 仲裁 → 红队；隔离靠代码强制，不靠提示词</span></div>', unsafe_allow_html=True)
    # ---- 第三期: 多智能体对撞区（只有走图版审计时才有数据，经典版自动不显示）
    _recs = _ex.get("records") or []
    _ROLE_KEYS = ("roles", "collider", "arbiter", "redteam", "report_findings")
    if any((r or {}).get("key") in _ROLE_KEYS for r in _recs):
        _role_recs = [r for r in _recs if (r or {}).get("key") in _ROLE_KEYS]
        _cc = _ex.get("crosscheck") or []
        _agree = sum(1 for x in _cc if (x or {}).get("verdict") == "互证")
        _diffs = [x for x in _cc if (x or {}).get("verdict") == "分歧"]
        _only_c = [x for x in _cc if (x or {}).get("verdict") == "仅对撞"]
        _head = (f"🧬 多智能体对撞区（三方独立抽取 → 对撞 → 仲裁 → 红队 · "
                 f"互证 {_agree} / 分歧 {len(_diffs)}")
        if _only_c:
            _head += f" / 对撞新增 {len(_only_c)}"
        # 有分歧时默认展开 —— 分歧是这条流水线最该被看见的输出；
        # 全互证时折叠，不占版面（结论已被 align 基线代表）。
        with st.expander(_head + "）", expanded=bool(_diffs)):
            st.caption(
                "三个角色各只看自己该看的材料——隔离在代码层面强制（读不到对方的字段），"
                "不是靠提示词求它别看。对撞与仲裁是纯规则、不调模型，所以可复现、可做消融对照。")
            # 关键结论用纯文本再说一遍：表格是画布渲染，读屏/复制/测试都取不到文字，
            # 而"分歧了哪几个参数"恰恰是这块最该被一眼看到的输出。
            if _diffs:
                st.warning("⚠️ 分歧参数：" + "、".join(
                    str(x.get("param_key")) for x in _diffs)
                    + "　—— 两条独立路径给出了不同结论，已如实并列保留，未做静默合并。")
            elif _cc:
                st.success("两条独立路径（align 基线 / 对撞区）结论一致，结论被互证。")
            st.markdown("**角色执行留痕**")
            _table(
                [{"角色": {"roles": "解析员 / 代码验证员 / 数值验证员",
                           "collider": "对撞（并排比对）",
                           "arbiter": "仲裁（按证据强度裁决）",
                           "redteam": "红队（专挑漏洞）",
                           "report_findings": "双路核对"}.get(r["key"], r["key"]),
                  "结果": {"done": "✅ 完成", "skipped": "⏭️ 闸门拦住",
                           "failed": "❌ 失败"}.get(r["status"], r["status"]),
                  "说明": r.get("reason") or "—",
                  "闸门": r.get("gate") or "—",
                  "耗时": f"{r.get('seconds', 0)}s"}
                 for r in _role_recs],
                use_container_width=True, hide_index=True)

            if _cc:
                st.markdown("**双路核对：align 基线 vs 对撞区独立结论**")
                st.caption(
                    "这是两条互相独立的推理路径：align（参数归一化+交叉匹配）与对撞区"
                    "（三方角色独立抽取）。两边都对上＝结论被互证；对不上＝**分歧本身也是发现**，"
                    "如实列出而不是让后跑的那条静默覆盖。")
                _table(
                    [{"参数": x.get("param_key"),
                      "核对结果": {"互证": "🤝 互证", "分歧": "⚠️ 分歧",
                                   "仅基线": "基线独有", "仅对撞": "对撞独有"
                                   }.get(x.get("verdict"), x.get("verdict")),
                      "基线（align）": {"consistent": "一致", "inconsistent": "不一致",
                                        "missing_in_code": "代码未找到",
                                        "missing_in_paper": "论文未声明",
                                        "not_fixed": "未固定",
                                        "unverifiable": "无法静态核验",
                                        "internal_inconsistent": "内部矛盾"
                                        }.get(x.get("baseline"), x.get("baseline") or "—"),
                      "对撞区": {"consistent": "一致", "inconsistent": "不一致",
                                 "missing_in_code": "代码未找到",
                                 "missing_in_paper": "论文未声明",
                                 "not_fixed": "未固定",
                                 "unverifiable": "无法静态核验",
                                 "internal_inconsistent": "内部矛盾"
                                 }.get(x.get("collision"), x.get("collision") or "—")}
                     for x in _cc],
                    use_container_width=True, hide_index=True)

            _arb = _ex.get("arbitrations") or []
            if _arb:
                st.markdown("**仲裁裁决（按证据强度，不是简单投票）**")
                _table(
                    [{"参数": a.get("param_key"),
                      "裁决": {"维持冲突": "⚠️ 维持冲突", "降级为疑似": "🔸 降级为疑似",
                               "撤销冲突": "✅ 撤销冲突", "升级": "🚨 升级"
                               }.get(a.get("verdict"), a.get("verdict")),
                      "理由": a.get("reason") or "—",
                      "置信": {"confirmed": "已确认", "inferred": "疑似"
                               }.get(a.get("confidence"), a.get("confidence") or "—"),
                      "需人工确认": "是" if a.get("requires_human") else "否"}
                     for a in _arb],
                    use_container_width=True, hide_index=True)

            _rt = _ex.get("red_team") or []
            if _rt:
                st.markdown("**红队笔记（对抗式验证，专挑结论漏洞）**")
                st.caption("挑漏洞的维度全部是程序能查的（证据缺出处 / 单侧定案 / 数值离谱 / "
                           "同义词未归并 / 置信不足），不是模型\"觉得\"。")
                _table(
                    [{"参数": n.get("param_key"),
                      "漏洞类型": n.get("flaw") or "—",
                      "严重度": {"high": "🔴 高", "medium": "🟠 中",
                                 "low": "🟡 低"}.get(n.get("severity"), n.get("severity") or "—"),
                      "细节": n.get("detail") or "—"}
                     for n in _rt],
                    use_container_width=True, hide_index=True)

elif _page == "loop":
    st.markdown('<div class="sec"><span class="sec-h">🔁 ④ 迭代站 · 复现闭环</span><span class="sec-sub">搜 / 算 / 做 + 迭代回边 + 根因追查 + 补丁草案；全为纯规则、离线可跑</span></div>', unsafe_allow_html=True)
    # ---- 阶段五（M2 闭环扩展）: 搜 / 算 / 做 + 迭代回边
    # 只有走闭环链时才有数据（经典版与纯多智能体版都不显示），自动判断、不空占版面。
    _cmp = _ex.get("compute") or {}
    _cit = _ex.get("citations") or {}
    _rpl = _ex.get("repro_plan") or {}
    _lp = _ex.get("loop") or {}
    if _cmp or _cit or _rpl or _lp:
        _grade = (_rpl.get("grade") or {})
        _lvl = _grade.get("level") or "—"
        _unres = _lp.get("unresolved") or []
        # 有"待人工确认项"或评分离及格线近时默认展开
        _head = (f"🔁 复现闭环（搜 / 算 / 做 + 迭代回边）"
                 f" · 可复现性 **{_lvl}（{_grade.get('score', '—')}/100）**")
        if _unres:
            _head += f" · 待人工确认 {len(_unres)} 项"
        with st.expander(_head, expanded=bool(_unres)):
            st.caption(
                "这一段的四个环节**全是程序规则、不调模型、离线可跑**：不开 API key 也能用，"
                "同一份材料永远得出同一个结果。「算」只做小规模复算与静态推演，"
                "**不执行任何训练、不运行你的代码**；「迭代回边」里的「修正」是内存里的推演，"
                "**不写任何文件、不改你的代码**。")

            # ---- 关键结论先用文字说一遍（下面的表格是画布渲染，读屏/复制都取不到文字）
            _rels = _cmp.get("relations") or []
            _bad = [r for r in _rels if r.get("verdict") == "mismatch"]
            _hint = [r for r in _rels if r.get("verdict") == "hint"]
            _mag = _cmp.get("magnitudes") or []
            _sum = []
            if _rels:
                _sum.append(f"「算」复算 {_cmp.get('checked', 0)} 条关系式"
                            + (f"，其中 {len(_bad)} 条换算后仍对不上" if _bad else "")
                            + (f"，{len(_hint)} 条给出推测性解释" if _hint else "")
                            + (f"；{_cmp.get('skipped', 0)} 条因缺输入**未算**（不用默认值补数）"
                               if _cmp.get("skipped") else ""))
            if _mag:
                _sum.append("「算」量纲提示：" + "、".join(
                    f"{m.get('param')}（{m.get('detail', '')}）" for m in _mag[:3]))
            _cov = _cit.get("coverage") or {}
            if _cov:
                _sum.append(f"「搜」引证覆盖 {_cov.get('cited', 0)}/{_cov.get('total', 0)}"
                            f"（{(_cov.get('cited_ratio') or 0):.0%}）"
                            + (f"，其中 {_cov.get('mismatched')} 条引文与原文不符"
                               if _cov.get("mismatched") else ""))
            _ext = _cit.get("external") or {}
            if _ext.get("items"):
                _sum.append(f"外部文献候选 {len(_ext['items'])} 条"
                            f"（{_ext.get('reached', 0)}/{_ext.get('of', 0)} 个源返回，均未核验）")
            if _rpl:
                _sum.append(f"「做」可复现性评级 **{_lvl}**"
                            f"（{_grade.get('score', '—')}/100，"
                            f"{len(_grade.get('deductions') or [])} 条扣分理由）")
            if _lp:
                _sum.append(f"「迭代回边」{len(_lp.get('rounds') or [])} 轮内收敛 "
                            f"{_lp.get('resolved', 0)} 条，"
                            f"升级为待人工确认 {len(_unres)} 条")
            if _sum:
                st.info("　".join(_sum))
            if _unres:
                st.warning("⚠️ 三轮推演后仍未收敛（升级为**待人工确认项**，系统不硬凑）："
                           + "、".join(str(u.get("display")) for u in _unres))

            # ---- 算
            if _rels or _cmp.get("plausibility"):
                st.markdown("**「算」小规模复算与静态推演**")
                st.caption("把已知的数代进已知公式核对。缺输入一律记「未算」——"
                           "**不用默认值补出一个数**；带假设的推测标「需人工确认」，"
                           "不与确定性结论混在一起。")
                if _rels:
                    _VL = {"consistent": "✅ 换算后一致", "mismatch": "⚠️ 复算对不上",
                           "hint": "💡 推测（需人工确认）", "info": "ℹ️ 推演（不作为结论）",
                           "skipped": "— 未算"}
                    _table(
                        [{"关系式": r.get("formula") or "—",
                          "结论": _VL.get(r.get("verdict"), r.get("verdict") or "—"),
                          "可信度": {"confirmed": "确凿", "inferred": "推测",
                                     "none": "—"}.get(r.get("confidence"),
                                                      r.get("confidence") or "—"),
                          "说明": r.get("detail") or "—"}
                         for r in _rels],
                        use_container_width=True, hide_index=True)
                for _p in (_cmp.get("plausibility") or []):
                    st.caption(f"· 数值合理性：`{_p.get('param')}`（{_p.get('side')} 侧 "
                               f"{_p.get('value')}）—— {_p.get('note')}")

            # ---- 搜
            if _cov:
                st.markdown("**「搜」文献依据检索与引证核验**")
                st.caption(f"检索方式：{_cit.get('mode', '字面检索')}。"
                           "检索不到的如实记「未检索到」，不拿不相关段落充数。")
                st.markdown(
                    f"- 结论总数 **{_cov.get('total', 0)}**，其中 "
                    f"**{_cov.get('cited', 0)}** 条有可定位出处"
                    f"（论文页码 或 代码 file:line）\n"
                    f"- 已检索到相关段落 **{_cov.get('with_retrieved', 0)}** 条\n"
                    f"- 仅 AI 推断、无原文出处 **{_cov.get('inferred_only', 0)}** 条\n"
                    f"- 引文与原文不符 **{_cov.get('mismatched', 0)}** 条")
                _items = _cit.get("items") or []
                _with = [it for it in _items if it.get("n_citations")]
                if _with:
                    _table(
                        [{"参数": it.get("display") or it.get("param_key"),
                          "一级出处": it.get("primary_source") or "—",
                          "检索到的段落": (it["citations"][0].get("snippet") or "")[:110]}
                         for it in _with[:12]],
                        use_container_width=True, hide_index=True)
                _lit_ext = _cit.get("external") or {}
                _lit = _lit_ext.get("items") or []
                if _lit:
                    st.markdown("**「搜」外部文献候选（真实多源检索）**")
                    st.caption(
                        f"6 个公开学术索引里查了 {_lit_ext.get('reached', 0)}/"
                        f"{_lit_ext.get('of', 0)} 个，返回 {len(_lit)} 条。"
                        "每条都能点开核（DOI / 落地页），但**没有一条被系统判为"
                        "「支持某条结论」** —— 那要你读完摘要自己决定。")
                    # 铁律：喂 _table 的值一律同一类型（这里全转字符串，
                    # 否则 None 与 int 混排会让 PyArrow 抛 ArrowTypeError，整页不渲染）
                    _table(
                        [{"年份": str(it.get("year") or "—"),
                          "标题": str(it.get("title") or "")[:90],
                          "收录库": "、".join(it.get("found_in") or []) or "—",
                          "DOI": str(it.get("doi") or "—"),
                          "链接": str(it.get("url") or "")} for it in _lit[:10]],
                        use_container_width=True, hide_index=True)
                    st.caption("点「链接」列可在浏览器打开；文献真实存在，"
                               "但相关性只做到「关键词命中 + 跨源互证」这一层。")
                elif _lit_ext.get("enabled"):
                    st.caption(f"· 外部文献：{_lit_ext.get('note') or '本次没有检索到相关文献'}")
                for _n in (_cit.get("notes") or [])[:3]:
                    st.caption(f"· {_n}")

            # ---- 做
            if _rpl:
                st.markdown("**「做」复现方案与可复现性评级**")
                _deds = _grade.get("deductions") or []
                if _deds:
                    st.caption("扣分理由（每一条都可复核，改一处材料后重新生成，分数会跟着变）：")
                    for _d in _deds[:10]:
                        st.markdown(f"- −**{_d.get('points')}** {_d.get('reason')}")
                _steps = _rpl.get("steps") or []
                for _s in _steps:
                    _flag = " **(必做)**" if _s.get("blocking") else ""
                    st.markdown(f"**{_s.get('n')}. {_s.get('title')}**{_flag}")
                    st.caption((_s.get("detail") or "")[:600])
                if _rpl.get("markdown"):
                    st.download_button(
                        "⬇️ 下载完整复现方案（Markdown）",
                        data=_rpl["markdown"].encode("utf-8"),
                        file_name="REPRO_PLAN.md", mime="text/markdown",
                        use_container_width=True,
                        help="含环境 / 数据 / 有序步骤 / 参数对照表 / 评级与扣分理由 / 待确认事项")

            # ---- 迭代回边
            if _lp:
                st.markdown("**「迭代回边」（对不上 → 定位根因 → 修正 → 再验证）**")
                st.caption(f"有界自主：上限 {_lp.get('max_rounds', 3)} 轮，"
                           "任何一轮没有改善就立即停止。「修正」是**内存里的反事实推演** ——"
                           "不写任何文件、不改你的代码；补丁草案只生成不落盘。")
                _rounds = _lp.get("rounds") or []
                for _r in _rounds:
                    st.markdown(f"**第 {_r.get('round')} 轮 · {_r.get('title')}**"
                                f" —— 尝试 {_r.get('tried', 0)} 条，"
                                f"收敛 {_r.get('improved', 0)} 条，"
                                f"仍未解决 {_r.get('remaining', 0)} 条")
                    _rows = []
                    for _it in (_r.get("items") or []):
                        _rows.append({
                            "参数": _it.get("display"),
                            "推演前": _it.get("before"),
                            "推演后": _it.get("after"),
                            "假设改成": _it.get("attempt") or "—",
                            "位置": _it.get("target_loc") or "—",
                            "结论": "✅ 收敛" if _it.get("resolved") else "· 未收敛",
                            "说明": (_it.get("reason") or "")[:150],
                        })
                    if _rows:
                        _table(_rows, use_container_width=True, hide_index=True)
                _ad = (_lp.get("auto_decisions") or [])
                if _ad:
                    st.markdown("**自主回头** —— 走不走由模型定，每一句都要过闸门：")
                    for _i, _d in enumerate(_ad, 1):
                        _act = _d.get("llm_action") or _d.get("action") or ""
                        _as = _d.get("assume") or {}
                        if _act == "retry":
                            _what = (f"再算一遍（`{_as.get('key') or '—'}` 的 "
                                     f"{_as.get('side') or '—'} 侧设成 "
                                     f"{_as.get('value') or '—'}）")
                        else:
                            _what = "收尾"
                        _tag = ("✅ 采纳" if _d.get("action") == "retry"
                                else ("⛔ 拦下" if _d.get("source") == "gate" else "收尾"))
                        _line = (f"第 {_i} 次：模型想{_what} → {_tag}。"
                                 f"{_d.get('note') or ''}")
                        if _d.get("outcome"):
                            _line += f"；事后核账：{_d['outcome']}"
                        st.caption(_line)
                if _lp.get("markdown"):
                    st.download_button(
                        "⬇️ 下载迭代推演留痕（Markdown）",
                        data=_lp["markdown"].encode("utf-8"),
                        file_name="ITERATION_TRACE.md", mime="text/markdown",
                        use_container_width=True,
                        help="逐轮的假设、复验结果与未收敛清单（含两侧原值）")
    # ---- 第三期: 根因追查 + 补丁草案（2026-09-28 起由流水线里的两站产出）
    # 之前这两件事只在 CLI 汇总脚本里算、界面一个字都没有 —— 用户问「补丁为什么被拒」时
    # 产品答不上来。现在它们与其它环节同权：这里能看结论、看 diff、看每一条拒绝理由。
    _tr = list(getattr(audit, "rootcause", None) or [])
    _pt = list(getattr(audit, "patches", None) or [])
    if _tr or _pt:
        _p_ok = [p for p in _pt if p.is_proposed]
        _p_no = [p for p in _pt if not p.is_proposed]
        _p_ai = sum(1 for p in _p_ok if getattr(p, "source", "rule") == "llm")
        _loc = sum(1 for t in _tr if getattr(t, "confidence", "none") != "none")
        with st.expander(
                f"🔧 根因追查与补丁草案（可定位 {_loc}/{len(_tr)} · "
                f"补丁给方案 {len(_p_ok)} / 拒绝 {len(_p_no)}"
                + (f" · 其中 {_p_ai} 条目标由 AI 提名" if _p_ai else "") + "）："):
            st.caption("补丁只展示、**绝不落盘**；只改配置项的值，拿不准的一律拒绝并写明理由。")
            if _tr:
                st.markdown("**根因追查**（只读代码，追不出就明说「无法定位」）")
                _table(
                    [{"参数": t.display,
                      "最终生效值": (t.verdict
                                     if getattr(t, "confidence", "none") != "none" else "—"),
                      "位置": (t.verdict_loc or "—"),
                      "置信": t.confidence,
                      "依据": (t.reason or "")[:120]}
                     for t in _tr],
                    use_container_width=True, hide_index=True)
            if _p_ok:
                st.markdown("**给方案的补丁**（原行与改后行都由程序生成，是否采纳由你定）")
                for _p in _p_ok:
                    _ai = getattr(_p, "source", "rule") == "llm"
                    st.markdown(("⚠️ " if _ai else "") + f"`{_p.target}`　**{_p.display}**"
                                + ("　—　**目标由 AI 提名，采纳前须人工确认**" if _ai else ""))
                    st.code(_p.diff or "", language="diff")
                    st.caption(_p.rationale)
                    for _s in (getattr(_p, "stamps", None) or []):
                        st.caption(f"　程序盖章：{_s}")
            if _p_no:
                st.markdown(f"**拒绝的补丁（{len(_p_no)} 条，逐条写明理由）**")
                for _p in _p_no:
                    st.caption(f"⛔ {_p.display} —— {_p.refuse_reason}")

elif _page == "ask":
    st.markdown('<div class="sec"><span class="sec-h">❓ 待确认问题</span><span class="sec-sub">程序判不了的部分如实问你；答复只记录，不自动改写结论</span></div>', unsafe_allow_html=True)
    # ---- 第二期第二批: 待确认问题（可直接填写答复, 答复只作为人工确认记录）
    try:
        from paper2lab.inquiry import (KIND_LABEL, apply_clarifications,
                                       build_questions)
        _qs = build_questions(audit)
    except Exception:      # noqa: BLE001
        _qs = []
    if _qs:
        _answered = len(audit.clarifications or [])
        with st.expander(
                f"❓ 待确认问题（{len(_qs)} 项，需要你判断"
                + (f" · 已确认 {_answered} 项" if _answered else "") + "）"):
            st.caption("答复写进报告作为人工确认记录：只记录，不自动改写结论。")
            _answers = {}
            for _q in _qs:
                st.markdown(f"**[{KIND_LABEL.get(_q.kind, _q.kind)}] {_q.title}**")
                st.caption(f"为什么问：{_q.basis}")
                for _e in _q.evidence:
                    st.caption(f"依据：{_e}")
                _answers[_q.qid] = st.text_input(
                    "你的答复", key=f"clar_{_q.qid}",
                    placeholder=" / ".join(_q.suggestions) if _q.suggestions else "")
                st.caption(f"不回答的影响：{_q.impact}")
                st.divider()
            if st.button("📝 应用澄清（重新生成报告）", key="apply_clar"):
                _rec = apply_clarifications(audit, _answers)
                if _rec:
                    from paper2lab.report import generate_report
                    st.session_state["report"] = generate_report(audit)
                    st.session_state["clar_done"] = len(_rec)
                    st.rerun()
                else:
                    st.info("尚未填写任何答复。")
            if st.session_state.get("clar_done"):
                st.success(f"已记录 {st.session_state.pop('clar_done')} 条人工确认，"
                           f"报告已同步更新。", icon="✅")

elif _page == "exec":
    st.markdown('<div class="sec"><span class="sec-h">📜 执行记录</span><span class="sec-sub">跑了什么、跳了什么、为什么；谁排的计划、被哪道闸门拦下</span></div>', unsafe_allow_html=True)
    # ---- 第二期第一批: 执行记录（两种模式各自如实交代"跑了什么、跳了什么、为什么"）
    _ex = getattr(audit, "execution_log", None) or {}
    if _ex:
        # 2026-10-25 删掉「经典模式」（不看材料、固定全跑）后只剩一条路:
        # 先排计划、再执行。这里 _planned 恒为真，只为让下面"把决策摊开"的逻辑少改。
        _planned = True
        _skipped = _ex.get("skipped", 0)
        _planner_kind = (_ex.get("plan") or {}).get("planner", "rule")
        _tag = ("🧠 模型排计划" if _planner_kind == "llm"
                else "🧭 规则版计划（模型未参与，如实标注）")
        _verdict = (f"按材料排计划：跳过 {_skipped} 个工具（材料不齐或模型判断无需跑）"
                    if _skipped else "材料齐备，计划内全部执行")
        with st.expander(
                f"{_tag} · 执行记录（执行 {_ex.get('ran', 0)} 个工具 / "
                f"跳过 {_skipped} 个 / 覆盖 {_ex.get('nodes_total', 0)} 个环节 · "
                f"{_ex.get('seconds', 0)}s）", expanded=True):
            st.caption(_verdict)

            # 计划模式: 把调度员**看了什么、据此下了什么决策**摊开
            if _planned:
                _plan = _ex.get("plan") or {}
                _dec = _plan.get("decisions") or []
                if _dec:
                    st.markdown(("**模型的决策**" if _planner_kind == "llm"
                                 else "**规则兜底的决策**")
                                + ("　（每一步都记着：材料事实 / 模型判断 / "
                                   "模型没表态时的默认处理；1C 起程序不再否决模型）"
                                   if _planner_kind == "llm" else ""))
                    _table(
                        [{"工具": d.get("title"),
                          "决策": "▶️ 执行" if d.get("action") == "run" else "⏭️ 跳过",
                          "来源": {"rule": "材料事实",
                                   "llm": "AI 判断"}.get(d.get("source"), "—"),
                          "理由": d.get("reason") or "—"}
                         for d in _dec],
                        use_container_width=True, hide_index=True)

            st.markdown("**实际执行**")
            _table(
                [{"环节": " → ".join(_s.get("checkpoints") or []) or "—",
                  "工具": _s["title"],
                  "结果": {"done": "✅ 执行", "skipped": "⏭️ 计划跳过",
                           "failed": "❌ 失败"}.get(_s["status"], _s["status"]),
                  "这次实际做出了什么": (_s["reason"]
                                     if _s["status"] != "done"
                                     else _tool_outcome(audit, _s["key"])),
                  "耗时": f"{_s['seconds']}s"}
                 for _s in _ex.get("steps", [])],
                use_container_width=True, hide_index=True)
            for _n in _ex.get("notes", []):
                st.caption(f"· {_n}")
            st.caption(f"{len(STEPS)} 环节 / {len(TOOLS)} 工具"
                       "（参数归一化 · 交叉匹配 · 结果核验 合为一个工具）")

elif _page == "report":
    _rc1, _rc2 = st.columns(2)
    with _rc1:
        st.download_button("⬇️ 下载审计报告 (Markdown)", report_md,
                           file_name="paper2lab_audit_report.md",
                           mime="text/markdown", use_container_width=True)
    with _rc2:
        _repro_md = ""
        try:
            from paper2lab.repro import build_repro_pack
            _repro_md = build_repro_pack(audit).to_markdown()
        except Exception:      # noqa: BLE001
            _repro_md = ""
        if _repro_md:
            st.download_button("📋 下载复现准备清单 (Markdown)", _repro_md,
                               file_name="paper2lab_repro_pack.md",
                               mime="text/markdown", use_container_width=True)
        else:
            st.caption("（复现准备清单生成失败）")
    st.caption("复现准备清单：环境依赖、超参对照（带出处）、待确认事项、已知风险。")
    st.markdown(report_md)
