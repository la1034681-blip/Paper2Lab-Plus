"""全套监控 —— 依次跑全部检查项, 记录耗时/退出码/报错关键词, 输出汇总报告。

用法: .venv\\Scripts\\python.exe run_all_checks.py
输出: Desktop\\Paper2Lab测试集\\monitor_suite_report.md

设计要点:
- 每项作为子进程运行, 单独设置超时(默认 900s), 超时如实标记 TIMEOUT;
- 记录 stdout+stderr, 并按关键词扫描异常(Traceback/❌/⚠️/异常/未通过/ERR);
- 先跑离线与流水线检查, 最后跑会重启服务的 UI 检查(避免互相干扰)。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
OUT_DIR = Path(r"C:\Users\lijiahui\Desktop\Paper2Lab测试集")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 全套体检**默认关掉外部文献检索**（2026-10-19）：
# 一次真查要 5~11 秒且依赖外网，而几十项检查都会跑到「搜」节点 ——
# 让体检结果取决于"今天网通不通"是错的（会把网络抖动记成程序问题）。
# 专项 `tests/verify_literature.py` 内部会把它显式打开，并用离线桩保证可跑。
os.environ["PAPER2LAB_LIT"] = "off"

# 本机服务探测必须**绕开 HTTP 代理**（2026-10-29 踩坑）。
# 沙箱 / 公司网络 / 抓包工具常设 `HTTP_PROXY`，而 `urllib.request.urlopen` 会照着走 ——
# 于是探针对 `http://127.0.0.1:8501/_stcore/health` 拿到 **502 Bad Gateway**，
# 判定「服务没起来」，接着重复起实例、连到错的那个实例，最后整项报「页面没渲染」。
# 实测代价：4 项界面检查全挂（其中一项白等 405 秒），而手工开服务后同样代码全过。
# `no_proxy` 是 urllib 认的标准变量；这里给足三种本机写法，并保留用户原有配置。
_np = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
_hosts = [h.strip() for h in _np.split(",") if h.strip()]
for _h in ("127.0.0.1", "localhost", "::1"):
    if _h not in _hosts:
        _hosts.append(_h)
os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(_hosts)

CHECKS: list[tuple[str, list[str], int]] = [
    ("五套真实论文样本审计",        ["tests/monitor_samples.py"], 1800),
    # ↑ 超时 2026-10-19 由 900 提到 1800: 五套里有 E 题（4.8MB 扫描版）。
    #   **缓存冷/热的耗时差三个数量级**（E 题清缓存后真读图单套就 10 分钟+），
    #   拿 900s 卡它会把"环境冷"误判成"程序坏"。

    # 超时给到 1200s（2026-10-18 实测）: E题论文 4.8MB 是扫描版, 跑真实读图时
    # **单这一套**就要吃掉 10 分钟以上; 缓存命中时才 3s —— 拿 600s 卡它会假超时。
    ("demo / D题 / E题 回归",       ["tests/verify_sets.py"], 1200),
    ("边缘集(诱饵值拒收)",          ["tests/verify_edge.py"], 600),
    ("图像通道 开关对比(mock)",     ["tests/verify_vision.py"], 600),
    ("图像通道 真实读图",           ["tests/verify_vision_real.py"], 900),
    # 覆盖策略: 默认读「文本稀薄页」+「正文提及参数/配置页的内嵌插图」。
    # 2026-09-27 实测: 八套样本共 128 个候选区域中 112 个(88%)是"文本已完整解析、
    # 且正文完全没提参数"页面上的插图(几乎必然是结果曲线/示意图) —— NeRF 的 103 个
    # 此类区域**全部零产出**却耗时 583s。设 PAPER2LAB_VISION_READ_ALL=1 可恢复全读。
    ("图像通道 默认覆盖策略",       ["tests/verify_vision_more.py", "0"], 900),
    ("DeepSeek 真实归因",           ["tests/verify_deepseek.py"], 600),
    ("界面 密钥状态与不预填",       ["tests/verify_keys_ui.py"], 600),
    # 模型 API 改造（2026-09-30 用户口径）: 页面输入区是通用的「模型 API」
    # （任意 OpenAI 兼容模型，Key 与 Base URL 成对）；多模态模型可复用模型 API 读图，
    # 单模态则必须单独配视觉 API，否则**如实提示无法读图**。
    # 为什么必须单列一项: 本机 .env 配了视觉 key，"完全没配读图通道"那几条分支
    # 在页面/真机上永远走不到 —— 靠页面点一遍是验证不到的，必须在函数层锁死。
    ("模型 API 与读图通道判定",     ["tests/verify_model_api.py"], 600),
    ("界面 结构与泄漏检查",         ["tests/verify_ui_live.py"], 600),
    ("界面 三路径真机实测",         ["tests/ui_test.py"], 900),
    # 端到端总验收: 像用户一样调用(不预传 key, 走 .env 兜底), 跑真实样本的完整流程。
    # 为什么必须固定跑这一项: 单元级验证脚本都是"作者自己写的", 会顺手传对参数、
    # 从而绕开真实调用路径 —— 2026-09-27 实测, 3 个缺陷(调度员误判跳过读图、Agent 拿不到
    # LLM key、读图问题清单类型错误)在单元级全绿的情况下依然存在, 端到端一跑才暴露。
    ("全流程端到端(真实样本·计划模式)", ["tests/demo_full_flow.py"], 900),
    # 新增守卫的快速回归: 分布参数守卫(分布均值不是取值; 设置型调用要放过)、
    # 读图语境守卫(默认不读"文本完整且未提参数"页的插图; READ_ALL=1 可恢复)、
    # D题种子不得被误补回。2026-09-27 由"全量跑真实样本"发现并修复, 固化为检查防回退。
    # 2026-09-28 并入第⑥节: 精度守卫(压平表格里的数字 / 方括号参考文献编号 / 论文说"迭代次数"
    # 而代码是 EPOCHS 时不得出补丁) —— 这三道由新样本集 v2 的"错补丁"逼出来。
    ("新增守卫 分布/读图/取值精度",  ["tests/verify_guards.py"], 300),
    # 「执行记录」表的两列（环节 / 这次实际做出了什么）由 paper2lab/execview.py 派生；
    # 界面里这张表是 canvas 渲染, DOM 读不到单元格, 所以必须在数据层断言。
    # 另: 界面级验证见 verify_modes_ui.py（四种组合, 约 75s, 按需跑, 未纳入本套）。
    # 超时 2026-10-19 由 300 提到 600: 这一项要跑 5 次真实审计，其中 04_PointNet
    # 的读图环节冷缓存下单独就 68s（实测），5 次叠起来 150~250s，
    # 300s 没余量 —— 之前那轮就是卡在这里。
    ("执行记录 环节与产出派生",    ["tests/verify_execview.py"], 600),
    # 计划确定性（2026-10-25 改）: 删除「经典模式」后，这一项守的是更要紧的一条 ——
    # **同一份材料跑两次，计划与结论必须一模一样**（执行序列、跳过理由、结论签名、
    # 就绪度、可信度逐项对比）。审计平台最怕的不是"跳过了某步"，而是每次跑结果都不一样。
    # 「调度不改变结论」的另一半证据在 verify_llm_planner.py（模型跳 13 件仍出结论）。
    # 超时 1500: 8 套真实材料 × 2 次 = 16 次审计，冷缓存下 PointNet 单套读图就 68s。
    ("计划确定性 跑两遍一致 8套",  ["tests/verify_ablation_modes.py"], 1500),
    # 无代码路径: 代码 ZIP 已是**可选**的常规用法, 必须验证"跳过代码扫描"后
    # repro / rootcause / patch / inquiry 四个下游模块都扛得住, 且不凭空给出补丁。
    ("无代码路径 下游模块健壮性",  ["tests/verify_nocode_flow.py"], 600),
    # 就绪度「未核验」语义: 没做核验的维度必须显式标未核验、且不进总分。
    # 2026-09-28 修正 —— 从前"没提供实测结果"会让「结果一致性」显示 100 分(绿色)、
    # 「实验记录完整度」固定给 60 分, 等于把"用户没上传"算成论文的问题。
    # 这条口径影响所有展示的分数, 因此逐样本(8 套)断言, 防回退。
    ("就绪度 未核验语义 8套",      ["tests/verify_readiness_semantics.py"], 600),
    # 第三期三件（复现清单 / 根因追查 / 补丁草案）此前**只在报告之外的脚本里跑**,
    # 没进全套, 也就没有防回退保护。2026-09-28 用户点出"这不是执行 agent 的责任吗"后,
    # 三件已作为流水线环节接入（见 toolbox 的 rootcause / patch 两站）, 与之一并纳入回归。
    # 其中 verify_patch.py 还带一道硬自检: **跑完代码包哈希必须一字不变**（补丁绝不落盘）。
    ("复现清单 出处与建议",        ["tests/verify_repro.py"], 600),
    ("根因追查 只读且可回溯源",     ["tests/verify_rootcause.py"], 600),
    ("补丁草案 闸门与绝不落盘",     ["tests/verify_patch.py"], 600),
    # 模型参与决策（2026-09-28 建，2026-10-15 阶段一 1C 翻转口径）:
    # ① 排计划 —— 现在**决定权在模型**（程序不设闸门，只递菜单/留痕/如实提示）；
    # ② 补丁提名目标 —— 仍归程序盖章（管的是"补丁内容"，不是"调不调工具"）。
    # **不调 API** —— 用注入的提案测程序这一侧的行为。确定性的东西不能靠模型当天心情。
    ("1C 模型排计划 无程序闸门",    ["tests/verify_llm_planner.py"], 600),
    # 两轮之间的清理口径（2026-09-29 用户要求）: 点下一次"开始审计"时,
    # 必须清掉上一轮的**运行缓存**与**上传临时文件**, 但跨任务记忆
    # （cache/audit_memory.json）必须一字不动 —— 它是记忆不是缓存。
    ("两轮清理 清运行态保记忆",     ["tests/verify_housekeeping.py"], 300),
    # 网页端默认即 agent 模式（2026-09-29 用户口径）: 输入材料启动，默认由模型
    # 参与决定跑哪些环节，不跑的环节在流程面板标「未执行」。
    # 真机断言两件事: ① 页面上没有「要不要 AI」的开关；② 执行记录里确实走了模型排计划
    # （若模型不可用，必须**如实说明已回退规则版**，不许悄悄退回还说自己是模型排的）。
    ("网页端默认 agent 模式",       ["tests/verify_agent_default.py"], 600),
    # 阶段五（M2 闭环扩展）: 「搜 / 算 / 做」+ 迭代回边。四个模块全是纯规则、
    # 零成本、离线可跑，所以必须进全套回归——否则"不开 key 也能用的能力"反而没人守。
    # 其中 verify_loop.py 带一条硬自检: 跑完**代码包哈希必须一字不变**（反事实推演）。
    ("「算」小规模复算与静态推演",   ["tests/verify_compute.py"], 300),
    ("「搜」引证检索与核验",        ["tests/verify_retrieval.py"], 300),
    # 阶段四（4A）: 外部文献检索从"让模型背课文"换成**真查 6 个公开索引**。
    # 这一项自带离线桩（每个源的响应 fixture 都是从真实响应裁下来的），
    # 所以断网也能跑；末尾那段真网加测在连不上时**自动跳过并打印**，不判红。
    ("「搜」外部多源文献检索",      ["tests/verify_literature.py"], 300),
    # 2E-化学: 分子式 / CAS / SMILES / 分子量对账（纯本地规则, 无网络无 key）
    ("「读」化学身份核验",          ["tests/verify_chem.py"], 300),
    # 2E-化学在**真机上看得见吗**（2026-10-24 补）：报告里有不等于页面有。
    # 这一项会真开浏览器、传 demo/chem_demo.pdf、点审计、断言卡片里能看见
    # 7732-18-6 / C8H10N4O2 / 对不上 —— 它抓出过"模型排计划跳过 read_figures,
    # 化学核验跟着消失"这个真 bug，光靠内存断言发现不了。
    ("「读」化学核验 界面真机",      ["tests/verify_chem_ui.py"], 420),
    ("「做」复现方案与可复现性评级", ["tests/verify_repro_plan.py"], 300),
    ("迭代回边 有界且绝不落盘",      ["tests/verify_loop.py"], 300),
    # 端到端: 挂上闭环链后跑真实 demo，做 classic / multagent / multagent+loop 三方
    # 消融对照 —— 「只加能力、不改结论」这条保证必须有对照证据，不能只靠承诺。
    ("阶段五 闭环端到端与消融",      ["tests/verify_closeloop.py"], 600),

    # ---- 阶段六 协议化（2026-10-14）: MCP/SCP 封装 + Agent Skills 打包 ----
    # verify_mcp.py 里有两条**真机**验证: stdio 握手 + Streamable HTTP 握手;
    # 前者还逐行断言 stdout 必须是合法 JSON-RPC（PyMuPDF 会污染它, 是踩过的坑）。
    ("协议化 MCP/SCP 封装",          ["tests/verify_mcp.py"], 600),
    ("协议化 Agent Skills 打包",     ["tests/verify_skill_pack.py"], 600),
    # 阶段七·评测打磨: 契约与书生·司南 SciEvalKit 逐字对齐（直接读官方源码比对）,
    # 判分四维 + 零重依赖自检; 端到端只跑 demo 一套（全量考卷见 tools/run_eval.py）。
    ("阶段七 评测 契约对齐与判分",      ["tests/verify_eval.py"], 600),

    # ---- 阶段一 1B（2026-10-15）: 三角色转正为「七位置」编制 ----
    # 换名分最容易把隔离换掉 —— 那是本项目生死线（自审自），所以隔离必须由
    # **可执行断言**盯着: 论文视野不得看到代码侧独有的键，反之亦然；
    # 并且 Parser ≡ 原 Analyst+CodeVerifier 逐字段相等（换编制不改行为）。
    ("阶段一 七位置编制与隔离",        ["tests/verify_team.py"], 300),
    # ---- 阶段二 2A（2026-10-15）: 假设卡 ----
    # 假设卡最危险的失败模式不是报错, 而是**一本正经地胡说** —— 把本来一致的参数
    # （32 vs 32、5e-4 vs 0.0005）当矛盾提出来。所以"不制造伪矛盾"那组断言是重点；
    # "判据与 needs_rerun 严格一致"则保证 ② 实验站不会拿到一堆静态就能判的活。
    ("阶段二 假设卡可证伪与口径",      ["tests/verify_hypothesis.py"], 300),
    # ---- 阶段二 2B（2026-10-15）: 符号推演与筛排 ----
    # 2A 最危险的失败模式是"一本正经地胡说"; 2B 最危险的是**空转** ——
    # 文件写得很像样, 规则却一条也没命中（第一版在 5 个真实样本 43 张卡上零命中）。
    # 所以这组断言大半是合成用例: 真实材料里能推的太少, 测不到「已排除」「已定位」两条路径。
    ("阶段二 符号推演与筛排",          ["tests/verify_derivation.py"], 300),
    # ---- 阶段二 2E（2026-10-25）: 影响量级（发现型假设） ----
    # 最危险的失败模式不是算错, 是**拿空话充数**: 两侧批大小按"相同"补齐时,
    # 倍数必然是 1, 报出来等于什么都没说（第一版实测在 DDPM / 03_NeRF 上各出现一次）。
    # 所以断言里专有一条守"算不出必须返回空", 另有单卡口径陷阱与口径歧义两条。
    ("阶段二 影响量级推演",            ["tests/verify_impact.py"], 300),
    # ③④ 的界面层：候选解释 / 翻盘 / 边界 / 对照 + 「提示层」口径, 必须在**真页面**上看得见。
    # 与上面那条的分工: 那边守字段与报告文字, 这边守"渲染出来没被吞掉"（组件异常会吞整页）。
    ("影响量级 界面真机",              ["tests/verify_impact_ui.py"], 420),
    # ---- 2026-10-25: 模型先判、程序后验（试点 ① 假设真假 / ③ 参数配对） ----
    # 这个改动最危险的失败模式不是"判错", 是**模型定调、程序背书** ——
    # 程序只复核模型说的那一项, 于是"两侧都抄错了"永远查不出来;
    # 其次是**覆盖**: 把 dict / status 直接改成模型的答案, 下游报告与断言会静默失准。
    # 所以断言守的是: 规则独立跑、结论不被覆盖、不一致两边都留、没 key 时如实记"没判"。
    # 全程离线（注入 ask）, 不花钱。
    ("模型先判程序后验",               ["tests/verify_judge_llm.py"], 180),
    # 2026-10-28 真 bug 的守卫：图状态 TypedDict 漏声明 `judge_llm`，
    # LangGraph 只透传声明过的键 ⇒ 网页默认档（多智能体）这个开关**空转**，
    # 而经典入口一直正常。静态核键 + Job 落地，毫秒级，必须常驻。
    ("图状态 开关不静默丢失",          ["tests/verify_graph_flags.py"], 120),
    # ---- 2026-10-25: 自主回头（L3 回边） ----
    # 最危险的失败模式不是"它不肯回头", 而是**它说什么就执行什么**:
    # 指名一个不存在的参数、把两侧现有值当新值提上来、或者一路回头到死。
    # 所以这组断言的绝大多数是"必须被拦下"的用例, 只有 B 组证明它真的有权。
    ("自主回头 L3 回边",               ["tests/verify_autonomy.py"], 240),
    # 2026-10-27 用户定版：上传区那一排开关**整块删掉**（默认全开、只留填 API 的一栏）。
    # 原来的「点总闸带开六个分项」真机测试失去了对象（开关都不在了），已退役到
    # .trash_tmp/；换成守「确实删干净了、六个值仍一路通到后端」。
    ("开关已移除 · 默认全开",          ["tests/verify_switches_removed.py"], 120),
    # 用户报过的真 bug：底部三个直达入口跳出去回不来（侧栏一收起就没出口）。
    # 这条真机走一遍「出去 → 点返回 → 回到深色指挥台」，同时确认材料三块还能点。
    ("三个直达入口 能出能回",          ["probes/probe_nav_return.py"], 420),
    # 2026-10-28 本轮新增：坏文件与空材料两条兜底的真机验收。
    # 抓出过两个真问题：土坏 PDF 时顶栏把「没产出」标成「已完成」；
    # 错误提示里让人去开一个已经被删掉的「演示数据」开关。
    ("异常输入与空材料兜底",            ["probes/probe_bad_input.py"], 420),
    # 2026-10-29 本轮新增：坏输入的**语义边界**（离线，秒级）。
    # 与上一条的区别：上一条走浏览器看「界面有没有崩」，这条直接测底层三个函数的契约 ——
    #   文件在但打不开 → 降级；路径不存在 → 必须抛错；压根没给 → 按"没有"处理。
    # 为什么单独守：上一轮修伪 PDF 崩溃时，第一版把"路径不存在"也吞了，
    # 会静默出一份空报告（用户以为审计做完了）。这类回退必须能被一键抓住。
    ("坏输入的语义边界",                ["tests/verify_bad_inputs.py"], 180),
    # 2026-10-29 本轮新增：**「内部不一致」的判定尺度**（离线，秒级）。
    # 人工标注抽 7 条、4 条判错，错法一样：把「按任务分别给学习率」当成论文自相矛盾。
    # 新尺度 = 同页 + 同章节 + 其余数字全同 才算矛盾；其余只出「多处取值」提示。
    # 这条守两头：误报要收住，真矛盾也不许判丢（末尾两项）。
    ("内部不一致的判定尺度",            ["tests/verify_internal_scope.py"], 120),
    # 2026-10-30 本轮新增：**表格里的结果指标**（离线，秒级，PDF 现场生成）。
    # 26 份论文抽不到结果指标，根因不在正则：表格的文本层是**按列输出**的，
    # 「AP」和它对应的数字隔着几十行，按句子切永远凑不到一起。
    # 这条守两头：真表要能还原（含跨中线的宽表），正文碎片不许当成表。
    ("表格结果通道",                    ["tests/verify_table_rows.py"], 180),
    # ---- 阶段二 2D（2026-10-15）: ② 实验站（实验设计与真跑） ----
    # ② 站最危险的失败模式是**"看起来做了"** —— 面板上有卡片、卡片上有字,
    # 但那个字不是任何工具算出来的。所以断言的重点是: 每条结论都能追到一次
    # 真实的工具调用（输入 → 输出 → 耗时）, 且**预期与实际不同源**（自证验不出东西）。
    # 真实材料里两侧凑得齐公式的场景很少, 所以「一致 / 对不上」两条路径用合成用例覆盖。
    ("阶段二 实验设计与真跑",          ["tests/verify_experiment.py"], 300),
    # ---- 阶段二 2E（2026-10-15）: 图表多模态解析 ----
    # 本层最危险的失败模式是**"假装看过图"** —— 面板上有"图表解析"四个字,
    # 结论却只是把图注复述一遍, 甚至把"说明 / 表明"这类**解释**当成观察。
    # 所以断言的重点是: ① 解释性表述一律丢; ② 没有图上逐字 anchor 的一律不采信;
    # ③ 正文里的引用（"Figure 1 shows …"）不算图注。读图路径用离线 mock 覆盖。
    ("阶段二 图表多模态解析",          ["tests/verify_multimodal.py"], 300),
    ("阶段二 执行与判定解耦",          ["tests/verify_split.py"], 300),
    # ---- 阶段三 3A（2026-10-17）: 四件具名科学计算工具 ----
    # 3A 最危险的失败模式是**"看起来做了"**: 菜单上写着四件工具、面板上有调用记录,
    # 但那个数不是真算出来的 —— 或者更坏, 它是**编的**（凑一条假曲线再拟合出漂亮 R²）。
    # 所以断言盯三条红线: ① 真能跑且物理结论对（1e-8 在 fp16 必须落成 0, 那是规格值）;
    # ② 不造数（输入只能来自材料, 凑不齐就如实说缺什么）;
    # ③ ② 只算数不下判断（一律不给 compare —— 判定归 ③ 站, 2C 顶的结构不许破）。
    ("阶段三 具名科学计算工具",        ["tests/verify_science_tools.py"], 300),
    # ---- 多用户隔离（2026-10-15）: 会话 id 从进程级全局改成线程局部 ----
    # "没坏过但一定会坏"的那类缺陷: 公开链接是一个进程服务所有访问者,
    # 全局变量在单人单机永远看不出问题。断言盯着它, 不靠注释保证。
    ("多用户隔离 会话不串",            ["tests/verify_session_isolation.py"], 300),
    # ---- 2026-10-25: 逐环节功能体检（用户口径: 每个环节单独测, 再串起来看衔接） ----
    # 与上面各专项的分工: 那些按**能力**测（假设/实验/化学/闭环…），这一项按**工序**测 ——
    # 14 件工序逐件单独跑一遍（产物为空与跑崩分开记），再走一次真实图看四站传没传下去，
    # 最后验四条兜底（无 key→rule / 化学补算 / 无代码包 / 坏路径）。
    # 全程离线、不加载 .env，模型与视觉一次都不真调，因此结果与"今天有没有额度"无关。
    ("逐环节功能与衔接",               ["tests/verify_stages.py"], 300),
    # ---- 2026-10-27: 问答窗口 ----
    # 用户口径: 入口要醒目、点开能多轮追问、由 LLM 答疑。这里只测**口径**：
    # 事实只能来自 audit（带页码/行号/原值）、没 key 时只列证据不编、问答前后判定一个字节不变。
    # 像素与点击由 probes/shot_qa.py 真机点（不进全套：那一项会真调模型、要花钱且有额度依赖）。
    ("问答窗口 只依据审计事实",         ["tests/verify_qa.py"], 120),
    # ---- 2026-10-27 第二版: 指挥台七块区域的详情视图 ----
    # 用户口径: 每块点开都要有自己的数据（位置看流程、四站看边界、事件看时间、
    # 判定看分布），且**不许互相抄**。这一项盯的就是"悄悄退回同一张 findings 表"，
    # 顺带守三条：数只从 audit 来、空态不炸、没跑不许写成 0。
    ("指挥台详情 各区域各说各话",       ["tests/verify_deck_views.py"], 180),
    # LLM 三级优先级（2026-10-30 用户定）：页面输入 > 本机已部署 > 服务端配置 > 无可用
    # （无可用时自动降级跑全套链路，并说明调度决策等环节准确度会下降）。
    # **追加在末尾**是为了不动既有编号 —— `tools/run_scene_checks.py` 与
    # `tools/run_repr_checks.py` 都按编号分场景，中间插一项会让后面全部错位。
    # 本项纯离线（探测 + 源码守卫，约 2s），放 UI 检查之后不影响"先离线后界面"的意图。
    ("LLM 三级优先级与降级提示",    ["tests/verify_llm_priority.py"], 180),
    ("点击即终止上一次并清理",       ["tests/verify_run_cancel.py"], 120),
    # 云端 429 限流的处理口径（2026-10-30 晚，用户口径修正后重写）。
    # 用户原话：「我就是故意用 dsh 的密钥进行提问，不是本地」——
    # 所以**绝不许**把请求偷偷改到本机模型（那等于把他特意选的云端降级成 8B，
    # 还慢一百秒）。该做的是：429 退避重试两次（3s、6s），仍不行再用人话说清
    # 「云端在限流，过半分钟再问，钱和 key 都没问题」。
    # 守四件事：① 429 会重试、② 非限流错误不重试、③ 本机端点不重试（白等 100s）、
    #          ④ 重试等待期间被新点击作废要立刻中止。
    # 全程替身调用，不联网、不花钱，与"今天还有没有云端额度"无关。
    ("云端限流自动重试（不换本机）",  ["tests/verify_qa_fallback.py"], 120),
    # 弹窗名额（2026-10-31 崩溃促成）。Streamlit 一轮脚本只许开一个 st.dialog，
    # 开第二个会抛 StreamlitInvalidLayoutContextError，而且**整页往下全不渲染**、
    # 进程退出。崩的场景很日常：先在指挥台点过某一块（焦点留着），再点「问一问」。
    # 守两件事：① 名额先到先得、占名额写在 try 之前；② 详情让位时不清焦点，
    #           问答关掉后下一页自然摊开。
    ("一轮只开一个弹窗（名额让位）",  ["tests/verify_dialog_slot.py"], 60),
    # ask_json 的两条硬口径（2026-10-31 本机模型实测抓到）。
    # ① 函数收了 base_url/model 却不用 ⇒ 静默用错模型（影响 9 个调用点）；
    # ② 超时写死 30s ⇒ 本机 8B 的 JSON 计划必超时 ⇒ planner 永远回落 rule。
    # 全程替身，不联网。
    ("ask_json 转发端点与超时折算",   ["tests/verify_llm_askjson.py"], 60),
    # 四站主干底线（2026-10-31 用户定版）：模型排计划可以砍步骤，但 ①② 不许空。
    # 证据是实测：给 demo 排计划时模型只排 5 件，假设/实验两站恒空，四站变两站。
    # 这一项盯「底线生效 + 不越权 + 只补自己需要的上游」，全程注入提案，不联网。
    ("四站主干底线（①② 不许空）",     ["tests/verify_baseline_stations.py"], 90),
]

#: 本套检查会用到的全部端口。**每项开跑前**清理这些端口上的残留服务。
#:
#: 为什么必须按端口清理（2026-10-30 踩坑）:
#:   服务是用 `python.exe -m streamlit run app.py` 起的，**进程名是 python.exe**，
#:   所以 `taskkill /IM streamlit.exe` 一个也杀不到 —— 残留实例一直占着端口。
#:   后一项界面检查连上去，拿到的是上一轮留下的页面状态，等不到自己要的元素，
#:   白等到超时。实测代价: 全套里 4 项界面失败（#28 白等 404s、#39 136s、
#:   #44 64s、#45 150s），而同一份代码**单独跑全部通过**。
#:   症状很好认: 「全套失败、单独跑全过」—— 先怀疑端口残留，别怀疑产品。
_TEST_PORTS = (8501, 8502, 8511, 8513, 8514, 8517, 8518, 8519,
               8521, 8527, 8531, 8599, 8791)


def _free_test_ports() -> int:
    """杀掉占用测试端口的 python 进程，返回清掉几个。只认 LISTENING、只杀 python。"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:                      # noqa: BLE001
        return 0
    pids: set[int] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3].upper() != "LISTENING":
            continue
        try:
            port = int(parts[1].rsplit(":", 1)[-1])
        except ValueError:
            continue
        if port in _TEST_PORTS and parts[4].isdigit() and int(parts[4]) > 0:
            pids.add(int(parts[4]))
    killed = 0
    for pid in pids:
        try:
            name = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=15).stdout.lower()
            if "python" not in name:
                continue
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=15)
            killed += 1
        except Exception:                  # noqa: BLE001
            continue
    return killed


ERR_PAT = ("Traceback", "❌", "异常", "未通过", "Error", "error", "ERR:",
           "FAIL", "超时", "TIMEOUT")

# 正常输出里会"看起来像异常"的行, 过滤掉以免误报(如"未通过 0"、空列表、故意造的诱饵文件)
_BENIGN = ("fitz", "pymupdf", "未通过 0", "监控方式", "监控脚本",
           "broken.py",   # 边缘集里故意放的语法错误文件, 被拒收才是正确行为
           "强制 Mock", "对照:", "模板样例",
           # 否定式表述("无异常/无错误")是正常结论, 不是异常
           "无页面异常", "无控制台错误", "无浏览器控制台错误", "无异常", "无错误",
           # 2026-10-18: 这些是**断言标题自身**在夸"不会抛异常" —— 含"异常"二字
           # 却被当成异常命中, 全套报告会平白多出一串⚠️(实测 6 项)。判据是句意, 不是字面。
           "不抛异常", "不是异常", "不能抛异常", "不把异常甩给上层",
           # 2026-10-19: 同类误报三例 —— 断言标题在夸"失败也不算崩"、
           # "卡住的源被标成超时"、"isError=False"，字面命中关键词但句意正确。
           "不是抛异常", "被标为超时", "isError=False",
           "errors 为空",
           # 2026-10-26: 「无任何…抛异常」这种带"任何"的否定式，前面的
           # "无异常"/"不抛异常"都覆盖不到（中间隔着字），全套会平白多一条 ⚠️。
           "无任何工序抛异常", "无任何异常", "没有任何异常")


def _benign(line: str) -> bool:
    if any(b in line for b in _BENIGN):
        return True
    s = line.rstrip()
    return s.endswith("[]") or s.endswith(": []")     # 空列表/空结果


def run_one(name: str, args: list[str], timeout: int):
    # 每项开跑前清掉端口上的残留服务 —— 不清的话后一项会连到前一项留下的旧实例，
    # 拿到上一轮的页面状态、等不到自己的元素，最后报"页面没渲染"（实际是连错了）。
    _freed = _free_test_ports()
    if _freed:
        print(f"    · 清理端口残留 {_freed} 个")
    # 用单调时钟计时：`time.time()` 会被系统时钟调整/休眠唤醒推着走，
    # 实测过一次机器休眠后单条记录虚报成 29767s（超时上限才 1800s）。
    t0 = time.monotonic()
    try:
        p = subprocess.run([PY, "-X", "utf8", *args], cwd=str(ROOT), env=os.environ.copy(),
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        code, timed_out = p.returncode, False
    except subprocess.TimeoutExpired as e:
        # text=True 时 stdout/stderr 已是 str; 未设置时为 bytes —— 两种都要能处理
        def _s(v):
            if v is None:
                return ""
            return v if isinstance(v, str) else v.decode("utf-8", "replace")
        out = _s(e.stdout) + _s(e.stderr)
        code, timed_out = -1, True
    el = time.monotonic() - t0
    hits = sorted({ln.strip() for ln in out.splitlines()
                   if any(k in ln for k in ERR_PAT)})
    hits = [h for h in hits if not _benign(h)]
    return {"name": name, "cmd": " ".join(args), "elapsed": el,
            "code": code, "timeout": timed_out, "hits": hits[:12],
            "tail": "\n".join(out.splitlines()[-12:])}


def main():
    results = []
    for i, (name, args, to) in enumerate(CHECKS, 1):
        print(f"[{i}/{len(CHECKS)}] {name} …", flush=True)
        r = run_one(name, args, to)
        results.append(r)
        status = "TIMEOUT" if r["timeout"] else ("OK" if r["code"] == 0 else f"exit={r['code']}")
        print(f"    -> {status}  {r['elapsed']:.1f}s", flush=True)

    bad = [r for r in results if r["timeout"] or r["code"] != 0]
    warn = [r for r in results if r not in bad and r["hits"]]

    md = ["# Paper2Lab 全套监控报告", "",
          f"- 运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
          f"- 检查项: {len(results)}　异常: {len(bad)}　含警示关键词: {len(warn)}", "",
          "| # | 检查项 | 命令 | 状态 | 耗时 |", "| --- | --- | --- | --- | --- |"]
    for i, r in enumerate(results, 1):
        st = "⏱️ 超时" if r["timeout"] else ("✅ 通过" if r["code"] == 0 else f"❌ exit={r['code']}")
        md.append(f"| {i} | {r['name']} | `{r['cmd']}` | {st} | {r['elapsed']:.1f}s |")

    md += ["", "## 异常与警示明细", ""]
    if not bad and not warn:
        md.append("- 无：全部检查通过，未发现报错、异常或超时。")
    for r in bad:
        st = "超时" if r["timeout"] else ("退出码 " + str(r["code"]))
        md += [f"### ❌ {r['name']}", "",
               f"- 状态: {st}",
               f"- 耗时: {r['elapsed']:.1f}s", "- 输出尾部:", "```text", r["tail"], "```", ""]
    for r in warn:
        md += [f"### ⚠️ {r['name']}（通过但有关键词）", ""]
        for h in r["hits"]:
            md.append(f"- `{h[:160]}`")
        md.append("")

    (OUT_DIR / "monitor_suite_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=================== 汇总 ===================")
    for r in results:
        st = "TIMEOUT" if r["timeout"] else ("OK" if r["code"] == 0 else f"exit={r['code']}")
        print(f"{r['name']:28} {st:10} {r['elapsed']:7.1f}s")
    print(f"\n异常 {len(bad)} 项 / 含警示 {len(warn)} 项")
    print(f"报告: {OUT_DIR / 'monitor_suite_report.md'}")

    # ---- 跑完自觉清理（2026-09-28）: 关掉残留进程 + 清运行缓存与临时草稿。
    # 必须放在**最后**: 界面类检查自己会抢占 8501 端口（跑完留着常驻服务白占内存）。
    # 缓存只是提速用的, 删了不影响任何结论; 想跑得快就用 --keep-cache 单独跑那个脚本。
    print("\n--- 跑完清理 ---", flush=True)
    sys.stdout.flush()          # 子进程直接写同一个 fd, 不 flush 会让日志顺序看起来错乱
    subprocess.run([PY, "-X", "utf8", "tools/cleanup_after_run.py"], cwd=str(ROOT))


if __name__ == "__main__":
    sys.exit(main())
