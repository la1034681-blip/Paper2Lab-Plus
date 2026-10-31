"""工具箱（第二期第一批）——把审计流水线的每一站登记成可单独调用的工具。

三条设计原则:
1. **工具只描述能力, 不含调度**: "怎么做"写在工具里, "做不做、按什么顺序做"
   交给调度员(planner)。工具函数本身不判断该不该跑。
2. **进度节点向后兼容**: 每个工具声明它覆盖的进度节点(checkpoints),
   `STEPS` 由工具顺序自动推导, 界面既有的十步进度条与 `pipeline.STEPS` 零改动。
3. **只有一份实现**: 单路径与多智能体对撞调用的是同一组工具函数,
   避免代码漂移（否则"等价重构"就无从谈起）。2026-10-25 已删除"经典模式"。

关于"十五个进度节点、十三个工具":
"参数归一化 / 交叉匹配 / 实验结果核验"在实现上是 `aligner.align` 的三段职责,
一次遍历同时产出; 硬拆成三个工具只会重复计算, 因此登记为一个工具、覆盖三个节点。
这一点在界面的「执行计划」里如实标注, 不做粉饰。
「假设生成」（阶段二 2A）、「实验设计与真跑」（2D）、「图表解析」（2E）同理:
各是一个工具、一个进度格。其中「假设生成」排在 align **之前**（提假设不需要先有判定,
而且这样「① 假设」在四站进度条上是连续的一段）。

**第三期两站为什么也登记进来**（2026-09-28 补）:
「根因追查 → 补丁草案」是一条**动作链的前后两半**, 少一半就不成立。早先它俩只在
报告之外的独立脚本里跑, 后果是三重的: ① 每个调用方自己拼一遍, 实测已漂移成三份
(`run_new_set` / `demo_full_flow` / `verify_patch`); ② 调度员管不到 —— 没有代码包时
本该跳过, 却没人知道该跳过; ③ 报告与界面完全看不到, 用户问"补丁为什么被拒"时
产品答不上来。现在它们与其它站同权: 结论落在 `AuditResult.rootcause / .patches`,
报告有专节、界面有面板、执行记录里看得见去向了。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .agent import fill_missing
from .aligner import align
from .code_analyzer import analyze_code
# 阶段二 2B: 提完假设先让「计算员」算一遍（真分歧 / 伪分歧 / 一侧写错）
from .derivation import derive, summarize
# 阶段二 2D: ② 实验站 —— 对推不掉的假设**真的动手算一遍**, 留下可核验的实验记录
from .experiment import EXPERIMENT_MAX, design_experiments, summarize as summarize_experiments
from .hypothesis import build_hypotheses
# 阶段二 2E-化学: 化学身份的可复算核验（分子式 / CAS / SMILES / 分子量对账）
from . import chem
from .llm import LLMAnalyzer
from .models import AuditResult, CodeInfo, PaperInfo
# 阶段二 2E: ① 假设站「读」这一步的多模态视野 —— 从图表里读结构化线索（图注 ＋ 读图）
from .multimodal import FIGURE_MAX, read_figures, summarize as summarize_figures
from .paper_parser import parse_paper
from .readiness import compute_readiness
from .report import generate_report
from .selfcheck import (build_manual_check, build_param_matrix, compute_coverage,
                        run_selfcheck)
from .vision import analyze_images


@dataclass
class Job:
    """一次审计任务的输入与中间产物——工具箱与执行器共用的载体。

    执行器只负责"调用工具、传递 Job、记录状态", 所有中间产物都挂在 Job 上,
    因此工具之间不需要互相知道对方的存在。
    """
    # ---- 输入
    pdf_path: str = ""
    code_path: str = ""
    user_results: Optional[dict] = None
    api_key: str = ""
    # 模型 API 的端点与模型名（2026-09-30）：页面可填，让用户换任何 OpenAI 兼容模型。
    # 留空则回落 .env / 环境变量 / 官方默认，**老调用点行为不变**。
    llm_base_url: str = ""
    llm_model: str = ""
    force_mock: bool = False
    vision_kwargs: dict = field(default_factory=dict)
    vision_max_regions: Optional[int] = None
    enable_agent: bool = True
    agent_budget: dict = field(default_factory=dict)
    # 让 AI 排计划: 规则版仍是兜底，见 planner._llm_refine。
    # 字段默认 False（CLI 与验证脚本需要可预测的确定性基线）；
    # **网页端默认 True**（agent 模式，见 app.py 的「AI 参与决策」开关）。
    llm_planner: bool = False
    # 让 AI 给补丁"提名目标": 只在"值可信、缺的只是选哪个"时启用，
    # 提名必须落在已抽到的候选里，再由程序盖章，产出标 proposed-llm（需人工确认）。
    # 同上：字段默认 False，网页端默认 True。
    llm_patch: bool = False
    # 阶段二 2D：② 实验站是否让模型挑计算工具（**默认关**，与 role_llm 同款口径）。
    # 关着时走纯规则挑路径; 开着时模型**只能挑工具键**, 输入仍由规则从材料里取（不造数）。
    # 不可用 / 输出不合法 → 回落规则版, 并在实验卡的 plan_reason 里如实写明。
    experiment_llm: bool = False
    # 2026-10-25：**模型先判、程序后验**（试点 ① 假设真假 / ③ 参数配对）。
    # 口径：模型先出判断 → 再跑规则（规则那步**独立跑完整判定**，不是只复核模型说的话）
    #      → 比对并列。规则结论 `verdict`/`status` 永远由规则写；模型的话只进 `llm_*`，
    #      不一致时**两边都进报告**，不静默取舍。
    # **默认关**（与 experiment_llm / role_llm 同款）：线上零成本、结论可复现是硬要求。
    # 关着时 `judge_llm` 一次都不调用，报告与界面不出现这一节。
    judge_llm: bool = False
    # 阶段三 3C：**消融档位**要「真跳过」的工具键（默认空 = 一个都不跳，行为完全不变）。
    # 与 planner 的 skip 不同: 那是调度员判断"这次不值得跑"，这是评测时**人为少开一项能力**，
    # 用来量"每加一档能力到底带来多少提升"。跳过会如实记进执行记录（status=skipped ＋ 理由），
    # 不是只把输出藏起来 —— 藏输出等于没做消融，查不得。
    # 受保护的三件**不许跳**（跳了就没有可评的答案），见 ABLATION_LOCKED。
    skip_tools: tuple = ()
    # ---- 产物
    paper: Optional[PaperInfo] = None
    code: Optional[CodeInfo] = None
    audit: Optional[AuditResult] = None
    report: str = ""
    agent_report: object = None          # AgentReport（第一期）
    rootcause: list = field(default_factory=list)   # [TraceResult]（第三期 3.2）
    patches: list = field(default_factory=list)     # [PatchProposal]（第三期 3.3）
    llm_mode: str = ""
    llm_real_calls: int = 0
    llm_cache_hits: int = 0
    llm_error: str = ""
    llm_shard: dict = field(default_factory=dict)      # 归因的分片并发记录
    llm_model_used: str = ""          # 归因实际用的模型名（界面/报告如实显示）
    execution_log: dict = field(default_factory=dict)
    fingerprint: str = ""                                # 论文指纹(跨任务记忆索引)
    memory_lines: list = field(default_factory=list)      # 记忆命中的线索(只提示)
    # 上下文装配器(阶段三第2步): 会话记忆按"论文侧 vs 代码侧"装配成对照屏
    context_screen: Optional[object] = None              # assembler.ContextScreen
    # 会话级复用(阶段三第1步): 同会话内同一输入第二次跑, 跳过解析
    needs_index: Optional[bool] = None                   # None=未知/未查; True=需解析; False=可复用
    snapshot: Optional[tuple] = None                     # (paper, code) 复用产物
    snapshot_reused: bool = False                       # 本次是否走了快路径(供界面展示)
    # ---- 阶段二 2A: 假设卡（① 假设站的产物, 四站叙事里"提"的落点）
    # 只挂新字段, 不动任何既有字段语义; execution_log 的结构也不碰（L8 铁律）。
    hypotheses: list = field(default_factory=list)      # [hypothesis.Hypothesis] 已排序
    hypothesis_total: int = 0                           # 共提出多少条（界面如实说"共 N 条, 列出 M 条"）
    # 阶段二 2B：推演统计。**按全部卡统计**, 不只是展示的那几条 ——
    # 否则会出现"界面说共 5 条、只统计了 3 条"的对不上。
    derivation_note: dict = field(default_factory=dict)
    # ① 站**全部**卡片（未被 HYPOTHESIS_MAX 截断）。2D 的实验站要对着全部卡设计实验,
    # 否则"实验覆盖了多少条假设"这个数会跟着展示上限走 —— 展示 8 条就成了"只有 8 条"。
    all_hypotheses: list = field(default_factory=list)
    # ---- 阶段二 2D: 实验卡（② 实验站的产物）
    # 与假设卡同款: `experiments` 只放展示用的前 N 条, `experiment_total` 如实报总数。
    experiments: list = field(default_factory=list)     # [experiment.ExperimentCard] 已排序
    experiment_total: int = 0
    experiment_note: dict = field(default_factory=dict)
    # ---- 阶段二 2E: 图表线索（① 假设站「读」这一步的多模态产物）
    # 与上面同款: `figure_clues` 只放展示用的前 N 条, `figure_total` 如实报总数。
    figure_clues: list = field(default_factory=list)    # [multimodal.FigureClue] 已排序
    figure_total: int = 0
    figure_note: dict = field(default_factory=dict)
    # 化学身份记录（阶段二 2E-化学）—— 与图表线索同一步产出, 但**分开挂**:
    # 一个是"图上看得见什么", 一个是"化学身份算得对不对", 混在一起两边都说不清。
    chem_items: list = field(default_factory=list)      # [chem.ChemItem] 已判定、已排序
    chem_total: int = 0
    chem_note: dict = field(default_factory=dict)
    chem_fallback: bool = False   # 结果是「模型跳过后程序兜底补算」的（报告/界面如实标注）

    # ---- 2026-10-25: 模型先判、程序后验的留痕（关着时全空, 界面与报告不出现这一节）
    # note 是"这次模型到底有没有参与、判了几条"的人话说明; stat 是逐项比对结果。
    judge_hyp_note: str = ""
    judge_find_note: str = ""
    judge_hyp_stat: dict = field(default_factory=dict)
    judge_find_stat: dict = field(default_factory=dict)
    chem_ran: bool = False        # 该步**真跑过**（即便是 0 条）—— 兜底据此判断要不要补


# ---------------------------------------------------------------- 工具执行体
# 以下函数由 pipeline.run_audit 的既有步骤**逐行搬运**而来, 行为不变。

def _run_parse_paper(job: Job, tick: Callable[[str], None]) -> None:
    if job.snapshot_reused:
        # 快路径: 复用上一轮的解析产物, 不重复读 PDF(仍打勾, 进度条如实)
        tick("论文解析")
        return
    job.paper = parse_paper(job.pdf_path)
    tick("论文解析")


def _run_analyze_images(job: Job, tick: Callable[[str], None]) -> None:
    job.paper.image_params, job.paper.image_meta = analyze_images(
        job.pdf_path, max_regions=job.vision_max_regions, **job.vision_kwargs)
    tick("图像解析")


def _run_read_figures(job: Job, tick: Callable[[str], None]) -> None:
    """图表多模态解析（阶段二 2E · ① 假设站「读」这一步）。

    与「图像解析」的分工: 那一步读的是**实验配置值**（把论文写的超参补回来）,
    这一步读的是**图本身**（图注原文 ＋ 图上看得见的坐标轴/趋势/峰位）——
    它是给假设站用的线索, 不是往参数表里塞数。
    两条通道: 图注（纯文本, 零成本, 总能跑）＋ 读图（可选, 需视觉 key 或离线 mock）。
    **不跑训练、不执行用户代码**; 读不出就说读不出（边界见 multimodal.figures 模块头）。
    """
    clues, stats = read_figures(job.paper, pdf_path=job.pdf_path,
                                vision_kwargs=dict(job.vision_kwargs or {}))
    job.figure_note = stats
    job.figure_total = len(clues)
    job.figure_clues = clues[:FIGURE_MAX]       # 总数如实报, 界面只展示前 N 条
    tick("图表解析")


def _run_chem_check(job: Job, tick: Callable[[str], None]) -> None:
    """化学身份核验（阶段二 2E-化学 · ① 假设站「读」这一步的**独立视野**）。

    2026-10-25 从 `read_figures` 里拆出来 —— 两件事本来就不相干（一个读图、一个算式子）。
    寄生在同一步的后果是：模型排计划跳过「图表解析」时，化学核验**跟着一起消失**
    （同一份 PDF：命令行核出 10 条、界面核验 0 条）。

    拆开后它在菜单上是一条**独立条目**，模型可以"不读图但核化学"——
    读图要花钱（调视觉模型），这一步不花钱（纯本地正则＋算术，毫秒级），两个决策彻底解耦。

    **只读正文, 不从像素猜结构**（见 chem 模块头）：看图认结构是幻觉重灾区,
    而"分子式对不上分子量"是确定性错误, 值得单列。
    """
    items, stats = chem.measure(job.paper)
    job.chem_items = chem.judge(items)[:chem.CHEM_MAX]
    job.chem_total = len(items)
    job.chem_note = stats
    job.chem_ran = True          # 真跑过了（0 条也算跑过）→ 兜底不必再算一遍
    job.chem_fallback = False    # 模型自己跑出来的，不是补算的
    # 这一步不占进度格（checkpoints=()）→ tick 的名字不在 STEPS 里, 回调会静默忽略。
    # 仍然传名字: 万一以后给它补了格子, 这里不用再改。
    tick("化学身份核验")


def _run_fill_missing(job: Job, tick: Callable[[str], None]) -> None:
    """抽取 Agent（第一期成果）——原样保留, 预算与闸门逻辑不变。"""
    _agent = fill_missing(
        job.paper, job.pdf_path,
        llm_api_key=job.api_key or "",
        llm_model=job.llm_model or "",
        llm_base_url=job.llm_base_url or "",
        vision_kwargs=dict(job.vision_kwargs),
        enabled=job.enable_agent, **dict(job.agent_budget or {}))
    for k, ev in _agent.filled.items():
        job.paper.params_all.setdefault(k, []).append(ev)
        job.paper.params.setdefault(k, ev)
    job.paper.agent_trace = _agent.to_dict()
    job.paper.agent_meta = _agent.to_dict()
    job.agent_report = _agent
    tick("Agent 补漏抽取")


def _run_analyze_code(job: Job, tick: Callable[[str], None]) -> None:
    if job.snapshot_reused:
        # 快路径: 代码扫描产物同上轮, 不重扫 AST
        tick("代码扫描")
        return
    job.code = analyze_code(job.code_path)
    tick("代码扫描")


# 假设卡展示上限（与 rootcause / patch 的 max_items 同口径, 不另定一个数）
HYPOTHESIS_MAX = 8


def attach_hypotheses(audit, cards, total: int = 0, stats=None) -> None:
    """把 ① 假设站的产物挂到审计结果上 —— **唯一挂载点**。

    为什么要抽成一个函数, 而不是在三处各写一行: 报告是在 `_run_report` 里生成的,
    而假设卡原本只挂在 `job` 上, 到收尾才搬到 `audit` —— 于是报告比挂载**更早**跑,
    生成出来的报告里「① 假设站: 0 条」, 界面同一份审计却列着 5 条。
    这是最不该出现的那种分裂: 两个出口对同一件事说了不同的话。
    现在三个调用点（报告生成前 / runner 收尾 / 图版 finalize）都走这里, 不会再漂移。

    总数取 `max(total, len(cards))`: 绝不出现「共 0 条」下面又列着卡片的荒唐排版。
    """
    if audit is None:
        return
    cards = list(cards or [])
    audit.hypotheses = cards
    audit.hypothesis_total = max(int(total or 0), len(cards))
    # 阶段二 2B：推演统计（**对全部卡**统计, 与"展示了几条"无关）。
    # 没传就留空 dict —— 界面据此判"这次到底推演没推演", 而不是把 0 条当成
    # "推演了但全是待验"。两者对用户是两个意思, 不能混。
    audit.hypothesis_stats = dict(stats or {})


def attach_experiments(audit, cards, total: int = 0, stats=None) -> None:
    """把 ② 实验站的产物挂到审计结果上 —— **唯一挂载点**（与 `attach_hypotheses` 同款）。

    为什么也要抽出来: 报告里有一节要写实验站, 而报告是在 `_run_report` 里生成的。
    若卡到收尾才挂, 报告就会写「本次没有做实验」, 而界面同一份审计列着好几张 ——
    这正是 `attach_hypotheses` 当初踩过的那一跤（报告写 0 条、界面列 5 条）。
    所以三个调用点（报告生成前 / runner 收尾 / 图版 finalize）都走这里。
    """
    if audit is None:
        return
    cards = list(cards or [])
    audit.experiments = cards
    audit.experiment_total = max(int(total or 0), len(cards))
    audit.experiment_stats = dict(stats or {})


def attach_figures(audit, clues, total: int = 0, stats=None) -> None:
    """把 ① 假设站「读」这一步的图表线索挂到审计结果上 —— **唯一挂载点**。

    与 `attach_hypotheses` / `attach_experiments` 同一条理由: 报告在 `_run_report`
    里生成, 若线索到收尾才挂, 报告会写「没有读到图表线索」而界面列着好几条 ——
    这是同一跤摔第三次, 所以从一开始就抽成一个函数, 三个调用点都走这里。
    """
    if audit is None:
        return
    clues = list(clues or [])
    audit.figure_clues = clues
    audit.figure_total = max(int(total or 0), len(clues))
    audit.figure_stats = dict(stats or {})


def attach_chemistry(audit, items, total: int = 0, stats=None,
                     fallback: bool = False) -> None:
    """把化学身份记录挂到审计结果上 —— **唯一挂载点**（与 `attach_figures` 同款理由）。

    报告在 `_run_report` 里生成: 若记录到收尾才挂, 报告会写"没有抽到化学身份"
    而界面列着好几条 —— 这跤摔过三次, 所以从一开始就只留一个入口。

    `fallback=True` 表示这批结果是**模型跳过该步、程序兜底补算**的:
    报告与界面会写明这件事, 免得"跳过"被误读成"论文里本来就没有化学内容"。
    """
    if audit is None:
        return
    items = list(items or [])
    audit.chem_items = items
    audit.chem_total = max(int(total or 0), len(items))
    audit.chem_stats = dict(stats or {})
    audit.chem_fallback = bool(fallback and items)


def ensure_chemistry(job) -> bool:
    """化学核验的**最后一道兜底**（纯规则、零成本、不调模型）。返回 True = 这次补算了。

    2026-10-25 之后：化学核验已是**独立工具** `chem_check`（不再寄生在图表解析里）。
    这里的口径随之收窄成一句话 —— **允许模型跳过，但不许结果静默消失**：

      · 模型主动跑了          → 什么都不做，直接复用；
      · 模型跳了 / 前置缺失 / 执行出错 → 只要材料在手，就在生成报告前补算一遍。

    为什么免费的东西还要兜底：它不花钱、不调模型、毫秒级，跳过它捞不到任何好处，
    却会让界面出现"核验 0 条"而命令行核出 10 条的分裂 —— 审计平台最不该有这个。
    补算的结果会标在 `job.chem_fallback` 上，由报告与界面如实写明
    「该步被跳过，程序已按『免费且必做』口径补算」，**跳过这件事本身照样留痕**。
    """
    if getattr(job, "chem_items", None) or getattr(job, "chem_total", 0):
        return False
    # 跑过但结果是 0 条（论文里没有化学内容）→ 也直接返回, 别白算第二遍
    if getattr(job, "chem_ran", False):
        return False
    paper = getattr(job, "paper", None)
    if paper is None:
        return False
    from . import chem

    raw, stats = chem.measure(paper)
    job.chem_items = chem.judge(raw)[:chem.CHEM_MAX]
    job.chem_total = len(raw)
    job.chem_note = stats
    # 如实标注：这批结果是**跳过后补算**的，不是模型主动跑出来的。
    # 没有化学标记的论文补算出 0 条 → 不算「补算」, 别在界面上无中生有。
    job.chem_fallback = bool(raw)
    # 补算也是「真跑过」（2026-10-25 逐环节体检发现）：
    # 不置位的话，非化学论文（补算出 0 条 → chem_items 空）会掉出上面第 335 行的
    # 短路条件，于是三个调用点（runner 收尾 / 报告前 / graph 收尾）**各重算一遍**。
    # chem_ran 问的是「这步跑没跑」，与「谁触发的」无关 —— 补算跑的也是同一个 _run_chem_check 的算法。
    job.chem_ran = True
    return True


def _run_hypothesize(job: Job, tick: Callable[[str], None]) -> None:
    """假设生成（阶段二 2A）—— 把"两边对不上"翻译成**可被推翻的陈述**。

    **排在 align 之前**：提假设不需要先有判定，而且这样「① 假设」在四站进度条上
    是连续的一段（读材料 → 提假设），不会出现同一站被劈成头尾两块。
    谁对谁错的判定留给 ③（参数归一化 / 交叉匹配那三步）。

    纯规则、零成本、不调模型、不碰文件；与判定链路共用同一套值比较口径
    （见 hypothesis 模块头的「口径纪律」），所以不会和 ③ 的结论互相打架。
    """
    cards = build_hypotheses(job.paper, job.code)
    # 2026-10-25：**模型先判**（刻意排在 derive 之前 —— 用户口径「模型先判、程序后判」）。
    # 拿不到就空着, 由 apply_hypotheses 如实记「模型没判」, 不回退成「假装判过」。
    llm_judged: dict = {}
    if job.judge_llm and cards:
        from .judge_llm import judge_hypotheses
        llm_judged, job.judge_hyp_note = judge_hypotheses(
            cards, api_key=job.api_key, force_mock=job.force_mock,
            base_url=getattr(job, "llm_base_url", "") or "",
            model=getattr(job, "llm_model", "") or "")
    # 阶段二 2B：提完先让「计算员」算一遍 —— 判出 真分歧 / 伪分歧 / 一侧写错。
    # 为什么就在这一步、不另开节点：「筛选归 ① 站」是既定口径（见四站分界），
    # ② 实验站只该接筛完的活；而且不新增节点就不动 TOOLS / STEPS / 进度条契约。
    # ⚠ 这一步**独立跑完整判定**, 不是只复核模型说的话（judge_llm 模块铁律一）。
    cards = derive(cards, job.paper, job.code)
    # **程序后验**：规则结论已经在手里, 这里只比对并列, 绝不覆盖 verdict。
    if job.judge_llm:
        from .judge_llm import apply_hypotheses, summary as _judge_summary
        job.judge_hyp_stat = apply_hypotheses(cards, llm_judged)
        job.judge_hyp_note += "；" + _judge_summary(job.judge_hyp_stat)
    job.derivation_note = summarize(cards)
    job.hypothesis_total = len(cards)
    job.all_hypotheses = cards                   # ② 站要对着**全部**卡设计实验
    job.hypotheses = cards[:HYPOTHESIS_MAX]      # 总数如实报, 界面只展示前 N 条
    tick("假设生成")


def _run_design_experiment(job: Job, tick: Callable[[str], None]) -> None:
    """实验设计与真跑（阶段二 2D · ② 实验站）—— 把「怀疑」变成「怎么验」, 并真的算一遍。

    与 ① 站的分工: ① 说"我们怀疑什么"（并把推得掉的先推掉）; ② 说"那怎么验",
    而且**真的动手**（调具名计算工具、留下输入 / 输出 / 耗时的执行记录）。
    判定（支持 / 推翻 / 待验）不在这里 —— 那是 ③ 站的事（② 站只负责测, 不造数也不判）。

    **不碰文件、不跑训练、不执行用户代码**; 小规模复算走 `compute` 的 ast 白名单。
    模型那一路（`job.experiment_llm`）默认关: 关着时是纯规则挑工具路径, 如实标 `plan_reason`。
    """
    cards = list(job.all_hypotheses or job.hypotheses or [])
    llm = _experiment_llm(job)
    exps = design_experiments(cards, job.paper, job.code, llm=llm)
    job.experiment_note = summarize_experiments(exps)
    job.experiment_total = len(exps)
    job.experiments = exps[:EXPERIMENT_MAX]      # 总数如实报, 界面只展示前 N 条
    tick("实验设计")


def _experiment_llm(job: Job):
    """② 站的"模型挑工具"通道 —— **默认关**，关着时返回 None（走纯规则）。

    为什么默认关: 与 `role_llm` 同款口径 —— 线上零成本、结论可复现是硬要求；
    而且模型这一路**只允许挑工具键**, 输入一律由规则从材料里取（不造数）。
    不可用 / 输出不合法 → 回落规则版, 并在实验卡的 `plan_reason` 里如实写明。
    """
    if not job.experiment_llm:
        return None
    key = str(job.api_key or "")
    if not key:
        return None

    def _ask(prompt: str):
        from .llm import ask_json
        # ask_json 返回 (dict | None, 说明) —— 这里只取对象; 拿不到就回落规则版。
        obj, _note = ask_json(prompt, key, mode="experiment",
                              force_mock=bool(getattr(job, "force_mock", False)),
                              base_url=getattr(job, "llm_base_url", "") or "",
                              model=getattr(job, "llm_model", "") or "")
        return obj

    return _ask


def _run_align(job: Job, tick: Callable[[str], None]) -> None:
    """归一化 + 交叉匹配 + 结果核验: 一次遍历同时产出, 覆盖三个进度节点。"""
    # 2026-10-25：**模型先判**（在 align 之前）—— 配对从两侧解析结果直接组。
    # 刻意不拿 Finding 去问: 它自带 status, 带着答案问等于让模型抄（见 judge_llm.JudgePair）。
    llm_judged: dict = {}
    if job.judge_llm:
        from .judge_llm import judge_findings, pairs_from_materials
        llm_judged, job.judge_find_note = judge_findings(
            pairs_from_materials(job.paper, job.code),
            api_key=job.api_key, force_mock=job.force_mock,
            base_url=getattr(job, "llm_base_url", "") or "",
            model=getattr(job, "llm_model", "") or "")
    job.audit = align(job.paper, job.code, job.user_results)
    # **程序后验**：align 已独立跑完整判定, 这里只比对并列, 绝不覆盖 status。
    if job.judge_llm:
        from .judge_llm import apply_findings, summary as _judge_summary
        job.judge_find_stat = apply_findings(job.audit.findings, llm_judged)
        job.judge_find_note += "；" + _judge_summary(job.judge_find_stat)
    tick("参数归一化")
    tick("交叉匹配")
    tick("实验结果核验")


def _run_llm_analyze(job: Job, tick: Callable[[str], None]) -> None:
    analyzer = LLMAnalyzer(api_key=job.api_key, force_mock=job.force_mock)
    analyzer.analyze_all(job.audit.findings)
    job.llm_mode = analyzer.mode
    job.llm_real_calls = analyzer.real_calls
    job.llm_cache_hits = analyzer.cache_hits
    job.llm_error = analyzer.last_error
    job.llm_shard = dict(getattr(analyzer, "shard_report", {}) or {})
    # 2026-10-26 修：这一行原来漏了 —— `llm_model_used` 从定义起就没人赋值，
    # 于是 readiness["llm_model"] 恒为空，界面只能兜底显示「模型 API」，
    # 报告里也写不出"这次用的是哪个模型"。取归因器实际用的模型名。
    job.llm_model_used = (getattr(analyzer, "model", "") or job.llm_model or "").strip()
    tick("AI 风险归因")


def _run_selfcheck(job: Job, tick: Callable[[str], None]) -> None:
    run_selfcheck(job.audit)        # 证据自证 + 结论回放
    compute_coverage(job.audit)     # 解析覆盖率与审计可信度
    job.audit.coverage["manual_check"] = build_manual_check(job.audit)
    job.audit.coverage["param_matrix"] = build_param_matrix(job.audit)

    compute_readiness(job.audit)
    # 2026-10-26：区分「配了 key 但这一步没跑」与「压根没配 key」。
    # 多智能体档的对撞区会替代「AI 风险归因」这一步（既定口径），此时 llm_mode 是空的；
    # 原来界面一律显示「内置模板（未配置 API key）」—— 明明配了 key，标注却像没配，
    # 答辩时会被当场问住。这里如实记 skipped，由界面翻成人话。
    job.audit.readiness["llm_mode"] = job.llm_mode or ("skipped" if job.api_key else "")
    job.audit.readiness["llm_model"] = job.llm_model_used
    job.audit.readiness["llm_real_calls"] = job.llm_real_calls
    job.audit.readiness["llm_cache_hits"] = job.llm_cache_hits
    job.audit.readiness["llm_error"] = job.llm_error
    job.audit.readiness["llm_shards"] = dict(job.llm_shard or {})
    tick("审计自检")


def _run_rootcause(job: Job, tick: Callable[[str], None]) -> None:
    """根因追查（第三期 3.2）：从多个赋值点里找出**最终生效**的那一个（只读代码）。"""
    from .rootcause import trace_findings
    traces = trace_findings(job.audit, max_items=8)
    job.rootcause = traces
    job.audit.rootcause = traces
    tick("根因追查")


def _run_patch(job: Job, tick: Callable[[str], None]) -> None:
    """补丁草案（第三期 3.3）：基于根因给出 diff 草案 —— **只展示，绝不落盘**。

    `job.llm_patch=True` 时额外开一条「AI 提名目标 + 程序盖章」通道（见 patch.propose_via_llm）。
    """
    from .patch import propose_patches
    patches = propose_patches(job.audit, job.rootcause or [], job.code_path,
                              max_items=8, llm=job.llm_patch,
                              api_key=job.api_key, force_mock=job.force_mock,
                              base_url=job.llm_base_url or "",
                              model=job.llm_model or "")
    job.patches = patches
    job.audit.patches = patches
    tick("补丁草案")


def _run_report(job: Job, tick: Callable[[str], None]) -> None:
    # 报告里含「① 假设站」（2A）、「② 实验站」（2D）与「图表线索」（2E）。卡片的挂载点见
    # attach_* —— 必须**在生成之前**挂: 否则报告会写「本次没有提出假设 / 没有做实验 /
    # 没读到图表线索」, 而界面列着好几条。
    attach_figures(job.audit, getattr(job, "figure_clues", None),
                   getattr(job, "figure_total", 0),
                   getattr(job, "figure_note", None))
    # 化学核验：报告在下面几行就生成，**必须先兜底跑过**。
    # 2026-10-25 起它是独立工具 chem_check（不再寄生在图表解析里），模型可以单独跳过；
    # 跳过了就在这里补算一遍，并标 fallback —— 允许跳过，但不许结果静默消失。
    ensure_chemistry(job)
    attach_chemistry(job.audit, getattr(job, "chem_items", None),
                     getattr(job, "chem_total", 0),
                     getattr(job, "chem_note", None),
                     fallback=bool(getattr(job, "chem_fallback", False)))
    attach_hypotheses(job.audit, getattr(job, "hypotheses", None),
                      getattr(job, "hypothesis_total", 0),
                      getattr(job, "derivation_note", None))
    attach_experiments(job.audit, getattr(job, "experiments", None),
                       getattr(job, "experiment_total", 0),
                       getattr(job, "experiment_note", None))
    job.report = generate_report(job.audit)
    tick("生成审计报告")


# ---------------------------------------------------------------- 工具登记表

@dataclass
class Tool:
    """一个可单独调用的工具。

    needs / gives 描述的是**产物依赖**, 调度员据此判断能否跳过;
    checkpoints 是它在界面进度条上覆盖的节点(顺序执行)。
    """
    key: str
    title: str
    checkpoints: tuple
    needs: tuple
    gives: tuple
    cost: str
    run: Callable
    core: bool = False        # 核心工具: 无论材料多残缺都必须执行


TOOLS: tuple = (
    Tool(key="parse_paper", title="论文解析",
         checkpoints=("论文解析",), needs=(), gives=("paper",),
         cost="本地解析", run=_run_parse_paper, core=True),
    Tool(key="analyze_images", title="图像解析",
         checkpoints=("图像解析",), needs=("paper",), gives=("image_params",),
         cost="按区域读图", run=_run_analyze_images),
    # 阶段二 2E：图表多模态解析（① 假设站「读」这一步）。
    # 与「图像解析」并列而不是合并: 一个读**超参数**、一个读**图本身**（图注 / 趋势 / 峰位）。
    # needs 只要 paper（图注来自正文文本层, 无代码包、无视觉 key 都能跑）。
    # 不标 core：材料残缺时由调度员决定跑不跑（与 2A/2D 同款）。
    Tool(key="read_figures", title="图表解析",
         checkpoints=("图表解析",), needs=("paper",),
         gives=("figure_clues",),
         cost="图注零成本 ＋ 按页读图（可选）", run=_run_read_figures),
    # 阶段二 2E-化学：化学身份核验（① 假设站「读」）。
    # 2026-10-25 从「图表解析」里**拆出来**: 合在一起时模型一跳过图表解析, 化学核验就
    # 跟着消失（同一份 PDF: 命令行 10 条、界面 0 条）。拆开后模型能单独决策,
    # 而且"不读图但核化学"是一条真实的省钱路径（读图花钱, 它不花钱）。
    # **不占进度格**（checkpoints=()）：它是"读"这一步的一种视野, 不是独立环节,
    #   与闭环复算同款口径 —— 不动 STEPS / 15 格契约。
    # 不标 core：纯本地、毫秒级, 由模型**知情**决定（排计划前程序已预扫正文,
    #   把"本文有 N 处化学标记"作为事实递过去, 见 planner._chem_survey）。
    Tool(key="chem_check", title="化学身份核验",
         checkpoints=(), needs=("paper",), gives=("chem_items",),
         cost="纯本地正则＋算术（毫秒级，零费用）", run=_run_chem_check),
    Tool(key="fill_missing", title="Agent 补漏抽取",
         checkpoints=("Agent 补漏抽取",), needs=("paper",), gives=("paper",),
         cost="最多 3 次 LLM / 2 次读图", run=_run_fill_missing),
    Tool(key="analyze_code", title="代码扫描",
         checkpoints=("代码扫描",), needs=(), gives=("code",),
         cost="本地解析", run=_run_analyze_code),
    # 阶段二 2A：假设生成。
    # needs 只要 paper —— **代码 ZIP 是可选的**：没有代码包时照样能提"论文独有"的卡
    # （那条假设反而更该问：论文声明的东西代码里根本找不到，到底怎么回事）。
    # 不标 core：材料太残缺时由调度员按材料决定跑不跑，不强制。
    Tool(key="hypothesize", title="假设生成",
         checkpoints=("假设生成",), needs=("paper",), gives=("hypotheses",),
         cost="纯规则零成本", run=_run_hypothesize),
    # 阶段二 2D：② 实验站 —— 把假设卡变成「怎么验」, 并**真的动手算一遍**。
    # needs 只要 hypotheses：没有假设卡就没有可设计的实验（不是"拦", 是真的无从下手）。
    # 排在 align（③ 站）之前: 先设计并做实验, 再拿实际结果去判 —— 顺序反了就没有"实际值"可判。
    # 不标 core：材料太残缺时由调度员按材料决定跑不跑（不强制, 与 2A/2B 同款）。
    Tool(key="design_experiment", title="实验设计与真跑",
         checkpoints=("实验设计",), needs=("hypotheses",), gives=("experiments",),
         cost="本地小规模复算（零成本, 不跑训练）", run=_run_design_experiment),
    Tool(key="align", title="参数对齐与结果核验",
         checkpoints=("参数归一化", "交叉匹配", "实验结果核验"),
         needs=("paper", "code"), gives=("findings",),
         cost="本地比对", run=_run_align, core=True),
    Tool(key="llm_analyze", title="AI 风险归因",
         checkpoints=("AI 风险归因",), needs=("findings",), gives=("ai_analysis",),
         cost="每次归因 1 次 LLM", run=_run_llm_analyze),
    Tool(key="selfcheck", title="审计自检",
         checkpoints=("审计自检",), needs=("findings",),
         gives=("selfcheck", "coverage", "readiness"),
         cost="本地校验", run=_run_selfcheck, core=True),
    Tool(key="rootcause", title="根因追查",
         checkpoints=("根因追查",), needs=("code", "findings"),
         gives=("rootcause",), cost="只读代码", run=_run_rootcause),
    Tool(key="patch", title="补丁草案",
         checkpoints=("补丁草案",), needs=("rootcause",),
         gives=("patches",), cost="只读代码、绝不落盘", run=_run_patch),
    Tool(key="report", title="生成审计报告",
         checkpoints=("生成审计报告",), needs=("findings",), gives=("report",),
         cost="本地渲染", run=_run_report, core=True),
)

TOOL_BY_KEY: dict = {t.key: t for t in TOOLS}

# ---- 消融档位（阶段三 3C）----
# 这三件**不许**被消融档位跳过: 跳了就没有「答案」可评（没有抽取=没参数; 没有比对=没结论;
# 没有报告=拿不到交付物）。是**断言**而不是注释 —— 档位写错要当场炸, 不能静默出一份空答卷。
ABLATION_LOCKED: frozenset = frozenset({"parse_paper", "align", "report"})
assert ABLATION_LOCKED <= set(TOOL_BY_KEY), "消融保护名单里出现了不存在的工具键"

# 进度节点: 由工具顺序自动推导, 保证与 pipeline.STEPS 完全一致
STEPS: list = [cp for t in TOOLS for cp in t.checkpoints]

# ---- 四站归属（阶段一 1A · 2026-10-15）----
# 只用于**界面分区展示**: 把进度条按「① 假设 / ② 实验 / ③ 验证 / ④ 迭代」分组,
# 让人一眼看出此刻在哪一站。**不参与调度、不改动 STEPS 本身、不影响任何执行行为。**
# 与「流水线工序」是同一批节点的两种看法: 工序看"哪件工具", 四站看"属于哪一段科研动作"。
#
# ② 实验站原先**没有环节**（实验设计在 2D 才落地）, 界面如实标「规划中」、不凭空塞一格充数;
# 2D 落地后按那时的承诺补上了「实验设计」一格 —— 现在四站都有真实环节。
STATIONS: tuple = ("① 假设", "② 实验", "③ 验证", "④ 迭代")

STATION_DESC: dict = {
    "① 假设": "读入材料（含图表多模态）· 提出可证伪的假设卡 · 先用推演筛掉伪分歧（2A 提假设 ＋ 2B 推演筛排 ＋ 2E 图表）",
    "② 实验": "对推不掉的假设设计实验并**真跑**（调用具名计算工具, 留下输入 / 输出 / 耗时的执行记录）",
    "③ 验证": "用实际结果判假设",
    "④ 迭代": "改完再验 · 出报告",
}

STATION_OF_STEP: dict = {
    # ① 假设 —— 读入材料（含图表）、形成线索, 再收敛成可证伪的假设卡（2A / 2E 已落地）
    "论文解析": "① 假设",
    "图像解析": "① 假设",
    "图表解析": "① 假设",
    "Agent 补漏抽取": "① 假设",
    "代码扫描": "① 假设",
    "假设生成": "① 假设",
    # ② 实验 —— 先设计、再真跑（阶段二 2D 落地后不再是空的）
    "实验设计": "② 实验",
    # ③ 验证 —— 判定与核验
    "参数归一化": "③ 验证",
    "交叉匹配": "③ 验证",
    "实验结果核验": "③ 验证",
    "AI 风险归因": "③ 验证",
    "审计自检": "③ 验证",
    # ④ 迭代 —— 追根因 / 出草案 / 出报告
    "根因追查": "④ 迭代",
    "补丁草案": "④ 迭代",
    "生成审计报告": "④ 迭代",
}
assert set(STATION_OF_STEP) == set(STEPS), (
    "四站归属必须与 STEPS 一一对应 —— 多一项或少一项都会让界面分区漏格")


# ================================================================ 工具总目录
# 2026-10-25 用户口径：「把工具全部整合到一个代码文件，其余通过调用这个文件来实现
# LLM 对工具的调用以及程序化的确定」—— 于是 `science_tools.py` 的登记部分搬到这里。
#
# 现在**看这一个文件就知道系统有哪些工具**：
#   · 上面 `TOOLS`            = 14 件审计工序（模型按计划逐个调）
#   · 下面 `SCI_TOOLS`        = 4 件具名科学计算工具（② 实验站算数用，执行体在 scifuncs）
#   · 再下面 `CATALOG`        = 把上面两批投影成给模型看的**同一张菜单**
#   · 文件末尾 `LOOP_NODES`   = 闭环扩展链的 4 个节点（不是工具，是"复核/推演"动作，不占进度格）
#
# 其余模块（planner / mcp_server / experiment）只 import 这里的登记表，
# **不许自己手抄一份** —— 抄一份就有两处口径，早晚漂。

# 3A 的具名科学计算工具：登记在这里，**执行体仍在 `scifuncs`**（那才是算法实现）。
# 为什么不在本文件重写一遍 run：工具表管"有哪些、怎么描述"，算法模块管"怎么算",
# 两者分开才看得清。这里只做**投影**，保证菜单与执行体一一对应、不会"菜单上有、跑不动"。
from .scifuncs import SCI_TOOLS  # noqa: E402  （放在这里是为了让上面 14 件工序先定义完）


@dataclass(frozen=True)
class CatalogEntry:
    """菜单上的一行。"""
    key: str                    # 工具键（模型只能从这个键里挑）
    title: str                  # 中文名
    cost: str                   # 成本说明（模型据此判断"值不值得跑"）
    needs: tuple = ()           # 需要什么产物
    gives: tuple = ()           # 产出什么
    core: bool = False          # 是否为核心步骤（如实登记；1C 后它**不再是程序闸门**，
                                # 只作为事实交给模型 —— 由模型自己决定要不要跳过）
    station: str = ""           # 归哪一站（四站口径；无进度格的登记为"（无进度格）"）
    kind: str = "audit"         # audit = 审计工序；science = 3A 的科学工具


def _station_of(tool) -> str:
    """工具归哪一站 —— 由它的**第一个进度格**决定（与进度条同一口径）。"""
    for cp in (tool.checkpoints or ()):
        st = STATION_OF_STEP.get(cp)
        if st:
            return st
    return "（无进度格）"


CATALOG: tuple = tuple(
    CatalogEntry(key=t.key, title=t.title, cost=t.cost,
                 needs=tuple(t.needs), gives=tuple(t.gives),
                 core=bool(t.core), station=_station_of(t), kind="audit")
    for t in TOOLS
)

# 科学工具归 ② 实验站 —— 2C 已把"② 只算数、③ 才判定"顶成结构了（见 scifuncs 模块头）。
SCIENCE_TOOLS: tuple = tuple(
    CatalogEntry(key=t.key, title=t.name, cost=t.cost,
                 needs=tuple(t.needs), gives=(t.gives,),
                 core=False, station="② 实验", kind="science")
    for t in SCI_TOOLS
)

# 防呆: 菜单必须与执行体同源同数（少一件就等于悄悄砍了一项能力）
assert len(CATALOG) == len(TOOLS), "菜单条目数必须等于执行体工具数"
assert {c.key for c in CATALOG} == {t.key for t in TOOLS}, "菜单与执行体的工具键必须一致"
assert len(SCIENCE_TOOLS) == len(SCI_TOOLS), "科学工具菜单条目数必须等于执行体件数"
assert {c.key for c in SCIENCE_TOOLS} == {t.key for t in SCI_TOOLS}, "科学工具菜单与执行体键必须一致"
# 菜单与执行体不许重名 —— 同名会让模型挑到哪一件变得说不清
assert not ({c.key for c in CATALOG} & {c.key for c in SCIENCE_TOOLS}), \
    "审计工序与科学工具的工具键不得重名"


def catalog_keys() -> frozenset:
    """白名单 —— 可被调用的**审计工序**工具键全集（清单外 = 不存在）。

    ⚠ 这里**只含审计工序**, 不含科学工具: 两者是两套东西, 分给两个调用方
       （调度员排工序 / ② 站挑计算工具）。要"全部可调用键"用 `all_keys()`。
    """
    return frozenset(c.key for c in CATALOG)


def all_keys() -> frozenset:
    """全部可调用工具键 = 审计工序 ＋ 科学工具（统一白名单, 用于"在不在清单上"的判断）。"""
    return frozenset(c.key for c in CATALOG) | frozenset(c.key for c in SCIENCE_TOOLS)


def menu() -> list:
    """给模型看的菜单（JSON 友好形状，字段名用中文，模型更好读）。"""
    return [{
        "key": c.key,
        "名称": c.title,
        "成本": c.cost,
        "需要产物": list(c.needs),
        "产出": list(c.gives),
        "核心步骤": c.core,
        "归属站": c.station,
    } for c in CATALOG]


def science_menu() -> list:
    """3A 的科学工具菜单（给 ② 站 / 答辩看的一行一条）。"""
    return [{"key": c.key, "名称": c.title, "成本": c.cost,
             "需要产物": list(c.needs), "产出": list(c.gives),
             "归属站": c.station} for c in SCIENCE_TOOLS]


def science_menu_for_model() -> list:
    """科学工具菜单的另一份形状 —— 带**说明**字段, 供模型挑工具时读。

    为什么与 `science_menu()` 分开而不是合并: `science_menu()` 是给人看的登记视图
    （字段名跟 CATALOG 一致, 免得两处口径漂）; 这份是**执行体的原样投影**, 字段名与
    `scifuncs.menu()` / `experiment.menu()` 完全同款, 三份菜单才能直接并起来递给模型。
    """
    from .scifuncs import menu as _impl
    return _impl()


# ---------------------------------------------------------------- 闭环扩展链的节点
# 这四件**不是工具**（模型不排它们、进度条不占格）: 它们是"核验之后再复核一遍"的动作,
# 由 `loop` 开关决定开不开。登记在这里只是为了"看一个文件就知道系统有哪几件动作"。
LOOP_NODES: tuple = (
    ("compute", "小规模复算", "把论文声称的数按公式重算一遍，看落不落得到同一个值"),
    ("search", "引证检索", "为结论找外部文献佐证，查不到就如实说查不到"),
    ("repro_plan", "复现方案", "把「怎么复现」写成可执行的步骤清单，并给可复现性评级"),
    ("loop", "迭代回边", "不通过的项回到假设站重来（≤3 轮，只在内存推演，绝不写文件）"),
)
