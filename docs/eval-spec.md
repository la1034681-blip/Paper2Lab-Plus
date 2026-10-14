# 评测口径说明（阶段七 · 评测打磨）

本文说明 Paper2Lab 怎么做量化评测、分数从哪来、以及**怎么挂进书生·司南
（SciEvalKit）**的评测流程。

## 1. 为什么是"自建考卷 + 契约对齐"

书生的科学智能评测工具链是 **SciEvalKit**
（`github.com/InternScience/SciEvalKit`，结果同步 OpenCompass 司南榜单）。
它评的是**通用科学能力** —— 会不会做科研题（符号推理 / 代码生成 / 图表理解 …）。
其榜单里**没有"审计"这一类任务**。

所以正确姿势是两件事分开做：

| 问题 | 做法 |
| --- | --- |
| 能不能被平台评测？ | 按 SciEvalKit 的接口把 Paper2Lab 包成"智能体" → 可被 `agent_runner.py` 拉起 |
| 审计抓错准不准？ | 用项目自有的真实论文+代码建**审计考卷**，判分格式对齐它 |

一句话：**接口用它的，题目用自己的**。不做"假装跑过它榜单"的叙述。

## 2. 三个契约（与 SciEvalKit 逐字一致）

`paper2lab/evalsuite/scieval_compat.py` 复刻了接入所需的全部契约，签名与官方一致
（`tests/verify_eval.py` 会**直接读官方源码**做比对）：

```text
agent   : .name / .model_version / .run(sample) -> EvalResult(success, final_answer)
sample  : EvalSample(prompt, images, files, metadata)
dataset : .dataset_name / __len__ / .build_agent_sample(idx) / .score_agent_sample(idx, ans, **kw)
落盘    : TrajectoryStore(root_dir).save_traj / load_traj / save_eval / load_eval
```

**为什么不直接装 SciEvalKit**：它的依赖很重（torch / vllm / transformers /
tensorflow-hub …），本地跑不动，也没必要。本层**零重依赖**（只用标准库），
因此评测在任何机器上都能离线复现。

## 3. 四维指标

| 指标 | 定义 | 看法 |
| --- | --- | --- |
| 全（recall） | 金标准"该抓到的参数"抓到了几成 | 越高越好 |
| 准（precision） | 报出的问题里，证据站得住（自证通过）的有几成 | 越高越好 |
| F1 | 准与全的调和平均 | 综合 |
| 自证率 | 全部结论的证据自证通过率 | 越高越好 |
| 诱饵误报 | "合法差异"被误升级为不一致的条数 | **越低越好** |
| 解析成功 | 作答能否被解析成结构化结果 | 必须为 1 |

**不做"通过/不通过"硬断言**：论文版本不同、个别超参可能只在附录，recall 合理低于 1
是正常的。评测的产出是一个**可横向比较的刻度**，不是开关 —— 这与
`verify_*.py` 那些断言型测试是两回事。

## 4. 金标准从哪来（防自证）

`paper2lab/evalsuite/goldset.py` 的 `expect`（该抓到的参数）与 `decoy`（诱饵）
**依据论文正文与官方开源实现的公开训练超参人工整理**，明确**不是**从本工具输出抄的。

| 类型 | 样本 | 用途 |
| --- | --- | --- |
| real | `Paper2Lab测试集\01_DDPM … 05_ViT`（arXiv 论文 + 官方代码） | 主考卷 |
| demo | `demo/`（演示样本）、比赛 D 题 | 回归 |
| edge | `demo/edge_code.zip`（刻意构造的合法差异） | 考精度 |
| external | 比赛 E 题（仓库外，缺失自动跳过） | 回归 |

例：DDPM 的 `expect = [learning_rate, batch_size, optimizer, warmup]`
（依据论文：Adam / batch 128 / lr 2e-4 带线性 warmup）。

## 5. 怎么跑

```bat
rem 单模式跑全套考卷
.venv\Scripts\python.exe tools\run_eval.py --agent-mode classic

rem 三模式同跑并对比（消融：只加能力、不改结论）
.venv\Scripts\python.exe tools\run_eval.py --agent-mode classic,multagent,closeloop

rem 只推理不判分 / 复用上次结果 / 另存报告
.venv\Scripts\python.exe tools\run_eval.py --mode infer
.venv\Scripts\python.exe tools\run_eval.py --reuse --report D:\eval_report.md
```

产出落 `eval_output\agent_eval\Paper2Lab-AuditBench\paper2lab\<版本>\`：
`sample_*_traj.json`（作答）、`sample_*_eval.json`（分数）、`summary.json`（汇总）。

## 6. 挂进真 SciEvalKit（三步）

本项目已按它的接口实现，接入时只需让它"认识"我们的 agent：

```bash
# 1. 克隆（轻量：只需 agents/ 与 dataset/ 相关部分）
git clone https://github.com/InternScience/SciEvalKit.git
cd SciEvalKit && pip install -e .     # 注意：会拉 torch 等重依赖

# 2. 把适配器放进它的 agents 包
copy <本仓库>\paper2lab\evalsuite\agent_adapter.py scieval\agents\paper2lab_agent.py
copy <本仓库>\paper2lab\evalsuite\scieval_compat.py scieval\agents\p2l_compat.py
```

```python
# 3. 在 scieval/agents/paper2lab_agent.py 顶部把 import 改指向它的包
from .p2l_compat import AgentBase, EvalResult, EvalSample   # 本地契约层
# 并让 Paper2LabAgent 直接使用它的实现（语义一致，仅路径不同）
```

再在 `agent_runner.py` 的 `_build_agent_from_config` 里登记一行：

```python
AGENT_MODULES = {
    "SmolAgentsAgent": "scieval.agents.smolagents",
    "Seed18Agent": "scieval.agents.seed18agent",
    "Deepseek32Agent": "scieval.agents.deepseek32agent",
    "Paper2LabAgent": "scieval.agents.paper2lab_agent",   # ← 新增
}
```

然后写一份 `agent_config.json`，跑 `python agent_runner.py --config agent_config.json` 即可。
`tools/agent_config.example.json` 是要改的样板。

## 7. 边界声明（答辩口径）

1. **不跑司南榜单**：Paper2Lab 是审计工具，司南榜里没有对应任务；我们**接口对齐、题目自建**。
2. **零成本**：评测全程 `force_mock`（不调模型、不花一分钱），可离线复现。
3. **不做"自证"**：金标准来自论文公开实现，判分器只度量不判定。
4. **本地可用、未发布**：服务与技能包按既定口径不上架（见 `docs/` 与 README §14）。
