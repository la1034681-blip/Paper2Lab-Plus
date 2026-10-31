# 参数清单与判定口径

本文件说明「两边到底在比什么」以及「怎么算对不上」。
需要更细的行为定义时，以项目源码 `paper2lab/synonyms.py` 与 `paper2lab/aligner.py` 为准。

## 支持的参数种类（22 类）

顺序即抽取优先级（前面的先抽、被引用的概率更高）。

| 参数键 | 显示名 | 常见写法（论文 / 代码） |
| --- | --- | --- |
| `learning_rate` | Learning Rate | lr、base_lr、eta、init_lr、学习率 |
| `batch_size` | Batch Size | batch、bs、per_device_train_batch_size、批大小 |
| `epochs` | Epochs | epoch、num_epochs、max_epochs、训练轮数 |
| `iterations` | 训练迭代次数 | iter、n_iter、max_steps、total_steps、训练步数 |
| `optimizer` | Optimizer | optim、solver、优化器 |
| `weight_decay` | Weight Decay | wd、l2_reg、权重衰减 |
| `momentum` | Momentum | beta1、动量 |
| `dropout` | Dropout | dropout_rate、drop_prob |
| `seed` | Random Seed | manual_seed、random_state、随机种子 |
| `dataset` | Dataset | dataset_name、数据集 |
| `model` | Model Architecture | arch、backbone、net、网络结构 |
| `input_size` | Input Size | image_size、crop_size、resolution、输入尺寸 |
| `warmup` | Warmup | warmup_epochs、warmup_steps |
| `scheduler` | LR Scheduler | lr_scheduler、学习率调度 |
| `num_clusters` | Cluster Count K | n_clusters、聚类数 |
| `threshold` | Threshold | thresh、阈值 |
| `sigma_xy` | 水平定位标准差 σ | 水平误差标准差（工程类论文） |
| `sigma_z` | 深度定位标准差 σ_z | 深度误差标准差（工程类论文） |
| `kill_radius` | 杀伤半径 R | （工程类论文） |
| `depth_est` | 深度定位值 h₀ | 中心深度定位值 |
| `min_depth` | 最小深度 | （工程类论文） |
| `num_bombs` | 投弹数量 | （工程类论文） |

> 后 6 类来自项目早期的工程题验证样本，保留是为了不破坏既有结论的可比性。

## 取值合理性区间

抽取后会做一次合理性校验，明显越界的值**不会**进入比对（否则会制造假冲突），
而是记入 `rejected` 清单如实示众。例如：

| 参数 | 合法区间 | 是否必须为整数 |
| --- | --- | --- |
| `learning_rate` | 0 ~ 1（开区间） | 否 |
| `batch_size` | 1 ~ 1,000,000 | 是 |
| `epochs` | 1 ~ 100,000 | 是 |
| `iterations` | 1 ~ 100,000,000 | 是 |
| `weight_decay` / `momentum` / `dropout` | 0 ~ 1 | 否 |

（区间定义见 `paper2lab/synonyms.py` 的取值表，此处只列常用的几条。）

## 一条结论的状态

| 状态 | 含义 |
| --- | --- |
| `consistent` | 归一化后两边相等 |
| `inconsistent` | 归一化后两边不等（**核心风险**） |
| `missing_in_code` | 论文声明了，代码里没找到对应实现 |
| `missing_in_paper` | 代码里有，论文未声明 |
| `not_fixed` | 论文要求固定（如随机种子），代码未固定 |
| `unverifiable` | 代码里是运行时动态决定的值，静态无法确认 |
| `internal_inconsistent` | 同一套设置里给了两个值才算：论文侧须**同页同章节**且其余细节一致；代码侧须**可能同时生效**（同文件 / 同目录的入口与配置 / 根目录平铺）。按任务分别取值、分目录的多套配置只出「多处取值」提示，不是此状态 |

## 归一化规则（为什么"看起来不一样"其实是同一个值）

比对前会做归一化，避免把等价写法报成冲突：

- 数值：`1e-4` = `0.0001`、`5e-4` = `0.0005`、`1/32` = `0.03125`、`50%` = `0.5`；
- 文本：小写化、去空格与连字符（`CIFAR-10` = `cifar10`）；
- 调度器：`CosineAnnealingLR` = `cosine annealing` = `余弦退火`；
- 优化器：`AdamW` ≠ `Adam`（**不合并**，这是真差异）。

## 会被报为「分歧」的典型情形

用大白话说，下面这些**不是 bug，是真实存在、需要人来判断的情况**：

1. **同一参数在论文里出现两处不同值** —— 例如正文写 1e-4、附录表格写 1e-3。
   系统会并列两边，不替你选。
2. **论文与代码对同一个概念的命名体系不同** —— 例如论文说「余弦退火」，
   代码写 `CosineAnnealingLR`，规则版可能报分歧；开启角色规范化后可自动归到同一标记。
3. **单位口径不同** —— 论文用步数（1000 步）、代码用比例（5%），
   「算」会尝试换算，换不上就如实记为「未算」而不是猜。
4. **一侧确实缺失** —— 论文没写种子，代码也没固定，这属于 `not_fixed`，是复现风险而非冲突。

> 遇到分歧时的正确做法：**如实并列给用户，让他判断是论文写得不清楚，还是实现跑偏了。**
> 不要在简报里把分歧说成"一致"，也不要替用户在两者之间选一个当真值。
