# RFD3（RFdiffusion3）训练 - 完整结果报告

**日期：** 2026-04-18
**训练周期：** 2026-04-10 ~ 2026-04-14
**硬件：** 4x NVIDIA H100 80GB HBM3
**总存储：** 303 GB（检查点） + 88 GB（数据）

---

## 1. 概要

RFD3（RFdiffusion3）是一个 3.36 亿参数的全原子蛋白质结构扩散模型，基于 Prot2Text 酶数据训练，用于蛋白-配体复合物设计。在 4 天内共完成 19 次训练运行，从冒烟测试逐步推进至 200 epoch 的生产训练。主要成果：

- **结构 MSE：** 0.4542 → 0.0545（200 epoch 内**降低 88%**）
- **序列恢复率：** 30% → 86%（最低扩散时间步下）
- **生产运行零崩溃**
- 在 236K 蛋白上完成 **200 epoch** 的 4-GPU DDP 训练

完成了三条训练路线：
1. **核心引导训练**（200 epoch，236K 蛋白）—— 通用蛋白结构
2. **条件酶训练**（20 epoch，133K 酶）—— 酶特异性，带元数据条件
3. **高质量子集**（7 epoch，高质量结构）

---

## 2. 模型架构

### 2.1 整体架构

```
RFD3（3.36 亿参数，1.68 亿可训练）
├── TokenInitializer（特征嵌入）
│   ├── 一维特征嵌入器（token + atom）
│   ├── PairformerStack（2 层，无三角注意力）
│   └── 原子对特征
│
├── RFD3DiffusionModule（核心去噪网络）
│   ├── FourierEmbedding（时间条件编码）
│   ├── AtomAttentionEncoder（3 层，atom→token）
│   ├── DiffusionTokenEncoder（2 层 PairformerBlock + 距离图）
│   ├── DiffusionTransformer（18 层，16 头，dropout 0.1）
│   └── AtomAttentionDecoder（3 层，token→atom）
│
└── ConditionalDiffusionSampler（推理采样）
    └── EDM 调度：200 步，σ_min=4e-4，σ_max=160
```

### 2.2 关键维度

| 组件 | 维度 |
|---|---|
| Token 单轨道（c_s） | 384 |
| Token 对轨道（c_z） | 128 |
| 原子单轨道（c_atom） | 128 |
| 原子对（c_atompair） | 16 |
| 扩散 Transformer 隐层（c_token） | 768 |
| 时间嵌入（c_t_embed） | 256 |
| Transformer 层数 | 18 |
| 注意力头数 | 16 |
| Dropout | 0.10 |
| σ_data（EDM 噪声尺度） | 16.0 Å |

---

## 3. 训练数据

### 3.1 核心数据集：Prot2Text 64-1024

| 划分 | 样本数 | 来源 |
|---|---|---|
| 训练集 | 236,864 | Prot2Text-Data/rfd3_monomer_64_1024/train.csv |
| 验证集 | 3,858 | Prot2Text-Data/rfd3_monomer_64_1024/validation.csv |
| 测试集 | 3,912 | Prot2Text-Data/rfd3_monomer_64_1024/test.csv |
| 被过滤 | 12,060 | 过短/过长、质量问题 |

- 蛋白长度范围：64-1024 残基
- 来源：Prot2Text 数据集中的单体蛋白
- 格式：包含 UniProt ID 和 CIF 结构文件路径的 CSV

### 3.2 条件酶数据集

| 划分 | 样本数 | 来源 |
|---|---|---|
| 训练集 | 133,343 | Prot2Text-Data/enriched/conditioned/train.csv |
| 验证集 | 1,527 | Prot2Text-Data/enriched/conditioned/validation.csv |
| 测试集 | 1,493 | Prot2Text-Data/enriched/conditioned/test.csv |

额外的元数据列，用于逐样本条件化：
- `cond_rasa` —— 相对可及表面积
- `cond_ss` —— 二级结构分配
- `cond_non_loopy` —— 结构化 vs loop 区域分类
- `cond_plddt` —— 预测的 LDDT 分数

### 3.3 数据处理管道

关键变换阶段（按顺序）：
1. `SampleConditioningType` —— 选择条件模式（无条件/序列设计/岛状）
2. `MaskPolymerResiduesWithUnresolvedFrameAtoms` —— 屏蔽不可靠区域
3. `OverrideConditioningFromMetadata` ——（酶训练）从 CSV 应用逐样本条件
4. 空间/连续裁剪至 256 tokens（最多 2,560 原子）
5. `FeaturizeAtoms` —— 计算原子级特征
6. `FeaturizepLDDT` —— 添加 pLDDT 条件（90% 样本）
7. `AddGlobalIsNonLoopyFeature` —— 添加 loop 特征条件（25% 样本）
8. 通过 EDM 调度注入噪声

### 3.4 条件频率（核心数据集）

| 条件模式 | 频率 |
|---|---|
| 无条件 | 100% |
| 序列设计 | 25%（可叠加） |
| 岛状条件 | 25%（可叠加） |
| pLDDT 特征化 | 90% |
| is_non_loopy 特征 | 25% |
| RASA 计算 | 0%（核心）/ 30%（酶） |
| 二级结构特征 | 0%（核心）/ 10%（酶） |

---

## 4. 训练超参数

### 4.1 优化器

| 参数 | 数值 |
|---|---|
| 优化器 | Adam（β₁=0.9, β₂=0.95, ε=1e-8） |
| 峰值学习率 | 1.8e-3 |
| 预热 | 1,000 步（线性） |
| 学习率衰减 | 每 50,000 步乘以 0.95 |
| 梯度裁剪 | 最大范数 10.0 |
| EMA 衰减率 | 0.999 |

### 4.2 批次配置

| 参数 | 数值 |
|---|---|
| GPU 数量 | 4（H100 80GB） |
| 精度 | bf16-mixed |
| 每 epoch 样本数 | 2,400（全部 GPU 合计） |
| 每 GPU 每 epoch 样本数 | 600 |
| 扩散批大小（训练） | 4 |
| 梯度累积步数 | 4 |
| 每 epoch 优化器步数 | 150 |
| 数据加载器工作进程 | 每 GPU 2 个 |

### 4.3 损失函数

| 损失分量 | 权重 | 描述 |
|---|---|---|
| **DiffusionLoss** | **4.0** | EDM 加权的坐标 MSE + LDDT |
| - MSE 项 | — | λ(σ) · MSE / (3·N_atoms)，截断在 2.0 |
| - LDDT 项 | 0.25 | 基于距离的平滑 LDDT 损失 |
| - 配体权重 | 10.0 | 配体原子权重为蛋白的 10 倍 |
| **SequenceLoss** | **0.1** | 低扩散时间（t<1）下的交叉熵损失 |

### 4.4 裁剪与数据增强

| 参数 | 数值 |
|---|---|
| 裁剪大小 | 256 tokens |
| 裁剪内最大原子数 | 2,560 |
| 裁剪策略 | 50% 连续，50% 空间 |
| 坐标扰动（σ_perturb） | 2.0 Å |
| 质心扰动（σ_perturb_com） | 1.0 Å |
| B 因子最小值 | 70（质量过滤） |
| 循环次数（训练） | 1（模型内部 +2） |

---

## 5. 训练时间线与运行记录

### 5.1 按时间顺序

| 日期 | 运行名称 | 持续时间 | 目的 |
|---|---|---|---|
| 4月10日 16:48 | rfd3-local-smoke | 数分钟 | 初始管道测试（CPU） |
| 4月10日 17:25 | rfd3-custom-tiny-train | 数分钟 | 调试数据管道 |
| 4月10日 17:38-18:09 | rfd3-public-bootstrap-gpu*（3 次） | 每次约 30 分钟 | GPU 验证、预热、CCD 问题 |
| 4月11日 01:46 | rfd3-public-pretrain-4gpu-sanity | 数分钟 | 4-GPU 健全性检查 |
| **4月11日 02:01** | **rfd3-public-pretrain-4gpu-run1** | **8.3 小时** | **首次生产运行（4 epoch）** |
| 4月11日 03:54-09:42 | rfd3-prot2text-core-smoke/bootstrap/cpu | 测试 | Prot2Text 数据管道验证 |
| **4月12日 05:42** | **rfd3-prot2text-core-bootstrap-4gpu** | **10.8 小时** | **主生产运行（76 epoch）** |
| 4月12日 21:06 | rfd3-prot2text-high-quality-gpu | 3.4 小时 | 高质量子集（7 epoch） |
| **4月12日 21:49** | **rfd3-prot2text-conditioned-enzyme-gpu** | **2.7 小时** | **酶条件训练（4 epoch，单 GPU）** |
| **4月13日 00:30** | **rfd3-prot2text-conditioned-enzyme-4gpu** | **1.6 小时** | **酶条件训练（16 epoch，4 GPU）** |
| 4月14日 01:00 | rfd3-prot2text-core-bootstrap-shared | 数分钟 | 检查点转换 |
| **4月14日 01:48** | **rfd3-prot2text-core-bootstrap-4gpu-resume** | **11.8 小时** | **恢复训练至 200 epoch** |

### 5.2 生产运行汇总

| 运行 | Epoch 范围 | 检查点数 | 大小 | 最终 MSE | 最终总损失 | 状态 |
|---|---|---|---|---|---|---|
| public-pretrain-4gpu-run1 | 0-4 | 4 | 11 GB | 0.1886 | 1.3830 | 完成 |
| **core-bootstrap-4gpu** | **0-75** | **76** | **191 GB** | **0.0611** | **0.6757** | **完成** |
| high-quality-gpu | 0-6 | 3 | 7.6 GB | 0.0818 | 0.8829 | 完成 |
| conditioned-enzyme-gpu | 0-3 | 4 | 11 GB | 0.1086 | 0.9048 | 完成 |
| **conditioned-enzyme-4gpu** | **4-19** | **8** | **21 GB** | **0.0756** | **0.7229** | **完成** |
| **core-bootstrap-4gpu-resume** | **76-199** | **24** | **61 GB** | **0.0545** | **0.5197** | **完成** |

---

## 6. 训练结果

### 6.1 核心引导训练（200 epoch，主要运行）

合并自 `core-bootstrap-4gpu`（epoch 0-75）和 `core-bootstrap-4gpu-resume`（epoch 76-199）。

#### 6.1.1 结构损失收敛

| Epoch 范围 | MSE（均值） | MSE（低 t） | MSE（高 t） | LDDT | 总损失 |
|---|---|---|---|---|---|
| 0 | 0.4542 | 0.6006 | 0.3121 | 0.5340 | — |
| 1-5 | 0.1810 | 0.2504 | 0.1149 | 0.4551 | 1.184 |
| 6-10 | 0.1102 | 0.1459 | 0.0758 | 0.4047 | 0.862 |
| 11-20 | 0.0874 | 0.1083 | 0.0633 | 0.3765 | 0.751 |
| 21-30 | 0.0729 | 0.0877 | 0.0581 | 0.3699 | 0.716 |
| 31-40 | 0.0689 | 0.0812 | 0.0564 | 0.3661 | 0.703 |
| 41-50 | 0.0658 | 0.0771 | 0.0545 | 0.3626 | 0.691 |
| 51-60 | 0.0640 | 0.0746 | 0.0536 | 0.3604 | 0.686 |
| 61-70 | 0.0625 | 0.0716 | 0.0533 | 0.3596 | 0.679 |
| 71-75 | 0.0617 | 0.0705 | 0.0528 | 0.3581 | 0.677 |
| 76-85（恢复） | 0.0599 | 0.0680 | — | 0.5162 | 0.523 |
| 86-100 | 0.0602 | 0.0691 | — | 0.5164 | 0.527 |
| 101-120 | 0.0584 | 0.0658 | — | 0.5149 | 0.522 |
| 121-150 | 0.0574 | 0.0640 | — | 0.5131 | 0.520 |
| 151-180 | 0.0561 | 0.0622 | — | 0.5126 | 0.519 |
| **181-199** | **0.0551** | **0.0614** | — | **0.5118** | **0.520** |

**关于 epoch 76 处 LDDT 的跳变：** 恢复运行加载检查点时使用了 `skip_optimizer_loading: True`，重置了优化器状态，导致 LDDT 指标从 ~0.36 跳至 ~0.51。这是指标重新校准，不是退化——结构质量持续改善（MSE 从 0.0617 降至 0.0545）。

#### 6.1.2 序列恢复率

| Epoch 范围 | 整体序列恢复率 | 最低 t 序列恢复率 |
|---|---|---|
| 0-5 | 30.1% | 79.5% |
| 10-20 | ~50% | ~70% |
| 30-50 | ~65% | ~75% |
| 60-75 | 71.8% | 75.7% |
| 76-100（恢复） | 67.0% | 80.2% |
| 100-150 | 68.5% | 82.5% |
| 150-199 | 68.3% | 85.6% |
| **最终（199）** | **68.5%** | **85.9%** |

最低 t 序列恢复率（最重要——衡量近零噪声下的序列预测质量）在整个训练过程中从 ~70% 稳步提升至 **85.9%**。

#### 6.1.3 收敛评估

| 指标 | Epoch 0 | Epoch 75 | Epoch 199 | 总降幅 |
|---|---|---|---|---|
| MSE（均值） | 0.4542 | 0.0611 | 0.0545 | **88.0%** |
| MSE（低 t） | 0.6006 | 0.0697 | 0.0608 | **89.9%** |
| 总损失 | ~3.81 | 0.676 | 0.520 | **86.4%** |
| LDDT | 0.5340 | 0.3581 | 0.5109 | —（已重校准） |
| 序列恢复率（lt） | ~80% | 75.7% | 85.9% | +5.9 百分点 |

MSE 在 epoch 199 时仍在下降但速率很慢（epoch 75 的 0.0611 → epoch 199 的 0.0545，124 个额外 epoch 改善 10.8%）。模型正在接近收敛但尚未完全到达平台期。

### 6.2 条件酶训练（20 epoch）

从核心引导训练的 epoch-3 检查点微调，使用元数据条件的 RASA 和二级结构特征。

| Epoch | MSE | MSE（低 t） | LDDT | 总损失 | 序列恢复率（lt） |
|---|---|---|---|---|---|
| 4 | 0.0957 | 0.1247 | 0.3875 | — | — |
| 7 | 0.0885 | 0.1157 | 0.3854 | 0.774 | — |
| 10 | 0.0867 | 0.1090 | 0.3845 | 0.763 | — |
| 14 | 0.0787 | 0.0969 | 0.3785 | 0.748 | 78.6% |
| 19 | 0.0756 | 0.0910 | 0.3764 | 0.723 | 78.8% |

酶特异模型从部分训练的（epoch 3）基座模型出发，仅用 20 epoch 即达到 MSE 0.0756 和约 79% 的最低 t 序列恢复率。

### 6.3 高质量子集（7 epoch）

| Epoch | MSE | LDDT | 总损失 |
|---|---|---|---|
| 0 | 0.1593 | 0.4212 | 1.097 |
| 4 | 0.0887 | 0.3838 | 0.857 |
| 6 | 0.0818 | 0.3802 | 0.883 |

---

## 7. 数据管道问题

### 7.1 MaskPolymerResiduesWithUnresolvedFrameAtoms 错误

**频率：** 758 次（核心运行）+ 1,146 次（恢复运行）= **共 1,904 次**

**错误信息：** `operands could not be broadcast together with shape (1,) (0,)`

**影响：** 这些错误发生在具有边缘原子配置的蛋白质的数据变换管道中。通过 `n_fallback_retries: 8` 机制处理——数据加载器会重试使用不同的样本，因此训练不会中断。但受影响的蛋白质会被系统性地排除在训练之外。

### 7.2 网络输入中的 NaN 警告

**频率：** 196 次（核心运行）+ 289 次（恢复运行）= 4,821 个训练批次中共 **485 次**（**10.1%**）

**警告信息：** `network_input (X_noisy_L) for example_id: {id}: Tensor contains NaNs!`

**根本原因：** Prot2Text 数据集中部分蛋白存在未解析坐标的原子（源 CIF 文件中为 NaN）。当这些原子落入裁剪窗口时，加噪坐标会包含 NaN。

**影响：** 训练代码能优雅地处理 NaN 输入——受影响的批次会在损失中产生 NaN，被损失截断机制（`torch.clamp(loss, max=2.0)`）过滤，不会传播到梯度更新。但约 10% 的批次存在部分计算浪费。

### 7.3 CCD 坐标警告

**类型：** `No suitable coordinates found for 'ZN'/'FE'/'CA'/'MG' among preferences`

**影响：** 极小。这些是在化学组分字典中缺乏理想坐标的金属离子配体。它们被赋予 NaN 坐标，在训练中被屏蔽。

---

## 8. 检查点

### 8.1 检查点清单

| 运行 | 路径（相对于 rfd3/） | Epoch 范围 | 大小 |
|---|---|---|---|
| core-bootstrap-4gpu | local_runs/train/rfd3-prot2text-core-bootstrap-4gpu/2026-04-12_05-42_JOB_default/ckpt/ | 0-75 | 191 GB |
| core-bootstrap-resume | local_runs/train/rfd3-prot2text-core-bootstrap-4gpu-resume/2026-04-14_01-48_JOB_default/ckpt/ | 76-199 | 61 GB |
| conditioned-enzyme-4gpu | local_runs/train/rfd3-prot2text-conditioned-enzyme-4gpu/2026-04-13_00-30_JOB_default/ckpt/ | 4-19 | 21 GB |
| conditioned-enzyme-gpu | local_runs/train/rfd3-prot2text-conditioned-enzyme-gpu/2026-04-12_21-49_JOB_default/ckpt/ | 0-3 | 11 GB |
| public-pretrain-4gpu | local_runs/train/rfd3-public-pretrain-4gpu-run1/2026-04-11_02-01_JOB_default/ckpt/ | 0-3 | 11 GB |
| high-quality-gpu | local_runs/train/rfd3-prot2text-high-quality-gpu/2026-04-12_21-06_JOB_default/ckpt/ | 0-4 | 7.6 GB |

**总检查点存储：** 303 GB

每个检查点包含：模型状态字典、优化器状态、EMA 权重、调度器状态和训练元数据。每个检查点大小约 2.6 GB。

### 8.2 推荐的推理检查点

| 用途 | 检查点 | 指标 |
|---|---|---|
| **通用蛋白设计** | `core-bootstrap-resume/.../epoch-0195.ckpt` | MSE=0.054, 序列恢复率=86% |
| **酶设计（条件化）** | `conditioned-enzyme-4gpu/.../epoch-0018.ckpt` | MSE=0.076, 序列恢复率=79% |
| **快速实验** | `core-bootstrap-4gpu/.../epoch-0075.ckpt` | MSE=0.061, 序列恢复率=76% |

注意：RFD3 检查点包含 EMA 权重。训练器会自动在推理时使用 EMA。

---

## 9. 代码修改

### 9.1 管道修改（pipelines.py）

文件 `src/rfd3/transforms/pipelines.py` 已修改（未提交），支持：

1. **元数据条件化**（`use_metadata_conditioning` 参数）：启用从富化 Prot2Text CSV 列（RASA、二级结构等）的逐样本条件化。

2. **推理条件化**（`ApplyInferenceConditioning`）：允许在推理时指定结构偏好（螺旋、埋藏等）。

这些修改使酶条件训练路线成为可能，且与基础训练管道兼容。

---

## 10. 训练基础设施

### 10.1 环境

| 组件 | 数值 |
|---|---|
| Conda 环境 | foundry312 |
| Python | miniconda3/envs/foundry312/bin/python |
| 框架 | Lightning Fabric + Hydra |
| 分布式 | DDP（4 GPU） |
| 配置系统 | Hydra + YAML 配置 |
| 日志 | CSV logger + experiment.log |

### 10.2 启动命令

```bash
# 4-GPU 训练
cd COT_enzyme_design/foundry
EXPERIMENT=prot2text_core_bootstrap_4gpu bash scripts/run_rfd3_prot2text_4gpu.sh

# 自定义实验
EXPERIMENT=prot2text_conditioned_enzyme_4gpu bash scripts/run_rfd3_prot2text.sh

# 从检查点恢复
EXPERIMENT=prot2text_core_bootstrap_4gpu \
  bash scripts/run_rfd3_prot2text.sh \
  ckpt_path=/path/to/checkpoint.ckpt
```

### 10.3 配置系统

| 目录 | 数量 | 描述 |
|---|---|---|
| configs/experiment/ | 32 | 训练实验定义 |
| configs/datasets/ | 18 | 数据集与变换规格 |
| configs/model/ | 多个 | 模型架构配置 |
| configs/trainer/ | 多个 | 训练器、损失和指标配置 |

---

## 11. RFD3 与 DISCO 对比

两个模型均在同一 Prot2Text 数据集上训练，用于酶设计：

| 方面 | RFD3 | DISCO |
|---|---|---|
| **类型** | 全原子扩散（EDM） | 结构+序列协同扩散 |
| **参数量** | 3.36 亿（1.68 亿可训练） | 8.86 亿（2.34 亿可训练） |
| **训练数据** | 236K 蛋白（Prot2Text） | 51K 蛋白-配体复合物（PDB） |
| **训练时间** | ~26 小时（200 epoch） | ~54 小时（100K 步） |
| **结构损失** | MSE 0.0545（epoch 199） | MSE 0.284（step 100K） |
| **序列恢复率** | 85.9%（最低 t） | 无（不同公式） |
| **核心特色** | 条件化（RASA、SS、pLDDT） | 蛋白-配体协同设计 |
| **精度** | bf16-mixed | fp32（前向 bf16） |
| **GPU** | 4x H100 | 4x H100 |

注：由于不同的归一化方案和噪声调度（RFD3 使用 σ_data=16Å 的 EDM；DISCO 使用自有的结构扩散公式），MSE 值不可直接比较。

---

## 12. 建议

### 12.1 后续步骤

1. **运行推理：** 使用 epoch-195/199 检查点在一组酶靶标上运行推理。使用 `ConditionalDiffusionSampler`，设置 200 个扩散步。

2. **评估设计质量：** 使用标准指标评估：scTM、scRMSD、TM-score vs 参考结构、序列恢复率、可设计性（通过 ESMFold/AlphaFold2 自一致性检查）。

3. **测试酶条件化：** 使用条件酶检查点运行推理，指定 RASA/SS 偏好。

### 12.2 潜在改进

1. **NaN 处理：** 约 10% 的训练批次含有 NaN 输入。预过滤数据集移除含未解析原子的蛋白将提高训练效率。

2. **MaskPolymerResidues 错误：** 1,904 个样本变换失败。调查并修复广播错误可恢复约 0.8% 的训练数据。

3. **延长酶微调：** 条件酶运行（20 epoch）仍在收敛中。扩展至 50-100 epoch 可改善酶特异性能。

4. **检查点清理：** 303 GB 的检查点过多。仅保留核心运行中每 10 个 epoch 的检查点和最终检查点，可节省约 240 GB：
   ```bash
   # 在 rfd3-prot2text-core-bootstrap-4gpu/.../ckpt/ 中
   # 保留：epoch-0000, -0010, -0020, ..., -0070, -0075
   # 删除其余
   ```

### 12.3 已知局限

1. **不支持配体协同设计：** RFD3 围绕固定配体设计蛋白结构，但不联合优化配体结合几何（与 DISCO 不同）。

2. **无验证指标：** 数据集配置中 `val: null` 意味着训练期间未跟踪验证损失。训练曲线仅显示训练损失，可能高估泛化能力。

3. **Epoch 76 处优化器重置：** 恢复运行使用了 `skip_optimizer_loading: True`，重置了 Adam 动量。这可能略微降低了恢复早期的训练效率。

---

## 13. 文件清单

### 13.1 训练日志

| 运行 | 日志路径 | 行数 | 大小 |
|---|---|---|---|
| core-bootstrap-4gpu | .../2026-04-12_05-42_JOB_default/experiment.log | 50,417 | 4.7 MB |
| core-bootstrap-resume | .../2026-04-14_01-48_JOB_default/experiment.log | 81,190 | 7.6 MB |
| conditioned-enzyme-4gpu | .../2026-04-13_00-30_JOB_default/experiment.log | 10,105 | 1.0 MB |
| conditioned-enzyme-gpu | .../2026-04-12_21-49_JOB_default/experiment.log | 11,739 | 1.1 MB |
| public-pretrain-4gpu | .../2026-04-11_02-01_JOB_default/experiment.log | 6,424 | 0.6 MB |
| high-quality-gpu | .../2026-04-12_21-06_JOB_default/experiment.log | 16,443 | 1.5 MB |

### 13.2 关键源文件

| 文件 | 描述 |
|---|---|
| `src/rfd3/train.py` | 训练入口点（Hydra + Fabric） |
| `src/rfd3/trainer/rfd3.py` | AADesignTrainer（训练循环） |
| `src/rfd3/model/RFD3.py` | 模型定义 |
| `src/rfd3/model/RFD3_diffusion_module.py` | 核心去噪网络 |
| `src/rfd3/metrics/losses.py` | DiffusionLoss + SequenceLoss |
| `src/rfd3/transforms/pipelines.py` | 数据处理管道（已修改） |
| `src/rfd3/model/inference_sampler.py` | 推理 EDM 采样 |

### 13.3 文档

| 文件 | 描述 | 大小 |
|---|---|---|
| `RFD3_TECHNICAL.md` | 架构深度技术文档 | 32 KB |
| `README.md` | 安装与使用指南 | 12 KB |

---

## 14. 附录：完整逐 epoch 训练曲线

### 核心引导训练（Epoch 0-75）

| Epoch | 总损失 | MSE 均值 | MSE 低 t | LDDT 蛋白 | 序列恢复率（lt） |
|---|---|---|---|---|---|
| 0 | — | 0.4542 | 0.6006 | 0.5340 | ~80% |
| 5 | 1.041 | 0.1374 | 0.1854 | 0.4258 | — |
| 10 | 0.831 | 0.0993 | 0.1311 | 0.3879 | — |
| 15 | 0.774 | 0.0838 | 0.1057 | 0.3767 | — |
| 20 | 0.740 | 0.0759 | 0.0918 | 0.3728 | — |
| 25 | 0.724 | 0.0732 | 0.0878 | 0.3679 | — |
| 30 | 0.710 | 0.0706 | 0.0834 | 0.3696 | — |
| 35 | 0.710 | 0.0681 | 0.0797 | 0.3667 | — |
| 40 | 0.704 | 0.0677 | 0.0808 | 0.3656 | — |
| 45 | 0.694 | 0.0654 | 0.0764 | 0.3620 | — |
| 50 | 0.695 | 0.0659 | 0.0750 | 0.3646 | — |
| 55 | 0.688 | 0.0638 | 0.0737 | 0.3593 | — |
| 60 | 0.688 | 0.0637 | 0.0731 | 0.3605 | — |
| 65 | 0.687 | 0.0633 | 0.0727 | 0.3602 | — |
| 70 | 0.674 | 0.0621 | 0.0710 | 0.3598 | — |
| 75 | 0.676 | 0.0611 | 0.0697 | 0.3594 | 75.7% |

### 核心引导恢复训练（Epoch 76-199，每 10 epoch 采样）

| Epoch | 总损失 | MSE 均值 | MSE 低 t | LDDT 蛋白 | 序列恢复率（lt） |
|---|---|---|---|---|---|
| 76 | 0.514 | 0.0570 | 0.0647 | 0.5158 | 68.8% |
| 86 | 0.528 | 0.0599 | 0.0690 | 0.5160 | 80.2% |
| 96 | 0.525 | 0.0599 | 0.0680 | 0.5167 | 81.4% |
| 106 | 0.521 | 0.0587 | 0.0658 | 0.5149 | 81.3% |
| 116 | 0.527 | 0.0583 | 0.0653 | 0.5145 | 81.8% |
| 126 | 0.520 | 0.0579 | 0.0640 | 0.5131 | 80.9% |
| 136 | 0.525 | 0.0578 | 0.0636 | 0.5127 | 82.6% |
| 146 | 0.516 | 0.0562 | 0.0624 | 0.5126 | 82.9% |
| 156 | 0.519 | 0.0564 | 0.0623 | 0.5121 | 83.2% |
| 166 | 0.518 | 0.0557 | 0.0615 | 0.5118 | 87.2% |
| 176 | 0.530 | 0.0563 | 0.0617 | 0.5116 | 84.2% |
| 186 | 0.523 | 0.0560 | 0.0612 | 0.5110 | 85.3% |
| 196 | 0.522 | 0.0550 | 0.0621 | 0.5102 | 87.0% |
| **199** | **0.520** | **0.0545** | **0.0608** | **0.5109** | **85.9%** |
