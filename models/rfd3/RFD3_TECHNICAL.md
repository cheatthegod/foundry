# RFdiffusion3 (RFD3) 技术深度解析

> 基于源码的完整技术文档，覆盖模型结构、训练流程与推理细节。
> 对应论文：*De novo Design of All-atom Biomolecular Interactions with RFdiffusion3*（bioRxiv, 2025）

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [模型超参数总览](#2-模型超参数总览)
3. [TokenInitializer：特征嵌入模块](#3-tokeninitializer特征嵌入模块)
4. [RFD3DiffusionModule：核心去噪网络](#4-rfd3diffusionmodule核心去噪网络)
5. [注意力机制详解](#5-注意力机制详解)
6. [自条件化（Self-Conditioning）与 Recycling](#6-自条件化self-conditioning与-recycling)
7. [条件控制系统](#7-条件控制系统)
8. [训练流程](#8-训练流程)
9. [损失函数](#9-损失函数)
10. [推理采样流程](#10-推理采样流程)
11. [Classifier-Free Guidance（CFG）](#11-classifier-free-guidancecfg)
12. [对称设计采样器](#12-对称设计采样器)
13. [Partial Diffusion（部分扩散）](#13-partial-diffusion部分扩散)
14. [低内存模式](#14-低内存模式)
15. [输入规格与推理引擎](#15-输入规格与推理引擎)
16. [序列设计头](#16-序列设计头)
17. [数据 Pipeline](#17-数据-pipeline)
18. [已知设计限制与工程细节](#18-已知设计限制与工程细节)

---

## 1. 整体架构概览

RFdiffusion3 是一个**全原子扩散模型**，能够联合生成蛋白质骨架坐标、全原子侧链、核酸、小分子配体等多类型生物分子的三维结构，并同步预测氨基酸序列。

在代码层级上，`RFD3`（`model/RFD3.py`）是最外层的 `nn.Module`，包含三个子模块：

```
RFD3
├── TokenInitializer          ← 特征编码（仅在推理开始时调用一次）
├── RFD3DiffusionModule       ← 每一步扩散的去噪网络
└── ConditionalDiffusionSampler  ← 推理时的完整扩散轨迹控制器
```

训练时，`forward()` 执行单步去噪（给定已加噪坐标和时间步 $t$）；
推理时，`forward()` 调用 `sample_diffusion_like_af3`，执行完整的 200 步扩散轨迹。

---

## 2. 模型超参数总览

所有维度均来自 `configs/model/components/rfd3_net.yaml`：

| 符号 | 配置键 | 值 | 含义 |
|------|--------|----|------|
| `c_s` | `c_s` | 384 | Token 单特征维度（Pairformer single track） |
| `c_z` | `c_z` | 128 | Token 对特征维度（Pairformer pair track） |
| `c_atom` | `c_atom` | 128 | 原子级单特征维度 |
| `c_atompair` | `c_atompair` | 16 | 原子对特征维度（P_LL） |
| `c_token` | `c_token` | 768 | Diffusion Transformer 隐层维度 |
| `c_t_embed` | `c_t_embed` | 256 | 时间步 Fourier 嵌入维度 |
| `σ_data` | `sigma_data` | 16 | EDM 数据噪声尺度（Å） |

---

## 3. TokenInitializer：特征嵌入模块

**源码**：`src/rfd3/model/layers/encoders.py` → `class TokenInitializer`

TokenInitializer 对应 AlphaFold3 的 trunk，但**移除了 MSA 和 Template 模块**，仅保留序列特征嵌入和轻量 Pairformer。它在每次推理/训练的开始执行**一次**，结果在扩散的所有步骤中共享复用。

### 3.1 Token 级 1D 特征嵌入

以下特征拼接后经线性变换到 `c_s = 384`：

| 特征键 | 维度 | 含义 |
|--------|------|------|
| `restype` | 32 | 残基/核苷酸/小分子的类型 one-hot |
| `ref_motif_token_type` | 3 | motif token 类型（0=非motif, 1=固定坐标, 2=固定序列） |
| `ref_plddt` | 1 | 预测的 pLDDT 置信度（训练时按概率提供） |
| `is_non_loopy` | 1 | 是否为二级结构（helices/sheets）的预测标志 |

> 所有嵌入通过 `OneDFeatureEmbedder` 实现：各特征独立线性投影，投影后相加（不是拼接）。

### 3.2 Atom 级 1D 特征嵌入（两次）

同一套原子特征分别投影到两个不同空间，用于不同目的：
- 第一次：→ `c_s = 384`，用于 downcast 到 token 级
- 第二次：→ `c_atom = 128`，作为原子级初始特征 `Q_L_init`

以下特征拼接（全部通过 `OneDFeatureEmbedder`）：

| 特征键 | 维度 | 含义 |
|--------|------|------|
| `ref_atom_name_chars` | 256 | 原子名字符编码（如"CA"、"CB"等，4字节×64维） |
| `ref_element` | 128 | 元素类型 one-hot（C/N/O/S/... ） |
| `ref_charge` | 1 | 形式电荷 |
| `ref_mask` | 1 | 坐标是否已解析 |
| `ref_pos` | 3 | 参考坐标（xyz，来自配体 CCD 或参考框架） |
| `ref_is_motif_atom_with_fixed_coord` | 1 | 该原子坐标是否被固定（motif） |
| `ref_is_motif_atom_unindexed` | 1 | 该原子是否为无序列索引 motif |
| `has_zero_occupancy` | 1 | 晶体学占位率是否为零 |
| `ref_atomwise_rasa` | 3 | **相对可及表面积**（3档：总/亲水/疏水）—用于 CFG |
| `active_donor` | 1 | 是否为氢键供体（需 HBPLUS）—用于 CFG |
| `active_acceptor` | 1 | 是否为氢键受体（需 HBPLUS）—用于 CFG |
| `is_atom_level_hotspot` | 1 | 是否为 PPI 热点原子 |

### 3.3 Token 对特征 Z_II 初始化

$$Z^{(0)}_{II} = \text{Linear}(S_I)_{\text{outer-sum}} + \text{RPE}(f) + \text{bond\_embedding}(f[\text{token\_bonds}]) + \text{ref\_pos\_embedding\_tok}$$

其中：
- **outer-sum**：`to_z_init_i(S_I)[i] + to_z_init_j(S_I)[j]`，类似 AF3 的 outer product
- **RPE**（`RelativePositionEncodingWithIndexRemoval`）：在 `r_max=32` 范围内编码残基距离、token 距离和链内对称 ID。对**无索引 motif** 之间的位置对，强制设为"未知位置" bin，防止错误的位置偏置
- **token_bonds**：共价键连接矩阵（整数 → 1D 线性投影）
- **ref_pos_embedding_tok**：配体/small molecule 内部参考坐标的逆平方距离嵌入（仅在 is_ca 层面有效）

### 3.4 轻量 Pairformer Stack

2 块 `PairformerBlock`（无三角更新、无 MSA），每块包含：
1. `Z_II += Transition(Z_II)`（SwiGLU MLP，expansion=4）
2. `S_I += AttentionPairBias(S_I, Z_II)`（16 头，KQ-norm）
3. `S_I += Transition(S_I)`

全程 **bfloat16**（`force_bfloat16=True`），使用 activation checkpointing 节省显存。

### 3.5 原子对特征 P_LL（L×L）

$$P_{LL} = P_{\text{motif}} + P_{\text{ref}} + \text{process\_single\_l}(C_L)[\text{outer\_sum}] + \text{process\_z}(Z_{II})[l][m] + \text{pair\_MLP}$$

各项含义：

| 项 | 来源 | 含义 |
|----|------|------|
| $P_{\text{motif}}$ | `SinusoidalDistEmbed` | motif 原子间距离的正弦嵌入（32 对 sin/cos频率 → `c_atompair`），仅在 `is_motif_atom_with_fixed_coord` 对上计算 |
| $P_{\text{ref}}$ | `PositionPairDistEmbedder` | 同一 token 内参考框架的逆平方距离（配体内部参考位置），仅在 `is_motif_atom_with_fixed_seq` 对上计算 |
| outer-sum | `process_single_l` + `process_single_m` | 原子级单特征的外加和 |
| token 投影 | `process_z` → index by tok_idx | 从 Z_II 降采样到原子对 |
| pair_MLP | 3层 ReLU MLP | 混合上述特征 |

此外，P_LL 还通过 `pairwise_mean_pool` 聚合回 token 对，作为 Z_II 的补充（`Z_II += project_pll(pooled_P_LL)`）。

> **P_LL 是 RFD3 的关键设计**：它把 motif 的全原子坐标信息和配体内部构象信息**预先编码为静态对特征**，之后扩散的每一步都能利用这些信息，而无需重新计算。

---

## 4. RFD3DiffusionModule：核心去噪网络

**源码**：`src/rfd3/model/RFD3_diffusion_module.py` → `class RFD3DiffusionModule`

这是每步去噪的实际执行体，接收：
- `X_noisy_L`：当前含噪的原子坐标 `[D, L, 3]`（D=扩散批大小）
- `t`：时间步标量或向量
- `f`：特征字典
- `Q_L_init`, `C_L`, `P_LL`, `S_I`, `Z_II`：来自 TokenInitializer 的固定特征

### 4.1 时间步处理

时间 $t$ 首先扩展为原子级时间向量：

$$t_L[i] = t \cdot \mathbb{1}[\text{is\_motif\_atom\_with\_fixed\_coord}[i] = 0]$$

即**固定 motif 原子的时间为 0**，不参与扩散。相应地 token 级 $t_I$ 也类似处理。

时间嵌入使用 **Fourier Embedding**（`FourierEmbedding`）：

$$n_t = \text{FourierEmbed}\left(\frac{1}{4}\log\frac{t}{\sigma_{\text{data}}}\right) \in \mathbb{R}^{c_{t\_embed}=256}$$

- 原子级时间条件：经 `RMSNorm + Linear` 投影到 `c_atom=128`，加到 `C_L`
- Token 级时间条件：投影到 `c_s=384`，加到 `S_I`
- 零时刻掩码：`t_L=0` 的位置时间特征置零（`C_L *= (t_L > 0).float()`）

### 4.2 位置输入的 EDM 缩放

$$R_{\text{noisy}} = \frac{X_{\text{noisy}}}{\sqrt{t^2 + \sigma_{\text{data}}^2}}$$

这是 EDM（Karras et al., 2022）的标准 input scaling，将噪声水平归一化到 $[-1, 1]$ 附近。另外还维护一个基于**全局时间 $t$**（忽略原子级差异）的 $R_{L,\text{uniform}}$，用于 DiffusionTokenEncoder 中的 Distogram 特征。

### 4.3 初始特征融合（进入 UNet 前）

```python
A_I = LinearEmbedWithPool(R_noisy)   # 坐标 → token 均值池化 [D, I, c_token]
S_I = Downcast_c(C_L, S_I)           # 原子 → token，cross-attention downcast
Q_L = Q_L_init + process_r(R_noisy)  # 原子初始特征 + 噪声位置投影
C_L = C_L + time_embed_atom          # 加时间条件
S_I = S_I + time_embed_token         # 加时间条件
C_L = C_L + process_c(C_L)           # 残差精炼
```

### 4.4 UNet 编码器：LocalAtomTransformer（3 块）

对原子级特征 `Q_L [D, L, c_atom]` 执行**局部原子自注意力**：

- 注意力键集合：每个查询原子关注**序列邻近 2 个 token 内的原子** + **3D 空间最近的 128 个原子**（由 `create_attention_indices` 动态计算）
- Pair Bias：来自预计算的 `P_LL` 的稀疏子集
- 条件化：通过 `AdaLN`（Adaptive Layer Norm）注入 `C_L`（含时间信息的原子特征）
- 输出门控：`output = sigmoid(Linear(C_L)) * attn_out`（adaLN-Zero 风格）
- Transition：`ConditionedTransitionBlock`（SwiGLU，expansion=2，AdaLN 条件化）

编码完成后：

```python
A_I = Downcast_q(Q_L, A_I, S_I)  # 更新 token 特征（cross-attention，4头）
```

### 4.5 Recycling 主循环

以下 3 步在 `forward_with_recycle` 中循环执行，非最后一轮使用 `torch.no_grad()`：

#### Step A：DiffusionTokenEncoder

融合自条件化和 Distogram 信息到 pair track：

1. `S_I += 2x Transition(S_I)`
2. 构建 `Z_II` 的多通道输入：
   - `Z_II_init`（来自 TokenInitializer，128维）
   - **Distogram**（65 bins，基于 $R_{L,\text{uniform}}$ 的 CA 距离）
   - **自条件化**（$D_{II,\text{self}}$，65 bins，来自上一轮预测的 CA 距离分桶）
3. 拼接后经 `RMSNorm + Linear` 压缩回 128 维
4. 2 块 Pairformer（无三角更新，16 头，bfloat16）精炼 `(S_I, Z_II)`

#### Step B：DiffusionTransformer（主干，18 块）

`LocalTokenTransformer` 对 token 级特征 `A_I [D, I, c_token=768]` 执行：

- 同样使用稀疏局部注意力（`n_keys=32`，序列邻居 `n_local_tokens=8`）
- Pair Bias 来自 `Z_II`（128 维 → 16 头）
- 条件化来自 `S_I`（AdaLN）
- 共 **18 块**，10% Dropout

#### Step C：CompactStreamingDecoder（3 块）

从 token 级信息解码回原子级坐标特征：

对每块（共 3 次）：
1. **Upcast**：`A_I → Q_L`，通过 Cross-Attention（query=原子, key/value=token，3路分裂，4头，`c_model=128`）
2. **LocalAtomTransformer**：原子级自注意力（同编码器结构）

最后：
```python
A_I = Downcast(Q_L.detach(), A_I.detach(), S_I.detach())  # detach 防止梯度回传
```

> 注意 Decoder 的 downcast 使用了 `.detach()`，这是为了避免反向传播时的梯度回路问题。

### 4.6 输出坐标的 EDM 反缩放

$$X_{\text{out}} = \frac{\sigma_{\text{data}}^2}{\sigma_{\text{data}}^2 + t^2} X_{\text{noisy}} + \frac{\sigma_{\text{data}} \cdot t}{\sqrt{\sigma_{\text{data}}^2 + t^2}} R_{\text{update}}$$

其中 $R_{\text{update}} = \text{to\_r\_update}(Q_L)$（`RMSNorm + Linear(c_atom, 3)`）。

这是 EDM 的标准 skip connection + 去噪方向，确保小 $t$ 时主要输出噪声原坐标，大 $t$ 时主要输出网络预测。

---

## 5. 注意力机制详解

### 5.1 稀疏局部注意力（`sparse_pairbias_attention`）

RFD3 的所有注意力层均使用**稀疏局部注意力**，不执行全序列 $O(L^2)$ 注意力。

**关键索引集合**的构建（`create_attention_indices`）：
- 序列邻居：±2 个 token 内的所有原子（`n_attn_seq_neighbours=2`）
- 空间邻居：基于当前 $X_{\text{noisy}}$ 的 kNN 最近 128 个原子（`n_attn_keys=128`）
- 合并去重，得到 `indices [D, L, k]`

稀疏注意力计算：

```python
K_gathered = K[batch_idx, indices]          # [D, L, k, c]
V_gathered = V[batch_idx, indices]          # [D, L, k, c]
B_gathered = B[query_idx, indices, :]       # [D, L, k, H]
attn = (Q @ K_gathered) / sqrt(c/H) + B    # [D, H, L, k]
out  = softmax(attn) @ V_gathered           # [D, H, L, c/H]
out  = G * out                              # 可学习门控
```

训练时使用 `full=True` 模式（完整 L×L 注意力矩阵 + 稀疏 mask），便于构建更优化的计算图；推理时使用 `full=False` 的 gather 模式节省内存。

### 5.2 Pair Bias 注意力（PairformerBlock 中）

`AttentionPairBiasPairformerDeepspeed`：
- 与普通注意力不同，attention logits 加上来自 pair track `Z_II` 的 per-head bias：`B_IIH = to_b(LayerNorm(Z_II)) + Beta_II`
- 全程 bfloat16
- 门控：`out = sigmoid(Linear(A_I)) * aggregated_values`

### 5.3 Upcast / Downcast（跨层级信息交换）

**Downcast（原子 → token）**（`cross_attention` 方法）：
- query = token 表示 `A_I [D, I, c_s]`（单 token 查询）
- key/value = 归属于该 token 的所有原子 `Q_IA [D, I, max14, c_atom]`
- 4 头，`c_model=128`，掩码排除 padding 原子

**Upcast（token → 原子）**（`cross_attention` 方法）：
- query = 原子 `Q_IA [D, I, max14, c_atom]`
- key/value = token 表示（分 3 段切分 `A_I`，每段 `c_token/3 = 256`）
- 这种分段设计降低了计算量，同时保留了全 token 信息

---

## 6. 自条件化（Self-Conditioning）与 Recycling

RFD3 实现了类 AF3 的**自条件化**机制：

1. 第一次 recycle：`D_II_self = zeros([B, I, I, 65])`（零初始化）
2. 每次 recycle 结束：`D_II_self = bucketize_distogram(X_out_CA.detach())`，将预测坐标的 CA 距离分桶到 65 个 bin（1-30 Å，EDM 缩放后）
3. 下一次 recycle 时，`D_II_self` 作为额外的 pair 信息输入 `DiffusionTokenEncoder`

**Recycle 次数**：
- 训练时：随机采样，从 1 到 `n_recycles_train=2` 之间均匀采样（每个 epoch 预先生成固定 schedule，保证同 batch 内所有 GPU 相同）
- 推理时：固定 `n_recycle=2`（来自模型配置）
- 非最后一轮使用 `torch.no_grad()` 节省显存
- 最后一轮开启梯度（用于反向传播）

> `torch.clear_autocast_cache()` 在每次有梯度的 recycle 前调用，这是针对 PyTorch autocast 已知 bug 的 workaround（issue #65766）。

---

## 7. 条件控制系统

### 7.1 核心原子注解（布尔掩码）

| 注解 | 训练时来源 | 推理时来源 |
|------|-----------|-----------|
| `is_motif_atom_with_fixed_coord` | `SampleConditioningFlags` 随机采样 | 输入 JSON/YAML 显式指定 |
| `is_motif_atom_with_fixed_seq` | 同上 | 同上 |
| `is_motif_atom_unindexed` | 同上（某些 motif 可"浮动"） | 同上 |

时间步处理时，固定坐标原子的 $t_L=0$，即这些原子不被加噪，模型"看到"其真实坐标。

### 7.2 训练条件采样分布

配置在 `configs/datasets/design_base.yaml` 的 `train_conditions`：

| 条件类型 | 频率权重 | 含义 |
|----------|---------|------|
| `unconditional` | 5.0 | 无任何条件，完全从噪声生成 |
| `sequence_design` | 2.0 | 固定骨架 + unfixed 序列（类 inverse folding） |
| `island` | 1.0 | 随机连续片段作为 motif |
| `tipatom` | 0.0（默认关闭） | 仅固定少数关键原子（tip atoms） |
| `ppi` | 0.0（默认关闭） | 蛋白质-蛋白质界面条件 |

### 7.3 Meta 条件概率

同样在 `design_base.yaml`：

| 概率参数 | 值 | 效果 |
|----------|-----|------|
| `calculate_hbonds` | 0.2 | 20% 概率提供 H 键注解（需 HBPLUS） |
| `calculate_rasa` | 0.6 | 60% 概率提供 RASA 注解 |
| `keep_protein_motif_rasa` | 0.1 | 10% 概率保留 motif 蛋白质的 RASA |
| `add_ppi_hotspots` | 0.75 | PPI 任务中 75% 概率提供热点信息 |
| `full_binder_crop` | 0.75 | PPI 任务中 75% 概率保留完整 binder |
| `featurize_plddt` | 0.9 | 单体蒸馏时 90% 概率提供 pLDDT |
| `add_global_is_non_loopy_feature` | 0.99 | 99% 概率提供二级结构特征 |
| `add_1d_ss_features` | 0.1 | 10% 概率提供残基级 SS 条件 |

### 7.4 无索引 Motif（Unindexed Motif）

部分 motif 可以"无序列位置"（如未知结合位点的配体）：
- 使用 `unindexing_pair_mask` 屏蔽其 RPE（位置编码中标记为"未知位置"bin）
- 损失函数中对 unindexed token 使用不同时间缩放因子（`unindexed_t_alpha=0.75`）
- 推理输出后通过 RMSD 最小化算法插回原始位置

---

## 8. 训练流程

### 8.1 整体配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam（`betas=[0.9, 0.95]`, `eps=1e-8`） |
| 学习率调度 | AF3Scheduler：`warmup_steps=1000`, `base_lr=1.8e-3`, 指数衰减 `decay_factor=0.95` 每 `decay_steps=50000` 步 |
| 精度 | `bf16-mixed`（bfloat16 混合精度） |
| 梯度裁剪 | `clip_grad_max_norm=10.0` |
| 梯度累积 | `grad_accum_steps=3`（默认，可通过启动脚本覆盖） |
| EMA | 衰减率 `0.999`（来自 AF3） |
| 每 epoch 样本数 | 2400 |
| 最大 epoch | 100,000 |
| 分布式策略 | `ddp`（Lightning Fabric） |
| Batch size（扩散） | 每样本 32 个噪声实例（`diffusion_batch_size_train=32`） |

### 8.2 数据裁剪

| 参数 | 值 |
|------|-----|
| token crop 大小 | 384 |
| 最大原子数（crop 内） | 3840（约 10× crop size） |
| 中心残基 | CB（空间裁剪的中心） |
| 扰动 $\sigma$ | 2.0 Å（坐标增广） |

### 8.3 训练步骤详解（`AADesignTrainer.training_step`）

```
1. 从 recycle_schedule 获取本 batch 的 n_cycle（所有 GPU 相同）
2. 从 AtomWorks 数据 pipeline 获取 example（crop 后的结构）
3. 构建网络输入：
   X_noisy_L = coord_atom_lvl_to_be_noised + noise  (来自 SampleEDMNoise)
4. 调用 model.forward(input, n_cycle=n_cycle)
   → TokenInitializer（一次）→ DiffusionModule（单步去噪，带 n_cycle 次 recycle）
5. 计算 DiffusionLoss + SequenceLoss
6. fabric.backward(total_loss)
7. 梯度累积达到 grad_accum_steps 时，optimizer.step() + EMA.update()
```

### 8.4 验证步骤（`AADesignTrainer.validation_step`）

验证步骤执行**完整的扩散轨迹**（200 步推理），并计算：
- 骨架度量（如 CA-RMSD、Ramachandran 分布）
- H键满足率（如提供）
- PPI 界面度量

---

## 9. 损失函数

### 9.1 DiffusionLoss（权重 4.0）

**源码**：`src/rfd3/metrics/losses.py` → `class DiffusionLoss`

$$\mathcal{L}_{\text{diff}} = w \cdot \overline{\lambda(\sigma) \cdot \text{MSE}(X_{\text{pred}}, X_{\text{gt}})}$$

其中 **EDM λ 权重**：

$$\lambda(\sigma) = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma \cdot \sigma_{\text{data}})^2}$$

这使得高噪声（大 $\sigma$）的步骤权重低，低噪声（小 $\sigma$）的精细步骤权重高。

**Per-Atom 权重系数**（影响 MSE 中各原子的权重）：

| 原子类型 | 权重 alpha | 配置项 |
|----------|-----------|--------|
| 一般原子 | 1.0 | - |
| 虚拟原子（guideposts） | 1.0 | `alpha_virtual_atom` |
| 极性原子 | 1.0 | `alpha_polar_residues` |
| **配体原子** | **10.0** | `alpha_ligand` |
| 无索引 motif | 1.0（时间缩放0.75） | `alpha_unindexed_diffused` |

**无索引 token 的特殊处理**：

$$\mathcal{L}_{\text{unindexed}} = \frac{\lambda(t \cdot \alpha_t)}{\lambda(t)} \cdot \mathcal{L}_{\text{base}}$$

其中 $\alpha_t = 0.75$，等效于给 unindexed token 使用更小的有效时间步，从而减轻其在高噪声阶段的损失权重（因为这些 token 的准确位置本就难以预测）。

**MSE 归一化**：除以 `3 * n_resolved_atoms`（即坐标维度数 × 有坐标原子数）。

**损失截断**：`torch.clamp(loss, max=2.0)` 防止梯度爆炸。

**可选 LDDT 损失**（权重 0.25）：

$$\mathcal{L}_{\text{LDDT}} = 1 - \frac{1}{4} \sum_{d \in \{0.5, 1.0, 2.0, 4.0\}} \sigma(d - |\Delta r_{ij}|) $$

仅对 CA-CA 距离 < 15 Å（蛋白质）或 < 30 Å（核酸）的原子对计算，不跨 token 内部计算，使用 activation checkpointing 节省显存。

### 9.2 SequenceLoss（权重 0.1）

$$\mathcal{L}_{\text{seq}} = w \cdot \overline{\text{CrossEntropy}(\hat{s}_I, s_I^{\text{gt}})} \quad \text{for } t \in [0, 1]$$

- 仅在有有效时间 $t$ 的 diffusion batch 上计算
- 按序列 valid mask 加权（只对有 ground truth 序列的 token 计算）
- 最大值截断：`min(loss, 4.0)`
- 记录 `seq_recovery` 和 `lowest_t_seq_recovery` 作为训练指标

---

## 10. 推理采样流程

### 10.1 噪声时间表

$$\hat{t}_k = \sigma_{\text{data}} \left[ s_{\max}^{1/p} + k \cdot \left( s_{\min}^{1/p} - s_{\max}^{1/p} \right) \right]^p$$

其中 $k$ 从 $t_{\min}=0$ 到 $t_{\max}=1$ 均匀采样 `num_timesteps=200` 个点。

默认参数：$p=7$，$s_{\min}=4\times10^{-4}$，$s_{\max}=160$，$\sigma_{\text{data}}=16$。

这产生一个从 $\hat{t}_0 \approx 160\text{ Å}$ 到 $\hat{t}_{199} \approx 4\times10^{-4}\text{ Å}$ 的指数衰减时间表。

### 10.2 初始结构

$$X_L^{(0)} = \hat{t}_0 \cdot \mathcal{N}(0, I) + \text{coord\_motif}$$

motif 原子的噪声被清零（`noise[is_motif]=0`），使扩散从 motif 附近开始。

### 10.3 主采样循环（AF3 风格随机 ODE）

对每对相邻时间步 $(c_{t-1}, c_t)$（从大到小）：

**Step 1：随机增广（可选）**
```python
if allow_realignment:
    X_L = random_rotation(X_L) + random_translation(scale=s_trans)
    X_L[motif] = rigid_align_motif(X_L, coord_motif)
```

**Step 2：加扰动噪声（Stochastic Sampler）**

$$\hat{t} = c_{t-1} \cdot (1 + \gamma)$$

$$\epsilon_L \sim \text{noise\_scale} \cdot \sqrt{\hat{t}^2 - c_{t-1}^2} \cdot \mathcal{N}(0, I)$$

$$X_{\text{noisy}} = X_L + \epsilon_L$$

其中 $\gamma = \gamma_0 = 0.6$（若 $c_t > \gamma_{\min}=1.0$ 则有扰动，否则 $\gamma=0$ 退化为确定性 ODE）。

motif 原子的 $\epsilon$ 置零。

**Step 3：模型去噪**
```python
outs = diffusion_module(X_noisy, t=t_hat, f=f, **tokenizer_outputs)
X_denoised = outs["X_L"]
```

**Step 4：梯度估计**
$$\delta_L = \frac{X_{\text{noisy}} - X_{\text{denoised}}}{\hat{t}}$$

**Step 5：更新坐标**
$$X_L^{\text{new}} = X_{\text{noisy}} + s_{\text{step}} \cdot (c_t - \hat{t}) \cdot \delta_L$$

默认 $s_{\text{step}} = 1.5$（步长缩放，大于 1 使步长更激进，倾向于生成更"可设计"但多样性稍低的结构）。

### 10.4 最终对齐

若存在固定 motif 且启用 `allow_realignment`：
1. 在最终 $X_L$ 上插入 GT motif（刚体对齐）
2. 将预测结构对齐到 GT motif 的原始位置

---

## 11. Classifier-Free Guidance（CFG）

**源码**：`src/rfd3/model/inference_sampler.py` + `configs/inference_engine/rfdiffusion3.yaml`

### 11.1 工作原理

在每个扩散步骤，额外执行一次**无条件前向传播**：

```python
f_uncond = strip_features(f, cfg_features)  # 将指定特征清零
X_denoised_uncond = model(X_noisy_stripped, f_uncond)
delta_uncond = (X_noisy - X_denoised_uncond) / t_hat

# CFG 混合
delta_guided = delta + (cfg_scale - 1) * (delta - delta_uncond)
```

### 11.2 CFG 特征集合

被清零的特征（即"条件"）：
- `ref_atomwise_rasa`：表面可及性（控制是否设计到特定暴露程度）
- `active_donor`：氢键供体条件
- `active_acceptor`：氢键受体条件

这意味着 RFD3 的 CFG 主要用于**增强氢键和溶剂暴露条件的遵守程度**。

### 11.3 默认参数

| 参数 | 默认值 |
|------|--------|
| `use_classifier_free_guidance` | False（推理默认关闭） |
| `cfg_scale` | 1.5 |
| `cfg_t_max` | None（全过程启用） |

启用 CFG 会导致每步两次模型前向，计算量翻倍。

---

## 12. 对称设计采样器

**源码**：`src/rfd3/model/inference_sampler.py` → `class SampleDiffusionWithSymmetry`

### 12.1 工作原理

在扩散的前 `sym_step_frac=0.9`（90%）的步骤中，每步对去噪坐标施加**对称变换**：

```python
if "X_L" in outs and c_t > gamma_min_sym:
    outs["X_L"] = apply_symmetry_to_xyz_atomwise(outs["X_L"], symmetry_feats)
```

这迫使模型的去噪方向始终符合对称约束，通过"对称投影"将结构拉向对称流形。

最后 10% 的步骤不强制对称，允许局部精修打破微小不对称（这也是为什么要求 `gamma_0 > 0.5`，保证此阶段仍有足够随机性）。

### 12.2 支持的对称类型

通过 `configs/model/samplers/symmetry.yaml` 配置，支持 $C_n$（cyclic）等点群对称，对称帧存储在 `f["sym_transform"]` 中。

---

## 13. Partial Diffusion（部分扩散）

当输入特征字典中存在 `f["partial_t"]` 时，噪声时间表被截断：

```python
noise_schedule = noise_schedule[noise_schedule <= partial_t]
```

这意味着只执行从 $t=\text{partial\_t}$ 到 $t=0$ 的扩散，相当于对输入结构添加中等水平噪声后再去噪，实现**对现有结构的局部重设计**。

输出时会计算 CA-RMSD（去噪结果 vs 输入结构）作为质量指标。

---

## 14. 低内存模式

**激活方式**：环境变量 `RFD3_LOW_MEMORY_MODE=1` 或推理配置 `low_memory_mode: True`

### 14.1 问题背景

`P_LL` 的尺寸为 `[L, L, c_atompair]`，当系统中原子数 $L$ 较大时（如 DNA/RNA 复合物），显存占用 $O(L^2)$ 会迅速增长。

### 14.2 解决方案：ChunkedPairwiseEmbedder

在低内存模式下：
1. TokenInitializer 不预计算完整 `P_LL`，而是预计算各原子的**静态 MLP 投影**（`cache_static_projections`）
2. 每次注意力层需要对特定原子对 `(q, k)` 的 pair feature 时，仅计算这些对（`forward_chunked`）
3. `ChunkedSinusoidalDistEmbed` 和 `ChunkedPositionPairDistEmbedder` 是对应的分块版本，**共享训练好的权重**

核心思想：将空间复杂度从 $O(L^2)$ 降为 $O(L \cdot k)$（$k$ 为注意力窗口大小）。

---

## 15. 输入规格与推理引擎

### 15.1 RFD3InferenceEngine

**源码**：`src/rfd3/engine.py`

推理引擎的主要流程：
```
inputs (JSON/YAML/AtomArray)
  ↓ process_input()
DesignInputSpecification (Pydantic 验证)
  ↓ _multiply_specifications() [跳过已存在的]
design_specifications dict
  ↓ assemble_distributed_inference_loader()
DataLoader
  ↓ trainer.validation_step()
RFD3Output (AtomArray + metadata)
  ↓ output.dump()
.cif.gz + .json
```

### 15.2 输入 JSON/YAML 格式

支持 `global_args` 键对所有样本共享参数：

```json
{
  "global_args": {
    "diffusion_batch_size": 16
  },
  "design_1": {
    "input": "path/to/input.pdb",
    "contigmap": {"contigs": ["A1-50/0 50-100"]},
    "hotspot_res": ["A30", "A35"]
  }
}
```

### 15.3 输出文件

每个设计产生：
- `{prefix}_model_{i}.cif.gz`：结构文件（biotite AtomArray 转 mmCIF 格式，gzip 压缩）
- `{prefix}_model_{i}.json`：元数据，包含：
  - `specification`：原始输入规格
  - `inference_sampler`：采样器配置
  - `ckpt_path`：使用的权重路径
  - `metrics`：骨架度量（CA-RMSD、Ramachandran 违例率等）
  - `diffused_index_map`：设计残基到输出链/残基号的映射

### 15.4 后处理流程

1. **清除 Guidepost**：若 `cleanup_guideposts=True`，移除辅助虚拟原子
2. **清除虚拟原子**：若 `cleanup_virtual_atoms=True`，恢复真实原子名和元素信息
3. **序列读取**：从 `sequence_head` 的 argmax 中提取预测序列
4. **无索引 motif 插入**：通过 RMSD 最小化将 unindexed motif 插回正确位置

---

## 16. 序列设计头

**源码**：`src/rfd3/model/layers/blocks.py` → `class LinearSequenceHead`

### 16.1 架构

```python
logits = Linear(c_token=768 → 32)(A_I)  # 简单线性层
```

输出 32 维 logits，对应 AF3 序列编码的 32 个 token（标准氨基酸 + 核苷酸 + 特殊符号）。

### 16.2 Decode 策略

不直接 argmax，而是先掩码掉不合法 token（UNK、X、DX、`<G>`），再归一化后取最大：

```python
probs = softmax(logits) * valid_out_mask
probs = probs / probs.sum()
indices = probs.argmax()
```

### 16.3 与旧版 SequenceHead 的区别

代码中同时保留了旧版 `SequenceHead`（结合 distogram 特征的 MLP）和新版 `LinearSequenceHead`（简单线性层）。训练和推理均使用 `LinearSequenceHead`（通过配置中的 `sequence_head` 指定）。

---

## 17. 数据 Pipeline

**源码**：`src/rfd3/transforms/pipelines.py` → `build_atom14_base_pipeline`

基于 AtomWorks 框架，训练 pipeline 的主要步骤：

| 步骤 | 变换 | 说明 |
|------|------|------|
| 1 | `RemoveHydrogens` | 去除氢原子 |
| 2 | `HandleUndesiredResTokens` | 处理非标准残基 |
| 3 | `FlagNonPolymersForAtomization` | 标记配体为"原子化"模式 |
| 4 | `AtomizeByCCDName` | 配体展开到全原子（来自 CCD 字典） |
| 5 | `CropSpatialLikeAF3` / `CropContiguousLikeAF3` | 空间或序列裁剪到 384 tokens |
| 6 | `SampleConditioningType` | 随机采样条件类型（unconditional/island/ppi等） |
| 7 | `SampleConditioningFlags` | 设置各原子的固定/浮动标志 |
| 8 | `CreateDesignReferenceFeatures` | 创建参考坐标特征 |
| 9 | `BatchStructuresForDiffusionNoising` | 生成多个扩散批次（32个noise实例） |
| 10 | `SampleEDMNoise` | 为每批次采样 EDM 噪声时间步 $t$ |
| 11 | `FeaturizeAtoms` | 最终特征化 |
| 12 | `ConvertToTorch` | 转为 Tensor |

训练数据集分为两种：
- **Interface 数据集**（`rfd3_train_interface`）：包含蛋白质-蛋白质界面的结构
- **PN Unit 数据集**（`rfd3_train_pn_unit`）：更大的蛋白质-核酸复合物

---

## 18. 已知设计限制与工程细节

### 18.1 代码中已标注的 Bug 和修复

1. **SwiGLU 实现错误**（`blocks.py:57-59`）：
   `ConditionedTransitionBlock` 中原始代码使用 `sigmoid(gate) * linear(x)`，而正确的 SwiGLU 应为 `silu(gate) * linear(x)`。代码中保留了注释说明并已修复，但错误版本可能仍存在于某些旧权重中。

2. **Protenix 技术报告 Bugfix**（`blocks.py:331`）：
   RPE 中链距离计算使用 `same_entity` 而非 `not same_chain`（与 AF3 原始伪代码不同，与 Protenix 技术报告对齐）。

3. **NVIDIA CUDA Kernel 未实现**（`attention.py:62`）：
   `cuequivariance_torch` 的 fused attention pair bias kernel 接口已定义但 `raise NotImplementedError`，当前实际使用 PyTorch 原生实现。

4. **Downcast 中的 detach**（`blocks.py:774`）：
   `CompactStreamingDecoder` 的最终 downcast 使用 `.detach()`，防止梯度通过序列预测头回流到坐标预测路径。

### 18.2 训练-推理差异

| 方面 | 训练 | 推理 |
|------|------|------|
| `n_recycle` | 随机 1-2，非最后轮 `no_grad` | 固定 2 |
| 注意力模式 | `full=True`（完整矩阵+mask） | `full=False`（稀疏 gather） |
| 初始坐标 | 从噪声 pipeline 提供 | 从高斯噪声初始化 |
| 批量大小 | 32 diffusion instances/样本 | 默认 8 |
| 数据增广 | 随机旋转+平移 | 可选 `allow_realignment` |

### 18.3 分布式训练注意事项

- 使用 Lightning Fabric DDP
- Recycle schedule 预生成确保**同 batch 内所有 GPU 使用相同的 recycle 数**（防止 DDP 同步不一致）
- 梯度累积 `grad_accum_steps=3`，需通过启动脚本确保有效 batch size 正确：
  ```bash
  GRAD_ACCUM=$((16 / (DEVICES_PER_NODE * NNODES)))
  ```
- EMA 更新在 optimizer.step() 后，decay=0.999

### 18.4 配体权重 10×

`alpha_ligand=10.0` 是当前配置中最显眼的超参数之一。小分子配体通常只有数十个原子，但在蛋白质复合物中数量远少于蛋白质原子。10× 上权重补偿了这种数量不平衡，迫使模型更精确地重建配体坐标——这对酶设计和小分子 binder 设计至关重要。

---

*本文档基于 `production` 分支源码分析，版本对应 commit `c5a9fbe`（2025）。*
