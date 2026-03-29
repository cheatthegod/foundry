# RFdiffusion3 代码导读

这份文档不是官方 README 的翻译，而是基于 `foundry/models/rfd3` 当前代码实现整理的源码级说明。目标是回答三个问题：

1. `rfd3` 到底在做什么。
2. 一次训练和一次推理，代码究竟经过哪些模块。
3. 哪些配置项和实现细节会真正改变模型行为。

## 1. 这部分代码的定位

`rfd3` 是 `foundry` 里负责 all-atom biomolecular design 的模型包。它不是只做“从随机噪声生成蛋白 backbone”的简化 diffusion，而是把下面这些任务统一到了同一个框架里：

- 无条件 de novo design
- motif scaffolding
- partial diffusion
- 固定骨架做 sequence design
- PPI binder design
- 对称体设计
- 带 hydrogen bond / RASA / hotspot / 1D secondary structure 等条件的设计

从代码结构上看，`rfd3` 的核心不是单一网络，而是三层耦合：

- 输入 DSL：把 JSON/YAML 中的人类约束转换成带注释的 `AtomArray`
- 数据流水线：把 `AtomArray` 变成模型特征、条件标签和 diffusion 噪声输入
- 模型与采样器：在 atom-level 和 token-level 表示之间来回投影，做单步去噪训练和多步 rollout 推理

对应入口文件：

- CLI: `src/rfd3/cli.py`
- 推理入口: `src/rfd3/run_inference.py`
- 推理引擎: `src/rfd3/engine.py`
- 训练入口: `src/rfd3/train.py`
- trainer: `src/rfd3/trainer/rfd3.py`
- 模型定义: `src/rfd3/model/RFD3.py`
- diffusion 主体: `src/rfd3/model/RFD3_diffusion_module.py`
- 采样器: `src/rfd3/model/inference_sampler.py`

## 2. 顶层执行链

### 2.1 推理执行链

命令行入口是：

```bash
rfd3 design ...
```

实际流程是：

1. `src/rfd3/cli.py`
   如果没有显式传 `inference_engine=...`，默认补成 `inference_engine=rfdiffusion3`。
2. `src/rfd3/run_inference.py`
   用 Hydra 组装配置，把 `inputs / n_batches / out_dir` 作为 `run()` 参数，其余部分作为 `RFD3InferenceEngine` 的初始化参数。
3. `src/rfd3/engine.py`
   `RFD3InferenceEngine.run()` 会：
   - 规范化输入
   - 对输入 specification 乘上 `n_batches`
   - 根据已有输出决定是否 `skip_existing`
   - 初始化 shared `BaseInferenceEngine`
   - 调用 `_run_multi()`
4. `src/rfd3/inference/datasets.py`
   `ContigJsonDataset` 把每个 design specification 变成 pipeline 输入，再送进 transform pipeline。
5. `src/rfd3/transforms/pipelines.py`
   构造完整 inference pipeline，把原子数组变成 `feats / t / noise / ground_truth / coord_atom_lvl_to_be_noised`。
6. `src/rfd3/trainer/rfd3.py`
   推理并没有单独写一套前向逻辑，而是复用 `AADesignTrainer.validation_step()`。
7. `src/rfd3/model/RFD3.py`
   eval 模式下不会走单步 denoise，而是调用 `ConditionalDiffusionSampler.sample_diffusion_like_af3()` 做完整 rollout。
8. `src/rfd3/engine.py`
   把 `validation_step()` 的输出重新封装成 `RFD3Output`，写出 `cif.gz`、可选 metadata json、可选 trajectory。

### 2.2 训练执行链

训练入口是：

```bash
uv run python models/rfd3/src/rfd3/train.py experiment=pretrain
```

流程是：

1. `src/rfd3/train.py`
   - 读取 Hydra 配置
   - 实例化 loggers / callbacks
   - 实例化 `AADesignTrainer`
   - 启动 Fabric 分布式环境
   - 构建模型、优化器、scheduler
   - 用 shared dataset 工具构建 train/val dataloaders
   - 调用 `trainer.fit(...)`
2. `foundry.trainers.fabric.FabricTrainer`
   提供训练主循环、EMA、梯度累计、checkpoint、validation、distributed orchestration。
3. `src/rfd3/trainer/rfd3.py`
   实现 RFD3 特有的 `training_step()` / `validation_step()` / 输出还原 / metrics 装配。

## 3. 输入 DSL：RFD3 最关键的一层

如果只看模型，很容易误判 RFD3 是一个“结构 diffusion 网络”。实际上它更像“带强约束语义的结构设计系统”。这些语义主要定义在：

- `src/rfd3/inference/input_parsing.py`
- `src/rfd3/inference/parsing.py`

### 3.1 `DesignInputSpecification`

核心类是 `DesignInputSpecification`。它把输入规范分成几类：

- 输入来源：
  - `input`
  - `atom_array_input`
  - `contig`
  - `unindex`
  - `length`
  - `ligand`
- 坐标/序列约束：
  - `select_fixed_atoms`
  - `select_unfixed_sequence`
- 条件信息：
  - `select_buried`
  - `select_partially_buried`
  - `select_exposed`
  - `select_hbond_acceptor`
  - `select_hbond_donor`
  - `select_hotspots`
  - `redesign_motif_sidechains`
- 全局条件：
  - `symmetry`
  - `ori_token`
  - `infer_ori_strategy`
  - `plddt_enhanced`
  - `is_non_loopy`
  - `partial_t`

这层最重要的不是字段本身，而是它内置了多轮验证和规范化：

- 必须提供 `input` 或 `contig/length`
- 如果给了 `input`，但没有用它参与 `contig/unindex/ligand/partial_t`，会报“input provided but unused”
- `partial_t` 开启时：
  - 禁止再给 `length`
  - 必须提供输入结构
- `contig` 和 `unindex` 不能重叠
- buried / partially_buried / exposed 三个选择必须互斥

### 3.2 `InputSelection`

`InputSelection` 是另一个关键抽象。它把这些形式统一成同一语义：

- 字符串 contig，如 `"A1-10,B2-5"`
- 布尔值 `True/False`
- 字典形式，如 `{"A1": "N,CA,C,O", "A2-5": "BKBN"}`

规范化后会得到三样东西：

- `data`: 每个 component 对应哪些原子名
- `mask`: 对整个 atom array 的布尔掩码
- `tokens`: 每个 component 对应的 `AtomArray`

因此 RFD3 的输入不是“你给一个 motif 字符串我后面再猜”，而是先变成严格、可验证、可重建的 selection object。

### 3.3 `build()` 真正做了什么

`DesignInputSpecification.build()` 会把 specification 构造成最终的推理输入结构：

1. `_build_init()`
   - 非 partial diffusion：
     - 根据 `contig` / `length` 采样设计模式
     - 从 indexed tokens 和 unindexed tokens 组装新结构
   - partial diffusion：
     - 从输入结构出发
     - 保留原有 protein token
     - 只重新定义哪些原子/序列固定
2. `_append_ligand()`
   追加 ligand，并对可选 conditioning annotation 做对齐，防止 Biotite 拼接时把 annotation 悄悄丢掉。
3. `_apply_symmetry()`
   如有 `symmetry.id`，扩展为对称结构。
4. `_set_origin()`
   - 普通 diffusion：把非 fixed atoms 坐标直接置零
   - partial diffusion：保留坐标，只做 COM/origin 处理
5. `_apply_globals()`
   写入 `is_non_loopy`、`ref_plddt`、`partial_t` 等全局条件。

最后 `to_pipeline_input()` 会调用 `prepare_pipeline_input_from_atom_array()`，把构造好的 `AtomArray` 再走一遍 AtomWorks parser，转成 pipeline 的标准输入格式。

这一步非常重要：RFD3 的推理输入并不是直接把 JSON 喂给模型，而是先还原成一个“像真实结构文件一样”的规范中间表示。

## 4. 数据流水线：真正定义训练任务分布的地方

主 pipeline 在：

- `src/rfd3/transforms/pipelines.py`

入口函数：

- `build_atom14_base_pipeline_()`
- `build_atom14_base_pipeline()`

### 4.1 pipeline 的宏观结构

一条完整 pipeline 大致分 8 段：

1. 初始化标记和分子类型
2. 训练时采样 conditioning route
3. pre-crop 结构清洗
4. crop
5. post-crop 条件计算
6. 设计相关变换
7. diffusion 封装与随机增强
8. key 子集化输出

### 4.2 训练 condition 是怎么采样的

训练时不是读一个固定标签，而是动态“生成任务”：

- `TrainingRoute(SampleConditioningType(...))`

配置主要在：

- `configs/datasets/design_base.yaml`
- `configs/datasets/conditions/*.yaml`
- `src/rfd3/transforms/training_conditions.py`

默认 `design_base.yaml` 里的任务频率是：

- `unconditional`: 5.0
- `sequence_design`: 2.0
- `island`: 1.0
- `tipatom`: 0.0
- `ppi`: 0.0

这说明默认训练更偏向：

- 大量无条件设计
- 一部分 inverse-folding / sequence design
- 一部分 motif island 任务
- 默认配置下 `tipatom` 和 `ppi` 没开

### 4.3 不同 condition 的真实含义

#### `unconditional`

- 不选 motif
- 蛋白区域基本全部扩散
- 不固定序列、不固定坐标、不 unindex

#### `island`

最典型的 motif scaffolding 条件。它会：

- 先在 protein tokens 中采样若干 island
- 再决定 motif 内是否只固定 backbone
- 或者是否只保留子图原子
- 再决定 motif sequence / motif coordinate / motif index 哪些固定

这不是一个“固定模板任务”，而是一类由概率控制的任务族。

#### `sequence_design`

它继承 `island`，但把 island 长度设得极大，相当于尽量覆盖整个骨架，更多地变成：

- 坐标大部分固定
- 序列被释放
- 更接近 inverse folding / fixed-backbone design

#### `tipatom`

更激进地做局部原子级 conditioning：

- island 长度基本是 1
- 经常只保留少量 subgraph atoms
- 用于训练模型从极少局部几何恢复完整 residue 环境

#### `ppi`

明确区分 binder 和 target：

- target 作为 motif，序列和坐标固定
- binder 区域扩散
- 不做 unindexed

### 4.4 pipeline 中的重要变换

按顺序看，最重要的几个节点是：

- `get_pre_crop_transforms(...)`
  - 去氢
  - 结构清洗
  - unresolved 处理
  - 非标准 token 原子化
- `get_crop_transform(...)`
  - contiguous crop
  - spatial crop
  - DNA-contact crop
- `CalculateRASA`
- `CalculateHbondsPlus`
- `UnindexFlaggedTokens`
- `PadTokensWithVirtualAtoms`
- `AddPPIHotspotFeature`
- `Add1DSSFeature`
- `EncodeAF3TokenLevelFeatures`
- `CreateDesignReferenceFeatures`
- `FeaturizeAtoms`
- `AddGroundTruthSequence`
- `AddSymmetryFeats`
- `get_diffusion_transforms(...)`
- `MotifCenterRandomAugmentation`
- `AugmentNoise`

其中最重要的两个设计是：

#### 4.4.1 `UnindexFlaggedTokens`

它把一部分 motif token 变成“结构保留，但不保留全局序号”的约束。这是 RFD3 相比早期 motif diffusion 更强的一点：它不仅能说“这些原子要保留”，还能说“这些原子存在，但不要求在输出中保留原始 index 对齐关系”。

#### 4.4.2 `PadTokensWithVirtualAtoms`

它把每个 token padding 到固定原子槽位。默认全局 `association_scheme` 来自：

- `configs/datasets/design_base.yaml`

默认是 `dense`。这会直接影响：

- 网络看到的 atom slot 语义
- sequence readout / cleanup 时如何把虚拟原子还原成真实 sidechain

## 5. 模型架构

模型入口：

- `src/rfd3/model/RFD3.py`

真正的网络主干：

- `src/rfd3/model/RFD3_diffusion_module.py`

配置在：

- `configs/model/components/rfd3_net.yaml`

### 5.1 `RFD3` 顶层只是一个调度壳

`RFD3` 本体很薄，只做三件事：

1. 构造 `TokenInitializer`
2. 构造 `RFD3DiffusionModule`
3. 构造 `ConditionalDiffusionSampler`

训练与推理在这里分叉：

- training:
  - 只做单步 denoise
- eval/inference:
  - 调用 sampler 做完整 rollout

如果开启 classifier-free guidance，它还会构造一个 strip 后的 `f_ref`，再跑一遍 reference forward。

### 5.2 配置给出的结构规模

`configs/model/components/rfd3_net.yaml` 里的关键维度是：

- `c_s = 384`
- `c_z = 128`
- `c_atom = 128`
- `c_atompair = 16`
- diffusion token hidden `c_token = 768`

模块深度大致是：

- `token_initializer.pairformer_blocks = 2`
- `atom_attention_encoder.n_blocks = 3`
- `diffusion_token_encoder.pairformer_blocks = 2`
- `diffusion_transformer.n_block = 18`
- `atom_attention_decoder.n_blocks = 3`

所以它不是一个很浅的“坐标 MLP + score network”，而是 atom/token 两层交互、再叠 18 层 token transformer 的中大型架构。

### 5.3 顶层表示流

`RFD3DiffusionModule.forward()` 的表示流可以概括为：

1. 输入 noisy atom coordinates `X_noisy_L`
2. 根据 `t` 和固定坐标 mask 生成 atom-level / token-level 时间特征
3. 把 atom 特征池化到 token
4. atom-level encoder 处理局部原子邻域
5. diffusion token encoder 混合 self-conditioning / distogram / token 初始化特征
6. token transformer 做全局 token 级建模
7. decoder 再把 token 信息回写到 atom
8. 输出：
   - 去噪坐标 `X_L`
   - 序列 logits
   - 序列 indices

### 5.4 原子层与 token 层是如何耦合的

关键模块有：

- `TokenInitializer`
  - 先根据参考 atom/token 特征构造初始 `Q_L_init / C_L / P_LL / S_I / Z_II`
- `LocalAtomTransformer`
  - 做 atom-level 局部建模
- `DiffusionTokenEncoder`
  - 把 atom 信息、自条件 distogram、token pair 表示汇总到 token 层
- `LocalTokenTransformer`
  - 18 层主干
- `CompactStreamingDecoder`
  - 把 token 表示解码回 atom-level 更新
- `LinearSequenceHead`
  - 在 token 层预测 residue type

这里一个关键点是：RFD3 不是“只预测 backbone 再单独接 sequence model”，而是 diffusion 过程本身就带 sequence head。

### 5.5 坐标缩放方式

`scale_positions_in()` 和 `scale_positions_out()` 实现的是 EDM 风格缩放：

- 输入缩放：`X_noisy / sqrt(t^2 + sigma_data^2)`
- 输出组合：把 noisy 坐标与网络预测更新按 EDM 公式混合

当前配置里：

- `f_pred = edm`
- `sigma_data = 16`

### 5.6 recycling

`forward_with_recycle()` 会循环调用 `process_()`：

- 训练时 recycle 数量来自 trainer 采样
- 推理时固定用模型配置里的 `n_recycle`

recycle 过程中会复用：

- 上一轮预测的 `X_L`
- 由 `X_L` bucketize 得到的自条件 distogram `D_II_self`

这意味着 RFD3 的单步网络并不独立，每个 denoise step 内部本身也有 recycle。

### 5.7 low-memory mode

如果在 engine 中设置 `low_memory_mode=True`，会注入环境变量：

```bash
RFD3_LOW_MEMORY_MODE=1
```

随后：

- `RFD3` 初始化时会启用 chunked pairwise path
- `diffusion_transformer` 的 `full=False`
- encoder / decoder 改走 chunked pairwise embedder

这不是简单减 batch size，而是切换了内部 pairwise 计算路径。

## 6. 训练细节

trainer 在：

- `src/rfd3/trainer/rfd3.py`

### 6.1 `AADesignTrainer` 的职责

它继承 shared `FabricTrainer`，但额外承担了 RFD3 的四个职责：

1. 组装网络输入
2. 训练时调用 loss
3. 验证/推理时调用 rollout
4. 把网络输出还原成原子级结构和 metadata

### 6.2 训练输入长什么样

`_assemble_network_inputs()` 会生成：

- `X_noisy_L = coord_atom_lvl_to_be_noised + noise`
- `t`
- `f`

其中一个容易忽略的实现细节是：

- 训练时如果 `X_noisy_L` 有 NaN，会用 `nan_to_num` 替换成 0
- 推理/验证时则直接报错

也就是说，训练端允许某些 crop/resolution 边缘情况以“从原点开始扩散”的方式继续跑，而推理端要求输入更严格。

### 6.3 recycle 调度

训练时 recycle 数量不是写死的，而是通过：

- `rfd3.trainer.recycling.get_recycle_schedule(...)`

预先为每个 epoch、每个 batch 采样好。这保证了多卡训练时同一个 distributed batch 使用一致 recycle 数。

### 6.4 训练配置默认值

`configs/trainer/rfd3_base.yaml` 的默认训练行为大致是：

- `n_examples_per_epoch: 2400`
- `checkpoint_every_n_epochs: 10`
- `validate_every_n_epochs: 4`
- `max_epochs: 100000`
- `n_recycles_train: 2`
- `grad_accum_steps: 3`
- `precision: bf16-mixed`
- `clip_grad_max_norm: 10.0`
- `prevalidate: False`

这里有两个容易误读的点：

- `cleanup_virtual_atoms: False`
  训练默认不清理虚拟原子，便于保持固定 tokenized 表示
- `output_full_json: False`
  训练侧默认不把整套 specification 全量塞进输出

## 7. Loss 与训练目标

loss 定义在：

- `src/rfd3/metrics/losses.py`

### 7.1 `SequenceLoss`

它做的不是简单 CE，而是带时间窗口的序列损失：

- 只在 `min_t <= t < max_t` 的样本上启用
- 对 token 级 logits 做 cross-entropy
- 统计 sequence recovery
- 特别记录最低噪声样本的 `lowest_t_seq_recovery`
- 最终对平均 token loss 做 `clamp(max=4)`

这说明 sequence head 并不是在所有噪声级别下都一视同仁地监督。

### 7.2 `DiffusionLoss`

坐标损失包含多种重权重策略：

- `alpha_unindexed_diffused`
- `alpha_virtual_atom`
- `alpha_polar_residues`
- `alpha_ligand`
- `unindexed_t_alpha`
- `lp_weight`
- `lddt_weight`

它的基本形式是：

- 先做加权坐标 MSE
- 再乘 EDM 风格的 `lambda(sigma)`
- 如果存在 unindexed token，则对 unindexed 区域做额外时间重标定
- 可选增加 unindexed 区域的 LP norm 正则
- 可选加 smoothed lDDT loss

几个重要结论：

- unindexed 区域不是普通 motif，它在 loss 里被特别强调
- virtual atoms 不是纯 padding，它们也参与训练权重设计
- 模型训练目标不是纯坐标 MSE，而是坐标、局部几何稳定性和序列共同优化

## 8. 推理采样细节

采样器定义在：

- `src/rfd3/model/inference_sampler.py`

### 8.1 `SampleDiffusionConfig`

推理配置包括三类参数：

1. 标准 EDM 参数
   - `num_timesteps`
   - `sigma_data`
   - `s_min`
   - `s_max`
   - `p`
   - `gamma_0`
   - `gamma_min`
   - `noise_scale`
   - `step_scale`
2. 设计相关参数
   - `center_option`
   - `s_trans`
   - `s_jitter_origin`
   - `fraction_of_steps_to_fix_motif`
   - `allow_realignment`
   - `insert_motif_at_end`
3. CFG 参数
   - `use_classifier_free_guidance`
   - `cfg_scale`
   - `cfg_t_max`

### 8.2 noise schedule

`_construct_inference_noise_schedule()` 使用 AF3 supplement 风格的 schedule：

```text
t_hat = sigma_data * (s_max^(1/p) + t * (s_min^(1/p) - s_max^(1/p)))^p
```

若给了 `partial_t`：

- 只保留 `t <= partial_t` 的 schedule
- 如果过滤后为空，会 fallback 到最后一步

这意味着 partial diffusion 不是“在完整 schedule 上加一个 mask”，而是直接截断 rollout 的起始噪声范围。

### 8.3 motif 处理

`SampleDiffusionWithMotif` 的 motif 逻辑包括：

- 初始加噪时 fixed motif 原子噪声置零
- 可选地每步围绕 motif 做重新中心化 / 刚体增强
- rollout 末尾可重新插回 motif
- 最后再对输出整体按 motif 做 rigid alignment

所以 motif 并不是单纯通过输入 mask 静态固定，而是贯穿整个采样过程。

### 8.4 classifier-free guidance

如果开启 CFG：

1. 先对 `f` 做 strip，去掉指定条件特征
2. 跑一遍 unconditional/reference denoise
3. 用：

```text
delta = delta + (cfg_scale - 1) * (delta - delta_ref)
```

做 guidance

默认 inference 配置里会 strip 的特征包括：

- `active_donor`
- `active_acceptor`
- `ref_atomwise_rasa`

### 8.5 对称采样

`SampleDiffusionWithSymmetry` 继承 motif sampler，并在 rollout 中的前一部分步骤对坐标施加 symmetry transform。它要求：

- `gamma_0 > 0.5`

而且 symmetry 不是对 noisy structure 施加，而是在 denoise/update 链条里保持对称约束。

## 9. 输出还原与后处理

这是 RFD3 非常有价值的一层，集中在：

- `src/rfd3/trainer/rfd3.py`
- `src/rfd3/trainer/trainer_utils.py`

### 9.1 从网络输出到结构

`_build_predicted_atom_array_stack()` 会做这些事：

1. 从源 `atom_array` 复制结构骨架
2. 把非固定序列区域暂时标成 `UNK`
3. 调用 `_build_atom_array_stack()` 把坐标写回
4. 根据 sequence head 或结构几何读出 residue type
5. 处理 unindexed motifs
6. 清理虚拟原子
7. 计算 backbone / hbond / partial diffusion 等指标
8. 组织 metadata json

### 9.2 sequence 是怎么读出来的

有两条路径：

#### 路径 A：读 sequence head

如果 `read_sequence_from_sequence_head=True` 且有 `sequence_logits_I`：

- 直接 decode 预测 token 序列
- 把 diffused 区域的 `res_name` 改成预测结果
- 同时把 sequence entropy 写进 `b_factor`

#### 路径 B：从结构反推序列

如果不读 sequence head，则：

- `_readout_seq_from_struc()` 根据虚拟原子占位模式匹配 association scheme
- 推断 residue type

这说明虚拟原子不仅是 padding，也是 sequence geometry 的编码容器。

### 9.3 unindexed 输出怎么恢复

`process_unindexed_outputs()` 会把 unindexed motif 与 diffused 区域重新做匹配，输出：

- `diffused_index_map`
- `insertion_rmsd`
- `insertion_rmsd_by_residue`

这正是 RFD3 能支持“结构存在但 index 不固定”的关键后处理能力。

### 9.4 cleanup virtual atoms

`_cleanup_virtual_atoms_and_assign_atom_name_elements()` 会：

- 根据预测 residue type 查 `association_scheme`
- 删除无效虚拟原子位
- 把真实 atom name 和 element 重新赋值
- 如果预测到非蛋白或不认识的 residue，则保留并标成 `UNK`

这也是为什么 `association_scheme` 不是小配置项，而是贯穿训练、输出和可视化的主轴配置。

## 10. 推理引擎配置

默认推理配置：

- `configs/inference_engine/rfdiffusion3.yaml`

一些高影响配置：

- `skip_existing: True`
  已有输出会直接跳过
- `diffusion_batch_size: 8`
  每个 specification 同时生成多少个样本
- `n_batches: 1`
  对同一个 specification 重复多少次
- `cleanup_guideposts: True`
- `cleanup_virtual_atoms: True`
- `read_sequence_from_sequence_head: True`
- `output_full_json: True`
- `dump_trajectories: False`
- `prevalidate_inputs: False`
- `low_memory_mode: False`

`skip_existing` 的行为容易忽略：它不是检查单个 model 文件，而是先把已有输出路径还原到 example id，再整组跳过。

## 11. 配置体系怎么看

如果你要真正读懂 RFD3，建议按这个顺序看配置：

1. `configs/inference_engine/rfdiffusion3.yaml`
   看推理默认行为
2. `configs/datasets/design_base.yaml`
   看训练任务分布和 transform 全局参数
3. `configs/datasets/conditions/*.yaml`
   看每种 condition 的具体采样规则
4. `configs/model/components/rfd3_net.yaml`
   看网络深度与维度
5. `configs/trainer/rfd3_base.yaml`
   看训练策略
6. `configs/experiment/*.yaml`
   看具体实验如何覆盖默认值

其中最关键的是：

- `design_base.yaml`
- `rfd3_net.yaml`
- `rfdiffusion3.yaml`

## 12. 实现上的几个关键判断

### 12.1 RFD3 的本质不是“一个 diffusion 网络”

它更准确地说是：

“一个把 design specification 编译成结构条件任务，再用 atom-token diffusion 网络求解的系统”

原因是模型性能和行为很大程度上取决于：

- input DSL 的表达方式
- training condition 的任务分布
- unindex 与 virtual atom 的结构编码

而不只取决于 backbone 网络本身。

### 12.2 trainer 和 inference engine 的边界设计很聪明

推理复用 `validation_step()`，带来的好处是：

- 训练验证和在线推理用同一套结构还原逻辑
- metrics / postprocess / unindexed handling 不会写两遍

代价是：

- trainer 职责明显偏重
- `validation_step()` 已经不只是 validation，而是“结构生成与后处理入口”

### 12.3 virtual atom 是整个系统的关键中介

如果只把 virtual atom 当 padding，会看不懂 RFD3。它实际承担了三层职责：

- 固定 token slot，使 atom-level network 输入维度稳定
- 编码 sidechain / atomization 模式
- 支持从结构反推 residue type 和最终 cleanup

### 12.4 unindexed motif 是 RFD3 与传统 motif scaffolding 的一个大分水岭

传统做法更像“给定固定 motif 和固定索引的 scaffolding”。RFD3 增加 unindexed 后，任务表达能力强得多：

- 可以固定几何而不固定全局位置编号
- 更适合复杂接口、局部化学基元、非连续 motif

## 13. 阅读源码的推荐顺序

如果你后面准备继续深读，建议按这个顺序：

1. `src/rfd3/inference/input_parsing.py`
2. `src/rfd3/inference/parsing.py`
3. `src/rfd3/transforms/pipelines.py`
4. `src/rfd3/transforms/training_conditions.py`
5. `src/rfd3/model/RFD3.py`
6. `src/rfd3/model/RFD3_diffusion_module.py`
7. `src/rfd3/model/inference_sampler.py`
8. `src/rfd3/trainer/rfd3.py`
9. `src/rfd3/trainer/trainer_utils.py`
10. `src/rfd3/metrics/losses.py`

这样读的顺序是：

- 先理解“输入想表达什么”
- 再理解“数据怎么变成网络输入”
- 最后再看“网络怎么解这个问题”

## 14. 测试覆盖说明

`models/rfd3/tests` 里已经有一批很有针对性的测试，重点覆盖：

- conditioning
- tokenization
- unindexing
- symmetry
- partial diffusion
- metrics
- legacy pipeline equivalence

这些测试名称本身就能反映出这个项目最脆弱也最核心的语义边界。

## 15. 一句话总结

RFdiffusion3 在这份代码里的真实形态，不是“一个从噪声到结构的模型”，而是“一个能够把复杂生物分子设计约束编译成 atom-level / token-level diffusion 问题，并在训练和推理时统一处理 motif、sequence、symmetry、unindex、局部化学条件的设计系统”。

如果后面要继续深入，最值得继续拆的两块是：

- `TokenInitializer` / `DiffusionTokenEncoder` 具体怎样编码 pairwise/self-conditioning
- `conditioning_utils` 与 `design_transforms` 怎样细化 island/subgraph/unindex 的采样语义
