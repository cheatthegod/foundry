# RFdiffusion3 最小 Demo 逐步拆解

这份文档只讲一个非常具体的东西：

- 输入文件：`models/rfd3/docs/examples/minimal_unconditional_cn.json`
- 运行脚本：`models/rfd3/docs/examples/run_minimal_inference_cpu.sh`
- 输出目录：`models/rfd3/docs/examples/minimal_demo_outputs/`

目标不是泛泛介绍 RFD3，而是把这个最小 demo 从命令行一直拆到最终输出文件，解释每一个关键对象、每一层张量、以及为什么它会长成这样。

---

## 1. 先说结论：这个 demo 在做什么

这个 demo 的输入是：

```json
{
  "toy_unconditional_32aa": {
    "length": 32,
    "extra": {
      "note": "Minimal unconditional RFdiffusion3 demo"
    }
  }
}
```

它的意思非常朴素：

- 不提供输入 PDB/CIF
- 不提供 contig
- 不提供 ligand
- 不提供 motif
- 也不做 partial diffusion
- 只要求模型“从零生成一个 32 residue 的蛋白”

所以这是 RFD3 里最简单的一类任务：

- unconditional protein design

这个 demo 的真正价值，不在于输出质量，而在于把 RFD3 的机械结构暴露出来：

1. 输入 specification 如何被编译成 `AtomArray`
2. `AtomArray` 如何被变成固定大小的 atom/token 张量
3. diffusion rollout 如何更新坐标
4. 预测出来的 padded atom slots 如何被清理成真实结构

---

## 2. 我实际跑的是哪条命令

脚本在：

- `models/rfd3/docs/examples/run_minimal_inference_cpu.sh`

核心命令是：

```bash
"$PYTHON_BIN" "$FOUNDRY_ROOT/models/rfd3/src/rfd3/run_inference.py" \
  ckpt_path="$CKPT_PATH" \
  out_dir="$OUT_DIR" \
  inputs="$INPUTS" \
  json_keys_subset=toy_unconditional_32aa \
  prevalidate_inputs=True \
  n_batches=1 \
  diffusion_batch_size=1 \
  inference_sampler.num_timesteps=4 \
  seed=0 \
  skip_existing=False \
  cleanup_virtual_atoms=True \
  dump_prediction_metadata_json=True
```

注意这里有几个非常关键的 override：

### 2.1 `json_keys_subset=toy_unconditional_32aa`

输入 JSON 里只跑这一条 example。

### 2.2 `diffusion_batch_size=1`

同一个 design specification 只生成 1 个 sample。

RFD3 里 `diffusion_batch_size` 不是 DataLoader batch size 的意思，而是：

- 同一个结构问题并行采样多少个扩散副本

如果这里设成 8，模型会对同一个 `length=32` 问题同时跑 8 条不同噪声轨迹。

### 2.3 `n_batches=1`

不重复这个 example。

### 2.4 `inference_sampler.num_timesteps=4`

这是为了让 CPU 上能快速跑通。

默认推理通常是 200 步左右，这里压成 4 步只是为了演示流程。

重要后果是：

- rollout 只有 3 次更新
- 输出质量不代表正常 RFD3 推理质量

### 2.5 `seed=0`

固定随机种子，方便复现。

### 2.6 `cleanup_virtual_atoms=True`

最后要把内部 padding 用的虚拟原子删掉，输出真实结构。

---

## 3. 命令行一启动，第一层发生了什么

执行路径是：

1. `run_inference.py`
2. `RFD3InferenceEngine`
3. `BaseInferenceEngine.initialize()`
4. pipeline 构建
5. trainer 构建
6. model 权重加载
7. 输入数据规范化
8. loader 构造
9. `validation_step()` 推理
10. 结构落盘

这条路径非常重要，因为 RFD3 推理不是自己单写了一套 forward，而是复用了 trainer 的 validation 逻辑。

换句话说：

- inference 和 validation 基本是同一套生成/还原/落盘流程

---

## 4. 输入 JSON 先变成了什么

### 4.1 `process_input()`

`engine.py` 里的 `process_input()` 会先做三件事：

1. 读 JSON
2. 取出键 `toy_unconditional_32aa`
3. 把这条记录变成一个 design specification

我实际打印出来的是：

```python
{
  "length": 32,
  "extra": {
    "note": "Minimal unconditional RFdiffusion3 demo",
    "example": "toy_unconditional_32aa"
  }
}
```

此时它还只是“设计要求”，还不是结构。

### 4.2 `DesignInputSpecification.safe_init(...)`

这个对象会验证：

- 至少要有 `input` 或 `contig/length`
- 这里虽然没有 `input`，但有 `length=32`
- 所以它是合法的 unconditional design

这个 demo 没有出现下面这些复杂逻辑：

- motif selection
- `select_fixed_atoms`
- `select_unfixed_sequence`
- `unindex`
- `partial_t`
- symmetry
- ligand append

也正因如此，这个 demo 能帮你先只看主干。

---

## 5. `DesignInputSpecification.build()` 到底构造了什么

这是整个 demo 第一个最关键的步骤。

因为没有 `input` 文件，也没有 `contig`，它会走：

- “纯 length unconditional accumulation” 这条路径

### 5.1 结果是什么

我实际打印出来的是：

- `num_tokens_in = 32`
- `num_residues_in = 32`
- `num_chains = 1`
- `num_atoms = 160`
- `sampled_contig = "32"`

也就是说，RFD3 先造了一个：

- 1 条链
- 32 个 residue
- 160 个原子的中间结构

### 5.2 为什么是 160 个原子

因为这时候不是最终 all-atom 结构，而是一个“最小设计骨架表示”。

对于这个 unconditional demo，可以把它近似理解成：

- 每个 residue 先只保留 backbone + 一个中心/侧链锚点

所以：

- `32 residue * 5 atom-like slots = 160`

这一步的本质是：

- 先把“长度 32 的蛋白设计任务”变成一个结构模板
- 这个模板不是最终输出，而是供 pipeline 继续加工的中间表示

### 5.3 这时的注释长什么样

我实际看到的 annotation 包括：

- `chain_id`
- `res_id`
- `res_name`
- `src_component`
- `is_motif_atom_with_fixed_coord`
- `is_motif_atom_with_fixed_seq`
- `is_motif_atom_unindexed`
- `is_motif_atom_unindexed_motif_breakpoint`
- `is_non_loopy`
- `ref_plddt`

这里因为是 unconditional：

- `fixed_coord` 基本都是 False
- `fixed_seq` 基本都是 False
- `unindexed` 基本都是 False

所以这个 demo 的问题定义非常纯：

- “所有位置都要由模型自己生成”

---

## 6. `prepare_pipeline_input_from_atom_array()` 做了什么

这一步看起来像是多余的，其实非常重要。

它做的是：

- 把刚刚构出来的 `AtomArray` 再喂进 AtomWorks parser
- 重新转成 pipeline 标准输入格式

我实际打印出来的 `pipeline_input` 键有：

- `atom_array`
- `chain_info`
- `ligand_info`
- `metadata`
- `example_id`
- `specification`

这里的逻辑意义是：

- RFD3 不区分“来自真实 PDB 的结构”和“由 specification 合成出来的结构”
- 两者最终都要统一成同一种 parser 输出格式

这是它工程上很漂亮的地方。

---

## 7. pipeline 真正把结构变成了什么

这一步在：

- `rfd3.transforms.pipelines.build_atom14_base_pipeline()`

### 7.1 先说最终结果

我实际拿到的 pipeline 输出键是：

- `atom_array`
- `coord_atom_lvl_to_be_noised`
- `example_id`
- `feats`
- `ground_truth`
- `noise`
- `sampled_condition_name`
- `specification`
- `t`

其中最核心的是：

- `t`
- `noise`
- `coord_atom_lvl_to_be_noised`
- `feats`

### 7.2 真实张量形状

这次 demo 我实际打印到的是：

- `t`: `(1,)`
- `noise`: `(1, 448, 3)`
- `coord_atom_lvl_to_be_noised`: `(1, 448, 3)`

这里突然从 160 原子变成了 448。

### 7.3 为什么会变成 448

因为 pipeline 中有一个关键变换：

- `PadTokensWithVirtualAtoms`

RFD3 内部把每个 token pad 到 14 个 atom slots。

这次 demo 是 32 个 residue，所以：

- `32 * 14 = 448`

因此：

- `L = 448` 是网络看到的 atom-level 长度
- 它不是“真实原子数”，而是“padding 后的固定槽位数”

### 7.4 为什么必须这样做

因为不同氨基酸、不同 ligand 的真实原子数不一样。

神经网络更喜欢固定形状。

所以 RFD3 的策略是：

1. 先给每个 token 固定 14 个槽位
2. 不够的用 virtual atom 补齐
3. 输出时再把虚拟原子删掉

这就是 RFD3 最重要的工程抽象之一。

---

## 8. `feats` 里到底有什么

`feats` 是模型静态条件的总包。

我实际看到的部分键有：

- `atom_to_token_map`
- `ref_pos`
- `ref_element`
- `is_motif_atom_with_fixed_coord`
- `is_motif_atom_with_fixed_seq`
- `is_virtual`
- `is_backbone`
- `is_ligand`
- `restype`
- `token_bonds`
- `motif_pos`
- `asym_id`
- `ref_atom_name_chars`
- `ref_motif_token_type`

### 8.1 这里最关键的几个字段

#### `atom_to_token_map`

形状：

- `(448,)`

它表示：

- 第几个 atom slot 属于第几个 token

这是 atom-level 和 token-level 两条表示流之间最关键的桥。

#### `ref_pos`

形状：

- `(448, 3)`

它是参考坐标。

对于这个 unconditional demo，它大部分是由 pipeline 构造出来的参考几何，而不是来自真实输入结构。

#### `is_virtual`

形状：

- `(448,)`

表示哪些 atom slot 是 padding 用的虚拟原子。

#### `is_motif_atom_with_fixed_coord`

形状：

- `(448,)`

这个 demo 几乎全是 False，因为没有 motif。

#### `restype`

它是 token-level 的 residue 类型特征，不是最终预测序列，而是当前输入 token 的类型编码。

---

## 9. `t`、`noise`、`coord_atom_lvl_to_be_noised` 各自是什么意思

这是 diffusion 里最容易混淆的地方。

### 9.1 `coord_atom_lvl_to_be_noised`

可以理解成：

- 扩散前的基准坐标

对于 unconditional demo，它代表的是“初始模板坐标”。

### 9.2 `noise`

高斯噪声。

### 9.3 `t`

噪声强度。

这次我真实拿到的值是：

```text
t = 0.3410351574420929
```

### 9.4 三者如何组成网络输入

trainer 里实际做的是：

```python
X_noisy_L = coord_atom_lvl_to_be_noised + noise
```

这次真实得到：

- `X_noisy_L`: `(1, 448, 3)`

所以你可以把网络输入理解成：

- 一份固定的条件特征 `f`
- 一份当前噪声结构 `X_noisy_L`
- 一份当前时间/噪声尺度 `t`

---

## 10. `TokenInitializer` 先产出了哪些张量

这一步发生在模型最开始，还没真正进入 diffusion 主干。

它的作用是：

- 把静态条件 `f` 编成 atom 和 token 的初始表示

我实际打印出的形状是：

- `Q_L_init`: `(448, 128)`
- `C_L`: `(448, 128)`
- `P_LL`: `(448, 448, 16)`
- `S_I`: `(32, 384)`
- `Z_II`: `(32, 32, 128)`

### 10.1 这些张量分别是什么意思

#### `Q_L_init`

atom-level 初始 single representation。

#### `C_L`

atom-level conditioning representation。

它把 token-level 信息也混回到 atom 层。

#### `P_LL`

atom-pair representation。

形状是：

- `448 x 448 x 16`

这说明 RFD3 在 atom 层显式维护 pair 特征。

#### `S_I`

token single representation。

这次 `I=32`，所以是：

- `32 x 384`

#### `Z_II`

token-pair representation。

这次是：

- `32 x 32 x 128`

### 10.2 这里一个特别重要的理解

`TokenInitializer` 产出的这些对象，在整个 rollout 中基本属于“静态底图”：

- 它们来自条件 `f`
- 不直接来自当前 `X_noisy_L`

真正随着 diffusion step 不断变化的，是：

- `X_L`
- `X_noisy_L`
- `attn_indices`
- self-conditioning distogram
- time embeddings

所以 RFD3 的计算图可以分成两部分：

1. 静态条件编码
2. 动态去噪迭代

---

## 11. attention indices 是怎么构造的

这是你当前最关心的那一段，我单独拆开。

代码在：

- `create_attention_indices()`
- `build_index_mask()`
- `extend_index_mask_with_neighbours()`

### 11.1 真实输出形状

这次 demo 实际得到：

- `attn_indices`: `(1, 448, 128)`

也就是说：

- 对每个 atom query
- 只保留 128 个 attention key

不是做完整 `448 x 448` 的 dense attention。

### 11.2 参数来自哪里

模型配置里：

- `n_attn_keys = 128`
- `n_attn_seq_neighbours = 2`

### 11.3 它不是单纯的 kNN

这一点非常关键。

构造逻辑其实是两段式：

#### 第一段：先强制保留序列局部邻域

`build_index_mask()` 先按 token 邻域构造一个硬 mask：

- token 差值 `<= 2` 的邻居 token 必须保留

注意这里是 token 级，不是 atom 级。

对于这个 demo：

- 每个 residue 有 14 个 atom slots
- 对于序列中间的 residue，会强制保留自身和左右各 2 个 token
- 也就是大约 `5 * 14 = 70` 个 atom slots

对于 N 端第一个 residue，只能看到 token `0,1,2`

- 所以强制邻域大约是 `3 * 14 = 42`

#### 第二段：剩余位置再用空间 kNN 填满

`extend_index_mask_with_neighbours()` 再用当前 `X_noisy_L` 的欧氏距离做最近邻补全，直到凑满 128 个 key。

所以：

- 先有一批“必须看的序列局部邻居”
- 再补一批“当前几何上最近的邻居”

### 11.4 这次 demo 的第一个 query 长什么样

我实际打印的前 20 个索引是：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
```

而且：

- `len(set(attn_indices[0,0])) = 128`

说明：

- 没有重复索引
- 第一行已经被填满到 128 个邻居

为什么前面看起来像连续编号？

因为对第一个 residue 来说，序列局部强制邻域天然就覆盖了前几个 token 的所有 atom slots，所以排完序后最前面会呈现一串连续小编号。

### 11.5 这件事最重要的意义

attention 图不是固定的。

它每一步都会根据当前 `X_noisy_L` 重新建图。

所以 RFD3 的局部原子图是：

- dynamic graph

而不是固定图。

---

## 12. 进入 diffusion module 后发生了什么

这一步在 `RFD3DiffusionModule.forward()`。

可以粗分为 6 步：

1. 根据当前 `X_noisy_L` 构造 `attn_indices`
2. 根据 `t` 生成 atom-level 和 token-level 时间嵌入
3. 先做 atom-level 编码
4. 再做 token-level 编码和 transformer
5. 再从 token 解码回 atom
6. 输出坐标和序列

### 12.1 为什么先 atom 再 token

因为 RFD3 想同时保留两种信息：

- atom 层的局部几何细节
- residue/token 层的全局折叠和序列语义

所以它不是单轨模型，而是双轨：

- atom track
- token track

### 12.2 sequence head 是什么时候出的

不是最后另接一个单独模型，而是在 diffusion module 里就直接读出：

- `sequence_logits_I`
- `sequence_indices_I`

这次真实输出是：

- `sequence_logits_I`: `(1, 32, 32)`
- `sequence_indices_I`: `(1, 32)`

所以每个 token 都会得到一个 32 类 residue 分布。

---

## 13. sampler 外层到底做了几次更新

这次我把：

- `num_timesteps = 4`

所以 noise schedule 有 4 个点。

但 rollout 代码是：

- `zip(noise_schedule, noise_schedule[1:])`

也就是说：

- 4 个时间点只会产生 3 次更新

这和我实际看到的输出完全一致：

- `X_noisy_L_traj`: 3 帧
- `X_denoised_L_traj`: 3 帧
- `t_hats`: 3 帧

所以这个 demo 的 diffusion 不是“4 次更新”，而是：

- 3 次 update step

---

## 14. `validation_step()` 为什么能当推理用

RFD3 的设计比较特殊。

推理并没有单独写一套“从网络输出变回结构”的逻辑，而是直接复用：

- `AADesignTrainer.validation_step()`

它做了四件事：

1. 组装 `network_input`
2. 调用模型 forward rollout
3. 把输出坐标和序列还原回 `AtomArray`
4. 生成 metadata

这个设计的好处是：

- validation 和 inference 用同一套后处理
- 不会出现“验证逻辑一套、在线推理逻辑另一套”的分叉

---

## 15. 为什么最终输出不是 448 个原子，而是 295 个

这是整个 demo 最关键的“反编译”步骤。

我这次实际看到：

- 网络内部：448 atom slots
- 最终结构：295 原子

### 15.1 原因

448 里有很多是 virtual atom。

最后 `_build_predicted_atom_array_stack()` 会：

1. 根据 `sequence_indices_I` 给每个 token 赋 residue type
2. 根据 `association_scheme` 判断：
   - 这个 residue 哪些 atom slot 对应真实原子
   - 哪些 slot 只是 padding
3. 删除虚拟原子
4. 重写 atom names 和 elements

### 15.2 所以最终发生的是

- token 数不变，仍然是 32
- 但每个 residue 的真实 sidechain 原子数不同
- 所以最后真实原子数从统一的 `32 * 14` 降回不规则的真实值

这也是为什么 RFD3 能同时兼顾：

- 神经网络喜欢的固定形状
- 结构文件需要的不规则原子数

---

## 16. 最终输出文件里到底有什么

这次实际生成了两个文件：

### 16.1 CIF 文件

- `minimal_unconditional_cn_toy_unconditional_32aa_0_model_0.cif.gz`

它是最终结构。

### 16.2 JSON 文件

- `minimal_unconditional_cn_toy_unconditional_32aa_0_model_0.json`

我实际看到的顶层键是：

- `diffused_index_map`
- `metrics`
- `specification`
- `ckpt_path`
- `seed`

### 16.3 `specification` 里有什么

这次包含：

- `length: "32"`
- `extra.note`
- `extra.sampled_contig = "32"`
- `extra.num_tokens_in = 32`
- `extra.num_atoms = 160`
- `extra.example_id = ...`

### 16.4 `metrics` 里有什么

这次包含：

- `max_ca_deviation`
- `n_chainbreaks`
- `n_clashing.interresidue_clashes_w_sidechain`
- `n_clashing.interresidue_clashes_w_backbone`
- `non_loop_fraction`
- `loop_fraction`
- `helix_fraction`
- `sheet_fraction`
- `num_ss_elements`
- `radius_of_gyration`
- `alanine_content`
- `glycine_content`
- `num_residues`
- `diffused_com`

所以输出 JSON 不是简单日志，而是：

- 输入 specification 的回填
- 推理参数记录
- 结构质量指标

---

## 17. 这次 demo 里哪些东西是“静态”的，哪些是“动态”的

这是理解 RFD3 的最高效视角。

### 17.1 静态对象

这些在 rollout 开始后基本不变：

- `specification`
- `atom_array` 的基本 token/chain 组织
- `feats`
- `atom_to_token_map`
- `Q_L_init`
- `C_L`
- `P_LL`
- `S_I`
- `Z_II`

### 17.2 动态对象

这些每一步都会变化：

- `t`
- `X_L`
- `X_noisy_L`
- `attn_indices`
- `sequence_logits_I`
- self-conditioning distogram

这就是为什么你会觉得 RFD3 很复杂：

- 它不是“一个前向”
- 而是“静态条件底图 + 多步动态图更新”

---

## 18. 这次 demo 为什么会有一些 warning

运行里你会看到一些 warning，含义如下：

### 18.1 `No GPUs/XPUs available`

说明这次是 CPU 跑的。

### 18.2 `Cached data not found for ALA`

说明某些 residue cache 没找到。

对这个最小 unconditional demo 不构成致命问题。

### 18.3 `extra_fields argument will be ignored if there is no CIF file input`

因为这个 demo 没有真实输入 PDB/CIF，是 specification 合成出来的结构。

### 18.4 有时会出现 sequence cleanup warning

因为这次只跑了 4 个 diffusion steps，输出质量故意不是生产设置。

---

## 19. 这个 demo 真正教会你的是什么

如果你只记住一件事，请记这个：

RFD3 的推理，不是“JSON -> 网络 -> PDB”。

而是：

1. `JSON specification`
2. `DesignInputSpecification`
3. `AtomArray` 中间结构
4. `pipeline` 把结构编译成固定 `atom14` 槽位张量
5. `TokenInitializer` 构造静态 atom/token 表示
6. `diffusion sampler` 多步更新 `X_L`
7. `trainer` 把 padded slots 反编译回真实结构
8. `CIF + JSON`

也就是说，RFD3 真正复杂的地方，不只是网络，而是：

- 问题定义
- 结构编译
- 固定槽位表示
- 反编译回真实化学结构

---

## 20. 如果你接下来还想继续深挖，最值得看的 3 个问题

### 20.1 为什么 `P_LL` 要做到 `(448, 448, 16)`

这决定了 atom-pair 几何信息怎样进入模型。

### 20.2 为什么 `attn_indices` 每步都要重建

这决定了 atom graph 是如何随当前噪声结构动态变化的。

### 20.3 为什么 sequence head 和坐标 diffusion 放在同一个模型里

这决定了 RFD3 为什么不像“先生 backbone，再单独跑 MPNN”的两阶段系统。

---

## 21. 一句话总结这个 demo

这个最小 unconditional demo 展示了 RFD3 的完整主干：

- 从一个只有 `length=32` 的 specification 出发
- 先构造 160 原子的中间骨架
- 再 pad 成 448 个 atom slots 供网络处理
- 再通过 3 次 diffusion 更新得到 `X_L`
- 再删掉虚拟原子，最终输出一个 295 原子的真实结构文件

它不是最有代表性的生物设计任务，但它是最适合看清 RFD3 机械结构的 demo。
