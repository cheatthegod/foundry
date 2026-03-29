"""
RFdiffusion3 完整推理追踪脚本
逐步打印每个阶段的输入和输出形状/值
"""

import sys
import os
import time
import json

# 设置 PYTHONPATH
sys.path.insert(0, "/home/ubuntu/cqr_files/protein_design/foundry/src")
sys.path.insert(0, "/home/ubuntu/cqr_files/protein_design/foundry/models/rfd3/src")
sys.path.insert(0, "/home/ubuntu/cqr_files/protein_design/foundry/models/mpnn/src")
sys.path.insert(0, "/home/ubuntu/cqr_files/protein_design/foundry/models/rf3/src")
os.chdir("/home/ubuntu/cqr_files/protein_design/foundry/models/rfd3/docs/examples")

import torch
import numpy as np
from dotenv import load_dotenv

load_dotenv("/home/ubuntu/cqr_files/protein_design/foundry/.env", override=True)

# ─── 工具函数 ────────────────────────────────────────────────────────────────

def sep(title="", width=72):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'='*pad} {title} {'='*(width-pad-len(title)-2)}")
    else:
        print("=" * width)

def shape_repr(x):
    if isinstance(x, torch.Tensor):
        return f"Tensor{list(x.shape)} dtype={x.dtype} device={x.device}"
    elif isinstance(x, np.ndarray):
        return f"ndarray{list(x.shape)} dtype={x.dtype}"
    elif isinstance(x, dict):
        return f"dict(keys={list(x.keys())[:6]}{'...' if len(x)>6 else ''})"
    elif isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    else:
        return repr(x)

def print_dict_shapes(d, prefix="  ", max_entries=30):
    count = 0
    for k, v in d.items():
        if count >= max_entries:
            print(f"{prefix}... ({len(d) - max_entries} more keys)")
            break
        print(f"{prefix}{k}: {shape_repr(v)}")
        count += 1

# ─── STEP 0: 加载输入 JSON ───────────────────────────────────────────────────

sep("STEP 0: 读取 demo.json 输入规格")

with open("demo.json") as f:
    demo = json.load(f)

print("\n[demo.json 内容]")
print(json.dumps(demo, indent=2))

# 只跑第一个样例 M0255_1mg5_unfixed
EXAMPLE_KEY = "M0255_1mg5_unfixed"
example_spec = demo[EXAMPLE_KEY]
print(f"\n[选取样例] key='{EXAMPLE_KEY}'")
print(json.dumps(example_spec, indent=2))

# ─── STEP 1: 解析 DesignInputSpecification ──────────────────────────────────

sep("STEP 1: DesignInputSpecification 解析与验证")

from rfd3.inference.input_parsing import DesignInputSpecification
from rfd3.engine import process_input

specs = process_input(
    inputs="demo.json",
    json_keys_subset=[EXAMPLE_KEY],
    validate=True,
)
spec_dict = specs[f"demo_{EXAMPLE_KEY}"]

print(f"\n[process_input 输出] 共 {len(specs)} 个规格")
print(f"example id prefix: demo_{EXAMPLE_KEY}")
print(f"\n[spec_dict 内容]")
print(json.dumps({k: str(v) for k, v in spec_dict.items()}, indent=2))

# ─── STEP 2: 构建 AtomArray（创建设计基底结构）────────────────────────────────

sep("STEP 2: 从 spec 构建 AtomArray")

from rfd3.inference.input_parsing import prepare_pipeline_input_from_atom_array
import inspect

dis = DesignInputSpecification(**spec_dict)
print(f"\n[DesignInputSpecification 字段]")
for field_name, field_val in dis.model_dump().items():
    if field_val is not None and field_val != {} and field_val != []:
        print(f"  {field_name}: {field_val}")

atom_array, spec_dumped = dis.build(return_metadata=True)
print(f"\n[构建后 AtomArray]")
print(f"  总原子数 (n_atoms): {len(atom_array)}")
print(f"  注解列 (annotations): {atom_array.get_annotation_categories()}")

# 分析条件注解
fixed_coord = atom_array.is_motif_atom_with_fixed_coord
fixed_seq   = atom_array.is_motif_atom_with_fixed_seq
unindexed   = atom_array.is_motif_atom_unindexed

print(f"\n  is_motif_atom_with_fixed_coord: {fixed_coord.sum()} 个原子固定坐标")
print(f"  is_motif_atom_with_fixed_seq:   {fixed_seq.sum()} 个原子固定序列")
print(f"  is_motif_atom_unindexed:        {unindexed.sum()} 个原子无位置索引")

chain_ids   = np.unique(atom_array.chain_id)
res_names   = np.unique(atom_array.res_name)
print(f"\n  链 (chain_ids): {chain_ids}")
print(f"  残基类型 (res_names 前10): {res_names[:10]}")

# 计算 motif 残基信息
motif_res_idx = np.unique(atom_array.res_id[fixed_coord])
print(f"\n  固定坐标的残基 res_id: {motif_res_idx}")

# ─── STEP 3: 数据 Pipeline Transform ─────────────────────────────────────────

sep("STEP 3: 数据 Pipeline（特征提取 + EDM 噪声采样）")

from rfd3.transforms.pipelines import build_atom14_base_pipeline

pipeline = build_atom14_base_pipeline(
    is_inference=True,
    n_atoms_per_token=14,
    sigma_data=16.0,           # from rfd3_net.yaml
    central_atom="CB",         # from design_base.yaml
    atom_1d_features={         # from rfd3_net.yaml token_initializer.atom_1d_features
        "ref_atom_name_chars": 256,
        "ref_element": 128,
        "ref_charge": 1,
        "ref_mask": 1,
        "ref_is_motif_atom_with_fixed_coord": 1,
        "ref_is_motif_atom_unindexed": 1,
        "has_zero_occupancy": 1,
        "ref_pos": 3,
        "ref_atomwise_rasa": 3,
        "active_donor": 1,
        "active_acceptor": 1,
        "is_atom_level_hotspot": 1,
    },
    token_1d_features={        # from rfd3_net.yaml token_initializer.token_1d_features
        "ref_motif_token_type": 3,
        "restype": 32,
        "ref_plddt": 1,
        "is_non_loopy": 1,
    },
    sigma_perturb=2.0,
    sigma_perturb_com=1.0,
    association_scheme="dense",
    diffusion_batch_size=4,
    center_option="diffuse",
    max_atoms_in_crop=3840,
    generate_conformers=True,
    generate_conformers_for_non_protein_only=True,
    provide_reference_conformer_when_unmasked=True,
    ground_truth_conformer_policy="IGNORE",
    provide_elements_for_unindexed_components=True,
    use_element_for_atom_names_of_atomized_tokens=True,
    keep_full_binder_in_spatial_crop=False,
    max_binder_length=170,
    max_ppi_hotspots_frac_to_provide=0.2,
    ppi_hotspot_max_distance=4.5,
    max_ss_frac_to_provide=0.4,
    min_ss_island_len=1,
    max_ss_island_len=10,
    meta_conditioning_probabilities={
        "calculate_hbonds": 0.0,
        "calculate_rasa": 0.0,
        "keep_protein_motif_rasa": 0.0,
        "hbond_subsample": 0.0,
        "unindex_leak_global_index": 0.0,
        "unindex_insert_random_break": 0.0,
        "unindex_remove_random_break": 0.0,
        "add_1d_ss_features": 0.0,
        "featurize_plddt": 0.0,
        "add_global_is_non_loopy_feature": 0.99,
        "add_ppi_hotspots": 0.0,
        "full_binder_crop": 0.0,
    },
    train_conditions={},
)

data_in = prepare_pipeline_input_from_atom_array(atom_array)
data_in["example_id"] = f"demo_{EXAMPLE_KEY}"
data_in["specification"] = spec_dumped

print(f"\n[pipeline 输入 data_in 的 keys]")
print_dict_shapes(data_in)

data_out = pipeline(data_in)
print(f"\n[pipeline 输出 data_out 的 keys]")
print_dict_shapes(data_out)

feats = data_out["feats"]
print(f"\n[data_out['feats'] 特征字典（共 {len(feats)} 项）]")
print_dict_shapes(feats, max_entries=50)

coord = data_out.get("coord_atom_lvl_to_be_noised")
noise = data_out.get("noise")
t_val = data_out.get("t")
print(f"\n[EDM 噪声采样结果]")
print(f"  coord_atom_lvl_to_be_noised: {shape_repr(coord)}")
if coord is not None and isinstance(coord, torch.Tensor):
    print(f"    center: {coord.mean(dim=[-2,-1])}")
print(f"  noise:   {shape_repr(noise)}")
print(f"  t (时间步): {shape_repr(t_val)}")
if t_val is not None and isinstance(t_val, torch.Tensor):
    print(f"    t values: {t_val}")

# ─── STEP 4: 加载模型和 Checkpoint ──────────────────────────────────────────

sep("STEP 4: 加载 RFD3 模型权重")

from foundry.inference_engines.base import BaseInferenceEngine

CKPT = "/home/ubuntu/cqr_files/protein_design/checkpoints/rfd3_latest.ckpt"
print(f"\n[Checkpoint 路径] {CKPT}")
ckpt_data = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"[Checkpoint 键] {list(ckpt_data.keys())}")
if "hyper_parameters" in ckpt_data:
    print(f"[hyper_parameters keys] {list(ckpt_data['hyper_parameters'].keys())}")

state_dict = ckpt_data.get("model", ckpt_data.get("state_dict", {}))
model_keys = [k for k in state_dict.keys() if not k.startswith("_")]
print(f"\n[模型权重参数总量] {len(model_keys)} 个 tensors")
total_params = sum(v.numel() for k, v in state_dict.items() if isinstance(v, torch.Tensor))
print(f"[总参数数量] {total_params:,} ({total_params/1e6:.1f}M)")

# 分模块统计参数量
prefix_counts = {}
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        parts = k.split(".")
        top2 = ".".join(parts[:3]) if len(parts) >= 3 else k
        prefix_counts[top2] = prefix_counts.get(top2, 0) + v.numel()

print(f"\n[各模块参数量 Top-20]")
for k, v in sorted(prefix_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  {k}: {v/1e6:.2f}M")

# ─── STEP 5: 初始化推理引擎 ──────────────────────────────────────────────────

sep("STEP 5: 初始化 RFD3InferenceEngine")

from rfd3.engine import RFD3InferenceEngine, RFD3InferenceConfig

OUT_DIR = "/tmp/rfd3_trace_out"
os.makedirs(OUT_DIR, exist_ok=True)

engine_cfg = RFD3InferenceConfig(
    ckpt_path=CKPT,
    diffusion_batch_size=2,
    skip_existing=False,
    prevalidate_inputs=False,
    dump_trajectories=False,
    dump_prediction_metadata_json=True,
    inference_sampler={
        "kind": "default",
        "num_timesteps": 20,       # 减少步数方便追踪，原始200步
        "step_scale": 1.5,
        "gamma_0": 0.6,
        "gamma_min": 1.0,
        "noise_scale": 1.003,
        "use_classifier_free_guidance": False,
        "allow_realignment": False,
    },
    low_memory_mode=False,
)

print(f"\n[RFD3InferenceConfig 关键参数]")
print(f"  ckpt_path: {engine_cfg.ckpt_path}")
print(f"  diffusion_batch_size: {engine_cfg.diffusion_batch_size}")
print(f"  num_timesteps: {engine_cfg.inference_sampler['num_timesteps']} (演示用缩短)")

engine = RFD3InferenceEngine(**engine_cfg)
print(f"\n[推理引擎初始化完成] type={type(engine)}")

# 初始化引擎（加载模型权重、构建 trainer）
print("\n[正在调用 engine.initialize() 加载模型...]")
engine.initialize()
print(f"[模型加载完成] trainer.state keys: {list(engine.trainer.state.keys())}")

# ─── STEP 6: Monkey-patch 追踪核心前向过程 ──────────────────────────────────

sep("STEP 6: 设置追踪钩子并运行推理")

# 保存追踪结果
trace = {}
step_counter = {"n": 0}

# 获取模型引用（FabricModule → EMA → shadow model used in eval/inference）
rfd3_model = engine.trainer.state["model"].module.shadow
print(f"\n[模型结构] type={type(rfd3_model)}")

# --- 追踪 TokenInitializer.forward ---
orig_token_init_fwd = rfd3_model.token_initializer.forward

def traced_token_initializer(f):
    sep("  [HOOK] TokenInitializer.forward 开始")
    tok_idx = f["atom_to_token_map"]
    L = len(tok_idx)
    I = tok_idx.max().item() + 1
    print(f"  输入 f: L={L} 个原子, I={I} 个 token")
    print(f"  关键输入特征:")
    for k in ["restype", "ref_element", "ref_pos", "is_motif_atom_with_fixed_coord",
              "residue_index", "atom_to_token_map", "token_bonds"]:
        if k in f:
            print(f"    f['{k}']: {shape_repr(f[k])}")

    out = orig_token_init_fwd(f)

    print(f"\n  [TokenInitializer 输出]")
    for k, v in out.items():
        print(f"    {k}: {shape_repr(v)}")
    trace["tokenizer_outputs"] = {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}
    sep("  [HOOK] TokenInitializer.forward 结束")
    return out

rfd3_model.token_initializer.forward = traced_token_initializer

# --- 追踪 DiffusionModule.forward ---
diff_mod = rfd3_model.diffusion_module
orig_diff_fwd = diff_mod.forward

def traced_diffusion_module(X_noisy_L, t, f, **kwargs):
    n = step_counter["n"]
    if n == 0:
        sep(f"  [HOOK] DiffusionModule.forward (第 {n+1} 步 / 20)")
        print(f"  X_noisy_L: {shape_repr(X_noisy_L)}")
        print(f"    min={X_noisy_L.min():.3f}, max={X_noisy_L.max():.3f}, "
              f"mean={X_noisy_L.mean():.3f}, std={X_noisy_L.std():.3f}")
        print(f"  t: {shape_repr(t)}, values={t[:4].tolist()}")

    out = orig_diff_fwd(X_noisy_L, t, f, **kwargs)

    if n == 0:
        print(f"\n  [DiffusionModule 输出]")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {shape_repr(v)}")
                if k == "X_L":
                    print(f"      min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")
                if k == "sequence_logits_I":
                    probs = torch.softmax(v[0], dim=-1)
                    top_tok = probs.argmax(-1)
                    print(f"      第1批 argmax token (前10个): {top_tok[:10].tolist()}")
        trace["first_step_out"] = {k: v.detach().cpu() for k, v in out.items()
                                    if isinstance(v, torch.Tensor)}
        sep(f"  [HOOK] DiffusionModule.forward 步骤 {n+1} 结束")
    elif n == 9:
        sep(f"  [HOOK] DiffusionModule.forward 中间步 (第10步/20)")
        print(f"  X_noisy_L std={X_noisy_L.std():.3f} (噪声水平下降了)")
        if isinstance(out, dict) and "X_L" in out:
            print(f"  X_denoised std={out['X_L'].std():.3f}")
    elif n == 19:
        sep(f"  [HOOK] DiffusionModule.forward 最后步 (第20步/20)")
        print(f"  X_noisy_L std={X_noisy_L.std():.3f}")
        if isinstance(out, dict) and "X_L" in out:
            print(f"  X_final std={out['X_L'].std():.3f}")
        trace["last_step_out"] = {k: v.detach().cpu() for k, v in out.items()
                                   if isinstance(v, torch.Tensor)}

    step_counter["n"] += 1
    return out

diff_mod.forward = traced_diffusion_module

# --- 追踪 DiffusionTokenEncoder ---
dte = diff_mod.diffusion_token_encoder
orig_dte_fwd = dte.forward

def traced_dte(f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs):
    if step_counter["n"] == 0:
        sep("    [HOOK] DiffusionTokenEncoder.forward")
        print(f"    输入:")
        print(f"      R_L (缩放后噪声坐标): {shape_repr(R_L)}")
        print(f"      S_init_I (token单特征): {shape_repr(S_init_I)}")
        print(f"      Z_init_II (token对特征): {shape_repr(Z_init_II)}")
        print(f"      C_L (原子单特征+时间): {shape_repr(C_L)}")
        if P_LL is not None:
            print(f"      P_LL (原子对特征): {shape_repr(P_LL)}")
        D_II_self = kwargs.get("D_II_self")
        print(f"      D_II_self (自条件化distogram): {shape_repr(D_II_self)}")
    S_I, Z_II = orig_dte_fwd(f, R_L, S_init_I, Z_init_II, C_L, P_LL, **kwargs)
    if step_counter["n"] == 0:
        print(f"    输出:")
        print(f"      S_I: {shape_repr(S_I)}")
        print(f"      Z_II: {shape_repr(Z_II)}")
    return S_I, Z_II

dte.forward = traced_dte

# --- 追踪 LocalTokenTransformer (主 diffusion transformer) ---
ltt = diff_mod.diffusion_transformer
orig_ltt_fwd = ltt.forward

def traced_ltt(A_I, S_I, Z_II, f, X_L, full=False):
    if step_counter["n"] == 0:
        sep("    [HOOK] LocalTokenTransformer (DiffusionTransformer, 18块)")
        print(f"    输入:")
        print(f"      A_I (token扩散特征): {shape_repr(A_I)}")
        print(f"        std={A_I.std():.4f}")
        print(f"      S_I (token单特征): {shape_repr(S_I)}")
        print(f"      Z_II (token对特征): {shape_repr(Z_II)}")
        print(f"      X_L (CA坐标 用于kNN索引): {shape_repr(X_L)}")
    out = orig_ltt_fwd(A_I, S_I, Z_II, f, X_L, full)
    if step_counter["n"] == 0:
        print(f"    输出:")
        print(f"      A_I_out: {shape_repr(out)}")
        print(f"        std={out.std():.4f}")
    return out

ltt.forward = traced_ltt

# ─── 运行推理 ─────────────────────────────────────────────────────────────────

sep("STEP 7: 执行推理（仅跑 M0255_1mg5_unfixed）")

t_start = time.time()
outputs = engine.run(
    inputs="demo.json",
    out_dir=OUT_DIR,
    n_batches=1,
)
t_end = time.time()

print(f"\n[推理总耗时] {t_end - t_start:.1f} 秒")

# ─── STEP 8: 分析输出 ────────────────────────────────────────────────────────

sep("STEP 8: 分析输出文件")

import glob
output_files = sorted(glob.glob(f"{OUT_DIR}/*"))
print(f"\n[输出文件]")
for f_path in output_files:
    size = os.path.getsize(f_path)
    print(f"  {os.path.basename(f_path)} ({size/1024:.1f} KB)")

# 读取 JSON 元数据
json_files = [f for f in output_files if f.endswith(".json")]
for jf in json_files[:3]:
    print(f"\n[{os.path.basename(jf)} 内容]")
    with open(jf) as fh:
        meta = json.load(fh)
    # 打印度量
    if "metrics" in meta:
        print(f"  metrics: {json.dumps(meta['metrics'], indent=4)}")
    if "diffused_index_map" in meta:
        print(f"  diffused_index_map: {meta['diffused_index_map']}")
    if "task" in meta:
        print(f"  task: {meta['task']}")

# 读取 CIF 结构
import gzip
cif_files = [f for f in output_files if f.endswith(".cif.gz")]
for cf in cif_files[:1]:
    print(f"\n[{os.path.basename(cf)} 前50行]")
    with gzip.open(cf, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= 50:
                print("  ...")
                break
            print(f"  {line}", end="")

# ─── STEP 9: 追踪结果总结 ────────────────────────────────────────────────────

sep("STEP 9: 关键中间量追踪汇总")

if "tokenizer_outputs" in trace:
    print("\n[TokenInitializer 输出维度汇总]")
    for k, v in trace["tokenizer_outputs"].items():
        print(f"  {k}: {list(v.shape)}")

if "first_step_out" in trace:
    print("\n[第1步去噪输出]")
    for k, v in trace["first_step_out"].items():
        print(f"  {k}: {list(v.shape)}")
        if k == "X_L":
            print(f"    RMSD to zero: {v.norm(dim=-1).mean():.3f} Å")

if "last_step_out" in trace:
    print("\n[最后步去噪输出]")
    for k, v in trace["last_step_out"].items():
        print(f"  {k}: {list(v.shape)}")
        if k == "X_L":
            print(f"    坐标范围: [{v.min():.2f}, {v.max():.2f}] Å")

sep("完成")
