#!/usr/bin/env python3
"""Quick checkpoint comparison on a fixed subset of training examples.

This script is designed for fast apples-to-apples comparisons between RFD3
checkpoints trained in this repository. It does not run a full inference
rollout. Instead, it reuses the training-time single-step denoising forward
pass and loss computation on a fixed materialized set of examples.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning.fabric import seed_everything
from lightning_utilities import apply_to_collection
from omegaconf import DictConfig, OmegaConf

from foundry.utils.datasets import (
    recursively_instantiate_datasets_and_samplers,
    subset_dataset_to_example_ids,
)


EXAMPLE_ID_RE = re.compile(r"\{\[[^\n]+?\]\}\{[^}]+\}\{[^}]+\}\{[^}]+\}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training run directory containing .hydra/config.yaml and ckpt/.",
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Checkpoint filenames or paths to compare.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="Number of fixed examples to compare on if --example-ids is omitted.",
    )
    parser.add_argument(
        "--example-ids",
        nargs="*",
        default=None,
        help="Explicit example IDs to evaluate.",
    )
    parser.add_argument(
        "--example-csv",
        type=Path,
        default=None,
        help=(
            "CSV containing an example_id column to evaluate. "
            "When provided, the evaluation dataset is also sourced from this CSV."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of examples loaded from --example-csv.",
    )
    parser.add_argument(
        "--n-cycle",
        type=int,
        default=1,
        help="Fixed recycle count for the quick forward pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed used to materialize the fixed examples.",
    )
    return parser.parse_args()


def resolve_ckpt_paths(run_dir: Path, ckpts: list[str]) -> list[Path]:
    ckpt_dir = run_dir / "ckpt"
    resolved = []
    for ckpt in ckpts:
        ckpt_path = Path(ckpt)
        if not ckpt_path.is_absolute():
            ckpt_path = ckpt_dir / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        resolved.append(ckpt_path.resolve())
    return resolved


def override_dataset_csv(cfg_section: Any, csv_path: Path) -> int:
    if cfg_section is None:
        return 0

    if isinstance(cfg_section, DictConfig) and "dataset" in cfg_section:
        dataset_cfg = cfg_section.dataset
        if (
            isinstance(dataset_cfg, DictConfig)
            and "dataset" in dataset_cfg
            and "data" in dataset_cfg.dataset
        ):
            dataset_cfg.dataset.data = str(csv_path)
            return 1

    updated = 0
    if isinstance(cfg_section, DictConfig):
        for _, nested_cfg in cfg_section.items():
            updated += override_dataset_csv(nested_cfg, csv_path)
    return updated


def load_cfg(run_dir: Path, device: str, example_csv: Path | None = None) -> Any:
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    root_dir = run_dir.parents[3]

    OmegaConf.set_struct(cfg, False)
    cfg.paths.root_dir = str(root_dir)
    cfg.paths.log_dir = str(root_dir / "models" / "rfd3" / "local_runs")
    cfg.paths.output_dir = "/tmp/rfd3_eval_quick"
    cfg.paths.work_dir = str(root_dir)
    cfg.trainer.output_dir = "/tmp/rfd3_eval_quick"
    cfg.trainer.accelerator = device
    cfg.trainer.devices_per_node = 1
    cfg.trainer.num_nodes = 1
    cfg.dataloader.train.dataloader_params.num_workers = 0
    cfg.dataloader.train.dataloader_params.prefetch_factor = None
    cfg.dataloader.train.dataloader_params.pin_memory = False
    if example_csv is not None:
        updated = override_dataset_csv(cfg.datasets.train, example_csv.resolve())
        if updated == 0:
            raise RuntimeError(f"Failed to override dataset CSV with: {example_csv}")
    if device == "cpu":
        cfg.trainer.precision = "32-true"
    return cfg


def pick_example_ids(run_dir: Path, cfg: Any, num_examples: int) -> list[str]:
    experiment_log = run_dir / "experiment.log"
    example_ids: list[str] = []
    if experiment_log.exists():
        text = experiment_log.read_text()
        for match in EXAMPLE_ID_RE.findall(text):
            if match not in example_ids:
                example_ids.append(match)
            if len(example_ids) >= num_examples:
                return example_ids

    csv_path = Path(cfg.datasets.train.prot2text_core.dataset.dataset.data)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = row["example_id"]
            if eid not in example_ids:
                example_ids.append(eid)
            if len(example_ids) >= num_examples:
                break
    return example_ids


def load_example_ids_from_csv(csv_path: Path, limit: int | None = None) -> list[str]:
    example_ids = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        if "example_id" not in (reader.fieldnames or []):
            raise RuntimeError(f"CSV is missing example_id column: {csv_path}")
        for row in reader:
            example_ids.append(row["example_id"])
            if limit is not None and len(example_ids) >= limit:
                break
    return example_ids


def preview_example_ids(example_ids: list[str], max_preview: int = 8) -> None:
    print(f"Prepared {len(example_ids)} example IDs for evaluation.")
    for eid in example_ids[:max_preview]:
        print(f"  - {eid}")
    if len(example_ids) > max_preview:
        print(f"  ... and {len(example_ids) - max_preview} more")


def iter_examples(cfg: Any, example_ids: list[str], seed: int) -> tuple[Any, list[str]]:
    seed_everything(seed, workers=True, verbose=False)
    dataset_and_sampler = recursively_instantiate_datasets_and_samplers(cfg.datasets.train)
    train_dataset = dataset_and_sampler["dataset"]
    subset = subset_dataset_to_example_ids(train_dataset, example_ids)

    kept_ids = []
    for i, requested_id in enumerate(example_ids):
        try:
            example = subset[i]
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] failed to materialize {requested_id}: {exc}")
            continue
        kept_ids.append(example["example_id"])
        yield example

    if len(kept_ids) != len(example_ids):
        print(f"Evaluating {len(kept_ids)} successfully materialized examples.")


def move_to_device(data: Any, device: torch.device) -> Any:
    return apply_to_collection(data, torch.Tensor, lambda x: x.to(device))


def get_eval_model(wrapped_model: Any) -> Any:
    model = wrapped_model.module if hasattr(wrapped_model, "module") else wrapped_model
    if hasattr(model, "shadow"):
        return model.shadow
    return model


def safe_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def evaluate_ckpt(
    trainer: Any,
    ckpt_path: Path,
    cfg: Any,
    example_ids: list[str],
    seed: int,
    n_cycle: int,
    device: torch.device,
) -> dict[str, float]:
    trainer.load_checkpoint(ckpt_path)
    eval_model = get_eval_model(trainer.state["model"])
    eval_model.eval()

    metrics = defaultdict(list)
    n_examples = 0
    for example in iter_examples(cfg, example_ids, seed):
        example_dev = move_to_device(example, device)
        with torch.no_grad():
            network_input = {
                "X_noisy_L": torch.nan_to_num(
                    example_dev["coord_atom_lvl_to_be_noised"] + example_dev["noise"]
                ),
                "t": example_dev["t"],
                "f": example_dev["feats"],
            }
            initializer_outputs = eval_model.token_initializer(network_input["f"])
            network_output = eval_model.diffusion_module(
                X_noisy_L=network_input["X_noisy_L"],
                t=network_input["t"],
                f=network_input["f"],
                n_recycle=n_cycle,
                **initializer_outputs,
            )
            loss_input = trainer._assemble_loss_extra_info(example_dev)
            total_loss, loss_dict = trainer.loss(
                network_input=network_input,
                network_output=network_output,
                loss_input=loss_input,
            )

        metrics["total_loss"].append(safe_float(total_loss))
        for key, value in loss_dict.items():
            metrics[key].append(safe_float(value))
        n_examples += 1

    if n_examples == 0:
        raise RuntimeError(f"No examples were evaluated for checkpoint: {ckpt_path}")

    return {key: sum(values) / len(values) for key, values in metrics.items() if values}


def build_trainer(cfg: Any) -> Any:
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        loggers=None,
        callbacks=None,
        _convert_="partial",
        _recursive_=False,
    )
    trainer.initialize_or_update_trainer_state({"train_cfg": cfg})
    trainer.fabric.launch()
    trainer.construct_model()
    trainer.construct_optimizer()
    trainer.construct_scheduler()
    trainer.setup_model_optimizers_and_schedulers()
    return trainer


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_paths = resolve_ckpt_paths(run_dir, args.ckpts)
    cfg = load_cfg(run_dir, args.device, example_csv=args.example_csv)

    if args.example_ids:
        example_ids = list(args.example_ids)
    elif args.example_csv:
        example_ids = load_example_ids_from_csv(args.example_csv.resolve(), limit=args.limit)
    else:
        example_ids = pick_example_ids(run_dir, cfg, args.num_examples)
    if not example_ids:
        raise RuntimeError("No example IDs available for evaluation.")

    preview_example_ids(example_ids)

    trainer = build_trainer(cfg)
    device = trainer.fabric.device

    rows = []
    for ckpt_path in ckpt_paths:
        metrics = evaluate_ckpt(
            trainer=trainer,
            ckpt_path=ckpt_path,
            cfg=cfg,
            example_ids=example_ids,
            seed=args.seed,
            n_cycle=args.n_cycle,
            device=device,
        )
        rows.append((ckpt_path.name, metrics))

    print("\nQuick checkpoint comparison")
    print(f"Run dir: {run_dir}")
    print(f"Device: {args.device}")
    print(f"Examples: {len(example_ids)}")
    if args.example_csv:
        print(f"Example CSV: {args.example_csv.resolve()}")
    print(f"n_cycle: {args.n_cycle}")
    print()

    wanted_keys = [
        "total_loss",
        "mse_loss_mean",
        "mean_lddt",
        "token_lvl_sequence_loss",
        "seq_recovery",
        "lowest_t_seq_recovery",
        "valid_t_fraction",
    ]

    header = ["checkpoint"] + wanted_keys
    print("\t".join(header))
    for ckpt_name, metrics in rows:
        values = [ckpt_name]
        for key in wanted_keys:
            value = metrics.get(key)
            values.append(f"{value:.4f}" if value is not None else "NA")
        print("\t".join(values))


if __name__ == "__main__":
    main()
