#!/usr/bin/env python3
"""Build a route-A constraint plan from task text to vanilla RFD3 inputs.

This script is intentionally conservative. It converts the current
requirement/task JSON into:

1. A normalized constraint plan that the controller can reason over.
2. An RFD3 input-spec draft that separates fields that are ready now from
   fields that still require downstream structure objects such as S2/S3.

It does not claim that free text alone is enough to fill atom-indexed RFD3
fields like ``select_fixed_atoms`` or ``select_hotspots``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-json",
        type=Path,
        default=None,
        help="Path to a task.json file that points to requirement_json.",
    )
    parser.add_argument(
        "--requirement-json",
        type=Path,
        default=None,
        help="Path to a requirement.json file. Used when --task-json is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for constraint_plan.json and rfd3_input_spec_draft.json.",
    )
    return parser.parse_args()


def pick_first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return value
    return None


def resolve_task_and_requirement(
    task_json: Path | None,
    requirement_json: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if task_json is None and requirement_json is None:
        raise ValueError("Provide either --task-json or --requirement-json.")

    if task_json is not None:
        task_path = task_json.resolve()
        task = read_json(task_path)
        root_dir = task_path.parent
        req_rel = task.get("paths", {}).get("requirement_json")
        requirement = (
            read_json((root_dir / req_rel).resolve())
            if req_rel
            else (read_json(requirement_json.resolve()) if requirement_json else {})
        )
        return task, requirement, root_dir

    requirement_path = requirement_json.resolve()
    requirement = read_json(requirement_path)
    task = {
        "task_id": requirement.get("task_id", requirement_path.stem),
        "substrate_name": requirement.get("substrate_or_ts"),
        "metal_type": [],
        "reaction_class": requirement.get("reaction_type"),
        "enzyme_family": None,
        "fold_class": None,
    }
    return task, requirement, requirement_path.parent


def normalize_goal_text(goal_text: str) -> str:
    return re.sub(r"\s+", " ", goal_text or "").strip()


def infer_text_cues(goal_text: str) -> dict[str, Any]:
    lower = goal_text.lower()
    return {
        "mentions_diiron": "diiron" in lower or "binuclear iron" in lower,
        "mentions_mixed_valent": "mixed-valent" in lower or "mixed valent" in lower,
        "mentions_lid": "lid" in lower or "loop-like element" in lower,
        "mentions_gate": "gate" in lower or "entrance" in lower,
        "mentions_semi_buried": "semi-buried" in lower or "semi buried" in lower,
        "mentions_buried": "buried" in lower,
        "mentions_polar": "polar" in lower,
        "mentions_hydrophobic_shell": "hydrophobic outer shell" in lower
        or "hydrophobic boundary" in lower
        or "hydrophobic wall" in lower,
        "mentions_helical": "alpha-helical" in lower
        or "helical fold" in lower
        or "predominantly a-helical" in lower
        or "predominantly \u03b1-helical" in goal_text,
        "mentions_rigid_core": "rigid protein core" in lower
        or "rigid core" in lower
        or "well packed" in lower,
        "mentions_hydroxyl_recognition": "hydroxyl" in lower,
        "mentions_oxygen_access": "open coordination position for o" in lower
        or "access to molecular oxygen" in lower,
    }


def infer_length_spec(length_preferences: dict[str, Any]) -> tuple[dict[str, Any], str | int | None]:
    min_len = length_preferences.get("min_len")
    max_len = length_preferences.get("max_len")
    target_len = length_preferences.get("target_len")
    if isinstance(target_len, int):
        return {
            "min_len": target_len,
            "max_len": target_len,
            "target_len": target_len,
        }, target_len
    if isinstance(min_len, int) and isinstance(max_len, int):
        return {
            "min_len": min_len,
            "max_len": max_len,
            "target_len": None,
        }, f"{min_len}-{max_len}"
    return {
        "min_len": min_len,
        "max_len": max_len,
        "target_len": target_len,
    }, None


def infer_burial(pocket_constraints: dict[str, Any], cues: dict[str, Any]) -> str | None:
    explicit = pocket_constraints.get("burial")
    if explicit and explicit != "unknown":
        return explicit
    if cues["mentions_semi_buried"]:
        return "semi_buried"
    if cues["mentions_buried"]:
        return "buried"
    return None


def infer_polarity_pattern(pocket_constraints: dict[str, Any], cues: dict[str, Any]) -> str | None:
    explicit = pocket_constraints.get("polarity_pattern")
    if explicit and explicit != "unknown":
        return explicit
    if cues["mentions_polar"] and cues["mentions_hydrophobic_shell"]:
        return "polar_core_hydrophobic_shell"
    if cues["mentions_polar"]:
        return "polar"
    return None


def infer_fold_bias(cues: dict[str, Any]) -> str | None:
    if cues["mentions_helical"]:
        return "helical"
    return None


def infer_ligand_name(task: dict[str, Any], requirement: dict[str, Any]) -> str | None:
    ligand = pick_first_nonempty(
        requirement.get("ligand"),
        requirement.get("substrate_or_ts"),
        task.get("substrate_name"),
    )
    if isinstance(ligand, str):
        ligand = ligand.strip()
        if ligand and ligand.lower() != "unknown" and re.fullmatch(r"[A-Z0-9]{1,4}", ligand):
            return ligand
    return None


def build_slot_plans(
    task: dict[str, Any],
    requirement: dict[str, Any],
    cues: dict[str, Any],
    length_summary: dict[str, Any],
    pocket_profile: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metal_types = task.get("metal_type", [])
    required_roles = requirement.get("required_roles", [])
    length_text = (
        f"{length_summary['min_len']}-{length_summary['max_len']}"
        if length_summary.get("min_len") and length_summary.get("max_len")
        else "unspecified"
    )

    s2_hard = []
    if metal_types:
        s2_hard.append(f"Include a metal-centered motif with metals: {metal_types}.")
    if required_roles:
        s2_hard.append(f"Cover catalytic roles: {required_roles}.")
    if cues["mentions_diiron"]:
        s2_hard.append("Preserve a diiron / binuclear-iron geometry prior.")
    if cues["mentions_hydroxyl_recognition"]:
        s2_hard.append("Support multidentate hydroxyl recognition around the substrate.")
    if cues["mentions_oxygen_access"]:
        s2_hard.append("Leave oxygen-access geometry available near the metal center.")

    s3_hard = []
    if pocket_profile.get("burial"):
        s3_hard.append(f"Pocket burial target: {pocket_profile['burial']}.")
    if pocket_profile.get("channel"):
        s3_hard.append(f"Pocket access requirement: {pocket_profile['channel']}.")
    if pocket_profile.get("polarity_pattern"):
        s3_hard.append(f"Pocket polarity target: {pocket_profile['polarity_pattern']}.")
    if cues["mentions_lid"] or cues["mentions_gate"]:
        s3_hard.append("Reserve gating / lid-like residues near the pocket entrance.")

    s4_soft = []
    if infer_fold_bias(cues):
        s4_soft.append("Prefer a compact alpha-helical scaffold bias.")
    if cues["mentions_rigid_core"]:
        s4_soft.append("Favor a rigid, well-packed core around the active site.")

    return {
        "E": {
            "objective": "Infer a biologically plausible catalytic prior from text.",
            "hard_constraints": [
                f"Reaction class hint: {task.get('reaction_class') or requirement.get('reaction_type') or 'unknown'}.",
                f"Substrate hint: {pick_first_nonempty(requirement.get('substrate_or_ts'), task.get('substrate_name')) or 'unknown'}.",
            ],
            "soft_constraints": [
                "Prefer literature-consistent catalytic chemistry over fold novelty.",
            ],
            "pending_rfd3_fields": [],
        },
        "S2": {
            "objective": "Propose the catalytic motif / theozyme object.",
            "hard_constraints": s2_hard,
            "soft_constraints": [
                "Keep the motif compact and compatible with later scaffolding.",
                "Treat second-shell shaping residues as revisable unless directly justified.",
            ],
            "pending_rfd3_fields": [
                "input",
                "unindex",
                "select_fixed_atoms",
            ],
        },
        "S3": {
            "objective": "Propose a pocket object around the motif.",
            "hard_constraints": s3_hard,
            "soft_constraints": [
                "Use a selective polar core with enough confinement for substrate orientation.",
                "Keep entrance residues revisable for later specificity tuning.",
            ],
            "pending_rfd3_fields": [
                "select_hotspots",
                "select_buried",
                "select_exposed",
            ],
        },
        "S4": {
            "objective": "Propose a scaffold/backbone that can host the motif and pocket.",
            "hard_constraints": [
                f"Global sequence length target: {length_text}.",
            ],
            "soft_constraints": s4_soft + [
                "Keep enough local flexibility near the pocket while preserving core rigidity.",
            ],
            "pending_rfd3_fields": [
                "contig",
            ],
        },
        "Q_seq": {
            "objective": "Design and refold a sequence consistent with the scaffold and active site.",
            "hard_constraints": [
                "Preserve motif geometry and pocket chemistry after sequence assignment.",
            ],
            "soft_constraints": [
                "Prefer sequence solutions that reduce pocket frustration and preserve packing.",
            ],
            "pending_rfd3_fields": [],
        },
        "S5": {
            "objective": "Finalize the enzyme-substrate complex.",
            "hard_constraints": [
                "Maintain productive substrate orientation in the final complex.",
                "Retain metal-centered catalytic geometry after completion/refinement.",
            ],
            "soft_constraints": [
                "Prefer solvent-shielded but accessible active sites.",
            ],
            "pending_rfd3_fields": [],
        },
    }


def build_rfd3_draft(
    task: dict[str, Any],
    requirement: dict[str, Any],
    cues: dict[str, Any],
    pocket_profile: dict[str, Any],
    length_spec: str | int | None,
) -> dict[str, Any]:
    ready_spec: dict[str, Any] = {}
    ligand_name = infer_ligand_name(task, requirement)
    if length_spec is not None:
        ready_spec["length"] = length_spec
    if ligand_name is not None:
        ready_spec["ligand"] = ligand_name

    suggested_overrides: dict[str, Any] = {}
    if cues["mentions_helical"] or cues["mentions_rigid_core"]:
        suggested_overrides["is_non_loopy"] = True

    pending_fields = {
        "input": "Need a motif/template structure object from S2 or a reference object from E.",
        "unindex": "Need motif residue groups after S2 proposes catalytic residues.",
        "select_fixed_atoms": "Need atom-level motif definitions from S2.",
        "select_hotspots": "Need residue or atom hotspots from the S3 pocket object.",
        "select_buried": "Need residue selections from the S3 pocket object.",
        "select_exposed": "Need residue selections from the S3 pocket object.",
        "contig": "Need a scaffolding policy or an existing template after S4 planning.",
        "ori_token": "Need an object-level geometry estimate before setting the diffusion origin.",
    }

    notes = [
        "This draft only fills RFD3 fields that can be justified directly from task text.",
        "Atom-indexed controls remain pending until controller/backends produce explicit objects.",
    ]
    if pocket_profile.get("burial"):
        notes.append(
            "Pocket burial should later be translated into residue selections for "
            "select_buried/select_exposed rather than guessed globally."
        )

    return {
        "task_id": task["task_id"],
        "ready_input_spec": {
            task["task_id"]: ready_spec,
        },
        "suggested_overrides": suggested_overrides,
        "pending_fields": pending_fields,
        "notes": notes,
    }


def build_constraint_plan(task: dict[str, Any], requirement: dict[str, Any]) -> dict[str, Any]:
    goal_text = normalize_goal_text(
        pick_first_nonempty(requirement.get("goal_text"), task.get("goal_text"), "")
    )
    cues = infer_text_cues(goal_text)
    length_summary, length_spec = infer_length_spec(
        requirement.get("length_preferences", {})
    )
    pocket_constraints = requirement.get("pocket_constraints", {})
    pocket_profile = {
        "burial": infer_burial(pocket_constraints, cues),
        "channel": pick_first_nonempty(
            pocket_constraints.get("channel"),
            "gated" if cues["mentions_lid"] or cues["mentions_gate"] else None,
        ),
        "polarity_pattern": infer_polarity_pattern(pocket_constraints, cues),
    }

    plan = {
        "task_id": task["task_id"],
        "route": "text_to_structured_constraints_to_vanilla_rfd3",
        "goal_text": goal_text,
        "global_summary": {
            "enzyme_family": task.get("enzyme_family"),
            "reaction_class": pick_first_nonempty(
                task.get("reaction_class"),
                requirement.get("reaction_type"),
            ),
            "substrate_name": pick_first_nonempty(
                requirement.get("substrate_or_ts"),
                task.get("substrate_name"),
            ),
            "metal_types": task.get("metal_type", []),
            "required_roles": requirement.get("required_roles", []),
            "reactive_atoms": requirement.get("reactive_atoms", []),
            "length_preferences": length_summary,
            "pocket_profile": pocket_profile,
            "forbidden_features": requirement.get("forbidden_features", []),
            "uncertainty_slots": requirement.get("uncertainty_slots", []),
            "fold_bias": infer_fold_bias(cues),
        },
        "text_cues": cues,
        "slot_plans": build_slot_plans(
            task=task,
            requirement=requirement,
            cues=cues,
            length_summary=length_summary,
            pocket_profile=pocket_profile,
        ),
        "rfd3_bridge": {
            "ready_now": [
                key
                for key in build_rfd3_draft(task, requirement, cues, pocket_profile, length_spec)[
                    "ready_input_spec"
                ][task["task_id"]].keys()
            ],
            "requires_intermediate_objects": [
                "input",
                "unindex",
                "select_fixed_atoms",
                "select_hotspots",
                "select_buried",
                "select_exposed",
                "contig",
                "ori_token",
            ],
        },
        "provenance": requirement.get("provenance", []),
    }
    return plan


def main() -> None:
    args = parse_args()
    task, requirement, _ = resolve_task_and_requirement(args.task_json, args.requirement_json)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    constraint_plan = build_constraint_plan(task, requirement)
    goal_text = normalize_goal_text(
        pick_first_nonempty(requirement.get("goal_text"), task.get("goal_text"), "")
    )
    cues = infer_text_cues(goal_text)
    pocket_profile = constraint_plan["global_summary"]["pocket_profile"]
    _, length_spec = infer_length_spec(requirement.get("length_preferences", {}))
    rfd3_draft = build_rfd3_draft(task, requirement, cues, pocket_profile, length_spec)

    write_json(output_dir / "constraint_plan.json", constraint_plan)
    write_json(output_dir / "rfd3_input_spec_draft.json", rfd3_draft)

    ready_fields = sorted(rfd3_draft["ready_input_spec"][task["task_id"]].keys())
    print("Constraint planning complete")
    print(f"task_id: {task['task_id']}")
    print(f"output_dir: {output_dir}")
    print(f"ready_rfd3_fields: {ready_fields if ready_fields else '[]'}")
    print(
        "pending_rfd3_fields:",
        sorted(rfd3_draft["pending_fields"].keys()),
    )


if __name__ == "__main__":
    main()
