#!/usr/bin/env python3
"""Bridge enriched Route A constraints to RFD3 planning format.

Converts a constraint record from prot2text_route_a_constraints.jsonl into
the task.json + requirement.json format that plan_text_to_rfd3_constraints.py
consumes, then invokes the planner to produce constraint_plan.json and
rfd3_input_spec_draft.json.

Usage:
  # Plan for a specific accession
  python enriched_to_rfd3_plan.py --accession A0R4Q6 --output-dir /tmp/plan_A0R4Q6

  # Plan for the first N enzymes
  python enriched_to_rfd3_plan.py --batch --limit 10 --output-dir /tmp/batch_plans

  # Plan for all enzymes of a specific reaction type
  python enriched_to_rfd3_plan.py --batch --filter-reaction oxidoreductase --output-dir /tmp/oxidoreductase_plans
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_CONSTRAINTS_PATH = PROJECT_DIR / "Prot2Text-Data" / "enriched" / "prot2text_route_a_constraints.jsonl"
PLANNER_SCRIPT = SCRIPT_DIR / "plan_text_to_rfd3_constraints.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--constraints-jsonl", type=Path, default=DEFAULT_CONSTRAINTS_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--accession", help="Process a single accession.")
    mode.add_argument("--batch", action="store_true", help="Process multiple constraints.")

    parser.add_argument("--limit", type=int, default=0, help="Limit number in batch mode.")
    parser.add_argument("--filter-reaction", help="Filter by reaction_type in batch mode.")
    parser.add_argument("--filter-confidence", choices=["high", "medium", "low"], help="Filter by confidence.")
    parser.add_argument("--skip-planner", action="store_true", help="Only generate JSONs, don't invoke planner.")
    return parser.parse_args()


def constraint_to_task_requirement(c: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert an enriched constraint record into task + requirement format."""
    task = {
        "task_id": c["accession"],
        "substrate_name": c.get("substrate_family"),
        "metal_type": c.get("metal_hint") or [],
        "reaction_class": c.get("reaction_type"),
        "enzyme_family": c.get("ec_number", "").split(".")[0] if c.get("ec_number") else None,
        "fold_class": c.get("fold_bias"),
    }

    # Build goal text from available enriched fields
    goal_parts = []
    if c.get("function_text"):
        goal_parts.append(c["function_text"])
    if c.get("protein_names"):
        goal_parts.append(f"Protein: {c['protein_names']}")
    if c.get("fold_bias") and c["fold_bias"] != "mixed":
        fold_map = {"helical": "alpha-helical fold", "sheet_rich": "beta-sheet-rich fold", "loop_dominated": "loop-dominated structure"}
        goal_parts.append(f"Structural bias: {fold_map.get(c['fold_bias'], c['fold_bias'])}")
    if c.get("pocket_profile"):
        pocket_map = {"deeply_buried": "buried active site", "semi_buried": "semi-buried pocket", "surface_exposed": "surface-exposed site"}
        goal_parts.append(f"Pocket: {pocket_map.get(c['pocket_profile'], c['pocket_profile'])}")

    requirement = {
        "task_id": c["accession"],
        "goal_text": " | ".join(goal_parts),
        "reaction_type": c.get("reaction_type"),
        "substrate_or_ts": c.get("substrate_family"),
        "length_preferences": {
            "min_len": max(64, c.get("sequence_length", 200) - 50) if c.get("sequence_length") else None,
            "max_len": min(1024, c.get("sequence_length", 200) + 50) if c.get("sequence_length") else None,
            "target_len": c.get("sequence_length"),
        },
        "required_roles": c.get("required_roles") or [],
        "reactive_atoms": [],
        "pocket_constraints": {
            "burial": c.get("pocket_profile"),
            "channel": None,
            "polarity_pattern": None,
        },
        "forbidden_features": [],
        "uncertainty_slots": [
            slot for slot in ["required_roles", "active_site_style"]
            if c.get(slot) is None
        ],
        "provenance": [
            {
                "source": "enriched_prot2text",
                "accession": c["accession"],
                "confidence": c.get("confidence"),
                "evidence": c.get("evidence_sources", []),
            }
        ],
    }

    # Add cofactor/metal info to goal text
    if c.get("cofactor_hint"):
        requirement["cofactors"] = c["cofactor_hint"]
    if c.get("metal_hint"):
        requirement["metals"] = c["metal_hint"]

    return task, requirement


def run_planner(task_path: Path, requirement_path: Path, output_dir: Path) -> bool:
    """Invoke plan_text_to_rfd3_constraints.py on the generated JSONs."""
    cmd = [
        sys.executable,
        str(PLANNER_SCRIPT),
        "--task-json", str(task_path),
        "--requirement-json", str(requirement_path),
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Planner failed: {result.stderr.strip()}", flush=True)
        return False
    return True


def process_one(c: dict[str, Any], output_dir: Path, skip_planner: bool) -> bool:
    """Process a single constraint record."""
    acc = c["accession"]
    acc_dir = output_dir / acc
    acc_dir.mkdir(parents=True, exist_ok=True)

    task, requirement = constraint_to_task_requirement(c)

    task_path = acc_dir / "task.json"
    req_path = acc_dir / "requirement.json"

    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False), encoding="utf-8")
    req_path.write_text(json.dumps(requirement, indent=2, ensure_ascii=False), encoding="utf-8")

    if skip_planner:
        return True

    return run_planner(task_path, req_path, acc_dir)


def run() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading constraints from {args.constraints_jsonl} ...", flush=True)
    constraints: list[dict[str, Any]] = []
    with args.constraints_jsonl.open("r") as f:
        for line in f:
            constraints.append(json.loads(line))
    print(f"  Loaded {len(constraints):,} constraints.", flush=True)

    # Filter
    if args.accession:
        constraints = [c for c in constraints if c["accession"] == args.accession]
        if not constraints:
            print(f"  Accession {args.accession} not found.", flush=True)
            return 1
    else:
        if args.filter_reaction:
            constraints = [c for c in constraints if c.get("reaction_type") == args.filter_reaction]
        if args.filter_confidence:
            constraints = [c for c in constraints if c.get("confidence") == args.filter_confidence]
        if args.limit > 0:
            constraints = constraints[:args.limit]

    print(f"  Processing {len(constraints):,} constraints ...", flush=True)

    success = 0
    for i, c in enumerate(constraints, 1):
        ok = process_one(c, args.output_dir, args.skip_planner)
        if ok:
            success += 1
        if i % 100 == 0 or i == len(constraints):
            print(f"  {i:,}/{len(constraints):,} ({success} ok)", flush=True)

    print(f"Done: {success}/{len(constraints)} plans generated in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
