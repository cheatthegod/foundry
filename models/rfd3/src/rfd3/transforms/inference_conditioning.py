"""Apply global conditioning annotations at inference from specification extra fields.

Reads preference signals from data["specification"]["extra"] and applies them
as global atom-level annotations, enabling typed enzyme generation without
modifying the model architecture.

Supported extra fields:
  prefer_helix: bool     → sets is_helix_conditioning=1 on all diffused atoms
  prefer_sheet: bool     → sets is_sheet_conditioning=1 on all diffused atoms
  prefer_loop: bool      → sets is_loop_conditioning=1 on all diffused atoms
  prefer_buried: bool    → sets rasa_bin=0 on all diffused atoms
  prefer_exposed: bool   → sets rasa_bin=2 on all diffused atoms

These work because the model was trained with meta_conditioning that associates
these annotations with actual structural features via OverrideConditioningFromMetadata.
"""

import numpy as np
from atomworks.ml.transforms.base import Transform

from rfd3.transforms.conditioning_base import get_motif_features


class ApplyInferenceConditioning(Transform):
    """Set global conditioning annotations from specification extra fields."""

    def forward(self, data: dict) -> dict:
        spec = data.get("specification", {})
        extra = spec.get("extra", {}) if isinstance(spec, dict) else {}
        atom_array = data.get("atom_array")

        if atom_array is None or not extra:
            return data

        n_atoms = atom_array.array_length()

        # Get diffused region mask (non-motif atoms)
        try:
            motif_features = get_motif_features(atom_array)
            is_motif = motif_features["is_motif_token"].astype(bool)
            diffused_mask = ~is_motif
        except Exception:
            diffused_mask = np.ones(n_atoms, dtype=bool)

        # SS conditioning
        if extra.get("prefer_helix"):
            annot = np.zeros(n_atoms, dtype=int)
            annot[diffused_mask] = 1
            atom_array.set_annotation("is_helix_conditioning", annot)

        if extra.get("prefer_sheet"):
            annot = np.zeros(n_atoms, dtype=int)
            annot[diffused_mask] = 1
            atom_array.set_annotation("is_sheet_conditioning", annot)

        if extra.get("prefer_loop"):
            annot = np.zeros(n_atoms, dtype=int)
            annot[diffused_mask] = 1
            atom_array.set_annotation("is_loop_conditioning", annot)

        # RASA burial conditioning
        if extra.get("prefer_buried"):
            annot = np.full(n_atoms, 3, dtype=int)  # 3 = no info (default)
            annot[diffused_mask] = 0  # 0 = buried
            atom_array.set_annotation("rasa_bin", annot)

        if extra.get("prefer_exposed"):
            annot = np.full(n_atoms, 3, dtype=int)
            annot[diffused_mask] = 2  # 2 = exposed
            atom_array.set_annotation("rasa_bin", annot)

        data["atom_array"] = atom_array
        return data
