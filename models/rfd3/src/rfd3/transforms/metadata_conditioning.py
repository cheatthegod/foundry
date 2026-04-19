"""Override conditioning flags based on per-sample metadata from enriched CSV.

When training with enriched Prot2Text data, each sample has metadata columns
(pocket_profile, fold_bias, active_site_style, etc.) that should drive which
conditioning features are active, rather than pure random sampling.

This transform reads metadata from data["extra_info"] (populated by
GenericDFParser from extra CSV columns) and overrides the randomly-sampled
flags in data["conditions"].

Insert this transform AFTER SampleConditioningType in the pipeline.

CSV columns consumed:
  cond_rasa          "always" | "never" | "random" (default: "random")
  cond_ss            "always" | "never" | "random"
  cond_non_loopy     "always" | "never" | "random"
  cond_plddt         "always" | "never" | "random"

These are pre-computed by the enrichment pipeline based on enzyme metadata:
  - deeply_buried pocket → cond_rasa=always
  - helical/sheet_rich fold → cond_ss=always
  - is_non_loopy structure → cond_non_loopy=always
"""

from atomworks.ml.transforms.base import Transform


class OverrideConditioningFromMetadata(Transform):
    """Read per-sample conditioning flags from extra_info and override data['conditions']."""

    # Map from CSV column name to conditions dict key
    COLUMN_TO_CONDITION = {
        "cond_rasa": "calculate_rasa",
        "cond_ss": "add_1d_ss_features",
        "cond_non_loopy": "add_global_is_non_loopy_feature",
        "cond_plddt": "featurize_plddt",
    }

    def forward(self, data: dict) -> dict:
        extra = data.get("extra_info", {})
        conditions = data.get("conditions", {})

        for col, cond_key in self.COLUMN_TO_CONDITION.items():
            val = extra.get(col)
            if val is None or val == "" or val == "random":
                continue  # keep the randomly sampled value
            if val == "always":
                conditions[cond_key] = True
            elif val == "never":
                conditions[cond_key] = False

        data["conditions"] = conditions
        return data
