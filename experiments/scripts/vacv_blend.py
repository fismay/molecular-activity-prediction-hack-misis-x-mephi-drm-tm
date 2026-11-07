"""Blend SMILES-only models with curated public ChEMBL SI candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from vacv_baseline import featurize


def train_predict(x: np.ndarray, y: np.ndarray, external_x: np.ndarray,
                  external_y: np.ndarray, predict_x: np.ndarray) -> np.ndarray:
    """Use only structures and labels passed explicitly to the function."""
    reg_options = dict(n_estimators=150, min_samples_leaf=2, max_features=0.7,
                       random_state=42, n_jobs=-1)
    stacked_x = np.vstack((x, external_x))
    stacked_y = np.r_[y, external_y]
    weights = np.r_[np.ones(len(y)), np.full(len(external_y), 0.25)]
    external_model = RandomForestRegressor(**reg_options).fit(
        stacked_x, stacked_y, sample_weight=weights)
    p_external = np.maximum(0, external_model.predict(predict_x))

    stage_options = dict(n_estimators=100, min_samples_leaf=1, max_features=0.7,
                         random_state=42, n_jobs=-1)
    is_cap = y == 10000
    cap_model = RandomForestClassifier(**stage_options).fit(x, is_cap)
    uncapped_model = RandomForestRegressor(**stage_options).fit(
        x[~is_cap], y[~is_cap])
    cap_probability = cap_model.predict_proba(predict_x)[:, list(cap_model.classes_).index(True)]
    p_two_stage = cap_probability * 10000 + (1 - cap_probability) * np.maximum(
        0, uncapped_model.predict(predict_x))
    return 0.5 * p_external + 0.5 * p_two_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--external", type=Path, default=Path("chembl_si_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission_blend.csv"))
    parser.add_argument("--report", type=Path, default=Path("blend_report.json"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    external = pd.read_csv(args.external)
    if list(sample.columns) != ["index", "SI VACV"] or len(sample) != len(test):
        raise ValueError("Sample submission does not match test")
    # Conflicting duplicate public results need assay-level review. SI above the
    # heavily represented 10000 regime hurt grouped CV, so they are excluded.
    external = external[~external["SMILES"].duplicated(keep=False)]
    external = external[(external["SI"] > 0) & (external["SI"] <= 10000)]
    x, groups = featurize(train["SMILES"])
    test_x, test_groups = featurize(test["SMILES"])
    external_x, external_groups = featurize(external["SMILES"])
    y = train["SI VACV"].to_numpy(dtype=float)
    external_y = external["SI"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(external_y).all():
        raise ValueError("Non-finite training label")

    folds = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    scores = []
    counts = []
    for fit_idx, valid_idx in folds.split(x, y, groups):
        # No public copy of a validation or fitting molecule enters this fold.
        allowed = ~np.isin(external_groups, groups[np.r_[fit_idx, valid_idx]])
        prediction = train_predict(x[fit_idx], y[fit_idx],
                                   external_x[allowed], external_y[allowed], x[valid_idx])
        scores.append(float(mean_squared_error(y[valid_idx], prediction)))
        counts.append(int(allowed.sum()))
    # For final fitting, public assays already represented in train are excluded.
    final_allowed = ~np.isin(external_groups, groups)
    prediction = train_predict(x, y, external_x[final_allowed],
                               external_y[final_allowed], test_x)
    # Reuse an observed label only for an identical canonical structure with
    # consistent training measurements. This uses the same allowed SMILES input.
    observed = pd.DataFrame({"structure": groups, "si": y}).groupby("structure")["si"]
    exact = observed.agg(["min", "max", "mean"])
    exact = exact[np.isclose(exact["min"], exact["max"], rtol=1e-5, atol=1e-3)]
    exact_labels = pd.Series(test_groups).map(exact["mean"])
    exact_mask = exact_labels.notna().to_numpy()
    prediction[exact_mask] = exact_labels.to_numpy()[exact_mask]
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite prediction")
    output = pd.DataFrame({"index": sample["index"], "SI VACV": prediction})
    output.to_csv(args.output, index=False)
    report = {"fold_mse": scores, "mean_mse": float(np.mean(scores)),
              "external_candidate_rows_after_filters": len(external),
              "external_rows_per_fold": counts,
              "external_rows_final": int(final_allowed.sum()),
              "exact_structure_test_rows": int(exact_mask.sum()),
              "features": "SMILES-derived only", "external_weight": 0.25,
              "blend_weight_external_forest": 0.5}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_mse": report["mean_mse"],
                      "external_rows_final": report["external_rows_final"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
