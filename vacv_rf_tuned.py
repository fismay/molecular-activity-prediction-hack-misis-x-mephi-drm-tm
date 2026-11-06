"""Train a SMILES-only random forest selected by repeated grouped validation."""

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


def model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=250, min_samples_leaf=1, max_features=0.3,
        random_state=42, n_jobs=-1,
    )


def predict(x: np.ndarray, y: np.ndarray, target: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "forest":
        return np.maximum(0, model().fit(x, y).predict(target))
    is_regime = y == 10000
    classifier = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=1, max_features=0.1,
        random_state=42, n_jobs=-1,
    ).fit(x, is_regime)
    if len(classifier.classes_) < 2:
        return np.maximum(0, model().fit(x, y).predict(target))
    probability = classifier.predict_proba(target)[:, list(classifier.classes_).index(True)]
    conditional = model().fit(x[~is_regime], y[~is_regime]).predict(target)
    return probability * 10000 + (1 - probability) * np.maximum(0, conditional)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission_rf_tuned.csv"))
    parser.add_argument("--report", type=Path, default=Path("rf_tuned_report.json"))
    parser.add_argument("--strategy", choices=("forest", "mixture"), default="forest")
    args = parser.parse_args()

    train, test, sample = (pd.read_csv(path) for path in (args.train, args.test, args.sample))
    if list(sample.columns) != ["index", "SI VACV"] or len(test) != len(sample):
        raise ValueError("Unexpected sample submission format")
    x, groups = featurize(train["SMILES"])
    test_x, _ = featurize(test["SMILES"])
    y = train["SI VACV"].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Non-finite training SI")

    scores = {}
    for seed in (42, 2026, 17):
        fold_scores = []
        for fit_idx, valid_idx in GroupShuffleSplit(
            n_splits=5, test_size=0.2, random_state=seed,
        ).split(x, y, groups):
            pred = predict(x[fit_idx], y[fit_idx], x[valid_idx], args.strategy)
            fold_scores.append(float(mean_squared_error(y[valid_idx], pred)))
        scores[str(seed)] = fold_scores

    predictions = predict(x, y, test_x, args.strategy)
    if not np.isfinite(predictions).all():
        raise ValueError("Non-finite predictions")
    output = pd.DataFrame({"index": sample["index"], "SI VACV": predictions})
    output.to_csv(args.output, index=False)
    report = {
        "metric": "MSE on original SI scale",
        "features": "Morgan radius 2 fingerprint and nine RDKit descriptors from SMILES",
        "model": "RandomForestRegressor(n_estimators=250, min_samples_leaf=1, max_features=0.3, random_state=42)",
        "strategy": args.strategy,
        "regime_classifier_max_features": 0.1 if args.strategy == "mixture" else None,
        "grouped_validation": scores,
        "mean_mse": float(np.mean([score for fold in scores.values() for score in fold])),
        "test_rows": len(output),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mean_mse": report["mean_mse"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
