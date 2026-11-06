"""Reproducible SMILES-only baseline for the VACV SI Kaggle competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")
DESCRIPTORS = (
    Descriptors.MolWt,
    Descriptors.MolLogP,
    Descriptors.TPSA,
    Descriptors.NumHDonors,
    Descriptors.NumHAcceptors,
    Descriptors.NumRotatableBonds,
    Descriptors.RingCount,
    Descriptors.HeavyAtomCount,
    Descriptors.FractionCSP3,
)
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def featurize(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return molecular features and canonical isomeric SMILES groups."""
    matrix = []
    groups = []
    for row, raw in enumerate(smiles.astype(str)):
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {row}: {raw!r}")
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        bits = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(FP.GetFingerprint(mol), bits)
        values = np.asarray([f(mol) for f in DESCRIPTORS], dtype=np.float32)
        matrix.append(np.concatenate((bits, values)))
        groups.append(canonical)
    return np.vstack(matrix), np.asarray(groups)


def candidates() -> dict[str, object]:
    return {
        "mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.7,
            random_state=42, n_jobs=-1,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, max_features=0.7,
            random_state=42, n_jobs=-1,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--report", type=Path, default=Path("baseline_report.json"))
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    if "SMILES" not in train or "SMILES" not in test or "SI VACV" not in train:
        raise ValueError("Missing required SMILES or SI VACV column")
    if list(sample.columns) != ["index", "SI VACV"] or len(sample) != len(test):
        raise ValueError("Sample submission columns or row count do not match test")
    if not np.isfinite(pd.to_numeric(train["SI VACV"], errors="coerce")).all():
        raise ValueError("Training SI contains missing or non-finite values")

    # The forbidden concentration columns are intentionally never read below.
    x_train, groups = featurize(train["SMILES"])
    x_test, test_groups = featurize(test["SMILES"])
    y = train["SI VACV"].to_numpy(dtype=np.float64)
    splits = list(GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42).split(x_train, y, groups))
    report: dict[str, object] = {
        "train_rows": len(train), "test_rows": len(test),
        "unique_train_structures": len(set(groups)),
        "train_test_structure_overlap": len(set(groups) & set(test_groups)),
        "metric": "MSE on original SI scale",
        "models": {},
    }
    for name, model in candidates().items():
        fold_mse = []
        for fit_idx, valid_idx in splits:
            fitted = clone(model).fit(x_train[fit_idx], y[fit_idx])
            prediction = np.maximum(0.0, fitted.predict(x_train[valid_idx]))
            fold_mse.append(float(mean_squared_error(y[valid_idx], prediction)))
        report["models"][name] = {"fold_mse": fold_mse, "mean_mse": float(np.mean(fold_mse))}

    winner = min(report["models"], key=lambda name: report["models"][name]["mean_mse"])
    report["selected_model"] = winner
    final = clone(candidates()[winner]).fit(x_train, y)
    predictions = np.maximum(0.0, final.predict(x_test))
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain non-finite values")
    submission = pd.DataFrame({"index": sample["index"], "SI VACV": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"selected_model": winner, "mean_mse": report["models"][winner]["mean_mse"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
