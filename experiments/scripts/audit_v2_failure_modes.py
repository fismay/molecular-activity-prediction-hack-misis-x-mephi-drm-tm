"""Aggregate failure modes of the confirmed SMILES-only model on grouped folds.

Uses the existing out-of-fold cache; never exports Kaggle rows or test labels.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from vacv_zero_hurdle import TRAIN_V2_SHA256


def main():
    train_path = Path('data/kaggle/train_v2.csv')
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != TRAIN_V2_SHA256:
        raise ValueError('Unexpected train_v2 snapshot')
    data = pd.read_csv(train_path)
    y = data['SI VACV'].to_numpy(float)
    scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(smiles=s)
                          for s in data.SMILES])
    cache_path = Path('artifacts/zero_v2_validation.npz')
    cache = np.load(cache_path)
    observations = []
    for seed in ('42', '2026', '17'):
        for fold in range(5):
            prefix = f'{seed}_{fold}'
            va = cache[f'{prefix}_va']
            prediction = cache[f'{prefix}_prediction_source_stereo_only__raw']
            if len(va) != len(prediction):
                raise ValueError('Mismatched OOF cache')
            for index, estimate in zip(va, prediction):
                observations.append((seed, int(index), float(y[index]), float(estimate)))
    frame = pd.DataFrame(observations, columns=['seed', 'row', 'actual', 'prediction'])
    frame['squared_error'] = (frame.actual-frame.prediction)**2
    total = frame.squared_error.sum()
    bins = {
        'zero': frame.actual == 0,
        'positive_lt_100': (frame.actual > 0) & (frame.actual < 100),
        '100_to_1000': (frame.actual >= 100) & (frame.actual < 1000),
        'ge_1000': frame.actual >= 1000,
    }
    strata = {name: {'heldout_observations': int(mask.sum()),
                     'mse': float(frame.loc[mask, 'squared_error'].mean()),
                     'error_share': float(frame.loc[mask, 'squared_error'].sum()/total)}
              for name, mask in bins.items()}
    by_row = frame.groupby('row').squared_error.sum().sort_values(ascending=False)
    top_row = int(by_row.index[0])
    top_scaffold = scaffolds[top_row]
    scaffold_rows = np.flatnonzero(scaffolds == top_scaffold)
    report = {
        'train_sha256': TRAIN_V2_SHA256,
        'cache_sha256': hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        'recipe': 'source_stereo_only__raw',
        'validation': '15 cached canonical-group folds, with repeated molecules across seeds',
        'heldout_observations': len(frame),
        'unique_heldout_rows': int(frame.row.nunique()),
        'pooled_mse': float(frame.squared_error.mean()),
        'strata': strata,
        'top_one_row_error_share': float(by_row.iloc[0]/total),
        'top_five_rows_error_share': float(by_row.iloc[:5].sum()/total),
        'top_error_scaffold_train_count': int(len(scaffold_rows)),
        'top_error_scaffold_target_range': [float(y[scaffold_rows].min()), float(y[scaffold_rows].max())],
        'top_error_scaffold_error_share': float(frame.loc[frame.row.isin(scaffold_rows), 'squared_error'].sum()/total),
        'note': 'Diagnostics use true training targets only after held-out prediction; no metadata or concentrations enter the model.',
    }
    path = Path('v2_failure_modes.json')
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
