"""Test a high-SI amplification hypothesis on saved grouped OOF predictions.

The screen is diagnostic: the same source molecule can recur across folds.
No Kaggle test target or concentration column is used.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv_zero_hurdle import TRAIN_V2_SHA256


TRAIN = Path('data/kaggle/train_v2.csv')
CACHES = {
    'screen': (Path('artifacts/zero_v2_validation.npz'),
               [f'{seed}_{fold}' for seed in (42, 2026, 17) for fold in range(5)]),
    'confirmation': (Path('artifacts/zero_v2_confirmation.npz'),
                     [f'confirmation67121_{fold}' for fold in range(5)]),
}
THRESHOLD = 5000.0
FACTOR = 2.0


def evaluate(path, prefixes, labels):
    cache = np.load(path)
    ids = np.concatenate([cache[f'{prefix}_va'].astype(int) for prefix in prefixes])
    baseline = np.concatenate([
        cache[f'{prefix}_prediction_source_stereo_only__raw'] for prefix in prefixes
    ])
    target = labels[ids]
    proposed = np.where(baseline >= THRESHOLD, FACTOR * baseline, baseline)
    worst_row = int(ids[np.argmax((target - baseline) ** 2)])
    retained = ids != worst_row
    per_row = []
    for row in np.unique(ids[baseline >= THRESHOLD]):
        mask = ids == row
        per_row.append({'train_row': int(row), 'true_si': float(labels[row]),
                        'mean_baseline': float(baseline[mask].mean()),
                        'visits': int(mask.sum())})
    return {
        'cache_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'heldout_observations': len(ids),
        'affected_observations': int((baseline >= THRESHOLD).sum()),
        'affected_unique_rows': per_row,
        'baseline_mse': float(np.mean((target - baseline) ** 2)),
        'amplified_mse': float(np.mean((target - proposed) ** 2)),
        'worst_baseline_row': worst_row,
        'baseline_mse_excluding_worst_row': float(np.mean((target[retained] - baseline[retained]) ** 2)),
        'amplified_mse_excluding_worst_row': float(np.mean((target[retained] - proposed[retained]) ** 2)),
    }


def main():
    if hashlib.sha256(TRAIN.read_bytes()).hexdigest() != TRAIN_V2_SHA256:
        raise ValueError('Unexpected training snapshot')
    labels = pd.read_csv(TRAIN)['SI VACV'].to_numpy(float)
    suites = {name: evaluate(*args, labels) for name, args in CACHES.items()}
    assert all(s['worst_baseline_row'] == 14 for s in suites.values())
    assert all(s['amplified_mse'] < s['baseline_mse'] for s in suites.values())
    assert all(s['amplified_mse_excluding_worst_row'] > s['baseline_mse_excluding_worst_row']
               for s in suites.values())
    report = {
        'hypothesis': 'Multiply saved SMILES-only baseline predictions >=5000 by 2',
        'train_sha256': TRAIN_V2_SHA256,
        'suites': suites,
        'decision': 'reject',
        'reason': 'Aggregate gain depends on one SI=36066.667 molecule; excluding it for diagnosis reverses the result. This is not a robust generalization signal.',
        'note': 'Excluding a row is diagnostic only; all rows remain in the actual competition metric. Confirmation folds reuse the same small dataset.',
    }
    Path('tail_amplification_screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'decision': report['decision'], 'suites': suites}, indent=2))


if __name__ == '__main__':
    main()
