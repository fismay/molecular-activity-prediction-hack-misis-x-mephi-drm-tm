"""Compare stored grouped OOF recipes with and without the largest-error row.

This is diagnostic only: it does not modify training, choose a test-row rule,
or use Kaggle test labels or concentration columns.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv_zero_hurdle import TRAIN_V2_SHA256


CACHE = Path('artifacts/zero_v2_validation.npz')
CACHE_SHA256 = '00231d2a535498a9217bd8eea4b9e66176c464ea68db63887e882612bbf3b896'
RECIPES = ('source_stereo_only__raw', 'source_stereo__raw', 'neighbors__raw')
SUITES = ('42', '2026', '17')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train_path = Path('data/kaggle/train_v2.csv')
    if sha(train_path) != TRAIN_V2_SHA256 or sha(CACHE) != CACHE_SHA256:
        raise ValueError('Unexpected train or validation cache snapshot')
    y = pd.read_csv(train_path)['SI VACV'].to_numpy(float)
    cache = np.load(CACHE)
    frames = {}
    for recipe in RECIPES:
        rows = []
        for suite in SUITES:
            for fold in range(5):
                prefix = f'{suite}_{fold}'
                indices = cache[f'{prefix}_va']
                pred = cache[f'{prefix}_prediction_{recipe}']
                if len(indices) != len(pred):
                    raise ValueError('Misaligned cached fold')
                rows.extend((suite, int(index), float((y[index]-estimate)**2))
                            for index, estimate in zip(indices, pred))
        frames[recipe] = pd.DataFrame(rows, columns=['suite', 'row', 'error'])
    reference = frames[RECIPES[0]]
    top_row = int(reference.groupby('row').error.sum().idxmax())
    report = {'train_sha256': TRAIN_V2_SHA256, 'cache_sha256': CACHE_SHA256,
              'folds': 15, 'heldout_observations': len(reference),
              'top_error_training_row': top_row, 'top_error_training_si': float(y[top_row]),
              'top_row_selection': 'largest aggregated squared OOF error of the existing confirmed recipe',
              'recipes': {},
              'interpretation': 'Excluding the largest-error row is a diagnostic sensitivity analysis, not a legitimate replacement for full MSE model selection. Folds reuse the same train rows.'}
    for recipe, frame in frames.items():
        if not np.array_equal(frame.row.to_numpy(), reference.row.to_numpy()):
            raise ValueError('Recipes have different heldout rows or order')
        report['recipes'][recipe] = {
            'full_pooled_mse': float(frame.error.mean()),
            'without_top_row_mse': float(frame.loc[frame.row != top_row, 'error'].mean()),
            'below_1000_si_mse': float(frame.loc[y[frame.row.to_numpy()] < 1000, 'error'].mean()),
            'top_row_error_share': float(frame.loc[frame.row == top_row, 'error'].sum()/frame.error.sum()),
            'suite_full_mse': {suite: float(frame.loc[frame.suite == suite, 'error'].mean())
                               for suite in SUITES},
        }
    path = Path('v2_outlier_sensitivity.json')
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
