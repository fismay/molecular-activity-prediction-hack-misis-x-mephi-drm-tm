"""Reproduce the initial fixed-grid search from existing grouped OOF cache."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv_zero_hurdle import TRAIN_V2_SHA256


CACHE = Path('artifacts/zero_v2_validation.npz')
CACHE_SHA256 = '00231d2a535498a9217bd8eea4b9e66176c464ea68db63887e882612bbf3b896'
THRESHOLDS = (100, 250, 500, 1000, 2000, 5000, 10000)
WEIGHTS = (.25, .5, .75, 1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train_path = Path('data/kaggle/train_v2.csv')
    if sha(train_path) != TRAIN_V2_SHA256 or sha(CACHE) != CACHE_SHA256:
        raise ValueError('Unexpected train or cache snapshot')
    y = pd.read_csv(train_path)['SI VACV'].to_numpy(float)
    cache = np.load(CACHE)
    records = []
    for threshold in THRESHOLDS:
        for weight in WEIGHTS:
            all_errors = []
            suite_mse = {}
            for suite in ('42', '2026', '17'):
                errors = []
                for fold in range(5):
                    prefix = f'{suite}_{fold}'
                    va = cache[f'{prefix}_va']
                    p = cache[f'{prefix}_p_raw']
                    base = (1-p)*cache[f'{prefix}_q_source_stereo_only']
                    alternate = (1-p)*cache[f'{prefix}_q_source_stereo']
                    candidate = base + weight*(alternate-base)*(base < threshold)
                    errors.extend((y[va]-candidate)**2)
                suite_mse[suite] = float(np.mean(errors))
                all_errors.extend(errors)
            records.append({'threshold': threshold, 'weight': weight,
                            'pooled_mse': float(np.mean(all_errors)),
                            'suite_mse': suite_mse})
    winner = min(records, key=lambda row: row['pooled_mse'])
    if winner['threshold'] != 5000 or winner['weight'] != 1.0:
        raise ValueError('Initial selection changed')
    report = {'train_sha256': TRAIN_V2_SHA256, 'cache_sha256': CACHE_SHA256,
              'selection_metric': 'pooled full-target MSE over 15 canonical-group OOF folds',
              'selected': winner, 'grid': records,
              'note': 'The confirmation seed 48373 was not part of this screen.'}
    Path('v2_adaptive_gate_screen.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(winner, indent=2))


if __name__ == '__main__':
    main()
