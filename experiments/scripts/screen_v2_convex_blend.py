"""Audit fixed convex blends of two SMILES-only SI recipes on grouped OOF data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vacv_zero_hurdle import TRAIN_V2_SHA256


TRAIN = Path('data/kaggle/train_v2.csv')
SCREEN = Path('artifacts/zero_v2_validation.npz')
CONFIRM = Path('artifacts/zero_v2_confirmation.npz')
SCREEN_SHA = '00231d2a535498a9217bd8eea4b9e66176c464ea68db63887e882612bbf3b896'
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(cache, prefixes, y, weight):
    errors, rows, folds = [], [], []
    for prefix in prefixes:
        va = cache[f'{prefix}_va']
        p = cache[f'{prefix}_p_raw']
        base = (1-p)*cache[f'{prefix}_q_source_stereo_only']
        alternative = (1-p)*cache[f'{prefix}_q_source_stereo']
        pred = base + weight*(alternative-base)
        squared = (y[va]-pred)**2
        errors.extend(squared.tolist())
        rows.extend(va.tolist())
        folds.append({'fold': prefix, 'mse': float(squared.mean())})
    errors, rows = np.asarray(errors), np.asarray(rows)
    unique, inv = np.unique(rows, return_inverse=True)
    row_sse = np.bincount(inv, weights=errors)
    worst = int(unique[np.argmax(row_sse)])
    return {'weight': weight, 'pooled_mse': float(errors.mean()),
            'mse_without_worst_row': float(errors[rows != worst].mean()),
            'worst_row': worst, 'worst_row_share': float(row_sse.max()/errors.sum()),
            'mse_si_below_1000': float(errors[y[rows] < 1000].mean()),
            'folds': folds}


def main():
    if sha(TRAIN) != TRAIN_V2_SHA256 or sha(SCREEN) != SCREEN_SHA:
        raise ValueError('Unexpected training data or screening cache')
    y = pd.read_csv(TRAIN)['SI VACV'].to_numpy(float)
    screen = np.load(SCREEN)
    confirm = np.load(CONFIRM)
    screen_prefixes = [f'{seed}_{fold}' for seed in (42, 2026, 17) for fold in range(5)]
    confirm_prefixes = [f'confirmation67121_{fold}' for fold in range(5)]
    screened = [evaluate(screen, screen_prefixes, y, w) for w in WEIGHTS]
    chosen = min(screened, key=lambda r: r['pooled_mse'])['weight']
    confirmed = [evaluate(confirm, confirm_prefixes, y, w) for w in sorted({0.0, 0.2, chosen})]
    report = {'train_sha256': sha(TRAIN), 'screen_cache_sha256': sha(SCREEN),
              'confirm_cache_sha256': sha(CONFIRM),
              'selection': 'minimum full pooled MSE on 15 grouped OOF folds',
              'selected_weight': chosen, 'screen': screened, 'confirmation': confirmed,
              'note': 'Weight 0.2 is shown diagnostically, not selected. Confirmation uses existing seed 67121, unseen by this blend screen but not an independent dataset. No Kaggle labels or forbidden concentration features.'}
    Path('v2_convex_blend_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'chosen': chosen, 'screen_mse': screened[0]['pooled_mse'],
                      'selected_screen_mse': next(r['pooled_mse'] for r in screened if r['weight']==chosen),
                      'confirmation': [{k: v for k, v in row.items() if k != 'folds'} for row in confirmed]}, indent=2))


if __name__ == '__main__':
    main()
