"""Confirm a frozen SI=5000 prediction gate on new grouped folds.

The threshold was selected using cached 15-fold OOF predictions. The new
GroupKFold seed below is fixed before running this independent computation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
from vacv_zero_hurdle import (TRAIN_V2_SHA256, conditional_components,
                              external_source, fit, probabilities)


THRESHOLD = 5000.0
SEED = 48373
SOURCE = Path('artifacts/external/bornyl_amides_expanded_source.csv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train_path = Path('data/kaggle/train_v2.csv')
    if sha(train_path) != TRAIN_V2_SHA256:
        raise ValueError('Unexpected train snapshot')
    data = pd.read_csv(train_path)
    smiles = data.SMILES
    y = data['SI VACV'].to_numpy(float)
    x, groups = featurize(smiles)
    syntax = featurize_syntax(smiles)
    external = external_source(SOURCE, smiles)
    splits = GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(x, y, groups)
    folds = []
    for number, (tr, va) in enumerate(splits):
        model = fit(smiles.iloc[tr], y[tr], x[tr], syntax[tr])
        p = probabilities(model, smiles.iloc[va], syntax[va])['raw']
        q = conditional_components(model, smiles.iloc[va], external, x[va])
        base = (1-p) * q['source_stereo_only']
        alternate = (1-p) * q['source_stereo']
        gated = np.where(base < THRESHOLD, alternate, base)
        folds.append({'fold': number, 'validation_rows': len(va),
                      'gated_rows': int(np.count_nonzero(base < THRESHOLD)),
                      'baseline_mse': float(np.mean((y[va]-base)**2)),
                      'candidate_mse': float(np.mean((y[va]-gated)**2)),
                      'baseline_sse': float(np.sum((y[va]-base)**2)),
                      'candidate_sse': float(np.sum((y[va]-gated)**2)),
                      'below_1000_baseline_sse': float(np.sum((y[va][y[va]<1000]-base[y[va]<1000])**2)),
                      'below_1000_candidate_sse': float(np.sum((y[va][y[va]<1000]-gated[y[va]<1000])**2))})
        print('fold', number, round(folds[-1]['baseline_mse']),
              round(folds[-1]['candidate_mse']), flush=True)
    base_sse = sum(f['baseline_sse'] for f in folds)
    gate_sse = sum(f['candidate_sse'] for f in folds)
    count = sum(f['validation_rows'] for f in folds)
    report = {'train_sha256': TRAIN_V2_SHA256, 'source_sha256': sha(SOURCE),
              'seed': SEED, 'groups': 'canonical molecule', 'fold_count': 5,
              'threshold_selected_on_prior_15_folds': THRESHOLD,
              'rule': 'source_stereo prediction when baseline predicted SI < 5000; otherwise source_stereo_only',
              'baseline_pooled_mse': base_sse/count,
              'candidate_pooled_mse': gate_sse/count,
              'relative_improvement': (base_sse-gate_sse)/base_sse,
              'all_folds_improved': all(f['candidate_mse'] < f['baseline_mse'] for f in folds),
              'folds': folds,
              'limitation': 'New grouped folds reuse the same small training set and are not an independent external cohort.'}
    Path('v2_adaptive_gate_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'folds'}, indent=2))


if __name__ == '__main__':
    main()
