"""Screen unanimous-zero molecular scaffolds on grouped out-of-fold predictions."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from vacv_zero_hurdle import TRAIN_V2_SHA256


TRAIN = Path('data/kaggle/train_v2.csv')
TEST = Path('data/kaggle/test.csv')
CACHE = Path('artifacts/zero_v2_validation.npz')
BASE = Path('submission_v2_cidofovir_a549.csv')
REPORT = Path('zero_scaffold_screen.json')
EXPECTED = {
    CACHE: '00231d2a535498a9217bd8eea4b9e66176c464ea68db63887e882612bbf3b896',
    BASE: '9dbf100872cb99818b765e382815db1a7edd7dcc65125ab2b1a88d0acd5dc263',
    TRAIN: TRAIN_V2_SHA256,
    TEST: '1b0c6a014bbe556c42ef2277e867d2e1a647ad31c157a9cd81bbc7e262481abd',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scaffold(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError('Invalid SMILES')
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    return Chem.MolToSmiles(core if core.GetNumAtoms() else molecule, isomericSmiles=False)


def selected(reference_indices, query_indices, scaffolds, target, minimum):
    groups = defaultdict(list)
    for row in reference_indices:
        groups[scaffolds[row]].append(target[row])
    return np.array([len(groups[scaffolds[row]]) >= minimum
                     and max(groups[scaffolds[row]]) == 0 for row in query_indices])


def main():
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train = pd.read_csv(TRAIN, usecols=['SMILES', 'SI VACV'])
    test = pd.read_csv(TEST, usecols=['index', 'SMILES'])
    base = pd.read_csv(BASE)
    if list(base) != ['index', 'SI VACV'] or not np.array_equal(test['index'], base['index']):
        raise ValueError('Misaligned submission')
    y = train['SI VACV'].to_numpy(float)
    scaffolds = train.SMILES.map(scaffold).tolist()
    cache = np.load(CACHE)
    results = {}
    for minimum in (2, 3, 4, 5):
        errors = []
        reference_errors = []
        positives = 0
        affected = 0
        per_suite = {}
        for seed in (42, 2026, 17):
            suite_error = []
            suite_baseline = []
            for fold in range(5):
                prefix = f'{seed}_{fold}'
                train_rows = cache[prefix + '_tr'].astype(int)
                valid_rows = cache[prefix + '_va'].astype(int)
                baseline = cache[prefix + '_prediction_source_stereo_only__raw'].astype(float)
                if len(baseline) != len(valid_rows):
                    raise ValueError('Misaligned out-of-fold predictions')
                mask = selected(train_rows, valid_rows, scaffolds, y, minimum)
                candidate = baseline.copy()
                candidate[mask] = 0
                actual = y[valid_rows]
                err = (actual - candidate) ** 2
                old = (actual - baseline) ** 2
                errors.extend(err)
                reference_errors.extend(old)
                suite_error.extend(err)
                suite_baseline.extend(old)
                positives += int((actual[mask] > 0).sum())
                affected += int(mask.sum())
            per_suite[str(seed)] = {'candidate_mse': float(np.mean(suite_error)),
                                    'baseline_mse': float(np.mean(suite_baseline))}
        results[str(minimum)] = {'candidate_mse': float(np.mean(errors)),
                                 'baseline_mse': float(np.mean(reference_errors)),
                                 'relative_improvement': float(1 - np.mean(errors)/np.mean(reference_errors)),
                                 'overrides': affected, 'positive_labels_overridden': positives,
                                 'suites': per_suite}
    test_scaffolds = test.SMILES.map(scaffold).tolist()
    groups = defaultdict(list)
    for core, value in zip(scaffolds, y):
        groups[core].append(value)
    mask = np.array([len(groups[core]) >= 3 and max(groups[core]) == 0 for core in test_scaffolds])
    report = {'input_sha256': {str(path): expected for path, expected in EXPECTED.items()},
              'rule': 'nonisomeric Murcko scaffold, at least N train compounds, all observed SI=0',
              'validation': results,
              'test_smiles_only_audit': {'threshold_3_matches': int(mask.sum()),
                                         'max_baseline_prediction_among_matches': float(base.loc[mask, 'SI VACV'].max())},
              'decision': 'reject: no threshold reaches 1% full-MSE improvement and some zero-scaffold folds override positive SI',
              'limitation': 'Grouped folds reuse the same molecules; test labels and concentration fields were not read.'}
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
