"""Produce the fixed SI=5000 adaptive structural recipe with cage measurements.

Run run_current_best.py first to create and verify the model and baseline CSV.
Only SMILES enter the predictor. Publication matching is uniform and automatic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from vacv_cage_publication import SOURCE_SHA256, achiral
from vacv_zero_hurdle import TRAIN_V2_SHA256, conditional_components, predict, probabilities


BASE_SHA256 = '0889724213860bd58db86e0b5ed0ed984160de4d67d9e943a0f199758ca96bae'
CONFIRMED_SHA256 = '8a0d40cf37f6a4a3a6d99815e6961b7d89f9493c2e031880b506590dab0d2bf4'
TEST_SHA256 = '1b0c6a014bbe556c42ef2277e867d2e1a647ad31c157a9cd81bbc7e262481abd'
SAMPLE_SHA256 = '3eea4d23f5700361c09da24cd45be05c5dbbecf76cc29b022c3676835606884c'
THRESHOLD = 5000.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('data/kaggle/train_v2.csv'))
    p.add_argument('--test', type=Path, default=Path('data/kaggle/test.csv'))
    p.add_argument('--sample', type=Path, default=Path('data/kaggle/sample_submit.csv'))
    p.add_argument('--base', type=Path, default=Path('submission_v2_stereo_hurdle.csv'))
    p.add_argument('--confirmed', type=Path, default=Path('submission_v2_cage_publication.csv'))
    p.add_argument('--source', type=Path, default=Path('cage_amide_primary.csv'))
    p.add_argument('--model', type=Path, default=Path('artifacts/zero_v2_hurdle_nn.joblib'))
    p.add_argument('--output', type=Path, default=Path('submission_v2_adaptive_gate.csv'))
    p.add_argument('--report', type=Path, default=Path('v2_adaptive_gate_release.json'))
    a = p.parse_args()
    snapshots = ((a.train, TRAIN_V2_SHA256), (a.test, TEST_SHA256),
                 (a.sample, SAMPLE_SHA256), (a.base, BASE_SHA256),
                 (a.confirmed, CONFIRMED_SHA256), (a.source, SOURCE_SHA256))
    for path, expected in snapshots:
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train, test, sample, base_csv, confirmed, source = [pd.read_csv(path) for path in
                                                        (a.train, a.test, a.sample,
                                                         a.base, a.confirmed, a.source)]
    if (list(base_csv) != ['index', 'SI VACV'] or list(confirmed) != ['index', 'SI VACV']
            or list(sample) != ['index', 'SI VACV'] or len(test) != 597
            or not test['index'].is_unique or not np.array_equal(test['index'], sample['index'])
            or not np.array_equal(test['index'], base_csv['index'])
            or not np.array_equal(test['index'], confirmed['index'])):
        raise ValueError('Submission format or order invalid')
    model = joblib.load(a.model)
    if (model.get('inputs') != ['SMILES'] or model.get('target_version') != 'train_v2'
            or model.get('calibration') != 'raw'
            or model.get('conditional') != 'source_stereo_only'):
        raise ValueError('Wrong trained model')
    smiles = test[['SMILES']].SMILES
    baseline = predict(model, smiles)
    if not np.allclose(baseline, base_csv['SI VACV'], rtol=0, atol=1e-8):
        raise ValueError('Saved model does not reproduce confirmed baseline')
    probability = probabilities(model, smiles)['raw']
    conditional = conditional_components(model, smiles, model['external'])
    alternative = (1-probability) * conditional['source_stereo']
    gated = np.where(baseline < THRESHOLD, alternative, baseline)
    exact = source.loc[source.status == 'exact_published_measurement'].copy()
    if (len(exact) != 12 or exact.DOI.nunique() != 1
            or exact.DOI.iloc[0] != '10.3390/v15010029'
            or set(exact.cell) != {'Vero'}
            or set(exact.virus) != {'Vaccinia Copenhagen'}
            or not np.allclose(exact.SI_derived,
                               exact.TC50_uM/exact.IC50_uM, rtol=1e-10)):
        raise ValueError('Cage publication validation failed')
    exact['key'] = exact.SMILES.map(achiral)
    train_keys = train.SMILES.map(achiral)
    test_keys = smiles.map(achiral)
    if not exact.key.is_unique:
        raise ValueError('Ambiguous publication structures')
    source_map = dict(zip(exact.key, exact.SI_derived))
    eligible = (test_keys.isin(source_map) & ~test_keys.isin(set(train_keys))
                & (test_keys.map(test_keys.value_counts()) == 1))
    if eligible.sum() != 8:
        raise ValueError('Unexpected publication coverage')
    # Verify the previously confirmed publication layer before applying it to
    # the new model prediction. This prevents an accidental change of source.
    check = baseline.copy()
    check[eligible.to_numpy()] = test_keys[eligible].map(source_map).to_numpy(float)
    if not np.allclose(check, confirmed['SI VACV'], rtol=0, atol=1e-8):
        raise ValueError('Publication layer differs from confirmed submission')
    gated[eligible.to_numpy()] = test_keys[eligible].map(source_map).to_numpy(float)
    if not np.isfinite(gated).all() or (gated < 0).any():
        raise ValueError('Invalid final predictions')
    output = pd.DataFrame({'index': sample['index'], 'SI VACV': gated})
    output.to_csv(a.output, index=False)
    report = {'model_rule': f'Use source_stereo if baseline predicted SI < {THRESHOLD}; otherwise use source_stereo_only',
              'source_rule': 'Published exact unique achiral match, not in train',
              'inputs_to_predictor': ['SMILES'], 'rows': len(output),
              'gated_rows_before_source': int(np.count_nonzero(baseline < THRESHOLD)),
              'publication_rows': int(eligible.sum()),
              'changed_from_confirmed': int(np.count_nonzero(np.abs(gated-confirmed['SI VACV'].to_numpy(float)) > 1e-8)),
              'finite_nonnegative': True,
              'sha256': {str(path): sha(path) for path in
                         (a.train, a.test, a.sample, a.base, a.confirmed,
                          a.source, a.model, a.output)},
              'limitations': 'Grouped CV reuses the same train; hidden Kaggle targets unavailable.'}
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
