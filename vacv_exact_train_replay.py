"""Replace predictions for unique exact train/test stereostructure matches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256


BASE_SHA256 = '9dbf100872cb99818b765e382815db1a7edd7dcc65125ab2b1a88d0acd5dc263'
TEST_SHA256 = '1b0c6a014bbe556c42ef2277e867d2e1a647ad31c157a9cd81bbc7e262481abd'
SAMPLE_SHA256 = '3eea4d23f5700361c09da24cd45be05c5dbbecf76cc29b022c3676835606884c'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_key(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/kaggle/train_v2.csv'))
    parser.add_argument('--test', type=Path, default=Path('data/kaggle/test.csv'))
    parser.add_argument('--sample', type=Path, default=Path('data/kaggle/sample_submit.csv'))
    parser.add_argument('--base', type=Path, default=Path('submission_v2_cidofovir_a549.csv'))
    parser.add_argument('--output', type=Path, default=Path('submission_v2_exact_train_replay.csv'))
    parser.add_argument('--report', type=Path, default=Path('exact_train_replay_release.json'))
    args = parser.parse_args()
    for path, expected in ((args.train, TRAIN_V2_SHA256), (args.test, TEST_SHA256),
                           (args.sample, SAMPLE_SHA256), (args.base, BASE_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train = pd.read_csv(args.train, usecols=['SMILES', 'SI VACV', 'strain'])
    test = pd.read_csv(args.test, usecols=['index', 'SMILES'])
    sample = pd.read_csv(args.sample)
    base = pd.read_csv(args.base)
    if (list(sample) != ['index', 'SI VACV'] or list(base) != list(sample)
            or len(test) != 597 or len(test) != len(sample) or len(base) != len(sample)
            or not np.array_equal(test['index'], sample['index'])
            or not np.array_equal(test['index'], base['index'])):
        raise ValueError('Submission layout or order changed')
    train['key'] = train.SMILES.map(exact_key)
    test['key'] = test.SMILES.map(exact_key)
    matched = test.key.isin(set(train.key))
    if matched.sum() != 29 or not test.loc[matched, 'key'].is_unique:
        raise ValueError('Expected 29 unique exact train/test stereostructures')
    matched_train = train.loc[train.key.isin(set(test.loc[matched, 'key']))]
    if not matched_train.key.is_unique:
        raise ValueError('Matched train stereostructure has multiple assay labels')
    # Reject source labels annotated only as another virus. Test metadata is not read.
    compatible_train = matched_train.loc[matched_train.strain.str.contains('vaccinia', case=False, na=False)]
    if len(compatible_train) != 27:
        raise ValueError('Expected 27 VACV-annotated exact source labels')
    source = dict(zip(compatible_train.key, compatible_train['SI VACV']))
    eligible = test.key.isin(source)
    prediction = base['SI VACV'].to_numpy(float).copy()
    prediction[eligible.to_numpy()] = test.loc[eligible, 'key'].map(source).to_numpy(float)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid prediction')
    pd.DataFrame({'index': sample['index'], 'SI VACV': prediction}).to_csv(args.output, index=False)
    change = prediction - base['SI VACV'].to_numpy(float)
    report = {'scope': 'Exact isomeric canonical SMILES train lookup; test SMILES only',
              'train_sha256': sha(args.train), 'test_sha256': sha(args.test),
              'sample_sha256': sha(args.sample), 'base_sha256': sha(args.base),
              'output_sha256': sha(args.output), 'exact_matches': int(matched.sum()),
              'vacv_annotated_matches': int(eligible.sum()),
              'changed_rows': int(np.count_nonzero(change)),
              'max_absolute_change': float(np.max(abs(change))),
              'squared_change_sum': float(np.sum(change**2)),
              'caveat': 'Same structure need not imply identical measurement in a different assay; grouped CV cannot score exact train/test duplicates.'}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
