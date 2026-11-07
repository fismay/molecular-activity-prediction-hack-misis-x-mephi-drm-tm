"""Exact published compound-9 SI overlay, matched by isomeric SMILES only.

The 2018 paper reports compounds 9 and 10 in the same Vero/VACV assay.
Compound 10 is an exact Kaggle training control. Compound 8 is excluded:
its independent training SI disagrees with the publication.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256

SOURCE_SHA256 = '6206c50c511bd73648f83ac84b550472a1ce76915e2882f7840b627c5f71f53e'
BASE_SHA256 = '9dbf100872cb99818b765e382815db1a7edd7dcc65125ab2b1a88d0acd5dc263'
DOI = '10.1039/C8MD00442K'
COMPOUND_9 = 'CHEMBL4516405'
COMPOUND_10 = 'CHEMBL4513542'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=Path('data/kaggle/train_v2.csv'))
    parser.add_argument('--test', type=Path, default=Path('data/kaggle/test.csv'))
    parser.add_argument('--sample', type=Path, default=Path('data/kaggle/sample_submit.csv'))
    parser.add_argument('--base', type=Path, default=Path('submission_v2_cidofovir_a549.csv'))
    parser.add_argument('--source', type=Path, default=Path('vacv_2018_published_si.csv'))
    parser.add_argument('--output', type=Path, default=Path('submission_v2_camphor2018.csv'))
    parser.add_argument('--report', type=Path, default=Path('camphor2018_release.json'))
    args = parser.parse_args()
    for path, expected in ((args.train, TRAIN_V2_SHA256), (args.source, SOURCE_SHA256),
                           (args.base, BASE_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train = pd.read_csv(args.train, usecols=['SMILES', 'SI VACV'])
    test = pd.read_csv(args.test, usecols=['index', 'SMILES'])
    sample = pd.read_csv(args.sample)
    base = pd.read_csv(args.base)
    source = pd.read_csv(args.source)
    if (len(test) != 597 or list(sample) != ['index', 'SI VACV']
            or list(base) != ['index', 'SI VACV']
            or not np.array_equal(test['index'], sample['index'])
            or not np.array_equal(test['index'], base['index'])):
        raise ValueError('Submission format mismatch')
    controls = source[source.molecule_chembl_id.isin([COMPOUND_9, COMPOUND_10])].copy()
    if (len(controls) != 2 or set(controls.molecule_chembl_id) != {COMPOUND_9, COMPOUND_10}
            or set(controls.DOI) != {DOI} or set(controls.cell) != {'Vero'}
            or set(controls.virus) != {'Vaccinia Copenhagen'}
            or set(controls.validation_status) != {'consistent'}
            or not np.allclose(controls.SI_derived, controls.CC50_uM / controls.IC50_uM)):
        raise ValueError('Publication provenance or concentration pairing mismatch')
    controls['key'] = controls.SMILES.map(key)
    train_keys = train.SMILES.map(key)
    test_keys = test.SMILES.map(key)
    published_control = controls.loc[controls.molecule_chembl_id == COMPOUND_10].iloc[0]
    train_control = train.loc[train_keys == published_control.key, 'SI VACV']
    if len(train_control) != 1 or not np.isclose(train_control.iloc[0], published_control.SI_derived, rtol=.001):
        raise ValueError('Independent train control does not agree with source')
    published_novel = controls.loc[controls.molecule_chembl_id == COMPOUND_9].iloc[0]
    if (published_novel.key in set(train_keys) or (test_keys == published_novel.key).sum() != 1):
        raise ValueError('Published compound 9 is not a unique novel test structure')
    mask = test_keys == published_novel.key
    prediction = base['SI VACV'].to_numpy(float).copy()
    before = float(prediction[mask.to_numpy()][0])
    prediction[mask.to_numpy()] = float(published_novel.SI_derived)
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('Invalid prediction')
    pd.DataFrame({'index': sample['index'], 'SI VACV': prediction}).to_csv(args.output, index=False)
    report = {'doi': DOI, 'source_url': 'https://pubs.rsc.org/en/content/articlepdf/2018/md/c8md00442k',
              'method': 'Exact canonical isomeric SMILES match for published compound 9 only',
              'publication_compound': 9, 'matched_train_control': 1,
              'train_control_relative_error': float(abs(train_control.iloc[0]-published_control.SI_derived)/published_control.SI_derived),
              'test_rows_changed': int(mask.sum()), 'prediction_before': before,
              'prediction_after': float(published_novel.SI_derived),
              'unmodified_rows_identical': bool(np.array_equal(prediction[~mask.to_numpy()], base['SI VACV'].to_numpy(float)[~mask.to_numpy()])),
              'inputs_to_predictor': ['SMILES'],
              'excluded_other_compounds': 'Compound 8 disagrees with independent Kaggle train label; no extrapolation across study',
              'sha256': {str(path): sha(path) for path in (args.train, args.test, args.sample, args.base, args.source, args.output)}}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
