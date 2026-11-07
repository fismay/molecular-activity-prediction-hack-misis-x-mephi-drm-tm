"""Apply exact published A549 VACV SI to identical, unseen stereostructures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256


SOURCE = Path('chembl_cidofovir_curated.csv')
SOURCE_SHA = '6918e12eee9ccf2ec00a1d09010ab88e1044725ce197d18997779a844ea5b998'
BASE = Path('submission_v2_cage_publication.csv')
BASE_SHA = '8a0d40cf37f6a4a3a6d99815e6961b7d89f9493c2e031880b506590dab0d2bf4'
TRAIN = Path('data/kaggle/train_v2.csv')
TEST = Path('data/kaggle/test.csv')
SAMPLE = Path('data/kaggle/sample_submit.csv')
OUTPUT = Path('submission_v2_cidofovir_a549.csv')
REPORT = Path('cidofovir_a549_release.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=TRAIN)
    parser.add_argument('--test', type=Path, default=TEST)
    parser.add_argument('--sample', type=Path, default=SAMPLE)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    for path, expected in ((SOURCE, SOURCE_SHA), (args.base, BASE_SHA), (args.train, TRAIN_V2_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    source = pd.read_csv(SOURCE)
    source = source.loc[source.cell.eq('A549')].copy()
    if (len(source) != 18 or source.DOI.nunique() != 1
            or source.DOI.iloc[0] != '10.1016/j.ejmech.2023.115601'
            or set(source.validation_status) != {'consistent'}
            or set(source.ic50_assay_chembl_id) != {'CHEMBL5625832'}
            or set(source.cc50_assay_chembl_id) != {'CHEMBL5625833'}
            or set(source.si_assay_chembl_id) != {'CHEMBL5625834'}
            or not (source.ic50_standard_relation.eq('=') & source.cc50_standard_relation.eq('=')
                    & source.si_standard_relation.eq('=')).all()
            or not np.isfinite(source[['SI', 'SI_derived']].to_numpy(float)).all()
            or not np.allclose(source.SI, source.SI_derived, rtol=.05)):
        raise ValueError('A549 publication provenance or measurements changed')
    source['key'] = source.SMILES.map(key)
    if not source.key.is_unique:
        raise ValueError('Ambiguous published stereostructure')
    train = pd.read_csv(args.train, usecols=['SMILES', 'SI VACV'])
    test = pd.read_csv(args.test, usecols=['index', 'SMILES'])
    sample = pd.read_csv(args.sample)
    base = pd.read_csv(args.base)
    if (list(base) != ['index', 'SI VACV'] or list(sample) != ['index', 'SI VACV']
            or len(test) != len(sample) or len(test) != len(base)
            or not np.array_equal(test['index'], sample['index'])
            or not np.array_equal(test['index'], base['index'])):
        raise ValueError('Competition submission layout changed')
    train_keys = train.SMILES.map(key)
    test_keys = test.SMILES.map(key)
    known = train.assign(key=train_keys).merge(source[['key', 'SI']], on='key')
    relative = abs(known['SI VACV']-known.SI)/known.SI
    if len(known) != 2 or not (relative < .01).all():
        raise ValueError('A549 source does not match two train controls')
    source_map = dict(zip(source.key, source.SI))
    eligible = test_keys.isin(source_map) & ~test_keys.isin(set(train_keys))
    if int(eligible.sum()) != 16 or (test_keys[eligible].value_counts() != 1).any():
        raise ValueError('Unexpected novel test match count or duplicates')
    pred = base['SI VACV'].to_numpy(float).copy()
    pred[eligible.to_numpy()] = test_keys[eligible].map(source_map).to_numpy(float)
    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError('Nonfinite or negative prediction')
    pd.DataFrame({'index': sample['index'], 'SI VACV': pred}).to_csv(args.output, index=False)
    report = {'doi': source.DOI.iloc[0], 'source_sha256': sha(SOURCE),
              'train_sha256': sha(args.train), 'test_sha256': sha(args.test), 'base_sha256': sha(args.base),
              'output_sha256': sha(args.output), 'source_cell': 'A549', 'source_rows': len(source),
              'exact_train_controls': len(known), 'max_control_relative_disagreement': float(relative.max()),
              'novel_test_rows_replaced': int(eligible.sum()),
              'maximum_absolute_change': float(np.max(abs(pred-base['SI VACV'].to_numpy(float)))),
              'input_features': ['SMILES'],
              'caveat': 'Published A549 assay retrieval; train controls support matching but do not prove unseen test labels use the same assay.'}
    REPORT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
