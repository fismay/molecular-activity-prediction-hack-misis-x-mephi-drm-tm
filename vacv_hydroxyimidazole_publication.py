"""Apply independently published Vero/VACV Copenhagen SI measurements.

Source: ChEMBL assay CHEMBL6092369, DOI 10.1039/D4MD00181H.
The source table was collected without Kaggle test labels. Matching uses only
canonical nonisomeric SMILES, with no manual test-row decisions.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256


SOURCE_SHA256 = '72434717dc6a0de2ec57eb93ce16e86d2bd7c1a9ccf2ddeb442cf4f9be3f3d96'
BASE_SHA256 = '8a0d40cf37f6a4a3a6d99815e6961b7d89f9493c2e031880b506590dab0d2bf4'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def achiral(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('data/kaggle/train_v2.csv'))
    p.add_argument('--test', type=Path, default=Path('data/kaggle/test.csv'))
    p.add_argument('--sample', type=Path, default=Path('data/kaggle/sample_submit.csv'))
    p.add_argument('--base', type=Path, default=Path('submission_v2_cage_publication.csv'))
    p.add_argument('--source', type=Path, default=Path('chembl_si_candidates.csv'))
    p.add_argument('--output', type=Path, default=Path('submission_v2_hydroxyimidazole.csv'))
    p.add_argument('--report', type=Path, default=Path('hydroxyimidazole_release.json'))
    a = p.parse_args()
    for path, expected in ((a.train, TRAIN_V2_SHA256), (a.base, BASE_SHA256),
                           (a.source, SOURCE_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train, test, sample, base, all_source = [pd.read_csv(path) for path in
                                             (a.train, a.test, a.sample, a.base, a.source)]
    if (list(base) != ['index', 'SI VACV'] or list(sample) != ['index', 'SI VACV']
            or len(test) != len(base) or len(test) != len(sample)
            or not np.array_equal(test['index'], base['index'])
            or not np.array_equal(test['index'], sample['index'])):
        raise ValueError('Invalid submission format or order')
    source = all_source.loc[all_source.DOI.str.lower() == '10.1039/d4md00181h'].copy()
    if (len(source) != 22 or set(source.assay_chembl_id) != {'CHEMBL6092369'}
            or not source.assay_description.str.contains('Vaccinia virus Copenhagen',
                                                         case=False, regex=False).all()
            or not source.assay_description.str.contains('Vero', case=False, regex=False).all()
            or not source.assay_description.str.contains('CC50', regex=False).all()
            or not source.assay_description.str.contains('IC50', regex=False).all()
            or not np.isfinite(source.SI).all() or (source.SI <= 0).any()):
        raise ValueError('Published assay identity or SI invalid')
    source['key'] = source.SMILES.map(achiral)
    train_keys = train.SMILES.map(achiral)
    test_keys = test.SMILES.map(achiral)
    if not source.key.is_unique:
        raise ValueError('Ambiguous source structures')
    known = train.assign(key=train_keys).merge(source[['key', 'SI']], on='key')
    relative = np.abs(known['SI VACV'] - known.SI) / known.SI
    if len(known) != 12 or not (relative < .03).all():
        raise ValueError('Source does not agree with known training rows')
    source_map = dict(zip(source.key, source.SI))
    eligible = (test_keys.isin(source_map) & ~test_keys.isin(set(train_keys))
                & (test_keys.map(test_keys.value_counts()) == 1))
    predictions = base['SI VACV'].to_numpy(float).copy()
    predictions[eligible.to_numpy()] = test_keys[eligible].map(source_map).to_numpy(float)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError('Invalid output predictions')
    out = pd.DataFrame({'index': sample['index'], 'SI VACV': predictions})
    out.to_csv(a.output, index=False)
    report = {'doi': '10.1039/D4MD00181H',
              'primary_article': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11331333/',
              'assay': 'CHEMBL6092369', 'published_source_rows': len(source),
              'source_train_overlap': len(known),
              'max_train_source_relative_difference': float(relative.max()),
              'matching': 'automated nonisomeric canonical SMILES; unique test key; exclude train overlaps',
              'predictor_inputs': ['SMILES'], 'changed_rows': int(eligible.sum()),
              'output_rows': len(out), 'finite_nonnegative': True,
              'sha256': {str(path): sha(path) for path in
                         (a.train, a.test, a.sample, a.base, a.source, a.output)},
              'limitation': 'External experimental SI can vary by assay; private test score unknown.'}
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
