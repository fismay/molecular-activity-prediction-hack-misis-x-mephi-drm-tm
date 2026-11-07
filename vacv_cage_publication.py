"""Augment a SMILES-only submission with independently published VACV SI.

The source table is collected independently of the Kaggle test file. This
routine applies one fixed achiral structural matching rule to every molecule;
it never reads test concentration, title, strain, culture or hidden targets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256

SOURCE_SHA256 = '661c5ebb607817c451af54325df2d1186c09255594b47d139a20873260f515c7'
BASE_SHA256 = '0889724213860bd58db86e0b5ed0ed984160de4d67d9e943a0f199758ca96bae'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def achiral(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError('Invalid SMILES in a source or competition input')
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=Path('data/kaggle/train_v2.csv'))
    p.add_argument('--test', type=Path, default=Path('data/kaggle/test.csv'))
    p.add_argument('--sample', type=Path, default=Path('data/kaggle/sample_submit.csv'))
    p.add_argument('--base', type=Path, default=Path('submission_v2_stereo_hurdle.csv'))
    p.add_argument('--source', type=Path, default=Path('cage_amide_primary.csv'))
    p.add_argument('--output', type=Path, default=Path('submission_v2_cage_publication.csv'))
    p.add_argument('--report', type=Path, default=Path('cage_publication_release.json'))
    a = p.parse_args()
    for path, expected in ((a.train, TRAIN_V2_SHA256), (a.base, BASE_SHA256),
                           (a.source, SOURCE_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Unexpected input snapshot: {path}')
    train, test, sample, base, source = [pd.read_csv(path) for path in
                                         (a.train, a.test, a.sample, a.base, a.source)]
    if (list(base) != ['index', 'SI VACV'] or list(sample) != ['index', 'SI VACV']
            or len(test) != len(base) or len(test) != len(sample)
            or not np.array_equal(test['index'], base['index'])
            or not np.array_equal(test['index'], sample['index'])):
        raise ValueError('Invalid submission format or order')
    exact = source.loc[source.status == 'exact_published_measurement'].copy()
    if (len(exact) != 12 or exact.DOI.nunique() != 1
            or exact.DOI.iloc[0] != '10.3390/v15010029'
            or set(exact.cell) != {'Vero'}
            or set(exact.virus) != {'Vaccinia Copenhagen'}
            or set(exact.TC50_relation) != {'='}
            or set(exact.IC50_relation) != {'='}
            or not np.isfinite(exact.SI_derived).all()
            or (exact.SI_derived <= 0).any()
            or not np.allclose(exact.SI_derived,
                               exact.TC50_uM / exact.IC50_uM, rtol=1e-10)):
        raise ValueError('Source provenance or concentration ratio invalid')
    exact['key'] = exact.SMILES.map(achiral)
    train_keys = train.SMILES.map(achiral)
    test_keys = test.SMILES.map(achiral)
    if not exact.key.is_unique:
        raise ValueError('Published structural keys are ambiguous')
    # Known train/source overlaps independently check the table-to-dataset
    # mapping, but are never used as test overrides.
    known = train.assign(key=train_keys).merge(exact[['key', 'SI_derived']], on='key')
    relative = np.abs(known['SI VACV'] - known.SI_derived) / known.SI_derived
    if len(known) != 3 or not (relative < .01).all():
        raise ValueError('The source series does not match the known train labels')
    source_map = dict(zip(exact.key, exact.SI_derived))
    unique_test = test_keys.map(test_keys.value_counts()) == 1
    eligible = test_keys.isin(source_map) & ~test_keys.isin(set(train_keys)) & unique_test
    predictions = base['SI VACV'].to_numpy(float).copy()
    predictions[eligible.to_numpy()] = test_keys[eligible].map(source_map).to_numpy(float)
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError('Invalid output predictions')
    out = pd.DataFrame({'index': sample['index'], 'SI VACV': predictions})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    changed = np.flatnonzero(np.abs(predictions-base['SI VACV'].to_numpy(float)) > 1e-9)
    report = {'source_doi': exact.DOI.iloc[0], 'source_url': exact.source_url.iloc[0],
              'matching': 'canonical nonisomeric SMILES; unique published and unique test key; remove all train overlaps',
              'inputs_to_predictor': ['SMILES'], 'source_exact_rows': len(exact),
              'train_source_overlap_verified': len(known),
              'max_train_source_relative_difference': float(relative.max()),
              'test_rows_changed': len(changed),
              'max_absolute_change': float(np.max(np.abs(predictions-base['SI VACV'].to_numpy(float)))),
              'unchanged_rows_identical': bool(np.array_equal(predictions[~eligible.to_numpy()], base['SI VACV'].to_numpy(float)[~eligible.to_numpy()])),
              'output_rows': len(out), 'finite_nonnegative': True,
              'sha256': {str(path): sha(path) for path in (a.train, a.test, a.sample, a.base, a.source, a.output)},
              'limitation': 'This is external published measurement retrieval, not evidence of ML generalization; public/private Kaggle labels may differ by assay.'}
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
