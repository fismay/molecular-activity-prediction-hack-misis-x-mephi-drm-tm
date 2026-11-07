"""Audit published VACV structures against competition SMILES without using test labels."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem

from vacv_zero_hurdle import TRAIN_V2_SHA256


TRAIN = Path('data/kaggle/train_v2.csv')
TEST = Path('data/kaggle/test.csv')
PUBLIC = Path('data/curated/vacv_database.csv')
REPORT = Path('external_structural_match_audit.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError('Invalid SMILES')
    return (Chem.MolToSmiles(molecule, isomericSmiles=True),
            Chem.MolToSmiles(molecule, isomericSmiles=False))


def main():
    if sha(TRAIN) != TRAIN_V2_SHA256:
        raise ValueError('Unexpected Kaggle train_v2 snapshot')
    train = pd.read_csv(TRAIN, usecols=['SMILES', 'SI VACV'])
    test = pd.read_csv(TEST, usecols=['SMILES'])
    public = pd.read_csv(PUBLIC)
    if list(public) != ['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'DOI']:
        raise ValueError('Unexpected public database columns')
    train_keys = train.SMILES.map(keys).tolist()
    test_keys = test.SMILES.map(keys).tolist()
    source_keys = public.SMILES.map(keys).tolist()
    train_exact_keys = {key[0] for key in train_keys}
    train_achiral_keys = {key[1] for key in train_keys}
    train_exact = defaultdict(list)
    train_achiral = defaultdict(list)
    test_exact = defaultdict(list)
    test_achiral = defaultdict(list)
    for i, (exact, achiral) in enumerate(train_keys):
        train_exact[exact].append(i)
        train_achiral[achiral].append(i)
    for i, (exact, achiral) in enumerate(test_keys):
        test_exact[exact].append(i)
        test_achiral[achiral].append(i)
    details = []
    for i, (exact, achiral) in enumerate(source_keys):
        exact_test = test_exact[exact]
        achiral_test = test_achiral[achiral]
        exact_train = train_exact[exact]
        achiral_train = train_achiral[achiral]
        if not exact_test and not achiral_test and not exact_train and not achiral_train:
            continue
        source_si = float(public.SI.iloc[i])
        train_si = [float(train['SI VACV'].iloc[j]) for j in achiral_train]
        exact_train_si = [float(train['SI VACV'].iloc[j]) for j in exact_train]
        details.append({'doi': str(public.DOI.iloc[i]),
                        'exact_test_rows': len(exact_test),
                        'achiral_test_rows': len(achiral_test),
                        'exact_train_rows': len(exact_train),
                        'achiral_train_rows': len(achiral_train),
                        'source_si': source_si,
                        'train_si': train_si,
                        'exact_train_si': exact_train_si,
                        'max_relative_train_disagreement': max(
                            (abs(value-source_si)/max(source_si, 1e-12) for value in train_si),
                            default=None)})
    doi_counts = {}
    for doi in sorted(public.DOI.unique()):
        found = [row for row in details if row['doi'] == doi]
        doi_counts[doi] = {'source_rows': int((public.DOI == doi).sum()),
                           'exact_test_structures': sum(row['exact_test_rows'] > 0 for row in found),
                           'achiral_test_structures': sum(row['achiral_test_rows'] > 0 for row in found),
                           'achiral_train_structures': sum(row['achiral_train_rows'] > 0 for row in found),
                           'novel_test_structures': sum(row['achiral_test_rows'] > 0 and row['achiral_train_rows'] == 0 for row in found),
                           'exact_train_measurements': sum(row['exact_train_rows'] for row in found),
                           'exact_train_disagreements_gt_5pct': sum(
                               abs(value-row['source_si'])/max(row['source_si'], 1e-12) > .05
                               for row in found for value in row['exact_train_si'])}
    report = {'scope': 'SMILES-only structure matching; no test metadata, concentration, or target column read',
              'source_sha256': sha(PUBLIC), 'train_sha256': sha(TRAIN), 'test_sha256': sha(TEST),
              'source_rows': len(public), 'train_rows': len(train), 'test_rows': len(test),
              'doi_counts': doi_counts,
              'novel_exact_test_rows': len({j for exact, _ in source_keys if exact not in train_exact_keys for j in test_exact[exact]}),
              'novel_achiral_test_rows': len({j for _, achiral in source_keys if achiral not in train_achiral_keys for j in test_achiral[achiral]}),
              'source_train_disagreements_gt_5pct': sum(row['max_relative_train_disagreement'] is not None and row['max_relative_train_disagreement'] > .05 for row in details),
              'limitations': 'Structural identity does not prove assay equivalence. Achiral matching can merge stereoisomers; review the source before any override.'}
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
