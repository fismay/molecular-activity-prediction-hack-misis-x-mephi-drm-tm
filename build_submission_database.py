"""Build and validate the five-column, provenance-audited public VACV database."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem


CURATED = Path('data/curated/vacv_new_observations.csv')
BORNyl = Path('artifacts/external/bornyl_amides_expanded_source.csv')
COLUMNS = ['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'DOI']


def canonical(value):
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        raise ValueError(f'Invalid SMILES: {value}')
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/curated/vacv_database.csv'))
    parser.add_argument('--report', type=Path, default=Path('data/curated/vacv_database_validation.json'))
    args = parser.parse_args()
    curated = pd.read_csv(CURATED)
    bornyl = pd.read_csv(BORNyl)
    if len(curated) != 34 or len(bornyl) != 10:
        raise ValueError('Unexpected source row count; review source revisions')
    if not curated[['SMILES', 'IC50_uM', 'CC50_uM', 'SI', 'DOI', 'ic50_url', 'cc50_url', 'si_url']].notna().all().all():
        raise ValueError('Missing curated value or ChEMBL provenance')
    if not bornyl[['SMILES', 'IC50_uM', 'CC50_uM', 'SI_derived', 'DOI', 'source_url']].notna().all().all():
        raise ValueError('Missing bornyl value or publication provenance')
    if not curated[['ic50_url', 'cc50_url', 'si_url']].map(lambda value: str(value).startswith('https://')).all().all():
        raise ValueError('Invalid curated source URL')
    if not bornyl.source_url.map(lambda value: str(value).startswith('https://')).all():
        raise ValueError('Invalid bornyl source URL')
    first = curated[COLUMNS].copy()
    second = bornyl[['SMILES', 'IC50_uM', 'CC50_uM', 'SI_derived', 'DOI']].rename(columns={'SI_derived': 'SI'})
    table = pd.concat([first, second], ignore_index=True)
    table['SMILES'] = table.SMILES.map(canonical)
    for column in ['IC50_uM', 'CC50_uM', 'SI']:
        table[column] = pd.to_numeric(table[column], errors='raise')
        if not table[column].map(lambda value: 0 < value < float('inf')).all():
            raise ValueError(f'Invalid {column}')
    if not table.DOI.map(lambda value: str(value).startswith('10.')).all():
        raise ValueError('Invalid or missing DOI')
    if table.SMILES.duplicated().any():
        raise ValueError('Canonical duplicate across source series')
    relative_error = ((table.CC50_uM / table.IC50_uM - table.SI).abs() / table.SI)
    if (relative_error > .05).any():
        raise ValueError('SI differs more than 5% from CC50/IC50')
    table = table.sort_values(['DOI', 'SMILES']).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, lineterminator='\n')
    roundtrip = pd.read_csv(args.output)
    if list(roundtrip) != COLUMNS or len(roundtrip) != 44 or not roundtrip.SMILES.is_unique:
        raise ValueError('Roundtrip validation failed')
    report = {
        'output': str(args.output), 'rows': len(table), 'columns': COLUMNS,
        'source_files': [str(CURATED), str(BORNyl)],
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [CURATED, BORNyl]},
        'doi_counts': {doi: int(count) for doi, count in table.DOI.value_counts().items()},
        'max_relative_si_ratio_error': float(relative_error.max()),
        'output_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'scope': 'Public source observations; Kaggle train/test files excluded; not all rows are used by the best model.',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
