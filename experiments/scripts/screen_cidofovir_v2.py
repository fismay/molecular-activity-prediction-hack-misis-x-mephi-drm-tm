"""Audit a published VACV Vero series and test its effect on fixed v2 folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemical_neighbors import nearest_recipe
from vacv_baseline import featurize
from vacv_zero_hurdle import external_source


def main():
    train = pd.read_csv('data/kaggle/train_v2.csv')
    y = train['SI VACV'].to_numpy(float)
    _, groups = featurize(train.SMILES)
    prior = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), train.SMILES)
    source = pd.read_csv('chembl_cidofovir_curated.csv')
    source = source.loc[source.cell.eq('Vero') & source.validation_status.eq('consistent')].copy()
    assert len(source) == 18
    assert source.DOI.eq('10.1016/j.ejmech.2023.115601').all()
    assert source.ic50_assay_chembl_id.eq('CHEMBL5625829').all()
    assert source.cc50_assay_chembl_id.eq('CHEMBL5625830').all()
    assert source.si_assay_chembl_id.eq('CHEMBL5625831').all()
    assert (source.ic50_standard_relation.eq('=') & source.cc50_standard_relation.eq('=') & source.si_standard_relation.eq('=')).all()
    assert np.isfinite(source[['SI', 'SI_derived']].to_numpy(float)).all()
    assert np.allclose(source.SI, source.SI_derived, rtol=.05)
    _, external_groups = featurize(source.SMILES)
    novel = source.loc[~np.isin(external_groups, groups)].copy()
    assert len(novel) == 16
    augmented = pd.concat([prior, novel[['SMILES', 'SI_derived']]], ignore_index=True)
    cache = np.load('artifacts/zero_v2_validation.npz')
    suites = {}
    for seed in ['42', '2026', '17']:
        scores = {'current': [], 'with_cidofovir': []}
        for fold in range(5):
            key = f'{seed}_{fold}'
            tr, va = cache[key+'_tr'], cache[key+'_va']
            p = cache[key+'_p_raw']
            current = (1-p) * cache[key+'_q_source_stereo_only']
            positives = tr[y[tr] > 0]
            ref_smiles = pd.concat([train.SMILES.iloc[positives], augmented.SMILES], ignore_index=True)
            ref_y = np.r_[y[positives], augmented.SI_derived.to_numpy(float)]
            candidate = (1-p) * nearest_recipe(ref_smiles, ref_y, train.SMILES.iloc[va])
            scores['current'].append(float(np.mean((y[va]-current)**2)))
            scores['with_cidofovir'].append(float(np.mean((y[va]-candidate)**2)))
        suites[seed] = scores
    means = {name: float(np.mean([v for s in suites.values() for v in s[name]])) for name in ['current','with_cidofovir']}
    report = {'source_doi': '10.1016/j.ejmech.2023.115601', 'source_document': 'CHEMBL5625166',
              'assays': ['CHEMBL5625829','CHEMBL5625830','CHEMBL5625831'],
              'context': 'Vero; directly published SI=CC50/IC50; VACV; 72h toxicity/96h antiviral protocol retained',
              'source_rows': len(source), 'novel_after_train_purge': len(novel),
              'selection_rule': 'first 15 fixed grouped folds mean improves >=1% and no seed cohort worsens',
              'suites': suites, 'mean_mse': means,
              'qualified': bool(means['with_cidofovir'] <= .99*means['current'] and all(np.mean(s['with_cidofovir']) <= np.mean(s['current']) for s in suites.values()))}
    Path('cidofovir_v2_screen.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: report[k] for k in ['source_rows','novel_after_train_purge','mean_mse','qualified']}))


if __name__ == '__main__':
    main()
