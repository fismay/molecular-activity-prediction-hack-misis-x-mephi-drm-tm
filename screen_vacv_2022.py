"""Screen the newly collected exact published VACV/Vero SI on frozen v2 folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemical_neighbors import nearest_recipe
from vacv_baseline import featurize
from vacv_zero_hurdle import external_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('2022','2018','2021'), default='2022')
    args = parser.parse_args()
    meta = {'2022': ('10.1016/j.bmcl.2022.129080','CHEMBL5223146',11,10),
            '2018': ('10.1039/C8MD00442K','CHEMBL4368876',5,3),
            '2021': ('10.1016/j.ejmech.2021.113485','CHEMBL4837211',5,5)}
    doi,document,n_consistent,n_novel = meta[args.study]
    train = pd.read_csv('data/kaggle/train_v2.csv')
    y = train['SI VACV'].to_numpy(float)
    _, groups = featurize(train.SMILES)
    prior = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), train.SMILES)
    raw = pd.read_csv(f'vacv_{args.study}_published_si.csv')
    source = raw.loc[raw.validation_status.eq('consistent')].copy()
    assert len(source) == n_consistent and source.DOI.str.lower().eq(doi.lower()).all()
    assert np.isfinite(source[['SI_published','SI_derived']].to_numpy(float)).all()
    assert np.allclose(source.SI_published,source.SI_derived,rtol=.05)
    _, source_groups = featurize(source.SMILES)
    novel = source.loc[~np.isin(source_groups,groups)].copy()
    assert len(novel) == n_novel
    new_rows = novel[['SMILES','SI_published']].rename(columns={'SI_published':'SI_derived'})
    augmented = pd.concat([prior, new_rows],ignore_index=True)
    cache = np.load('artifacts/zero_v2_validation.npz')
    suites = {}
    for seed in ['42','2026','17']:
        scores = {'current': [], 'with_study': []}
        for fold in range(5):
            key = f'{seed}_{fold}'
            tr,va = cache[key+'_tr'],cache[key+'_va']
            p = cache[key+'_p_raw']
            current = (1-p)*cache[key+'_q_source_stereo_only']
            positive = tr[y[tr]>0]
            ref = pd.concat([train.SMILES.iloc[positive],augmented.SMILES],ignore_index=True)
            values = np.r_[y[positive],augmented.SI_derived.to_numpy(float)]
            candidate = (1-p)*nearest_recipe(ref,values,train.SMILES.iloc[va])
            scores['current'].append(float(np.mean((y[va]-current)**2)))
            scores['with_study'].append(float(np.mean((y[va]-candidate)**2)))
        suites[seed] = scores
    means = {name:float(np.mean([v for s in suites.values() for v in s[name]])) for name in ['current','with_study']}
    qualified = means['with_study'] <= .99*means['current'] and all(np.mean(s['with_study']) <= np.mean(s['current']) for s in suites.values())
    report = {'source_doi':doi,'source_document':document,
              'exact_si_rows':len(raw),'consistent_triples':n_consistent,
              'validation_status':{str(k):int(v) for k,v in raw.validation_status.value_counts().items()},
              'novel_consistent_after_train_purge':len(novel),
              'mismatch_policy':'inconsistent ratio and published-SI-only rows excluded',
              'selection_rule':'first 15 fixed grouped folds mean improves >=1% and no seed cohort worsens',
              'suites':suites,'mean_mse':means,'qualified':bool(qualified)}
    Path(f'vacv_{args.study}_v2_screen.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'mean_mse':means,'qualified':qualified}))


if __name__=='__main__': main()
