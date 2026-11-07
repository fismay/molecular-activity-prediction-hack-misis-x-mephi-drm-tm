"""Evaluate the frozen best model with all newly audited Vero/VACV observations."""

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
    base = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), train.SMILES)
    new = pd.read_csv('data/curated/vacv_new_observations.csv')
    assert len(new) == 34 and new.SMILES.is_unique
    _,train_group = featurize(train.SMILES)
    _,new_group = featurize(new.SMILES)
    assert not set(train_group) & set(new_group)
    ref = pd.concat([base,new.rename(columns={'SI':'SI_derived'})[['SMILES','SI_derived']]],ignore_index=True)
    cache=np.load('artifacts/zero_v2_validation.npz')
    suites={}
    for seed in ['42','2026','17']:
        scores={'current':[],'all_34':[]}
        for fold in range(5):
            key=f'{seed}_{fold}';tr,va=cache[key+'_tr'],cache[key+'_va'];p=cache[key+'_p_raw']
            positive=tr[y[tr]>0]
            current=(1-p)*cache[key+'_q_source_stereo_only']
            smiles=pd.concat([train.SMILES.iloc[positive],ref.SMILES],ignore_index=True)
            values=np.r_[y[positive],ref.SI_derived.to_numpy(float)]
            candidate=(1-p)*nearest_recipe(smiles,values,train.SMILES.iloc[va])
            scores['current'].append(float(np.mean((y[va]-current)**2)))
            scores['all_34'].append(float(np.mean((y[va]-candidate)**2)))
        suites[seed]=scores
    means={name:float(np.mean([v for s in suites.values() for v in s[name]])) for name in ['current','all_34']}
    qualified=means['all_34'] <= .99*means['current'] and all(np.mean(s['all_34'])<=np.mean(s['current']) for s in suites.values())
    report={'source':'data/curated/vacv_new_observations.csv','rows':len(new),
            'selection_rule':'first15 mean >=1% improvement and no cohort worse',
            'suites':suites,'mean_mse':means,'qualified':bool(qualified)}
    Path('all_new_sources_v2_screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mean_mse':means,'qualified':qualified}))


if __name__=='__main__':main()
