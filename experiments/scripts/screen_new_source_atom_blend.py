"""Predeclared 50/50 Morgan baseline and atom-pair source augmentation screen."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chemical_neighbors import nearest_recipe
from vacv_baseline import featurize
from vacv_zero_hurdle import external_source


def main():
    train=pd.read_csv('data/kaggle/train_v2.csv')
    y=train['SI VACV'].to_numpy(float)
    old=external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'),train.SMILES)
    new=pd.read_csv('data/curated/vacv_new_observations.csv')
    assert len(new)==34
    _,a=featurize(train.SMILES);_,b=featurize(new.SMILES)
    assert not set(a)&set(b)
    extra=pd.concat([old,new[['SMILES','SI']].rename(columns={'SI':'SI_derived'})],ignore_index=True)
    cached=np.load('artifacts/zero_v2_validation.npz')
    suites={}
    for seed in ['42','2026','17']:
        scores={'current':[],'half_atom_new':[]}
        for fold in range(5):
            key=f'{seed}_{fold}';tr,va=cached[key+'_tr'],cached[key+'_va'];p=cached[key+'_p_raw']
            positive=tr[y[tr]>0]
            ref=pd.concat([train.SMILES.iloc[positive],extra.SMILES],ignore_index=True)
            values=np.r_[y[positive],extra.SI_derived.to_numpy(float)]
            old_q=cached[key+'_q_source_stereo_only']
            atom_q=nearest_recipe(ref,values,train.SMILES.iloc[va],feature='atom_pair_sparse')
            scores['current'].append(float(np.mean((y[va]-(1-p)*old_q)**2)))
            scores['half_atom_new'].append(float(np.mean((y[va]-(1-p)*(.5*old_q+.5*atom_q))**2)))
        suites[seed]=scores
    means={name:float(np.mean([v for s in suites.values() for v in s[name]])) for name in ['current','half_atom_new']}
    qualified=means['half_atom_new']<=.99*means['current'] and all(np.mean(s['half_atom_new'])<=np.mean(s['current']) for s in suites.values())
    report={'candidate':'fixed half current Morgan positive estimate + half atom-pair k1 using all 34 new audited rows',
            'input_features':['SMILES'],'selection_rule':'first 15 mean >=1% improvement and no cohort worse',
            'suites':suites,'mean_mse':means,'qualified':bool(qualified)}
    Path('new_source_atom_blend_screen.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'mean_mse':means,'qualified':qualified}))


if __name__=='__main__':main()
