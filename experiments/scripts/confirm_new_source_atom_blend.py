"""Confirm one frozen Morgan/atom-pair source recipe on untouched grouped folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from chemical_neighbors import nearest_recipe
from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
from vacv_zero_hurdle import external_source, fit_classifier, raw_probability


SEEDS = [35471, 68011]


def main():
    screened=json.loads(Path('new_source_atom_blend_screen.json').read_text())
    assert screened['qualified'] is True
    train=pd.read_csv('data/kaggle/train_v2.csv')
    y=train['SI VACV'].to_numpy(float)
    _,groups=featurize(train.SMILES)
    syntax=featurize_syntax(train.SMILES)
    old=external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'),train.SMILES)
    new=pd.read_csv('data/curated/vacv_new_observations.csv')
    _,new_groups=featurize(new.SMILES)
    assert len(new)==34 and not set(groups)&set(new_groups)
    all_extra=pd.concat([old,new[['SMILES','SI']].rename(columns={'SI':'SI_derived'})],ignore_index=True)
    report={'candidate':'fixed 50/50 Morgan baseline and atom-pair k1 with 34 audited external rows',
            'inputs':['SMILES'],'seeds':SEEDS,
            'selection_rule':'pooled mean MSE improves >=1% and neither new seed cohort worse',
            'suites':{}}
    cache={}
    for seed in SEEDS:
        scores={'current':[],'candidate':[]}
        for fold,(tr,va) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(train.SMILES,y,groups)):
            clf=fit_classifier(train.SMILES.iloc[tr],y[tr]==0,syntax[tr])
            p=raw_probability(clf,train.SMILES.iloc[va],syntax[va])
            positives=tr[y[tr]>0]
            old_smiles=pd.concat([train.SMILES.iloc[positives],old.SMILES],ignore_index=True)
            old_y=np.r_[y[positives],old.SI_derived.to_numpy(float)]
            old_q=nearest_recipe(old_smiles,old_y,train.SMILES.iloc[va],feature='morgan_count_r2')
            new_smiles=pd.concat([train.SMILES.iloc[positives],all_extra.SMILES],ignore_index=True)
            new_y=np.r_[y[positives],all_extra.SI_derived.to_numpy(float)]
            atom_q=nearest_recipe(new_smiles,new_y,train.SMILES.iloc[va],feature='atom_pair_sparse')
            current=(1-p)*old_q
            candidate=(1-p)*(.5*old_q+.5*atom_q)
            scores['current'].append(float(np.mean((y[va]-current)**2)))
            scores['candidate'].append(float(np.mean((y[va]-candidate)**2)))
            key=f'{seed}_{fold}'
            cache[key+'_tr'],cache[key+'_va'],cache[key+'_current'],cache[key+'_candidate']=tr,va,current,candidate
            print(seed,fold,round(scores['current'][-1]),round(scores['candidate'][-1]),flush=True)
        report['suites'][str(seed)]=scores
        Path('new_source_atom_blend_confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed('artifacts/new_source_atom_blend_confirmation.npz',**cache)
    means={name:float(np.mean([v for s in report['suites'].values() for v in s[name]])) for name in ['current','candidate']}
    report['pooled_mse']=means
    report['qualified']=bool(means['candidate']<=.99*means['current'] and all(np.mean(s['candidate'])<=np.mean(s['current']) for s in report['suites'].values()))
    Path('new_source_atom_blend_confirmation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pooled_mse':means,'qualified':report['qualified']}),flush=True)


if __name__=='__main__':main()
