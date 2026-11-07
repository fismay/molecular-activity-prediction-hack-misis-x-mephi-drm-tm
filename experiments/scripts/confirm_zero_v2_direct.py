"""Confirm the frozen all-target neighbor winner on two predeclared seeds."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from chemical_neighbors import nearest_recipe
from vacv_zero_hurdle import external_source, fit_classifier, raw_probability
from vacv_baseline import featurize
from smiles_syntax import featurize_syntax


def main():
 screen=json.loads(Path('zero_v2_direct_screen.json').read_text());assert screen['qualified_for_confirmation']
 recipe=screen['recipes'][screen['selected']]
 d=pd.read_csv('data/kaggle/train_v2.csv');y=d['SI VACV'].to_numpy(float);_,groups=featurize(d.SMILES);syntax=featurize_syntax(d.SMILES)
 e=external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'),d.SMILES)
 report={'inputs':['SMILES'],'selected_before_confirmation':screen['selected'],'recipe':recipe,'seeds':screen['confirmation_seeds_fixed_before_screen'],'gate':'>=1% lower pooled MSE, neither new suite mean worse','suites':{}};cache={}
 for seed in report['seeds']:
  scores={'current':[],'candidate':[]}
  for fold,(tr,va) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=seed).split(d.SMILES,y,groups)):
   classifier=fit_classifier(d.SMILES.iloc[tr],y[tr]==0,syntax[tr]);prob=raw_probability(classifier,d.SMILES.iloc[va],syntax[va])
   positive=tr[y[tr]>0]
   ref=pd.concat([d.SMILES.iloc[positive],e.SMILES],ignore_index=True);values=np.r_[y[positive],e.SI_derived.to_numpy(float)]
   current=(1-prob)*nearest_recipe(ref,values,d.SMILES.iloc[va])
   all_ref=pd.concat([d.SMILES.iloc[tr],e.SMILES],ignore_index=True);all_values=np.r_[y[tr],e.SI_derived.to_numpy(float)]
   direct=nearest_recipe(all_ref,all_values,d.SMILES.iloc[va],recipe['feature'],recipe['k'],recipe['power'])
   pred=(1-recipe['current_weight'])*direct+recipe['current_weight']*current
   key=f'{seed}_{fold}'
   for name,values in [('current',current),('candidate',pred)]:scores[name].append(float(np.mean((y[va]-values)**2)));cache[key+'_'+name]=values
   cache[key+'_tr'],cache[key+'_va']=tr,va
   print(seed,fold,{k:round(v[-1]) for k,v in scores.items()},flush=True)
  report['suites'][str(seed)]=scores
  Path('zero_v2_direct_confirmation.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('artifacts/zero_v2_direct_confirmation.npz',**cache)
  print(seed,{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
 report['pooled_mse']={name:float(np.mean([v for s in report['suites'].values() for v in s[name]])) for name in ['current','candidate']}
 report['qualified']=report['pooled_mse']['candidate']<.99*report['pooled_mse']['current'] and all(np.mean(s['candidate'])<=np.mean(s['current']) for s in report['suites'].values())
 Path('zero_v2_direct_confirmation.json').write_text(json.dumps(report,indent=2)+'\n');print('qualified',report['qualified'],report['pooled_mse'],flush=True)
if __name__=='__main__':main()
