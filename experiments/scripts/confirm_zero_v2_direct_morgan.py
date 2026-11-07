"""Confirm the pre-screened best Morgan all-target recipe on saved new splits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from chemical_neighbors import nearest_recipe
from vacv_zero_hurdle import external_source


def main():
 screen=json.loads(Path('zero_v2_direct_screen.json').read_text())
 names=[n for n in screen['recipes'] if n.startswith('morgan_count_r2_')]
 selected=min(names,key=lambda n:screen['mean_mse'][n]);recipe=screen['recipes'][selected]
 assert screen['mean_mse'][selected]<.99*screen['mean_mse']['current']
 assert all(np.mean(s[selected])<np.mean(s['current']) for s in screen['suites'].values())
 d=pd.read_csv('data/kaggle/train_v2.csv');y=d['SI VACV'].to_numpy(float)
 e=external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'),d.SMILES)
 old=np.load('artifacts/zero_v2_direct_confirmation.npz')
 report={'inputs':['SMILES'],'selected_before_confirmation':selected,'recipe':recipe,'confirmation_seeds':[61453,40511],'selection':'lowest mean Morgan recipe from existing pre-public screen; same held-out grouped partitions; no public-score weight tuning','suites':{}};cache={}
 for seed in report['confirmation_seeds']:
  scores={'current':[],'candidate':[]}
  for fold in range(5):
   key=f'{seed}_{fold}';tr=old[key+'_tr'];va=old[key+'_va'];current=old[key+'_current']
   ref=pd.concat([d.SMILES.iloc[tr],e.SMILES],ignore_index=True);values=np.r_[y[tr],e.SI_derived.to_numpy(float)]
   direct=nearest_recipe(ref,values,d.SMILES.iloc[va],recipe['feature'],recipe['k'],recipe['power'])
   pred=(1-recipe['current_weight'])*direct+recipe['current_weight']*current
   for name,prediction in [('current',current),('candidate',pred)]:scores[name].append(float(np.mean((y[va]-prediction)**2)));cache[key+'_'+name]=prediction
   cache[key+'_tr'],cache[key+'_va']=tr,va
  report['suites'][str(seed)]=scores
  print(seed,{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
 report['pooled_mse']={name:float(np.mean([v for s in report['suites'].values() for v in s[name]])) for name in ['current','candidate']}
 report['qualified']=report['pooled_mse']['candidate']<.99*report['pooled_mse']['current'] and all(np.mean(s['candidate'])<=np.mean(s['current']) for s in report['suites'].values())
 Path('zero_v2_direct_morgan_confirmation.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('artifacts/zero_v2_direct_morgan_confirmation.npz',**cache)
 print('qualified',report['qualified'],report['pooled_mse'],flush=True)
if __name__=='__main__':main()
