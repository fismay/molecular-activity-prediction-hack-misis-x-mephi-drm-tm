"""Training-only screen of direct target neighbors, including corrected zeros."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from chemical_neighbors import similarities, neighbor_values
from vacv_zero_hurdle import external_source


def main():
 d=pd.read_csv('data/kaggle/train_v2.csv');y=d['SI VACV'].to_numpy(float)
 e=external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'),d.SMILES)
 strings=pd.concat([d.SMILES,e.SMILES],ignore_index=True);values=np.r_[y,e.SI_derived.to_numpy(float)];external=np.arange(len(d),len(strings))
 old=np.load('artifacts/zero_v2_validation.npz');predictions={}
 report={'inputs':['SMILES'],'selection':'15 grouped splits only; >=1% lower overall mean MSE and no seed-suite mean worse; then new seeds61453,40511 with >=1% pooled improvement and neither suite worse','confirmation_seeds_fixed_before_screen':[61453,40511],'suites':{},'recipes':{}}
 matrices={f:similarities(strings,strings,f) for f in ['morgan_count_r2','atom_pair_sparse','torsion_sparse']}
 for suite in ['42','2026','17']:
  scores={}
  for fold in range(5):
   key=f'{suite}_{fold}';tr=old[key+'_tr'];va=old[key+'_va'];reference=np.r_[tr,external];p=old[key+'_p_raw'];base=(1-p)*old[key+'_q_source_stereo_only']
   scores.setdefault('current',[]).append(float(np.mean((y[va]-base)**2)))
   for feature,matrix in matrices.items():
    sim=matrix[np.ix_(va,reference)]
    for k in [1,3,5,10]:
     for power in ([2] if k==1 else [2,4]):
      direct=neighbor_values(sim,values[reference],k,power)
      for blend in [0.,.5]:
       pred=(1-blend)*direct+blend*base
       name=f'{feature}_k{k}_p{power}_current{blend}'
       report['recipes'][name]={'feature':feature,'k':k,'power':power,'current_weight':blend}
       scores.setdefault(name,[]).append(float(np.mean((y[va]-pred)**2)));predictions[key+'__'+name]=pred
  report['suites'][suite]=scores
  print(suite,sorted(((k,round(np.mean(v))) for k,v in scores.items()),key=lambda z:z[1])[:5],flush=True)
 report['mean_mse']={name:float(np.mean([v for s in report['suites'].values() for v in s[name]])) for name in scores}
 selected=min(report['mean_mse'],key=report['mean_mse'].get);report['selected']=selected
 report['qualified_for_confirmation']=selected!='current' and report['mean_mse'][selected]<.99*report['mean_mse']['current'] and all(np.mean(s[selected])<=np.mean(s['current']) for s in report['suites'].values())
 Path('zero_v2_direct_screen.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('artifacts/zero_v2_direct_screen.npz',**predictions)
 print('best',selected,report['mean_mse'][selected],'qualified',report['qualified_for_confirmation'],flush=True)
if __name__=='__main__':main()
