"""Compare frozen SMILES zero-classifier recipes using training OOF only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from smiles_syntax import featurize_syntax


def main():
 d=pd.read_csv('data/kaggle/train_v2.csv'); y=d['SI VACV'].to_numpy(float); syntax=featurize_syntax(d.SMILES)
 old=np.load('artifacts/zero_v2_validation.npz'); report={'inputs':['SMILES'],'selection':'first15 OOF; >=1% overall mean improvement and no suite worse, then fixed recipe confirmation', 'suites':{},'recipes':{}}; cache={}
 for suite in ['42','2026','17']:
  scores={}
  for fold in range(5):
   key=f'{suite}_{fold}'; tr=old[key+'_tr']; va=old[key+'_va']; q=old[key+'_q_source_stereo_only']; label=y[tr]==0
   vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(3,7),lowercase=False); x=vectorizer.fit_transform(d.SMILES.iloc[tr]); xv=vectorizer.transform(d.SMILES.iloc[va])
   text={c:LogisticRegression(C=c,max_iter=3000).fit(x,label).predict_proba(xv)[:,1] for c in [1,10,100,1000]}
   tree={mf:ExtraTreesClassifier(n_estimators=400,max_features=mf,random_state=42,n_jobs=1).fit(syntax[tr],label).predict_proba(syntax[va])[:,1] for mf in [.3,.7,1.]}
   for c,pt in text.items():
    for mf,ps in tree.items():
     for w in [.25,.5,.75]:
      name=f'C{c}_mf{mf}_text{w}'; p=w*pt+(1-w)*ps; pred=(1-p)*q
      scores.setdefault(name,[]).append(float(np.mean((y[va]-pred)**2)));cache[key+'__'+name]=p
      report['recipes'][name]={'C':c,'max_features':mf,'text_weight':w}
   assert np.allclose(.5*text[100]+.5*tree[.7],old[key+'_p_raw'],atol=1e-12,rtol=0)
   print(suite,fold,flush=True)
  report['suites'][suite]=scores
 report['mean_mse']={name:float(np.mean([v for suite in report['suites'].values() for v in suite[name]])) for name in scores}
 current='C100_mf0.7_text0.5'; selected=min(report['mean_mse'],key=report['mean_mse'].get)
 report['current']=current;report['selected']=selected
 report['qualified_for_confirmation']=report['mean_mse'][selected]<.99*report['mean_mse'][current] and all(np.mean(s[selected])<=np.mean(s[current]) for s in report['suites'].values())
 Path('zero_v2_classifier_screen.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed('artifacts/zero_v2_classifier_screen.npz',**cache)
 print('ranked',sorted(report['mean_mse'].items(),key=lambda z:z[1])[:8],flush=True)
 print('qualified',report['qualified_for_confirmation'],flush=True)
if __name__=='__main__':main()
