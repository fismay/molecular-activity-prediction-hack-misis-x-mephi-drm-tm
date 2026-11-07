"""Optimize final SI MSE on strictly inner grouped predictions only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as F
from vacv_baseline import featurize, FP

D = pd.read_csv('data/kaggle/train.csv')
y = D['SI VACV'].to_numpy(float)
X, groups = featurize(D.SMILES)
mols = [Chem.MolFromSmiles(s) for s in D.SMILES]
recipes = [(FP, False, 3, 4), (F.GetMorganGenerator(radius=2,includeChirality=True), True, 1, 2), (F.GetRDKitFPGenerator(), True, 1, 2)]
similarities=[]
for gen, count, k, power in recipes:
    fps=[gen.GetCountFingerprint(m) if count else gen.GetFingerprint(m) for m in mols]
    similarities.append(np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps]))
def neighbors(tr,va):
    ordinary=tr[y[tr]!=10000]
    values=[]
    for S,(_,_,k,power) in zip(similarities,recipes):
        s=S[np.ix_(va,ordinary)]
        cutoff=np.sort(s,axis=1)[:,-k]
        w=np.where(s>=cutoff[:,None]-1e-12,s**power,0)
        values.append(w@y[ordinary]/np.maximum(w.sum(axis=1),1e-9))
    return np.mean(values,axis=0)
base=np.load('artifacts/calibrated_neighbor_validation.npz')
stack=np.load('artifacts/probability_stack_cache.npz')
path=Path('artifacts/end_to_end_calibration_cache.npz')
cache=dict(np.load(path)) if path.exists() else {}
report={'inputs':['SMILES'],'regularization':.001,'suites':{}}
for suite in ['42','2026','17','fresh_groupkfold901','new_groupkfold137']:
    scores={n:[] for n in ['current_brier','mse_probability','mse_probability_and_scale']}
    parameters={n:[] for n in scores}
    for i in range(5):
        key=suite+'_'+str(i)
        tr,va=base[key+'_tr'],base[key+'_va']
        if key+'_inner_conditional' not in cache:
            qinner=np.zeros(len(tr))
            for a,b in GroupKFold(n_splits=3,shuffle=True,random_state=42).split(tr,y[tr],groups[tr]):
                fitidx,validx=tr[a],tr[b]
                ordinary=fitidx[y[fitidx]!=10000]
                reg=RandomForestRegressor(n_estimators=250,min_samples_leaf=1,max_features=.3,random_state=42,n_jobs=-1).fit(X[ordinary],y[ordinary])
                qinner[b]=.5*np.maximum(0,reg.predict(X[validx]))+.5*neighbors(fitidx,validx)
            cache[key+'_inner_conditional']=qinner
            np.savez(path,**cache)
        qinner=cache[key+'_inner_conditional']/10000
        f=logit(np.clip(stack[key+'_inner'].mean(axis=1),1e-8,1-1e-8))
        fv=logit(np.clip(stack[key+'_outer'].mean(axis=1),1e-8,1-1e-8))
        q=.5*base[key+'_q']+.5*neighbors(tr,va)
        labels=(y[tr]==10000).astype(float)
        def brier(w):return np.mean((expit(w[0]*f+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
        opts={'current_brier':minimize(brier,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])}
        for scale in [False,True]:
            prior=np.array([1.,0.,1.]) if scale else np.array([1.,0.])
            def objective(w):
                p=expit(w[0]*f+w[1])
                s=w[2] if scale else 1.
                pred=p+(1-p)*qinner*s
                return np.mean((pred-y[tr]/10000)**2)+.001*np.sum((w-prior)**2)
            name='mse_probability_and_scale' if scale else 'mse_probability'
            opts[name]=minimize(objective,prior,method='L-BFGS-B',bounds=[(0,10),(-10,10)]+([(0,2)] if scale else []))
        for name,opt in opts.items():
            if not opt.success:raise RuntimeError(name+': '+opt.message)
            p=expit(opt.x[0]*fv+opt.x[1]);s=opt.x[2] if len(opt.x)==3 else 1.
            pred=p*10000+(1-p)*q*s
            scores[name].append(float(mean_squared_error(y[va],pred)));parameters[name].append(opt.x.tolist())
    report['suites'][suite]={'mse':scores,'parameters':parameters}
    print(suite,{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
Path('end_to_end_calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
