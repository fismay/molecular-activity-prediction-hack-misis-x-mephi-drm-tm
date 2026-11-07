"""Grouped nested calibration of a joint text/syntax kernel signal."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.optimize import minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances,mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from smiles_syntax import featurize_syntax
from vacv_baseline import featurize

D=pd.read_csv('data/kaggle/train.csv');y=D['SI VACV'].to_numpy(float);_,groups=featurize(D.SMILES);S=featurize_syntax(D.SMILES)
base=np.load('artifacts/calibrated_neighbor_validation.npz');stack=np.load('artifacts/probability_stack_cache.npz');conditional=np.load('artifacts/end_to_end_calibration_cache.npz')
path=Path('artifacts/joint_kernel_cache.npz');cache=dict(np.load(path)) if path.exists() else {}

def margin(tr,va):
    vector=TfidfVectorizer(analyzer='char',ngram_range=(3,7),lowercase=False)
    a=vector.fit_transform(D.SMILES.iloc[tr]);b=vector.transform(D.SMILES.iloc[va])
    scaler=StandardScaler().fit(S[tr]);sa=np.clip(scaler.transform(S[tr]),-5,5);sb=np.clip(scaler.transform(S[va]),-5,5)
    ka=((a@a.T).toarray()+np.exp(-pairwise_distances(sa,metric='sqeuclidean')/S.shape[1]))/2
    kb=((b@a.T).toarray()+np.exp(-pairwise_distances(sb,sa,metric='sqeuclidean')/S.shape[1]))/2
    return SVC(C=10,kernel='precomputed').fit(ka,y[tr]==10000).decision_function(kb)

report={'inputs':['SMILES'],'classifier':'SVC C10; mean(text cosine, syntax RBF)','regularization':.001,'suites':{}}
for suite in ['42','2026','17','fresh_groupkfold901','new_groupkfold137']:
    scores={'previous_final_mse':[],'joint_signal':[]};params=[]
    for i in range(5):
        key=suite+'_'+str(i);tr=base[key+'_tr'];va=base[key+'_va']
        if key+'_inner_margin' not in cache:
            inner=np.zeros(len(tr))
            for a,b in GroupKFold(n_splits=3,shuffle=True,random_state=42).split(tr,y[tr],groups[tr]):inner[b]=margin(tr[a],tr[b])
            cache[key+'_inner_margin']=inner;cache[key+'_outer_margin']=margin(tr,va);np.savez(path,**cache)
        f=logit(np.clip(stack[key+'_inner'].mean(axis=1),1e-8,1-1e-8));fv=logit(np.clip(stack[key+'_outer'].mean(axis=1),1e-8,1-1e-8));qinner=conditional[key+'_inner_conditional']/10000;q=stack[key+'_conditional']
        for joint in [False,True]:
            prior=np.array([1.,0.,0.,1.]) if joint else np.array([1.,0.,1.])
            def loss(w):
                if joint:p=expit(w[0]*f+w[1]*cache[key+'_inner_margin']+w[2])
                else:p=expit(w[0]*f+w[1])
                pred=p+(1-p)*qinner*w[-1]
                return np.mean((pred-y[tr]/10000)**2)+.001*np.sum((w-prior)**2)
            bounds=[(0,10),(0,10),(-10,10),(0,2)] if joint else [(0,10),(-10,10),(0,2)]
            opt=minimize(loss,prior,method='L-BFGS-B',bounds=bounds)
            if not opt.success:raise RuntimeError(opt.message)
            if joint:p=expit(opt.x[0]*fv+opt.x[1]*cache[key+'_outer_margin']+opt.x[2]);name='joint_signal';params.append(opt.x.tolist())
            else:p=expit(opt.x[0]*fv+opt.x[1]);name='previous_final_mse'
            scores[name].append(float(mean_squared_error(y[va],p*10000+(1-p)*q*opt.x[-1])))
    report['suites'][suite]={'mse':scores,'joint_parameters':params};print(suite,{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
Path('joint_calibration_report.json').write_text(json.dumps(report,indent=2)+'\n')
