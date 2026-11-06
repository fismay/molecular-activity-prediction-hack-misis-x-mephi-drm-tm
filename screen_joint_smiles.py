"""Explore joint text/syntax classifiers on fixed grouped folds; not a release."""
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error,pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit
from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
D=pd.read_csv('data/kaggle/train.csv');y=D['SI VACV'].to_numpy(float);X,g=featurize(D.SMILES);S=featurize_syntax(D.SMILES);old=np.load('artifacts/probability_oof.npz')
splits=[(tr,va) for seed in [42,2026,17] for tr,va in GroupShuffleSplit(n_splits=5,test_size=.2,random_state=seed).split(X,y,g)]
recipes=[('linear',weight,C) for weight in [.1,.3] for C in [10,100]]+[('kernel_add',gamma,10) for gamma in [.1,1,3]]+[('kernel_product',gamma,10) for gamma in [.1,1,3]]
results={str(r):[] for r in recipes};baseline=[]
for i,(tr,va) in enumerate(splits):
    v=TfidfVectorizer(analyzer='char',ngram_range=(3,7),lowercase=False);a=v.fit_transform(D.SMILES.iloc[tr]);b=v.transform(D.SMILES.iloc[va]);scaler=StandardScaler().fit(S[tr]);sa=np.clip(scaler.transform(S[tr]),-5,5);sb=np.clip(scaler.transform(S[va]),-5,5)
    kt=(a@a.T).toarray();kv=(b@a.T).toarray();da=pairwise_distances(sa,metric='sqeuclidean')/S.shape[1];db=pairwise_distances(sb,sa,metric='sqeuclidean')/S.shape[1]
    p=old[f'{i}_p'];q=old[f'{i}_q'];baseline.append(float(mean_squared_error(y[va],p*10000+(1-p)*q)))
    for r in recipes:
        kind,w,C=r
        if kind=='linear':
            aa=sparse.hstack([a,sparse.csr_matrix(w*sa)],format='csr');bb=sparse.hstack([b,sparse.csr_matrix(w*sb)],format='csr');model=LogisticRegression(C=C,max_iter=3000).fit(aa,y[tr]==10000);c=model.predict_proba(bb)[:,1]
        else:
            ka,kb=np.exp(-w*da),np.exp(-w*db)
            aa,bb=((kt+ka)/2,(kv+kb)/2) if kind=='kernel_add' else (kt*ka,kv*kb)
            model=SVC(C=C,kernel='precomputed',probability=True,random_state=42).fit(aa,y[tr]==10000);c=model.predict_proba(bb)[:,1]
        results[str(r)].append([float(mean_squared_error(y[va],pp*10000+(1-pp)*q)) for pp in [c,.5*c+.5*p,.25*c+.75*p]])
    print('fold',i,flush=True)
report={'baseline_mean':float(np.mean(baseline)),'blend_weights':['candidate_only','50_percent_candidate','25_percent_candidate'],'results':results,'note':'Exploratory screening; any selected classifier requires nested grouped calibration before release.'}
json.dump(report,open('joint_smiles_screening_report.json','w'),indent=2)
for k,v in results.items():print(k,[round(z) for z in np.mean(v,axis=0)],flush=True)
