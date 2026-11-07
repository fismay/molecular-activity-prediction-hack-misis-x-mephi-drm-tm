"""Fixed end-to-end MSE calibration versus current Brier on fresh grouped folds."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from vacv_baseline import featurize
from vacv_text_mixture import fit, probability
from vacv_knn_blend import nearest_ensemble
D=pd.read_csv('data/kaggle/train.csv');y=D['SI VACV'].to_numpy(float);X,g=featurize(D.SMILES)
scores={'current_brier':[],'mse_probability_and_scale':[]};parameters=[]
for i,(tr,va) in enumerate(GroupKFold(n_splits=5,shuffle=True,random_state=271828).split(X,y,g)):
    inner=np.zeros(len(tr));qi=np.zeros(len(tr))
    for a,b in GroupKFold(n_splits=3,shuffle=True,random_state=42).split(tr,y[tr],g[tr]):
        m=fit(D.SMILES.iloc[tr[a]],y[tr[a]],True)
        inner[b]=probability(m,D.SMILES.iloc[tr[b]])
        qi[b]=.5*np.maximum(0,m['regressor'].predict(X[tr[b]]))+.5*nearest_ensemble(D.SMILES.iloc[tr[a]].reset_index(drop=True),y[tr[a]],D.SMILES.iloc[tr[b]])
    f=logit(np.clip(inner,1e-8,1-1e-8));labels=(y[tr]==10000).astype(float)
    def brier(w):return np.mean((expit(w[0]*f+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
    def final_mse(w):
        p=expit(w[0]*f+w[1]);pred=p+(1-p)*qi*w[2]/10000
        return np.mean((pred-y[tr]/10000)**2)+.001*((w[0]-1)**2+w[1]**2+(w[2]-1)**2)
    old=minimize(brier,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])
    new=minimize(final_mse,[1.,0.,1.],method='L-BFGS-B',bounds=[(0,10),(-10,10),(0,2)])
    if not old.success or not new.success:raise RuntimeError('optimization failed')
    m=fit(D.SMILES.iloc[tr],y[tr],True);p0=probability(m,D.SMILES.iloc[va]);fv=logit(np.clip(p0,1e-8,1-1e-8))
    q=.5*np.maximum(0,m['regressor'].predict(X[va]))+.5*nearest_ensemble(D.SMILES.iloc[tr].reset_index(drop=True),y[tr],D.SMILES.iloc[va])
    for name,opt in [('current_brier',old),('mse_probability_and_scale',new)]:
        p=expit(opt.x[0]*fv+opt.x[1]);scale=opt.x[2] if len(opt.x)==3 else 1.
        scores[name].append(float(mean_squared_error(y[va],p*10000+(1-p)*q*scale)))
    parameters.append(new.x.tolist());print(i,{k:round(v[-1]) for k,v in scores.items()},flush=True)
print('mean',{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
json.dump({'seed':271828,'folds':scores,'parameters':parameters,'regularization':.001,'inner_seed':42},open('end_to_end_fresh_report.json','w'),indent=2)
