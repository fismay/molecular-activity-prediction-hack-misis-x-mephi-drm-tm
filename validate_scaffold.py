"""Scaffold-held-out comparison; all inner folds also exclude full scaffolds."""
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import json
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from vacv_baseline import featurize
from vacv_text_mixture import fit, probability
from vacv_knn_blend import nearest_ensemble
D = pd.read_csv('data/kaggle/train.csv')
y = D['SI VACV'].to_numpy(float)
X, canonical = featurize(D.SMILES)
g = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) or c for s,c in zip(D.SMILES,canonical)])
scores = {'baseline': [], 'brier': []}
for i, (tr, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=20261107).split(X,y,g)):
    inner = np.zeros(len(tr))
    for a,b in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(tr,y[tr],g[tr]):
        m = fit(D.SMILES.iloc[tr[a]], y[tr[a]], True)
        inner[b] = probability(m, D.SMILES.iloc[tr[b]])
    f = logit(np.clip(inner,1e-8,1-1e-8))
    labels = (y[tr] == 10000).astype(float)
    m = fit(D.SMILES.iloc[tr],y[tr],True)
    p0 = probability(m,D.SMILES.iloc[va])
    fv = logit(np.clip(p0,1e-8,1-1e-8))
    sigmoid = LogisticRegression(C=1,max_iter=2000).fit(f[:,None],labels)
    def loss(w):return np.mean((expit(w[0]*f+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
    opt = minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])
    q = .5*np.maximum(0,m['regressor'].predict(X[va]))+.5*nearest_ensemble(D.SMILES.iloc[tr].reset_index(drop=True),y[tr],D.SMILES.iloc[va])
    for name,p in [('baseline',sigmoid.predict_proba(fv[:,None])[:,1]),('brier',expit(opt.x[0]*fv+opt.x[1]))]:
        scores[name].append(float(mean_squared_error(y[va],p*10000+(1-p)*q)))
    print(i,{k:round(v[-1]) for k,v in scores.items()},flush=True)
print('mean',{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
json.dump({'seed':20261107, 'grouping':'Murcko scaffold, canonical structure fallback for acyclic molecules','folds':scores,'brier_regularization':.001,'inner_seed':42},open('scaffold_validation_report.json','w'),indent=2)
