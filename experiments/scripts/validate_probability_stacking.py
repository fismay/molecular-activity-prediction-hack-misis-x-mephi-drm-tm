"""Nested grouped evaluation of separate text/syntax probability calibration."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
from vacv_knn_blend import nearest_ensemble

D = pd.read_csv('data/kaggle/train.csv')
y = D['SI VACV'].to_numpy(float)
_, groups = featurize(D.SMILES)
S = featurize_syntax(D.SMILES)
base = np.load('artifacts/calibrated_neighbor_validation.npz')
path = Path('artifacts/probability_stack_cache.npz')
cache = dict(np.load(path)) if path.exists() else {}

def components(tr, va):
    v = TfidfVectorizer(analyzer='char', ngram_range=(3, 7), lowercase=False)
    a = v.fit_transform(D.SMILES.iloc[tr])
    text = LogisticRegression(C=100, max_iter=3000).fit(a, y[tr] == 10000)
    syntax = ExtraTreesClassifier(n_estimators=400, max_features=.7, n_jobs=-1, random_state=42).fit(S[tr], y[tr] == 10000)
    return np.c_[text.predict_proba(v.transform(D.SMILES.iloc[va]))[:, 1], syntax.predict_proba(S[va])[:, 1]]

suites = ['42', '2026', '17', 'fresh_groupkfold901', 'new_groupkfold137']
report = {'inputs': ['SMILES'], 'suites': {}}
for suite in suites:
    scores = {name: [] for name in ['previous', 'stack_logit_C1', 'stack_logit_C01', 'stack_raw_C1', 'brier_1d']}
    for i in range(5):
        key = suite+'_'+str(i)
        tr, va = base[key+'_tr'], base[key+'_va']
        if key+'_inner' not in cache:
            inner = np.zeros((len(tr), 2))
            for a, b in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(tr, y[tr], groups[tr]):
                inner[b] = components(tr[a], tr[b])
            cache[key+'_inner'] = inner
            cache[key+'_outer'] = components(tr, va)
            np.savez(path, **cache)
        inner, outer = cache[key+'_inner'], cache[key+'_outer']
        if key+'_conditional' not in cache:
            kn = nearest_ensemble(D.SMILES.iloc[tr].reset_index(drop=True), y[tr], D.SMILES.iloc[va])
            cache[key+'_conditional'] = .5*kn+.5*base[key+'_q']
            np.savez(path, **cache)
        q = cache[key+'_conditional']
        p = base[key+'_p']
        scores['previous'].append(float(mean_squared_error(y[va], 10000*p+(1-p)*q)))
        f = logit(np.clip(inner.mean(axis=1),1e-8,1-1e-8))
        fv = logit(np.clip(outer.mean(axis=1),1e-8,1-1e-8))
        labels = (y[tr] == 10000).astype(float)
        def loss(w):
            return np.mean((expit(w[0]*f+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
        opt = minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])
        p = expit(opt.x[0]*fv+opt.x[1])
        scores['brier_1d'].append(float(mean_squared_error(y[va],10000*p+(1-p)*q)))
        for name, C, use_logit in [('stack_logit_C1', 1, True), ('stack_logit_C01', .1, True), ('stack_raw_C1', 1, False)]:
            a, b = inner, outer
            if use_logit:
                a, b = logit(np.clip(a, 1e-8, 1-1e-8)), logit(np.clip(b, 1e-8, 1-1e-8))
            m = LogisticRegression(C=C, max_iter=2000).fit(a, y[tr] == 10000)
            p = m.predict_proba(b)[:, 1]
            scores[name].append(float(mean_squared_error(y[va], 10000*p+(1-p)*q)))
    report['suites'][suite] = scores
    print(suite, {k: round(np.mean(v)) for k, v in scores.items()}, flush=True)
Path('probability_stacking_report.json').write_text(json.dumps(report, indent=2)+'\n')
