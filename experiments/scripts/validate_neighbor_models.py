"""Compare fixed conditional nearest-neighbor candidates with grouped validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as F
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error
from vacv_baseline import featurize, FP
from vacv_text_mixture import fit, probability

D = pd.read_csv('data/kaggle/train.csv')
y = D['SI VACV'].to_numpy(float)
X, groups = featurize(D.SMILES)
mols = [Chem.MolFromSmiles(s) for s in D.SMILES]
variants = {'binary3': (FP, False, 3, 4),
            'chiral_counts1': (F.GetMorganGenerator(radius=2, includeChirality=True), True, 1, 2),
            'rdkit_counts1': (F.GetRDKitFPGenerator(), True, 1, 2)}
similarities = {}
for name, (gen, counts, k, power) in variants.items():
    fps = [(gen.GetCountFingerprint(m) if counts else gen.GetFingerprint(m)) for m in mols]
    similarities[name] = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
suites = {str(seed): list(GroupShuffleSplit(n_splits=5, test_size=.2, random_state=seed).split(X, y, groups)) for seed in [42, 2026, 17]}
suites['fresh_groupkfold901'] = list(GroupKFold(n_splits=5, shuffle=True, random_state=901).split(X, y, groups))
suites['new_groupkfold137'] = list(GroupKFold(n_splits=5, shuffle=True, random_state=137).split(X, y, groups))
report = {'inputs': ['SMILES'], 'suites': {}, 'knn_weight': .5}
cache_path = Path('artifacts/calibrated_neighbor_validation.npz')
cache = dict(np.load(cache_path)) if cache_path.exists() else {}
for suite, splits in suites.items():
    scores = {name: [] for name in ['baseline'] + list(variants) + ['neighbor_ensemble']}
    for i, (tr, va) in enumerate(splits):
        key = suite+'_'+str(i)
        if key+'_p' not in cache:
            model = fit(D.SMILES.iloc[tr], y[tr], True, 'raw', .7, True)
            cache[key+'_p'] = probability(model, D.SMILES.iloc[va])
            cache[key+'_q'] = np.maximum(0, model['regressor'].predict(X[va]))
            cache[key+'_tr'], cache[key+'_va'] = tr, va
            np.savez(cache_path, **cache)
        assert np.array_equal(tr, cache[key+'_tr']) and np.array_equal(va, cache[key+'_va'])
        p, q = cache[key+'_p'], cache[key+'_q']
        scores['baseline'].append(float(mean_squared_error(y[va], 10000*p+(1-p)*q)))
        ordinary = tr[y[tr] != 10000]
        ensemble = []
        for name, (_, _, k, power) in variants.items():
            sim = similarities[name][np.ix_(va, ordinary)]
            ix = np.argsort(-sim, axis=1)[:, :k]
            cutoff = np.sort(sim, axis=1)[:, -k]
            weights = np.where(sim >= cutoff[:, None]-1e-12, sim**power, 0)
            nearest = weights@y[ordinary]/np.maximum(weights.sum(axis=1), 1e-9)
            pred = 10000*p+(1-p)*(.5*q+.5*nearest)
            scores[name].append(float(mean_squared_error(y[va], pred)))
            ensemble.append(pred)
        scores['neighbor_ensemble'].append(float(mean_squared_error(y[va], np.mean(ensemble, axis=0))))
    report['suites'][suite] = scores
    print(suite, {k: round(np.mean(v)) for k, v in scores.items()}, flush=True)
Path('neighbor_validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
