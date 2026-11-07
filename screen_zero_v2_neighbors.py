"""Screen chemical similarity recipes using corrected-label grouped OOF only."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from chemical_neighbors import FEATURES, similarities, neighbor_values
from vacv_zero_hurdle import external_source


def main():
    d = pd.read_csv('data/kaggle/train_v2.csv')
    y = d['SI VACV'].to_numpy(float)
    external = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), d.SMILES)
    smiles = pd.concat([d.SMILES, external.SMILES], ignore_index=True)
    target = np.r_[y, external.SI_derived.to_numpy(float)]
    source_indices = np.arange(len(d), len(smiles))
    cached = np.load('artifacts/zero_v2_validation.npz')
    report = {'inputs': ['SMILES'], 'selection': 'first15 grouped OOF; >=1% lower mean MSE; confirm fixed winner on separate folds before any public score',
              'source_purge': 'canonical copies of all320 train molecules excluded', 'suites': {}, 'recipes': {}}
    matrices = {feature: similarities(smiles, smiles, feature) for feature in FEATURES}
    kernel_features = ['morgan_sparse_r2', 'morgan_sparse_r3', 'atom_pair_sparse']
    kernels = {feature: similarities(smiles, smiles, feature, gate=False) for feature in kernel_features}
    predictions = {}
    for suite in ['42', '2026', '17']:
        scores = {}
        for fold in range(5):
            key = suite+'_'+str(fold)
            tr, va = cached[key+'_tr'], cached[key+'_va']
            reference = np.r_[tr[y[tr] > 0], source_indices]
            values = target[reference]
            p = cached[key+'_p_raw']
            baseline_q = cached[key+'_q_source_stereo_only']
            baseline = (1-p)*baseline_q
            scores.setdefault('current', []).append(float(np.mean((y[va]-baseline)**2)))
            for feature, matrix in matrices.items():
                sim = matrix[np.ix_(va, reference)]
                for k, power in [(1, 2), (3, 4), (5, 4)]:
                    q = neighbor_values(sim, values, k, power)
                    if feature == 'morgan_count_r2' and k == 1:
                        assert np.allclose(q, baseline_q, rtol=1e-12, atol=1e-8)
                    name = feature+f'_k{k}_p{power}'
                    pred = (1-p)*q
                    scores.setdefault(name, []).append(float(np.mean((y[va]-pred)**2)))
                    report['recipes'][name] = {'kind': 'neighbors', 'feature': feature, 'k': k, 'power': power}
                    predictions[key+'__'+name] = pred
            for feature, matrix in kernels.items():
                gram = matrix[np.ix_(reference, reference)]
                query = matrix[np.ix_(va, reference)]
                for alpha in [.01, .1, 1.]:
                    coefficients = np.linalg.solve(gram+alpha*np.eye(len(reference)), values)
                    q = np.maximum(0, query @ coefficients)
                    name = feature+f'_kernel_alpha{alpha}'
                    pred = (1-p)*q
                    scores.setdefault(name, []).append(float(np.mean((y[va]-pred)**2)))
                    report['recipes'][name] = {'kind': 'kernel', 'feature': feature, 'alpha': alpha}
                    predictions[key+'__'+name] = pred
        report['suites'][suite] = scores
        print(suite, sorted(((name, round(np.mean(v))) for name, v in scores.items()), key=lambda z:z[1])[:8], flush=True)
    report['mean_mse'] = {name: float(np.mean([v for suite in report['suites'].values() for v in suite[name]])) for name in report['suites']['42']}
    ranked = sorted(report['mean_mse'], key=report['mean_mse'].get)
    report['selected'] = ranked[0]
    report['qualified_for_confirmation'] = report['mean_mse'][ranked[0]] < .99*report['mean_mse']['current']
    Path('zero_v2_neighbor_screen.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed('artifacts/zero_v2_neighbor_screen.npz', **predictions)
    print('best', [(name, round(report['mean_mse'][name])) for name in ranked[:12]], flush=True)
    print('qualified', report['qualified_for_confirmation'], flush=True)


if __name__ == '__main__':
    main()
