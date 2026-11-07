"""Confirm one frozen molecular-neighbor recipe on grouped corrected labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from chemical_neighbors import nearest_recipe
from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
from vacv_zero_hurdle import external_source, fit_classifier, raw_probability


def main():
    selected = json.loads(Path('zero_v2_neighbor_screen.json').read_text())
    recipe = selected['recipes'][selected['selected']]
    if not selected['qualified_for_confirmation'] or recipe['kind'] != 'neighbors':
        raise ValueError('No qualified frozen nearest-neighbor recipe')
    d = pd.read_csv('data/kaggle/train_v2.csv')
    y = d['SI VACV'].to_numpy(float)
    external = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), d.SMILES)
    _, groups = featurize(d.SMILES)
    syntax = featurize_syntax(d.SMILES)
    cache = {}
    report = {'inputs': ['SMILES'], 'selected_before_confirmation': selected['selected'],
              'recipe': recipe, 'source_purge': 'all train molecule copies excluded', 'suites': {}}
    for suite in ['confirmation67121', 'fresh82741']:
        if suite == 'confirmation67121':
            old = np.load('artifacts/zero_v2_confirmation.npz')
            splits = [(old[f'{suite}_{fold}_tr'], old[f'{suite}_{fold}_va']) for fold in range(5)]
        else:
            splits = list(GroupKFold(n_splits=5, shuffle=True, random_state=82741).split(d.SMILES, y, groups))
        scores = {'current': [], 'candidate': []}
        for fold, (tr, va) in enumerate(splits):
            key = suite+'_'+str(fold)
            if suite == 'confirmation67121':
                prob = old[key+'_p_raw']
                current_q = old[key+'_q_source_stereo_only']
            else:
                classifier = fit_classifier(d.SMILES.iloc[tr], y[tr] == 0, syntax[tr])
                prob = raw_probability(classifier, d.SMILES.iloc[va], syntax[va])
            positive = tr[y[tr] > 0]
            reference = pd.concat([d.SMILES.iloc[positive], external.SMILES], ignore_index=True)
            values = np.r_[y[positive], external.SI_derived.to_numpy(float)]
            if suite != 'confirmation67121':
                current_q = nearest_recipe(reference, values, d.SMILES.iloc[va])
            q = nearest_recipe(reference, values, d.SMILES.iloc[va], recipe['feature'], recipe['k'], recipe['power'])
            for name, conditional in [('current', current_q), ('candidate', q)]:
                prediction = (1-prob)*conditional
                scores[name].append(float(np.mean((y[va]-prediction)**2)))
                cache[key+'_'+name] = prediction
            cache[key+'_tr'], cache[key+'_va'], cache[key+'_p'] = tr, va, prob
            print(suite, fold, {k: round(v[-1]) for k,v in scores.items()}, flush=True)
        report['suites'][suite] = scores
        Path('zero_v2_neighbor_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed('artifacts/zero_v2_neighbor_confirmation.npz', **cache)
        print(suite, {k: round(np.mean(v)) for k,v in scores.items()}, flush=True)
    report['qualified'] = all(np.mean(s['candidate']) < .99*np.mean(s['current']) for s in report['suites'].values())
    Path('zero_v2_neighbor_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('qualified', report['qualified'], flush=True)


if __name__ == '__main__':
    main()
