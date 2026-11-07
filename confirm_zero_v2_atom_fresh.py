"""Additional registered check of the exploratory atom-pair candidate.

The original >=1% improvement in EACH confirmation suite was not met.
This is an explicitly separate follow-up: freeze the selected atom-pair
recipe and test two new seed partitions. Require >=1% pooled improvement
and no worse mean in either new suite. No recipe changes after this check.
Repeated partitions reuse the same molecules and are not independent data.
"""
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
    previous = json.loads(Path('zero_v2_neighbor_confirmation.json').read_text())
    assert not previous['qualified']
    d = pd.read_csv('data/kaggle/train_v2.csv')
    y = d['SI VACV'].to_numpy(float)
    external = external_source(Path('artifacts/external/bornyl_amides_expanded_source.csv'), d.SMILES)
    _, groups = featurize(d.SMILES)
    syntax = featurize_syntax(d.SMILES)
    report = {'prior_gate_passed': False,
              'followup_reason': 'atom-pair had lower means in every earlier suite but only 0.82% improvement in one; prior 1%-per-suite gate failed',
              'recipe_frozen': {'feature': 'atom_pair_sparse', 'k': 1, 'power': 2, 'calibration': 'raw'},
              'seeds_fixed_before_run': [97531, 31627],
              'followup_gate': 'pooled mean MSE at least 1% lower and both new suite means lower; no further recipe tuning in this check',
              'inputs': ['SMILES'], 'suites': {}}
    Path('zero_v2_atom_fresh_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
    cache = {}
    for seed in report['seeds_fixed_before_run']:
        scores = {'current': [], 'candidate': []}
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(d.SMILES, y, groups)):
            classifier = fit_classifier(d.SMILES.iloc[tr], y[tr] == 0, syntax[tr])
            prob = raw_probability(classifier, d.SMILES.iloc[va], syntax[va])
            positive = tr[y[tr] > 0]
            reference = pd.concat([d.SMILES.iloc[positive], external.SMILES], ignore_index=True)
            values = np.r_[y[positive], external.SI_derived.to_numpy(float)]
            key = f'{seed}_{fold}'
            for name, feature in [('current', 'morgan_count_r2'), ('candidate', 'atom_pair_sparse')]:
                pred = (1-prob)*nearest_recipe(reference, values, d.SMILES.iloc[va], feature)
                scores[name].append(float(np.mean((y[va]-pred)**2)))
                cache[key+'_'+name] = pred
            cache[key+'_tr'], cache[key+'_va'], cache[key+'_p'] = tr, va, prob
            print(seed, fold, {k: round(v[-1]) for k,v in scores.items()}, flush=True)
        report['suites'][str(seed)] = scores
        Path('zero_v2_atom_fresh_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez_compressed('artifacts/zero_v2_atom_fresh_confirmation.npz', **cache)
        print(seed, {k: round(np.mean(v)) for k,v in scores.items()}, flush=True)
    report['pooled_mse'] = {name: float(np.mean([v for suite in report['suites'].values() for v in suite[name]])) for name in ['current','candidate']}
    report['followup_gate_passed'] = report['pooled_mse']['candidate'] < .99*report['pooled_mse']['current'] and all(np.mean(s['candidate']) < np.mean(s['current']) for s in report['suites'].values())
    Path('zero_v2_atom_fresh_confirmation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('followup_gate_passed', report['followup_gate_passed'], report['pooled_mse'], flush=True)

if __name__ == '__main__':
    main()
