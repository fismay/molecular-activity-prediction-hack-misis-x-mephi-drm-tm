"""Compare SMILES-only pipelines against the corrected train_v2 targets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from smiles_syntax import featurize_syntax
from vacv_baseline import featurize
from vacv_zero_hurdle import TRAIN_V2_SHA256, conditional_components, external_source, fit, probabilities


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--confirm', help='Fixed recipe selected before the new confirmation splits')
    a = p.parse_args()
    train_path = Path('data/kaggle/train_v2.csv')
    if hashlib.sha256(train_path.read_bytes()).hexdigest() != TRAIN_V2_SHA256:
        raise ValueError('Unexpected corrected target snapshot')
    d = pd.read_csv(train_path)
    old = pd.read_csv('data/kaggle/train.csv')['SI VACV'].to_numpy(float)
    smiles = d.SMILES
    y = d['SI VACV'].to_numpy(float)
    x, groups = featurize(smiles)
    syntax = featurize_syntax(smiles)
    external = external_source(a.source, smiles)
    if a.confirm:
        suites = {'confirmation67121': list(GroupKFold(n_splits=5, shuffle=True, random_state=67121).split(x, y, groups))}
    else:
        suites = {str(seed): list(GroupShuffleSplit(n_splits=5, test_size=.2, random_state=seed).split(x, y, groups)) for seed in [42, 2026, 17]}
    report = {'target_sha256': TRAIN_V2_SHA256, 'inputs': ['SMILES'],
              'zero_labels': int((y == 0).sum()), 'external_novel_rows': len(external),
              'source_purge': 'all canonical copies of any outer training OR validation molecule excluded',
              'probability_calibration': 'inner3 canonical GroupKFold42 only',
              'selected_recipe_before_confirmation': a.confirm,
              'suites': {}}
    cache = {}
    for suite, splits in suites.items():
        scores = {}
        briers = {}
        for fold, (tr, va) in enumerate(splits):
            model = fit(smiles.iloc[tr], y[tr], x[tr], syntax[tr])
            ps = probabilities(model, smiles.iloc[va], syntax[va])
            qs = conditional_components(model, smiles.iloc[va], external, x[va])
            preds = {}
            for name, values, leaf, fraction in [
                ('old_baseline_evaluated_on_v2', old, 2, .7),
                ('refitted_baseline_v2', y, 2, .7),
                ('refitted_tuned_forest_v2', y, 1, .3),
            ]:
                regressor = RandomForestRegressor(n_estimators=250, min_samples_leaf=leaf,
                                                  max_features=fraction, random_state=42, n_jobs=1)
                preds[name] = np.maximum(0, regressor.fit(x[tr], values[tr]).predict(x[va]))
            for conditional, q in qs.items():
                for calibration, prob in ps.items():
                    name = conditional + '__' + calibration
                    if not a.confirm or name == a.confirm:
                        preds[name] = (1-prob)*q
            if not a.confirm:
                preds['source_stereo_without_hurdle'] = qs['source_stereo']
            if a.confirm and a.confirm not in preds:
                raise ValueError('Unknown confirmation recipe')
            for name, pred in preds.items():
                scores.setdefault(name, []).append(float(np.mean((y[va]-pred)**2)))
                cache[f'{suite}_{fold}_prediction_{name}'] = pred
            for name, prob in ps.items():
                briers.setdefault(name, []).append(float(np.mean((prob-(y[va] == 0))**2)))
                cache[f'{suite}_{fold}_p_{name}'] = prob
            for name, q in qs.items():
                cache[f'{suite}_{fold}_q_{name}'] = q
            cache[f'{suite}_{fold}_tr'] = tr
            cache[f'{suite}_{fold}_va'] = va
            report['suites'][suite] = {'fold_mse': scores, 'class_brier': briers}
            a.report.parent.mkdir(parents=True, exist_ok=True)
            a.cache.parent.mkdir(parents=True, exist_ok=True)
            a.report.write_text(json.dumps(report, indent=2)+'\n')
            np.savez_compressed(a.cache, **cache)
            top = sorted(((k, v[-1]) for k, v in scores.items()), key=lambda z: z[1])[:4]
            print(suite, fold, {k: round(v) for k, v in top}, flush=True)
        report['suites'][suite]['mean_mse'] = {k: float(np.mean(v)) for k, v in scores.items()}
        a.report.write_text(json.dumps(report, indent=2)+'\n')
        print('suite', suite, sorted(((k, round(np.mean(v))) for k, v in scores.items()), key=lambda z: z[1])[:8], flush=True)
    names = next(iter(report['suites'].values()))['fold_mse']
    report['mean_fold_mse'] = {name: float(np.mean([v for s in report['suites'].values() for v in s['fold_mse'][name]])) for name in names}
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print('final', sorted(((k, round(v)) for k, v in report['mean_fold_mse'].items()), key=lambda z: z[1]), flush=True)


if __name__ == '__main__':
    main()
