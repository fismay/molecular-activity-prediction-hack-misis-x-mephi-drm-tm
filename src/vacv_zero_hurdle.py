"""Train_v2 SMILES-only regression with an observed-zero-label hurdle.

The zero component is a statistical property of the supplied target labels,
not a claim of measured biological inactivity. No Kaggle concentration or
metadata column enters any fitted transform, classifier, or regressor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from smiles_syntax import featurize_syntax
from stereo_neighbors import nearest_stereo
from vacv_baseline import featurize
from vacv_external_neighbors import validate_source
from vacv_knn_blend import nearest_ensemble

TRAIN_V2_SHA256 = 'd9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8'


def fit_classifier(smiles, label, syntax=None):
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 7), lowercase=False)
    text = vectorizer.fit_transform(smiles)
    text_model = LogisticRegression(C=100, max_iter=3000).fit(text, label)
    if syntax is None:
        syntax = featurize_syntax(smiles)
    syntax_model = ExtraTreesClassifier(n_estimators=400, max_features=.7,
                                       random_state=42, n_jobs=1).fit(syntax, label)
    return {'vectorizer': vectorizer, 'text': text_model, 'syntax': syntax_model}


def raw_probability(model, smiles, syntax=None):
    if syntax is None:
        syntax = featurize_syntax(smiles)
    return .5 * model['text'].predict_proba(model['vectorizer'].transform(smiles))[:, 1] + .5 * model['syntax'].predict_proba(syntax)[:, 1]


def fit(smiles, y, structural=None, syntax=None):
    smiles = smiles.reset_index(drop=True)
    if structural is None:
        structural, groups = featurize(smiles)
    else:
        _, groups = featurize(smiles)
    if syntax is None:
        syntax = featurize_syntax(smiles)
    zero = y == 0
    if len(np.unique(zero)) != 2 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Expected finite nonnegative targets and both zero/positive labels')
    inner = np.zeros(len(y))
    for tr, va in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(smiles, y, groups):
        classifier = fit_classifier(smiles.iloc[tr], zero[tr], syntax[tr])
        inner[va] = raw_probability(classifier, smiles.iloc[va], syntax[va])
    logits = logit(np.clip(inner, 1e-8, 1-1e-8))

    def objective(w):
        return np.mean((expit(w[0] * logits + w[1]) - zero) ** 2) + .001 * ((w[0]-1)**2 + w[1]**2)

    opt = minimize(objective, [1., 0.], method='L-BFGS-B', bounds=[(0, 10), (-10, 10)])
    if not opt.success:
        raise ValueError('Brier calibration failed: ' + opt.message)
    regressor = RandomForestRegressor(n_estimators=250, max_features=.3,
                                      min_samples_leaf=1, random_state=42, n_jobs=1)
    regressor.fit(structural[~zero], y[~zero])
    return {'classifier': fit_classifier(smiles, zero, syntax),
            'regressor': regressor, 'brier': opt.x,
            'isotonic': IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(inner, zero),
            'train_smiles': smiles, 'y': np.asarray(y, float),
            'inputs': ['SMILES'], 'target_version': 'train_v2'}


def probabilities(model, smiles, syntax=None):
    raw = raw_probability(model['classifier'], smiles, syntax)
    a, b = model['brier']
    brier = expit(a * logit(np.clip(raw, 1e-8, 1-1e-8)) + b)
    iso = model['isotonic'].predict(raw)
    return {'raw': raw, 'brier': brier, 'isotonic_half': .5*brier + .5*iso,
            'isotonic': iso}


def external_source(path, all_train_smiles):
    source = pd.read_csv(path)
    validate_source(source)
    _, groups = featurize(all_train_smiles)
    _, external_groups = featurize(source.SMILES)
    # Purge exact copies of every outer train OR validation molecule.
    return source.loc[~np.isin(external_groups, groups), ['SMILES', 'SI_derived']].reset_index(drop=True)


def conditional_components(model, smiles, external=None, structural=None):
    if structural is None:
        structural, _ = featurize(smiles)
    forest = np.maximum(0, model['regressor'].predict(structural))
    positive = model['y'] > 0
    reference = model['train_smiles'].loc[positive].reset_index(drop=True)
    values = model['y'][positive]
    ensemble = nearest_ensemble(reference, values, smiles, exclude_value=None)
    result = {'forest': forest, 'neighbors': .5*forest + .5*ensemble}
    if external is not None and len(external):
        reference = pd.concat([reference, external.SMILES], ignore_index=True)
        values = np.r_[values, external.SI_derived.to_numpy(float)]
        source_ensemble = nearest_ensemble(reference, values, smiles, exclude_value=None)
        stereo = nearest_stereo(reference, values, smiles, exclude_value=None)
        result['source_ensemble'] = .5*forest + .5*source_ensemble
        result['source_stereo'] = .5*forest + .5*stereo
        result['source_stereo_only'] = stereo
    return result


def predict(model, smiles):
    if model.get('strategy') == 'forest':
        x, _ = featurize(smiles)
        return np.maximum(0, model['regressor'].predict(x))
    p = probabilities(model, smiles)[model['calibration']]
    if model.get('neighbor_feature'):
        from chemical_neighbors import nearest_recipe
        positive = model['y'] > 0
        reference = model['train_smiles'].loc[positive].reset_index(drop=True)
        values = model['y'][positive]
        external = model.get('external')
        if external is not None:
            reference = pd.concat([reference, external.SMILES], ignore_index=True)
            values = np.r_[values, external.SI_derived.to_numpy(float)]
        q = nearest_recipe(reference, values, smiles, model['neighbor_feature'],
                           model.get('neighbor_k', 1), model.get('neighbor_power', 2))
    else:
        q = conditional_components(model, smiles, model.get('external'))[model['conditional']]
    result = (1-p) * q
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid predictions')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['train', 'test', 'sample', 'model', 'output', 'report']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--strategy', choices=['hurdle', 'forest'], default='hurdle')
    parser.add_argument('--calibration', choices=['raw', 'brier', 'isotonic_half', 'isotonic'], default='raw')
    parser.add_argument('--conditional', choices=['forest', 'neighbors', 'source_ensemble', 'source_stereo', 'source_stereo_only'], default='source_stereo_only')
    parser.add_argument('--neighbor-feature')
    parser.add_argument('--neighbor-k', type=int, default=1)
    parser.add_argument('--neighbor-power', type=int, default=2)
    a = parser.parse_args()
    if a.neighbor_feature:
        from chemical_neighbors import FEATURES
        if a.neighbor_feature not in FEATURES or a.strategy != 'hurdle' or a.conditional != 'source_stereo_only':
            raise ValueError('Custom fingerprint requires the positive-neighbor hurdle recipe')
    if hashlib.sha256(a.train.read_bytes()).hexdigest() != TRAIN_V2_SHA256:
        raise ValueError('Unexpected target snapshot; expected audited train_v2.csv')
    train, test, sample = (pd.read_csv(p) for p in [a.train, a.test, a.sample])
    if len(test) != 597 or list(sample) != ['index', 'SI VACV'] or not sample['index'].is_unique or not np.array_equal(test['index'], sample['index']):
        raise ValueError('Invalid sample format/order')
    if a.strategy == 'forest':
        x, _ = featurize(train.SMILES)
        y = train['SI VACV'].to_numpy(float)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Invalid targets')
        regressor = RandomForestRegressor(n_estimators=250, max_features=.3,
                                          min_samples_leaf=1, random_state=42, n_jobs=1).fit(x, y)
        model = {'strategy': 'forest', 'regressor': regressor, 'y': y,
                 'inputs': ['SMILES'], 'target_version': 'train_v2',
                 'external': None, 'calibration': None, 'conditional': None}
    else:
        if a.conditional.startswith('source_') and not a.source:
            raise ValueError('Source recipe requires an audited external source')
        model = fit(train.SMILES, train['SI VACV'].to_numpy(float))
        model.update(calibration=a.calibration, conditional=a.conditional,
                     external=external_source(a.source, train.SMILES) if a.source else None,
                     neighbor_feature=a.neighbor_feature, neighbor_k=a.neighbor_k,
                     neighbor_power=a.neighbor_power)
    predictions = predict(model, test.SMILES)
    mutated = test.copy()
    for column in test:
        if column not in ['index', 'SMILES']:
            mutated[column] = 'MUTATED'
    assert np.array_equal(predictions, predict(model, mutated.SMILES))
    assert np.array_equal(predictions, predict(model, test[['SMILES']].SMILES))
    a.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, a.model)
    assert np.array_equal(predictions, predict(joblib.load(a.model), test.SMILES))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': sample['index'], 'SI VACV': predictions}).to_csv(a.output, index=False)
    report = {'inputs': ['SMILES'], 'target_snapshot': 'train_v2.csv',
              'zero_labels': int((model['y'] == 0).sum()), 'strategy': a.strategy,
              'prediction': '(1 - probability_of_zero_label) * conditional_SI' if a.strategy == 'hurdle' else 'RandomForestRegressor on all corrected targets',
              'calibration': model['calibration'], 'conditional': model['conditional'],
              'neighbor_feature': model.get('neighbor_feature'),
              'neighbor_k': model.get('neighbor_k'), 'neighbor_power': model.get('neighbor_power'),
              'external_novel_rows': len(model['external']) if model['external'] is not None else 0,
              'checks': {'rows': 597, 'exact_order': True, 'finite_nonnegative': True,
                         'forbidden_columns_mutation_identical': True,
                         'other_columns_removed_identical': True, 'saved_model_identical': True},
              'range': [float(predictions.min()), float(predictions.max())],
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.train, a.test, a.sample, a.output]}}
    if a.source:
        report['source_sha256'] = hashlib.sha256(a.source.read_bytes()).hexdigest()
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
