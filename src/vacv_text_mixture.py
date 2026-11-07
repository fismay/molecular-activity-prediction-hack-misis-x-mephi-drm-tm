"""Predict SI from original SMILES text and structure; no assay metadata inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.special import logit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from vacv_baseline import featurize
from smiles_syntax import featurize_syntax


def fit(smiles: pd.Series, y: np.ndarray, syntax_ensemble: bool = False,
        conditional_transform: str = 'raw', syntax_max_features: float = 0.7,
        calibrate_probability: bool = False) -> dict:
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 7), lowercase=False)
    text = vectorizer.fit_transform(smiles)
    classifier = LogisticRegression(C=100, max_iter=3000).fit(text, y == 10000)
    x, _ = featurize(smiles)
    ordinary = y != 10000
    target = y[ordinary]
    if conditional_transform == 'sqrt':
        target = np.sqrt(target)
    elif conditional_transform == 'log':
        target = np.log1p(target)
    regressor = RandomForestRegressor(
        n_estimators=250, min_samples_leaf=1, max_features=0.3,
        random_state=42, n_jobs=-1,
    ).fit(x[ordinary], target)
    result = {'vectorizer': vectorizer, 'classifier': classifier, 'regressor': regressor,
              'conditional_transform': conditional_transform}
    if syntax_ensemble:
        result['syntax_classifier'] = ExtraTreesClassifier(
            n_estimators=400, max_features=syntax_max_features, random_state=42, n_jobs=-1,
        ).fit(featurize_syntax(smiles), y == 10000)
    if calibrate_probability:
        _, groups = featurize(smiles)
        oof = np.zeros(len(y))
        for tr, va in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(x, y, groups):
            inner = fit(smiles.iloc[tr], y[tr], syntax_ensemble,
                        conditional_transform, syntax_max_features, False)
            oof[va] = probability(inner, smiles.iloc[va])
        result['probability_calibrator'] = LogisticRegression(C=1, max_iter=2000).fit(
            logit(np.clip(oof, 1e-8, 1-1e-8)).reshape(-1, 1), y == 10000)
    return result


def probability(model: dict, smiles: pd.Series) -> np.ndarray:
    text = model['vectorizer'].transform(smiles)
    classifier = model['classifier']
    p = classifier.predict_proba(text)[:, list(classifier.classes_).index(True)]
    if 'syntax_classifier' in model:
        syntax = model['syntax_classifier']
        syntax_probability = syntax.predict_proba(featurize_syntax(smiles))[:, list(syntax.classes_).index(True)]
        p = 0.5 * p + 0.5 * syntax_probability
    if 'probability_calibrator' in model:
        calibrator = model['probability_calibrator']
        p = calibrator.predict_proba(logit(np.clip(p, 1e-8, 1-1e-8)).reshape(-1, 1))[:, 1]
    return p


def predict(model: dict, smiles: pd.Series) -> np.ndarray:
    p = probability(model, smiles)
    x, _ = featurize(smiles)
    conditional = np.maximum(0, model['regressor'].predict(x))
    if model.get('conditional_transform') == 'sqrt':
        conditional = conditional ** 2
    elif model.get('conditional_transform') == 'log':
        conditional = np.expm1(conditional)
    return p * 10000 + (1 - p) * conditional


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission_smiles_text.csv'))
    parser.add_argument('--report', type=Path, default=Path('smiles_text_report.json'))
    parser.add_argument('--model', type=Path, default=Path('artifacts/smiles_text.joblib'))
    parser.add_argument('--syntax-ensemble', action='store_true')
    parser.add_argument('--syntax-max-features', type=float, default=0.7)
    parser.add_argument('--conditional-transform', choices=('raw', 'sqrt', 'log'), default='raw')
    parser.add_argument('--calibrate-probability', action='store_true')
    args = parser.parse_args()
    train, test, sample = (pd.read_csv(path) for path in (args.train, args.test, args.sample))
    if list(sample.columns) != ['index', 'SI VACV'] or len(sample) != len(test):
        raise ValueError('Unexpected sample submission')
    y = train['SI VACV'].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError('Non-finite SI labels')
    x, groups = featurize(train['SMILES'])
    scores = {}
    for seed in (42, 2026, 17):
        fold_scores = []
        for tr, va in GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=seed).split(x, y, groups):
            model = fit(train['SMILES'].iloc[tr], y[tr], args.syntax_ensemble,
                        args.conditional_transform, args.syntax_max_features,
                        args.calibrate_probability)
            prediction = predict(model, train['SMILES'].iloc[va])
            fold_scores.append(float(mean_squared_error(y[va], prediction)))
        scores[str(seed)] = fold_scores
    model = fit(train['SMILES'], y, args.syntax_ensemble,
                args.conditional_transform, args.syntax_max_features,
                args.calibrate_probability)
    prediction = predict(model, test['SMILES'])
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite prediction')
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    output = pd.DataFrame({'index': sample['index'], 'SI VACV': prediction})
    output.to_csv(args.output, index=False)
    report = {
        'inputs': ['SMILES'], 'metric': 'MSE on original SI scale',
        'grouped_validation': scores,
        'mean_mse': float(np.mean([s for fold in scores.values() for s in fold])),
        'classifier': 'LogisticRegression(C=100, max_iter=3000)',
        'syntax_ensemble': args.syntax_ensemble,
        'syntax_classifier': f'ExtraTrees(n_estimators=400, max_features={args.syntax_max_features}, random_state=42)' if args.syntax_ensemble else None,
        'syntax_ensemble_weight': 0.5 if args.syntax_ensemble else None,
        'text': {'ngram_range': [3, 7], 'lowercase': False, 'canonicalized': False},
        'conditional_regressor': 'RandomForest(n_estimators=250, max_features=0.3, min_samples_leaf=1, random_state=42)',
        'conditional_transform': args.conditional_transform,
        'probability_calibration': 'LogisticRegression(C=1) on logit probabilities from inner 3-fold GroupKFold' if args.calibrate_probability else None,
        'test_rows': len(test), 'sklearn_version': sklearn.__version__,
        'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.train, args.test, args.sample)},
        'limitation': 'Original SMILES syntax can encode source-specific formatting; predictions may change under an equivalent SMILES serialization.',
    }
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mean_mse': report['mean_mse'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
