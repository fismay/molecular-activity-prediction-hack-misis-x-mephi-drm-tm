"""Blend calibrated SMILES model with three nearest non-10000 structures."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as F
from vacv_baseline import FP
from vacv_text_mixture import probability


def nearest(smiles_train, y, smiles_test):
    keep = np.flatnonzero(y != 10000)
    reference = [FP.GetFingerprint(Chem.MolFromSmiles(smiles_train.iloc[i])) for i in keep]
    values = y[keep]
    out = []
    for s in smiles_test:
        fp = FP.GetFingerprint(Chem.MolFromSmiles(s))
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, reference))
        ix = np.argsort(-sims)[:3]
        weights = sims[ix] ** 4
        out.append(float(np.dot(weights, values[ix]) / max(weights.sum(), 1e-9)))
    return np.asarray(out)


def nearest_ensemble(smiles_train, y, smiles_test):
    keep = np.flatnonzero(y != 10000)
    train_mols = [Chem.MolFromSmiles(smiles_train.iloc[i]) for i in keep]
    test_mols = [Chem.MolFromSmiles(s) for s in smiles_test]
    recipes = [(FP, False, 3, 4),
               (F.GetMorganGenerator(radius=2, includeChirality=True), True, 1, 2),
               (F.GetRDKitFPGenerator(), True, 1, 2)]
    predictions = []
    for gen, counts, k, power in recipes:
        def make(m):
            return gen.GetCountFingerprint(m) if counts else gen.GetFingerprint(m)
        reference = [make(m) for m in train_mols]
        out = []
        for m in test_mols:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(make(m), reference))
            cutoff = np.sort(sims)[-k]
            weights = np.where(sims >= cutoff-1e-12, sims**power, 0)
            out.append(float(weights@y[keep]/max(weights.sum(), 1e-9)))
        predictions.append(out)
    return np.mean(predictions, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ensemble', action='store_true')
    args = parser.parse_args()
    train, test, sample = (pd.read_csv(p) for p in (args.train, args.test, args.sample))
    if list(sample.columns) != ['index', 'SI VACV'] or len(sample) != len(test):
        raise ValueError('Unexpected sample submission')
    y = train['SI VACV'].to_numpy(float)
    model = joblib.load(args.model)
    if 'probability_calibrator' not in model:
        raise ValueError('Expected calibrated model')
    p = probability(model, test.SMILES)
    from vacv_baseline import featurize
    x, _ = featurize(test.SMILES)
    rf = np.maximum(0, model['regressor'].predict(x))
    kn = (nearest_ensemble if args.ensemble else nearest)(train.SMILES, y, test.SMILES)
    pred = p*10000 + (1-p)*(.5*rf+.5*kn)
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index':sample['index'],'SI VACV':pred}).to_csv(args.output,index=False)
    print(json.dumps({'rows':len(pred),'min':float(pred.min()),'max':float(pred.max()),'output':str(args.output)}))

if __name__ == '__main__':
    main()
