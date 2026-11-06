"""Audit distribution shift using SMILES only; no private rows in report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def main():
    train = pd.read_csv('data/kaggle/train.csv', usecols=['SMILES'])
    test = pd.read_csv('data/kaggle/test.csv', usecols=['SMILES'])
    raw = pd.concat([train, test]).SMILES.astype(str).to_numpy()
    mols = [Chem.MolFromSmiles(s) for s in raw]
    if any(m is None for m in mols):
        raise ValueError('Invalid structure')
    canonical = np.array([Chem.MolToSmiles(m) for m in mols])
    scaffolds = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols])
    n = len(train)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    nearest = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, fps[:n])) for fp in fps[n:]])
    labels = np.r_[np.zeros(n), np.ones(len(test))]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=20261107)
    pipeline = make_pipeline(TfidfVectorizer(analyzer='char', ngram_range=(3,7), min_df=2),
                             LogisticRegression(C=1, max_iter=2000))
    prob = cross_val_predict(pipeline, raw, labels, groups=canonical, cv=cv,
                            method='predict_proba', n_jobs=1)[:,1]
    report = {
        'input_columns': ['SMILES'], 'train_rows': n, 'test_rows': len(test),
        'exact_canonical_overlap_test_rows': int(np.isin(canonical[n:], canonical[:n]).sum()),
        'test_rows_with_unseen_scaffold': int((~np.isin(scaffolds[n:], scaffolds[:n])).sum()),
        'test_empty_scaffold_rows': int((scaffolds[n:] == '').sum()),
        'nearest_train_tanimoto_quantiles': dict(zip(['min','q25','median','q75','max'],
                                                    np.quantile(nearest,[0,.25,.5,.75,1]).tolist())),
        'test_rows_nearest_tanimoto_below_0_3': int((nearest < .3).sum()),
        'grouped_domain_classifier_auc': float(roc_auc_score(labels, prob)),
        'interpretation': 'AUC measures distinguishability of train/test SMILES, not target prediction quality. Empty Murcko scaffold is a shared acyclic category, not evidence of chemical similarity.'
    }
    Path('train_test_shift_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
