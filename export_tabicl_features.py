"""Export SMILES-derived features for an isolated optional TabICL experiment."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from smiles_syntax import featurize_syntax, FEATURE_NAMES
from vacv_baseline import featurize

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/tabicl/tabicl_features.npz'))
    args=parser.parse_args()
    train=pd.read_csv(args.train)
    syntax=featurize_syntax(train.SMILES)
    structure,groups=featurize(train.SMILES)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,syntax=syntax,structure=structure,y=train['SI VACV'].to_numpy(float),groups=groups,feature_names=np.asarray(FEATURE_NAMES),train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest())
