"""Fixed half/half blend registered in the all-target-neighbor OOF screen."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from vacv_zero_hurdle import predict as hurdle_predict
from vacv_direct_neighbors import predict as direct_predict


def predict(model, smiles):
    return .5*hurdle_predict(model['hurdle'], smiles) + .5*direct_predict(model['direct'], smiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['hurdle', 'direct', 'test', 'sample', 'model', 'output', 'report']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(Path('zero_v2_direct_morgan_blend_plan.json').read_text())
    if not plan['qualified'] or plan['weight'] != .5:
        raise ValueError('Fixed blend has not passed the registered check')
    model = {'hurdle': joblib.load(args.hurdle), 'direct': joblib.load(args.direct),
             'inputs': ['SMILES'], 'weight': .5}
    h, d = model['hurdle'], model['direct']
    if h['target_version'] != 'train_v2' or d['target_version'] != 'train_v2' or not np.array_equal(h['y'], d['y']) or list(h['train_smiles']) != list(d['train_smiles']):
        raise ValueError('Target or molecule snapshot mismatch')
    if h['calibration'] != 'raw' or h['conditional'] != 'source_stereo_only' or h.get('neighbor_feature') or d['feature'] != 'morgan_count_r2':
        raise ValueError('Unexpected blend components')
    if not h['external'].equals(d['external']):
        raise ValueError('External source snapshot mismatch')
    test, sample = (pd.read_csv(p) for p in [args.test, args.sample])
    if len(test) != 597 or len(sample) != 597 or list(sample) != ['index', 'SI VACV'] or not sample['index'].is_unique or not np.array_equal(test['index'], sample['index']):
        raise ValueError('Invalid submission order')
    pred = predict(model, test.SMILES)
    assert np.isfinite(pred).all() and (pred >= 0).all()
    mutated = test.copy()
    for column in test:
        if column != 'SMILES':
            mutated[column] = 'MUTATED'
    assert np.array_equal(pred, predict(model, mutated.SMILES))
    assert np.array_equal(pred, predict(model, test[['SMILES']].SMILES))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    assert np.array_equal(pred, predict(joblib.load(args.model), test.SMILES))
    pd.DataFrame({'index': sample['index'], 'SI VACV': pred}).to_csv(args.output, index=False)
    report = {'inputs': ['SMILES'], 'weight': .5, 'weight_selected_in_prepublic_training_screen': True,
              'first15_mse': plan['first15_mse'], 'confirmation10_mse': plan['pooled_mse'],
              'checks': {'rows': 597, 'sample_order': True, 'finite_nonnegative': True,
                         'forbidden_columns_independent': True, 'saved_model_identical': True},
              'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.hurdle, args.direct, args.test, args.sample]},
              'csv_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
