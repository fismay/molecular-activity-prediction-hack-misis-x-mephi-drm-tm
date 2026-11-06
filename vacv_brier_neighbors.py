"""SMILES-only neighbor ensemble with nested Brier or final-MSE calibration."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from vacv_baseline import featurize
from vacv_text_mixture import fit, probability
from vacv_knn_blend import nearest_ensemble


def fit_brier(smiles, y, base, final_mse=False):
    x, groups = featurize(smiles)
    inner = np.zeros(len(y))
    inner_q = np.zeros(len(y))
    for a,b in GroupKFold(n_splits=3, shuffle=True, random_state=42).split(smiles,y,groups):
        model = fit(smiles.iloc[a], y[a], True)
        inner[b] = probability(model, smiles.iloc[b])
        if final_mse:
            inner_q[b] = .5*np.maximum(0,model['regressor'].predict(x[b]))+.5*nearest_ensemble(
                smiles.iloc[a].reset_index(drop=True),y[a],smiles.iloc[b])
    f = logit(np.clip(inner, 1e-8, 1-1e-8))
    labels = (y == 10000).astype(float)
    def objective(w):
        if final_mse:
            p = expit(w[0]*f+w[1])
            pred = p+(1-p)*inner_q*w[2]/10000
            return np.mean((pred-y/10000)**2)+.001*((w[0]-1)**2+w[1]**2+(w[2]-1)**2)
        return np.mean((expit(w[0]*f+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
    initial = [1.,0.,1.] if final_mse else [1.,0.]
    bounds = [(0,10),(-10,10)]+([(0,2)] if final_mse else [])
    opt = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)
    if not opt.success:
        raise ValueError('Calibration optimization failed: '+opt.message)
    base = dict(base)
    base.pop('probability_calibrator', None)
    return {'base': base, 'brier_parameters': opt.x[:2],
            'conditional_scale':float(opt.x[2]) if final_mse else 1.,
            'calibration_method':'final_mse' if final_mse else 'brier',
            'train_smiles': smiles.reset_index(drop=True), 'y': y}


def predict(model, smiles):
    p = probability(model['base'], smiles)
    slope, intercept = model['brier_parameters']
    p = expit(slope*logit(np.clip(p,1e-8,1-1e-8))+intercept)
    x, _ = featurize(smiles)
    q = np.maximum(0,model['base']['regressor'].predict(x))
    kn = nearest_ensemble(model['train_smiles'],model['y'],smiles)
    return p*10000+(1-p)*(.5*q+.5*kn)*model.get('conditional_scale',1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['train','test','sample','base-model','model','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--final-mse',action='store_true')
    args = parser.parse_args()
    train,test,sample = (pd.read_csv(p) for p in [args.train,args.test,args.sample])
    if list(sample.columns) != ['index','SI VACV'] or len(sample) != len(test):
        raise ValueError('Unexpected sample')
    model = fit_brier(train.SMILES,train['SI VACV'].to_numpy(float),joblib.load(args.base_model),args.final_mse)
    args.model.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,args.model)
    prediction = predict(joblib.load(args.model),test.SMILES)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite predictions')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':sample['index'],'SI VACV':prediction}).to_csv(args.output,index=False)
    print(json.dumps({'rows':len(prediction),'brier_parameters':model['brier_parameters'].tolist(),
                      'conditional_scale':model['conditional_scale'],'output':str(args.output)}))

if __name__ == '__main__':
    main()
