"""Reproduce convex blend scores on the 25 cached grouped validation folds."""
import json
import numpy as np
import pandas as pd
from scipy.special import logit,expit
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
D=pd.read_csv('data/kaggle/train.csv');y=D['SI VACV'].to_numpy(float)
base=np.load('artifacts/calibrated_neighbor_validation.npz');stack=np.load('artifacts/probability_stack_cache.npz');cal=json.load(open('end_to_end_calibration_report.json'));report={}
for suite in ['42','2026','17','fresh_groupkfold901','new_groupkfold137']:
    scores={str(w):[] for w in [0,.25,.5,.75,1]}
    for i in range(5):
        key=suite+'_'+str(i);tr=base[key+'_tr'];va=base[key+'_va'];q=stack[key+'_conditional']
        inner=logit(np.clip(stack[key+'_inner'].mean(axis=1),1e-8,1-1e-8));outer=logit(np.clip(stack[key+'_outer'].mean(axis=1),1e-8,1-1e-8));labels=y[tr]==10000
        def loss(w):return np.mean((expit(w[0]*inner+w[1])-labels)**2)+.001*((w[0]-1)**2+w[1]**2)
        opt=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])
        if not opt.success:raise RuntimeError(opt.message)
        p=expit(opt.x[0]*outer+opt.x[1]);best=p*10000+(1-p)*q
        slope,intercept,scale=cal['suites'][suite]['parameters']['mse_probability_and_scale'][i];p=expit(slope*outer+intercept);candidate=p*10000+(1-p)*q*scale
        for w in [0,.25,.5,.75,1]:scores[str(w)].append(float(mean_squared_error(y[va],(1-w)*best+w*candidate)))
    report[suite]=scores
with open('conservative_blend_validation.json','w') as f:json.dump(report,f,indent=2)
print({w:round(np.mean([v for suite in report.values() for v in suite[w]])) for w in ['0','0.25','0.5','0.75','1']})
