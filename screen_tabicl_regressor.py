"""TabICL conditional SI regression with train-only fingerprint SVD."""
import os
from pathlib import Path
repo=Path(__file__).resolve().parent
root=repo/'artifacts'/'tabicl'
root.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(root/'hf_cache');os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['TORCH_HOME']=str(root/'torch_cache')
import time,json,hashlib
import numpy as np
import torch
from scipy.special import expit,logit
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from tabicl import TabICLRegressor
data=np.load(root/'tabicl_features.npz');X=data['structure'];y=data['y'];groups=data['groups']
base=np.load(repo/'artifacts/calibrated_neighbor_validation.npz');stack=np.load(repo/'artifacts/probability_stack_cache.npz');cal=json.loads((repo/'end_to_end_calibration_report.json').read_text())
assert str(data['train_sha256']) == json.loads((repo/'calibrated_report.json').read_text())['data_sha256']['train.csv']
torch.set_num_threads(4)
report={'version':'2.2.0','checkpoint':'tabicl-regressor-v2-20260212.ckpt','n_estimators':2,'input':'32 train-only fingerprint SVD components and 9 molecular descriptors','folds':[]}
report.update(train_sha256=str(data['train_sha256']),torch_version=torch.__version__,device='cpu',source='https://github.com/soda-inria/tabicl',status='rejected_experiment_no_new_submission')
for i in range(5):
    key='42_'+str(i);tr=base[key+'_tr'];va=base[key+'_va'];ordinary=tr[y[tr]!=10000]
    start=time.monotonic();svd=TruncatedSVD(n_components=32,random_state=42);a=svd.fit_transform(X[ordinary,:2048]);b=svd.transform(X[va,:2048]);a=np.c_[a,X[ordinary,2048:]];b=np.c_[b,X[va,2048:]]
    model=TabICLRegressor(n_estimators=2,model_path=str(root/'tabicl-regressor-v2-20260212.ckpt'),device='cpu',n_jobs=4,random_state=42,verbose=False)
    model.fit(a,y[ordinary]);c=np.maximum(0,model.predict(b));u,v,scale=cal['suites']['42']['parameters']['mse_probability_and_scale'][i]
    p=expit(u*logit(np.clip(stack[key+'_outer'].mean(axis=1),1e-8,1-1e-8))+v);q=scale*stack[key+'_conditional']
    scores={str(w):float(mean_squared_error(y[va],p*10000+(1-p)*((1-w)*q+w*c))) for w in [0,.25,.5,1]}
    report['checkpoint_sha256']=hashlib.file_digest((root/report['checkpoint']).open('rb'),'sha256').hexdigest()
    report['folds'].append(scores);(repo/'tabicl_regression_screen.json').write_text(json.dumps(report,indent=2)+'\n');print(i,'seconds',round(time.monotonic()-start,1),{k:round(v) for k,v in scores.items()},flush=True)
print('mean',{k:round(np.mean([f[k] for f in report['folds']])) for k in report['folds'][0]},flush=True)
