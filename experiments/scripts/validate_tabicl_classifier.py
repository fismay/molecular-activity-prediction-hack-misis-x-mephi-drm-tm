"""Fixed 25% TabICL probability blend versus the latest nested MSE candidate."""
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
from tabicl import TabICLClassifier
data=np.load(root/'tabicl_features.npz');S=data['syntax'];y=data['y'];groups=data['groups']
base=np.load(repo/'artifacts/calibrated_neighbor_validation.npz');stack=np.load(repo/'artifacts/probability_stack_cache.npz');cal=json.loads((repo/'end_to_end_calibration_report.json').read_text())
cachepath=root/'tabicl8_predictions.npz';cache=dict(np.load(cachepath)) if cachepath.exists() else {}
report={'package':'tabicl','version':'2.2.0','checkpoint':'tabicl-classifier-v2-20260212.ckpt','n_estimators':8,'fixed_weight':.25,'suites':{}}
assert str(data['train_sha256']) == json.loads((repo/'calibrated_report.json').read_text())['data_sha256']['train.csv']
torch.set_num_threads(4)
report.update(train_sha256=str(data['train_sha256']),torch_version=torch.__version__,device='cpu',source='https://github.com/soda-inria/tabicl',status='rejected_experiment_no_new_submission')
for suite in ['42','2026','17','fresh_groupkfold901','new_groupkfold137']:
    scores={'previous_final_mse':[],'tabicl_probability_blend':[]}
    for i in range(5):
        key=suite+'_'+str(i);tr=base[key+'_tr'];va=base[key+'_va'];assert not set(groups[tr])&set(groups[va])
        if key not in cache:
            start=time.monotonic();mask=np.ptp(S[tr],axis=0)>0
            clf=TabICLClassifier(n_estimators=8,model_path=str(root/'tabicl-classifier-v2-20260212.ckpt'),device='cpu',n_jobs=4,random_state=42,verbose=False)
            clf.fit(S[tr][:,mask],y[tr]==10000);cache[key]=clf.predict_proba(S[va][:,mask])[:,1];np.savez(cachepath,**cache)
            print('inference',key,'seconds',round(time.monotonic()-start,1),flush=True)
        a,b,scale=cal['suites'][suite]['parameters']['mse_probability_and_scale'][i]
        p=expit(a*logit(np.clip(stack[key+'_outer'].mean(axis=1),1e-8,1-1e-8))+b);q=scale*stack[key+'_conditional'];c=cache[key]
        for name,pp in [('previous_final_mse',p),('tabicl_probability_blend',.75*p+.25*c)]:scores[name].append(float(mean_squared_error(y[va],10000*pp+(1-pp)*q)))
    report['checkpoint_sha256']=hashlib.file_digest((root/report['checkpoint']).open('rb'),'sha256').hexdigest() if (root/report['checkpoint']).exists() else None
    report['suites'][suite]=scores;(repo/'tabicl_blend_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(suite,{k:round(np.mean(v)) for k,v in scores.items()},flush=True)
print('overall',{k:round(np.mean([v for s in report['suites'].values() for v in s[k]])) for k in scores},flush=True)
