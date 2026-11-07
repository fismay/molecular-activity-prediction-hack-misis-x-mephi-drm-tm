"""SMILES-only calibrated graph and frozen ChemBERTa conditional blend.

The graph is transductive: unlabeled test SMILES shape its neighborhoods.
Hyperparameters and both 25% blend weights were fixed before public scoring.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.sparse import csr_matrix,eye
from scipy.sparse.linalg import factorized
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from export_chemberta_smiles import smiles_sha256
from vacv_baseline import featurize
from vacv_text_mixture import probability
from vacv_knn_blend import nearest_ensemble

def graph_probability(train_smiles,y,test_smiles,labeled_indices=None,query_indices=None,mode_value=10000):
    strings=pd.concat([train_smiles.reset_index(drop=True),test_smiles.reset_index(drop=True)],ignore_index=True)
    text=TfidfVectorizer(analyzer='char',ngram_range=(3,7),lowercase=False).fit_transform(strings)
    sim=cosine_similarity(text)**2;np.fill_diagonal(sim,0);k=15
    neighbors=np.argpartition(-sim,k,axis=1)[:,:k];rows=np.repeat(np.arange(len(strings)),k);cols=neighbors.ravel()
    W=csr_matrix((sim[rows,cols],(rows,cols)),shape=sim.shape);W=(W+W.T)*.5
    degree=np.asarray(W.sum(1)).ravel();scale=1/np.sqrt(np.maximum(degree,1e-12));S=W.multiply(scale[:,None]).multiply(scale[None,:])
    solve=factorized((eye(len(strings))-.5*S).tocsc());labels=y==mode_value
    def masked_probability(labeled,query):
        targets=np.zeros((len(strings),2));targets[labeled,labels[labeled].astype(int)]=1
        diffused=np.maximum(0,solve(targets));return diffused[query,1]/np.maximum(diffused[query].sum(1),1e-12)
    labeled=np.arange(len(y)) if labeled_indices is None else np.asarray(labeled_indices)
    query=np.arange(len(y),len(strings)) if query_indices is None else np.asarray(query_indices)
    if len(set(labeled)&set(query)):raise ValueError('Query labels must be hidden')
    _,groups=featurize(train_smiles.iloc[labeled]);inner=np.zeros(len(labeled))
    for tr,va in GroupKFold(n_splits=3,shuffle=True,random_state=42).split(labeled,y[labeled],groups):inner[va]=masked_probability(labeled[tr],labeled[va])
    f=logit(np.clip(inner,1e-6,1-1e-6))
    def loss(w):return np.mean((expit(w[0]*f+w[1])-labels[labeled])**2)+.001*((w[0]-1)**2+w[1]**2)
    opt=minimize(loss,[1.,0.],method='L-BFGS-B',bounds=[(0,10),(-10,10)])
    if not opt.success:raise RuntimeError(opt.message)
    raw=masked_probability(labeled,query)
    return expit(opt.x[0]*logit(np.clip(raw,1e-6,1-1e-6))+opt.x[1]),opt.x

def predict(model,test,embeddings):
    strings=list(model['train_smiles'])+list(test.SMILES)
    if smiles_sha256(strings)!=model['smiles_sha256']:raise ValueError('Feature row identity/order mismatch')
    base=model['brier'];p=probability(base['base'],test.SMILES);slope,intercept=base['brier_parameters'];p=expit(slope*logit(np.clip(p,1e-8,1-1e-8))+intercept)
    x,_=featurize(test.SMILES);q=.5*np.maximum(0,base['base']['regressor'].predict(x))+.5*nearest_ensemble(model['train_smiles'],model['y'],test.SMILES)
    graphp,_=graph_probability(model['train_smiles'],model['y'],test.SMILES)
    p=.75*p+.25*graphp
    chemical=np.maximum(0,model['conditional'].predict(embeddings[len(model['y']):]))**2;q=.75*q+.25*chemical
    result=p*10000+(1-p)*q
    if not np.isfinite(result).all():raise ValueError('Nonfinite predictions')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['train','test','sample','features','encoder-report','base-model','model','output','report']:parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();train,test,sample=(pd.read_csv(p) for p in [a.train,a.test,a.sample]);features=np.load(a.features);y=train['SI VACV'].to_numpy(float)
    if len(test)!=597 or list(sample.columns)!=['index','SI VACV'] or len(sample)!=597 or not np.array_equal(test['index'],sample['index']) or not sample['index'].is_unique:raise ValueError('Unexpected sample/order')
    strings=list(train.SMILES)+list(test.SMILES);digest=smiles_sha256(strings)
    if str(features['smiles_sha256'])!=digest or int(features['train_rows'])!=len(train) or int(features['test_rows'])!=len(test):raise ValueError('Feature manifest mismatch')
    E=features['original_mean']
    if E.shape!=(len(strings),768) or not np.isfinite(E).all() or not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid features/targets')
    base=joblib.load(a.base_model)
    if list(base['train_smiles'])!=list(train.SMILES) or not np.array_equal(base['y'],y) or base.get('calibration_method','brier')!='brier' or base.get('conditional_scale',1.)!=1.:raise ValueError('Wrong baseline model')
    # Sequential forest reduction gives bit-identical predictions for release
    # invariance/round-trip checks, avoiding thread-order summation noise.
    for name in ['regressor','syntax_classifier']:
        if name in base['base']:base['base'][name].set_params(n_jobs=1)
    keep=np.flatnonzero(y!=10000);reg=make_pipeline(StandardScaler(),Ridge(alpha=100)).fit(E[keep],np.sqrt(y[keep]))
    model={'brier':base,'conditional':reg,'train_smiles':train.SMILES.reset_index(drop=True),'y':y,'smiles_sha256':digest,'graph_weight':.25,'chemical_conditional_weight':.25,'encoder':json.loads(a.encoder_report.read_text())}
    pred=predict(model,test,E)
    mutant=test.copy()
    for col in mutant.columns:
        if col not in ['SMILES','index']:mutant[col]='changed_for_invariance_check'
    assert np.array_equal(pred,predict(model,mutant,E))
    assert np.array_equal(pred,predict(model,test[['SMILES','index']],E))
    a.model.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,a.model);assert np.array_equal(pred,predict(joblib.load(a.model),test,E))
    a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'index':sample['index'],'SI VACV':pred}).to_csv(a.output,index=False)
    saved=pd.read_csv(a.output);assert list(saved.columns)==['index','SI VACV'] and len(saved)==597 and np.isfinite(saved['SI VACV']).all() and np.array_equal(saved['index'],sample['index'])
    report={'inputs':['SMILES'],'graph':{'transductive_unlabeled_test_smiles':True,'k':15,'alpha':.5,'affinity':'squared char3-7 TFIDF cosine','weight':.25,'calibration':'inner3 grouped Brier'},'conditional':{'encoder':model['encoder'],'regressor':'StandardScaler + Ridge(alpha100) on sqrt SI, y!=10000','weight':.25},'release_checks':{'rows':597,'exact_sample_order':True,'finite':True,'forbidden_columns_mutation_identical':True,'all_other_columns_removed_identical':True,'model_roundtrip_identical':True},'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.train,a.test,a.sample,a.output]},'prediction_range':[float(pred.min()),float(pred.max())]}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
