"""Add provenance-audited public SI records only to the neighbor component.

Source observations duplicating any training canonical molecule are removed.
Validation must additionally purge source copies of outer validation molecules.
The classifier and forest are retained from the baseline. Optional isotonic
calibration uses only inner grouped out-of-fold probabilities.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit,logit
from vacv_baseline import featurize
from vacv_text_mixture import probability
from vacv_knn_blend import nearest_ensemble

def validate_source(source):
    required=['SMILES','SI_derived','SI_reported','CC50_uM','IC50_uM','DOI','ic50_activity_id','source_url','identity_resolution','endpoint_relation','cell','virus']
    if not set(required)<=set(source) or source[required].isna().any().any():raise ValueError('Missing source evidence')
    values=source[['CC50_uM','IC50_uM','SI_derived','SI_reported']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,:2]<=0).any() or (values[:,2:]<0).any():raise ValueError('Invalid source point measurements')
    if not source.endpoint_relation.eq('=').all():raise ValueError('Censored source observations are not exact labels')
    if not source.cell.eq('Vero').all() or not source.virus.eq('Vaccinia Copenhagen').all():raise ValueError('This source recipe requires the audited Vero/VACV Copenhagen context')
    if not np.allclose(source.SI_derived,source.CC50_uM/source.IC50_uM):raise ValueError('Source ratio mismatch')

def predict(model,test):
    b=model['base'];rawp=probability(b['base'],test.SMILES);a,c=b['brier_parameters'];p=expit(a*logit(np.clip(rawp,1e-8,1-1e-8))+c)
    isotonic_weight=model.get('isotonic_weight',0.)
    if isotonic_weight:p=(1-isotonic_weight)*p+isotonic_weight*model['isotonic'].predict(rawp)
    weight=model.get('graph_weight',0.)
    if weight:
        from vacv_graph_chemberta import graph_probability
        graph,_=graph_probability(model['train_smiles'],model['y'],test.SMILES)
        p=(1-weight)*p+weight*graph
    x,_=featurize(test.SMILES);forest=np.maximum(0,b['base']['regressor'].predict(x))
    sm=pd.concat([model['train_smiles'],model['external_smiles']],ignore_index=True);values=np.r_[model['y'],model['external_si']]
    recipe=model.get('neighbor_recipe','ensemble')
    if recipe=='ensemble':neighbors=nearest_ensemble(sm,values,test.SMILES)
    else:
        from stereo_neighbors import nearest_stereo
        neighbors=nearest_stereo(sm,values,test.SMILES,enforce_scaffold=recipe=='scaffold-stereo')
    pred=p*10000+(1-p)*(.5*forest+.5*neighbors)
    if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
    return pred

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train','test','sample','source','base-model','model','output','report']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-policy',choices=['derived','reported'],default='derived');p.add_argument('--graph-weight',type=float,default=0.);p.add_argument('--isotonic-weight',type=float,default=0.);p.add_argument('--neighbor-recipe',choices=['ensemble','chiral2','scaffold-stereo'],default='ensemble');a=p.parse_args()
    if not 0<=a.graph_weight<=1 or not 0<=a.isotonic_weight<=1:raise ValueError('Invalid probability blend weight')
    D,T,S=(pd.read_csv(f) for f in [a.train,a.test,a.sample]);E=pd.read_csv(a.source);y=D['SI VACV'].to_numpy(float)
    if len(T)!=597 or len(S)!=597 or list(S.columns)!=['index','SI VACV'] or not np.array_equal(T['index'],S['index']) or not S['index'].is_unique:raise ValueError('Unexpected submission/sample order')
    validate_source(E)
    _,g=featurize(D.SMILES);_,eg=featurize(E.SMILES);keep=~np.isin(eg,g);novel=E.loc[keep].reset_index(drop=True)
    values=novel['SI_'+a.source_policy].to_numpy(float)
    if not len(novel) or not np.isfinite(values).all() or (values<0).any():raise ValueError('No usable source records')
    base=joblib.load(a.base_model)
    if list(base['train_smiles'])!=list(D.SMILES) or not np.array_equal(base['y'],y) or base.get('conditional_scale',1.)!=1. or base.get('calibration_method','brier')!='brier':raise ValueError('Wrong baseline')
    for name in ['regressor','syntax_classifier']:
        if name in base['base']:base['base'][name].set_params(n_jobs=1)
    model={'base':base,'train_smiles':D.SMILES.reset_index(drop=True),'y':y,'external_smiles':novel.SMILES,'external_si':values,'source_policy':a.source_policy,'source_sha256':hashlib.sha256(a.source.read_bytes()).hexdigest(),'source_doi':sorted(set(E.DOI)),'graph_weight':a.graph_weight,'neighbor_recipe':a.neighbor_recipe,'isotonic_weight':a.isotonic_weight}
    if a.isotonic_weight:
        from sklearn.model_selection import GroupKFold
        from sklearn.isotonic import IsotonicRegression
        from vacv_text_mixture import fit
        inner=np.zeros(len(y))
        for tr,va in GroupKFold(n_splits=3,shuffle=True,random_state=42).split(D.SMILES,y,g):
            m=fit(D.SMILES.iloc[tr],y[tr],True);inner[va]=probability(m,D.SMILES.iloc[va])
        model['isotonic']=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(inner,y==10000)
    pred=predict(model,T);mutant=T.copy()
    for col in mutant:
        if col not in ['SMILES','index']:mutant[col]='mutated'
    assert np.array_equal(pred,predict(model,mutant)) and np.array_equal(pred,predict(model,T[['SMILES','index']]))
    a.model.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,a.model);assert np.array_equal(pred,predict(joblib.load(a.model),T))
    a.output.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'index':S['index'],'SI VACV':pred}).to_csv(a.output,index=False)
    saved=pd.read_csv(a.output);assert len(saved)==597 and list(saved)==['index','SI VACV'] and np.isfinite(saved['SI VACV']).all() and np.array_equal(saved['index'],S['index'])
    report={'inputs':['SMILES'],'changed':'tie-aware neighbor component gains verified external measurements; optional stereo compatibility and probability calibration','neighbor_recipe':a.neighbor_recipe,'isotonic_weight':a.isotonic_weight,'isotonic_calibration':'inner3 grouped raw probabilities' if a.isotonic_weight else None,'graph_weight':a.graph_weight,'graph_transductive_unlabeled_test_smiles':bool(a.graph_weight),'source_doi':model['source_doi'],'source_policy':a.source_policy,'source_rows':len(E),'source_rows_removed_training_duplicates':int((~keep).sum()),'source_novel_rows':len(novel),'source_sha256':model['source_sha256'],'source_identity_resolution':sorted(set(E.identity_resolution)),
        'checks':{'rows':597,'exact_order':True,'finite':True,'forbidden_columns_mutation_identical':True,'other_columns_removed_identical':True,'saved_model_identical':True},'range':[float(pred.min()),float(pred.max())],'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.train,a.test,a.sample,a.output]}}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
