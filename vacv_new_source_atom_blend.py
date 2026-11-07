"""Produce a validated SMILES-only submission from the frozen confirmed recipe."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from chemical_neighbors import nearest_recipe
from vacv_zero_hurdle import predict as current_predict, probabilities


TRAIN_SHA='d9a876a67b5623d9cdb6592fc53d31ffaa7c535b09968c3a538138740fe246e8'
TEST_SHA='1b0c6a014bbe556c42ef2277e867d2e1a647ad31c157a9cd81bbc7e262481abd'
SAMPLE_SHA='3eea4d23f5700361c09da24cd45be05c5dbbecf76cc29b022c3676835606884c'
NEW_SHA='b14dfd1904c2d4ff7da53fed75f2d648807c1c9717a2a08fe589342fb275d2e6'
BEST_SHA='0889724213860bd58db86e0b5ed0ed984160de4d67d9e943a0f199758ca96bae'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(model, new, smiles):
    base=current_predict(model,smiles)
    p=probabilities(model,smiles)['raw']
    positive=model['y']>0
    old=model['external']
    ref=pd.concat([model['train_smiles'].loc[positive].reset_index(drop=True),
                   old.SMILES.reset_index(drop=True),new.SMILES.reset_index(drop=True)],ignore_index=True)
    values=np.r_[model['y'][positive],old.SI_derived.to_numpy(float),new.SI.to_numpy(float)]
    atom=nearest_recipe(ref,values,smiles,feature='atom_pair_sparse')
    return .5*base+.5*(1-p)*atom,base


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['train','test','sample','hurdle','source','best','model','output','report']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    for path,expected in [(args.train,TRAIN_SHA),(args.test,TEST_SHA),(args.sample,SAMPLE_SHA),(args.source,NEW_SHA),(args.best,BEST_SHA)]:
        if sha(path)!=expected:raise ValueError('Unexpected input snapshot: '+str(path))
    screen=json.loads(Path('new_source_atom_blend_screen.json').read_text())
    confirmation=json.loads(Path('new_source_atom_blend_confirmation.json').read_text())
    if not screen['qualified'] or not confirmation['qualified'] or confirmation['seeds']!=[35471,68011]:
        raise ValueError('Fixed recipe not confirmed')
    train,test,sample=[pd.read_csv(path) for path in [args.train,args.test,args.sample]]
    best=pd.read_csv(args.best)
    if len(test)!=597 or len(sample)!=597 or list(sample)!=['index','SI VACV'] or not sample['index'].is_unique or not np.array_equal(test['index'],sample['index']):
        raise ValueError('Invalid test/sample alignment')
    model=joblib.load(args.hurdle)
    if model.get('target_version')!='train_v2' or model.get('calibration')!='raw' or model.get('conditional')!='source_stereo_only':
        raise ValueError('Unexpected baseline model')
    if not np.array_equal(model['y'],train['SI VACV'].to_numpy(float)) or list(model['train_smiles'])!=list(train.SMILES):
        raise ValueError('Baseline train snapshot mismatch')
    new=pd.read_csv(args.source)
    if len(new)!=34 or not new.SMILES.is_unique or not np.isfinite(new.SI.to_numpy(float)).all() or (new.SI<=0).any():
        raise ValueError('Invalid curated source')
    pred,base=predict(model,new,test.SMILES)
    if not np.allclose(base,best['SI VACV'].to_numpy(float),rtol=0,atol=1e-9):
        raise ValueError('Current model does not reproduce confirmed best predictions')
    if not np.isfinite(pred).all() or (pred<0).any():raise ValueError('Invalid candidate predictions')
    modified=test.copy()
    for column in test:
        if column!='SMILES':modified[column]='MUTATED'
    assert np.array_equal(pred,predict(model,new,modified.SMILES)[0])
    saved={'inputs':['SMILES'],'hurdle':model,'new':new[['SMILES','SI']].copy(),'weight':.5,'feature':'atom_pair_sparse'}
    args.model.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(saved,args.model)
    loaded=joblib.load(args.model)
    assert np.array_equal(pred,predict(loaded['hurdle'],loaded['new'],test.SMILES)[0])
    result=pd.DataFrame({'index':sample['index'],'SI VACV':pred})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    report={'inputs':['SMILES'],'new_source_rows':34,'weight':.5,'feature':'atom_pair_sparse',
            'screen_mean_mse':screen['mean_mse'],'confirmation_mean_mse':confirmation['pooled_mse'],
            'checks':{'rows':597,'sample_order':True,'finite_nonnegative':True,
                      'forbidden_columns_independent':True,'saved_model_identical':True,
                      'baseline_matches_confirmed_best':True},
            'sha256':{'train':sha(args.train),'test':sha(args.test),'sample':sha(args.sample),
                      'new_source':sha(args.source),'baseline':sha(args.best),'candidate':sha(args.output)}}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
