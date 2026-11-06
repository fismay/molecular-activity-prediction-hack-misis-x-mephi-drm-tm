"""Create a fixed convex prediction blend with strict sample/index checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--best',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--weight',type=float,default=.25)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not 0<=args.weight<=1:raise ValueError('Weight must be within 0..1')
    a,b,sample=(pd.read_csv(p) for p in [args.best,args.candidate,args.sample])
    for frame in [a,b,sample]:
        if list(frame)!=['index','SI VACV'] or len(frame)!=597 or frame['index'].duplicated().any():raise ValueError('Unexpected format')
        if not frame['index'].equals(sample['index']):raise ValueError('Index order differs')
    pa,pb=(f['SI VACV'].to_numpy(float) for f in [a,b])
    if not np.isfinite(pa).all() or not np.isfinite(pb).all():raise ValueError('Nonfinite input')
    prediction=(1-args.weight)*pa+args.weight*pb
    if not np.isfinite(prediction).all():raise ValueError('Nonfinite output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'index':sample['index'],'SI VACV':prediction}).to_csv(args.output,index=False)
    report={'weight':args.weight,'rows':len(prediction),'checks':{'columns':True,'index_order':True,'finite':True},'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.best,args.candidate]},'submission_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'public_score':None}
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
