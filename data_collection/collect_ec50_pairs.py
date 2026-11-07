"""Discover external CC50/EC50 pairs without equating EC50 and IC50."""
import concurrent.futures
import csv
import json
import math
from pathlib import Path
from collect_pairs import cached_pages
from collect_chembl import BASE,request_json
raw=Path('data/raw/pairs')
A=json.loads((raw/'vaccinia_assays.json').read_text())['assays']
aids=sorted({a['assay_chembl_id'] for a in A});docs=sorted({a['document_chembl_id'] for a in A if a.get('document_chembl_id')})
jobs=[]
for kind,ids,field in [('EC50',aids,'assay_chembl_id__in'),('CC50',docs,'document_chembl_id__in')]:
    for i in range(0,len(ids),25):jobs.append(({'standard_type':kind,field:','.join(ids[i:i+25])},raw/f'{kind}_{i}.json'))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:batches=list(ex.map(lambda job:cached_pages(*job),jobs))
activities=[a for b in batches for a in b];groups={}
for a in activities:
    if a.get('standard_relation')!='=' or a.get('standard_units')!='nM' or a.get('data_validity_comment') or a.get('potential_duplicate') or not a.get('canonical_smiles'):continue
    try:v=float(a['standard_value'])
    except (TypeError,ValueError):continue
    if v<=0 or not math.isfinite(v):continue
    groups.setdefault((a['document_chembl_id'],a['molecule_chembl_id']),{}).setdefault(a['standard_type'],[]).append(a)
rows=[]
for (doc,mol),types in groups.items():
    if set(types)!={'EC50','CC50'} or any(len(v)!=1 for v in types.values()):continue
    ec,cc=types['EC50'][0],types['CC50'][0]
    if ec['canonical_smiles']!=cc['canonical_smiles']:continue
    rows.append({'SMILES':ec['canonical_smiles'],'EC50_uM':float(ec['standard_value'])/1000,'CC50_uM':float(cc['standard_value'])/1000,'selectivity_candidate':float(cc['standard_value'])/float(ec['standard_value']),
                 'ec50_original_value':ec['standard_value'],'ec50_original_unit':ec['standard_units'],'ec50_relation':ec['standard_relation'],'cc50_original_value':cc['standard_value'],'cc50_original_unit':cc['standard_units'],'cc50_relation':cc['standard_relation'],'definition':'derived CC50/EC50; not CC50/IC50','document_chembl_id':doc,'molecule_chembl_id':mol,'ec50_activity_id':ec['activity_id'],'cc50_activity_id':cc['activity_id'],
                 'ec50_assay_id':ec['assay_chembl_id'],'cc50_assay_id':cc['assay_chembl_id'],'ec50_description':ec['assay_description'],'cc50_description':cc['assay_description'],
                 'ec50_url':BASE+'/activity/'+str(ec['activity_id'])+'.json','cc50_url':BASE+'/activity/'+str(cc['activity_id'])+'.json','status':'candidate_context_review'})
def document(i):
    p=raw/(i+'.json')
    if not p.exists():p.write_text(json.dumps(request_json(BASE+'/document/'+i+'.json')))
    return i,json.loads(p.read_text()).get('doi')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:dois=dict(ex.map(document,sorted({r['document_chembl_id'] for r in rows})))
for r in rows:r['DOI']=dois[r['document_chembl_id']]
if rows:
    with open('chembl_ec50_pair_candidates.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps({'activities_scanned':len(activities),'candidate_pairs':len(rows),'documents':len(docs)}),flush=True)
