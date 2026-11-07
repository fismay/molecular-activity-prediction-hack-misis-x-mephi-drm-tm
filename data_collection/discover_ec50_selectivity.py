"""Collect directly published VACV CC50/EC50 selectivity; keep endpoint identity."""
import concurrent.futures
import csv
import json
import re
from pathlib import Path
from collect_chembl import BASE, request_json, all_activities

raw = Path('data/raw/ec50_selectivity')
raw.mkdir(parents=True, exist_ok=True)
assay_path = Path('data/raw/vaccinia_assays.json')
if not assay_path.exists():
    payload = request_json(BASE+'/assay.json?description__icontains=vaccinia&limit=1000')
    if payload['page_meta']['next']:
        raise ValueError('Assay search requires pagination')
    assay_path.write_text(json.dumps(payload))
A = json.loads(assay_path.read_text())['assays']
chosen = [a for a in A if re.search('selectivity', a.get('description',''),re.I)
          and 'CC50' in a['description'] and 'EC50' in a['description']
          and not re.search('cowpox|HCV|EC90',a['description'],re.I)]
def download(a):
    p = raw/(a['assay_chembl_id']+'.json')
    if not p.exists():p.write_text(json.dumps(all_activities(a['assay_chembl_id'])))
    return a,json.loads(p.read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    batches=list(ex.map(download,chosen))
rows=[]
for a,acts in batches:
    print(a['assay_chembl_id'],len(acts),sorted({str(x.get('standard_type')) for x in acts}),flush=True)
    for x in acts:
        if not x.get('canonical_smiles'):continue
        try:v=float(x['standard_value'])
        except (TypeError,ValueError):continue
        if v<0:continue
        rows.append({'SMILES':x['canonical_smiles'],'SI':v,'standard_type':x.get('standard_type'),
                     'relation':x.get('standard_relation'),'unit':x.get('standard_units'),
                     'definition':'direct published CC50/EC50 selectivity; not CC50/IC50',
                     'assay_chembl_id':a['assay_chembl_id'],'assay_description':a['description'],
                     'document_chembl_id':a['document_chembl_id'],'molecule_chembl_id':x['molecule_chembl_id'],
                     'activity_id':x['activity_id'],'data_validity_comment':x.get('data_validity_comment'),
                     'potential_duplicate':x.get('potential_duplicate'),
                     'source_url':BASE+'/activity/'+str(x['activity_id'])+'.json',
                     'status':'candidate_endpoint_context_review'})
docs={r['document_chembl_id'] for r in rows}
def doc(i):
    p=raw/(i+'.json')
    if not p.exists():p.write_text(json.dumps(request_json(BASE+'/document/'+i+'.json')))
    return i,json.loads(p.read_text()).get('doi')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:doi=dict(ex.map(doc,sorted(docs)))
for r in rows:r['DOI']=doi[r['document_chembl_id']]
with open('chembl_ec50_si_candidates.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print('assays',len(chosen),'rows',len(rows),flush=True)
