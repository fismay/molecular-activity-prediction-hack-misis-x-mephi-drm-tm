import concurrent.futures,csv,sys,re,json
from pathlib import Path
from rdkit import Chem
sys.path.insert(0,'.')
from collect_chembl import request_json,all_activities,BASE
assays=request_json(BASE+'/assay.json?description__icontains=vaccinia&limit=1000')['assays']
chosen=[a for a in assays if re.search('selectivity index',a.get('description') or '',re.I) and re.search('CC50',a.get('description') or '',re.I) and re.search('IC50',a.get('description') or '',re.I) and not re.search('EC50|EC90',a.get('description') or '',re.I)]
print('assays',len(chosen),flush=True)
def download(a):
 try:return a,all_activities(a['assay_chembl_id'])
 except Exception as e:return a,{'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p:downloads=list(p.map(download,chosen))
docs=set(a['document_chembl_id'] for a,_ in downloads if a.get('document_chembl_id'))
def doc(d):
 try:return d,request_json(BASE+'/document/'+d+'.json').get('doi')
 except Exception:return d,None
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as p:dois=dict(p.map(doc,docs))
rows=[]
for a,acts in downloads:
 if isinstance(acts,dict):print('error',a['assay_chembl_id'],acts,flush=True);continue
 print(a['assay_chembl_id'],len(acts),dict((t,sum(x.get('standard_type')==t for x in acts)) for t in set(x.get('standard_type') for x in acts)),flush=True)
 for x in acts:
  if x.get('standard_relation')!='=' or x.get('data_validity_comment') or x.get('potential_duplicate') or not x.get('canonical_smiles'):continue
  try:v=float(x['standard_value'])
  except (ValueError,TypeError):continue
  if v<=0:continue
  m=Chem.MolFromSmiles(x['canonical_smiles'])
  if m is None:continue
  rows.append({'SMILES':Chem.MolToSmiles(m,isomericSmiles=True),'SI':v,'DOI':dois.get(a['document_chembl_id']) or '', 'assay_chembl_id':a['assay_chembl_id'],'activity_id':x['activity_id'],'molecule_chembl_id':x['molecule_chembl_id'],'standard_type':x.get('standard_type'),'assay_description':a['description'],'source_url':BASE+'/activity/'+str(x['activity_id'])+'.json'})
print('candidate rows',len(rows),'unique structures',len(set(r['SMILES'] for r in rows)),flush=True)
with open('chembl_si_candidates.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
