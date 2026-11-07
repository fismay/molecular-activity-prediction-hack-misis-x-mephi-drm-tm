"""Build a provenance-preserving table of novel audited Vero/VACV observations."""

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem


BASE = 'https://www.ebi.ac.uk/chembl/api/data/activity/'
OUTPUT = Path('data/curated/vacv_new_observations.csv')
COLUMNS = ['SMILES','SI','IC50_uM','CC50_uM','DOI','cell','virus','document_chembl_id',
           'molecule_chembl_id','ic50_activity_id','cc50_activity_id','si_activity_id',
           'ic50_url','cc50_url','si_url','source_file','label_policy']


def canonical(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError('Invalid source SMILES')
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def number(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('Nonexact, nonpositive or nonfinite observation')
    return result


def main():
    train = pd.read_csv('data/kaggle/train_v2.csv')
    train_keys = {canonical(s) for s in train.SMILES}
    rows = []
    old = pd.read_csv('chembl_cidofovir_curated.csv')
    for _,item in old.loc[old.cell.eq('Vero') & old.validation_status.eq('consistent')].iterrows():
        if item.DOI != '10.1016/j.ejmech.2023.115601':
            raise ValueError('Unexpected cidofovir source')
        si,ic,cc = number(item.SI),number(item.ic50_standard_value)/1000,number(item.cc50_standard_value)/1000
        if abs(cc/ic-si)/si > .05:
            raise ValueError('Inconsistent cidofovir ratio')
        rows.append({'SMILES':canonical(item.SMILES),'SI':si,'IC50_uM':ic,'CC50_uM':cc,'DOI':item.DOI,
                     'cell':'Vero','virus':'VACV','document_chembl_id':item.document_chembl_id,
                     'molecule_chembl_id':item.molecule_chembl_id,
                     'ic50_activity_id':item.ic50_activity_id,'cc50_activity_id':item.cc50_activity_id,
                     'si_activity_id':item.si_activity_id,'source_file':'chembl_cidofovir_curated.csv',
                     'label_policy':'exact published SI; matching exact IC50/CC50; source 72h/96h protocols'})
    for year,doi in [('2022','10.1016/j.bmcl.2022.129080'),
                     ('2018','10.1039/C8MD00442K'),
                     ('2021','10.1016/j.ejmech.2021.113485')]:
        file = f'vacv_{year}_published_si.csv'
        current = pd.read_csv(file)
        for _,item in current.loc[current.validation_status.eq('consistent')].iterrows():
            si,ic,cc = number(item.SI_published),number(item.IC50_uM),number(item.CC50_uM)
            if str(item.DOI).lower()!=doi.lower() or abs(cc/ic-si)/si > .05:
                raise ValueError('Inconsistent linked study')
            rows.append({'SMILES':canonical(item.SMILES),'SI':si,'IC50_uM':ic,'CC50_uM':cc,'DOI':item.DOI,
                         'cell':item.cell,'virus':item.virus,'document_chembl_id':item.document_chembl_id,
                         'molecule_chembl_id':item.molecule_chembl_id,
                         'ic50_activity_id':item.ic50_activity_id,'cc50_activity_id':item.cc50_activity_id,
                         'si_activity_id':item.si_activity_id,'source_file':file,
                         'label_policy':'exact published SI; matching exact IC50/CC50'})
    unique = {}
    excluded_train = 0
    for row in rows:
        key = row['SMILES']
        if key in train_keys:
            excluded_train += 1
            continue
        if key in unique:
            raise ValueError('Cross-source canonical duplicate requires manual review')
        for field in ['ic50_activity_id','cc50_activity_id','si_activity_id']:
            row[field.replace('_activity_id','_url')] = BASE+str(row[field])+'.json'
        unique[key] = row
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS,lineterminator='\n')
        writer.writeheader();writer.writerows(unique[k] for k in sorted(unique))
    report={'output':str(OUTPUT),'rows':len(unique),'excluded_train_copies':excluded_train,
            'source_rows_before_purge':len(rows),'doi_counts':{k:int(v) for k,v in pd.Series([r['DOI'] for r in unique.values()]).value_counts().items()},
            'sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
            'policy':'These measurements are research candidates, not automatically used in the best model; Kaggle concentration fields are never model inputs.'}
    Path('data/curated/vacv_new_observations_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
