"""Collect published Vero/VACV Copenhagen selectivity from one linked ChEMBL study."""

import argparse
import csv
import json
import math
from pathlib import Path

from rdkit import Chem

from collect_chembl import BASE, all_activities, request_json


STUDIES = {
    '2022': ('CHEMBL5223146', '10.1016/j.bmcl.2022.129080',
             {'IC50': 'CHEMBL5225279', 'CC50': 'CHEMBL5225278', 'SI': 'CHEMBL5225280'}),
    '2018': ('CHEMBL4368876', '10.1039/C8MD00442K',
             {'IC50': 'CHEMBL4369494', 'CC50': 'CHEMBL4369495', 'SI': 'CHEMBL4369496'}),
    '2021': ('CHEMBL4837211', '10.1016/j.ejmech.2021.113485',
             {'IC50': 'CHEMBL4838274', 'CC50': 'CHEMBL4838273', 'SI': 'CHEMBL4838275'}),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=STUDIES, default='2022')
    args = parser.parse_args()
    doc, doi, assays = STUDIES[args.study]
    document = request_json(f'{BASE}/document/{doc}.json')
    if document.get('doi', '').lower() != doi.lower():
        raise ValueError('Unexpected publication')
    descriptions = {}
    activities = {}
    for kind, assay_id in assays.items():
        assay = request_json(f'{BASE}/assay/{assay_id}.json')
        if assay.get('document_chembl_id') != doc:
            raise ValueError('Assay document mismatch')
        descriptions[kind] = assay['description']
        activities[kind] = all_activities(assay_id)
    if not all('vero' in desc.lower() for desc in descriptions.values()):
        raise ValueError('Wrong cell context')
    if 'vaccinia virus copenhagen' not in descriptions['IC50'].lower() or 'vaccinia virus copenhagen' not in descriptions['SI'].lower():
        raise ValueError('Wrong viral context')

    indexed = {}
    rejected = {}
    for kind, rows in activities.items():
        values = {}
        ambiguous = set()
        reasons = {}
        for row in rows:
            reason = None
            expected = 'Ratio CC50/IC50' if kind == 'SI' else kind
            if row.get('document_chembl_id') != doc or row.get('standard_type') != expected:
                reason = 'wrong_document_or_type'
            elif row.get('standard_relation') != '=' or row.get('data_validity_comment') or row.get('potential_duplicate'):
                reason = 'censored_or_flagged'
            elif kind != 'SI' and row.get('standard_units') != 'nM':
                reason = 'wrong_unit'
            else:
                try:
                    value = float(row['standard_value'])
                except (TypeError, ValueError):
                    reason = 'invalid_value'
                if reason is None and (not math.isfinite(value) or value <= 0):
                    reason = 'invalid_value'
            if reason is None:
                mol = Chem.MolFromSmiles(row.get('canonical_smiles') or '')
                if mol is None:
                    reason = 'invalid_smiles'
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            key = row['molecule_chembl_id']
            if key in values:
                ambiguous.add(key)
            else:
                values[key] = (row, Chem.MolToSmiles(mol, isomericSmiles=True), value)
        for key in ambiguous:
            values.pop(key, None)
        indexed[kind] = values
        rejected[kind] = {'reasons': reasons, 'ambiguous_molecules': len(ambiguous)}

    output = []
    for key, (si, smiles, reported) in sorted(indexed['SI'].items()):
        row = {'SMILES': smiles, 'SI_published': reported, 'DOI': doi, 'document_chembl_id': doc,
               'molecule_chembl_id': key, 'si_activity_id': si['activity_id'],
               'si_assay_id': assays['SI'], 'si_source_url': f'{BASE}/activity/{si["activity_id"]}.json',
               'ic50_activity_id': '', 'cc50_activity_id': '', 'IC50_uM': '', 'CC50_uM': '',
               'SI_derived': '', 'validation_status': 'published_si_only',
               'cell': 'Vero', 'virus': 'Vaccinia Copenhagen'}
        if key in indexed['IC50'] and key in indexed['CC50']:
            ic, ic_smiles, ic_value = indexed['IC50'][key]
            cc, cc_smiles, cc_value = indexed['CC50'][key]
            if smiles == ic_smiles == cc_smiles:
                derived = cc_value / ic_value
                row.update(ic50_activity_id=ic['activity_id'], cc50_activity_id=cc['activity_id'],
                           IC50_uM=ic_value/1000, CC50_uM=cc_value/1000, SI_derived=derived,
                           validation_status='consistent' if abs(derived-reported)/reported <= .05 else 'mismatch_review')
            else:
                row['validation_status'] = 'structure_mismatch'
        output.append(row)

    path = Path(f'vacv_{args.study}_published_si.csv')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(output)
    from collections import Counter
    report = {'source': BASE, 'document': doc, 'doi': doi, 'assays': assays,
              'descriptions': descriptions, 'raw_counts': {k: len(v) for k, v in activities.items()},
              'rejected': rejected, 'rows': len(output),
              'validation_status': dict(Counter(x['validation_status'] for x in output)),
              'policy': 'Published exact SI kept distinct from concentrations; only consistent triples may become matched labels.'}
    Path(f'vacv_{args.study}_source_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'rows': len(output), 'status': report['validation_status']}))


if __name__ == '__main__':
    main()
