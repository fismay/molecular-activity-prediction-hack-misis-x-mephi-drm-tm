"""Collect public VACV IC50 and linked CC50 candidates; retain assay provenance.

Pairs are candidates for context review, not automatically validated labels.
Only exact, unflagged nM records and unambiguous document/molecule pairs enter
the output. Assay descriptions are included so cell/strain/protocol can be checked.
"""
import concurrent.futures
import csv
import json
from pathlib import Path
from urllib.parse import urlencode

from collect_chembl import BASE, request_json


def cached_pages(params, cache):
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    out = []
    offset = 0
    while True:
        query = dict(params, limit=1000, offset=offset)
        data = request_json(BASE + '/activity.json?' + urlencode(query))
        out.extend(data['activities'])
        if not data['page_meta']['next']:
            break
        offset += len(data['activities'])
    cache.write_text(json.dumps(out))
    return out


def main():
    raw = Path('data/raw/pairs')
    raw.mkdir(parents=True, exist_ok=True)
    assay_file = raw / 'vaccinia_assays.json'
    if not assay_file.exists():
        assay_file.write_text(json.dumps(request_json(BASE + '/assay.json?description__icontains=vaccinia&limit=1000')))
    assays = json.loads(assay_file.read_text())['assays']
    if len(assays) >= 1000:
        raise ValueError('Assay search needs pagination')
    aid = sorted({a['assay_chembl_id'] for a in assays})
    docs = sorted({a['document_chembl_id'] for a in assays if a.get('document_chembl_id')})
    jobs = []
    for kind, ids, field in [('IC50', aid, 'assay_chembl_id__in'), ('CC50', docs, 'document_chembl_id__in')]:
        for i in range(0, len(ids), 25):
            jobs.append(({'standard_type': kind, field: ','.join(ids[i:i+25])}, raw / f'{kind}_{i}.json'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        batches = list(pool.map(lambda job: cached_pages(*job), jobs))
    activities = [a for b in batches for a in b]
    grouped = {}
    for a in activities:
        if a.get('standard_relation') != '=' or a.get('standard_units') != 'nM':
            continue
        if a.get('data_validity_comment') or a.get('potential_duplicate') or not a.get('canonical_smiles'):
            continue
        try:
            value = float(a['standard_value'])
        except (ValueError, TypeError):
            continue
        if value <= 0:
            continue
        key = (a['document_chembl_id'], a['molecule_chembl_id'])
        grouped.setdefault(key, {}).setdefault(a['standard_type'], []).append(a)
    rows = []
    for (doc, mol), measurements in grouped.items():
        if set(measurements) != {'IC50', 'CC50'}:
            continue
        # A repeated value in different assays still needs protocol review.
        if any(len(items) != 1 for items in measurements.values()):
            continue
        ic, cc = measurements['IC50'][0], measurements['CC50'][0]
        if ic['canonical_smiles'] != cc['canonical_smiles']:
            continue
        rows.append({'SMILES': ic['canonical_smiles'], 'IC50_uM': float(ic['standard_value'])/1000,
                     'CC50_uM': float(cc['standard_value'])/1000,
                     'SI_candidate': float(cc['standard_value'])/float(ic['standard_value']),
                     'document_chembl_id': doc, 'molecule_chembl_id': mol,
                     'ic50_activity_id': ic['activity_id'], 'cc50_activity_id': cc['activity_id'],
                     'ic50_assay_id': ic['assay_chembl_id'], 'cc50_assay_id': cc['assay_chembl_id'],
                     'ic50_description': ic['assay_description'], 'cc50_description': cc['assay_description'],
                     'validation_status': 'candidate_context_review',
                     'source_url': BASE+'/document/'+doc+'.json'})
    if rows:
        candidate_docs = sorted({row['document_chembl_id'] for row in rows})
        def get_document(identifier):
            cache = raw / (identifier + '.json')
            if not cache.exists():
                cache.write_text(json.dumps(request_json(BASE + '/document/' + identifier + '.json')))
            return identifier, json.loads(cache.read_text()).get('doi') or ''
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            dois = dict(pool.map(get_document, candidate_docs))
        for row in rows:
            row['DOI'] = dois[row['document_chembl_id']]
            row['ic50_source_url'] = BASE + '/activity/' + str(row['ic50_activity_id']) + '.json'
            row['cc50_source_url'] = BASE + '/activity/' + str(row['cc50_activity_id']) + '.json'
        with open('chembl_pair_candidates.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'activities': len(activities), 'candidate_pairs': len(rows), 'documents': len(docs)}))


if __name__ == '__main__':
    main()
