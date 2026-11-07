"""Audit PubChem VACV/cytotoxicity measurements by exact substance ID.

Candidate ratios are intentionally excluded from training until assay conditions
and the meaning of AC50 versus the competition's IC50/CC50 are validated.
"""
import csv
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

AIDS = (1117274, 1053167, 1117284, 977617, 1053115)
PAIRS = ((1117274, 1117284), (1117274, 977617),
         (1117274, 1053115), (1053167, 1117284),
         (1053167, 977617), (1053167, 1053115))
RAW = Path('data/raw/pubchem')
RAW.mkdir(parents=True, exist_ok=True)

def rows(aid):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV'
    path = RAW / f'assay_{aid}.csv'
    if not path.exists():
        path.write_bytes(urlopen(url, timeout=60).read())
    data = csv.DictReader(io.StringIO(path.read_text(encoding='utf-8-sig')))
    return {r['PUBCHEM_SID']: r for r in data if r['PUBCHEM_SID'].isdigit()}

def main():
    assays = {aid: rows(aid) for aid in AIDS}
    report = {'status': 'candidate_pairs_excluded_from_training', 'assay_rows':
              {str(k): len(v) for k, v in assays.items()}, 'pairs': []}
    for activity, toxicity in PAIRS:
        shared = sorted(set(assays[activity]) & set(assays[toxicity]))
        exact = []
        for sid in shared:
            a, t = assays[activity][sid], assays[toxicity][sid]
            if not (a.get('AC50_Qualifier') == t.get('AC50_Qualifier') == '='):
                continue
            try:
                av, tv = float(a['AC50_uM']), float(t['AC50_uM'])
            except (ValueError, KeyError):
                continue
            if av <= 0 or tv <= 0:
                continue
            exact.append({'sid': sid, 'activity_ac50_uM': av,
                          'toxicity_ac50_uM': tv, 'candidate_ratio': tv / av,
                          'activity_url': f'https://pubchem.ncbi.nlm.nih.gov/bioassay/{activity}',
                          'toxicity_url': f'https://pubchem.ncbi.nlm.nih.gov/bioassay/{toxicity}'})
        report['pairs'].append({'activity_aid': activity, 'toxicity_aid': toxicity,
                                'shared_sid': len(shared), 'exact_numeric_pairs': exact})
    report['reason_excluded'] = ('The assays report AC50 rather than the competition IC50/CC50; '
                                 'exposure duration and readouts differ across protocols. '
                                 'An exact SID and = qualifiers do not validate a comparable SI.')
    Path('pubchem_pair_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({f"{p['activity_aid']}/{p['toxicity_aid']}":
                      [p['shared_sid'], len(p['exact_numeric_pairs'])] for p in report['pairs']}))

if __name__ == '__main__':
    main()
