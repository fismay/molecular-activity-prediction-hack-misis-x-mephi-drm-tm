"""Download and validate a linked VACV IC50/CC50/SI study from ChEMBL.

This first curated source is a reproducible seed for the larger external dataset.
The three assay IDs refer to the same publication, cell line and VACV strain.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem


BASE = "https://www.ebi.ac.uk/chembl/api/data"
ASSAYS = {
    "IC50": "CHEMBL4700293",
    "CC50": "CHEMBL4700294",
    "SI": "CHEMBL4700295",
}
DOCUMENT = "CHEMBL4699424"
COLUMNS = ["SMILES", "IC50_uM", "CC50_uM", "SI", "DOI", "molecule_chembl_id", "document_chembl_id", "ic50_activity_id", "cc50_activity_id", "si_activity_id", "source_url", "validation_status"]


def request_json(url: str) -> dict:
    for attempt in range(4):
        try:
            req = Request(url, headers={"User-Agent": "VACV-hackathon-reproducible-research/0.1"})
            with urlopen(req, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def all_activities(assay_id: str) -> list[dict]:
    result = []
    offset = 0
    while True:
        url = f"{BASE}/activity.json?" + urlencode({"assay_chembl_id": assay_id, "limit": 1000, "offset": offset})
        payload = request_json(url)
        batch = payload["activities"]
        result.extend(batch)
        if not payload["page_meta"]["next"]:
            return result
        offset += len(batch)


def clean_activity(activity: dict, expected: str) -> dict | None:
    if activity.get("potential_duplicate") or activity.get("data_validity_comment"):
        return None
    if activity.get("standard_relation") != "=" or not activity.get("canonical_smiles"):
        return None
    if activity.get("document_chembl_id") != DOCUMENT:
        return None
    typ = activity.get("standard_type")
    if expected == "SI":
        if typ != "Ratio CC50/IC50":
            return None
    elif typ != expected or activity.get("standard_units") != "nM":
        return None
    try:
        value = float(activity["standard_value"])
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    smiles = Chem.MolFromSmiles(activity["canonical_smiles"])
    if smiles is None:
        return None
    return {"smiles": Chem.MolToSmiles(smiles, isomericSmiles=True), "value": value, "id": activity["activity_id"], "molecule": activity["molecule_chembl_id"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("chembl_vacv_curated.csv"))
    parser.add_argument("--report", type=Path, default=Path("chembl_validation.json"))
    args = parser.parse_args()

    assays = {kind: request_json(f"{BASE}/assay/{identifier}.json") for kind, identifier in ASSAYS.items()}
    if any(a["document_chembl_id"] != DOCUMENT for a in assays.values()):
        raise ValueError("Assays are no longer linked to the expected publication")
    document = request_json(f"{BASE}/document/{DOCUMENT}.json")
    datasets = {kind: all_activities(identifier) for kind, identifier in ASSAYS.items()}
    indexed: dict[str, dict[tuple[str, str], dict]] = {}
    duplicates = {}
    for kind, activities in datasets.items():
        current = {}
        duplicate_keys = set()
        for raw in activities:
            clean = clean_activity(raw, kind)
            if clean is None:
                continue
            key = (clean["molecule"], clean["smiles"])
            if key in current:
                duplicate_keys.add(key)
            else:
                current[key] = clean
        for key in duplicate_keys:
            current.pop(key)
        indexed[kind] = current
        duplicates[kind] = len(duplicate_keys)

    records = []
    mismatches = []
    for key in sorted(set.intersection(*(set(items) for items in indexed.values()))):
        ic, cc, si = (indexed[kind][key] for kind in ("IC50", "CC50", "SI"))
        derived = cc["value"] / ic["value"]
        rel_error = abs(derived - si["value"]) / si["value"]
        status = "consistent" if rel_error <= 0.05 else "mismatch_review"
        if status != "consistent":
            mismatches.append({"molecule": key[0], "reported_si": si["value"], "derived_si": derived})
        records.append({
            "SMILES": key[1], "IC50_uM": ic["value"] / 1000,
            "CC50_uM": cc["value"] / 1000, "SI": si["value"],
            "DOI": document.get("doi") or "", "molecule_chembl_id": key[0],
            "document_chembl_id": DOCUMENT, "ic50_activity_id": ic["id"],
            "cc50_activity_id": cc["id"], "si_activity_id": si["id"],
            "source_url": f"{BASE}/document/{DOCUMENT}.json", "validation_status": status,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report = {"source": BASE, "document": DOCUMENT, "doi": document.get("doi"),
              "assays": ASSAYS, "raw_counts": {k: len(v) for k, v in datasets.items()},
              "filtered_counts": {k: len(v) for k, v in indexed.items()},
              "duplicate_molecule_assay_keys": duplicates,
              "complete_rows": len(records), "consistent_rows": len(records) - len(mismatches),
              "mismatches": mismatches}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete_rows": len(records), "consistent_rows": report["consistent_rows"], "doi": report["doi"]}))


if __name__ == "__main__":
    main()
