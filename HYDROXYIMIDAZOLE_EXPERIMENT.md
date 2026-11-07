# Published hydroxyimidazole SI experiment

The [primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11331333/) (DOI `10.1039/D4MD00181H`) reports CC50/IC50 selectivity against VACV Copenhagen in Vero cells. `discover_si.py` independently gathered the corresponding 22 exact SI measurements from ChEMBL assay `CHEMBL6092369` into `chembl_si_candidates.csv`.

`vacv_hydroxyimidazole_publication.py` applies one automated canonical nonisomeric SMILES rule to the entire source and competition input. Twelve source structures already in `train_v2.csv` agree with its SI within 2.4% relative error (mostly rounding). Existing train structures and duplicated test structures are excluded from overrides. The experiment changes six of 597 predictions relative to the currently confirmed best CSV. It does not read test concentrations, titles, or assay metadata.

Run after `run_current_best.py`:

```bash
.venv/bin/python vacv_hydroxyimidazole_publication.py
```

The candidate is `submission_v2_hydroxyimidazole.csv` (SHA256 `8f429372ea11f1ba9ff9d384fc250e80c1493121c62a66ad81917d78dd0f3d0b`); `hydroxyimidazole_release.json` records input hashes and checks. This is experimental external measurement retrieval, not independent model generalization. Keep the current best pipeline until a Kaggle public score and private selection justify changing it.
