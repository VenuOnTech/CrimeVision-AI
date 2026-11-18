# Conformal artifact migration

The active implementation now requires schema **3**, which stores a recording-group calibration unit and fit/calibration recording IDs. Schema-2 and earlier JSON artifacts are historical and cannot be loaded by the new inference code.

Regenerate from separate fit and calibration CSVs:

```bash
cd research-engine
python calibrate_conformal.py \
  --fit-csv datasets/uca/fit.csv \
  --calibration-split-csv datasets/uca/calibration.csv \
  --output models/uca_conformal_group_v3.json \
  --alpha 0.10
```

Then test the artifact on the separate recording-disjoint test split:

```bash
python evaluate_research_engine.py \
  --conformal models/uca_conformal_group_v3.json \
  --test-csv datasets/uca/test.csv \
  --output reports/uca_conformal_group_metrics.json \
  --sweep-alpha 0.05,0.10,0.20,0.30
```

Do not edit the `schema_version` field on an old artifact to bypass this check. Recalibration is required because the calibration unit and score aggregation changed.
