# clinifs examples

| File | Description |
|---|---|
| `build_example_data.py` | Generate `example_microarray.csv` and `example_labels.csv` (synthetic, 200 × 1000) |
| `example_microarray.csv` | 200 samples × 1000 synthetic genes (auto-generated; included for convenience) |
| `example_labels.csv` | Binary labels for the 200 samples |
| `quickstart.py` | End-to-end demo: adaptive selection → RRA → stability report |

## Run

```bash
# From package root
python examples/quickstart.py
# or
clinifs demo
```

Expected output (< 5 s on a modern laptop):

```
Loaded:  X (200, 1000)  |  classes [0 1]

[1] Auto tier  : easy_medium
    Selected   : [  3  17  42  58  91 …] (20 genes)
    X_panel    : (200, 20)

[2] RRA(ANOVA+MI) top-5 ρ-scores: [0.0001  0.0003  0.0012  0.0019  0.0031]

[3] ANOVA top-10: [ 3 17 42 58 91 … ]

[4] Stability (5-fold ANOVA, k=20)
    Nogueira Φ     : 0.8xxx
    Kuncheva Jaccard: 0.7xxx

✓ Quickstart complete.
```

## Data note

`example_microarray.csv` is fully synthetic (`sklearn.datasets.make_classification`, seed=42). It contains no real patient data.
