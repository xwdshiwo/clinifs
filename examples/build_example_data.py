"""
Generate synthetic example data for clinifs demos.

Output files (written to this directory):
    example_microarray.csv   — 200 samples × 1000 genes (float32)
    example_labels.csv       — 200 samples × 1 label column (0/1)

These are fully synthetic (make_classification with fixed seed = 42) and do NOT
contain any real patient data.  They are intended only for API demonstration.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

HERE = pathlib.Path(__file__).parent

N_SAMPLES    = 200
N_FEATURES   = 1000
N_INFORMATIVE = 50
N_REDUNDANT   = 30
RANDOM_STATE  = 42


def build():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        n_redundant=N_REDUNDANT,
        n_classes=2,
        flip_y=0.02,
        random_state=RANDOM_STATE,
    )
    X = X.astype(np.float32)

    gene_names  = [f"GENE_{i:04d}" for i in range(N_FEATURES)]
    sample_ids  = [f"S{i:03d}" for i in range(N_SAMPLES)]

    df_X = pd.DataFrame(X, index=sample_ids, columns=gene_names)
    df_y = pd.DataFrame({"label": y}, index=sample_ids)

    df_X.to_csv(HERE / "example_microarray.csv")
    df_y.to_csv(HERE / "example_labels.csv")
    print(f"Saved: example_microarray.csv  ({N_SAMPLES} × {N_FEATURES})")
    print(f"Saved: example_labels.csv      ({N_SAMPLES} samples, classes: {np.unique(y)})")


if __name__ == "__main__":
    build()
