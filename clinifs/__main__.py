"""
clinifs command-line interface.

Usage
-----
    clinifs --help
    clinifs version
    clinifs methods
    clinifs demo
    clinifs run --x X.csv --y y.csv [--method auto] [--k 20] [--out selected.csv]
"""
import argparse
import sys


def _cmd_version(args):
    from clinifs import __version__
    print(f"clinifs {__version__}")


def _cmd_methods(args):
    rows = [
        ("Filter",   "variance",   "VarianceSelector",  "unsupervised top-k by sample variance",    "⚡"),
        ("Filter",   "anova",      "ANOVASelector",     "top-k by ANOVA F-score",                   "⚡"),
        ("Filter",   "mi",         "MISelector",        "top-k by Mutual Information",              "⚡"),
        ("Filter",   "mrmr",       "mRMRSelector",      "min-redundancy max-relevance (Ding 2005)",  "🔵"),
        ("Filter",   "relieff",    "ReliefFSelector",   "ReliefF (skrebate)",                       "🔵"),
        ("Embedded", "l1",         "L1LogisticSelector","L1-regularised logistic regression",        "⚡"),
        ("Embedded", "elasticnet", "ElasticNetSelector","ElasticNet (L1+L2)",                        "🔵"),
        ("Embedded", "linsvc",     "LinSVCL1Selector",  "LinearSVC with L1 penalty",                "⚡"),
        ("Wrapper",  "boruta",     "BorutaSelector",    "Boruta all-relevant selection",             "🔴"),
        ("Wrapper",  "extratrees", "ExtraTreesSelector","ExtraTrees importance ranking",             "⚡"),
        ("Wrapper",  "rfecv",      "RFECVSelector",     "Recursive Feature Elimination + CV",       "🔵"),
        ("Wrapper",  "ga",         "GASelector",        "Genetic Algorithm (custom)",                "🔴"),
        ("Wrapper",  "bpso",       "BPSOSelector",      "Binary Particle Swarm Optimisation",        "🔴"),
        ("RRA",      "rra/auto",   "RankAggregateFilter","Adaptive Robust Rank Aggregation",         "⚡"),
    ]
    print(f"\n{'Family':<12} {'Key':<12} {'Class':<22} {'Description':<45} Speed")
    print("-" * 105)
    for fam, key, cls, desc, spd in rows:
        print(f"{fam:<12} {key:<12} {cls:<22} {desc:<45} {spd}")
    print("\nSpeed legend: ⚡ <1s  🔵 1–10s  🔴 30–120s  (microarray, after ANOVA pre-filter)\n")


def _cmd_demo(args):
    import os, pathlib
    here = pathlib.Path(__file__).parent
    demo_script = here.parent / "examples" / "quickstart.py"
    if not demo_script.exists():
        print(f"[clinifs demo] example script not found at {demo_script}")
        print("Run: clinifs run --x examples/example_microarray.csv "
              "--y examples/example_labels.csv --method auto --k 20")
        sys.exit(1)
    import runpy
    runpy.run_path(str(demo_script), run_name="__main__")


def _cmd_run(args):
    import numpy as np
    import pandas as pd
    from clinifs import FeatureSelector

    try:
        X = pd.read_csv(args.x, index_col=0).values.astype(float)
    except Exception as e:
        print(f"[clinifs run] Cannot read X from '{args.x}': {e}")
        sys.exit(1)
    try:
        y_raw = pd.read_csv(args.y, index_col=0).values.ravel()
        from sklearn.preprocessing import LabelEncoder
        y = LabelEncoder().fit_transform(y_raw)
    except Exception as e:
        print(f"[clinifs run] Cannot read y from '{args.y}': {e}")
        sys.exit(1)

    print(f"[clinifs run] X shape: {X.shape}, classes: {np.unique(y)}")
    print(f"[clinifs run] method={args.method}, k={args.k}")

    fs = FeatureSelector(method=args.method, k=args.k)
    fs.fit(X, y)

    tier = getattr(fs, "tier_", "N/A")
    print(f"[clinifs run] Tier detected : {tier}")
    print(f"[clinifs run] Selected {len(fs.selected_indices_)} features")

    out_df = pd.DataFrame({"feature_index": fs.selected_indices_})
    out_df.to_csv(args.out, index=False)
    print(f"[clinifs run] Results saved to '{args.out}'")


def main():
    parser = argparse.ArgumentParser(
        prog="clinifs",
        description="clinifs — Clinical Feature Selection Toolkit",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    sub.add_parser("version", help="Print version and exit")
    sub.add_parser("methods", help="List all available feature selection methods")
    sub.add_parser("demo",    help="Run the built-in quickstart demo")

    run_p = sub.add_parser("run", help="Run feature selection on your data")
    run_p.add_argument("--x",      required=True,      help="Path to feature matrix CSV (rows=samples, cols=genes)")
    run_p.add_argument("--y",      required=True,      help="Path to label CSV (single column)")
    run_p.add_argument("--method", default="auto",     help="Method key (default: auto)")
    run_p.add_argument("--k",      type=int, default=20, help="Panel size k (default: 20)")
    run_p.add_argument("--out",    default="selected_features.csv", help="Output CSV path")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "version": _cmd_version,
        "methods": _cmd_methods,
        "demo":    _cmd_demo,
        "run":     _cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
