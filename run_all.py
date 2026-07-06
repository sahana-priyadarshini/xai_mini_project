"""
run_all.py
==========
Single entry point to reproduce the full pipeline in one command:

    python run_all.py

Runs, in order:
    1. 01_data_download.py  - fetch/build the DBpedia graph (reuses data/
                               cache if present; fetches fresh from DBpedia
                               otherwise, including all predicates needed
                               -- no separate "expand predicates" step is
                               needed for a fresh run)
    2. 02_train_model.py    - train the R-GCN (final tuned architecture)
    3. 03_explain.py        - generate explanations (GNNExplainer,
                               PGExplainer, SubGraphX)
    4. 04_evaluate.py       - quantitatively evaluate the explanations
                               (fidelity, sparsity, stability)

Each step is run as a subprocess using the same Python interpreter that
is running this script (so it automatically uses your active conda
environment). If any step fails, this stops immediately and reports which
step failed and its exit code, rather than silently continuing with a
broken pipeline.

All results (tables, plots, model checkpoint, explanations) are written
to outputs/ as each step completes.
"""

import subprocess
import sys
import time

STEPS = [
    ("Step 1/4 - Data download & graph construction", "01_data_download.py"),
    ("Step 2/4 - Model training", "02_train_model.py"),
    ("Step 3/4 - Generating explanations", "03_explain.py"),
    ("Step 4/4 - Evaluating explanations", "04_evaluate.py"),
]


def main() -> None:
    overall_start = time.time()

    for label, script in STEPS:
        print("\n" + "=" * 70)
        print(f"  {label}  ({script})")
        print("=" * 70)

        t0 = time.time()
        result = subprocess.run([sys.executable, script])
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n[run_all] FAILED at '{script}' (exit code {result.returncode}) "
                  f"after {elapsed:.1f}s. Stopping - see the error output above.")
            sys.exit(result.returncode)

        print(f"[run_all] '{script}' completed in {elapsed:.1f}s.")

    total = time.time() - overall_start
    print("\n" + "=" * 70)
    print(f"  ALL STEPS COMPLETE in {total / 60:.1f} minutes.")
    print("  See outputs/ for all results (tables, plots, explanations).")
    print("=" * 70)


if __name__ == "__main__":
    main()
