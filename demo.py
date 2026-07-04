"""Minimal reproducibility demo for the consensus simulation repository."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
EXAMPLE_DATA = os.path.join(DATA_DIR, "example_input.csv")


def make_example_data(path: str, n: int = 80, seed: int = 42) -> None:
    """Create a small synthetic input file with the required column names."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame({
        "Group": rng.integers(1, 5, size=n),
        "ProAtti": rng.integers(1, 6, size=n),
        "ConQual1": rng.integers(2, 6, size=n),
        "CogResource": rng.integers(1, 6, size=n),
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    make_example_data(EXAMPLE_DATA)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    run([
        sys.executable, "static_consensus.py",
        "--data_path", EXAMPLE_DATA,
        "--out_dir", os.path.join(RESULTS_DIR, "demo_static"),
        "--run_single",
        "--omega", "0.5",
        "--R", "3",
        "--bootstrap_B", "20",
        "--perm_B", "20",
    ])

    run([
        sys.executable, "dynamic_perturbation.py",
        "--data_path", EXAMPLE_DATA,
        "--out_dir", os.path.join(RESULTS_DIR, "demo_dynamic"),
        "--omega", "0.5",
        "--p_list", "0.02,0.10",
        "--R", "2",
        "--steps", "12",
        "--shock_t", "5",
        "--rewire_mode", "behavioral",
    ])

    print("\nDemo completed. See the results/ folder for outputs.")


if __name__ == "__main__":
    main()
