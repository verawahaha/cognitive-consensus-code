# Cognitive Constraints and Collective Consensus

This repository contains the supplementary simulation code for a study of how communication strategies, cognitive capacity constraints, information geometry, and dynamic network perturbations shape collective consensus formation.

The code implements two related simulation modules:

1. `static_consensus.py` — static Watts-Strogatz communication networks with Communicability-Forman (CF) curvature, cognitive-cost-based attention allocation, elaborative and bounded-rationality communication strategies, Difference-in-Differences summaries, bootstrap confidence intervals, permutation tests, and mechanism regressions.
2. `dynamic_perturbation.py` — dynamic networks under exogenous structural disturbance, random or behavioral degree-preserving rewiring, time-series network metrics, and edge-series exports for network visualization.

A minimal synthetic dataset and `demo.py` are included so that the repository can be executed immediately after installation.

## Repository structure

```text
README.md
requirements.txt
demo.py
static_consensus.py
dynamic_perturbation.py
data/
  README.md
  example_input.csv
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation because of the local execution policy, either run the activation command in a temporary process policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

or use `cmd.exe` instead:

```cmd
.venv\Scripts\activate.bat
```

## Quick demo

```bash
python demo.py
```

This creates a synthetic input file and runs small static and dynamic examples. Outputs are written to `results/`.

## Input data

The empirical initialization file should be a CSV with the following columns:

| Column | Description | Scale |
|---|---|---|
| `Group` | Optional group identifier | integer or string |
| `ProAtti` | Initial attitude, used to initialize `x_i(0)` | 1-5 Likert-type item |
| `ConQual1` | Expert or target reference, used to estimate `x*` | 1-5 Likert-type item |
| `CogResource` | Cognitive resource / cognitive capacity measure, used to initialize `r_i` | 1-5 Likert-type item |

All 1-5 Likert-type items are normalized to `[0, 1]` using:

```text
x = (x_raw - 1) / 4
```

In the supplied scripts, `ProAtti` initializes the individual opinion state `x_i(0)`, `CogResource` initializes the individual cognitive-resource parameter `r_i`, and `ConQual1` is normalized and averaged across participants to initialize the target reference value `x*`.

## Static-network simulations

Run a single social-reference intensity `omega`:

```bash
python static_consensus.py \
  --data_path data/example_input.csv \
  --out_dir results/static_single \
  --run_single \
  --omega 0.5 \
  --R 100 \
  --seed 42
```

Run an omega sweep:

```bash
python static_consensus.py \
  --data_path data/example_input.csv \
  --out_dir results/static_sweep \
  --sweep \
  --omegas 0,0.25,0.5,0.75,1 \
  --R 50 \
  --seed 42
```

All output paths below are relative to the directory specified by `--out_dir`.

Main static outputs:

- `results_single_omega_*.csv` — Monte Carlo results for the four Strategy × Modulation conditions.
- `summary_single_omega_*.csv` — mean consensus quality and DiD estimate.
- `results_omega_sweep.csv` — full simulation table across omega values.
- `did_omega_curve.csv` — omega-level DiD, bootstrap confidence intervals, and permutation p values.
- `reg_Bbar.txt` and `reg_Q_on_Bbar.txt` — mechanism regressions.

## Dynamic-network perturbation simulations

Run behavioral rewiring for several perturbation probabilities:

```bash
python dynamic_perturbation.py \
  --data_path data/example_input.csv \
  --out_dir results/dynamic_behavioral \
  --omega 0.5 \
  --p_list 0.02,0.10,0.30 \
  --R 50 \
  --steps 30 \
  --shock_t 10 \
  --rewire_mode behavioral \
  --seed 42
```

Run the random-rewiring control:

```bash
python dynamic_perturbation.py \
  --data_path data/example_input.csv \
  --out_dir results/dynamic_random \
  --omega 0.5 \
  --p_list 0.02,0.10,0.30 \
  --R 50 \
  --steps 30 \
  --shock_t 10 \
  --rewire_mode random \
  --seed 42
```

All output paths below are relative to the directory specified by `--out_dir`.

Main dynamic outputs:

- `dynamic_summary_*.csv` — run-level consensus quality, convergence time, and volatility.
- `timeseries/timeseries_*.csv` — time-series network and consensus metrics, including `deltaE`, `LCC`, `Q_mod`, `p_neg`, `x_var`, and `Q`.
- `edges/edges_*.csv` — edge list at each time step, useful for reconstructing network snapshots.

## Key command-line parameters

The main commands expose a `--seed` argument. We recommend reporting the seed in any reproduction command because the simulations use random initialization, Watts-Strogatz graph generation, attention sampling, Monte Carlo resampling, bootstrap confidence intervals, permutation tests, edge deletion, and rewiring.

| Argument | Used in | Meaning | Recommended reporting |
|---|---|---|---|
| `--seed` | both modules | Base random seed controlling Monte Carlo runs and stochastic network operations | Always report; default is `42` |
| `--R` | both modules | Number of Monte Carlo repetitions | Report with seed |
| `--omega` | both modules | Social-reference intensity | Report |
| `--omegas` | `static_consensus.py` | Comma-separated omega grid for threshold analysis | Report full grid |
| `--p` / `--p_list` | `dynamic_perturbation.py` | Perturbation probability or perturbation grid | Report full grid |
| `--rewire_mode` | `dynamic_perturbation.py` | Rewiring mechanism: `behavioral` or `random` | Report |
| `--shock_t` | `dynamic_perturbation.py` | Time step at which the single structural shock is applied | Report |

The seed is treated as a reproducibility control rather than a model parameter. For robustness checks, run the same parameter grid under multiple seeds and report whether the qualitative conclusions are stable.

## Model summary

### Static networks

Agents are initialized from empirical attitude and cognitive-resource variables. A Watts-Strogatz communication network is constructed, and each edge receives a CF-curvature value. Agents allocate limited attention to a subset of neighbors through a softmax rule that combines bridge preference and cognitive cost. The code compares bounded-rationality and elaborative communication strategies under cognitive modulation and no-modulation conditions.

### Dynamic networks

At the designated shock time, each existing edge is deleted with probability `p`. The network is repaired by approximately degree-preserving rewiring. Random rewiring selects replacement ties uniformly. Behavioral rewiring selects replacement ties according to structural bridge potential, cognitive cost, and opinion similarity. Before candidate edges exist, bridge potential is approximated by Fiedler-vector distance. After rewiring, CF curvature and attention sets are recalculated on the updated network.

## Reproducibility notes

- All random number generation is controlled by command-line seeds.
- The synthetic `data/example_input.csv` is for code testing only and does not reproduce manuscript results.
- To reproduce manuscript analyses, replace `data/example_input.csv` with the empirical initialization data and use the Monte Carlo repetitions, seed values, and parameter grids reported in the supplementary materials.
- CF curvature requires a matrix exponential and can be computationally expensive for large networks and many Monte Carlo repetitions.

## Citation

Please cite the associated manuscript when using this code.
