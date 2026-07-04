"""
Static consensus simulations under cognitive capacity constraints.

This script implements the static-network component of the supplementary
simulation code for the manuscript. It combines empirically initialized agent
attributes, Watts-Strogatz communication networks, Communicability-Forman (CF)
curvature, attention allocation, and strategy-dependent opinion updating.

Required input CSV columns
--------------------------
Group       : optional group identifier
ProAtti     : initial attitude, 1-5 Likert-type item
ConQual1    : expert/target consensus-quality reference, 1-5 Likert-type item
CogResource : cognitive resource / cognitive capacity measure, 1-5 Likert-type item

Example
-------
python static_consensus.py --data_path data/example_input.csv --out_dir results/static \
    --run_single --omega 0.5 --R 50

python static_consensus.py --data_path data/example_input.csv --out_dir results/sweep \
    --sweep --omegas 0,0.25,0.5,0.75,1 --R 20
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import expm
import statsmodels.formula.api as smf
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


def normalize_likert_1_5(x: np.ndarray | pd.Series) -> np.ndarray:
    """Map 1-5 Likert-type values to [0, 1]."""
    return (np.asarray(x, dtype=float) - 1.0) / 4.0


def softmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = x - np.nanmax(x)
    ex = np.exp(x)
    s = float(np.nansum(ex))
    if not np.isfinite(s) or s <= eps:
        return np.ones_like(x, dtype=float) / len(x)
    return ex / s


def standardize_series(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def safe_mean(values: Iterable[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def gini_coefficient(p: np.ndarray, eps: float = 1e-12) -> float:
    """Gini coefficient for a probability vector."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= eps or p.size == 0:
        return 0.0
    p = np.sort(p / total)
    n = p.size
    i = np.arange(1, n + 1)
    gini = 1.0 - 2.0 * np.sum((n + 1 - i) * p) / n
    return float(np.clip(gini, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Communicability-Forman curvature
# -----------------------------------------------------------------------------

def communicability_kernel(A: np.ndarray, beta: float = 0.8) -> np.ndarray:
    """Communicability kernel exp(beta A)."""
    return expm(beta * A)


def communicability_distance(kernel: np.ndarray, i: int, j: int, eps: float = 1e-12) -> float:
    val = kernel[i, i] + kernel[j, j] - 2.0 * kernel[i, j]
    return float(np.sqrt(max(val, 0.0)) + eps)


def compute_cf_curvature(
    graph: nx.Graph,
    beta: float = 0.8,
    normalize_by_degree: bool = True,
    eps: float = 1e-12,
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    """
    Degree-normalized Communicability-Forman curvature for each edge.

    The normalized version compares the communicability distance of a focal edge
    with the average communicability distances of adjacent edges. Negative
    curvature identifies structurally consequential bridging channels; positive
    curvature indicates more redundant intra-community channels.
    """
    if graph.number_of_edges() == 0:
        return {}, {"mu_C": np.nan, "var_C": np.nan, "p_neg": np.nan}

    A = nx.to_numpy_array(graph, dtype=float)
    K = communicability_kernel(A, beta=beta)

    xi_edge: Dict[Tuple[int, int], float] = {}
    for u, v in graph.edges():
        a, b = sorted((int(u), int(v)))
        xi_edge[(a, b)] = communicability_distance(K, a, b, eps=eps)

    def xi(u: int, v: int) -> float:
        a, b = sorted((int(u), int(v)))
        return xi_edge.get((a, b), communicability_distance(K, a, b, eps=eps))

    curvature: Dict[Tuple[int, int], float] = {}
    values: List[float] = []

    for u, v in graph.edges():
        a, b = sorted((int(u), int(v)))
        xuv = xi(a, b)
        n_a = [k for k in graph.neighbors(a) if k != b]
        n_b = [k for k in graph.neighbors(b) if k != a]

        terms_a = [xuv / xi(a, k) for k in n_a]
        terms_b = [xuv / xi(b, k) for k in n_b]

        if normalize_by_degree:
            s_a = float(np.mean(terms_a)) if terms_a else 0.0
            s_b = float(np.mean(terms_b)) if terms_b else 0.0
        else:
            s_a = float(np.sum(terms_a))
            s_b = float(np.sum(terms_b))

        c_uv = 2.0 - s_a - s_b
        curvature[(a, b)] = float(c_uv)
        values.append(float(c_uv))

    values_arr = np.asarray(values, dtype=float)
    stats = {
        "mu_C": float(np.mean(values_arr)),
        "var_C": float(np.var(values_arr)),
        "p_neg": float(np.mean(values_arr < 0.0)),
    }
    return curvature, stats


# -----------------------------------------------------------------------------
# Simulation model
# -----------------------------------------------------------------------------

@dataclass
class StaticParams:
    # Network parameters
    ws_k: int = 8
    ws_rewire_p: float = 0.10
    beta_comm: float = 0.8

    # Dynamics
    T_max: int = 80
    conv_eps: float = 1e-4
    delta_eps: float = 1e-4
    min_steps: int = 5
    jitter_sd: float = 0.01
    noise_sd: float = 0.0

    # Attention and cognitive cost
    m: int = 5
    bridge_prefer: bool = True
    alpha_cost: float = 1.0
    lambda0: float = 1.0
    lambda1: float = 2.0
    gamma0: float = 1.0
    kappa_E: float = 6.0
    theta_E: float = 0.5

    # Social reference and strategy-specific parameters
    alpha_social: float = 0.7
    omega: float = 0.5
    rho_bias: float = 0.10
    bias_mode: str = "debias"       # "debias" or "motivated"
    beta_L: float = 8.0
    eta_sd: float = 0.05

    # Neighbor aggregation
    neighbor_weight_mode: str = "uniform"  # "uniform", "prob", or "curvature"


def load_empirical_data(path: str) -> Tuple[np.ndarray, np.ndarray, float, pd.DataFrame]:
    """Load and normalize empirical initialization data."""
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = ["ProAtti", "ConQual1", "CogResource"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(data.columns)}")

    x0 = normalize_likert_1_5(data["ProAtti"].values)
    r = normalize_likert_1_5(data["CogResource"].values)
    x_star = float(normalize_likert_1_5(data["ConQual1"].values).mean())
    return x0, r, x_star, data


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return tuple(sorted((int(u), int(v))))


def run_one_simulation(
    x0: np.ndarray,
    r: np.ndarray,
    x_star: float,
    strategy: str,
    modulation: bool,
    params: StaticParams,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Run one Monte Carlo simulation under one strategy/modulation condition."""
    if strategy not in {"finite", "elaborated"}:
        raise ValueError("strategy must be 'finite' or 'elaborated'.")

    N = len(x0)
    graph = nx.watts_strogatz_graph(
        n=N,
        k=params.ws_k,
        p=params.ws_rewire_p,
        seed=int(rng.integers(1, 1_000_000_000)),
    )

    curvature, net_stats = compute_cf_curvature(graph, beta=params.beta_comm)

    if modulation:
        E = sigmoid(params.kappa_E * (r - params.theta_E))
    else:
        E = np.full(N, float(np.mean(r)))

    x = np.asarray(x0, dtype=float).copy()
    if params.jitter_sd > 0:
        x = np.clip(x + rng.normal(0.0, params.jitter_sd, size=N), 0.0, 1.0)

    x_mean_prev = float(x.mean())
    vol_acc = 0.0
    T_conv = params.T_max

    H_list: List[float] = []
    B_list: List[float] = []
    top1_list: List[float] = []
    gini_list: List[float] = []

    def gC(cij: float) -> float:
        return -cij if params.bridge_prefer else cij

    for t in range(params.T_max):
        s_t = params.alpha_social * float(x.mean()) + (1.0 - params.alpha_social) * float(x_star)
        x_new = x.copy()

        for i in range(N):
            neighbors = list(graph.neighbors(i))
            if not neighbors:
                continue
            m_i = min(params.m, len(neighbors))

            gamma_i = params.gamma0 * (1.0 - float(r[i]))
            lambda_i = params.lambda0 + params.lambda1 * float(E[i])

            logits = []
            curvatures = []
            for j in neighbors:
                cij = curvature.get(_edge_key(i, j), 0.0)
                curvatures.append(cij)
                cost = abs(cij) ** params.alpha_cost
                logits.append(lambda_i * gC(cij) - gamma_i * cost)

            p_all = softmax(np.asarray(logits, dtype=float))
            chosen_idx = rng.choice(len(neighbors), size=m_i, replace=False, p=p_all)
            chosen = [neighbors[k] for k in chosen_idx]

            p_chosen = p_all[chosen_idx]
            p_chosen = p_chosen / max(float(p_chosen.sum()), 1e-12)
            H_list.append(float(-(p_chosen * np.log(np.clip(p_chosen, 1e-12, 1.0))).sum()))
            B_list.append(float(np.mean([curvature.get(_edge_key(i, j), 0.0) < 0.0 for j in chosen])))
            top1_list.append(float(np.max(p_all)))
            gini_list.append(gini_coefficient(p_all))

            if params.neighbor_weight_mode == "uniform":
                w = np.ones(m_i, dtype=float) / m_i
            elif params.neighbor_weight_mode == "prob":
                w = p_chosen
            elif params.neighbor_weight_mode == "curvature":
                selected_curv = np.asarray([curvature.get(_edge_key(i, j), 0.0) for j in chosen], dtype=float)
                w = softmax(np.asarray([gC(c) for c in selected_curv], dtype=float))
            else:
                raise ValueError("neighbor_weight_mode must be 'uniform', 'prob', or 'curvature'.")

            neighbor_term = float(np.sum(w * np.asarray([x[j] for j in chosen], dtype=float)))

            if strategy == "finite":
                x_new[i] = (1.0 - params.omega) * neighbor_term + params.omega * s_t
            else:
                if params.bias_mode == "debias":
                    bias_i = params.rho_bias * (-(float(r[i]) - 0.5))
                elif params.bias_mode == "motivated":
                    bias_i = params.rho_bias * (+(float(r[i]) - 0.5))
                else:
                    raise ValueError("bias_mode must be 'debias' or 'motivated'.")

                private_signal = float(x_star + bias_i + rng.normal(0.0, params.eta_sd))
                load_i = float(m_i) / float(max(params.m, 1))
                qA = float(np.exp(-params.beta_L * max(0.0, load_i - float(r[i])) ** 2))
                x_new[i] = (1.0 - params.omega) * ((1.0 - qA) * neighbor_term + qA * private_signal) + params.omega * s_t

            if params.noise_sd > 0:
                x_new[i] += rng.normal(0.0, params.noise_sd)
            x_new[i] = float(np.clip(x_new[i], 0.0, 1.0))

        vol_acc += abs(float(x_new.mean()) - x_mean_prev)
        x_mean_prev = float(x_new.mean())

        if t >= params.min_steps and np.var(x_new) < params.conv_eps and np.max(np.abs(x_new - x)) < params.delta_eps:
            T_conv = t
            x = x_new
            break
        x = x_new

    consensus = 1.0 - float(np.var(x))
    accuracy = 1.0 - abs(float(x.mean()) - float(x_star))
    Q = float(consensus * accuracy)

    out = {
        "Q": Q,
        "Consensus": float(consensus),
        "Accuracy": float(accuracy),
        "T_conv": float(T_conv),
        "Vol": float(vol_acc) / float(max(T_conv, 1)),
        "B_bar": safe_mean(B_list),
        "H_bar": safe_mean(H_list),
        "AttentionTop1_bar": safe_mean(top1_list),
        "AttentionGini_bar": safe_mean(gini_list),
    }
    out.update(net_stats)
    return out


def run_monte_carlo(
    data: pd.DataFrame,
    params: StaticParams,
    R: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run the 2 x 2 strategy/modulation Monte Carlo design."""
    rng = np.random.default_rng(seed)
    x0 = normalize_likert_1_5(data["ProAtti"].values.astype(float))
    r = normalize_likert_1_5(data["CogResource"].values.astype(float))
    x_star = float(normalize_likert_1_5(data["ConQual1"].values.astype(float)).mean())

    conditions = [
        ("finite", False),
        ("finite", True),
        ("elaborated", False),
        ("elaborated", True),
    ]

    rows = []
    for run_id in tqdm(range(R), desc=f"Static MC omega={params.omega:.2f}", unit="run"):
        perm = rng.permutation(len(x0))
        x0_run = x0[perm]
        r_run = r[perm]

        for strategy, modulation in conditions:
            out = run_one_simulation(
                x0=x0_run,
                r=r_run,
                x_star=x_star,
                strategy=strategy,
                modulation=modulation,
                params=params,
                rng=rng,
            )
            out.update({
                "run": int(run_id),
                "Strategy": strategy,
                "Modulation": bool(modulation),
                "omega": float(params.omega),
                "x_star": float(x_star),
            })
            rows.append(out)

    df = pd.DataFrame(rows)
    meanQ = df.groupby(["Strategy", "Modulation"])["Q"].mean()
    summary = {
        "omega": float(params.omega),
        "R": int(R),
        "x_star": float(x_star),
        "Q_C1_finite_no_mod": float(meanQ.get(("finite", False), np.nan)),
        "Q_C2_finite_mod": float(meanQ.get(("finite", True), np.nan)),
        "Q_C3_elaborated_no_mod": float(meanQ.get(("elaborated", False), np.nan)),
        "Q_C4_elaborated_mod": float(meanQ.get(("elaborated", True), np.nan)),
        "DiD_Q": calc_did(df, ycol="Q"),
    }
    return df, summary


# -----------------------------------------------------------------------------
# Statistical summaries
# -----------------------------------------------------------------------------

def calc_did(df: pd.DataFrame, ycol: str = "Q") -> float:
    """DiD = (C4 - C3) - (C2 - C1)."""
    m = df.groupby(["Strategy", "Modulation"])[ycol].mean()
    C1 = float(m.get(("finite", False), np.nan))
    C2 = float(m.get(("finite", True), np.nan))
    C3 = float(m.get(("elaborated", False), np.nan))
    C4 = float(m.get(("elaborated", True), np.nan))
    return float((C4 - C3) - (C2 - C1))


def bootstrap_did_ci(df: pd.DataFrame, B: int = 1000, ycol: str = "Q", seed: int = 123) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    runs = df["run"].unique()
    dids = []
    for _ in range(B):
        sampled_runs = rng.choice(runs, size=len(runs), replace=True)
        sdf = pd.concat([df[df["run"] == r] for r in sampled_runs], ignore_index=True)
        dids.append(calc_did(sdf, ycol=ycol))
    return float(np.quantile(dids, 0.025)), float(np.quantile(dids, 0.975))


def permutation_test_did(df: pd.DataFrame, B: int = 1000, ycol: str = "Q", seed: int = 456) -> float:
    rng = np.random.default_rng(seed)
    observed = calc_did(df, ycol=ycol)
    null = []
    for _ in range(B):
        sdf = df.copy()
        for _, g in sdf.groupby("run"):
            labels = list(zip(g["Strategy"].values, g["Modulation"].values))
            rng.shuffle(labels)
            sdf.loc[g.index, "Strategy"] = [a for a, _ in labels]
            sdf.loc[g.index, "Modulation"] = [b for _, b in labels]
        null.append(calc_did(sdf, ycol=ycol))
    null = np.asarray(null, dtype=float)
    return float(np.mean(np.abs(null) >= abs(observed)))


def run_regressions(df: pd.DataFrame, out_dir: str) -> None:
    """Save heteroskedasticity-robust OLS summaries used for mechanism checks."""
    d = df.copy()
    for c in ["B_bar", "H_bar", "p_neg", "var_C", "omega", "T_conv", "Vol", "Q"]:
        if c in d.columns:
            d[c + "_z"] = standardize_series(d[c])

    if {"B_bar_z", "p_neg_z", "var_C_z"}.issubset(d.columns):
        m1 = smf.ols("B_bar_z ~ C(Strategy) * C(Modulation) + p_neg_z + var_C_z", data=d).fit(cov_type="HC3")
        with open(os.path.join(out_dir, "reg_Bbar.txt"), "w", encoding="utf-8") as f:
            f.write(m1.summary().as_text())

    if {"Q_z", "B_bar_z", "p_neg_z", "var_C_z"}.issubset(d.columns):
        m2 = smf.ols("Q_z ~ B_bar_z + C(Strategy) + C(Modulation) + p_neg_z + var_C_z", data=d).fit(cov_type="HC3")
        with open(os.path.join(out_dir, "reg_Q_on_Bbar.txt"), "w", encoding="utf-8") as f:
            f.write(m2.summary().as_text())


def parse_omegas(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Static consensus simulations under cognitive constraints.")
    parser.add_argument("--data_path", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--out_dir", type=str, default="results/static", help="Output directory.")
    parser.add_argument("--run_single", action="store_true", help="Run a single omega value.")
    parser.add_argument("--sweep", action="store_true", help="Run an omega sweep.")
    parser.add_argument("--omega", type=float, default=0.5, help="Social reference intensity.")
    parser.add_argument("--omegas", type=str, default="0,0.25,0.5,0.75,1", help="Comma-separated omega values for sweep.")
    parser.add_argument("--R", type=int, default=100, help="Monte Carlo repetitions per omega.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bootstrap_B", type=int, default=1000, help="Bootstrap iterations for DiD CI.")
    parser.add_argument("--perm_B", type=int, default=1000, help="Permutation iterations for DiD p value.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = pd.read_csv(args.data_path, encoding="utf-8-sig")

    if args.sweep:
        all_rows = []
        summary_rows = []
        for idx, omega in enumerate(parse_omegas(args.omegas)):
            params = StaticParams(omega=float(omega))
            df, summary = run_monte_carlo(data=data, params=params, R=args.R, seed=args.seed + idx * 97)
            ci_lo, ci_hi = bootstrap_did_ci(df, B=args.bootstrap_B, ycol="Q", seed=args.seed + idx * 131)
            p_perm = permutation_test_did(df, B=args.perm_B, ycol="Q", seed=args.seed + idx * 151)
            summary.update({"CI_lo": ci_lo, "CI_hi": ci_hi, "p_perm": p_perm})
            all_rows.append(df)
            summary_rows.append(summary)

        all_df = pd.concat(all_rows, ignore_index=True)
        summary_df = pd.DataFrame(summary_rows)
        all_df.to_csv(os.path.join(args.out_dir, "results_omega_sweep.csv"), index=False, encoding="utf-8-sig")
        summary_df.to_csv(os.path.join(args.out_dir, "did_omega_curve.csv"), index=False, encoding="utf-8-sig")
        run_regressions(all_df, args.out_dir)
        print(f"[ok] Saved omega-sweep outputs to {args.out_dir}")

    else:
        params = StaticParams(omega=float(args.omega))
        df, summary = run_monte_carlo(data=data, params=params, R=args.R, seed=args.seed)
        ci_lo, ci_hi = bootstrap_did_ci(df, B=args.bootstrap_B, ycol="Q", seed=args.seed + 1)
        p_perm = permutation_test_did(df, B=args.perm_B, ycol="Q", seed=args.seed + 2)
        summary.update({"CI_lo": ci_lo, "CI_hi": ci_hi, "p_perm": p_perm})

        df.to_csv(os.path.join(args.out_dir, f"results_single_omega_{args.omega:.2f}.csv"), index=False, encoding="utf-8-sig")
        pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, f"summary_single_omega_{args.omega:.2f}.csv"), index=False, encoding="utf-8-sig")
        run_regressions(df, args.out_dir)
        print(f"[ok] Saved single-omega outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
