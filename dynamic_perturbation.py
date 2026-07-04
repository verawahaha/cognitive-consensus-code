# -*- coding: utf-8 -*-
"""
Dynamic consensus simulations under exogenous perturbation and rewiring.

This script implements the dynamic-network component of the supplementary code.
A Watts-Strogatz communication network is perturbed by stochastic edge deletion
and then repaired through either random rewiring or behavioral rewiring.

The behavioral rewiring rule uses a spectral proxy for bridge potential before
new edges exist: candidate ties with larger Fiedler-vector distance are treated
as more likely cross-region bridges. After rewiring, CF curvature and attention
sets are recalculated on the updated network.

Example
-------
python dynamic_perturbation.py --data_path data/example_input.csv \
    --out_dir results/dynamic --omega 0.5 --p_list 0.02,0.10,0.30 \
    --R 30 --steps 30 --shock_t 10 --rewire_mode behavioral
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.linalg import eigsh
from tqdm import tqdm

from static_consensus import (
    StaticParams,
    normalize_likert_1_5,
    compute_cf_curvature,
    softmax,
    safe_mean,
)


# -----------------------------------------------------------------------------
# Dynamic-network utilities
# -----------------------------------------------------------------------------

def edge_set_undirected(graph: nx.Graph) -> set[Tuple[int, int]]:
    return {tuple(sorted((int(u), int(v)))) for u, v in graph.edges()}


def largest_connected_component_share(graph: nx.Graph) -> float:
    n = graph.number_of_nodes()
    if n == 0:
        return float("nan")
    if graph.number_of_edges() == 0:
        return 1.0 / n
    largest = max((len(c) for c in nx.connected_components(graph)), default=0)
    return float(largest / n)


def greedy_modularity(graph: nx.Graph) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        communities = list(greedy_modularity_communities(graph))
        if len(communities) <= 1:
            return 0.0
        return float(modularity(graph, communities))
    except Exception:
        return float("nan")


def random_edge_deletion(graph: nx.Graph, p: float, rng: np.random.Generator) -> int:
    """Delete each existing edge with probability p."""
    deleted = 0
    for u, v in list(graph.edges()):
        if rng.random() < p and graph.has_edge(u, v):
            graph.remove_edge(u, v)
            deleted += 1
    return deleted


def fiedler_proxy(graph: nx.Graph) -> np.ndarray:
    """Return standardized Fiedler vector as a spectral proxy for bridge potential."""
    n = graph.number_of_nodes()
    if graph.number_of_edges() == 0 or n <= 2:
        return np.zeros(n, dtype=float)

    L_sparse = nx.laplacian_matrix(graph).astype(float)
    try:
        _, vecs = eigsh(L_sparse, k=2, which="SM", maxiter=20000, tol=1e-6)
        f2 = vecs[:, 1]
    except Exception:
        L_dense = L_sparse.toarray()
        _, vecs = np.linalg.eigh(L_dense)
        f2 = vecs[:, 1] if vecs.shape[1] >= 2 else vecs[:, 0]

    sd = float(np.std(f2))
    if sd < 1e-12:
        return np.zeros_like(f2)
    return (f2 - float(np.mean(f2))) / sd


@dataclass
class DynamicParams(StaticParams):
    perturb_p: float = 0.02
    shock_t: int = 10
    steps: int = 30
    rewire_mode: str = "behavioral"   # "behavioral" or "random"
    tau_homophily: float = 1.0
    max_rewire_tries: int = 8000


def degree_preserving_rewire(
    graph: nx.Graph,
    target_degree: Dict[int, int],
    x: np.ndarray,
    r: np.ndarray,
    params: DynamicParams,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Repair deleted edges while approximately preserving baseline node degree."""
    z = fiedler_proxy(graph)
    nodes = list(graph.nodes())
    Cnew_list: List[float] = []
    Dnew_list: List[float] = []
    Costnew_list: List[float] = []
    tries = 0

    for i in nodes:
        while graph.degree(i) < int(target_degree[i]):
            tries += 1
            if tries > params.max_rewire_tries:
                break

            neighbors = set(graph.neighbors(i))
            candidates = [k for k in nodes if k != i and k not in neighbors]
            if not candidates:
                break

            if params.rewire_mode == "random":
                k = int(rng.choice(candidates))
                c_tilde = 0.0
            elif params.rewire_mode == "behavioral":
                logits = []
                c_cache = []
                lambda_i = params.lambda0 + params.lambda1 * float(r[i])
                gamma_i = params.gamma0 * (1.0 - float(r[i]))

                for k_candidate in candidates:
                    c_tilde_candidate = -abs(float(z[i] - z[k_candidate]))
                    bridge_value = -c_tilde_candidate if params.bridge_prefer else c_tilde_candidate
                    cost = abs(c_tilde_candidate) ** params.alpha_cost
                    homophily_penalty = abs(float(x[i] - x[k_candidate]))
                    logits.append(lambda_i * bridge_value - gamma_i * cost - params.tau_homophily * homophily_penalty)
                    c_cache.append(c_tilde_candidate)

                probs = softmax(np.asarray(logits, dtype=float))
                idx = int(rng.choice(len(candidates), p=probs))
                k = int(candidates[idx])
                c_tilde = float(c_cache[idx])
            else:
                raise ValueError("rewire_mode must be 'behavioral' or 'random'.")

            graph.add_edge(int(i), int(k))
            Cnew_list.append(float(c_tilde))
            Dnew_list.append(float(abs(x[i] - x[k])))
            Costnew_list.append(float(abs(c_tilde) ** params.alpha_cost))

    return {
        "Cnew_bar": safe_mean(Cnew_list),
        "Dnew_bar": safe_mean(Dnew_list),
        "Costnew_bar": safe_mean(Costnew_list),
        "rewire_tries": float(tries),
    }


def _edge_key(u: int, v: int) -> Tuple[int, int]:
    return tuple(sorted((int(u), int(v))))


def attention_selection(graph: nx.Graph, curvature: Dict[Tuple[int, int], float], params: DynamicParams, rng: np.random.Generator) -> Dict[int, List[int]]:
    """Select an attention set for each node using curvature-biased sampling."""
    selected: Dict[int, List[int]] = {}
    for i in graph.nodes():
        neighbors = list(graph.neighbors(i))
        if len(neighbors) <= params.m:
            selected[int(i)] = [int(j) for j in neighbors]
            continue
        logits = []
        for j in neighbors:
            c_ij = curvature.get(_edge_key(i, j), 0.0)
            bridge_value = -c_ij if params.bridge_prefer else c_ij
            cost = abs(c_ij) ** params.alpha_cost
            logits.append(params.lambda0 * bridge_value - params.gamma0 * cost)
        probs = softmax(np.asarray(logits, dtype=float))
        selected[int(i)] = [int(neighbors[k]) for k in rng.choice(len(neighbors), size=params.m, replace=False, p=probs)]
    return selected


def update_opinions(x: np.ndarray, attention: Dict[int, List[int]], omega: float, social_signal: float) -> np.ndarray:
    """Bounded-rationality opinion update used in the dynamic perturbation model."""
    x_new = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        neighbors = attention.get(i, [])
        neighbor_mean = float(np.mean([x[j] for j in neighbors])) if neighbors else float(x[i])
        x_new[i] = (1.0 - omega) * neighbor_mean + omega * social_signal
    return np.clip(x_new, 0.0, 1.0)


def compute_consensus_metrics(x: np.ndarray, x_star: float) -> Tuple[float, float, float]:
    variance = float(np.var(x))
    consensus = 1.0 - variance
    accuracy = 1.0 - abs(float(np.mean(x)) - float(x_star))
    Q = float(consensus * accuracy)
    return Q, consensus, accuracy


def simulate_dynamic_once(
    x0: np.ndarray,
    r: np.ndarray,
    x_star: float,
    params: DynamicParams,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Run one dynamic-network simulation and return summary, time series, and edge series."""
    N = len(x0)
    graph = nx.watts_strogatz_graph(
        n=N,
        k=params.ws_k,
        p=params.ws_rewire_p,
        seed=int(rng.integers(1, 1_000_000_000)),
    )
    target_degree = dict(graph.degree())
    x = np.asarray(x0, dtype=float).copy()

    curvature, net_stats = compute_cf_curvature(graph, beta=params.beta_comm)
    prev_edges = edge_set_undirected(graph)

    Q_values: List[float] = []
    variance_values: List[float] = []
    ts_records: List[Dict[str, float]] = []
    edge_records: List[Dict[str, int]] = []

    for t in range(params.steps):
        graph_changed = False
        deleted_edges = 0
        rewire_stats = {"Cnew_bar": np.nan, "Dnew_bar": np.nan, "Costnew_bar": np.nan, "rewire_tries": 0.0}

        if t == params.shock_t:
            deleted_edges = random_edge_deletion(graph, params.perturb_p, rng)
            if deleted_edges > 0:
                graph_changed = True
            rewire_stats = degree_preserving_rewire(graph, target_degree, x, r, params, rng)
            graph_changed = True

        current_edges = edge_set_undirected(graph)
        deltaE = float(len(prev_edges.symmetric_difference(current_edges)) / max(1, len(prev_edges)))
        prev_edges = current_edges

        if graph_changed:
            curvature, net_stats = compute_cf_curvature(graph, beta=params.beta_comm)

        attention = attention_selection(graph, curvature, params, rng)
        social_signal = float(x_star)
        x = update_opinions(x, attention, omega=params.omega, social_signal=social_signal)

        Q, consensus, accuracy = compute_consensus_metrics(x, x_star)
        Q_values.append(Q)
        variance_values.append(float(np.var(x)))

        ts_records.append({
            "t": int(t),
            "deltaE": float(deltaE),
            "deleted_edges": float(deleted_edges),
            "LCC": largest_connected_component_share(graph),
            "Q_mod": greedy_modularity(graph),
            "p_neg": float(net_stats.get("p_neg", np.nan)),
            "mu_C": float(net_stats.get("mu_C", np.nan)),
            "var_C": float(net_stats.get("var_C", np.nan)),
            "x_mean": float(np.mean(x)),
            "x_var": float(np.var(x)),
            "Q": float(Q),
            "Consensus": float(consensus),
            "Accuracy": float(accuracy),
            **rewire_stats,
        })

        for u, v in graph.edges():
            edge_records.append({"t": int(t), "source": int(u), "target": int(v)})

    T_conv = next((idx for idx, v in enumerate(variance_values) if v < params.conv_eps), params.steps)
    Vol = float(np.mean(np.abs(np.diff(Q_values)))) if len(Q_values) > 1 else 0.0

    return {
        "Q": float(Q_values[-1]),
        "T_conv": int(T_conv),
        "Vol": float(Vol),
        "ts": pd.DataFrame(ts_records),
        "edges": pd.DataFrame(edge_records),
    }


def run_dynamic_experiment(
    data_path: str,
    out_dir: str,
    omega: float,
    p: float,
    R: int,
    steps: int,
    shock_t: int,
    rewire_mode: str,
    seed: int = 42,
) -> None:
    """Run R independent dynamic simulations and save outputs."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    required = ["ProAtti", "CogResource"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(data.columns)}")

    x0 = normalize_likert_1_5(data["ProAtti"].values.astype(float))
    r = normalize_likert_1_5(data["CogResource"].values.astype(float))
    if "ConQual1" in data.columns:
        x_star = float(normalize_likert_1_5(data["ConQual1"].values.astype(float)).mean())
    else:
        x_star = 0.5

    params = DynamicParams(omega=omega, perturb_p=p, steps=steps, shock_t=shock_t, rewire_mode=rewire_mode)

    summary_rows = []
    ts_all = []
    edges_all = []

    for run_id in tqdm(range(R), desc=f"Dynamic {rewire_mode} p={p}", unit="run"):
        perm = rng.permutation(len(x0))
        result = simulate_dynamic_once(x0=x0[perm], r=r[perm], x_star=x_star, params=params, rng=rng)

        summary_rows.append({
            "run_id": int(run_id),
            "omega": float(omega),
            "p": float(p),
            "rewire_mode": rewire_mode,
            "Q": result["Q"],
            "T_conv": result["T_conv"],
            "Vol": result["Vol"],
        })

        ts_df = result["ts"].copy()
        ts_df["run_id"] = int(run_id)
        ts_df["omega"] = float(omega)
        ts_df["p"] = float(p)
        ts_df["rewire_mode"] = rewire_mode
        ts_all.append(ts_df)

        edge_df = result["edges"].copy()
        edge_df["run_id"] = int(run_id)
        edge_df["omega"] = float(omega)
        edge_df["p"] = float(p)
        edge_df["rewire_mode"] = rewire_mode
        edges_all.append(edge_df)

    p_label = f"{p:g}".replace(".", "p")
    omega_label = f"{omega:g}".replace(".", "p")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, f"dynamic_summary_{rewire_mode}_p{p_label}_omega{omega_label}.csv"), index=False, encoding="utf-8-sig")

    ts_dir = os.path.join(out_dir, "timeseries")
    edge_dir = os.path.join(out_dir, "edges")
    os.makedirs(ts_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)
    pd.concat(ts_all, ignore_index=True).to_csv(os.path.join(ts_dir, f"timeseries_{rewire_mode}_p{p_label}_omega{omega_label}.csv"), index=False, encoding="utf-8-sig")
    pd.concat(edges_all, ignore_index=True).to_csv(os.path.join(edge_dir, f"edges_{rewire_mode}_p{p_label}_omega{omega_label}.csv"), index=False, encoding="utf-8-sig")

    print(f"[ok] Saved dynamic outputs to {out_dir}")
    print(summary[["Q", "T_conv", "Vol"]].mean())


def parse_p_list(text: str | None, default_p: float) -> List[float]:
    if text is None:
        return [default_p]
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic consensus simulations under exogenous perturbation.")
    parser.add_argument("--data_path", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--out_dir", type=str, default="results/dynamic", help="Output directory.")
    parser.add_argument("--omega", type=float, default=0.5, help="Social reference intensity.")
    parser.add_argument("--p", type=float, default=0.02, help="Single perturbation probability.")
    parser.add_argument("--p_list", type=str, default=None, help="Comma-separated perturbation probabilities.")
    parser.add_argument("--R", type=int, default=50, help="Monte Carlo repetitions.")
    parser.add_argument("--steps", type=int, default=30, help="Number of time steps.")
    parser.add_argument("--shock_t", type=int, default=10, help="Shock time for single-shock simulations.")
    parser.add_argument("--rewire_mode", type=str, default="behavioral", choices=["behavioral", "random"], help="Rewiring mechanism.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for idx, p in enumerate(parse_p_list(args.p_list, args.p)):
        run_dynamic_experiment(
            data_path=args.data_path,
            out_dir=args.out_dir,
            omega=args.omega,
            p=float(p),
            R=args.R,
            steps=args.steps,
            shock_t=args.shock_t,
            rewire_mode=args.rewire_mode,
            seed=args.seed + idx * 97,
        )


if __name__ == "__main__":
    main()
