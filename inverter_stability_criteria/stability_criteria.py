#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to check the stability criteria developed in the paper."""

import numpy as np
import pandas as pd

from inverter_stability_criteria.system_classes import System
from inverter_stability_criteria.formulate_matrices import edges_from_BB, build_A_matrices
from inverter_stability_criteria.powerflow_solution import solve_lossless_pv_pq_power_flow


def solve_operating_point_system(system: System, metadata: dict, theta0: np.ndarray | None = None,
                                 V0: np.ndarray | None = None, tol: float = 1e-10,
                                 max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Solve the slack/PV/PQ operating point for a System.

    Bus sets and base state are taken from `metadata`, which must contain
    slack_bus, pv_buses, pq_buses, theta_base, V_base and V_set.
    They must be indexed by Ybus row, not by pandapower bus number.
    
    With theta0 = None, the solver starts from the base state (theta_base,
    V_base). Passing theta0, and optionally V0, the solver starts from that
    instead. If theta0 is given without V0, the starting voltage
    is taken from system.volt_operating_point.
    """
    slack = int(metadata["slack_bus"])
    pv_buses = np.asarray(metadata["pv_buses"], dtype=int)
    pq_buses = np.asarray(metadata["pq_buses"], dtype=int)

    if theta0 is None:
        theta_init = np.asarray(metadata["theta_base"], dtype=float).copy()
        V_init = np.asarray(metadata["V_base"], dtype=float).copy()

    else:
        theta_init = np.asarray(theta0, dtype=float).copy()
        V_init = (system.volt_operating_point.copy() if V0 is None else np.asarray(V0, dtype=float).copy())

    V_set = np.asarray(metadata["V_set"], dtype=float)

    theta, V, info = solve_lossless_pv_pq_power_flow(system=system, theta0=theta_init, V0=V_init, slack_bus=slack,
                                                         pv_buses=pv_buses, pq_buses=pq_buses, V_set=V_set, tol=tol,
                                                         max_iter=max_iter)

    return theta, V, info

def empty_stability_metrics() -> dict:
    """NaN-filled stability metrics, used for non-converged operating points."""    
    return {
        "k_exact": np.nan,
        "k_cor1": np.nan,
        "k_cor2": np.nan,
        "k_cor3": np.nan,
        "k_cor4": np.nan,
        "k_ratio": np.nan,
        "cycle_on_critical_mode": np.nan,
        "max_angle": np.nan,
        "cycle_eigs": np.array([]),
    }


# Nonuniform gains

def prepare_gain_profile(h: np.ndarray, n_nodes: int) -> np.ndarray:

    """Validate and return the gain profile h."""

    h = np.asarray(h, dtype=float)
    if h.shape != (n_nodes,) or np.any(~np.isfinite(h)) or np.any(h <= 0):
        raise ValueError(f"Gain profile must have shape {(n_nodes,)} and be finite and positive, got {h}.")
    return h

def critical_alpha_from_A(A: np.ndarray, h: np.ndarray, tol: float = 1e-12) -> float:
    """
    Largest alpha with diag(1/(alpha h)) + A positive definite.

    For k_i^q = alpha h_i, this is the critical droop level. The condition is equivalent to
        alpha * diag(sqrt(h)) A diag(sqrt(h)) > -1 => alpha_crit = -1 / lambda_min.
    Returns inf if A is positive semi-definite, i.e. if finite gain cannot destabilize the system.
    """
    sqrt_h = np.sqrt(h)
    A_tilde = (sqrt_h[:, None] * A) * sqrt_h[None, :]

    lam_min = np.linalg.eigvalsh(A_tilde).min()

    if lam_min >= -tol:
        return np.inf

    return -1.0 / lam_min

def cor2_margin_profile(system: System, theta: np.ndarray, V: np.ndarray,
                        alpha: float, h: np.ndarray, data: dict) -> float:
    """
    Corollary 2 margin at the gain profile k_i^q = alpha h_i.

    Returns the smallest eigenvalue of the edge-space matrix
    Psi = W^{-1} - F^T R^{-1} F, with R_ii = 1/k_i^q - 2 B_ii (V_i)^2.
    A positive value certifies stability.
    Returns -inf when some R_ii <= 0 (R > 0 is required for the
    Schur-complement argument in the corollary).

    `data` is the dict from build_A_matrices, which supplies F and the edge
    weights W.
    """

    BB = np.asarray(system.BB_matrix, dtype=float)
    V = np.asarray(V, dtype=float)

    F = data["F"]
    W_diag = data["W_diag"]

    kq = alpha * h
    R_diag = (1.0 / kq) - 2.0 * np.diag(BB) * V**2

    if np.any(R_diag <= 0):
        return -np.inf

    Psi = np.diag(1.0 / W_diag) - F.T @ ((1.0 / R_diag)[:, None] * F)

    return np.linalg.eigvalsh(Psi).min()


def cor3_critical_alpha_profile(system: System, theta: np.ndarray, V: np.ndarray, h: np.ndarray) -> float:
    """
    Critical alpha from the nodal certificate of Corollary 3.

    The bound is 1 / max_i h_i gamma_i with
        gamma_i = 2 V_i sum_j B_ij V_j / cos(theta_i - theta_j),
    so the certificate is set by the single worst node.
    Returns NaN if cos <= 0 on any edge.
    """
    BB = np.asarray(system.BB_matrix, dtype=float)
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(V, dtype=float)

    edges = edges_from_BB(BB)

    rhs = np.diag(BB) * V

    for i, j in edges:
        c = np.cos(theta[i] - theta[j])
        if c <= 0:
            return np.nan

        rhs[i] += BB[i, j] * V[j] / c
        rhs[j] += BB[j, i] * V[i] / c

    gamma_nodes = 2.0 * V * rhs

    # If k_i^q = alpha * h_i, then beta_i^q = alpha * h_i * V_i,
    # so the constraint is 1/alpha > h_i * gamma_i.
    profile_gamma = h * gamma_nodes
    profile_gamma_max = np.max(profile_gamma)

    if profile_gamma_max <= 0:
        return np.inf

    return 1.0 / profile_gamma_max


def evaluate_stability_at_operating_point(system: System, metadata: dict, theta: np.ndarray, V: np.ndarray, 
                                          info: dict, gain_profile_builder, p_f: float | None = None) -> dict:
    """
    Evaluate Theorem 2, Corollaries 1-4 and cycle diagnostics
    for a PQ/PV System at a solved operating point.

    The returned quantities k_exact, k_cor1, ... are critical values of alpha.
    """
    row = {
        "p_f": p_f,
        "converged": bool(info.get("converged", False)),
        "theta": theta,
        "V": V,
    }

    if not row["converged"]:
        row.update(empty_stability_metrics())
        return row

    data = build_A_matrices(system, theta, V)

    if data is None:
        row.update(empty_stability_metrics())
        row["all_cos_positive"] = False
        return row

    row["all_cos_positive"] = True

    h = prepare_gain_profile(gain_profile_builder(system, theta, V, metadata), n_nodes=system.n_nodes)

    row["gain_profile"] = h

    A_exact = data["A_exact"]
    A_cor1 = data["A_cor1"]
    cycle = data["cycle"]

    k_exact = critical_alpha_from_A(A_exact, h)
    k_cor1 = critical_alpha_from_A(A_cor1, h)

    if np.isfinite(k_exact) and np.isfinite(k_cor1) and k_cor1 > 0:
        k_ratio = k_exact / k_cor1
    else:
        k_ratio = np.nan

    k_cor2 = bisection_critical_alpha(lambda alpha: cor2_margin_profile(system=system, theta=theta, V=V,
                                                                        alpha=alpha, h=h, data=data),
                                                                        alpha_start=1.0)

    k_cor3 = cor3_critical_alpha_profile(system, theta, V, h)

    k_cor4 = bisection_critical_alpha(lambda alpha: cor4_margin_from_edges(system=system, theta=theta, V=V,
                                                                           alpha=alpha, h=h, metadata=metadata),
                                                                           alpha_start=1.0)
    # The critical direction for the generalized scan is the minimum eigenvector of
    # diag(sqrt(h)) A_cor1 diag(sqrt(h)).
    sqrt_h = np.sqrt(h)
    A_cor1_tilde = (sqrt_h[:, None] * A_cor1) * sqrt_h[None, :]

    evals_tilde, evecs_tilde = np.linalg.eigh(A_cor1_tilde)

    ycrit = evecs_tilde[:, 0]
    vcrit = sqrt_h * ycrit
    vcrit = vcrit / np.linalg.norm(vcrit)

    cycle_eigs = np.linalg.eigvalsh(cycle)
    cycle_on_critical_mode = float(vcrit.T @ cycle @ vcrit)

    max_angle = max(abs(theta[i] - theta[j]) for i, j in data["edges"])

    row.update({
        "k_exact": k_exact,
        "k_cor1": k_cor1,
        "k_cor2": k_cor2,
        "k_cor3": k_cor3,
        "k_cor4": k_cor4,
        "k_ratio": k_ratio,
        "cycle_on_critical_mode": cycle_on_critical_mode,
        "max_angle": max_angle,
        "cycle_eigs": cycle_eigs,
    })

    return row

def scan_stability_case(case_builder, p_f_values, gain_profile_builder, tol: float = 1e-10, max_iter: int = 80,
                        use_continuation: bool = True) -> pd.DataFrame:
    """
    Evaluate all certificates along a set of p_f_values, with one DataFrame row for each p_f.

    With use_continuation=True each solver is initialized from the previous
    converged operating point (this keeps the branch continuous as
    p_f approaches the point where the load flow stops converging).
    """
    rows = []
    theta0 = None
    V0 = None

    for p_f in p_f_values:
        system, metadata = case_builder(p_f)

        if not use_continuation:
            theta0 = None
            V0 = None

        theta, V, info = solve_operating_point_system(system, theta0=theta0, V0=V0, tol=tol,
                                                      max_iter=max_iter, metadata=metadata)

        row = evaluate_stability_at_operating_point(system=system, theta=theta, V=V, info=info, p_f=p_f,
                                                    metadata=metadata, gain_profile_builder=gain_profile_builder)

        rows.append(row)

        if row["converged"] and use_continuation:
            theta0 = theta
            V0 = V
        else:
            theta0 = None
            V0 = None

    return pd.DataFrame(rows)

def pv_slack_embedded_gain_profile(system: System, theta: np.ndarray, V: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Droop-gain profile h for the embedded-inverter protocol.

    Grid-forming inverters at the slack and PV buses have full gain,
    h_i = 1, while the PQ buses carry weaker embedded participation,
    h_i = embedded_share. With k_i^q = alpha h_i and h un-normalized, the
    critical alpha returned by the scan is the scalar droop level
    bar{k}^q from Fig. 4.
    """
    h = np.full(system.n_nodes, float(metadata["embedded_share"]), dtype=float)
    h[np.asarray(metadata["large_inverter_buses"], dtype=int)] = 1.0
    return h


### Check Cor 3 and 4 behaviors

def _bus_kind(i: int, metadata: dict) -> str:
    """Classify bus i as 'slack', 'PV', or 'PQ' from metadata."""
    if int(i) == int(metadata["slack_bus"]):
        return "slack"
    if int(i) in set(np.asarray(metadata["pv_buses"], dtype=int).tolist()):
        return "PV"
    return "PQ"

def cor3_binding_table(system: System, theta: np.ndarray, V: np.ndarray, h: np.ndarray,
                       metadata: dict) -> tuple[float, int, pd.DataFrame]:
    """
    Corollary 3 node diagnostic for k_i^q = alpha h_i.

    Corollary 3 requires
    1/(k_i^q V_i) > 2 sum_j B_ij V_j / cos(theta_i - theta_j) at every node.
    With k_i^q = alpha h_i the certified bound is alpha < 1 / max_i h_i gamma_i
    and a single node sets it.

    Returns (alpha_cor3, binding_node, node_df), where node_df has one row per
    node sorted by h_i gamma_i, worst first, so that node_df.iloc[0] is the
    limiting node.
    Raises ValueError if cos(theta_i - theta_j) <= 0 on any edge, since it is
    outside the regime where the corollary applies.
    """

    BB = np.asarray(system.BB_matrix, dtype=float)
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(V, dtype=float)
    h = prepare_gain_profile(h, system.n_nodes)

    edges = edges_from_BB(BB)

    rhs = np.diag(BB) * V

    for i, j in edges:
        c = np.cos(theta[i] - theta[j])
        if c <= 0:
            raise ValueError(f"Nonpositive cosine on edge {(i, j)}: cos={c}")

        rhs[i] += BB[i, j] * V[j] / c
        rhs[j] += BB[j, i] * V[i] / c

    gamma_nodes = 2.0 * V * rhs
    profile_gamma = h * gamma_nodes

    rows = []
    for i in range(system.n_nodes):
        alpha_bound_i = np.inf if profile_gamma[i] <= 0 else 1.0 / profile_gamma[i]
        rows.append({
            "node": int(i),
            "kind": _bus_kind(i, metadata),
            "h": float(h[i]),
            "V": float(V[i]),
            "gamma_i": float(gamma_nodes[i]),
            "profile_gamma_i": float(profile_gamma[i]),
            "alpha_bound_i": float(alpha_bound_i),
        })

    node_df = pd.DataFrame(rows).sort_values("profile_gamma_i", ascending=False).reset_index(drop=True)

    alpha_cor3 = np.inf if node_df.loc[0, "profile_gamma_i"] <= 0 else 1.0 / node_df.loc[0, "profile_gamma_i"]
    binding_node = int(node_df.loc[0, "node"])

    return alpha_cor3, binding_node, node_df


def cor4_edge_table_at_alpha(system: System, theta: np.ndarray, V: np.ndarray, h: np.ndarray,
                             alpha: float, metadata: dict) -> pd.DataFrame:
    """
    Corollary 4 edge diagnostic at a fixed alpha.

    Returns one row per edge with the minimum eigenvalue of the local
    2x2 edge matrix. The binding edge is the row with smallest lam_min.
    """

    BB = np.asarray(system.BB_matrix, dtype=float)
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(V, dtype=float)
    h = prepare_gain_profile(h, system.n_nodes)

    edges = edges_from_BB(BB)
    kq = alpha * h

    Bsh = BB.sum(axis=1)
    S = BB.sum(axis=1) - np.diag(BB)

    rows = []

    for _, (i, j) in enumerate(edges):
        c = np.cos(theta[i] - theta[j])
        if c <= 0:
            lam_min = -np.inf
            H11 = H22 = H12 = np.nan
        elif S[i] <= 0 or S[j] <= 0:
            lam_min = -np.inf
            H11 = H22 = H12 = np.nan
        else:
            beta_i = kq[i] * V[i]
            beta_j = kq[j] * V[j]

            H11 = 2.0 * V[i] - V[j] / c + ((1.0 / beta_i) - 2.0 * Bsh[i] * V[i]) / S[i]
            H22 = 2.0 * V[j] - V[i] / c + ((1.0 / beta_j) - 2.0 * Bsh[j] * V[j]) / S[j]
            H12 = -np.sqrt(V[i] * V[j]) / c

            H = np.array([[H11, H12], [H12, H22]])
            lam_min = float(np.linalg.eigvalsh(H).min())

        rows.append({
            "edge": (int(i), int(j)),
            "lam_min": float(lam_min),
        })

    edge_df = pd.DataFrame(rows).sort_values("lam_min", ascending=True).reset_index(drop=True)
    return edge_df

def cor4_margin_from_edges(system: System, theta: np.ndarray, V: np.ndarray,
                           alpha: float, h: np.ndarray, metadata: dict) -> float:
    """Smallest 2x2 edge eigenvalue of Corollary 4.
    Positivity certifies stability."""
    edge_df = cor4_edge_table_at_alpha(system, theta, V, h, alpha, metadata=metadata)
    return float(edge_df["lam_min"].min())

def bisection_critical_alpha(margin_function, alpha_min: float = 1e-8,
                             alpha_start: float = 1.0, alpha_max: float = 1e6,
                             n_iter: int = 80) -> float:
    """
    Largest alpha with margin_function(alpha) > 0, by bracketing and bisection.

    Assumes the margin decreases monotonically in alpha. This holds because
    alpha enters only through R_ii = 1/(alpha h_i) - 2 B_ii V_i^2.
    Returns NaN if even alpha_min fails and inf if alpha_max still passes.
    """
    if margin_function(alpha_min) <= 0:
        return np.nan

    alpha_hi = alpha_start
    while margin_function(alpha_hi) > 0 and alpha_hi < alpha_max:
        alpha_hi *= 2.0

    if alpha_hi >= alpha_max and margin_function(alpha_hi) > 0:
        return np.inf

    alpha_lo = alpha_min

    for _ in range(n_iter):
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        if margin_function(alpha_mid) > 0:
            alpha_lo = alpha_mid
        else:
            alpha_hi = alpha_mid

    return alpha_lo

def local_binding_diagnostics(system: System, theta: np.ndarray, V: np.ndarray,
                              h: np.ndarray, metadata: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Compare Corollaries 3 and 4 at one operating point.

    Returns (summary, node_df, edge_df). This is a scalar summary naming which
    certificate binds and which node/edge limits it, plus the full per-node
    and per-edge tables sorted worst-first.
    """
    alpha_cor3, binding_node, node_df = cor3_binding_table(system, theta, V, h, metadata=metadata)

    alpha_cor4 = bisection_critical_alpha(lambda a: cor4_margin_from_edges(system, theta, V, a, h, metadata=metadata), alpha_start=1.0)

    edge_df = cor4_edge_table_at_alpha(system, theta, V, h, alpha_cor4, metadata=metadata)

    binding_edge_row = edge_df.iloc[0]

    summary = {
        "alpha_cor3": float(alpha_cor3),
        "alpha_cor4": float(alpha_cor4),
        "cor4_over_cor3": float(alpha_cor4 / alpha_cor3) if np.isfinite(alpha_cor3) and alpha_cor3 > 0 else np.nan,
        "more_restrictive": "Cor 3" if alpha_cor3 < alpha_cor4 else "Cor 4",
        "binding_node_cor3": int(binding_node),
        "binding_node_kind": str(node_df.iloc[0]["kind"]),
        "binding_edge_cor4": binding_edge_row["edge"]
    }

    return summary, node_df, edge_df

def binding_switch_table(diag_df: pd.DataFrame, certificate: str = "cor3") -> pd.DataFrame:
    """
    Locate loadings where the limiting element of a local certificate changes.

    Corollary 3 is limited by the worst node (argmax_i h_i gamma_i) and
    Corollary 4 by the worst edge (argmin over edges of lam_min).
    This function tracks one of the two, selected by `certificate`.

    For every switch between consecutive converged samples p_f[k] -> p_f[k+1]
    it reports the limiting element and the certified bound alpha on both
    sides, together with the change in the local logarithmic slope
    d log(alpha) / d p_f across the switch. The transitional interval itself
    is skipped, so the slopes compared are those of the two smooth branches.
    """
    if certificate == "cor3":
        label_col, alpha_col = "binding_node_cor3", "alpha_cor3"
    elif certificate == "cor4":
        label_col, alpha_col = "binding_edge_cor4", "alpha_cor4"
    else:
        raise ValueError("certificate must be 'cor3' or 'cor4'.")

    df = diag_df[diag_df["converged"]].sort_values("p_f").reset_index(drop=True)

    p_f = df["p_f"].to_numpy(dtype=float)
    alpha = df[alpha_col].to_numpy(dtype=float)
    label = df[label_col].astype(str).to_numpy()

    # Local logarithmic slope on interval k = [p_f[k], p_f[k+1]].
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.diff(np.log(alpha)) / np.diff(p_f)

    rows = []

    # k indexes the interval across which the switch happens.
    # k >= 1 and k <= len-3 so that a slope exists on both intervals.
    for k in range(1, len(df) - 2):
        if label[k] == label[k + 1]:
            continue

        rows.append({
            "certificate": "Cor. 3 (node)" if certificate == "cor3" else "Cor. 4 (edge)",
            "p_f_before": float(p_f[k]),
            "p_f_after": float(p_f[k + 1]),
            "limiting_before": label[k],
            "limiting_after": label[k + 1],
            "alpha_before": float(alpha[k]),
            "alpha_after": float(alpha[k + 1]),
            "slope_before": float(slope[k - 1]),
            "slope_after": float(slope[k + 1]),
            "slope_change": float(slope[k + 1] - slope[k - 1]),
        })

    return pd.DataFrame(rows)

def diagnose_local_binding_scan(case_builder, p_f_values: np.ndarray, gain_profile_builder,
                                use_continuation: bool = True, tol: float = 1e-10,
                                max_iter: int = 80) -> tuple[pd.DataFrame, dict, dict]: 
    """
    Track which node and which edge limit the local certificates along a loading scan.

    At each p_f the operating point is solved and Corollaries 3 and 4 are
    evaluated with local_binding_diagnostics, giving the certified bounds
    alpha_cor3 and alpha_cor4 together with the limiting node and edge.

    Returns (diag_df, node_tables, edge_tables): one summary row per p_f, plus
    the full per-node and per-edge tables keyed by p_f for closer inspection.
    Non-converged points appear in diag_df with converged=False and no summary
    columns.

    use_continuation=True initializes each solver from the previous converged operating point.
    """
    rows = []
    node_tables = {}
    edge_tables = {}

    theta0 = None
    V0 = None

    for p_f in p_f_values:
        system, metadata = case_builder(p_f)

        if not use_continuation:
            theta0 = None
            V0 = None

        theta, V, info = solve_operating_point_system(system, theta0=theta0, V0=V0, tol=tol, max_iter=max_iter, metadata=metadata)

        if not info.get("converged", False):
            rows.append({"p_f": p_f, "converged": False})
            theta0 = None
            V0 = None
            continue

        h = gain_profile_builder(system, theta, V, metadata)
        h = prepare_gain_profile(h, system.n_nodes)

        summary, node_df, edge_df = local_binding_diagnostics(system, theta, V, h, metadata=metadata)

        summary.update({
            "p_f": float(p_f),
            "converged": True,
        })

        rows.append(summary)
        node_tables[float(p_f)] = node_df
        edge_tables[float(p_f)] = edge_df

        if use_continuation:
            theta0 = theta
            V0 = V

    diag_df = pd.DataFrame(rows)

    return diag_df, node_tables, edge_tables