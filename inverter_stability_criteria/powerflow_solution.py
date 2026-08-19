#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Power flow solvers for lossless networks with purely imaginary Ybus, used for
finding an operating point for the system.
This is required to evaluate the stability criteria at a given operating point.
Conventions used here are consistent with System.BB_matrix = imag(Ybus).

Active injection: P_i = sum_j V_i V_j * B_ij * sin(theta_i - theta_j)
Reactive injection: Q_i = - sum_j V_i V_j * B_ij * cos(theta_i - theta_j)
"""

import numpy as np

from inverter_stability_criteria.system_classes import System

__all__ = [
    "active_power_flow",
    "reactive_power_flow",
    "jacobian",
    "solve_lossless_pv_pq_power_flow",
]

def active_power_flow(BB_matrix: np.ndarray, phase_arr: np.ndarray, volt_arr: np.ndarray) -> np.ndarray:
    """Compute active power injections P(theta, V) for a lossless Ybus with BB = imag(Ybus)."""
    
    BB = np.asarray(BB_matrix, dtype=float)
    theta = np.asarray(phase_arr, dtype=float)
    V = np.asarray(volt_arr, dtype=float)

    N = BB.shape[0]
    
    assert theta.shape == (N,)
    assert V.shape == (N,)

    diff_theta = theta[:, None] - theta[None, :]

    P = (V[:, None] * V[None, :] * BB * np.sin(diff_theta)).sum(axis=1)
    
    return P
    
def reactive_power_flow(BB_matrix: np.ndarray, phase_arr: np.ndarray, volt_arr: np.ndarray) -> np.ndarray:
    """Compute reactive power injections Q with fixed voltages for a lossless Ybus with BB = imag(Ybus)"""
    BB = np.asarray(BB_matrix, dtype=float)
    theta = np.asarray(phase_arr, dtype=float)
    V = np.asarray(volt_arr, dtype=float)

    N = BB.shape[0]
    assert theta.shape == (N,)
    assert V.shape == (N,)

    diff_theta = theta[:, None] - theta[None, :]
    Q = -((V[:, None] * V[None, :]) * BB * np.cos(diff_theta)).sum(axis=1)

    return Q

def jacobian(theta: np.ndarray, V: np.ndarray, BB: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Jacobian of [P; Q] wrt [theta; V].
    Returns full (N x N) blocks: J_Ptheta, J_PV, J_Qtheta, J_QV.
    """
    
    N = BB.shape[0]
    dtheta = theta[:, None] - theta[None, :]
    sin = np.sin(dtheta)
    cos = np.cos(dtheta)

    # Off-diagonal elements
    J_Ptheta = -(V[:, None] * V[None, :] * BB * cos)
    J_Qtheta = -(V[:, None] * V[None, :] * BB * sin)

    J_PV =  (V[:, None] * BB * sin) # dP_i/dV_k for k!=i
    J_QV = -(V[:, None] * BB * cos) # dQ_i/dV_k for k!=i

    # Diagonal elements
    np.fill_diagonal(J_Ptheta, 0.0)
    np.fill_diagonal(J_Qtheta, 0.0)
    np.fill_diagonal(J_PV, 0.0)
    np.fill_diagonal(J_QV, 0.0)

    J_Ptheta[np.arange(N), np.arange(N)] = -J_Ptheta.sum(axis=1)
    J_Qtheta[np.arange(N), np.arange(N)] = -J_Qtheta.sum(axis=1)

    P = (V[:, None] * V[None, :] * BB * sin).sum(axis=1)
    Q = -(V[:, None] * V[None, :] * BB * cos).sum(axis=1)

    J_PV[np.arange(N), np.arange(N)] = P / V # dP_i/dV_i
    J_QV[np.arange(N), np.arange(N)] = Q / V - np.diag(BB) * V # dQ_i/dV_i

    return J_Ptheta, J_PV, J_Qtheta, J_QV

def solve_lossless_pv_pq_power_flow(system: System, theta0: np.ndarray, V0: np.ndarray, slack_bus: int,
                                    pv_buses: np.ndarray, pq_buses: np.ndarray, V_set: np.ndarray | None = None,
                                    tol: float = 1e-10, max_iter: int = 80) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Solve a standard lossless slack/PV/PQ power flow.

    Model:
        P_i(theta, V) = Psp_i       for all non-slack buses
        Q_i(theta, V) = Qsp_i       for PQ buses only

    Fixed:
        theta_slack = 0
        V_slack fixed
        V_i fixed for PV buses

    Unknowns:
        theta_i for all non-slack buses
        V_i for PQ buses

    Notes:
        - Slack active power is not enforced.
        - PV reactive power is not enforced.
        - Qsp values at slack/PV buses may be stored in system but are ignored.
    """
    BB = np.asarray(system.BB_matrix, dtype=float)
    Psp = np.asarray(system.power_setpoints_active, dtype=float)
    Qsp = np.asarray(system.power_setpoints_reactive, dtype=float)

    N = BB.shape[0]

    theta = np.asarray(theta0, dtype=float).copy()
    V = np.asarray(V0, dtype=float).copy()

    if V_set is None:
        V_set = np.asarray(system.volt_operating_point, dtype=float)
    else:
        V_set = np.asarray(V_set, dtype=float)

    pv_buses = np.asarray(pv_buses, dtype=int)
    pq_buses = np.asarray(pq_buses, dtype=int)

    if theta.shape != (N,):
        raise ValueError(f"theta0 has shape {theta.shape}, expected {(N,)}.")

    if V.shape != (N,):
        raise ValueError(f"V0 has shape {V.shape}, expected {(N,)}.")

    if V_set.shape != (N,):
        raise ValueError(f"V_set has shape {V_set.shape}, expected {(N,)}.")

    all_buses = np.arange(N, dtype=int)

    non_slack = all_buses[all_buses != slack_bus]

    fixed_voltage_buses = np.unique(np.concatenate(([slack_bus], pv_buses))).astype(int)

    # Enforce reference angle and fixed voltage magnitudes.
    theta[slack_bus] = 0.0
    V[fixed_voltage_buses] = V_set[fixed_voltage_buses]

    n_theta = len(non_slack)
    n_pq = len(pq_buses)

    info = {
        "converged": False,
        "iterations": 0,
        "final_residual_inf": np.nan,
        "slack_bus": int(slack_bus),
        "n_pv": int(len(pv_buses)),
        "n_pq": int(len(pq_buses)),
        "bad_voltage": False,
    }

    for it in range(1, max_iter + 1):
        P = active_power_flow(BB, theta, V)
        Q = reactive_power_flow(BB, theta, V)

        fP = (P - Psp)[non_slack]
        fQ = (Q - Qsp)[pq_buses]
        f = np.concatenate([fP, fQ])

        res_inf = float(np.linalg.norm(f, ord=np.inf))
        info["iterations"] = it
        info["final_residual_inf"] = res_inf

        if res_inf < tol:
            info["converged"] = True
            return theta, V, info

        J_Pth, J_PV, J_Qth, J_QV = jacobian(theta, V, BB)

        J11 = J_Pth[non_slack][:, non_slack]
        J12 = J_PV[non_slack][:, pq_buses]
        J21 = J_Qth[pq_buses][:, non_slack]
        J22 = J_QV[pq_buses][:, pq_buses]

        if n_pq == 0:
            J = J11
        else:
            J = np.block([
                [J11, J12],
                [J21, J22],
            ])

        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            info["converged"] = False
            info["singular_jacobian"] = True
            return theta, V, info

        theta[non_slack] += dx[:n_theta]

        if n_pq > 0:
            V[pq_buses] += dx[n_theta:]

        theta[slack_bus] = 0.0
        V[fixed_voltage_buses] = V_set[fixed_voltage_buses]

        if np.any(~np.isfinite(V)) or np.any(V <= 0):
            info["converged"] = False
            info["bad_voltage"] = True
            return theta, V, info

    return theta, V, info
