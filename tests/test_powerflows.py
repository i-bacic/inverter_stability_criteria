#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the power flow solver in 'powerflow_solution.py'."""

import numpy as np

from inverter_stability_criteria.system_classes import case9_sparse_pv_pq_embedded_builder
from inverter_stability_criteria.powerflow_solution import active_power_flow, reactive_power_flow
from inverter_stability_criteria.stability_criteria import solve_operating_point_system

def test_pv_pq_power_flow_residual(p_f: float = 1.5, atol: float = 1e-8):
    """
    The solved operating point satisfies the equations the solver enforces.

    The lossless slack/PV/PQ model enforces P at every non-slack bus and Q at
    the PQ buses only. Slack active power and PV reactive power are free, and
    are therefore excluded from the check. Voltage magnitudes at the slack and
    PV buses must equal their setpoints, and the slack angle must be zero.
    """
    system, metadata = case9_sparse_pv_pq_embedded_builder(p_f)

    theta, V, info = solve_operating_point_system(system, metadata)

    assert info["converged"], f"Power flow did not converge: {info}"

    slack = int(metadata["slack_bus"])
    pv_buses = np.asarray(metadata["pv_buses"], dtype=int)
    pq_buses = np.asarray(metadata["pq_buses"], dtype=int)

    non_slack = np.setdiff1d(np.arange(system.n_nodes), [slack])

    P = active_power_flow(system.BB_matrix, theta, V)
    Q = reactive_power_flow(system.BB_matrix, theta, V)

    assert np.allclose(P[non_slack], system.power_setpoints_active[non_slack], atol=atol), \
        "Active power mismatch at the non-slack buses."

    assert np.allclose(Q[pq_buses], system.power_setpoints_reactive[pq_buses], atol=atol), \
        "Reactive power mismatch at the PQ buses."

    fixed = np.concatenate(([slack], pv_buses))
    assert np.allclose(V[fixed], np.asarray(metadata["V_set"])[fixed], atol=atol), \
        "Voltage magnitude not held at the slack and PV buses."

    assert abs(theta[slack]) < atol, "Slack angle is not zero."