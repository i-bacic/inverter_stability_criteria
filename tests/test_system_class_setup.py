#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the System class and the case builders in 'system_classes.py'."""

import numpy as np
import pytest

from inverter_stability_criteria.system_classes import (
    System,
    case9_sparse_pv_pq_embedded_builder,
    case30_sparse_pv_pq_embedded_builder,
    case118_sparse_pv_pq_embedded_builder,
)

BUILDERS = [
    case9_sparse_pv_pq_embedded_builder,
    case30_sparse_pv_pq_embedded_builder,
    case118_sparse_pv_pq_embedded_builder,
]

REQUIRED_METADATA = ["slack_bus", "pv_buses", "pq_buses", "large_inverter_buses",
                     "embedded_share", "theta_base", "V_base", "V_set"]


@pytest.mark.parametrize("builder", BUILDERS)
def test_builder_constructs_valid_system(builder):
    """
    Each case builder returns a System satisfying the paper's conventions and
    metadata complete enough for the operating-point solver.
    """
    system, metadata = builder(1.0)

    assert isinstance(system, System)

    n = system.n_nodes
    BB = system.BB_matrix

    # Lossless, shuntless, symmetric Laplacian with nonnegative couplings.
    assert np.allclose(system.GG_matrix, 0.0)
    assert np.allclose(BB, BB.T)
    assert np.allclose(BB.sum(axis=1), 0.0, atol=1e-8)

    offdiag = BB - np.diag(np.diag(BB))
    assert (offdiag >= -1e-12).all(), "Off-diagonal susceptances must be nonnegative."
    assert (np.diag(BB) < 0).all(), "B_ii must be negative."

    for key in REQUIRED_METADATA:
        assert key in metadata, f"metadata is missing '{key}'."

    # The three bus sets partition the buses, in Ybus row order.
    slack = int(metadata["slack_bus"])
    pv = np.asarray(metadata["pv_buses"], dtype=int)
    pq = np.asarray(metadata["pq_buses"], dtype=int)

    all_buses = np.sort(np.concatenate(([slack], pv, pq)))
    assert np.array_equal(all_buses, np.arange(n)), \
        "slack/PV/PQ buses must partition the buses."

    for key in ["theta_base", "V_base", "V_set"]:
        assert np.asarray(metadata[key]).shape == (n,)

    assert abs(metadata["theta_base"][slack]) < 1e-12


@pytest.mark.parametrize("builder", BUILDERS)
def test_active_power_scales_with_loading(builder):
    """
    The loading factor scales the active injections at the non-slack buses.

    The slack entry absorbs the rebalancing and is not enforced by the solver,
    so it is excluded.
    """
    p_f = 2.0
    system_1, metadata = builder(1.0)
    system_2, _ = builder(p_f)

    non_slack = np.setdiff1d(np.arange(system_1.n_nodes), [int(metadata["slack_bus"])])

    assert np.allclose(system_2.power_setpoints_active[non_slack], p_f * system_1.power_setpoints_active[non_slack])


def test_system_rejects_lossy_admittance():
    """System refuses a matrix with nonzero conductance."""
    Y = np.array([[-1.0 + 1.0j, 1.0 - 1.0j],
                  [1.0 - 1.0j, -1.0 + 1.0j]])

    with pytest.raises(AssertionError):
        System(admittance_matrix=Y,
               volt_operating_point=np.ones(2),
               power_setpoints_active=np.array([0.5, -0.5]),
               power_setpoints_reactive=np.zeros(2))


def test_system_rejects_unbalanced_active_power():
    """System refuses active setpoints that do not sum to zero."""
    Y = 1j * np.array([[-1.0, 1.0], [1.0, -1.0]])

    with pytest.raises(AssertionError):
        System(admittance_matrix=Y,
               volt_operating_point=np.ones(2),
               power_setpoints_active=np.array([0.5, 0.5]),
               power_setpoints_reactive=np.zeros(2))