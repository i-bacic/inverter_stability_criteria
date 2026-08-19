#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the stability certificates in 'stability_criteria.py'."""

import numpy as np
import pytest

from inverter_stability_criteria.system_classes import case9_sparse_pv_pq_embedded_builder
from inverter_stability_criteria.stability_criteria import (
    scan_stability_case,
    pv_slack_embedded_gain_profile,
    solve_operating_point_system,
    critical_alpha_from_A,
)
from inverter_stability_criteria.formulate_matrices import build_A_matrices


P_F_VALUES = [1.0, 1.5, 2.0, 2.5]


@pytest.fixture(scope="module")
def scan():
    return scan_stability_case(case_builder=case9_sparse_pv_pq_embedded_builder,
                               p_f_values=P_F_VALUES,
                               gain_profile_builder=pv_slack_embedded_gain_profile)


def test_certificates_are_nested(scan, rtol=1e-9):
    """
    The certificates form a chain of increasingly conservative bounds.

    Corollaries 3 and 4 are decentralizations of Corollary 1, which drops the
    positive semi-definite cycle term from the exact condition of Theorem 2.
    Every critical gain must therefore satisfy
        k_cor3, k_cor4 <= k_cor1 <= k_exact.
    """
    assert scan["converged"].all()

    for _, row in scan.iterrows():
        assert row["k_cor3"] <= row["k_cor1"] * (1 + rtol)
        assert row["k_cor4"] <= row["k_cor1"] * (1 + rtol)
        assert row["k_cor1"] <= row["k_exact"] * (1 + rtol)


def test_cor1_and_cor2_agree(scan, rtol=1e-6):
    """
    Corollaries 1 and 2 certify the same boundary.

    Upsilon and Psi are Schur complements of the same block matrix, so they
    are positive definite together and must give the same critical gain.
    """
    assert np.allclose(scan["k_cor1"], scan["k_cor2"], rtol=rtol)


def test_cycle_contribution_is_psd():
    """
    The cycle correction is positive semi-definite, hence Upsilon_hat >= Upsilon.

    This is what makes Corollary 1 sufficient: dropping Z Lambda^{-1} Z^T can
    only tighten the bound.
    """
    system, metadata = case9_sparse_pv_pq_embedded_builder(2.0)
    theta, V, info = solve_operating_point_system(system, metadata)
    assert info["converged"]

    data = build_A_matrices(system, theta, V)
    assert data is not None

    cycle_eigs = np.linalg.eigvalsh(data["cycle"])
    assert (cycle_eigs > -1e-10).all()

    assert np.allclose(data["A_exact"], data["A_cor1"] + data["cycle"])


def test_critical_alpha_from_A_on_scalar_case():
    """
    critical_alpha_from_A inverts the smallest eigenvalue.

    For A = diag(-2, -1) and h = 1, positive definiteness of diag(1/alpha) + A
    fails first in the direction with eigenvalue -2, so alpha_crit = 1/2.
    """
    A = np.diag([-2.0, -1.0])
    h = np.ones(2)

    assert np.isclose(critical_alpha_from_A(A, h), 0.5)

    # A positive semi-definite matrix cannot be destabilized by any finite gain.
    assert np.isinf(critical_alpha_from_A(np.diag([1.0, 2.0]), h))