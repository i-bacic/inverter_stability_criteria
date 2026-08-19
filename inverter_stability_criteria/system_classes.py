#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Here, the system classes are defined."""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

from dataclasses import dataclass, field

@dataclass(frozen=True)
class System:
    """This class describes the power system.
    Lossless, shuntless power system: susceptance matrix, operating-point voltages, and power setpoints.
    """
    
    admittance_matrix: np.ndarray
    volt_operating_point: np.ndarray
    power_setpoints_active: np.ndarray
    power_setpoints_reactive: np.ndarray
    
    # Members defined in __post_init__
    n_nodes: int = field(init=False)
    BB_matrix: np.ndarray = field(init=False)
    GG_matrix: np.ndarray = field(init=False)
        
    def __post_init__(self):
        """Validate parameters after dataclass initialization and set the number of nodes."""
        
        # Types
        all_np_array = all([isinstance(xx, np.ndarray) for xx in [
            self.admittance_matrix,
            self.volt_operating_point,
            self.power_setpoints_active,
            self.power_setpoints_reactive
        ]])
        
        assert all_np_array, "All attributes must be of type 'np.ndarray'."
    
        assert self.admittance_matrix.ndim == 2, "Admittance matrix must be 2D."
        assert self.admittance_matrix.shape[0] == self.admittance_matrix.shape[1], "Admittance matrix must be square."
            
        assert (abs(np.sum(self.admittance_matrix, axis=0)) < 1e-8).all(), "Admittance matrix must be balanced (sum of columns must be zero)."
        
        n = self.admittance_matrix.shape[0]
        assert (self.volt_operating_point.shape == (n,) and self.power_setpoints_active.shape == (n,)
                and self.power_setpoints_reactive.shape == (n,)), "Voltages and power setpoints must have length n."
        
        object.__setattr__(self, 'n_nodes', self.admittance_matrix.shape[0])
        object.__setattr__(self, 'BB_matrix', np.imag(self.admittance_matrix))
        object.__setattr__(self, 'GG_matrix', np.real(self.admittance_matrix))
        
        assert (self.GG_matrix == 0).all(), "The conductance matrix must be zero (lossless system)."
        
        assert abs(sum(self.power_setpoints_active)) < 1e-8, "The sum of active power setpoints must be zero."
        
# Functions for studying sparse grids

def _vector_from_res_bus_in_ppc_order(net, values_by_bus: pd.Series) -> np.ndarray:
    """
    Convert a pandapower bus-indexed Series/array into the internal ppc/Ybus order.

    This matters because Ybus is stored in pandapower's internal ppc bus ordering,
    while net.res_bus is indexed by pandapower bus indices.
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    n_ppc = int(net["_ppc"]["bus"].shape[0])

    out = np.zeros(n_ppc, dtype=float)

    for bus_idx, value in values_by_bus.items():
        ppc_idx = int(bus_lookup[int(bus_idx)])
        if ppc_idx >= 0:
            out[ppc_idx] = float(value)

    return out

def BB_shuntless_laplacian_from_pandapower_ybus(Ybus, tol: float = 1e-12) -> np.ndarray:
    """
    Extract BB = imag(Ybus), discard shunts, and reconstruct a Laplacian diagonal.

    This gives the sparse, shunt-free network:
        B_ij = imag(Ybus_ij), i != j
        B_ii = - sum_{j != i} B_ij

    The resulting matrix has row sums zero.
    """
    Y = Ybus.toarray() if hasattr(Ybus, "toarray") else np.asarray(Ybus)
    BB = np.imag(Y).astype(float)

    # Keep only off-diagonal series couplings.
    np.fill_diagonal(BB, 0.0)

    BB[np.abs(BB) < tol] = 0.0

    # With the convention used in the paper,
    # off-diagonal inductive couplings should be nonnegative.
    offdiag = BB.copy()
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -tol):
        min_val = offdiag.min()
        raise ValueError("Found negative off-diagonal susceptance entries after extraction. "
            f"Minimum off-diagonal value: {min_val:.3e}."
        )

    # Reconstruct shunt-free Laplacian diagonal.
    np.fill_diagonal(BB, -BB.sum(axis=1))

    return BB

## PV/PQ case builder

def _bus_type_sets_from_ppc(net) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Return (slack_bus, pv_buses, pq_buses), indexed by Ybus row rather than
    pandapower bus number.
    MATPOWER/PPC bus types: 1 = PQ, 2 = PV, 3 = REF/slack.
    """
    bus_types = net["_ppc"]["bus"][:, 1].astype(int)

    slack_buses = np.where(bus_types == 3)[0].astype(int)
    pv_buses = np.where(bus_types == 2)[0].astype(int)
    pq_buses = np.where(bus_types == 1)[0].astype(int)

    slack_bus = int(slack_buses[0])

    return slack_bus, pv_buses, pq_buses

def pandapower_sparse_shuntless_pv_pq_embedded_inverter_system(case_factory, pwr_factor: float = 1.0,
                                                               embedded_share: float = 0.1) -> tuple[System, dict]:
    """
    Sparse, shuntless, lossless slack/PV/PQ benchmark with embedded droop.
    P0 and Q0 are defined from the shuntless lossless model evaluated at p_f=1.
    Protocol:
        - keep the original sparse topology,
        - discard conductance and shunts,
        - keep slack/PV voltage magnitudes fixed,
        - scale active injections at all buses by pwr_factor,
        - scale reactive injections at PQ buses by pwr_factor,
        - solve P equations at all non-slack buses,
        - solve Q equations at PQ buses only,
        - use h_i = 1 on slack/PV buses,
        - use h_i = embedded_share on PQ buses.

    Since h_i = 1 on the slack and PV buses, the critical alpha reported by
    the scan is the scalar droop level bar{k}^q.

    Returns (system, metadata). Everything in metadata is indexed by Ybus row,
    not by pandapower bus number.
    pandapower indices are mapped through net["_pd2ppc_lookups"]["bus"].
    """
    # Avoid circular import
    from inverter_stability_criteria.powerflow_solution import active_power_flow, reactive_power_flow

    net = case_factory()
    pp.runpp(net, calculate_voltage_angles=True)

    Ybus = net["_ppc"]["internal"]["Ybus"]
    BB = BB_shuntless_laplacian_from_pandapower_ybus(Ybus)

    n = BB.shape[0]

    slack_bus, pv_buses, pq_buses = _bus_type_sets_from_ppc(net)

    theta_base = _vector_from_res_bus_in_ppc_order(net, np.deg2rad(net.res_bus.va_degree))
    theta_base = theta_base - theta_base[slack_bus]

    V_base = _vector_from_res_bus_in_ppc_order(net, net.res_bus.vm_pu)

    P0 = active_power_flow(BB, theta_base, V_base)
    Q0 = reactive_power_flow(BB, theta_base, V_base)

    # Scale active injections at all buses.
    Psp = pwr_factor * P0

    # Enforce exact active-power balance for System construction.
    Psp[slack_bus] -= Psp.sum()

    # Scale Q only at PQ buses.
    Qsp = Q0.copy()
    Qsp[pq_buses] = pwr_factor * Q0[pq_buses]

    system = System(admittance_matrix=1j * BB, volt_operating_point=V_base, power_setpoints_active=Psp,
                    power_setpoints_reactive=Qsp)

    large_inverter_buses = np.unique(np.concatenate(([slack_bus], pv_buses))).astype(int)

    metadata = {
        "slack_bus": slack_bus,
        "pv_buses": pv_buses,
        "pq_buses": pq_buses,
        "large_inverter_buses": large_inverter_buses,
        "embedded_share": float(embedded_share),
        "theta_base": theta_base,
        "V_base": V_base,
        "V_set": V_base.copy()
    }

    return (system, metadata)

def case9_sparse_pv_pq_embedded_builder(p_f: float) -> tuple[System, dict]:
    """IEEE Case 9 at loading factor p_f, for the embedded-droop protocol."""
    return pandapower_sparse_shuntless_pv_pq_embedded_inverter_system(pn.case9, pwr_factor=p_f, embedded_share=0.1)

def case30_sparse_pv_pq_embedded_builder(p_f: float) -> tuple[System, dict]:
    """IEEE Case 30 at loading factor p_f, for the embedded-droop protocol."""
    return pandapower_sparse_shuntless_pv_pq_embedded_inverter_system(pn.case30, pwr_factor=p_f, embedded_share=0.1)

def case118_sparse_pv_pq_embedded_builder(p_f: float) -> tuple[System, dict]:
    """IEEE Case 118 at loading factor p_f, for the embedded-droop protocol."""
    return pandapower_sparse_shuntless_pv_pq_embedded_inverter_system(pn.case118, pwr_factor=p_f, embedded_share=0.1)
