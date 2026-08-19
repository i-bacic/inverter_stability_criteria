#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""These functions setup the matrices needed to describe the power system.
Input are the system classes defined in system_classes.py.

Edge convention:
  For each undirected edge {i,j} with BB[i,j] != 0, there is one directed edge
  from the smaller index to the larger index: (u=min(i,j), v=max(i,j)).
"""

import numpy as np
import networkx as nx

from inverter_stability_criteria.system_classes import System

__all__ = ["edges_from_BB",
           "edge_cycle_incidence_matrix", 
           "node_edge_unsigned_incidence_matrix",
           "edge_quantities",
           "build_A_matrices"]

def edges_from_BB(BB: np.ndarray) -> list[tuple[int, int]]:
    """Return the edge list of BB, each edge (u, v) oriented u->v with u < v."""
    N = BB.shape[0]
    edges = []

    # Triangular matrix, we always know from(edge) and to(edge) this way
    for u in range(N):
        for v in range(u + 1, N):
            if abs(BB[u, v]) > 1e-12:
                edges.append((u, v))  # oriented u -> v when u < v
                
    return edges

def edge_cycle_incidence_matrix(system: System) -> np.ndarray:
    """
    Cycle-edge incidence matrix C in R^{m x c} where columns correspond to a fundamental cycle basis.
    Edges are assumed oriented as (u, v) with u < v.
    Sign convention: +1 if cycle traverses edge in stored orientation, -1 otherwise.
    """
    BB = np.asarray(system.BB_matrix, dtype=float)
    N = BB.shape[0]
    edges = edges_from_BB(BB)
    M = len(edges)
    edge_index = {e: k for k, e in enumerate(edges)}

    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edges)

    cycles = nx.cycle_basis(G)  # list of node lists
    c = len(cycles)
    CC_matrix = np.zeros((M, c), dtype=float)

    for col, cyc_nodes in enumerate(cycles):
        # Close the loop
        cyc = cyc_nodes + [cyc_nodes[0]]
        for i in range(len(cyc) - 1):
            a = cyc[i]
            b = cyc[i+1]
            u, v = (a, b) if a < b else (b, a)
            k = edge_index[(u, v)]
            # Traversal direction relative to orientation u -> v
            sign = +1.0 if (a == u and b == v) else -1.0
            CC_matrix[k, col] = sign

    return CC_matrix

def node_edge_unsigned_incidence_matrix(system: System) -> np.ndarray:
    """Unsigned node-edge incidence F in R^{N x M}."""
    
    BB = np.asarray(system.BB_matrix, dtype=float)
    N = BB.shape[0]
    edges = edges_from_BB(BB)
    M = len(edges)

    FF_matrix = np.zeros((N, M), dtype=float)
    for a, (u, v) in enumerate(edges):
        FF_matrix[u, a] = +1.0
        FF_matrix[v, a] = +1.0
    return FF_matrix

def edge_quantities(system: System, theta: np.ndarray, V: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonal edge quantities at the operating point (theta, V).

    Returns (edges, P, Q, W, cos), where
    - P_a and Q_a are the diagonal entries of the edge-space matrices P and Q
    - W_a = Q_a + P_a^2 / Q_a, which reduces to W_a = B_ij V_i V_j / cos(theta_i - theta_j) since the matrices are diagonal.
    - cos_a = cos(theta_i - theta_j)
    """
    BB = np.asarray(system.BB_matrix, dtype=float)
    theta = np.asarray(theta, dtype=float)
    V = np.asarray(V, dtype=float)

    edges = edges_from_BB(BB)
    M = len(edges)

    P_diag = np.zeros(M)
    Q_diag = np.zeros(M)
    W_diag = np.zeros(M)
    cos_diag = np.zeros(M)

    for a, (i, j) in enumerate(edges):
        dtheta = theta[i] - theta[j]
        c = np.cos(dtheta)

        cos_diag[a] = c
        P_diag[a] = BB[i, j] * V[i] * V[j] * np.sin(dtheta)
        Q_diag[a] = BB[i, j] * V[i] * V[j] * c
        W_diag[a] = Q_diag[a] + (P_diag[a] * P_diag[a]) / Q_diag[a]

    return edges, P_diag, Q_diag, W_diag, cos_diag

def build_A_matrices(system: System, theta: np.ndarray, V: np.ndarray) -> dict | None:
    """
    Assemble the gain-independent parts of the stability matrices.

    Writing R_ii = 1/k_i^q - 2 B_ii (V_i)^2, 
    - Upsilon = diag(1/k^q) + A_cor1,
    - Upsilon_hat = diag(1/k^q) + A_exact,
    with A_cor1 = D - F W F^T, D = diag(-2 B_ii V_i^2), and A_exact = A_cor1 + cycle,
    where cycle is the positive semi-definite cycle correction Z Lambda^{-1} Z^T.
    Separating out diag(1/k^q) lets us obtain the critical droop level from a single eigenvalue.

    Returns None if cos(theta_i - theta_j) <= 0 on any edge, i.e. outside the
    regime where the certificates apply.
    """
    BB = np.asarray(system.BB_matrix, dtype=float)
    V = np.asarray(V, dtype=float)
    N = system.n_nodes

    F = node_edge_unsigned_incidence_matrix(system)
    C = edge_cycle_incidence_matrix(system)

    edges, P_diag, Q_diag, W_diag, cos_diag = edge_quantities(system, theta, V)

    if np.any(cos_diag <= 0):
        return None

    D = np.diag(-2.0 * np.diag(BB) * V**2)

    A_cor1 = D - (F * W_diag) @ F.T

    if C.shape[1] == 0:
        cycle = np.zeros((N, N))
    else:
        A = ((P_diag / Q_diag)[:, None]) * C
        Gamma = C.T @ ((1.0 / Q_diag)[:, None] * C)
        cycle = F @ A @ np.linalg.solve(Gamma, A.T @ F.T)

    A_exact = A_cor1 + cycle

    return {
        "A_exact": A_exact,
        "A_cor1": A_cor1,
        "cycle": cycle,
        "F": F,
        "C": C,
        "edges": edges,
        "P_diag": P_diag,
        "Q_diag": Q_diag,
        "W_diag": W_diag,
        "cos_diag": cos_diag,
    }
