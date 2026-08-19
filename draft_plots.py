#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions to plot the results."""

import numpy as np

from inverter_stability_criteria.system_classes import (
    case9_sparse_pv_pq_embedded_builder,
    case30_sparse_pv_pq_embedded_builder,
    case118_sparse_pv_pq_embedded_builder
)

from inverter_stability_criteria.stability_criteria import scan_stability_case, pv_slack_embedded_gain_profile
from inverter_stability_criteria.plotting import plot_Figure3, plot_combined_certificate_cycle_grid

# Figure 3
plot_Figure3(savepath = "Two_identical_inverters.pdf")

# Figure 4
p_f_case9_sparse_pv_pq = np.linspace(1.0, 3.1, 100)
p_f_case30_sparse_pv_pq = np.linspace(1.0, 7.0, 100)
p_f_case118_sparse_pv_pq = np.linspace(1.0, 4.0, 100)

print('doing case 9...')
df9_sparse_pv_pq = scan_stability_case(case_builder=case9_sparse_pv_pq_embedded_builder,
                                       p_f_values=p_f_case9_sparse_pv_pq,
                                       gain_profile_builder=pv_slack_embedded_gain_profile)
print('doing case 30...')
df30_sparse_pv_pq = scan_stability_case(case_builder=case30_sparse_pv_pq_embedded_builder,
                                        p_f_values=p_f_case30_sparse_pv_pq,
                                        gain_profile_builder=pv_slack_embedded_gain_profile)
print('doing case 118...')
df118_sparse_pv_pq = scan_stability_case(case_builder=case118_sparse_pv_pq_embedded_builder,
                                         p_f_values=p_f_case118_sparse_pv_pq,
                                         gain_profile_builder=pv_slack_embedded_gain_profile)

dfs_sparse_pv_pq = [df9_sparse_pv_pq, df30_sparse_pv_pq, df118_sparse_pv_pq]

case_labels_sparse_pv_pq = ["Case 9", "Case 30", "Case 118"]

plot_combined_certificate_cycle_grid(dfs_sparse_pv_pq, case_labels_sparse_pv_pq, savepath = "Testcases_figure.pdf")
