#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostics supporting two claims in Sec. VI-B:
(1) the ordering of the Corollary 3 and Corollary 4 bounds is not universal (Cases 9 and 30),
(2) the limiting node (Cor. 3) or edge (Cor. 4) changes with loading, producing the kinks in
Fig. 4 (Cases 9 and 118).
"""

import numpy as np
import pandas as pd

from inverter_stability_criteria.system_classes import (
    case9_sparse_pv_pq_embedded_builder,
    case30_sparse_pv_pq_embedded_builder,
    case118_sparse_pv_pq_embedded_builder
)

from inverter_stability_criteria.stability_criteria import (
    diagnose_local_binding_scan,
    pv_slack_embedded_gain_profile,
    binding_switch_table
)

from IPython.display import display

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

p_f_case9_sparse_pv_pq = np.linspace(1.0, 3.1, 100)
p_f_case30_sparse_pv_pq = np.linspace(1.0, 7.0, 100)
p_f_case118_sparse_pv_pq = np.linspace(1.0, 4.0, 100)

diag9, nodes9, edges9 = diagnose_local_binding_scan(case9_sparse_pv_pq_embedded_builder,
                                                    p_f_case9_sparse_pv_pq,
                                                    gain_profile_builder=pv_slack_embedded_gain_profile)

diag30, nodes30, edges30 = diagnose_local_binding_scan(case30_sparse_pv_pq_embedded_builder,
                                                       p_f_case30_sparse_pv_pq,
                                                       gain_profile_builder=pv_slack_embedded_gain_profile)

diag118, nodes118, edges118 = diagnose_local_binding_scan(case118_sparse_pv_pq_embedded_builder,
                                                          p_f_case118_sparse_pv_pq,
                                                          gain_profile_builder=pv_slack_embedded_gain_profile)

# (1) Ordering of the two local certificates.
print("=== Ordering of the Corollary 3 and Corollary 4 bounds ===")

for name, diag in [("Case 9", diag9), ("Case 30", diag30), ("Case 118", diag118)]:
    df = diag[diag["converged"]].sort_values("p_f").reset_index(drop=True)
    flipped = df["more_restrictive"].ne(df["more_restrictive"].shift())
    flipped.iloc[0] = False

    print(f"\n{name}: more restrictive at p_f = {df['p_f'].iloc[0]:.3g} is "
          f"{df['more_restrictive'].iloc[0]}")

    if not flipped.any():
        print("  no change of ordering over the scanned range")
    else:
        print(df.loc[flipped, ["p_f", "alpha_cor3", "alpha_cor4",
                               "cor4_over_cor3", "more_restrictive"]].to_string(index=False))


# (2) Switches of the limiting node / edge.
print("\n\n=== Limiting element of each local certificate ===")

for name, diag in [("Case 9", diag9), ("Case 118", diag118)]:
    for cert, what in [("cor3", "Corollary 3, limiting node"),
                       ("cor4", "Corollary 4, limiting edge")]:
        tbl = binding_switch_table(diag, cert)
        print(f"\n{name} - {what}")

        if len(tbl) == 0:
            print("  no switch of the limiting element over the scanned range")
        else:
            print(tbl.sort_values("slope_change", key=np.abs, ascending=False).to_string(index=False))
