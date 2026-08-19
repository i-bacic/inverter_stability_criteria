# Stability Criteria for Inverter-Based Systems

Code accompanying *A graph theoretic view on small signal stability of inverter-based power grids* (I. Bačić, J. Niehues, P. C. Böttcher, C. Dieball, L. R. Gorjão, A. Benigni, F. Hellmann, D. Witthaut): https://arxiv.org/abs/2607.08260

Given a lossless, shuntless power system at an operating point, the package evaluates the exact small-signal stability condition of Theorem 2 and the decentralized sufficient certificates of Corollaries 1–4, returning for each the critical reactive droop gain.

## Installation

This project was developed using `Python=3.10`. You can install this package in development mode using pip into your desired python environment, which uses the pyproject.toml file.

```bash
pip install -e .
```

## Basic Usage
```python
import numpy as np
from inverter_stability_criteria.system_classes import case9_sparse_pv_pq_embedded_builder
from inverter_stability_criteria.stability_criteria import (
    scan_stability_case, pv_slack_embedded_gain_profile,
)

df = scan_stability_case(
    case_builder=case9_sparse_pv_pq_embedded_builder,
    p_f_values=np.linspace(1.0, 3.1, 100),
    gain_profile_builder=pv_slack_embedded_gain_profile,
)
```

Each row of `df` corresponds to one loading factor `p_f`. The gain profile sets
`k_i^q = alpha * h_i`, with `h_i = 1` at the slack and PV buses and
`h_i = 0.1` at the PQ buses, so the reported critical values are upper bounds
on the scalar droop level `k̄^q`:

| column | certificate |
| --- | --- |
| `k_exact` | Theorem 2, exact (`Υ̂ ≻ 0`) |
| `k_cor1`, `k_cor2` | Corollaries 1 and 2, cycle contribution neglected |
| `k_cor3` | Corollary 3, nodal certificate |
| `k_cor4` | Corollary 4, edge certificate |


## Reproducing the Results

```bash
python draft_plots.py     # Figs. 3 and 4
python draft_results.py   # diagnostics quoted in Sec. VI-B
```

`draft_plots.py` writes `Two_identical_inverters.pdf` (Fig. 3) and
`Testcases_figure.pdf` (Fig. 4).
`draft_results.py` prints the tables behind two claims in Sec. VI-B: that the ordering of the Corollary 3 and 4 bounds is not universal, and that the limiting node or edge changes with loading.

Running both scripts takes a few minutes. Case 118 takes the longest.

## Citation

If you use this code, please cite:

```bibtex
@misc{bacic2026graph,
  title         = {A graph theoretic view on small signal stability of
                   inverter-based power grids},
  author        = {Ba{\v c}i{\'c}, Iva and Niehues, Jakob and
                   B{\"o}ttcher, Philipp C. and Dieball, Cai and
                   Rydin Gorj{\~a}o, Leonardo and Benigni, Andrea and
                   Hellmann, Frank and Witthaut, Dirk},
  year          = {2026},
  eprint        = {2607.08260},
  archivePrefix = {arXiv},
  primaryClass  = {eess.SY},
  url           = {https://arxiv.org/abs/2607.08260}
}
```


## Contributors
- Iva Bačić [Orcid](https://orcid.org/0000-0003-2987-5065) 

- Philipp C. Böttcher [Orcid](https://orcid.org/0000-0002-3240-0442) 
