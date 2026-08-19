#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions used to plot the results."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Adapting figures to IEEE standards
IEEE_ONE_COL = 3.50 # inches, IEEE one-column width
IEEE_TWO_COL = 7.16 # inches, IEEE two-column width

IEEE_FONT = 9
IEEE_SMALL_FONT = 8
IEEE_TITLE_FONT = 9.5
IEEE_LINEWIDTH = 1.15
IEEE_GRID_ALPHA = 0.22

# Colorblind friendly palette
colors = {
    "lemma3": "#000000",  # black
    "cor1":   "#0072B2",  # blue
    "cor2":   "#56B4E9",  # sky blue
    "cor3":   "#D55E00",  # vermillion
    "cor4":   "#E69F00"   # orange
}

line_styles = {
    "lemma3": dict(color=colors["lemma3"], linestyle="-",  marker=None),
    "cor1":   dict(color=colors["cor1"],   linestyle="--", marker="s"),
    "cor2":   dict(color=colors["cor2"],   linestyle=":",  marker="D"),
    "cor3":   dict(color=colors["cor3"],   linestyle="-.", marker="o"),
    "cor4":   dict(color=colors["cor4"],   linestyle=(0, (5, 2, 1, 2)), marker="^")
}

def set_ieee_mpl_style():
    """Apply a consistent IEEE-friendly Matplotlib style."""
    mpl.rcParams.update({
        # Typography
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": IEEE_FONT,
        "axes.labelsize": IEEE_FONT,
        "axes.titlesize": IEEE_TITLE_FONT,
        "xtick.labelsize": IEEE_SMALL_FONT,
        "ytick.labelsize": IEEE_SMALL_FONT,
        "legend.fontsize": IEEE_SMALL_FONT,
        "legend.title_fontsize": IEEE_SMALL_FONT,

        # Lines and axes
        "axes.linewidth": 0.7,
        "lines.linewidth": IEEE_LINEWIDTH,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.25,

        # Export: keep text editable/searchable in PDF
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.03,

        # Layout
        "figure.constrained_layout.use": True,
    })

def _markevery(n, target_markers=8):
    """Sparse markers to avoid clutter on dense curves."""
    return max(1, int(np.ceil(n / target_markers)))

def _apply_axis_style(ax):
    ax.grid(True, which="major", alpha=IEEE_GRID_ALPHA, linewidth=0.5)
    ax.tick_params(direction="out")
    ax.set_axisbelow(True)

def _ieee_plot(ax, x, y, key, label, *, logy=False, lw=IEEE_LINEWIDTH, zorder=None):
    style = line_styles[key].copy()
    marker = style.pop("marker")

    kwargs = dict(label=label, lw=lw, zorder=zorder, **style)

    if marker is not None:
        kwargs.update(marker=marker, markevery=_markevery(len(x)), markerfacecolor="none", markeredgewidth=0.75)

    if logy:
        return ax.semilogy(x, y, **kwargs)
    return ax.plot(x, y, **kwargs)

def save_ieee_figure(fig, savepath):
    """
    Save figure as vector PDF.
    """
    if savepath is not None:
        fig.savefig(savepath, dpi=600)

# === Figure 3 ===
def plot_Figure3(B=1.0, v=1.0, c = np.linspace(0.02, 1.0, 500), figsize=(IEEE_ONE_COL, 1.5), savepath = None):
    """Plots certificates for two identical inverters (no cycles)."""
    set_ieee_mpl_style()

    k_exact = c / (2 * B * v**2 * (1 - c))
    k_cor3 = c / (2 * B * v**2* (1 - c))

    fig, ax = plt.subplots(figsize=figsize)

    _ieee_plot(ax, c, k_exact, "lemma3", r"Theorem 2", logy=False, lw=1.45, zorder=2)
    _ieee_plot(ax, c, k_cor3, "cor3", r"Corollary 3", logy=False, lw=1.05, zorder=3)

    plt.ylim(0, 7)
    plt.xlabel(r"$\cos(\Delta\theta_{12}^{\circ})$")
    plt.ylabel(r"$k^q$")
    _apply_axis_style(ax)
    plt.legend()
    plt.grid(True, which="major", alpha=0.25)
    if savepath is not None:
        save_ieee_figure(fig, savepath)

# === Figure 4 ===
def _good_df(df):
    good = df.copy()

    if "converged" in good.columns:
        good = good[good["converged"]].copy()

    if "all_cos_positive" in good.columns:
        good = good[good["all_cos_positive"]].copy()

    return good.sort_values("p_f")

def plot_combined_certificate_cycle_grid(dfs, case_labels,  figsize=(IEEE_TWO_COL, 4.85), savepath=None, show=True):
    """
    Figure 4: 3 columns and 4 rows.

    Columns:
      Case 9, Case 30, Case 118.

    Rows:
      1. Stability certificates.
      2. Maximum angle difference.
      3. Cycle spectrum and critical-mode projection.
      4. Relative boundary gap.
    """

    set_ieee_mpl_style()

    fig, axs = plt.subplots(4, 3, figsize=figsize, sharex="col",
                            gridspec_kw={"height_ratios": [2.35, 0.85, 1.75, 0.85], "hspace": 0.055, "wspace": 0.14}
                            )

    for col, (df, case_label) in enumerate(zip(dfs, case_labels)):
        good = _good_df(df)
        p_f = good["p_f"].to_numpy(dtype=float)

        # Row 1: Stability certificates
        ax = axs[0, col]

        _ieee_plot(ax, p_f, good["k_exact"], "lemma3", r"Th. 2, $\widehat{\Upsilon}$", logy=True)
        _ieee_plot(ax, p_f, good["k_cor1"], "cor1", r"Cor. 1, $\Upsilon$", logy=True)
        _ieee_plot(ax, p_f, good["k_cor2"], "cor2", r"Cor. 2, $\Psi$", logy=True)
        _ieee_plot(ax, p_f, good["k_cor3"], "cor3", r"Cor. 3", logy=True)
        _ieee_plot(ax, p_f, good["k_cor4"], "cor4", r"Cor. 4", logy=True)

        if col == 0:
            ax.set_ylabel(r"$\bar{k}^q_{\rm crit}$")

        ax.set_title(case_label)
        ax.set_xlim(float(np.min(p_f)), float(np.max(p_f)))
        ax.margins(x=0)
        ax.tick_params(axis="x", labelbottom=False)
        _apply_axis_style(ax)

        if col == 0:
            ax.legend(loc="upper right", ncol=1, frameon=False)

        # Row 2: Maximum angle difference
        ax = axs[1, col]

        ax.plot(p_f, good["max_angle"] / (0.5 * np.pi), lw=IEEE_LINEWIDTH, color="k")

        if col == 0:
            ax.set_ylabel(r"$\max|\Delta\theta_{ij}^{\circ}|/(\pi/2)$")

        ax.set_xlim(float(np.min(p_f)), float(np.max(p_f)))
        ax.margins(x=0)
        ax.tick_params(axis="x", labelbottom=False)
        _apply_axis_style(ax)

        # Row 3: Cycle spectrum + projection
        eig_lists = []
        max_len = 0
        for eigs in good["cycle_eigs"]:
            eigs = np.asarray(eigs, dtype=float)
            eigs = eigs[eigs > 1e-12]
            eigs = np.sort(eigs)[::-1]
            eig_lists.append(eigs)
            max_len = max(max_len, len(eigs))

        eig_arr = np.full((len(good), max_len), np.nan)

        for i, eigs in enumerate(eig_lists):
            eig_arr[i, :len(eigs)] = eigs

        projection = good["cycle_on_critical_mode"].to_numpy(dtype=float)

        ax = axs[2, col]

        cmap = plt.cm.viridis
        eig_colors = cmap(np.linspace(0.12, 0.88, max(max_len, 1)))

        for j in range(max_len):
            if max_len == 1:
                label = r"$\lambda_{1}(\Upsilon_{\rm cycle})$"
            elif j == 0:
                label = r"$\lambda_{1}(\Upsilon_{\rm cycle})$"
            elif j == max_len - 1:
                label = rf"$\lambda_{{{max_len}}}(\Upsilon_{{\rm cycle}})$"
            else:
                label = None

            ax.semilogy(p_f, eig_arr[:, j], lw=0.75, alpha=0.9, color=eig_colors[j], label=label)

        positive_projection = np.where(projection > 0, projection, np.nan)

        ax.semilogy(p_f, positive_projection, "k--", lw=1.35, label=r"$x_{\min}^{\top}\Upsilon_{\rm cycle}x_{\min}$")

        if col == 0:
            ax.set_ylabel(r"Cycle contribution")

        ax.set_xlim(float(np.min(p_f)), float(np.max(p_f)))
        ax.margins(x=0)
        ax.tick_params(axis="x", labelbottom=False)
        _apply_axis_style(ax)

        ax.legend(loc="lower right", ncol=1, frameon=False)

        # Row 4: Relative boundary gap
        gap = good["k_ratio"].to_numpy(dtype=float) - 1.0
        positive_gap = np.where(gap > 0, gap, np.nan)

        ax = axs[3, col]

        ax.semilogy(p_f, positive_gap, lw=IEEE_LINEWIDTH, color="k")

        if col == 0:
            ax.set_ylabel(r"$\bar{k}^q_{\rm crit}(\widehat{\Upsilon})/\bar{k}^q_{\rm crit}(\Upsilon)-1$")

        ax.set_xlabel(r"$p_f$")
        ax.set_xlim(float(np.min(p_f)), float(np.max(p_f)))
        ax.margins(x=0)
        _apply_axis_style(ax)

    if savepath is not None:
        save_ieee_figure(fig, savepath)

    if show:
        plt.show()

    return fig, axs