"""Diagnostic plots of the synthetic population's orbital and physical properties.

Four figures are written next to the catalog (PNG, like PyLEADER's outputs):

* ``<name>_orbits.png``      -- (a, e) and (a, i) planes; in family mode the
                                real members are underplotted in gray so the
                                synthetic family can be judged against its
                                observed counterpart.
* ``<name>_sfd.png``         -- cumulative size-frequency distribution N(>D)
                                with the configured sampling law as reference.
* ``<name>_shape_pole.png``  -- shape elongation b/a, pole latitude |beta|
                                (with an approximate-obliquity axis), pole
                                longitude (uniformity check), and the joint
                                (b/a, |beta|) density actually sampled.
* ``<name>_physical.png``    -- albedo and beaming eta with their fitted
                                models overlaid, rotation period, and the
                                derived H distribution.

Every figure is explicitly marked as a **synthetic** population: titles carry
"Synthetic <name>" and a footer states that these are model draws, not observed
asteroids -- so nobody skimming the figures mistakes them for measured
distributions of the real family.

Colors follow the Okabe-Ito colorblind-safe palette (Wong 2011): synthetic
population in blue, real/observed context in gray, fitted models in vermillion.
Series identity is always carried by a legend and mark type, never color alone.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless; we only write files
import matplotlib.pyplot as plt

# Okabe-Ito (colorblind-safe) role assignments
C_SYNTH = "#0072B2"   # synthetic population
C_REAL = "#999999"    # observed/real context layer (step-histogram overlays)
C_OBS = "#E69F00"     # observed points over a synthetic density field
C_FIT = "#D55E00"     # fitted model curves / reference lines

_GRID = dict(alpha=0.25, linewidth=0.6)

_FOOTER = ("Fully SYNTHETIC population (SynthPop model draws), not observed asteroids. "
           "Distributions follow models fit to real measurements (PyLEADER, NEOWISE).")


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(True, **_GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _finish(fig, path: Path) -> Path:
    """Stamp the synthetic-data footer and write the figure."""
    # reserve a strip below the axes so the footer never collides with labels
    # (rect is (left, bottom, WIDTH, HEIGHT) -- height must shrink to fit)
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.955))
    fig.text(0.005, 0.005, _FOOTER, fontsize=7, style="italic",
             color="#666666", ha="left", va="bottom")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _broken_uniform_hist(fig, ax, vals, xlabel, title, nbins=36, bins=None,
                         log_x=False):
    """Histogram of an expected-uniform quantity with a broken y-axis.

    The top segment zooms the bar-top region and draws the uniform
    expectation with its Poisson +/-1 sigma band, so per-bin wiggle reads as
    quantified counting noise; the bottom segment anchors the bars at zero.
    Pass log-spaced ``bins`` (+``log_x``) for a quantity that is uniform per
    *log* bin (e.g. a log-uniform rotation-period draw) -- equal expected
    counts per bin still hold. Replaces ``ax`` with two stacked axes.
    """
    spec = ax.get_subplotspec()
    ax.remove()
    sub = spec.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
    top = fig.add_subplot(sub[0])
    bot = fig.add_subplot(sub[1], sharex=top)

    vals = np.asarray(vals)
    if bins is None:
        bins = nbins
    else:
        nbins = len(bins) - 1
    mean = vals.size / nbins
    sig = np.sqrt(mean)
    for a in (top, bot):
        a.hist(vals, bins=bins, color=C_SYNTH, edgecolor="white", linewidth=0.5)
    if log_x:
        top.set_xscale("log")  # shared x: bottom follows
    top.axhspan(mean - sig, mean + sig, color=C_FIT, alpha=0.15,
                label=r"Poisson $\pm1\sigma$")
    top.axhline(mean, color=C_FIT, linewidth=1.5, linestyle="--",
                label="uniform expectation")
    top.set_ylim(mean - 5 * sig, mean + 5 * sig)
    bot.set_ylim(0, 2.5 * sig)

    _style(top, "", "count", title)
    _style(bot, xlabel, "", "")
    top.legend(frameon=False, fontsize=8)
    top.tick_params(labelbottom=False)
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    bot.set_yticks([0])

    # diagonal break markers
    d = 0.5
    kw = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle="none",
              color="k", mec="k", mew=1, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **kw)
    bot.plot([0], [1], transform=bot.transAxes, **kw)


def _contour_levels(H, fractions=(0.95, 0.68, 0.39)):
    """Histogram levels enclosing the given fractions of the total counts."""
    vals = np.sort(H.ravel())[::-1]
    cum = np.cumsum(vals) / vals.sum()
    return sorted({float(vals[np.searchsorted(cum, f)]) for f in fractions
                   if np.searchsorted(cum, f) < vals.size})


def plot_orbits(df, out: Path, name: str, real=None) -> Path:
    """Marginal a/e/i comparisons, the (a, e)/(a, i) planes, and angle checks.

    Row 1: one-dimensional distributions of a, e, and i -- the synthetic
    population against the real members, the form in which the two are most
    directly comparable. Row 2: the (a, e) and (a, i) planes, with the
    synthetic population as a density field and the real members as density
    contours (enclosing 39/68/95% of members) -- two dense point sets cannot
    be overlaid legibly, but contours over a field can. Row 3: the secular
    angles against the uniform expectation (the portal element files carry no
    angles, so no observed reference exists there).
    """
    disp = f"Synthetic {name}"
    fig = plt.figure(figsize=(13.5, 12.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    real_i = (np.degrees(np.arcsin(np.clip(real["sini"], -1, 1)))
              if real else None)

    # --- row 1: marginal distributions of a, e, i --------------------------
    marginals = [
        ("semi-major axis a (AU)", df["a_au"], real["a"] if real else None),
        ("eccentricity e", df["e"], real["e"] if real else None),
        ("inclination (deg)", df["i_deg"], real_i),
    ]
    for k, (xlabel, syn, obs) in enumerate(marginals):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        _, bins, _ = ax.hist(syn, bins=40, density=True, color=C_SYNTH,
                             edgecolor="white", linewidth=0.5,
                             label="synthetic")
        if obs is not None:
            ax.hist(obs, bins=bins, density=True, histtype="step",
                    color="#555555", linewidth=1.4,
                    label="real members (observed)")
        _style(ax, xlabel, "density", f"{disp}: {xlabel.split(' (')[0]}")
        ax.legend(frameon=False, fontsize=7)

    # --- row 2: (a, e) and (a, i) planes -----------------------------------
    HEXBIN_ABOVE = 3000       # scatter saturates above this; use a field
    CONTOUR_ABOVE = 300       # contours need enough members to be stable
    as_density = len(df) > HEXBIN_ABOVE

    planes = [
        ("eccentricity e", df["e"], real["e"] if real else None),
        ("inclination (deg)", df["i_deg"], real_i),
    ]
    for k, (ylabel, y_syn, y_real) in enumerate(planes):
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        handles = []
        if as_density:
            hb = ax.hexbin(df["a_au"], y_syn, gridsize=40, cmap="Blues",
                           mincnt=1, bins="log")
            # both colorbars show the same quantity; label only the outermost
            # so the inner label cannot collide with its neighbor's y-axis
            fig.colorbar(hb, ax=ax,
                         label="synthetic count (log)" if k == 1 else "")
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=C_SYNTH,
                                         label="synthetic (model density)"))
        else:
            ax.scatter(df["a_au"], y_syn, s=3, c=C_SYNTH, alpha=0.35,
                       linewidths=0, label="synthetic (model)",
                       rasterized=True)
        if y_real is not None:
            if len(y_real) >= CONTOUR_ABOVE:
                H, xe, ye = np.histogram2d(real["a"], y_real, bins=36)
                try:
                    from scipy.ndimage import gaussian_filter
                    H = gaussian_filter(H, 1.0)
                except ImportError:
                    pass
                xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
                levels = _contour_levels(H)
                if levels:
                    ax.contour(xc, yc, H.T, levels=levels, colors=C_OBS,
                               linewidths=1.2)
                handles.append(plt.Line2D([], [], color=C_OBS,
                                          label="real members (39/68/95% contours)"))
            else:
                ax.scatter(real["a"], y_real, s=5, facecolors="none",
                           edgecolors=C_OBS, linewidths=0.5, alpha=0.5,
                           label="real members (observed)", rasterized=True)
        _style(ax, "semi-major axis a (AU)", ylabel, f"{disp}: a vs {ylabel}")
        h, _ = ax.get_legend_handles_labels()
        leg = ax.legend(handles=h + handles, frameon=False, fontsize=8,
                        markerscale=3)
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)

    # --- row 3: secular-angle uniformity checks ----------------------------
    angle_panels = [
        ("longitude of ascending node Omega (deg)", df["node_deg"]),
        ("argument of perihelion omega (deg)", df["argperi_deg"]),
    ]
    for k, (xlabel, vals) in enumerate(angle_panels):
        ax = fig.add_subplot(gs[2, 3 * k:3 * k + 3])
        _broken_uniform_hist(fig, ax, vals, xlabel,
                             f"{disp}: {xlabel.split(' (')[0]}")

    return _finish(fig, out / f"{name}_orbits.png")


def plot_sfd(df, out: Path, name: str, kind: str = "powerlaw",
             slope: float | None = None, real_D=None) -> Path:
    """Cumulative N(>D), log-log, with the input sampling law as reference.

    ``real_D`` optionally overlays the real members' NEOWISE diameters. Note
    that curve is *detection-biased* (only members NEOWISE measured), which is
    exactly the incompleteness Simmer is built to correct.
    """
    disp = f"Synthetic {name}"
    fig, ax = plt.subplots(figsize=(5.6, 4.7), constrained_layout=True)
    d = np.sort(np.asarray(df["diam_km"]))[::-1]
    n_gt = np.arange(1, d.size + 1)
    ax.loglog(d, n_gt, color=C_SYNTH, linewidth=2, label="synthetic N(>D)")
    if real_D is not None and len(real_D):
        rd = np.sort(np.asarray(real_D)[np.isfinite(real_D)])[::-1]
        ax.loglog(rd, np.arange(1, rd.size + 1), color=C_REAL, linewidth=1.5,
                  drawstyle="steps-post",
                  label="real members (NEOWISE-detected)")

    d_ref = np.geomspace(d.min(), d.max(), 100)
    if kind == "loguniform":
        # expectation for equal counts per log-D bin
        n_ref = d.size * np.log(d.max() / d_ref) / np.log(d.max() / d.min())
        ax.loglog(d_ref, np.clip(n_ref, 1, None), color=C_FIT, linewidth=1.5,
                  linestyle="--", label="log-uniform expectation")
    elif slope is not None:
        anchor_d = np.median(d)
        anchor_n = np.interp(anchor_d, d[::-1], n_gt[::-1])
        ax.loglog(d_ref, anchor_n * (d_ref / anchor_d) ** (-slope),
                  color=C_FIT, linewidth=1.5, linestyle="--",
                  label=f"input slope {slope:g}")
    _style(ax, "diameter D (km)", "N(>D)",
           f"{disp}: size-frequency distribution ({kind})")
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, out / f"{name}_sfd.png")


def plot_shape_pole(df, out: Path, name: str, model_marginals=None) -> Path:
    """b/a, pole latitude |beta| (+obliquity axis), pole longitude, joint.

    ``model_marginals`` (from :func:`synthpop.shape_pole.mixture_marginals`)
    overlays the PyLEADER source-model marginals on the sampled histograms,
    verifying the joint sampler reproduces its input distribution.
    """
    disp = f"Synthetic {name}"
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.hist(df["b_over_a"], bins=30, density=True, color=C_SYNTH,
            edgecolor="white", linewidth=0.5, label="synthetic (sampled)")
    if model_marginals is not None:
        gx, gp = model_marginals["p"]
        ax.plot(gx, gp, color=C_FIT, linewidth=1.8,
                label="PyLEADER model marginal")
    _style(ax, "axis ratio b/a", "density", f"{disp}: shape elongation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.hist(df["pole_beta_deg"], bins=30, density=True, color=C_SYNTH,
            edgecolor="white", linewidth=0.5, label="synthetic (sampled)")
    if model_marginals is not None:
        gb, pb = model_marginals["beta"]   # |beta| marginal
        ax.plot(gb, pb, color=C_FIT, linewidth=1.8,
                label="PyLEADER model marginal")
    _style(ax, "pole ecliptic latitude |beta| (deg)", "density",
           f"{disp}: pole latitude (hemisphere-degenerate)")
    ax.legend(frameon=False, fontsize=8)
    # pole_beta_deg is the pole's |ecliptic latitude|; for a low-inclination
    # orbit the obliquity is ~ 90 - |beta| deg (prograde branch; the
    # retrograde branch is 90 + |beta|). KNOWN ISSUE (shape_pole.py): legacy
    # catalogs hold the polar angle, so for them both axis labels are wrong.
    sec = ax.secondary_xaxis("top", functions=(lambda b: 90.0 - b,
                                               lambda e: 90.0 - e))
    sec.set_xlabel("approx. obliquity (deg; prograde branch)", fontsize=8)
    sec.tick_params(labelsize=8)

    _broken_uniform_hist(fig, axes[1, 0], df["pole_lambda_deg"],
                         "pole ecliptic longitude lambda (deg)",
                         f"{disp}: pole longitude")

    ax = axes[1, 1]
    hb = ax.hexbin(df["b_over_a"], df["pole_beta_deg"].abs(), gridsize=24,
                   cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="count")
    _style(ax, "axis ratio b/a", "|beta| (deg)",
           f"{disp}: joint shape-spin density (as sampled)")

    return _finish(fig, out / f"{name}_shape_pole.png")


def plot_physical(df, out: Path, name: str, albedo_fit=None, eta_fit=None,
                  real_phys=None, albedo_min: float = 0.0) -> Path:
    """pV and eta with their fitted models, rotation period, derived H.

    ``albedo_fit`` / ``eta_fit`` are the fitted distribution objects from
    :mod:`synthpop.distributions` (each exposing ``.pdf``), if available.
    ``real_phys`` optionally overlays the real members' measured values as
    gray step histograms: a dict with any of ``pV``, ``eta``, ``H``.
    """
    disp = f"Synthetic {name}"
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6), constrained_layout=True)

    def _real_overlay(ax, key, bins):
        if real_phys is None:
            return
        vals = np.asarray(real_phys.get(key, []))
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    color="#555555", linewidth=1.4,
                    label="real members (measured)")

    ax = axes[0, 0]
    # log-spaced bins + log x: albedo spans a decade-plus and dark families
    # concentrate at pV ~ 0.02-0.1, which a linear axis squashes. density=True
    # normalises per *linear* pV either way, so the fitted pdf overlays as-is.
    pv = df["pV"].clip(lower=1e-3)
    bins = np.geomspace(pv.min(), max(0.6, float(pv.max())), 40)
    ax.hist(pv, bins=bins, density=True, color=C_SYNTH,
            edgecolor="white", linewidth=0.5, label="synthetic")
    _real_overlay(ax, "pV", bins)
    if albedo_fit is not None:
        x = np.geomspace(max(bins[0], albedo_min or bins[0]), bins[-1], 400)
        pdf = albedo_fit.pdf(x)
        # name the fitted model honestly
        label = {"TruncatedGaussian": "Gaussian fit",
                 "LogNormal": "lognormal fit",
                 "DoubleLogNormal": "double-lognormal fit",
                 "DoubleRayleigh": "double-Rayleigh fit",
                 "LogNormalMixture": "lognormal-mixture fit"}.get(
                     type(albedo_fit).__name__, "model fit")
        if albedo_min > 0:
            # renormalise over the truncated support so the curve matches the
            # (floored) sampled density
            pdf = pdf / max(1.0 - float(albedo_fit.cdf(albedo_min)), 1e-12)
            label += f" (pV >= {albedo_min:g})"
        ax.plot(x, pdf, color=C_FIT, linewidth=1.8, label=label)
    ax.set_xscale("log")
    _style(ax, "visible albedo pV", "density", f"{disp}: albedo")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    _, bins, _ = ax.hist(df["eta"], bins=40, density=True, color=C_SYNTH,
                         edgecolor="white", linewidth=0.5, label="synthetic")
    _real_overlay(ax, "eta", bins)
    if eta_fit is not None:
        x = np.linspace(float(df["eta"].min()), float(df["eta"].max()), 400)
        ax.plot(x, eta_fit.pdf(x), color=C_FIT, linewidth=1.8, label="Gaussian fit")
    _style(ax, "beaming parameter eta", "density", f"{disp}: beaming")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if "rot_period_h" in df.columns:
        # the draw is log-uniform, so counts are expected-uniform per log bin;
        # same broken-axis treatment as the angle panels
        bins = np.geomspace(df["rot_period_h"].min(), df["rot_period_h"].max(), 31)
        _broken_uniform_hist(fig, ax, df["rot_period_h"], "rotation period (h)",
                             f"{disp}: rotation period (log-uniform draw)",
                             bins=bins, log_x=True)
    else:
        ax.set_axis_off()

    ax = axes[1, 1]
    _, bins, _ = ax.hist(df["H"], bins=40, density=True, color=C_SYNTH,
                         edgecolor="white", linewidth=0.5, label="synthetic")
    _real_overlay(ax, "H", bins)
    _style(ax, "absolute magnitude H", "density", f"{disp}: H (from D and pV)")
    ax.legend(frameon=False, fontsize=8)

    return _finish(fig, out / f"{name}_physical.png")


def make_all_plots(df, cfg, real_elements=None, real_phys=None,
                   model_marginals=None, albedo_fit=None, eta_fit=None) -> list:
    """Write the full diagnostic set into ``cfg.out_dir``; returns the paths."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    real_D = real_phys.get("D") if real_phys else None
    return [
        plot_orbits(df, out, cfg.name, real=real_elements),
        plot_sfd(df, out, cfg.name, kind=cfg.sfd_kind, slope=cfg.sfd_slope,
                 real_D=real_D),
        plot_shape_pole(df, out, cfg.name, model_marginals=model_marginals),
        plot_physical(df, out, cfg.name, albedo_fit=albedo_fit, eta_fit=eta_fit,
                      real_phys=real_phys, albedo_min=cfg.albedo_min),
    ]
