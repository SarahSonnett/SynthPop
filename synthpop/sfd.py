"""Assign diameters to the synthetic population (the size-frequency distribution).

Diameter is the master physical parameter: it selects which PyLEADER diameter
bin supplies an object's shape/pole (:mod:`synthpop.shape_pole`), it drives the
size-orbit relation for families (:mod:`synthpop.orbits_family`), and together
with the sampled albedo it fixes the absolute magnitude ``H`` needed by the
downstream NEOWISE-survey simulator.

We sample diameters from a (optionally broken) cumulative power law
``N(>D) propto D**-slope`` by inverse-transform sampling. A broken power law
lets the small-size slope differ from the large-size slope, as is typical for
collisionally evolved populations.
"""

from __future__ import annotations

import numpy as np
from .config import SynthConfig


def _single_power_law(n, dmin, dmax, slope, rng):
    # pdf(D) propto D**-(slope+1) on [dmin, dmax]; inverse-CDF sample.
    a = slope  # cumulative exponent
    if np.isclose(a, 0.0):
        return dmin * (dmax / dmin) ** rng.random(n)
    lo, hi = dmin ** (-a), dmax ** (-a)
    u = rng.random(n)
    return (lo + u * (hi - lo)) ** (-1.0 / a)


def sample_diameters(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw ``cfg.n_objects`` diameters (km) from the configured SFD."""
    n, dmin, dmax = cfg.n_objects, cfg.diam_min_km, cfg.diam_max_km

    if cfg.sfd_kind == "loguniform":
        # Equal counts per log-D bin: not a realistic SFD, but it gives every
        # size range (incl. rare large objects) the same statistical weight --
        # what you want when the population's job is estimating the detection
        # probability P(detect | D) per size bin for survey debiasing.
        return dmin * (dmax / dmin) ** rng.random(n)
    if cfg.sfd_kind != "powerlaw":
        raise ValueError(f"unknown sfd_kind {cfg.sfd_kind!r} "
                         "(expected 'powerlaw' or 'loguniform')")

    if cfg.sfd_break_km is None or cfg.sfd_slope_small is None:
        return _single_power_law(n, dmin, dmax, cfg.sfd_slope, rng)

    # Broken power law: two segments joined continuously at the break diameter.
    dbreak = cfg.sfd_break_km
    s_small, s_large = cfg.sfd_slope_small, cfg.sfd_slope

    def seg_mass(d0, d1, slope):
        a = slope
        if np.isclose(a, 0.0):
            return np.log(d1 / d0)
        return (d0 ** -a - d1 ** -a) / a

    # weight each segment by its integrated number, matching density at dbreak
    w_small = seg_mass(dmin, dbreak, s_small)
    # continuity factor so the large segment's amplitude matches at dbreak
    cont = dbreak ** (s_large - s_small)
    w_large = cont * seg_mass(dbreak, dmax, s_large)
    p_small = w_small / (w_small + w_large)

    pick_small = rng.random(n) < p_small
    out = np.empty(n)
    ns = int(pick_small.sum())
    out[pick_small] = _single_power_law(ns, dmin, dbreak, s_small, rng)
    out[~pick_small] = _single_power_law(n - ns, dbreak, dmax, s_large, rng)
    return out
