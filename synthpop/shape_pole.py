"""Assign shape elongation (b/a) and spin-pole coordinates.

For each synthetic asteroid we:

* select the PyLEADER diameter bin whose [low, high) range contains the
  object's diameter (falling back to the nearest bin if it lies outside all of
  them);
* draw the **axis ratio** ``b/a`` and **pole latitude** magnitude ``|beta|``
  *jointly* from that bin's PyLEADER joint distribution ``f_mean(p, beta)``
  (``population_distribution.npz``), preserving the shape/spin correlation
  rather than sampling the two marginals independently. LEADER is
  hemisphere-degenerate, so ``|beta|`` (in [0, 90] deg) is stored as-is --
  no hemisphere sign is invented;
* draw the **pole longitude** ``lambda`` uniformly on [0, 360) deg -- LEADER is
  insensitive to longitude, so it is unconstrained by prior studies;
* draw the **rotation period** log-uniformly between the configured bounds
  (default 2.2-20 h: from the cohesionless spin barrier up through typical MBA
  periods).

Note on conventions: PyLEADER's beta follows Nortunen & Kaasalainen (2017) --
it is the **ecliptic latitude of the spin pole** (a magnitude in [0, 90] deg;
hemispheres are indistinguishable), *not* the obliquity. For a low-inclination
orbit the obliquity is approximately ``90 - |beta|`` deg (prograde branch) or
``90 + |beta|`` deg (retrograde branch).

Distributions are loaded once per bin and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SynthConfig, DiameterBin
from .pyleader_io import load_distributions, LeaderDistributions


@dataclass
class ShapePole:
    b_over_a: np.ndarray       # axis ratio b/a in (0, 1]
    pole_beta_deg: np.ndarray  # |ecliptic latitude| of spin pole, [0, 90] deg
    pole_lambda_deg: np.ndarray  # ecliptic longitude of spin pole, [0, 360)
    rot_period_h: np.ndarray   # rotation period, hours (log-uniform draw)


def _select_bin(bins, diam_km: float) -> DiameterBin:
    # Among bins that contain the diameter, prefer the narrowest -- PyLEADER
    # often emits a wide "all sizes" bin (e.g. 1-100 km) alongside fine bins;
    # the fine bin is the more specific distribution.
    containing = [b for b in bins if b.contains(diam_km)]
    if containing:
        return min(containing, key=lambda b: b.high_km - b.low_km)
    # outside all bins -> nearest bin by center
    centers = [0.5 * (b.low_km + b.high_km) for b in bins]
    return bins[int(np.argmin(np.abs(np.array(centers) - diam_km)))]


def assign_shape_pole(cfg: SynthConfig, diameters: np.ndarray,
                      rng: np.random.Generator) -> ShapePole:
    if not cfg.diameter_bins:
        raise ValueError("cfg.diameter_bins is empty; cannot assign shape/pole")

    cache: dict = {}

    def dists(b: DiameterBin) -> LeaderDistributions:
        key = str(b.dist_dir)
        if key not in cache:
            cache[key] = load_distributions(b.dist_dir)
        return cache[key]

    n = diameters.size
    ba = np.empty(n)
    beta = np.empty(n)

    # group objects by bin so each joint PDF is sampled in one vectorised call
    bin_of = [_select_bin(cfg.diameter_bins, d) for d in diameters]
    keys = np.array([str(b.dist_dir) for b in bin_of])
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        b = bin_of[idx[0]]
        p, bta = dists(b).joint.sample(idx.size, rng)   # joint (b/a, |beta|) draw
        ba[idx] = np.clip(p, 1e-3, 1.0)
        beta[idx] = bta

    # LEADER constrains only |beta|; store the magnitude (no invented sign)
    pole_beta = np.clip(np.abs(beta), 0.0, 90.0)
    pole_lambda = rng.random(n) * 360.0

    # rotation period: log-uniform between the configured bounds
    lo, hi = cfg.rot_period_min_h, cfg.rot_period_max_h
    rot_period = lo * (hi / lo) ** rng.random(n)

    return ShapePole(b_over_a=ba, pole_beta_deg=pole_beta,
                     pole_lambda_deg=pole_lambda, rot_period_h=rot_period)


def mixture_marginals(cfg: SynthConfig, diameters: np.ndarray):
    """The p and |beta| marginal PDFs of the PyLEADER model actually sampled.

    Objects draw from different diameter bins, so the population-level source
    distribution is the bin-count-weighted mixture of the per-bin joint
    ``f_mean`` marginals. Overlaying these curves on the sampled histograms
    verifies the joint sampler against its source.

    Returns ``{"p": (grid, pdf), "beta": (grid, pdf)}`` with each pdf
    normalised to unit integral over its grid.
    """
    from .sampling import _edges_from_centers

    bin_of = [_select_bin(cfg.diameter_bins, d) for d in diameters]
    keys = np.array([str(b.dist_dir) for b in bin_of])
    out = {}
    acc = {}  # dist_dir -> (weight, LeaderDistributions)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        b = bin_of[idx[0]]
        acc[key] = (idx.size / diameters.size, load_distributions(b.dist_dir))

    # all bins share PyLEADER's canonical grids; use the first as reference
    first = next(iter(acc.values()))[1]
    for axis, grid in (("p", first.p_grid), ("beta", first.b_grid)):
        pdf = np.zeros_like(grid)
        for w, dist in acc.values():
            dp = np.diff(_edges_from_centers(dist.p_grid))    # p bin widths
            db = np.diff(_edges_from_centers(dist.b_grid))    # beta bin widths
            if axis == "p":       # integrate over beta -> density in p
                marg = (dist.f_mean * db[None, :]).sum(axis=1)
                widths = dp
            else:                 # integrate over p -> density in beta
                marg = (dist.f_mean * dp[:, None]).sum(axis=0)
                widths = db
            total = (marg * widths).sum()
            if total > 0:
                pdf += w * marg / total   # unit integral per bin, then weight
        out[axis] = (grid, pdf)
    return out
