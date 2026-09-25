"""Generate synthetic family orbits from the observed family.

Two generators are provided (``cfg.family_orbit_model``): the parametric
Gaussian + D_orb(D) model of Schunova-Lilly et al. (2017) ("dorb", the
default, described below) and a structure-preserving size-matched bootstrap
of the real member elements ("resample",
:func:`generate_family_orbits_resample`).

This implements the family-orbit recipe in the project spec (after
Schunova-Lilly et al. 2017):

1. **Fit the family center.** Fit a Gaussian to each proper element of the
   observed family -- semi-major axis ``a``, eccentricity ``e``, and sine of
   inclination ``sin(i)`` -- giving a center ``mu`` and width ``sigma`` per
   element.

2. **Orbital distance vs size.** For each *real* member compute its orbital
   distance from the family center, standardised by the Gaussian widths,

       D_orb = sqrt( ((a-mu_a)/sig_a)^2 + ((e-mu_e)/sig_e)^2
                     + ((sini-mu_sini)/sig_sini)^2 ),

   and fit a curve of ``D_orb`` versus diameter ``D``. Yarkovsky drift spreads
   small members farther from the center, so ``D_orb`` decreases with size; we
   fit a power law ``D_orb = A * D**gamma`` (gamma < 0) by linear regression in
   log-log space, and record the scatter of the residuals.

3. **Generate synthetic members.** For each synthetic object of assigned
   diameter ``D`` predict ``D_orb`` from the fit, add Gaussian scatter
   (the residual spread), place it at that standardised radius in a random
   direction in (a, e, sin i) space, and map back to physical elements via
   ``element = mu + sigma * offset``. Secular angles (node, argument of
   perihelion, mean anomaly) are drawn uniformly, as proper-element fuzzing
   leaves them unconstrained.

4. **Clip to observed bounds.** Reject and redraw any synthetic element falling
   outside the observed family's (a, e, sin i) range, so the synthetic family
   occupies an orbital space consistent with its real counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SynthConfig


@dataclass
class FamilyFit:
    mu: dict          # element -> center  (keys: 'a', 'e', 'sini')
    sigma: dict       # element -> Gaussian width
    bounds: dict      # element -> (min, max) of observed members
    dorb_A: float     # D_orb = A * D**gamma
    dorb_gamma: float
    dorb_scatter: float  # std of log10(D_orb) residuals about the fit
    n_members: int


@dataclass
class Orbits:
    a_au: np.ndarray
    e: np.ndarray
    i_deg: np.ndarray
    node_deg: np.ndarray
    argperi_deg: np.ndarray
    M_deg: np.ndarray


def load_family_members(cfg: SynthConfig):
    """Read the AFP family file -> (designation, H, a, e, sin_i) arrays.

    The ID column may hold either a packed number or a provisional designation
    (e.g. ``2013QB81``), so it is read as a string; the element columns are
    read as floats.
    """
    cols = cfg.family_cols
    designation = np.genfromtxt(cfg.family_members_file, usecols=[cols[0]], dtype=str)
    elems = np.genfromtxt(cfg.family_members_file,
                          usecols=[cols[1], cols[2], cols[3], cols[4]], dtype=float)
    H, a, e, sini = elems[:, 0], elems[:, 1], elems[:, 2], elems[:, 3]
    return designation, H, a, e, sini


def fit_family(a, e, sini, diameters) -> FamilyFit:
    """Fit the Gaussian center/width per element and the D_orb-vs-D relation."""
    elems = {"a": np.asarray(a, float), "e": np.asarray(e, float),
             "sini": np.asarray(sini, float)}
    mu = {k: float(np.mean(v)) for k, v in elems.items()}
    sigma = {k: float(np.std(v, ddof=1)) for k, v in elems.items()}
    bounds = {k: (float(np.min(v)), float(np.max(v))) for k, v in elems.items()}

    # standardised orbital distance of each member from the center
    z2 = np.zeros_like(elems["a"])
    for k, v in elems.items():
        if sigma[k] > 0:
            z2 += ((v - mu[k]) / sigma[k]) ** 2
    dorb = np.sqrt(z2)

    # fit log10(D_orb) = log10(A) + gamma * log10(D); guard against D_orb==0
    D = np.asarray(diameters, float)
    good = np.isfinite(D) & (D > 0) & (dorb > 0)
    lx, ly = np.log10(D[good]), np.log10(dorb[good])
    gamma, logA = np.polyfit(lx, ly, 1)
    resid = ly - (logA + gamma * lx)
    scatter = float(np.std(resid, ddof=1)) if resid.size > 2 else 0.1

    return FamilyFit(mu=mu, sigma=sigma, bounds=bounds,
                     dorb_A=float(10 ** logA), dorb_gamma=float(gamma),
                     dorb_scatter=scatter, n_members=int(elems["a"].size))


def generate_family_orbits_resample(cfg: SynthConfig, a, e, sini,
                                    member_D, diameters: np.ndarray,
                                    rng: np.random.Generator,
                                    max_redraws: int = 50):
    """Size-matched bootstrap of real member elements with kernel smoothing.

    For each synthetic object, a real member of similar diameter is selected
    (uniformly among the ``cfg.resample_k_neighbors`` nearest members in
    log D), its ``(a, e, sin i)`` adopted, and a Gaussian kernel jitter added
    (Scott's-rule bandwidth per element). This preserves the observed
    multimodality, resonance edges, and element correlations that the
    parametric ``dorb`` model cannot represent, together with the empirical
    size-orbit coupling; the cost is that the member sample's Poisson noise
    and selection effects are inherited rather than smoothed away. Elements
    are clipped to the observed member bounds, as in the ``dorb`` model.

    Returns ``(Orbits, info_dict)``; the info dict is written to provenance.
    """
    elems = np.column_stack([np.asarray(a, float), np.asarray(e, float),
                             np.asarray(sini, float)])
    D = np.asarray(member_D, float)
    good = np.isfinite(D) & (D > 0) & np.isfinite(elems).all(axis=1)
    if good.sum() < 5:
        raise ValueError("resample orbit model needs >= 5 members with diameters")
    elems, logD = elems[good], np.log10(D[good])

    order = np.argsort(logD)
    logD_s, elems_s = logD[order], elems[order]
    n_pool = logD_s.size
    k = int(min(max(cfg.resample_k_neighbors, 1), n_pool))
    # Scott's-rule kernel bandwidth per element
    h = 1.06 * elems_s.std(axis=0, ddof=1) * n_pool ** (-0.2)
    lo_b = elems_s.min(axis=0)
    hi_b = elems_s.max(axis=0)

    # size matching: a window of the k nearest members in log D per object
    n = diameters.size
    pos = np.searchsorted(logD_s, np.log10(diameters))
    win = np.clip(pos - k // 2, 0, n_pool - k)
    base = elems_s[win + rng.integers(0, k, size=n)]

    out = np.full((n, 3), np.nan)
    todo = np.arange(n)
    for _ in range(max_redraws):
        if todo.size == 0:
            break
        cand = base[todo] + rng.normal(size=(todo.size, 3)) * h
        inside = np.ones(todo.size, bool)
        if cfg.family_clip_to_bounds:
            inside &= ((cand >= lo_b) & (cand <= hi_b)).all(axis=1)
        inside &= (cand[:, 1] >= 0) & (cand[:, 1] < 1)
        inside &= (cand[:, 2] >= 0) & (cand[:, 2] < 1)
        out[todo[inside]] = cand[inside]
        todo = todo[~inside]
    if todo.size:                      # stragglers keep their member's elements
        out[todo] = base[todo]

    orbits = Orbits(
        a_au=out[:, 0], e=out[:, 1],
        i_deg=np.degrees(np.arcsin(np.clip(out[:, 2], 0, 1))),
        node_deg=rng.random(n) * 360.0,
        argperi_deg=rng.random(n) * 360.0,
        M_deg=rng.random(n) * 360.0,
    )
    info = {"n_pool": int(n_pool), "k_neighbors": k,
            "kernel_bandwidth": {"a": float(h[0]), "e": float(h[1]),
                                 "sini": float(h[2])}}
    return orbits, info


def generate_family_orbits(cfg: SynthConfig, fit: FamilyFit, diameters: np.ndarray,
                           rng: np.random.Generator,
                           max_redraws: int = 50) -> Orbits:
    """Draw synthetic (a, e, sin i) for each diameter, clipped to family bounds."""
    n = diameters.size
    # predicted standardised radius from the size-orbit relation, + scatter
    log_dorb = (np.log10(fit.dorb_A) + fit.dorb_gamma * np.log10(diameters))
    if cfg.family_clip_to_bounds:
        pass  # bounds applied below

    keys = ("a", "e", "sini")
    out = {k: np.full(n, np.nan) for k in keys}
    todo = np.arange(n)

    for _ in range(max_redraws):
        m = todo.size
        if m == 0:
            break
        radius = 10 ** (log_dorb[todo] + rng.normal(0.0, fit.dorb_scatter, m))
        # random isotropic direction in 3-D standardised element space
        direction = rng.normal(size=(m, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        offset = radius[:, None] * direction  # standardised offsets per element

        cand = {}
        for j, k in enumerate(keys):
            cand[k] = fit.mu[k] + fit.sigma[k] * offset[:, j]

        # accept those within observed bounds (if clipping enabled)
        if cfg.family_clip_to_bounds:
            inside = np.ones(m, bool)
            for k in keys:
                lo, hi = fit.bounds[k]
                inside &= (cand[k] >= lo) & (cand[k] <= hi)
            # additionally keep physical: e in [0,1), sini in [0,1)
            inside &= (cand["e"] >= 0) & (cand["e"] < 1)
            inside &= (cand["sini"] >= 0) & (cand["sini"] < 1)
        else:
            inside = (cand["e"] >= 0) & (cand["e"] < 1) & (cand["sini"] >= 0) & (cand["sini"] < 1)

        acc = todo[inside]
        for k in keys:
            out[k][acc] = cand[k][inside]
        todo = todo[~inside]

    # any stragglers: clamp into bounds rather than leave NaN
    if todo.size:
        for k in keys:
            lo, hi = fit.bounds[k]
            out[k][todo] = np.clip(fit.mu[k], lo, hi)

    i_deg = np.degrees(np.arcsin(np.clip(out["sini"], 0, 1)))
    return Orbits(
        a_au=out["a"], e=out["e"], i_deg=i_deg,
        node_deg=rng.random(n) * 360.0,
        argperi_deg=rng.random(n) * 360.0,
        M_deg=rng.random(n) * 360.0,
    )
