"""``run_synthpop`` -- the end-to-end driver, mirroring PyLEADER's ``run_analysis``.

Stages, in order:

1. **Size** -- draw diameters from the SFD (:mod:`synthpop.sfd`).
2. **Shape & pole** -- draw b/a, beta, lambda from the PyLEADER distributions
   for each object's diameter bin (:mod:`synthpop.shape_pole`).
3. **Orbits** -- family generation (:mod:`synthpop.orbits_family`: Gaussian
   D_orb model or size-matched member resampling) or s3m
   bootstrap (:mod:`synthpop.orbits_bg`), per ``cfg.mode``.
4. **Physical** -- sample pV and eta from the population's fitted NEOWISE
   distributions; derive H (:mod:`synthpop.physical`).
5. **Assemble & write** -- join into one catalog and write to disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

from .config import SynthConfig, FAMILY, BACKGROUND
from . import sfd, shape_pole, orbits_family, orbits_bg, physical, assemble, io


@dataclass
class PopulationInputs:
    """Data-derived fits + the raw member data they were fit from.

    Everything needed to *generate* a synthetic population, computed once from
    the population's measured data. The raw arrays are retained so an ensemble
    can bootstrap-resample them and refit (:mod:`synthpop.ensemble`), turning
    fit uncertainty into a spread of plausible populations.
    """
    phys_fit: physical.PhysicalFit
    neowise_pV: np.ndarray            # measured albedos the albedo/eta fit used
    neowise_eta: np.ndarray
    neowise_eta_fitted: Optional[np.ndarray] = None  # PDS fit-code 'B' mask
    fam_fit: Optional[orbits_family.FamilyFit] = None
    # family member data (for bootstrap); None in background mode
    fam_a: Optional[np.ndarray] = None
    fam_e: Optional[np.ndarray] = None
    fam_sini: Optional[np.ndarray] = None
    fam_member_D: Optional[np.ndarray] = None
    # overlays / provenance carried through for plots (single-run only)
    real_elements: Optional[dict] = None
    real_phys: Optional[dict] = None
    prov_extra: dict = field(default_factory=dict)


def _family_member_diameters(cfg, number, H, pV_fit):
    """Diameters of real family members: cross-match to NEOWISE, else H->albedo.

    Members present in the NEOWISE catalog use the measured diameter; the rest
    fall back to the diameter implied by H and the fitted median albedo.
    """
    cat_num, cat_pV, _, _ = physical.load_neowise(cfg)
    # NEOWISE diameter for matched members
    c = physical.NEOWISE_COLS
    arr = np.genfromtxt(cfg.neowise_file, delimiter=",",
                        usecols=(c["number"], c["D"]), dtype=float, invalid_raise=False)
    num_to_D = {int(nb): d for nb, d in arr if np.isfinite(d) and d > 0}

    median_pV = float(np.median(cat_pV)) if cat_pV.size else 0.1
    D = np.empty(number.size)
    for k, (nb, h) in enumerate(zip(number, H)):
        try:
            num = int(nb)
        except (TypeError, ValueError):
            num = None  # provisional designation -> no NEOWISE number match
        if num is not None and num in num_to_D:
            D[k] = num_to_D[num]
        elif h > -9.0:
            D[k] = physical.H_SUN_CONST / np.sqrt(median_pV) * 10 ** (-h / 5.0)
        else:
            # AFP files flag a missing H as -9.99; with no NEOWISE match either,
            # the diameter is unknown -> NaN (fit_family drops non-finite D).
            D[k] = np.nan
    return D


def _family_real_physical(cfg, number, H):
    """Real family members' measured D/pV/eta (NEOWISE cross-match) and H.

    Returns a dict of arrays for the plot overlays: {"D", "pV", "eta", "H"}.
    Only numbered members present in NEOWISE contribute to D/pV/eta; H comes
    from the family file itself (missing-H sentinels excluded).
    """
    cat_num, cat_D, cat_pV, cat_eta = physical.load_neowise_full(cfg)
    ok = np.isfinite(cat_num)
    lookup = {}
    for nb, d, p, e in zip(cat_num[ok], cat_D[ok], cat_pV[ok], cat_eta[ok]):
        lookup.setdefault(int(nb), (d, p, e))
    D, pV, eta = [], [], []
    for nb in number:
        try:
            hit = lookup.get(int(nb))
        except (TypeError, ValueError):
            hit = None  # provisional designation
        if hit is not None:
            D.append(hit[0]); pV.append(hit[1]); eta.append(hit[2])
    H = np.asarray(H, float)
    return {"D": np.array(D), "pV": np.array(pV), "eta": np.array(eta),
            "H": H[H > -9.0]}


def prepare_inputs(cfg: SynthConfig, verbose: bool = True) -> PopulationInputs:
    """Load the population's measured data and fit its property distributions.

    Data-only; independent of any random draw, so it runs *once* for a whole
    ensemble (:mod:`synthpop.ensemble`).
    """
    def log(msg):
        if verbose:
            print(f"[synthpop:{cfg.name}] {msg}")

    _, pV, eta, eta_fitted = physical.load_neowise(cfg)
    phys_fit = physical.fit_physical_from_arrays(cfg, pV, eta,
                                                 eta_fitted=eta_fitted)
    log(f"fit NEOWISE albedo ({phys_fit.albedo_model}) + eta"
        + (f" [bimodality: dBIC={phys_fit.bimodality['delta_bic']:.1f}, "
           f"D={phys_fit.bimodality['ashman_d']:.2f}, "
           f"w_minor={phys_fit.bimodality['w_minor']:.2f}]"
           if phys_fit.bimodality else ""))

    gi = PopulationInputs(phys_fit=phys_fit, neowise_pV=pV, neowise_eta=eta,
                          neowise_eta_fitted=eta_fitted)

    if cfg.mode == FAMILY:
        log("fitting family center + D_orb(D)")
        number, H, a, e, sini = orbits_family.load_family_members(cfg)
        member_D = _family_member_diameters(cfg, number, H, None)
        gi.fam_fit = orbits_family.fit_family(a, e, sini, member_D)
        gi.fam_a, gi.fam_e, gi.fam_sini, gi.fam_member_D = a, e, sini, member_D
        gi.real_elements = {"a": a, "e": e, "sini": sini}
        gi.real_phys = _family_real_physical(cfg, number, H)
        gi.prov_extra["family_fit"] = {
            "mu": gi.fam_fit.mu, "sigma": gi.fam_fit.sigma, "bounds": gi.fam_fit.bounds,
            "dorb_A": gi.fam_fit.dorb_A, "dorb_gamma": gi.fam_fit.dorb_gamma,
            "dorb_scatter": gi.fam_fit.dorb_scatter, "n_members": gi.fam_fit.n_members}
    elif cfg.mode != BACKGROUND:
        raise ValueError(cfg.mode)
    return gi


def generate_catalog(cfg: SynthConfig, gi: PopulationInputs,
                     rng: np.random.Generator, verbose: bool = True):
    """Draw one synthetic population from prepared fits. Returns ``(df, prov)``."""
    from dataclasses import asdict

    def log(msg):
        if verbose:
            print(f"[synthpop:{cfg.name}] {msg}")

    # Independent per-stage RNG streams, spawned deterministically from the
    # caller's generator before any draws: runs that differ in one stage's
    # configuration (e.g. the family orbit model) still produce IDENTICAL
    # draws in every other stage, so such runs form matched pairs for
    # downstream systematic comparisons.
    rng_sfd, rng_shape, rng_orbit, rng_phys = rng.spawn(4)

    log("sampling diameters from the SFD")
    diameters = sfd.sample_diameters(cfg, rng_sfd)

    log("assigning shape (b/a) and pole (beta, lambda) from PyLEADER")
    sp = shape_pole.assign_shape_pole(cfg, diameters, rng_shape)

    resample_info = None
    if cfg.mode == FAMILY:
        if cfg.family_orbit_model == "resample":
            log("generating orbits by size-matched resampling of real members")
            orbits, resample_info = orbits_family.generate_family_orbits_resample(
                cfg, gi.fam_a, gi.fam_e, gi.fam_sini, gi.fam_member_D,
                diameters, rng_orbit)
        elif cfg.family_orbit_model == "dorb":
            log("generating family orbits from the D_orb(D) model")
            orbits = orbits_family.generate_family_orbits(cfg, gi.fam_fit, diameters, rng_orbit)
        else:
            raise ValueError(f"unknown family_orbit_model {cfg.family_orbit_model!r}")
    else:
        log("bootstrapping orbits from the s3m model")
        orbits = orbits_bg.generate_background_orbits(cfg, rng_orbit)

    log("MCMC-sampling pV + eta")
    phys = physical.assign_physical(cfg, gi.phys_fit, diameters, rng_phys)

    df = assemble.build_catalog(cfg.name, diameters, sp, orbits, phys)
    prov = {"name": cfg.name, "mode": cfg.mode, "n_objects": cfg.n_objects,
            "seed": cfg.seed,
            # beta contract keys, copied from the PyLEADER products
            **shape_pole.pole_provenance(sp.sources),
            "shape_pole_bins": sp.sources,
            **({"family_orbit_model": cfg.family_orbit_model,
                **({"resample": resample_info} if resample_info else {})}
               if cfg.mode == FAMILY else {}),
            **gi.prov_extra,
            "physical_fit": {
                "albedo_model": gi.phys_fit.albedo_model,
                "albedo": asdict(gi.phys_fit.albedo),
                **({"bimodality": gi.phys_fit.bimodality}
                   if gi.phys_fit.bimodality else {}),
                "eta": {"mu": gi.phys_fit.eta.mu, "sigma": gi.phys_fit.eta.sigma,
                        "fitted_only": cfg.eta_fitted_only,
                        "n_eta_fit": getattr(gi.phys_fit, "n_eta_fit", None)},
                "n_neowise": gi.phys_fit.n_sample}}
    return df, prov


def run_synthpop(cfg: SynthConfig, verbose: bool = True) -> Path:
    def log(msg):
        if verbose:
            print(f"[synthpop:{cfg.name}] {msg}")

    gi = prepare_inputs(cfg, verbose=verbose)
    rng = np.random.default_rng(cfg.seed)
    df, prov = generate_catalog(cfg, gi, rng, verbose=verbose)
    out = io.write_catalog(df, cfg, provenance=prov)
    log(f"wrote {len(df)} objects -> {out}")

    if cfg.make_plots:
        from . import plotting  # lazy: pulls in matplotlib only when needed
        model_marginals = shape_pole.mixture_marginals(cfg, df["diam_km"].to_numpy())
        paths = plotting.make_all_plots(df, cfg, real_elements=gi.real_elements,
                                        real_phys=gi.real_phys,
                                        model_marginals=model_marginals,
                                        albedo_fit=gi.phys_fit.albedo,
                                        eta_fit=gi.phys_fit.eta)
        log("wrote plots: " + ", ".join(p.name for p in paths))
    return out
