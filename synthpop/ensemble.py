"""Generate an *ensemble* of synthetic populations for systematic-error work.

Simmer turns one synthetic population into a detection efficiency eta(D) with a
*statistical* (Wilson) error bar. But eta(D) also depends on what we assumed
about the population -- its albedo distribution, beaming, orbital spread -- and
those assumptions are themselves fit from a finite set of measured members, so
they carry uncertainty. This module turns that fit uncertainty into a spread of
plausible populations: run each through Simmer and the spread of eta(D) across
them is the *systematic* band.

The widths are not supplied by hand. Each ensemble member **bootstrap-resamples
the population's own measured data** (the NEOWISE albedos/beaming; the family's
proper elements) with replacement and refits, so each member is the population a
differently-sampled-but-equally-valid survey would have inferred. The band
therefore scales automatically with how well-constrained the population is: many
measured members -> tight band, few -> wide band.

Member 0 is the nominal (unresampled) fit; members 1..N-1 are bootstrap draws.
A ``manifest.json`` records every member's catalog path and drawn fit summary, so
the systematic band is fully traceable. Simmer's ``ensemble`` step consumes it.
The manifest also carries the beta contract keys ``pole_beta_convention`` and
``geometry_frame``, taken from the PyLEADER bins all members drew from
(:func:`synthpop.shape_pole.pole_provenance`).

Note: the shape/pole (b/a, beta) distribution comes from a precomputed PyLEADER
joint PDF, not a parametric fit of raw data here, so it is held fixed for now;
its uncertainty lives upstream in PyLEADER. The SFD slope is deliberately *not*
varied -- eta(D) = P(detect | D) is independent of the input size distribution,
so ensemble members should use ``sfd_kind="loguniform"`` (as for any Simmer run).
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict, field
from pathlib import Path
from typing import Optional
import json
import numpy as np

from .config import SynthConfig, FAMILY
from . import physical, orbits_family, io, shape_pole
from .pipeline import prepare_inputs, generate_catalog, PopulationInputs


@dataclass
class EnsembleConfig:
    """Specify an ensemble: a base population + how many bootstrap realizations."""
    base: SynthConfig
    n_members: int = 50
    seed: int = 0
    bootstrap_physical: bool = True     # resample NEOWISE albedo/beaming members
    bootstrap_orbits: bool = True       # resample family proper elements (family mode)
    out_dir: Optional[Path] = None      # default: <base.out_dir>/<base.name>_ensemble

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir) if self.out_dir is not None \
            else Path(self.base.out_dir) / f"{self.base.name}_ensemble"


def _bootstrap_inputs(ecfg: EnsembleConfig, base_gi: PopulationInputs,
                      member: int, rng: np.random.Generator) -> PopulationInputs:
    """Return per-member fits: nominal for member 0, else bootstrap-refit."""
    if member == 0:
        return base_gi                                   # nominal reference

    gi = replace(base_gi)                                # shallow copy of the bundle
    cfg = ecfg.base

    if ecfg.bootstrap_physical:
        n = base_gi.neowise_pV.size
        idx = rng.integers(0, n, n)                      # joint resample keeps pV-eta pairing
        mask = (base_gi.neowise_eta_fitted[idx]
                if base_gi.neowise_eta_fitted is not None else None)
        gi.phys_fit = physical.fit_physical_from_arrays(
            cfg, base_gi.neowise_pV[idx], base_gi.neowise_eta[idx],
            eta_fitted=mask)

    if ecfg.bootstrap_orbits and cfg.mode == FAMILY and base_gi.fam_a is not None:
        m = base_gi.fam_a.size
        j = rng.integers(0, m, m)
        # store the resampled member arrays too: the "resample" orbit model
        # draws from these directly, not from fam_fit
        gi.fam_a = base_gi.fam_a[j]
        gi.fam_e = base_gi.fam_e[j]
        gi.fam_sini = base_gi.fam_sini[j]
        gi.fam_member_D = base_gi.fam_member_D[j]
        gi.fam_fit = orbits_family.fit_family(
            gi.fam_a, gi.fam_e, gi.fam_sini, gi.fam_member_D)
    return gi


def _fit_summary(gi: PopulationInputs) -> dict:
    """Compact record of a member's drawn distribution parameters (for the manifest)."""
    # albedo_model per member: with the auto bimodality rule, a borderline
    # family can legitimately flip models between bootstrap resamples --
    # model-selection uncertainty is then part of the systematic band
    s = {"albedo_model": gi.phys_fit.albedo_model,
         "albedo": asdict(gi.phys_fit.albedo),
         "eta": {"mu": gi.phys_fit.eta.mu, "sigma": gi.phys_fit.eta.sigma}}
    if gi.fam_fit is not None:
        s["family"] = {"sigma": gi.fam_fit.sigma, "dorb_A": gi.fam_fit.dorb_A,
                       "dorb_gamma": gi.fam_fit.dorb_gamma}
    return s


def run_ensemble(ecfg: EnsembleConfig, verbose: bool = True) -> Path:
    """Generate the ensemble; write M catalogs + a manifest. Returns manifest path.

    Each member is an independent, fully-generated synthetic population (different
    orbits and physical draws), ready to be fed through Simmer in parallel.
    """
    def log(msg):
        if verbose:
            print(f"[ensemble:{ecfg.base.name}] {msg}")

    out_dir = Path(ecfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("fitting the population's measured data once")
    base_gi = prepare_inputs(ecfg.base, verbose=False)

    # One independent RNG stream per member, reproducible from ecfg.seed.
    seeds = np.random.SeedSequence(ecfg.seed).spawn(ecfg.n_members)
    members = []
    pole_sources = []
    for m in range(ecfg.n_members):
        boot_rng = np.random.default_rng(seeds[m])          # bootstrap resampling
        gen_rng = np.random.default_rng(seeds[m].spawn(1)[0])  # population draw
        gi = _bootstrap_inputs(ecfg, base_gi, m, boot_rng)

        name = f"{ecfg.base.name}_m{m:03d}"
        mcfg = replace(ecfg.base, name=name, out_dir=out_dir,
                       make_plots=False, seed=ecfg.seed + m)
        df, prov = generate_catalog(mcfg, gi, gen_rng, verbose=False)
        prov["ensemble_member"] = m
        prov["nominal"] = (m == 0)
        pole_sources.extend(prov["shape_pole_bins"])
        cat_path = io.write_catalog(df, mcfg, provenance=prov)
        members.append({"member": m, "nominal": m == 0,
                        "catalog": cat_path.name, "n_objects": int(len(df)),
                        "fit": _fit_summary(gi)})
        log(f"member {m:3d}/{ecfg.n_members} -> {cat_path.name}")

    manifest = {
        "name": ecfg.base.name, "mode": ecfg.base.mode,
        "n_members": ecfg.n_members, "seed": ecfg.seed,
        "bootstrap_physical": ecfg.bootstrap_physical,
        "bootstrap_orbits": ecfg.bootstrap_orbits,
        **shape_pole.pole_provenance(pole_sources),
        "n_neowise_members": int(base_gi.neowise_pV.size),
        "n_family_members": (int(base_gi.fam_a.size)
                             if base_gi.fam_a is not None else None),
        "members": members,
    }
    manifest_path = out_dir / f"{ecfg.base.name}_ensemble_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    log(f"wrote manifest -> {manifest_path.name}")
    return manifest_path
