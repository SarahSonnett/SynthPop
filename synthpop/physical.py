"""Assign visible albedo (pV) and beaming parameter (eta) from NEOWISE fits.

Following Lilly et al. (2021) and Wright et al. (2016), we fit analytic
distributions to the relevant complete NEOWISE subpopulation and MCMC-sample
synthetic values from those fits:

* ``pV`` -- model chosen by population type: a collisional **family** gets a
  single **lognormal** (Gaussian in log-albedo), upgraded automatically to a
  **double lognormal** when the member albedos test bimodal
  (:func:`synthpop.distributions.assess_albedo_bimodality`; e.g. Phocaea,
  Tirela, Juno, Flora, Eos); a **background** population mixes C- and S-type
  taxonomies, so a **double Rayleigh** (Wright+2016). A sigma-clipped linear
  Gaussian is also available. Override via ``cfg.albedo_model``
  ("gaussian" | "lognormal" | "double_lognormal" | "double_rayleigh").
* ``eta`` -- **Gaussian** (truncated to positive). By default fit over ALL
  rows, including the PDS table's quantized ASSUMED beaming values (fit code
  without 'B'; 0.7/0.95/1.0/1.2/1.4), which bias the fit toward eta=1;
  ``cfg.eta_fitted_only`` restricts the fit to measured rows (>=10 required,
  else ValueError) as a sensitivity option -- see README, 'Beaming parameter
  and assumed values'.

We assume no intrinsic dependence between ``pV``, ``eta`` and the other
properties within a subpopulation, so each is sampled independently.

Once ``D`` (from the SFD) and ``pV`` are known, the absolute magnitude follows
from the standard relation ``D = 1329 / sqrt(pV) * 10**(-H/5)``; we compute
``H`` per object for self-consistency, since the downstream NEOWISE-survey
simulator needs it.

The NEOWISE input is the PDS SBN diameters/albedos table (headerless CSV).
Column indices (0-based) match that product:
``H=3, D=11, D_err=12, pV=13, pV_err=14, pIR=15, pIR_err=16, eta=17, eta_err=18``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SynthConfig
from .distributions import (DoubleRayleigh, TruncatedGaussian, LogNormal,
                            DoubleLogNormal, LogNormalMixture,
                            assess_albedo_bimodality)
from .sampling import metropolis

# PDS SBN NEOWISE diameters/albedos column indices (0-based).
NEOWISE_COLS = {"number": 0, "H": 3, "fitcode": 10, "D": 11, "D_err": 12,
                "pV": 13, "pV_err": 14, "pIR": 15, "pIR_err": 16,
                "eta": 17, "eta_err": 18}

H_SUN_CONST = 1329.0  # km; D = H_SUN_CONST / sqrt(pV) * 10**(-H/5)


@dataclass
class PhysicalFit:
    albedo: object          # fitted albedo model (pdf/logpdf/cdf/mode/linear_scale)
    eta: TruncatedGaussian
    n_sample: int
    albedo_model: str = ""            # the model actually fit
    bimodality: dict | None = None    # assess_albedo_bimodality stats (family mode)


def diameter_to_H(D_km: np.ndarray, pV: np.ndarray) -> np.ndarray:
    """Absolute magnitude from diameter (km) and visible albedo."""
    return 5.0 * np.log10(H_SUN_CONST / (np.asarray(D_km) * np.sqrt(np.asarray(pV))))


def load_neowise(cfg: SynthConfig):
    """Read pV, eta, and the eta-was-fit flag (and numbers) from the NEOWISE CSV.

    Returns ``(numbers, pV, eta, eta_fitted)``: rows with non-physical/blank
    entries removed and, if ``cfg.neowise_member_numbers`` is given, restricted
    to that membership. ``eta_fitted`` is True where the PDS fit code contains
    ``'B'`` (beaming actually fit rather than assumed).
    """
    c = NEOWISE_COLS
    raw = np.genfromtxt(cfg.neowise_file, delimiter=",",
                        usecols=(c["number"], c["pV"], c["eta"], c["fitcode"]),
                        dtype=str, invalid_raise=False)
    number = raw[:, 0].astype(float)
    pV = raw[:, 1].astype(float)
    eta = raw[:, 2].astype(float)
    code = np.char.replace(np.char.strip(raw[:, 3]), '"', "")
    eta_fitted = np.char.find(code, "B") >= 0
    ok = np.isfinite(pV) & np.isfinite(eta) & (pV > 0) & (eta > 0)
    number, pV, eta, eta_fitted = number[ok], pV[ok], eta[ok], eta_fitted[ok]

    if cfg.neowise_member_numbers is not None:
        members = set(int(x) for x in cfg.neowise_member_numbers)
        keep = np.array([int(nb) in members for nb in number])
        number, pV, eta, eta_fitted = (number[keep], pV[keep], eta[keep],
                                       eta_fitted[keep])
    return number, pV, eta, eta_fitted


def load_neowise_full(cfg: SynthConfig):
    """Read (number, D, pV, eta) for every NEOWISE row, unfiltered by membership.

    Used to build 'real member' overlay distributions for the diagnostic plots
    (cross-matched against a family's member numbers by the caller).
    """
    c = NEOWISE_COLS
    arr = np.genfromtxt(cfg.neowise_file, delimiter=",",
                        usecols=(c["number"], c["D"], c["pV"], c["eta"]),
                        dtype=float, invalid_raise=False)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


ALBEDO_MODELS = ("gaussian", "lognormal", "double_lognormal", "double_rayleigh",
                 "mixture")


def resolve_albedo_model(cfg: SynthConfig) -> str:
    """The albedo model to use: explicit config value, else auto by mode.

    For family mode the auto default is ``lognormal``, upgraded to
    ``double_lognormal`` at fit time when the member albedos test bimodal
    (:func:`synthpop.distributions.assess_albedo_bimodality`); the model
    actually fit is recorded in ``PhysicalFit.albedo_model``.
    """
    if cfg.albedo_model:
        model = cfg.albedo_model.replace("-", "_").lower()
        if model not in ALBEDO_MODELS:
            raise ValueError(f"unknown albedo_model {cfg.albedo_model!r}")
        return model
    from .config import FAMILY
    return "lognormal" if cfg.mode == FAMILY else "double_rayleigh"


def fit_physical(cfg: SynthConfig) -> PhysicalFit:
    _, pV, eta, eta_fitted = load_neowise(cfg)
    return fit_physical_from_arrays(cfg, pV, eta, eta_fitted=eta_fitted)


def fit_physical_from_arrays(cfg: SynthConfig, pV, eta,
                             eta_fitted=None) -> PhysicalFit:
    """Fit the albedo + beaming distributions to given pV/eta arrays.

    Split out from :func:`fit_physical` so an ensemble can pass bootstrap-
    resampled members (see :mod:`synthpop.ensemble`) and get the fit that a
    differently-sampled version of the same population would have produced.
    """
    from .config import FAMILY
    pV = np.asarray(pV, float)
    eta = np.asarray(eta, float)
    model = resolve_albedo_model(cfg)

    # Family mode: test the member albedos for bimodality (some families --
    # e.g. Phocaea, Flora -- carry two albedo components). With the auto
    # default, a bimodal sample upgrades the model to the double lognormal.
    bimod = None
    if cfg.mode == FAMILY and pV.size >= 30:
        bimod = assess_albedo_bimodality(pV)
        if cfg.albedo_model is None and bimod["use_mixture"]:
            model = "double_lognormal"

    # DEFAULT PATH: 3-sigma clipping (in log-albedo) before the fit. This is
    # a deliberate *membership* cut, not just numerical hygiene: HCM family
    # lists carry interlopers from nearby families and the background, and in
    # bright families those contaminants form a short dark tail 3-4 sigma
    # below the log-mean (e.g. Vesta, Eunomia, Eos, Baptistina). The clip
    # excludes that tail so the synthetic family models the family, not its
    # contamination. To *include* the dark tail instead, request
    # albedo_model="mixture": a BIC-selected k-component lognormal mixture
    # fit on the UNCLIPPED sample, in which a genuine few-percent dark group
    # receives its own component. (If used for debiasing, the detected-count
    # side of the ratio should then keep its contaminants too.)
    if model == "gaussian":
        albedo = TruncatedGaussian.fit(pV, lo=0.0, clip_sigma=3.0)
    elif model == "lognormal":
        albedo = LogNormal.fit(pV, clip_sigma=3.0)
    elif model == "double_lognormal":
        albedo = (bimod["mixture"] if bimod and bimod["mixture"] is not None
                  else DoubleLogNormal.fit(pV))
    elif model == "mixture":
        albedo = LogNormalMixture.fit_bic(pV)
    else:
        albedo = DoubleRayleigh.fit(pV)
    bimod_stats = ({k: v for k, v in bimod.items() if k != "mixture"}
                   if bimod else None)
    # eta fit sample: optionally only rows where beaming was actually fit
    # (fit code 'B'), excluding the quantized assumed values (sensitivity
    # option cfg.eta_fitted_only; default keeps all rows)
    eta_sample = np.asarray(eta, float)
    if cfg.eta_fitted_only:
        if eta_fitted is None:
            raise ValueError("eta_fitted_only=True requires the eta_fitted mask")
        eta_sample = eta_sample[np.asarray(eta_fitted, bool)]
        if eta_sample.size < 10:
            raise ValueError(
                f"only {eta_sample.size} fitted-eta rows; cannot fit beaming")
    fit = PhysicalFit(
        albedo=albedo,
        eta=TruncatedGaussian.fit(eta_sample, lo=0.0),
        n_sample=int(pV.size),
        albedo_model=model,
        bimodality=bimod_stats,
    )
    fit.n_eta_fit = int(eta_sample.size)
    return fit


@dataclass
class Physical:
    pV: np.ndarray
    eta: np.ndarray
    H: np.ndarray


def assign_physical(cfg: SynthConfig, fit: PhysicalFit, diameters: np.ndarray,
                    rng: np.random.Generator) -> Physical:
    """MCMC-sample pV and eta from the fits; derive H from D and pV."""
    n = cfg.n_objects

    # Truncate at cfg.albedo_min: albedos below ~1% have never been measured,
    # so the fitted Rayleigh tail there is unphysical extrapolation.
    pv_floor = cfg.albedo_min
    if hasattr(fit.albedo, "sample_truncated"):
        # exact sampler (mixture models): a random-walk chain mixes poorly
        # between well-separated modes, mis-weighting the components
        pV = fit.albedo.sample_truncated(n, rng, lo=pv_floor, hi=1.0)
    else:
        pv_scale = fit.albedo.linear_scale  # per-model linear-space step size
        pV = metropolis(
            log_pdf=lambda x: float(fit.albedo.logpdf(x)),
            n=n, x0=max(fit.albedo.mode, pv_floor),
            proposal_scale=max(pv_scale, 1e-3),
            rng=rng, burn=cfg.mcmc_burn, thin=cfg.mcmc_thin, bounds=(pv_floor, 1.0),
        )
    eta = metropolis(
        log_pdf=lambda x: float(fit.eta.logpdf(x)),
        n=n, x0=fit.eta.mode, proposal_scale=max(fit.eta.sigma, 1e-3),
        rng=rng, burn=cfg.mcmc_burn, thin=cfg.mcmc_thin, bounds=(1e-3, np.inf),
    )

    H = diameter_to_H(diameters, pV)
    return Physical(pV=pV, eta=eta, H=H)
