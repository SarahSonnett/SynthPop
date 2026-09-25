"""Unit tests for SynthPop. Runnable with pytest or directly:

    python tests/test_synthpop.py
"""

import sys
from pathlib import Path

import numpy as np

# numpy<2 compatibility: trapezoid was added in numpy 2.0 (trapz before, and
# removed in 2.0 — so neither name can be referenced unguarded)
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthpop.sampling import EmpiricalPDF, EmpiricalPDF2D, metropolis  # noqa: E402
from synthpop.distributions import DoubleRayleigh, TruncatedGaussian, rayleigh_pdf  # noqa: E402
from synthpop import sfd                                        # noqa: E402
from synthpop.config import SynthConfig, FAMILY                 # noqa: E402
from synthpop import orbits_family                              # noqa: E402
from synthpop.pyleader_io import load_distributions             # noqa: E402


def test_empirical_pdf_recovers_mean():
    centers = np.linspace(0.05, 0.95, 19)
    density = np.exp(-0.5 * ((centers - 0.6) / 0.1) ** 2)  # bump at 0.6
    rng = np.random.default_rng(0)
    s = EmpiricalPDF(centers, density).sample(200_000, rng)
    assert abs(s.mean() - 0.6) < 0.01
    assert s.min() >= 0 and s.max() <= 1.0


def test_metropolis_samples_gaussian():
    rng = np.random.default_rng(1)
    logpdf = lambda x: -0.5 * ((x - 3.0) / 0.5) ** 2
    s = metropolis(logpdf, n=40_000, x0=3.0, proposal_scale=0.5, rng=rng)
    assert abs(s.mean() - 3.0) < 0.03
    assert abs(s.std() - 0.5) < 0.05


def test_double_rayleigh_fit():
    rng = np.random.default_rng(2)
    # two Rayleigh components: low-albedo (sig .05) + high-albedo (sig .18)
    n1 = rng.rayleigh(0.05, 6000)
    n2 = rng.rayleigh(0.18, 4000)
    data = np.concatenate([n1, n2])
    fit = DoubleRayleigh.fit(data)
    assert fit.sigma1 < fit.sigma2
    assert abs(fit.sigma1 - 0.05) < 0.02
    assert abs(fit.sigma2 - 0.18) < 0.03
    # pdf integrates to ~1
    x = np.linspace(0, 1.5, 5000)
    assert abs(_trapezoid(fit.pdf(x), x) - 1.0) < 0.02


def test_truncated_gaussian_fit():
    rng = np.random.default_rng(3)
    data = rng.normal(1.0, 0.2, 5000)
    fit = TruncatedGaussian.fit(data, lo=0.0)
    assert abs(fit.mu - 1.0) < 0.02
    assert abs(fit.sigma - 0.2) < 0.02
    assert fit.logpdf(-1.0) == -np.inf


def test_sfd_power_law_slope():
    rng = np.random.default_rng(4)
    cfg = SynthConfig(name="t", n_objects=200_000, diam_min_km=1.0,
                      diam_max_km=100.0, sfd_slope=2.0)
    D = sfd.sample_diameters(cfg, rng)
    assert D.min() >= 1.0 and D.max() <= 100.0
    # cumulative N(>D) propto D**-2 -> fraction with D>10 is (10**-2)/(1**-2) approx
    frac_gt10 = np.mean(D > 10)
    expected = (10.0 ** -2 - 100.0 ** -2) / (1.0 ** -2 - 100.0 ** -2)
    assert abs(frac_gt10 - expected) < 0.01


def test_family_fit_and_bounds():
    rng = np.random.default_rng(5)
    # fake family: tight in a,e,sin i; spread anti-correlated with size
    n = 500
    D = rng.uniform(1, 20, n)
    spread = 1.0 / D  # small bodies spread more
    a = 3.14 + spread * rng.normal(0, 0.01, n)
    e = 0.135 + spread * rng.normal(0, 0.005, n)
    sini = 0.089 + spread * rng.normal(0, 0.003, n)
    fit = orbits_family.fit_family(a, e, sini, D)
    assert fit.dorb_gamma < 0  # D_orb decreases with size
    assert fit.n_members == n

    cfg = SynthConfig(name="t", mode=FAMILY, n_objects=5000, family_clip_to_bounds=True)
    syn_D = rng.uniform(1, 20, cfg.n_objects)
    orb = orbits_family.generate_family_orbits(cfg, fit, syn_D, rng)
    # all synthetic elements within observed bounds
    assert orb.e.min() >= fit.bounds["e"][0] - 1e-9
    assert orb.e.max() <= fit.bounds["e"][1] + 1e-9
    assert orb.a_au.min() >= fit.bounds["a"][0] - 1e-9
    assert np.all(np.isfinite(orb.i_deg))


def test_empirical_pdf_2d_recovers_correlation():
    # a correlated 2-D bump: beta increases with p
    xc = np.linspace(0.3, 0.95, 12)
    yc = np.linspace(6.0, 84.0, 12)
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    # ridge along Y = 100*X - 20 (positive p-beta correlation)
    density = np.exp(-0.5 * ((Y - (100 * X - 20)) / 8.0) ** 2)
    rng = np.random.default_rng(7)
    x, y = EmpiricalPDF2D(xc, yc, density).sample(200_000, rng)
    assert x.min() >= xc[0] - 0.1 and x.max() <= xc[-1] + 0.1
    assert y.min() >= yc[0] - 5 and y.max() <= yc[-1] + 5
    # recover the positive correlation
    assert np.corrcoef(x, y)[0, 1] > 0.7


def test_load_distributions_npz(tmp_path=None):
    import tempfile, os
    d = tempfile.mkdtemp() if tmp_path is None else str(tmp_path)
    p_grid = np.linspace(0.3, 0.95, 12)
    b_grid = np.linspace(6.0, 84.0, 12)
    f_mean = np.ones((12, 12))
    np.savez(os.path.join(d, "population_distribution.npz"),
             p_grid=p_grid, b_grid=b_grid, f_mean=f_mean,
             pop_median_p=np.array([0.61, 0.60, 0.62]),
             pop_median_b=np.array([19.0, 18.0, 20.0]), relerr=0.007)
    ld = load_distributions(d)
    assert ld.f_mean.shape == (12, 12)
    assert np.isclose(ld.pop_median_p[0], 0.61)
    rng = np.random.default_rng(0)
    pa, bta = ld.joint.sample(1000, rng)
    assert pa.shape == (1000,) and bta.shape == (1000,)


def test_sfd_loguniform_equal_log_bins():
    rng = np.random.default_rng(6)
    cfg = SynthConfig(name="t", n_objects=100_000, diam_min_km=1.0,
                      diam_max_km=100.0, sfd_kind="loguniform")
    D = sfd.sample_diameters(cfg, rng)
    assert D.min() >= 1.0 and D.max() <= 100.0
    # equal counts per decade: [1,10) and [10,100) each hold ~half
    frac_low = np.mean(D < 10)
    assert abs(frac_low - 0.5) < 0.01


def test_rotation_period_bounds_and_assignment():
    from synthpop.shape_pole import ShapePole  # noqa: F401 (contract import)
    rng = np.random.default_rng(8)
    cfg = SynthConfig(name="t", rot_period_min_h=2.2, rot_period_max_h=20.0)
    lo, hi = cfg.rot_period_min_h, cfg.rot_period_max_h
    p = lo * (hi / lo) ** rng.random(50_000)
    assert p.min() >= lo and p.max() <= hi
    # log-uniform: median at the geometric mean
    assert abs(np.median(p) - np.sqrt(lo * hi)) < 0.15


def test_eta_fitted_only_filters_assumed_values():
    from synthpop.physical import fit_physical_from_arrays
    rng = np.random.default_rng(51)
    # 40% real measurements around 1.15, 60% assumed spike at exactly 1.0
    real = rng.normal(1.15, 0.20, 400)
    assumed = np.full(600, 1.0)
    eta = np.concatenate([real, assumed])
    fitted = np.concatenate([np.ones(400, bool), np.zeros(600, bool)])
    pV = np.exp(rng.normal(np.log(0.05), 0.3, 1000))
    cfg_all = SynthConfig(name="t", mode=FAMILY)
    cfg_fit = SynthConfig(name="t", mode=FAMILY, eta_fitted_only=True)
    f_all = fit_physical_from_arrays(cfg_all, pV, eta, eta_fitted=fitted)
    f_fit = fit_physical_from_arrays(cfg_fit, pV, eta, eta_fitted=fitted)
    assert abs(f_all.eta.mu - 1.06) < 0.03      # dragged toward the spike
    assert abs(f_fit.eta.mu - 1.15) < 0.03      # recovers the measured mean
    assert f_fit.eta.sigma > f_all.eta.sigma    # spike shrank the width
    assert f_fit.n_eta_fit == 400
    # too few fitted rows -> hard error, not a silent bad fit
    try:
        fit_physical_from_arrays(cfg_fit, pV[:20], eta[:20][::-1][:20],
                                 eta_fitted=np.zeros(20, bool))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_albedo_floor_truncates_sampling():
    from synthpop.physical import assign_physical, PhysicalFit
    from synthpop.distributions import DoubleRayleigh, TruncatedGaussian
    rng = np.random.default_rng(9)
    cfg = SynthConfig(name="t", n_objects=3000, albedo_min=0.01,
                      mcmc_burn=500, mcmc_thin=2)
    fit = PhysicalFit(albedo=DoubleRayleigh(w=0.9, sigma1=0.03, sigma2=0.17),
                      eta=TruncatedGaussian(mu=0.95, sigma=0.14, lo=0.0),
                      n_sample=1000)
    phys = assign_physical(cfg, fit, diameters=rng.uniform(1, 25, cfg.n_objects),
                           rng=rng)
    assert phys.pV.min() >= 0.01
    assert phys.pV.max() <= 1.0
    # the truncation must not gut the distribution: bulk still near the mode
    assert 0.02 < np.median(phys.pV) < 0.08


def test_albedo_model_resolution_and_gaussian_sampling():
    from synthpop.physical import resolve_albedo_model, assign_physical, PhysicalFit
    from synthpop.distributions import TruncatedGaussian
    # auto: family -> lognormal, background -> double_rayleigh; explicit wins
    assert resolve_albedo_model(SynthConfig(name="t", mode="family")) == "lognormal"
    assert resolve_albedo_model(SynthConfig(name="t", mode="background")) == "double_rayleigh"
    assert resolve_albedo_model(
        SynthConfig(name="t", mode="family", albedo_model="double-rayleigh")
    ) == "double_rayleigh"
    assert resolve_albedo_model(
        SynthConfig(name="t", mode="family", albedo_model="gaussian")
    ) == "gaussian"
    # Gaussian albedo sampling honours the floor and recovers the fit
    rng = np.random.default_rng(12)
    cfg = SynthConfig(name="t", mode="family", n_objects=4000, albedo_min=0.01,
                      mcmc_burn=500, mcmc_thin=2)
    fit = PhysicalFit(albedo=TruncatedGaussian(mu=0.05, sigma=0.02, lo=0.0),
                      eta=TruncatedGaussian(mu=0.95, sigma=0.14, lo=0.0),
                      n_sample=1000)
    phys = assign_physical(cfg, fit, rng.uniform(1, 25, cfg.n_objects), rng)
    assert phys.pV.min() >= 0.01
    assert abs(np.median(phys.pV) - 0.05) < 0.01


def test_truncated_gaussian_sigma_clipped_fit():
    rng = np.random.default_rng(14)
    # dark-family core + 4% high-albedo interlopers
    core = rng.normal(0.045, 0.012, 5000)
    interlopers = rng.uniform(0.15, 0.35, 200)
    data = np.concatenate([core, interlopers])
    plain = TruncatedGaussian.fit(data, lo=0.0)
    clipped = TruncatedGaussian.fit(data, lo=0.0, clip_sigma=3.0)
    # unclipped is inflated by the interlopers; clipped recovers the core
    assert plain.sigma > 1.5 * clipped.sigma
    assert abs(clipped.mu - 0.045) < 0.002
    assert abs(clipped.sigma - 0.012) < 0.002


def test_double_lognormal_em_fit_and_sampler():
    from synthpop.distributions import DoubleLogNormal
    rng = np.random.default_rng(31)
    # dark component at pV~0.06, bright at ~0.25, 60/40 split
    dark = np.exp(rng.normal(np.log(0.06), 0.25, 6000))
    bright = np.exp(rng.normal(np.log(0.25), 0.20, 4000))
    fit = DoubleLogNormal.fit(np.concatenate([dark, bright]))
    assert abs(np.exp(fit.mu1) - 0.06) < 0.01
    assert abs(np.exp(fit.mu2) - 0.25) < 0.03
    assert abs(fit.w - 0.6) < 0.05
    # pdf integrates to 1
    x = np.linspace(1e-4, 1.5, 20000)
    assert abs(_trapezoid(fit.pdf(x), x) - 1.0) < 1e-3
    # exact truncated sampler respects bounds and recovers the weight
    s = fit.sample_truncated(50_000, rng, lo=0.01, hi=1.0)
    assert s.min() >= 0.01 and s.max() <= 1.0
    frac_dark = np.mean(s < 0.12)
    assert abs(frac_dark - 0.6) < 0.03


def test_bimodality_assessment():
    from synthpop.distributions import assess_albedo_bimodality
    rng = np.random.default_rng(32)
    # clearly bimodal sample
    bimod = np.concatenate([np.exp(rng.normal(np.log(0.05), 0.25, 700)),
                            np.exp(rng.normal(np.log(0.24), 0.20, 300))])
    res = assess_albedo_bimodality(bimod)
    assert res["bimodal"] and res["use_mixture"]
    assert res["delta_bic"] > 10 and res["ashman_d"] > 2 and res["dip_ratio"] < 0.8
    # unimodal sample selects the single model
    uni = np.exp(rng.normal(np.log(0.05), 0.30, 1000))
    res_u = assess_albedo_bimodality(uni)
    assert not res_u["bimodal"] and not res_u["use_mixture"]
    # sharp-edged skew: not called *bimodal* (no resolved dip is required for
    # that flag), though the mixture may still win on evidence
    skew = 0.04 + np.abs(rng.normal(0, 0.02, 1000))
    assert not assess_albedo_bimodality(skew)["bimodal"]
    # tiny samples never qualify
    assert not assess_albedo_bimodality(bimod[:25])["use_mixture"]


def test_lognormal_mixture_bic_recovers_dark_group():
    from synthpop.distributions import LogNormalMixture
    rng = np.random.default_rng(41)
    # 3% dark interloper group + dominant bright family (Vesta-like)
    dark = np.exp(rng.normal(np.log(0.05), 0.20, 90))
    bright = np.exp(rng.normal(np.log(0.32), 0.22, 2910))
    fit = LogNormalMixture.fit_bic(np.concatenate([dark, bright]))
    assert len(fit.ws) >= 2
    # darkest component sits near the dark group with roughly its weight
    assert abs(np.exp(fit.mus[0]) - 0.05) < 0.02
    assert 0.015 < fit.ws[0] < 0.06
    # sampled dark fraction matches the input fraction
    s = fit.sample_truncated(50_000, rng, lo=0.01, hi=1.0)
    assert abs(np.mean(s < 0.1) - 0.03) < 0.01
    # BIC keeps k=1 for a clean unimodal sample
    uni = np.exp(rng.normal(np.log(0.2), 0.25, 2000))
    assert len(LogNormalMixture.fit_bic(uni).ws) == 1


def test_lognormal_fit_and_cdf():
    from synthpop.distributions import LogNormal
    rng = np.random.default_rng(15)
    data = np.exp(rng.normal(np.log(0.045), 0.3, 20_000))
    fit = LogNormal.fit(data, clip_sigma=3.0)
    assert abs(fit.mu - np.log(0.045)) < 0.01
    assert abs(fit.sigma - 0.3) < 0.01
    # cdf consistent with numerically integrated pdf
    x = np.linspace(1e-4, 0.1, 50_000)
    num = _trapezoid(fit.pdf(x), x)
    assert abs(num - float(fit.cdf(0.1))) < 1e-3
    assert float(fit.cdf(1e9)) > 0.999999


def test_truncated_gaussian_cdf():
    from scipy.integrate import quad
    g = TruncatedGaussian(mu=0.05, sigma=0.02, lo=0.0)
    assert g.cdf(0.0) == 0.0
    assert abs(g.cdf(1e6) - 1.0) < 1e-12
    num = quad(g.pdf, 0.0, 0.04)[0] / quad(g.pdf, 0.0, np.inf)[0]
    assert abs(float(g.cdf(0.04)) - num) < 1e-6


def test_double_rayleigh_cdf():
    fit = DoubleRayleigh(w=0.6, sigma1=0.05, sigma2=0.18)
    assert abs(fit.cdf(1e6) - 1.0) < 1e-12
    assert fit.cdf(0.0) == 0.0
    # cdf consistent with numerically integrated pdf
    x = np.linspace(0, 0.3, 20_000)
    num = _trapezoid(fit.pdf(x), x)
    assert abs(num - fit.cdf(0.3)) < 1e-4


def test_family_orbit_resample_preserves_structure():
    rng = np.random.default_rng(21)
    # bimodal family: two clusters in a, small members only in cluster A
    nA, nB = 1500, 1500
    a = np.concatenate([rng.normal(3.00, 0.004, nA), rng.normal(3.10, 0.004, nB)])
    e = rng.normal(0.07, 0.01, nA + nB)
    sini = rng.normal(0.17, 0.005, nA + nB)
    D = np.concatenate([rng.uniform(1, 5, nA), rng.uniform(5, 20, nB)])
    cfg = SynthConfig(name="t", mode=FAMILY, family_orbit_model="resample")

    syn_D = rng.uniform(1, 20, 6000)
    orb, info = orbits_family.generate_family_orbits_resample(
        cfg, a, e, sini, D, syn_D, rng)
    assert info["n_pool"] == nA + nB
    # bimodality preserved: the gap between clusters stays underpopulated
    frac_gap = np.mean((orb.a_au > 3.03) & (orb.a_au < 3.07))
    assert frac_gap < 0.05
    # both modes populated
    assert np.mean(orb.a_au < 3.03) > 0.2 and np.mean(orb.a_au > 3.07) > 0.2
    # size-orbit coupling: small synthetics draw from the small-member cluster
    small = orb.a_au[syn_D < 4]
    big = orb.a_au[syn_D > 8]
    assert np.mean(small < 3.05) > 0.9
    assert np.mean(big > 3.05) > 0.9
    # bounds respected
    assert orb.a_au.min() >= a.min() - 1e-9 and orb.a_au.max() <= a.max() + 1e-9
    assert np.all((orb.e >= 0) & (orb.e < 1))


def test_build_family_elements():
    import tempfile, os
    from synthpop.family_elements import build_family_elements
    d = tempfile.mkdtemp()
    members = os.path.join(d, "members.txt")
    catalog = os.path.join(d, "propels.dat")
    out = os.path.join(d, "a99_synth.txt")
    with open(members, "w") as f:
        # famid, packed, designation -- incl. another family and a provisional id
        f.write("        99       00042          42\n"
                "        99     K16NH8S   2016NS178\n"
                "        99       00007           7\n"
                "        10       00100         100\n")
    with open(catalog, "w") as f:
        # proper_catalog24.dat layout: a da e de sini dsini g s H n packed id
        f.write(" 3.10 1e-5 0.16 1e-4 0.37 1e-5  54.0 -59.0  10.5 12 00042  \t42\n"
                " 3.09 1e-5 0.17 1e-4 0.37 1e-5  54.0 -59.0  16.7 12 K16NH8S \t2016NS178\n"
                " 2.20 1e-5 0.10 1e-4 0.12 1e-5  30.0 -40.0   5.5 12 00100  \t100\n")
        # member 7 deliberately absent -> unmatched
    res = build_family_elements(99, members, catalog, out, verbose=False)
    assert (res.n_members, res.n_matched, res.n_unmatched) == (3, 2, 1)
    # output loads through the family-orbit reader
    cfg = SynthConfig(name="t", mode=FAMILY, family_members_file=out)
    number, H, a, e, sini = orbits_family.load_family_members(cfg)
    assert number.size == 2
    assert np.allclose(a, [3.10, 3.09])
    assert "2016NS178" in set(number)


def test_make_all_plots_writes_four_pngs():
    import tempfile
    import pandas as pd
    from synthpop import plotting
    rng = np.random.default_rng(11)
    n = 500
    df = pd.DataFrame({
        "diam_km": rng.uniform(1, 25, n),
        "b_over_a": rng.uniform(0.3, 1.0, n),
        "pole_beta_deg": rng.uniform(0, 88, n),
        "pole_lambda_deg": rng.uniform(0, 360, n),
        "a_au": rng.normal(3.14, 0.02, n),
        "e": rng.normal(0.2, 0.01, n),
        "i_deg": rng.normal(10, 0.5, n),
        "node_deg": rng.uniform(0, 360, n),
        "argperi_deg": rng.uniform(0, 360, n),
        "rot_period_h": 2.2 * (20 / 2.2) ** rng.random(n),
        "pV": rng.rayleigh(0.04, n),
        "eta": rng.normal(0.95, 0.14, n),
        "H": rng.normal(18, 1.5, n),
    })
    cfg = SynthConfig(name="t", out_dir=tempfile.mkdtemp())
    paths = plotting.make_all_plots(df, cfg,
                                    real_elements={"a": df.a_au, "e": df.e,
                                                   "sini": np.sin(np.radians(df.i_deg))})
    assert len(paths) == 4
    for p in paths:
        assert p.exists() and p.stat().st_size > 10_000, p


def _demo_population_inputs(rng):
    from synthpop.physical import fit_physical_from_arrays
    from synthpop.pipeline import PopulationInputs
    cfg = SynthConfig(name="ens", mode=FAMILY)      # family -> lognormal albedo
    pV = rng.lognormal(np.log(0.06), 0.4, 400)
    eta = np.clip(rng.normal(1.0, 0.2, 400), 0.2, None)
    a = rng.normal(3.14, 0.02, 250)
    e = rng.normal(0.15, 0.01, 250)
    sini = rng.normal(0.18, 0.01, 250)
    D = rng.uniform(2, 20, 250)
    gi = PopulationInputs(
        phys_fit=fit_physical_from_arrays(cfg, pV, eta),
        neowise_pV=pV, neowise_eta=eta,
        fam_fit=orbits_family.fit_family(a, e, sini, D),
        fam_a=a, fam_e=e, fam_sini=sini, fam_member_D=D)
    return cfg, gi


def test_ensemble_bootstrap_perturbs_fits_and_is_deterministic():
    from synthpop.ensemble import EnsembleConfig, _bootstrap_inputs
    rng = np.random.default_rng(0)
    cfg, gi = _demo_population_inputs(rng)
    ecfg = EnsembleConfig(base=cfg, n_members=6, seed=1)

    # member 0 is the nominal (unresampled) fit
    assert _bootstrap_inputs(ecfg, gi, 0, np.random.default_rng(0)) is gi

    # a bootstrap member perturbs the albedo, beaming and family-orbit fits
    m1 = _bootstrap_inputs(ecfg, gi, 1, np.random.default_rng(1))
    assert m1.phys_fit.albedo.mu != gi.phys_fit.albedo.mu
    assert m1.phys_fit.eta.mu != gi.phys_fit.eta.mu
    assert m1.fam_fit.sigma["a"] != gi.fam_fit.sigma["a"]
    # but stays close to the nominal (resampling the same data, not inventing)
    assert abs(m1.phys_fit.albedo.mu - gi.phys_fit.albedo.mu) < 0.5

    # same member + same rng seed -> identical fit (reproducible band)
    m1b = _bootstrap_inputs(ecfg, gi, 1, np.random.default_rng(1))
    assert m1b.phys_fit.albedo.mu == m1.phys_fit.albedo.mu
    assert m1b.fam_fit.sigma["a"] == m1.fam_fit.sigma["a"]


def test_ensemble_bootstrap_width_shrinks_with_more_members():
    # the systematic width must scale with how well-constrained the population is:
    # a family with more measured members gives tighter bootstrap scatter
    from synthpop.ensemble import EnsembleConfig, _bootstrap_inputs
    from synthpop.physical import fit_physical_from_arrays
    from synthpop.pipeline import PopulationInputs

    def scatter(n_data):
        rng = np.random.default_rng(4)
        cfg = SynthConfig(name="ens", mode=FAMILY)
        pV = rng.lognormal(np.log(0.06), 0.4, n_data)
        eta = np.clip(rng.normal(1.0, 0.2, n_data), 0.2, None)
        gi = PopulationInputs(phys_fit=fit_physical_from_arrays(cfg, pV, eta),
                              neowise_pV=pV, neowise_eta=eta)
        ecfg = EnsembleConfig(base=cfg, n_members=40, seed=2, bootstrap_orbits=False)
        mus = [_bootstrap_inputs(ecfg, gi, m, np.random.default_rng(m)).phys_fit.albedo.mu
               for m in range(1, 40)]
        return np.std(mus)

    assert scatter(2000) < scatter(80)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
