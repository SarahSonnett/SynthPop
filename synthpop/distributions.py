"""Analytic distributions fitted to NEOWISE physical parameters.

Following Lilly et al. (2021) (the NEO Surveyor synthetic-population model) and
Wright et al. (2016):

* visible albedo ``pV`` is modelled as a **double Rayleigh** distribution (a
  mixture of two Rayleigh components, capturing the low-albedo C-type and
  high-albedo S-type peaks);
* the infrared beaming parameter ``eta`` is modelled as a single (truncated)
  **Gaussian**.

Each model exposes a ``pdf``, a ``logpdf`` (for the Metropolis sampler in
:mod:`synthpop.sampling`), and a ``fit`` classmethod that estimates parameters
from a data sample by maximum likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


def rayleigh_pdf(x: np.ndarray, sigma: float) -> np.ndarray:
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = x >= 0
    out[m] = (x[m] / sigma**2) * np.exp(-(x[m] ** 2) / (2 * sigma**2))
    return out


@dataclass
class DoubleRayleigh:
    """Mixture ``w * Rayleigh(sigma1) + (1 - w) * Rayleigh(sigma2)``."""

    w: float
    sigma1: float
    sigma2: float

    def pdf(self, x):
        return self.w * rayleigh_pdf(x, self.sigma1) + (1 - self.w) * rayleigh_pdf(x, self.sigma2)

    def logpdf(self, x):
        p = self.pdf(x)
        return np.log(p) if np.all(p > 0) else np.where(p > 0, np.log(np.clip(p, 1e-300, None)), -np.inf)

    def cdf(self, x):
        x = np.asarray(x, float)
        c1 = 1.0 - np.exp(-np.clip(x, 0, None) ** 2 / (2 * self.sigma1**2))
        c2 = 1.0 - np.exp(-np.clip(x, 0, None) ** 2 / (2 * self.sigma2**2))
        return self.w * c1 + (1 - self.w) * c2

    @property
    def support(self):
        return (0.0, np.inf)

    @property
    def mode(self) -> float:
        # the smaller-sigma component's mode is a good MCMC start
        return min(self.sigma1, self.sigma2)

    @property
    def linear_scale(self) -> float:
        return self.sigma1

    @classmethod
    def fit(cls, data: np.ndarray) -> "DoubleRayleigh":
        data = np.asarray(data, float)
        data = data[np.isfinite(data) & (data > 0)]
        if data.size < 5:
            raise ValueError("need at least 5 positive albedo values to fit")
        s = np.sqrt(np.mean(data**2) / 2)  # method-of-moments Rayleigh sigma

        def negloglike(theta):
            wl, l1, l2 = theta
            w = 1 / (1 + np.exp(-wl))          # logit -> (0,1)
            sig1, sig2 = np.exp(l1), np.exp(l2)  # log -> (0, inf)
            p = w * rayleigh_pdf(data, sig1) + (1 - w) * rayleigh_pdf(data, sig2)
            p = np.clip(p, 1e-300, None)
            return -np.sum(np.log(p))

        # init: two components straddling the moment estimate (C-type low, S-type high)
        x0 = [0.0, np.log(0.6 * s), np.log(1.6 * s)]
        res = minimize(negloglike, x0, method="Nelder-Mead",
                       options=dict(maxiter=10_000, xatol=1e-6, fatol=1e-6))
        wl, l1, l2 = res.x
        w = 1 / (1 + np.exp(-wl))
        sig1, sig2 = np.exp(l1), np.exp(l2)
        # order components so sigma1 <= sigma2 (low- then high-albedo)
        if sig1 > sig2:
            sig1, sig2, w = sig2, sig1, 1 - w
        return cls(w=float(w), sigma1=float(sig1), sigma2=float(sig2))


@dataclass
class LogNormal:
    """Gaussian in ln(x): ``mu``/``sigma`` are the mean/std of ln(x).

    Often the better model for positive-definite quantities with
    multiplicative scatter (e.g. albedo); the median is ``exp(mu)``.
    """

    mu: float       # mean of ln(x)
    sigma: float    # std of ln(x)

    def pdf(self, x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = (np.exp(-0.5 * ((np.log(x[m]) - self.mu) / self.sigma) ** 2)
                  / (x[m] * self.sigma * np.sqrt(2 * np.pi)))
        return out

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        lx = np.log(x)
        return (-0.5 * ((lx - self.mu) / self.sigma) ** 2
                - lx - np.log(self.sigma * np.sqrt(2 * np.pi)))

    def cdf(self, x):
        from scipy.special import erf
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        out[m] = 0.5 * (1.0 + erf((np.log(x[m]) - self.mu)
                                  / (self.sigma * np.sqrt(2))))
        return out

    @property
    def support(self):
        return (0.0, np.inf)

    @property
    def mode(self) -> float:
        return float(np.exp(self.mu - self.sigma**2))

    @property
    def linear_scale(self) -> float:
        """Approximate linear-space width, for MCMC proposal steps."""
        return float(np.exp(self.mu) * self.sigma)

    @classmethod
    def fit(cls, data: np.ndarray, clip_sigma: float | None = None,
            max_iter: int = 10) -> "LogNormal":
        """Moments fit in ln-space, optionally with iterative sigma-clipping."""
        data = np.asarray(data, float)
        ln = np.log(data[np.isfinite(data) & (data > 0)])
        if ln.size < 3:
            raise ValueError("need at least 3 positive values to fit a lognormal")
        mu, sigma = float(np.mean(ln)), float(np.std(ln, ddof=1))
        if clip_sigma is not None:
            for _ in range(max_iter):
                mask = np.abs(ln - mu) <= clip_sigma * sigma
                if mask.all() or mask.sum() < 3:
                    break
                ln = ln[mask]
                mu, sigma = float(np.mean(ln)), float(np.std(ln, ddof=1))
        return cls(mu=mu, sigma=sigma)


def _norm_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


@dataclass
class DoubleLogNormal:
    """Two-component Gaussian mixture in ln(x) (a "double lognormal").

    The bimodal extension of :class:`LogNormal`, for families whose albedo
    distribution shows two components (e.g. overlapping C- and S-complex
    material). Component 1 is the darker one (``mu1 < mu2``); parameters are
    the means/widths of ln(x) and the weight ``w`` of component 1.
    """

    w: float
    mu1: float
    sigma1: float
    mu2: float
    sigma2: float

    def pdf(self, x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        lx = np.log(x[m])
        out[m] = (self.w * _norm_pdf(lx, self.mu1, self.sigma1)
                  + (1 - self.w) * _norm_pdf(lx, self.mu2, self.sigma2)) / x[m]
        return out

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        p = float(self.pdf(np.array([x]))[0])
        return np.log(max(p, 1e-300))

    def cdf(self, x):
        from scipy.special import erf
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        lx = np.log(x[m])
        phi1 = 0.5 * (1 + erf((lx - self.mu1) / (self.sigma1 * np.sqrt(2))))
        phi2 = 0.5 * (1 + erf((lx - self.mu2) / (self.sigma2 * np.sqrt(2))))
        out[m] = self.w * phi1 + (1 - self.w) * phi2
        return out

    @property
    def support(self):
        return (0.0, np.inf)

    @property
    def mode(self) -> float:
        # dominant component's lognormal mode: a robust MCMC starting point
        mu, s = ((self.mu1, self.sigma1) if self.w >= 0.5
                 else (self.mu2, self.sigma2))
        return float(np.exp(mu - s**2))

    @property
    def linear_scale(self) -> float:
        """Proposal scale must span BOTH modes for a random-walk sampler."""
        sep = abs(np.exp(self.mu2) - np.exp(self.mu1))
        return float(max(sep, np.exp(self.mu1) * self.sigma1))

    def sample_truncated(self, n: int, rng, lo: float = 0.0,
                         hi: float = np.inf) -> np.ndarray:
        """Exact draw (component choice + lognormal), rejecting outside
        ``[lo, hi]``. Preferred over random-walk MCMC for a bimodal target,
        whose modes a short chain mixes between poorly."""
        out = np.empty(n)
        filled = 0
        for _ in range(1000):   # guard: acceptance ~0 only if [lo, hi] misses
            if filled >= n:     # essentially all the mixture mass
                break
            m = n - filled
            comp1 = rng.random(m) < self.w
            mu = np.where(comp1, self.mu1, self.mu2)
            sig = np.where(comp1, self.sigma1, self.sigma2)
            draw = np.exp(rng.normal(mu, sig))
            draw = draw[(draw >= lo) & (draw <= hi)]
            out[filled:filled + draw.size] = draw
            filled += draw.size
        if filled < n:
            raise RuntimeError(
                f"sample_truncated: bounds [{lo}, {hi}] exclude nearly all "
                "mixture mass")
        return out

    @classmethod
    def fit(cls, data: np.ndarray, max_iter: int = 500,
            tol: float = 1e-9) -> "DoubleLogNormal":
        """Maximum-likelihood EM fit of the two-component mixture in ln(x)."""
        x = np.asarray(data, float)
        x = np.log(x[np.isfinite(x) & (x > 0)])
        if x.size < 10:
            raise ValueError("need at least 10 positive values for a mixture fit")
        # initialise the components at the outer quartiles
        mu1, mu2 = np.percentile(x, [20, 80])
        s1 = s2 = max(float(np.std(x, ddof=1)) / 2, 1e-3)
        w = 0.5
        last = -np.inf
        for _ in range(max_iter):
            p1 = w * _norm_pdf(x, mu1, s1)
            p2 = (1 - w) * _norm_pdf(x, mu2, s2)
            tot = np.clip(p1 + p2, 1e-300, None)
            r = p1 / tot
            r1, r2 = r.sum(), (1 - r).sum()
            if r1 < 1e-6 or r2 < 1e-6:
                break  # degenerate: one component vanished
            w = float(r.mean())
            mu1 = float((r * x).sum() / r1)
            mu2 = float(((1 - r) * x).sum() / r2)
            s1 = max(float(np.sqrt((r * (x - mu1) ** 2).sum() / r1)), 1e-3)
            s2 = max(float(np.sqrt(((1 - r) * (x - mu2) ** 2).sum() / r2)), 1e-3)
            ll = float(np.log(tot).sum())
            if abs(ll - last) < tol:
                break
            last = ll
        if mu1 > mu2:  # order: component 1 is the darker
            mu1, mu2, s1, s2, w = mu2, mu1, s2, s1, 1 - w
        return cls(w=float(w), mu1=mu1, sigma1=s1, mu2=mu2, sigma2=s2)


@dataclass
class LogNormalMixture:
    """k-component Gaussian mixture in ln(x), k selected by BIC.

    The opt-in ``"mixture"`` albedo model: fit on the *unclipped* member
    sample, so a genuine few-percent dark group (which the default's
    interloper sigma-clip deliberately excludes; see README) gets its own
    component instead of being rejected. Parameters are stored as plain
    lists so the model serialises directly into provenance JSON.
    """

    ws: list        # component weights (sum to 1)
    mus: list       # means of ln(x)
    sigmas: list    # widths of ln(x)

    def _arrs(self):
        return (np.asarray(self.ws, float), np.asarray(self.mus, float),
                np.asarray(self.sigmas, float))

    def pdf(self, x):
        ws, mus, sigs = self._arrs()
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        lx = np.log(x[m])
        out[m] = sum(w * _norm_pdf(lx, mu, s)
                     for w, mu, s in zip(ws, mus, sigs)) / x[m]
        return out

    def logpdf(self, x):
        if x <= 0:
            return -np.inf
        return np.log(max(float(self.pdf(np.array([x]))[0]), 1e-300))

    def cdf(self, x):
        from scipy.special import erf
        ws, mus, sigs = self._arrs()
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        m = x > 0
        lx = np.log(x[m])
        out[m] = sum(w * 0.5 * (1 + erf((lx - mu) / (s * np.sqrt(2))))
                     for w, mu, s in zip(ws, mus, sigs))
        return out

    @property
    def support(self):
        return (0.0, np.inf)

    @property
    def mode(self) -> float:
        ws, mus, sigs = self._arrs()
        j = int(np.argmax(ws))
        return float(np.exp(mus[j] - sigs[j] ** 2))

    @property
    def linear_scale(self) -> float:
        ws, mus, sigs = self._arrs()
        meds = np.exp(mus)
        sep = meds.max() - meds.min()
        return float(max(sep, meds[int(np.argmax(ws))] * sigs[int(np.argmax(ws))]))

    def sample_truncated(self, n: int, rng, lo: float = 0.0,
                         hi: float = np.inf) -> np.ndarray:
        ws, mus, sigs = self._arrs()
        out = np.empty(n)
        filled = 0
        for _ in range(1000):
            if filled >= n:
                break
            m = n - filled
            comp = rng.choice(ws.size, size=m, p=ws / ws.sum())
            draw = np.exp(rng.normal(mus[comp], sigs[comp]))
            draw = draw[(draw >= lo) & (draw <= hi)]
            out[filled:filled + draw.size] = draw
            filled += draw.size
        if filled < n:
            raise RuntimeError(
                f"sample_truncated: bounds [{lo}, {hi}] exclude nearly all "
                "mixture mass")
        return out

    @classmethod
    def fit_k(cls, data: np.ndarray, k: int, max_iter: int = 500,
              tol: float = 1e-9) -> tuple:
        """EM fit with fixed k; returns ``(model, loglike)``."""
        x = np.asarray(data, float)
        x = np.log(x[np.isfinite(x) & (x > 0)])
        if x.size < 10 * k:
            raise ValueError(f"need at least {10 * k} values for k={k}")
        qs = np.linspace(10, 90, k)
        mus = np.percentile(x, qs).astype(float)
        sigs = np.full(k, max(float(np.std(x, ddof=1)) / k, 1e-3))
        ws = np.full(k, 1.0 / k)
        last = -np.inf
        for _ in range(max_iter):
            comp = np.stack([w * _norm_pdf(x, mu, s)
                             for w, mu, s in zip(ws, mus, sigs)])
            tot = np.clip(comp.sum(axis=0), 1e-300, None)
            r = comp / tot
            rs = r.sum(axis=1)
            if (rs < 1e-6).any():
                break
            ws = rs / x.size
            mus = (r * x).sum(axis=1) / rs
            sigs = np.maximum(np.sqrt((r * (x - mus[:, None]) ** 2).sum(axis=1) / rs),
                              1e-3)
            ll = float(np.log(tot).sum())
            if abs(ll - last) < tol:
                break
            last = ll
        order = np.argsort(mus)   # darkest component first
        return (cls(ws=ws[order].tolist(), mus=mus[order].tolist(),
                    sigmas=sigs[order].tolist()), last)

    @classmethod
    def fit_bic(cls, data: np.ndarray, k_max: int = 3, w_floor: float = 0.02,
                min_members: int = 20) -> "LogNormalMixture":
        """Fit k = 1..k_max on the (unclipped) sample and keep the best BIC.

        A candidate k is admissible only if every component holds at least
        ``w_floor`` of the weight and ``min_members`` expected members, so a
        handful of stray objects cannot claim a component -- but a genuine
        few-percent dark group can.
        """
        data = np.asarray(data, float)
        n = data[np.isfinite(data) & (data > 0)].size
        best, best_bic = None, np.inf
        for k in range(1, k_max + 1):
            if n < 10 * k:
                break
            model, ll = cls.fit_k(data, k)
            ws = np.asarray(model.ws)
            if k > 1 and ((ws < w_floor).any() or (ws * n < min_members).any()):
                continue
            bic = (3 * k - 1) * np.log(n) - 2 * ll
            if bic < best_bic:
                best, best_bic = model, bic
        return best


def assess_albedo_bimodality(pV: np.ndarray, clip_sigma: float = 3.0) -> dict:
    """Test a family's albedo sample for bimodality in log-albedo.

    Compares a single Gaussian against the two-component mixture in ln(pV) by
    BIC, and requires the recovered components to be genuinely separated
    (Ashman's D > 2) and both substantially populated (minor weight >= 0.1) --
    so mere skewness or a handful of interlopers is not called bimodal.

    Returns a dict with ``bimodal`` (bool), ``delta_bic`` (positive favors the
    mixture), ``ashman_d``, ``w_minor``, ``n``, and the fitted ``mixture``.
    """
    x = np.asarray(pV, float)
    x = np.log(x[np.isfinite(x) & (x > 0)])
    # mild clipping so extreme outliers cannot masquerade as a mode
    mu, s = float(np.mean(x)), float(np.std(x, ddof=1))
    for _ in range(10):
        m = np.abs(x - mu) <= clip_sigma * s
        if m.all() or m.sum() < 10:
            break
        x = x[m]
        mu, s = float(np.mean(x)), float(np.std(x, ddof=1))
    n = x.size
    if n < 30:   # too small to claim bimodality
        return {"bimodal": False, "two_component": False, "use_mixture": False,
                "delta_bic": 0.0, "ashman_d": 0.0, "w_minor": 0.0,
                "dip_ratio": 1.0, "n": int(n), "mixture": None}

    ll1 = float(np.log(np.clip(_norm_pdf(x, mu, s), 1e-300, None)).sum())
    mix = DoubleLogNormal.fit(np.exp(x))
    p = (mix.w * _norm_pdf(x, mix.mu1, mix.sigma1)
         + (1 - mix.w) * _norm_pdf(x, mix.mu2, mix.sigma2))
    ll2 = float(np.log(np.clip(p, 1e-300, None)).sum())
    bic1 = 2 * np.log(n) - 2 * ll1          # 2 params
    bic2 = 5 * np.log(n) - 2 * ll2          # 5 params
    delta_bic = float(bic1 - bic2)
    ashman_d = float(np.sqrt(2) * abs(mix.mu2 - mix.mu1)
                     / np.sqrt(mix.sigma1**2 + mix.sigma2**2))
    w_minor = float(min(mix.w, 1 - mix.w))

    # Antimode check: a mixture also fits mere *skewness* well (inflating
    # delta_BIC), so demand the fitted density actually has two peaks with a
    # genuine dip between them.
    grid = np.linspace(min(mix.mu1 - 3 * mix.sigma1, mix.mu2 - 3 * mix.sigma2),
                       max(mix.mu1 + 3 * mix.sigma1, mix.mu2 + 3 * mix.sigma2),
                       1024)
    dens = (mix.w * _norm_pdf(grid, mix.mu1, mix.sigma1)
            + (1 - mix.w) * _norm_pdf(grid, mix.mu2, mix.sigma2))
    interior = (dens[1:-1] > dens[:-2]) & (dens[1:-1] > dens[2:])
    peaks = np.where(interior)[0] + 1
    dip_ratio = 1.0
    if peaks.size >= 2:
        top2 = peaks[np.argsort(dens[peaks])[-2:]]
        lo_i, hi_i = sorted(top2)
        valley = dens[lo_i:hi_i + 1].min()
        dip_ratio = float(valley / min(dens[lo_i], dens[hi_i]))

    # Two regimes select the mixture model:
    # * "bimodal" -- two resolved peaks with a genuine dip between them
    #   (e.g. Phocaea, Tirela, Juno);
    # * "two_component" -- overwhelming evidence for two populations whose
    #   overlap leaves no visible dip (e.g. Flora, Eos: a blended C+S mix).
    #   Threshold dBIC > 50 is far beyond the "very strong" evidence scale of
    #   Kass & Raftery (1995); both components must be substantially
    #   populated so a handful of outliers cannot qualify.
    # dip < 0.7: the resolved families sit far below (Phocaea 0.12,
    # Tirela 0.16, Juno 0.19) while sharp-edged skew artifacts sit above
    bimodal = (delta_bic > 10.0) and (ashman_d > 2.0) and (w_minor >= 0.10) \
        and (peaks.size >= 2) and (dip_ratio < 0.7)
    two_component = (delta_bic > 50.0) and (w_minor >= 0.15)
    return {"bimodal": bool(bimodal), "two_component": bool(two_component),
            "use_mixture": bool(bimodal or two_component),
            "delta_bic": delta_bic, "ashman_d": ashman_d, "w_minor": w_minor,
            "dip_ratio": dip_ratio, "n": int(n), "mixture": mix}


@dataclass
class TruncatedGaussian:
    """Gaussian truncated to ``support`` (default: positive reals)."""

    mu: float
    sigma: float
    lo: float = 0.0
    hi: float = np.inf

    def pdf(self, x):
        x = np.asarray(x, float)
        out = np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2) / (self.sigma * np.sqrt(2 * np.pi))
        return np.where((x >= self.lo) & (x <= self.hi), out, 0.0)

    def logpdf(self, x):
        if x < self.lo or x > self.hi:
            return -np.inf
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - np.log(self.sigma * np.sqrt(2 * np.pi))

    def cdf(self, x):
        """CDF of the *truncated* distribution (0 at lo, 1 at hi)."""
        from scipy.special import erf

        def phi(v):  # standard normal CDF at (v - mu) / sigma
            return 0.5 * (1.0 + erf((v - self.mu) / (self.sigma * np.sqrt(2))))

        x = np.clip(np.asarray(x, float), self.lo, self.hi)
        lo_m = phi(self.lo)
        hi_m = phi(self.hi) if np.isfinite(self.hi) else 1.0
        return (phi(x) - lo_m) / max(hi_m - lo_m, 1e-300)

    @property
    def support(self):
        return (self.lo, self.hi)

    @property
    def mode(self) -> float:
        return float(np.clip(self.mu, self.lo, self.hi))

    @property
    def linear_scale(self) -> float:
        return self.sigma

    @classmethod
    def fit(cls, data: np.ndarray, lo: float = 0.0, hi: float = np.inf,
            clip_sigma: float | None = None, max_iter: int = 10) -> "TruncatedGaussian":
        """Moments fit, optionally with iterative sigma-clipping.

        ``clip_sigma`` (e.g. 3.0) iteratively rejects points more than that
        many sigma from the mean before re-estimating -- essential when the
        sample contains interlopers (e.g. high-albedo non-family objects in a
        family member list), which otherwise inflate the fitted width.
        """
        data = np.asarray(data, float)
        data = data[np.isfinite(data) & (data >= lo) & (data <= hi)]
        if data.size < 3:
            raise ValueError("need at least 3 values to fit a Gaussian")
        keep = data
        mu, sigma = float(np.mean(keep)), float(np.std(keep, ddof=1))
        if clip_sigma is not None:
            for _ in range(max_iter):
                mask = np.abs(keep - mu) <= clip_sigma * sigma
                if mask.all() or mask.sum() < 3:
                    break
                keep = keep[mask]
                mu, sigma = float(np.mean(keep)), float(np.std(keep, ddof=1))
        return cls(mu=mu, sigma=sigma, lo=lo, hi=hi)
