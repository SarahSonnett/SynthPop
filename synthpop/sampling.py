"""Generic sampling utilities shared by every stage.

Three tools are used throughout SynthPop:

* :func:`EmpiricalPDF` / :func:`sample_from_pdf` -- inverse-CDF (inverse
  transform) sampling from a *binned* 1-D empirical distribution (e.g. the SFD).

* :func:`EmpiricalPDF2D` -- the 2-D analog, sampling ``(x, y)`` *pairs* from a
  binned joint density. Used for PyLEADER's joint shape/spin distribution
  ``f_mean(p, beta)`` (``population_distribution.npz``), preserving the
  correlation between shape elongation and spin-axis orientation.

* :func:`metropolis` -- a small, dependency-free Metropolis MCMC sampler. The
  physical-parameter stage fits analytic distributions (double Rayleigh for
  albedo, Gaussian for beaming) and then MCMC-samples synthetic values from the
  fitted density, following Lilly et al. (2021). We roll our own rather than
  depend on ``emcee`` so the install footprint matches PyLEADER
  (numpy/scipy/matplotlib only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class EmpiricalPDF:
    """A binned probability density, sampleable by inverse transform.

    Parameters
    ----------
    centers : bin-center values of the independent variable.
    density : (unnormalised) probability density in each bin.
    """

    centers: np.ndarray
    density: np.ndarray

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, float)
        self.density = np.clip(np.asarray(self.density, float), 0.0, None)
        if self.centers.size != self.density.size:
            raise ValueError("centers and density must have equal length")
        if not np.any(self.density > 0):
            raise ValueError("density is all zero; nothing to sample")
        # Bin edges midway between centers; extrapolate the outer half-bins.
        c = self.centers
        edges = np.empty(c.size + 1)
        edges[1:-1] = 0.5 * (c[:-1] + c[1:])
        edges[0] = c[0] - 0.5 * (c[1] - c[0])
        edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
        self.edges = edges
        widths = np.diff(edges)
        mass = self.density * widths
        self.cdf = np.concatenate([[0.0], np.cumsum(mass)])
        self.cdf /= self.cdf[-1]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` values. Within a bin the value is interpolated linearly,
        giving a continuous (piecewise-uniform) draw rather than discrete centers."""
        u = rng.random(n)
        # locate the bin each u falls into along the cumulative mass
        idx = np.searchsorted(self.cdf, u, side="right") - 1
        idx = np.clip(idx, 0, self.edges.size - 2)
        lo = self.cdf[idx]
        hi = self.cdf[idx + 1]
        frac = np.where(hi > lo, (u - lo) / (hi - lo), 0.0)
        return self.edges[idx] + frac * (self.edges[idx + 1] - self.edges[idx])


def sample_from_pdf(centers, density, n, rng) -> np.ndarray:
    return EmpiricalPDF(centers, density).sample(n, rng)


def _edges_from_centers(c: np.ndarray) -> np.ndarray:
    """Bin edges midway between centers, extrapolating the outer half-bins."""
    c = np.asarray(c, float)
    edges = np.empty(c.size + 1)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    return edges


@dataclass
class EmpiricalPDF2D:
    """A binned 2-D joint density, sampleable by inverse transform.

    Samples ``(x, y)`` *pairs* from the full joint density, preserving any
    correlation between the two variables -- unlike sampling two 1-D marginals
    independently. Used for the PyLEADER joint shape/spin distribution
    ``f_mean(p, beta)``.

    Parameters
    ----------
    x_centers, y_centers : bin-center grids for the two axes (lengths ``nx``, ``ny``).
    density : ``(nx, ny)`` (unnormalised) density; ``density[i, j]`` is the mass
        density at ``(x_centers[i], y_centers[j])``.
    """

    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray

    def __post_init__(self) -> None:
        self.x_centers = np.asarray(self.x_centers, float)
        self.y_centers = np.asarray(self.y_centers, float)
        self.density = np.clip(np.asarray(self.density, float), 0.0, None)
        if self.density.shape != (self.x_centers.size, self.y_centers.size):
            raise ValueError("density shape must be (len(x_centers), len(y_centers))")
        if not np.any(self.density > 0):
            raise ValueError("density is all zero; nothing to sample")
        self.x_edges = _edges_from_centers(self.x_centers)
        self.y_edges = _edges_from_centers(self.y_centers)
        # per-cell probability mass = density * cell area
        cell_area = np.outer(np.diff(self.x_edges), np.diff(self.y_edges))
        mass = (self.density * cell_area).ravel()
        self.flat_cdf = np.concatenate([[0.0], np.cumsum(mass)])
        self.flat_cdf /= self.flat_cdf[-1]
        self.ny = self.y_centers.size

    def sample(self, n: int, rng: np.random.Generator):
        """Draw ``n`` ``(x, y)`` pairs; within a cell the position is uniform."""
        u = rng.random(n)
        cell = np.searchsorted(self.flat_cdf, u, side="right") - 1
        cell = np.clip(cell, 0, self.flat_cdf.size - 2)
        ix, iy = np.divmod(cell, self.ny)
        x = self.x_edges[ix] + rng.random(n) * (self.x_edges[ix + 1] - self.x_edges[ix])
        y = self.y_edges[iy] + rng.random(n) * (self.y_edges[iy + 1] - self.y_edges[iy])
        return x, y


def metropolis(
    log_pdf: Callable[[float], float],
    n: int,
    x0: float,
    proposal_scale: float,
    rng: np.random.Generator,
    burn: int = 2_000,
    thin: int = 5,
    bounds: tuple = (-np.inf, np.inf),
) -> np.ndarray:
    """Sample ``n`` values from an (unnormalised) 1-D density via Metropolis.

    Parameters
    ----------
    log_pdf : returns log density (may be -inf outside support).
    x0 : starting point.
    proposal_scale : std of the Gaussian random-walk proposal.
    burn, thin : discard ``burn`` initial steps, keep every ``thin``-th sample.
    bounds : hard support limits; proposals outside are rejected.
    """
    lo, hi = bounds
    total = burn + n * thin
    x = float(x0)
    lp = log_pdf(x)
    out = np.empty(n)
    kept = 0
    for step in range(total):
        xp = x + rng.normal(0.0, proposal_scale)
        if xp < lo or xp > hi:
            lpp = -np.inf
        else:
            lpp = log_pdf(xp)
        if np.log(rng.random()) < lpp - lp:
            x, lp = xp, lpp
        if step >= burn and (step - burn) % thin == 0 and kept < n:
            out[kept] = x
            kept += 1
    return out
