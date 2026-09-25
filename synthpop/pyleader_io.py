"""Read PyLEADER population output into a sampleable joint distribution.

As of the 2026-07 PyLEADER rewrite, each run is organised as::

    <run_label>/
      Fam<id>_<cat>_<filter>_<low>km_to_<high>km_analysis/   (per-Trial diagnostics)
      Fam<id>_<cat>_<filter>_<low>km_to_<high>km_basis/
      Fam<id>_<cat>_<filter>_<low>km_to_<high>km_biasmap/
      Fam<id>_<cat>_<filter>_<low>km_to_<high>km_summary/     <-- what we read

The ``*_summary`` directory holds the population-level products. The one we
consume is:

* ``population_distribution.npz`` -- the debiased **joint** distribution of
  shape elongation ``p = b/a`` and spin-axis latitude ``beta``:
    - ``p_grid`` (nx,), ``b_grid`` (ny,)  -- bin centers
    - ``f_mean`` (nx, ny)                 -- joint density (with ``f_lo``/``f_hi`` bounds)
    - ``pop_median_p`` / ``pop_median_b`` -- (median, lo, hi) population summaries
    - ``relerr``                          -- inversion relative error

This replaces the old independent ``DF_p_all.txt`` / ``DF_b_all.txt`` text
histograms; we now sample ``(p, beta)`` jointly, preserving their correlation.

``posterior_{peak,median}.npz`` (the Bayesian posterior on the population's
*point estimate*, not a per-object distribution) is exposed via
:func:`load_summary` for callers that want the population value + uncertainty.

This module only *reads*; turning a distribution into samples is
:mod:`synthpop.sampling`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np

from .sampling import EmpiricalPDF2D

# canonical PyLEADER subdir: "...Fam<id>_..._<low>km_to_<high>km_<kind>"
_BIN_RE = re.compile(r"_([0-9.]+)km_to_([0-9.]+)km_(?:summary|analysis|basis|biasmap)$")
_FAMID_RE = re.compile(r"(?:^|/)Fam(\d+)_")

DISTRIBUTION_FILE = "population_distribution.npz"


@dataclass
class LeaderDistributions:
    """The joint (p, beta) distribution for one diameter bin."""

    joint: EmpiricalPDF2D          # f_mean(p, beta); sample (b/a, |beta|) pairs
    p_grid: np.ndarray             # shape-elongation bin centers (b/a)
    b_grid: np.ndarray             # spin-latitude bin centers, degrees
    f_mean: np.ndarray             # (nx, ny) joint density
    pop_median_p: np.ndarray       # (median, lo, hi)
    pop_median_b: np.ndarray       # (median, lo, hi)
    relerr: float
    source_dir: Path


def load_distributions(dist_dir: Path) -> LeaderDistributions:
    """Load the joint (p, beta) distribution from a PyLEADER ``*_summary`` dir."""
    dist_dir = Path(dist_dir)
    z = np.load(dist_dir / DISTRIBUTION_FILE, allow_pickle=False)
    p_grid = z["p_grid"]
    b_grid = z["b_grid"]
    f_mean = z["f_mean"]
    return LeaderDistributions(
        joint=EmpiricalPDF2D(p_grid, b_grid, f_mean),
        p_grid=p_grid, b_grid=b_grid, f_mean=f_mean,
        pop_median_p=z["pop_median_p"] if "pop_median_p" in z.files else np.array([np.nan] * 3),
        pop_median_b=z["pop_median_b"] if "pop_median_b" in z.files else np.array([np.nan] * 3),
        relerr=float(z["relerr"]) if "relerr" in z.files else float("nan"),
        source_dir=dist_dir,
    )


def discover_diameter_bins(base_dir: Path, tag: str):
    """Find PyLEADER ``*_summary`` dirs for ``tag`` and parse their diameter ranges.

    Searches ``base_dir`` (recursively) for ``<tag>_..._<low>km_to_<high>km_summary``
    directories that contain ``population_distribution.npz``. A family ``tag``
    may be given as ``"Fam10"``, ``"10"``, or ``"fam10"``; a background tag is
    used verbatim (e.g. ``"BG_IB_Ctypes"``). Returns a list of ``DiameterBin``
    sorted by low edge.

    The match is on the *canonical inner directory name* (which always carries
    the population tag), so it is robust to the free-form outer run-label
    folder (e.g. ``170_1km_3km`` whose inner dir is ``Fam170_..._summary``).
    """
    from .config import DiameterBin  # local import to avoid a cycle
    base_dir = Path(base_dir)
    tag_s = str(tag)
    if tag_s.upper().startswith("BG"):
        pattern = f"{tag_s}_*km_to_*km_summary"
    else:
        m = re.search(r"(\d+)", tag_s)
        if not m:
            raise ValueError(f"could not parse a family id from tag {tag!r}")
        pattern = f"Fam{m.group(1)}_*km_to_*km_summary"

    bins = []
    for d in base_dir.rglob(pattern):
        mm = _BIN_RE.search(d.name)
        if mm and (d / DISTRIBUTION_FILE).exists():
            bins.append(DiameterBin(low_km=float(mm.group(1)),
                                    high_km=float(mm.group(2)), dist_dir=d))
    # de-duplicate identical [low, high) bins (keep the first found)
    seen, unique = set(), []
    for b in sorted(bins, key=lambda b: (b.low_km, b.high_km)):
        key = (b.low_km, b.high_km)
        if key not in seen:
            seen.add(key)
            unique.append(b)
    return unique


def load_summary(dist_dir: Path):
    """Load the Bayesian posterior summaries from ``posterior_*.npz``.

    Returns ``{"peak": {...}, "median": {...}}`` where each channel has
    ``p`` and ``beta`` entries ``(mode, median, ci68_lo, ci68_hi, ci95_lo,
    ci95_hi)`` (the ``p_stats`` / ``b_stats`` layout PyLEADER writes). These are
    the population point estimate and its uncertainty, not a per-object PDF.
    """
    dist_dir = Path(dist_dir)
    out = {}
    for channel in ("peak", "median"):
        f = dist_dir / f"posterior_{channel}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            out[channel] = {"p": z["p_stats"], "beta": z["b_stats"],
                            "observed": z["observed"]}
    if not out:
        raise FileNotFoundError(f"no posterior_*.npz in {dist_dir}")
    return out
