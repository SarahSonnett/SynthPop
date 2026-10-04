"""Generate synthetic background orbits from the s3m model (Grav et al. 2011).

Background subpopulations (inner/middle/pristine/outer Main Belt, Hungarias,
Cybeles) are *not* clustered like families, so instead of fuzzing we draw their
orbital elements directly from the matching s3m synthetic-population file, which
already encodes the correct orbital distribution for that region. We bootstrap
(sample rows with replacement) to the requested ``n_objects`` and add a small
optional jitter so repeated draws are not bit-identical.

The s3m column layout is configured via ``cfg.s3m_columns`` so this loader is
not hard-wired to one file flavour; see ``data/s3m_schema.md`` for the two
supported element formats:

* ``"KEP"`` -- elements given as (a, e, i, node, argperi, M).
* ``"COM"`` -- cometary elements (q, e, i, node, argperi, t_peri); we convert
  perihelion distance to semi-major axis via ``a = q / (1 - e)``.

Angles are assumed to be in degrees and ``a``/``q`` in AU unless the mapping
says otherwise.
"""

from __future__ import annotations

import numpy as np

from .config import SynthConfig
from .orbits_family import Orbits


# Default mapping for MPC-derived s3m files (layout: id S3M q e i node argperi t_peri H G epoch n_opp MPC). Override via
# cfg.s3m_columns for other flavours.
_DEFAULT_S3M_COLUMNS = {
    "format": "COM",   # "COM" or "KEP"
    "q": 2, "e": 3, "i": 4, "node": 5, "argperi": 6, "t_peri": 7,
    # for KEP format instead: "a", "e", "i", "node", "argperi", "M"
}


def load_s3m_elements(cfg: SynthConfig):
    """Read the s3m file -> dict of element arrays (a, e, i, node, argperi, M)."""
    if cfg.s3m_file is None:
        raise ValueError("cfg.s3m_file is not set; provide the s3m model path")
    cmap = dict(_DEFAULT_S3M_COLUMNS)
    if cfg.s3m_columns:
        cmap.update(cfg.s3m_columns)

    # Read ONLY the element columns: s3m rows carry string fields (the id and
    # the "S3M"/"MPC" markers), which would turn whole-row float parsing into
    # NaNs and silently drop every orbit.
    keys = (("a", "e", "i", "node", "argperi", "M")
            if cmap["format"].upper() == "KEP"
            else ("q", "e", "i", "node", "argperi", "t_peri"))
    usecols = [cmap[k] for k in keys]
    raw = np.genfromtxt(cfg.s3m_file, comments="!", usecols=usecols,
                        dtype=float, invalid_raise=False)
    raw = np.atleast_2d(raw)
    raw = raw[np.isfinite(raw).all(axis=1)]
    col = {k: raw[:, j] for j, k in enumerate(keys)}

    e = col["e"]
    i = col["i"]
    node = col["node"]
    argperi = col["argperi"]

    if cmap["format"].upper() == "KEP":
        a = col["a"]
        M = col["M"]
    else:  # COM: q, e (+ time of perihelion -> random mean anomaly)
        a = col["q"] / (1.0 - e)
        M = None  # filled with uniform draws by the caller

    return {"a": a, "e": e, "i": i, "node": node, "argperi": argperi, "M": M}


def generate_background_orbits(cfg: SynthConfig, rng: np.random.Generator,
                               jitter_frac: float = 0.0) -> Orbits:
    """Bootstrap ``cfg.n_objects`` orbits from the region's s3m population."""
    el = load_s3m_elements(cfg)
    n_src = el["a"].size
    idx = rng.integers(0, n_src, cfg.n_objects)

    a = el["a"][idx]
    e = el["e"][idx]
    i = el["i"][idx]
    node = el["node"][idx]
    argperi = el["argperi"][idx]
    M = el["M"][idx] if el["M"] is not None else rng.random(cfg.n_objects) * 360.0

    if jitter_frac > 0:
        a = a * (1 + rng.normal(0, jitter_frac, a.size))
        e = np.clip(e * (1 + rng.normal(0, jitter_frac, e.size)), 0, 0.999)
        i = np.clip(i * (1 + rng.normal(0, jitter_frac, i.size)), 0, None)

    return Orbits(a_au=a, e=e, i_deg=i, node_deg=node, argperi_deg=argperi, M_deg=M)
