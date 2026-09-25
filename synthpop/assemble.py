"""Join the per-stage arrays into a single synthetic-asteroid catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .shape_pole import ShapePole
from .orbits_family import Orbits
from .physical import Physical


# Final catalog columns, in order. Everything the NEOWISE-survey simulator
# (the second pipeline) needs to place and "detect" each synthetic body.
COLUMNS = [
    "id",
    "diam_km",          # diameter (km), from the SFD
    "b_over_a",         # shape elongation b/a in (0,1], from PyLEADER
    "pole_beta_deg",    # |spin-pole ecliptic latitude| in [0, 90] (LEADER beta
                        #   is hemisphere-degenerate; obliquity ~ 90 -/+ |beta|)
    "pole_lambda_deg",  # spin-pole ecliptic longitude [0,360)
    "rot_period_h",     # rotation period (hours), log-uniform draw
    "a_au", "e", "i_deg", "node_deg", "argperi_deg", "M_deg",  # orbit
    "pV",               # visible albedo, from NEOWISE double-Rayleigh fit
    "eta",              # beaming parameter, from NEOWISE Gaussian fit
    "H",                # absolute magnitude, derived from D and pV
]


def build_catalog(name: str, diameters: np.ndarray, sp: ShapePole,
                  orbits: Orbits, phys: Physical) -> pd.DataFrame:
    n = diameters.size
    ids = np.array([f"{name}_{k:06d}" for k in range(n)])
    return pd.DataFrame({
        "id": ids,
        "diam_km": diameters,
        "b_over_a": sp.b_over_a,
        "pole_beta_deg": sp.pole_beta_deg,
        "pole_lambda_deg": sp.pole_lambda_deg,
        "rot_period_h": sp.rot_period_h,
        "a_au": orbits.a_au,
        "e": orbits.e,
        "i_deg": orbits.i_deg,
        "node_deg": orbits.node_deg,
        "argperi_deg": orbits.argperi_deg,
        "M_deg": orbits.M_deg,
        "pV": phys.pV,
        "eta": phys.eta,
        "H": phys.H,
    }, columns=COLUMNS)
