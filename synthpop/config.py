"""Configuration for the SynthPop synthetic-population pipeline.

``SynthConfig`` plays the same role here that ``AnalysisConfig`` plays in
PyLEADER: it replaces the notebook "top cell" with a single, documented,
serialisable object that every stage reads from. One ``SynthConfig`` fully
specifies one synthetic population (one dynamically distinct subpopulation:
either a collisional family or a background region).

Nothing in this module does science; it only declares *what* to build and
*where the inputs live*. The stage modules (``sfd``, ``shape_pole``,
``orbits_family``/``orbits_bg``, ``physical``) consume it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence
import json


# Population mode -----------------------------------------------------------
FAMILY = "family"          # a collisional family (orbits from the observed members)
BACKGROUND = "background"  # a non-family region, orbits from the s3m model (Grav+2011)


@dataclass
class DiameterBin:
    """A PyLEADER diameter bin and the ``*_summary`` dir holding its results.

    PyLEADER writes one ``Fam<id>_..._<low>km_to_<high>km_summary`` directory per
    diameter bin, containing ``population_distribution.npz`` -- the joint
    distribution of shape elongation (p = b/a) and pole latitude (beta).
    We sample shape and pole from the bin whose [low, high) range contains an
    object's diameter.
    """

    low_km: float
    high_km: float
    dist_dir: Path       # the PyLEADER *_summary directory (holds population_distribution.npz)

    def contains(self, diam_km: float) -> bool:
        return self.low_km <= diam_km < self.high_km

    def __post_init__(self) -> None:
        self.dist_dir = Path(self.dist_dir)


@dataclass
class SynthConfig:
    """Everything needed to generate one synthetic subpopulation."""

    # --- identity ---------------------------------------------------------
    name: str                                   # e.g. "Hygiea" or "BG_IB_Ctypes"
    mode: str = BACKGROUND                       # FAMILY or BACKGROUND
    n_objects: int = 10_000                      # number of synthetic asteroids to draw

    # --- size-frequency distribution (sfd.py) -----------------------------
    # sfd_kind selects the diameter sampling law:
    #   "powerlaw"   -- cumulative N(>D) propto D**-slope (optionally broken):
    #                   a realistic SFD shape.
    #   "loguniform" -- equal counts per logarithmic diameter bin: an
    #                   *unrealistic* SFD that maximises statistics at every
    #                   size, incl. rare large objects. Use for detection-
    #                   efficiency work (Simmer debiasing), where the synthetic
    #                   population only estimates P(detect | D) per size bin
    #                   and its absolute SFD shape is irrelevant.
    sfd_kind: str = "powerlaw"
    diam_min_km: float = 1.0
    diam_max_km: float = 100.0
    sfd_slope: float = 2.5                        # single-slope default (powerlaw)
    sfd_slope_small: Optional[float] = None       # if set, slope below d_break
    sfd_break_km: Optional[float] = None          # break diameter for a broken power law

    # --- shape & pole (shape_pole.py, from PyLEADER) ----------------------
    diameter_bins: Sequence[DiameterBin] = field(default_factory=list)
    # beta is the pole's ecliptic latitude (standard convention; what N&K 2017
    # call beta is the polar angle, 90 - |beta|). LEADER is hemisphere-
    # degenerate: only |beta| in [0, 90] deg is constrained, so the catalog
    # stores the magnitude (no invented sign). Legacy (unlabeled) PyLEADER
    # files hold the polar angle and are refused; catalogs produced before
    # 2026-10-19 hold it too (KNOWN ISSUE note in shape_pole.py).
    # rotation period, log-uniform between these bounds (hours); typical MBA
    # range, matching the spin-barrier floor at ~2.2 h
    rot_period_min_h: float = 2.2
    rot_period_max_h: float = 20.0

    # --- orbits: family mode (orbits_family.py) ---------------------------
    family_members_file: Optional[Path] = None    # AFP_families/a<famid>v<vcut>.txt
    # column indices (0-based) in the family file: number, H, a, e, sin_i, famid
    family_cols: tuple = (0, 1, 2, 3, 4, 5)
    family_clip_to_bounds: bool = True            # exclude synthetic elements outside observed bounds
    # Orbit generator:
    #   "dorb"     -- Gaussian family + D_orb(D) scaling (Schunova-Lilly+2017).
    #                 Low variance, but cannot represent multimodal or
    #                 resonance-sculpted families.
    #   "resample" -- size-matched bootstrap of real member elements with
    #                 kernel smoothing. Preserves the observed (a, e, sin i)
    #                 structure and the empirical size-orbit coupling, at the
    #                 cost of carrying the member sample's noise and selection
    #                 effects. Run Simmer with both to bound the systematic.
    family_orbit_model: str = "dorb"
    resample_k_neighbors: int = 50                # size-match pool per object

    # --- orbits: background mode (orbits_bg.py) ---------------------------
    s3m_file: Optional[Path] = None               # s3m model file (Grav+2011) for this region
    s3m_columns: Optional[dict] = None            # column mapping; see data/s3m_schema.md

    # --- physical params: albedo & beaming (physical.py) ------------------
    neowise_file: Optional[Path] = None           # PDS SBN NEOWISE diameters/albedos CSV
    # restrict the NEOWISE fit sample to this region's members (numbers), optional
    neowise_member_numbers: Optional[Sequence[int]] = None
    # Albedo model. None = auto: family -> "lognormal", upgraded to
    # "double_lognormal" when the member albedos test bimodal (see
    # distributions.assess_albedo_bimodality); background -> "double_rayleigh"
    # (mixed C/S taxonomies, Wright+2016). Explicit values force a model:
    # "gaussian" | "lognormal" | "double_lognormal" | "double_rayleigh".
    albedo_model: Optional[str] = None
    # Fit the beaming parameter only to members whose eta was actually FIT
    # by NEOWISE (fit-code column contains 'B'). The default (False) uses all
    # rows, including the ~40-80% per family carrying quantized ASSUMED values
    # (0.7/0.95/1.0/1.2/1.4), which drag the fit toward eta=1 and shrink its
    # width. Sensitivity option; see README.
    eta_fitted_only: bool = False
    # Physical floor on sampled albedo. Albedos below ~1% have never been
    # measured for asteroids, so a fitted tail below this is unphysical
    # extrapolation; the sampled distribution is truncated here.
    albedo_min: float = 0.01
    mcmc_burn: int = 2_000
    mcmc_thin: int = 5

    # --- output -----------------------------------------------------------
    out_dir: Path = Path("synthpop_out")
    out_format: str = "csv"                       # "csv" or "parquet"
    make_plots: bool = True                       # write diagnostic PNGs (plotting.py)
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in (FAMILY, BACKGROUND):
            raise ValueError(f"mode must be {FAMILY!r} or {BACKGROUND!r}, got {self.mode!r}")
        for p in ("family_members_file", "s3m_file", "neowise_file", "out_dir"):
            v = getattr(self, p)
            if v is not None:
                setattr(self, p, Path(v).expanduser())
        self.diameter_bins = [
            b if isinstance(b, DiameterBin) else DiameterBin(**b)
            for b in self.diameter_bins
        ]

    # --- (de)serialisation -------------------------------------------------
    def to_json(self, path: Path) -> None:
        d = asdict(self)
        d = json.loads(json.dumps(d, default=str))  # Path -> str
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "SynthConfig":
        return cls(**json.loads(Path(path).read_text()))
