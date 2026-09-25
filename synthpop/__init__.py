"""SynthPop -- synthetic Main-Belt asteroid population generator.

Builds a database of synthetic asteroids whose physical and orbital properties
match the expected distributions from prior studies. Shape elongation (b/a) and
spin-pole latitude come from PyLEADER distributions; pole longitude is uniform;
orbits come from the s3m model (backgrounds) or the observed family
(collisional families); albedo and beaming come from NEOWISE fits sampled by MCMC.

The output catalog is the input to a second pipeline (built separately) that
simulates the NEOWISE survey and "detects" each synthetic body.
"""

from .config import SynthConfig, DiameterBin, FAMILY, BACKGROUND
from .pipeline import run_synthpop

__all__ = ["SynthConfig", "DiameterBin", "FAMILY", "BACKGROUND", "run_synthpop"]
__version__ = "0.1.0"
