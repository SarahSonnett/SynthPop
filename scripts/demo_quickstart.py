"""Self-contained quickstart: generate a synthetic asteroid family.

Runs entirely from this repository — no external data needed. Fabricates a
small but realistic input set for a *fictitious* collisional family
("Demonstranda", family 9999): a PyLEADER-style joint shape/spin distribution
`f_mean(p, beta)` in two diameter bins, a proper-elements member file with
Yarkovsky-like size-dependent dispersion, and a NEOWISE-style table of
measured diameters, albedos, and beaming parameters. It then runs the full
SynthPop pipeline against those inputs — joint (b/a, |beta|) sampling,
Schunova-Lilly family orbit generation, lognormal albedo + truncated-Gaussian
eta fits — and writes the catalog, provenance, and the four diagnostic
figures.

    python scripts/demo_quickstart.py        # ~30 seconds
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthpop import SynthConfig, run_synthpop
from synthpop.pyleader_io import discover_diameter_bins

rng = np.random.default_rng(9)
tmp = Path(tempfile.mkdtemp(prefix="synthpop_demo_"))

# --- 1. A PyLEADER-style joint (p, beta) distribution, two diameter bins ----
# p = b/a (shape elongation), beta = |spin-pole ecliptic latitude| in degrees.
# Small members are modeled slightly more elongated than large ones, with a
# mild p-beta correlation — the kind of structure PyLEADER recovers.
p_grid = np.linspace(0.30, 0.95, 12)
b_grid = np.linspace(3.75, 86.25, 12)
for lo, hi, p0 in [(1, 5, 0.66), (5, 25, 0.76)]:
    P, B = np.meshgrid(p_grid, b_grid, indexing="ij")
    beta0 = 30.0 + 25.0 * (p0 - P)          # more elongated -> lower |beta|
    f = np.exp(-0.5 * ((P - p0) / 0.11) ** 2 - 0.5 * ((B - beta0) / 24.0) ** 2)
    d = tmp / "leader" / f"Fam9999_demo_V_{lo}km_to_{hi}km_summary"
    d.mkdir(parents=True)
    np.savez(d / "population_distribution.npz",
             p_grid=p_grid, b_grid=b_grid, f_mean=f / f.sum())

# --- 2. A proper-elements member file (id, H, a, e, sin i, famid) -----------
N_MEM = 400
a0, e0, sini0 = 2.62, 0.150, 0.145
H = rng.uniform(12.5, 17.5, N_MEM)
D_true = 1329.0 / np.sqrt(0.20) * 10 ** (-H / 5.0)   # km, at the median albedo
spread = (D_true / 5.0) ** -0.25                     # Yarkovsky-like: small = dispersed
a = a0 + rng.normal(0, 0.012, N_MEM) * spread
e = e0 + rng.normal(0, 0.010, N_MEM) * spread
sini = sini0 + rng.normal(0, 0.008, N_MEM) * spread
members = tmp / "a9999v50.txt"
np.savetxt(members, np.column_stack([np.arange(1, N_MEM + 1), H, a, e, sini,
                                     np.full(N_MEM, 9999)]),
           fmt=["%d", "%.2f", "%.6f", "%.6f", "%.6f", "%d"])

# --- 3. A NEOWISE-style table (PDS column order) for the pV / eta fits ------
# 70% of members carry a NEOWISE row; beaming is genuinely fit (fit code 'B')
# for 60% of those, and takes the assumed value eta = 1.0 for the rest.
det = rng.random(N_MEM) < 0.70
num = np.arange(1, N_MEM + 1)[det]
pV = np.exp(rng.normal(np.log(0.20), 0.30, num.size))
D_meas = 1329.0 / np.sqrt(pV) * 10 ** (-H[det] / 5.0)
fitB = rng.random(num.size) < 0.60
eta = np.where(fitB, np.clip(rng.normal(1.05, 0.18, num.size), 0.4, 2.5), 1.0)
neowise = tmp / "neowise_demo.csv"
with open(neowise, "w") as f:
    for k in range(num.size):
        row = [""] * 19
        row[0] = f"{num[k]:d}"; row[3] = f"{H[det][k]:.2f}"
        row[10] = '"DVBI"' if fitB[k] else '"DVI"'
        row[11] = f"{D_meas[k]:.3f}"; row[12] = f"{0.1 * D_meas[k]:.3f}"
        row[13] = f"{pV[k]:.4f}"; row[14] = f"{0.2 * pV[k]:.4f}"
        row[17] = f"{eta[k]:.3f}"; row[18] = "0.100"
        f.write(",".join(row) + "\n")

# --- 4. Run the full pipeline ------------------------------------------------
cfg = SynthConfig(
    name="Demonstranda", mode="family", n_objects=4000,
    sfd_kind="loguniform", diam_min_km=1.0, diam_max_km=25.0,
    diameter_bins=discover_diameter_bins(tmp / "leader", "Fam9999"),
    family_members_file=members, neowise_file=neowise,
    neowise_member_numbers=[int(n) for n in num],
    seed=7, out_dir=Path("synthpop_out/Demonstranda"),
)
out = run_synthpop(cfg)
print(f"\nDemo family written under {out}")
print("Open the four Demonstranda_*.png figures to see the synthetic "
      "population verified against its (fabricated) inputs.")
