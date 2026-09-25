#!/usr/bin/env python
"""Generate an ensemble of synthetic populations for systematic-error work.

Each ensemble member bootstrap-resamples the population's measured data
(NEOWISE albedos/beaming; family proper elements) and refits, so the spread of
Simmer's debiased SFD across members is the fit-uncertainty systematic band.
Member 0 is the nominal (unresampled) fit. See synthpop/ensemble.py.

Example (a 50-member, 50,000-object ensemble):

  python scripts/run_ensemble.py \\
      --name Lixiaohua --mode family \\
      --leader-base ~/pyleader_runs --leader-tag Fam3556 \\
      --neowise ~/Desktop/work/MBA_SFDs/neowise_mainbelt.csv \\
      --family ~/Desktop/work/MBA_SFDs/AFP_families/a3556v40.txt \\
      --restrict-neowise --sfd loguniform --family-orbit-model resample \\
      --n 50000 --n-members 50 --dmin 1 --dmax 40 \\
      --out ~/synthpop_runs/ensembles
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthpop import SynthConfig                          # noqa: E402
from synthpop.ensemble import EnsembleConfig, run_ensemble  # noqa: E402
from synthpop.pyleader_io import discover_diameter_bins     # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["family", "background"], default="family")
    p.add_argument("--leader-base", type=Path, required=True)
    p.add_argument("--leader-tag", required=True)
    p.add_argument("--neowise", type=Path, required=True)
    p.add_argument("--family", type=Path)
    p.add_argument("--s3m", type=Path)
    p.add_argument("--restrict-neowise", action="store_true")
    p.add_argument("--sfd", choices=["powerlaw", "loguniform"],
                   default="loguniform", dest="sfd_kind",
                   help="(default loguniform: ensembles exist for "
                        "detection-efficiency work)")
    p.add_argument("--family-orbit-model", choices=["dorb", "resample"],
                   default="dorb", dest="family_orbit_model")
    p.add_argument("--albedo-model", default=None,
                   choices=["gaussian", "lognormal", "double-lognormal",
                            "double-rayleigh", "mixture"])
    p.add_argument("--eta-fitted-only", action="store_true",
                   help="fit the beaming parameter only to rows whose eta was "
                        "actually FIT by NEOWISE (fit-code 'B'), excluding the "
                        "quantized assumed values (sensitivity option)")
    p.add_argument("--albedo-min", type=float, default=0.01)
    p.add_argument("--n", type=int, default=50_000, dest="n_objects")
    p.add_argument("--dmin", type=float, default=1.0)
    p.add_argument("--dmax", type=float, default=100.0)
    p.add_argument("--n-members", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bootstrap-orbits", action="store_true",
                   help="hold the orbital fit fixed across members")
    p.add_argument("--no-bootstrap-physical", action="store_true",
                   help="hold the albedo/beaming fits fixed across members")
    p.add_argument("--out", type=Path, default=Path("synthpop_out"),
                   help="parent dir; the ensemble lands in <out>/<name>_ensemble/")
    args = p.parse_args(argv)

    bins = discover_diameter_bins(args.leader_base, args.leader_tag)
    if not bins:
        p.error(f"no PyLEADER bins for tag {args.leader_tag!r} under {args.leader_base}")

    member_numbers = None
    if args.restrict_neowise:
        if not args.family:
            p.error("--restrict-neowise requires --family")
        ids = np.genfromtxt(args.family.expanduser(), usecols=[0], dtype=str)
        member_numbers = [int(s) for s in ids if s.lstrip("-").isdigit()]

    base = SynthConfig(
        name=args.name, mode=args.mode, n_objects=args.n_objects,
        diam_min_km=args.dmin, diam_max_km=args.dmax, sfd_kind=args.sfd_kind,
        diameter_bins=bins,
        family_members_file=args.family,
        family_orbit_model=args.family_orbit_model,
        s3m_file=args.s3m,
        neowise_file=args.neowise,
        neowise_member_numbers=member_numbers,
        albedo_model=args.albedo_model,
        albedo_min=args.albedo_min,
        eta_fitted_only=args.eta_fitted_only,
        make_plots=False,
        out_dir=args.out,
    )
    run_ensemble(EnsembleConfig(
        base=base, n_members=args.n_members, seed=args.seed,
        bootstrap_physical=not args.no_bootstrap_physical,
        bootstrap_orbits=not args.no_bootstrap_orbits,
    ))


if __name__ == "__main__":
    main()
