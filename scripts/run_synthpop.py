#!/usr/bin/env python
"""CLI for the SynthPop synthetic-population pipeline.

Two ways to run:

  # From a saved JSON config
  python scripts/run_synthpop.py --config myconfig.json

  # Built from command-line flags (PyLEADER analysis dirs auto-discovered)
  python scripts/run_synthpop.py \
      --name BG_IB_Ctypes --mode background \
      --leader-base ~/Desktop/work/MBA_SFDs --leader-tag BG_IB_Ctypes \
      --neowise ~/Desktop/work/MBA_SFDs/neowise_mainbelt.csv \
      --s3m ~/data/s3m_innerbelt.txt \
      --n 10000 --out synthpop_out

  python scripts/run_synthpop.py \
      --name Hygiea --mode family \
      --leader-base ~/Desktop/work/MBA_SFDs --leader-tag Fam10 \
      --neowise ~/Desktop/work/MBA_SFDs/neowise_mainbelt.csv \
      --family ~/Desktop/work/MBA_SFDs/AFP_families/a10v50.txt \
      --n 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthpop import SynthConfig, run_synthpop          # noqa: E402
from synthpop.pyleader_io import discover_diameter_bins  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, help="load a saved SynthConfig JSON (overrides flags)")
    p.add_argument("--name")
    p.add_argument("--mode", choices=["family", "background"], default="background")
    p.add_argument("--n", type=int, default=10_000, dest="n_objects")
    p.add_argument("--seed", type=int, default=0)

    # PyLEADER distributions
    p.add_argument("--leader-base", type=Path, help="dir containing *_analysis_* bins")
    p.add_argument("--leader-tag", help="subpopulation tag, e.g. BG_IB_Ctypes or Fam10")

    # SFD
    p.add_argument("--dmin", type=float, default=1.0)
    p.add_argument("--dmax", type=float, default=100.0)
    p.add_argument("--slope", type=float, default=2.5)
    p.add_argument("--sfd", choices=["powerlaw", "loguniform"], default="powerlaw",
                   dest="sfd_kind",
                   help="diameter sampling law: 'powerlaw' (realistic SFD shape) "
                        "or 'loguniform' (equal counts per log-D bin -- for "
                        "detection-efficiency/debiasing work where large-object "
                        "statistics matter and the SFD shape does not)")

    # orbits
    p.add_argument("--family", type=Path, help="AFP family members file (family mode)")
    p.add_argument("--s3m", type=Path, help="s3m model file (background mode)")

    # physical
    p.add_argument("--neowise", type=Path, help="PDS SBN NEOWISE diameters/albedos CSV")
    p.add_argument("--family-orbit-model", choices=["dorb", "resample"],
                   default="dorb", dest="family_orbit_model",
                   help="family orbit generator: 'dorb' (Gaussian family + "
                        "D_orb(D) scaling, Schunova-Lilly+2017; default) or "
                        "'resample' (size-matched bootstrap of real member "
                        "elements with kernel smoothing -- preserves "
                        "multimodal/resonance-sculpted structure)")
    p.add_argument("--albedo-model",
                   choices=["gaussian", "lognormal", "double-lognormal",
                            "double-rayleigh", "mixture"],
                   default=None,
                   help="albedo fit model (default: auto -- 'lognormal' for "
                        "family mode, upgraded to 'double-lognormal' when the "
                        "member albedos test bimodal; 'double-rayleigh' for "
                        "background mode; 'gaussian' fits in linear albedo; "
                        "'mixture' fits a BIC-selected k-component lognormal "
                        "mixture on the UNCLIPPED sample, representing the "
                        "interloper dark tail that the default's sigma-clip "
                        "deliberately excludes)")
    p.add_argument("--eta-fitted-only", action="store_true",
                   help="fit the beaming parameter only to rows whose eta was "
                        "actually FIT by NEOWISE (fit-code 'B'), excluding the "
                        "quantized assumed values (sensitivity option)")
    p.add_argument("--albedo-min", type=float, default=0.01,
                   help="physical floor on sampled pV (default 0.01: albedos "
                        "below ~1%% have never been measured)")
    p.add_argument("--restrict-neowise", action="store_true",
                   help="fit albedo/eta only to the family's own members (family "
                        "mode; numbered members of --family cross-matched to "
                        "--neowise) instead of all NEOWISE MBAs")

    # output
    p.add_argument("--out", type=Path, default=Path("synthpop_out"))
    p.add_argument("--format", choices=["csv", "parquet"], default="csv", dest="out_format")
    p.add_argument("--no-plots", action="store_true",
                   help="skip the diagnostic PNGs (orbits, SFD, shape/pole, physical)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.config:
        cfg = SynthConfig.from_json(args.config)
    else:
        if not (args.name and args.leader_base and args.leader_tag):
            build_parser().error("--name, --leader-base and --leader-tag are required "
                                 "unless --config is given")
        bins = discover_diameter_bins(args.leader_base, args.leader_tag)
        if not bins:
            build_parser().error(
                f"no PyLEADER analysis bins found for tag {args.leader_tag!r} "
                f"under {args.leader_base}")

        member_numbers = None
        if args.restrict_neowise:
            if not args.family:
                build_parser().error("--restrict-neowise requires --family")
            # numbered members only; provisional-designation rows can't be
            # cross-matched to the NEOWISE catalog by number
            ids = np.genfromtxt(args.family, usecols=[0], dtype=str)
            member_numbers = [int(s) for s in ids if s.lstrip("-").isdigit()]

        cfg = SynthConfig(
            name=args.name, mode=args.mode, n_objects=args.n_objects, seed=args.seed,
            diam_min_km=args.dmin, diam_max_km=args.dmax, sfd_slope=args.slope,
            sfd_kind=args.sfd_kind,
            diameter_bins=bins,
            family_members_file=args.family,
            family_orbit_model=args.family_orbit_model,
            s3m_file=args.s3m,
            neowise_file=args.neowise,
            neowise_member_numbers=member_numbers,
            albedo_model=args.albedo_model,
            albedo_min=args.albedo_min,
            eta_fitted_only=args.eta_fitted_only,
            make_plots=not args.no_plots,
            out_dir=args.out, out_format=args.out_format,
        )

    run_synthpop(cfg)


if __name__ == "__main__":
    main()
