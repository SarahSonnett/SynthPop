#!/usr/bin/env python
"""Build a proper-elements file for a designated background selection.

Cross-matches a PyLEADER background object list (``BGobjs_<REGION>_<TYPE>type_
neowise.txt``: columns number, designation, packed id, D, pV) against a
proper-elements catalog and writes the six-column file (``id, H, a, e, sin(i),
0``) that ``run_synthpop.py --family`` accepts — so a background population can
be generated through the same empirical member-resampling machinery as a
collisional family, modeling exactly the selection PyLEADER analyzed.

Example:

  python scripts/build_background_elements.py \\
      --bgobjs ~/Desktop/work/MBA_SFDs/BGobjs_IB_Ctype_neowise.txt \\
      --catalog ~/Desktop/work/MBA_SFDs/proper_catalog24.dat \\
      --out BG_IB_Ctypes_elements.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthpop.family_elements import build_elements_from_ids, DEFAULT_CATALOG_COLS  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bgobjs", type=Path, required=True,
                   help="background object list (number, designation, ... rows)")
    p.add_argument("--catalog", type=Path, required=True,
                   help="proper-elements catalog, e.g. proper_catalog24.dat")
    p.add_argument("--out", type=Path, required=True,
                   help="output element file")
    p.add_argument("--cols", default=None,
                   help="override the catalog column mapping as "
                        "'id=11,a=0,e=2,sini=4,H=8' (0-based; default matches "
                        f"proper_catalog24.dat: {DEFAULT_CATALOG_COLS})")
    args = p.parse_args(argv)

    ids = []
    with open(args.bgobjs.expanduser()) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            # prefer the number (col 0); fall back to the designation (col 1)
            ids.append(parts[0] if parts[0].isdigit() and int(parts[0]) > 0
                       else (parts[1] if len(parts) > 1 else parts[0]))

    cols = None
    if args.cols:
        cols = {k.strip(): int(v) for k, v in
                (item.split("=") for item in args.cols.split(","))}

    build_elements_from_ids(ids, args.catalog.expanduser(),
                            args.out.expanduser(), label=0, catalog_cols=cols,
                            what=args.bgobjs.name)


if __name__ == "__main__":
    main()
