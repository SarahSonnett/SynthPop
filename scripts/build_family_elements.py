#!/usr/bin/env python
"""Build a family proper-elements file for families without a portal file.

Cross-matches a family membership list against a proper-elements catalog and
writes the six-column file (``id, H, a, e, sin(i), famid``) that
``run_synthpop.py --family`` expects. Intended for the 136 families of
Nesvorny et al. (2024), which have membership lists but no Asteroid Families
Portal element file.

Example (family 1312, one of the Nesvorny et al. 2024 identifications):

  python scripts/build_family_elements.py --famid 1312 \\
      --members ~/Desktop/work/MBA_SFDs/AllMBAFamilyMembers.txt \\
      --catalog ~/Desktop/work/MBA_SFDs/proper_catalog24.dat \\
      --out a1312_nesv24.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthpop.family_elements import build_family_elements, DEFAULT_CATALOG_COLS  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--famid", type=int, required=True,
                   help="family id (integer) to extract")
    p.add_argument("--members", type=Path, required=True,
                   help="membership list (famid, packed_id, designation rows), "
                        "e.g. AllMBAFamilyMembers.txt")
    p.add_argument("--catalog", type=Path, required=True,
                   help="proper-elements catalog, e.g. proper_catalog24.dat "
                        "(Nesvorny et al. 2024)")
    p.add_argument("--out", type=Path, default=None,
                   help="output element file (default a<famid>_synth.txt)")
    p.add_argument("--cols", default=None,
                   help="override the catalog column mapping as "
                        "'id=11,a=0,e=2,sini=4,H=8' (0-based; default matches "
                        f"proper_catalog24.dat: {DEFAULT_CATALOG_COLS})")
    args = p.parse_args(argv)

    cols = None
    if args.cols:
        cols = {k.strip(): int(v) for k, v in
                (item.split("=") for item in args.cols.split(","))}

    out = args.out or Path(f"a{args.famid}_synth.txt")
    build_family_elements(args.famid, args.members.expanduser(),
                          args.catalog.expanduser(), out.expanduser(),
                          catalog_cols=cols)


if __name__ == "__main__":
    main()
