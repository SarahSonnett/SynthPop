"""Build a family proper-elements file for families without a portal file.

The Asteroid Families Portal provides ``a<famid>v<vcut>.txt`` element files for
its 87 families only; the 136 families newly identified by Nesvorny et al.
(2024) have membership lists but no portal element file. This module
constructs an equivalent six-column file (``id, H, a, e, sin(i), famid``) by
cross-matching a family membership list against a proper-elements catalog, so
any family can be supplied to ``--family``.

Inputs
------
* **membership list** -- whitespace-delimited ``famid, packed_id, designation``
  rows (the format of ``AllMBAFamilyMembers.txt`` and
  ``Nesvorny2024_newfamilies.txt``).
* **proper-elements catalog** -- one row per object. The default column
  mapping matches the Nesvorny et al. (2024) catalog of proper orbits
  (``proper_catalog24.dat``): ``a=0, e=2, sini=4, H=8, id=11`` (0-based),
  with uncertainties and frequencies in the intervening columns. Any
  whitespace-delimited catalog can be used by overriding the mapping.

The output file is directly consumable by
:func:`synthpop.orbits_family.load_family_members`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Default 0-based column mapping for the Nesvorny et al. (2024) proper-orbit
# catalog (proper_catalog24.dat): a, da, e, de, sini, dsini, g, s, H, nobs?,
# packed, designation.
DEFAULT_CATALOG_COLS = {"a": 0, "e": 2, "sini": 4, "H": 8, "id": 11}


@dataclass
class BuildResult:
    out_path: Path
    n_members: int      # membership rows for the family
    n_matched: int      # members found in the proper-elements catalog
    n_unmatched: int


def _norm_id(s: str) -> str:
    """Normalise an object identifier for matching: strip, drop leading zeros
    on purely numeric ids (``00123`` == ``123``), case-fold designations."""
    s = s.strip()
    if s.lstrip("-").isdigit():
        return str(int(s))
    return s.upper()


def load_membership(members_file: Path, famid: int) -> list:
    """Designations of the members of ``famid`` from a membership list."""
    designations = []
    with open(members_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                fid = int(parts[0])
            except ValueError:
                continue
            if fid == famid:
                designations.append(parts[2])
    return designations


def build_elements_from_ids(ids, catalog_file: Path, out_path: Path,
                            label: int = 0, catalog_cols: dict | None = None,
                            verbose: bool = True,
                            what: str = "population") -> BuildResult:
    """Cross-match a list of object identifiers against a proper-elements
    catalog and write the six-column element file (``id, H, a, e, sin(i),
    label``). Used for families (via :func:`build_family_elements`) and for
    designated background selections (``scripts/build_background_elements.py``)."""
    cols = dict(DEFAULT_CATALOG_COLS)
    if catalog_cols:
        cols.update(catalog_cols)

    members = [str(i) for i in ids]
    if not members:
        raise ValueError(f"empty id list for {what}")
    wanted = {_norm_id(m): m for m in members}

    matched = {}
    need = max(cols.values()) + 1
    with open(catalog_file) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < need:
                continue
            key = _norm_id(parts[cols["id"]])
            if key in wanted:
                try:
                    matched[key] = (float(parts[cols["H"]]),
                                    float(parts[cols["a"]]),
                                    float(parts[cols["e"]]),
                                    float(parts[cols["sini"]]))
                except ValueError:
                    continue

    if not matched:
        raise ValueError(
            f"no members of {what} found in {catalog_file}; "
            "check the catalog column mapping")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as out:
        for key, desig in wanted.items():
            if key in matched:
                H, a, e, sini = matched[key]
                out.write(f"{desig:>11s} {H:8.2f} {a:13.6f} {e:13.6f} "
                          f"{sini:13.6f} {label:>9d}\n")

    res = BuildResult(out_path=out_path, n_members=len(members),
                      n_matched=len(matched),
                      n_unmatched=len(members) - len(matched))
    if verbose:
        print(f"[build_elements] {what}: "
              f"{res.n_matched}/{res.n_members} members matched in the "
              f"proper-elements catalog -> {out_path}"
              + (f" ({res.n_unmatched} unmatched)" if res.n_unmatched else ""))
    return res


def build_family_elements(famid: int, members_file: Path, catalog_file: Path,
                          out_path: Path, catalog_cols: dict | None = None,
                          verbose: bool = True) -> BuildResult:
    """Cross-match a family's members against a proper-elements catalog and
    write the six-column element file."""
    members = load_membership(members_file, famid)
    if not members:
        raise ValueError(f"no members of family {famid} in {members_file}")
    return build_elements_from_ids(members, catalog_file, out_path,
                                   label=famid, catalog_cols=catalog_cols,
                                   verbose=verbose, what=f"family {famid}")
