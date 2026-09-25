"""Write the synthetic-population database and its provenance to disk."""

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .config import SynthConfig


def write_catalog(df: pd.DataFrame, cfg: SynthConfig, provenance: dict | None = None) -> Path:
    """Write the catalog plus the config and a provenance summary.

    Returns the path to the catalog file. Output layout::

        <out_dir>/<name>_synthpop.csv      (or .parquet)
        <out_dir>/<name>_config.json
        <out_dir>/<name>_provenance.json
    """
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{cfg.name}_synthpop"

    if cfg.out_format == "parquet":
        cat_path = stem.with_suffix(".parquet")
        df.to_parquet(cat_path, index=False)
    else:
        cat_path = stem.with_suffix(".csv")
        df.to_csv(cat_path, index=False, float_format="%.6g")

    cfg.to_json(out_dir / f"{cfg.name}_config.json")
    if provenance is not None:
        (out_dir / f"{cfg.name}_provenance.json").write_text(
            json.dumps(provenance, indent=2, default=str)
        )
    return cat_path
