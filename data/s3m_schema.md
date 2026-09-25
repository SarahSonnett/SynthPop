# s3m model input schema

The background-orbit stage (`synthpop/orbits_bg.py`) reads orbital elements from
the **s3m** synthetic population model (Grav et al. 2011, *The Pan-STARRS
Synthetic Solar System Model*). One file per background region is expected:
inner / middle / pristine / outer Main Belt, Hungarias, and Cybeles.

`orbits_bg` does **not** assume a fixed column layout. It uses the mapping in
`SynthConfig.s3m_columns` (defaulting to the standard S3M cometary format) to
pick out the columns it needs. Whitespace-delimited; lines beginning with `!`
are treated as comments/headers and skipped.

## Supported element formats

Set `s3m_columns["format"]` to one of:

### `"COM"` (default) — cometary elements
| key       | meaning                              | units |
|-----------|--------------------------------------|-------|
| `q`       | perihelion distance                  | AU    |
| `e`       | eccentricity                         | —     |
| `i`       | inclination                          | deg   |
| `node`    | longitude of ascending node (Ω)      | deg   |
| `argperi` | argument of perihelion (ω)           | deg   |
| `t_peri`  | time of perihelion passage           | (any) |

Semi-major axis is recovered as `a = q / (1 - e)`. Mean anomaly is drawn
uniformly (the survey simulator re-propagates to its own epoch anyway).

### `"KEP"` — Keplerian elements
| key       | meaning                         | units |
|-----------|---------------------------------|-------|
| `a`       | semi-major axis                 | AU    |
| `e`       | eccentricity                    | —     |
| `i`       | inclination                     | deg   |
| `node`    | longitude of ascending node (Ω) | deg   |
| `argperi` | argument of perihelion (ω)      | deg   |
| `M`       | mean anomaly                    | deg   |

## Example mapping

If your inner-belt KEP file has columns
`id  H  a  e  i  node  argperi  M  epoch` (0-based indices), configure:

```python
cfg.s3m_file = "/path/to/s3m_innerbelt.kep"
cfg.s3m_columns = {
    "format": "KEP",
    "a": 2, "e": 3, "i": 4, "node": 5, "argperi": 6, "M": 7,
}
```

The defaults in `orbits_bg._DEFAULT_S3M_COLUMNS` assume a COM file with columns
`id q e i node argperi t_peri ...` (q at index 1). Adjust to match your files.
