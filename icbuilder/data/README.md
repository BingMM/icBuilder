# Zhang--Paxton lookup data

`zhang_paxton_e0_lookup.nc` is generated from the repository root with:

```bash
python scripts/make_zhang_paxton_lookup.py --workers 4
```

The NetCDF file contains direct latitude collapses for Kp 0.00--9.00 in
steps of 0.01 on the fixed 36-by-36 IMAGE Cubed-Sphere grid. It stores the
area-weighted mean electron energy `E0`, the profile-derived spread `dE0`, and
the area-weighted median `E0_median`. The ZhangPaxton2008 package version and
the provisional latitude-collapse settings are stored as file attributes.

## Definitive GFZ Kp

`gfz_kp_2000_2001.json` is the unchanged response from the official GFZ JSON
API for the documented IMAGE interval. It contains 5,848 uninterrupted
three-hour values, all with definitive (`def`) status.

- Acquired: 2026-07-30
- Source: GFZ Potsdam
- DOI: `10.5880/Kp.0001`
- Licence: CC BY 4.0
- SHA-256:
  `259cc539ac6578510ea9f54691bbe2f15913a19060db3e390f4490c49226f91e`
- Query:
  `https://kp.gfz.de/app/json/?start=2000-01-01T00:00:00Z&end=2001-12-31T23:59:59Z&index=Kp&status=def`

Re-download and validate the source response with:

```bash
python scripts/download_gfz_kp.py
```

Orbit processing reads this local file and never contacts GFZ.
The loader checks the SHA-256 before parsing the response, and the download
script refuses to replace it with bytes that do not match this acquisition.
