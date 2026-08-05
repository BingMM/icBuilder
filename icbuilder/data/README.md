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

`gfz_kp_2000_2003.json` is the response from the official GFZ JSON API for
the IMAGE interval plus a one-month buffer after the last June 2003 orbit.
It contains 10,464 uninterrupted
three-hour values, all with definitive (`def`) status.

- Acquired: 2026-08-05
- Source: GFZ Potsdam
- DOI: `10.5880/Kp.0001`
- Licence: CC BY 4.0
- SHA-256:
  `29218cf7ab50d7629c92f3fbd2709554dabe593e34f31d82b64bc75b29b2d362`
- Query:
  `https://kp.gfz.de/app/json/?start=2000-01-01T00:00:00Z&end=2003-07-31T23:59:59Z&index=Kp&status=def`

Re-download and validate the source response with:

```bash
python scripts/download_gfz_kp.py
```

Orbit processing reads this local file and never contacts GFZ.
The downloader and loader check the data shape, uninterrupted three-hour
cadence, definitive status, and physical Kp range. The loader calculates the
SHA-256 of the file it actually reads and includes it in output provenance.
This records the exact input without preventing a deliberate date-range
extension.
