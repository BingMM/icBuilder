"""Exact regression tests for the serial ``BinnedImage`` optimizations."""

from datetime import datetime

import numpy as np
from scipy.stats import chi2, t

import icbuilder.binnedimage as binned_module
from icbuilder.binnedimage import BinnedImage


class _SyntheticGrid:
    """Small grid whose longitude/latitude directly identify each cell."""

    shape = (2, 3)
    size = 6

    def ingrid(self, lon, lat):
        return (
            np.isfinite(lon)
            & np.isfinite(lat)
            & (lon >= 0)
            & (lon < self.shape[1])
            & (lat >= 0)
            & (lat < self.shape[0])
        )

    def bin_index(self, lon, lat):
        lon = np.asarray(lon).flatten()
        lat = np.asarray(lat).flatten()
        inside = self.ingrid(lon, lat)
        j = np.full(lon.size, -1, dtype=int)
        k = np.full(lon.size, -1, dtype=int)
        j[inside] = lat[inside].astype(int)
        k[inside] = lon[inside].astype(int)
        return j, k

    def count(self, lon, lat):
        j, k = self.bin_index(lon, lat)
        flat_bin = j[j >= 0] * self.shape[1] + k[k >= 0]
        return np.bincount(flat_bin, minlength=self.size).reshape(self.shape)


class _SyntheticPreImage:
    """Two frames containing empty, partial, negative, and NaN bins."""

    shape = (2, 3, 4)
    sensor = "WIC"
    ssalon = np.array([0.0, 1.0])

    _lon = np.array([
        [[0.2, 0.2, 0.2, 1.2], [1.2, 2.2, 2.2, 0.2],
         [0.2, 1.2, 2.2, 3.2]],
        [[0.2, 0.2, 1.2, 1.2], [2.2, 2.2, 0.2, 0.2],
         [1.2, 1.2, 2.2, 2.2]],
    ])
    _lat = np.array([
        [[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 1.2],
         [1.2, 1.2, 1.2, 1.2]],
        [[0.2, 0.2, 0.2, 0.2], [0.2, 0.2, 1.2, 1.2],
         [1.2, 1.2, 1.2, 1.2]],
    ])
    _img = np.array([
        [[1.0, np.nan, 3.0, np.nan], [4.0, -4.0, -2.0, 8.0],
         [10.0, 12.0, 14.0, 99.0]],
        [[5.0, 7.0, 9.0, 11.0], [13.0, np.nan, 15.0, 17.0],
         [19.0, 21.0, 23.0, 25.0]],
    ])
    _weight = np.arange(24, dtype=float).reshape(shape)
    _weight[0, 0, 1] = np.nan
    _sza = 60.0 + np.arange(24, dtype=float).reshape(shape)
    _sza[1, 2, 0] = np.nan
    _dza = 10.0 + np.arange(24, dtype=float).reshape(shape)
    _dza[1, 2, 1] = np.nan

    def get_mcoords(self, i):
        zeros = np.zeros_like(self._lat[i])
        return self._lat[i], zeros, self._lon[i] / 15.0, self.ssalon[i]

    def get_img(self, i):
        return self._img[i]

    def get_dgw(self, i):
        return self._weight[i]

    def get_shw(self, i):
        return np.ones_like(self._weight[i])

    def get_SZA(self, i):
        return self._sza[i]

    def get_DZA(self, i):
        return self._dza[i]


def _old_binning(preimage, grid):
    """Frozen reference calculation from the former cell-scanning loop."""

    shape = (preimage.shape[0], *grid.shape)
    result = {
        "counts": np.zeros(shape),
        "mu": np.full(shape, np.nan),
        "sigma": np.full(shape, np.nan),
        "w": np.full(shape, np.nan),
        "sza": np.full(shape, np.nan),
        "dza": np.full(shape, np.nan),
        "los_factor": np.full(shape, np.nan),
    }

    for i in range(preimage.shape[0]):
        lat, _, mlt, _ = preimage.get_mcoords(i)
        lon = mlt * 15
        inside = grid.ingrid(lon, lat)
        result["counts"][i] = grid.count(lon[inside], lat[inside])
        image = preimage.get_img(i)
        weights = preimage.get_dgw(i) * preimage.get_shw(i)
        sza = preimage.get_SZA(i)
        dza = preimage.get_DZA(i)
        j, k = grid.bin_index(lon, lat)

        for jj in range(grid.shape[0]):
            for kk in range(grid.shape[1]):
                index = (i, jj, kk)
                if result["counts"][index] < 2:
                    continue
                mask = (j == jj) & (k == kk)
                values = image.flatten()[mask]
                image_nan = np.isnan(values)
                if np.sum(~image_nan) < 2:
                    continue
                if np.any(image_nan):
                    result["counts"][index] -= np.sum(image_nan)

                result["mu"][index] = max(np.nanmedian(values), 0)
                result["sigma"][index] = np.nanstd(values)
                valid_image = ~image_nan
                sza_values = sza.flatten()[mask][valid_image]
                dza_values = dza.flatten()[mask][valid_image]
                finite_sza = np.isfinite(sza_values)
                finite_dza = np.isfinite(dza_values)
                if np.any(finite_sza):
                    result["sza"][index] = np.median(sza_values[finite_sza])
                if np.any(finite_dza):
                    result["dza"][index] = np.median(dza_values[finite_dza])
                    result["los_factor"][index] = np.median(
                        np.cos(np.radians(dza_values[finite_dza]))
                    )
                result["w"][index] = np.nanmedian(weights.flatten()[mask])

    return result


def test_grouped_binning_is_exactly_equal_to_cell_scanning():
    """Grouping populated bins must preserve every historical output field."""

    preimage = _SyntheticPreImage()
    grid = _SyntheticGrid()
    reference = _old_binning(preimage, grid)
    binned = BinnedImage(
        preimage,
        grid,
        [datetime(2001, 1, 1), datetime(2001, 1, 1, 0, 1)],
        correction=None,
        los_correction=False,
    )

    for name, expected in reference.items():
        np.testing.assert_array_equal(getattr(binned, name), expected)


def _old_uncertainty(counts, sigma, alpha_mean=0.32, alpha_std=0.32):
    """Frozen scalar uncertainty loop used before multiplier caching."""

    inflated = sigma.copy()
    for index in np.ndindex(counts.shape):
        df = counts[index] - 1
        if df < 1:
            continue
        t_multiplier = t.ppf(1 - alpha_mean / 2, df)
        mean_unc = t_multiplier * inflated[index] / np.sqrt(counts[index])
        chi2_lower = chi2.ppf(alpha_std / 2, df)
        std_inflation = inflated[index] * np.sqrt(df / chi2_lower)
        inflated[index] = np.sqrt(mean_unc**2 + std_inflation**2)
    return inflated


def test_cached_uncertainty_multipliers_preserve_scalar_result(monkeypatch):
    """Distribution functions run once per count without changing any bit."""

    counts = np.array([[[0, 1, 2], [2, 4, 4]], [[8, 2, 1], [8, 4, 0]]],
                      dtype=float)
    sigma = np.arange(12, dtype=float).reshape(counts.shape) / 10
    sigma[0, 1, 2] = np.nan
    expected = _old_uncertainty(counts, sigma)

    calls = {"t": 0, "chi2": 0}
    original_t_ppf = binned_module.t.ppf
    original_chi2_ppf = binned_module.chi2.ppf

    def count_t_calls(*args, **kwargs):
        calls["t"] += 1
        return original_t_ppf(*args, **kwargs)

    def count_chi2_calls(*args, **kwargs):
        calls["chi2"] += 1
        return original_chi2_ppf(*args, **kwargs)

    monkeypatch.setattr(binned_module.t, "ppf", count_t_calls)
    monkeypatch.setattr(binned_module.chi2, "ppf", count_chi2_calls)

    binned = BinnedImage.__new__(BinnedImage)
    binned.counts = counts.copy()
    binned.sigma = sigma.copy()
    binned.shape = counts.shape
    binned._inflate_uncertainty()

    np.testing.assert_array_equal(binned.sigma, expected)
    assert calls == {"t": 3, "chi2": 3}

