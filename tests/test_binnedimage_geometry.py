"""Regression tests for viewing geometry retained by ``BinnedImage``."""

import numpy as np

from icbuilder.binnedimage import BinnedImage


class _OneCellGrid:
    """Minimal grid interface needed by ``BinnedImage`` for one test cell."""

    shape = (1, 1)
    size = 1
    xi = np.array([[0.0]])
    eta = np.array([[0.0]])

    def ingrid(self, lon, lat):
        return np.isfinite(lon) & np.isfinite(lat)

    def count(self, lon, lat):
        return np.array([[lon.size]], dtype=float)

    def bin_index(self, lon, lat):
        return (
            np.zeros(lon.size, dtype=int),
            np.zeros(lat.size, dtype=int),
        )


class _PreImage:
    """Small deterministic ``PreImage`` stand-in for geometry binning."""

    shape = (1, 1, 3)
    ssalon = np.array([0.0])

    _img = np.array([[[10.0, 20.0, 30.0]]])
    _sza = np.array([[[80.0, 90.0, 100.0]]])
    _dza = np.array([[[0.0, 60.0, 60.0]]])
    _weight = np.ones_like(_img)
    _lat = np.full_like(_img, 70.0)
    _mlt = np.zeros_like(_img)

    def get_mcoords(self, i):
        zeros = np.zeros_like(self._lat[i])
        return self._lat[i], zeros, self._mlt[i], self.ssalon[i]

    def get_img(self, i):
        return self._img[i]

    def get_img_los(self, i):
        return self._img[i] * np.cos(np.radians(self._dza[i]))

    def get_dgw(self, i):
        return self._weight[i]

    def get_shw(self, i):
        return self._weight[i]

    def get_SZA(self, i):
        return self._sza[i]

    def get_DZA(self, i):
        return self._dza[i]


def test_binned_geometry_and_los_factor_are_preserved():
    """Geometry uses the same pixels as the corrected image statistic."""

    binned = BinnedImage(
        _PreImage(),
        _OneCellGrid(),
        correction=None,
        los_correction=True,
    )

    # Corrected pixels are [10, 10, 15].
    assert np.isclose(binned.mu[0, 0, 0], 10.0)
    assert binned.sza[0, 0, 0] == 90.0
    assert binned.dza[0, 0, 0] == 60.0
    assert np.isclose(binned.los_factor[0, 0, 0], 0.5)
    assert binned.los_correction is True


def test_geometry_is_retained_when_los_correction_is_disabled():
    """Turning correction off changes brightness, not geometry provenance."""

    binned = BinnedImage(
        _PreImage(),
        _OneCellGrid(),
        correction=None,
        los_correction=False,
    )

    assert binned.mu[0, 0, 0] == 20.0
    assert binned.sza[0, 0, 0] == 90.0
    assert binned.dza[0, 0, 0] == 60.0
    assert np.isclose(binned.los_factor[0, 0, 0], 0.5)
    assert binned.los_correction is False
