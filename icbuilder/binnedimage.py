#%% Imports

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid
from copy import deepcopy as dcopy
from scipy.stats import t, chi2
from scipy.interpolate import griddata
from .preimage import PreImage

#%% BinnedImage class

class BinnedImage:
    """
    A class to bin IMAGE data onto a CSgrid.

    This class processes a `PreImage` object and computes binned statistics
    (median and standard deviation) for each grid cell, optionally inflating the
    uncertainties or interpolating to a different grid.

    Attributes
    ----------
    grid : CSgrid
        The CSgrid to which data is binned (or interpolated).
    counts : np.ndarray
        Number of samples contributing to each grid cell.
    mu : np.ndarray
        Mean (median) of the binned values in each grid cell.
    sigma : np.ndarray
        Standard deviation of the values in each grid cell.
    sza : np.ndarray
        Median solar zenith angle of contributing pixels [degrees].
    dza : np.ndarray
        Median detector zenith angle of contributing pixels [degrees].
    los_factor : np.ndarray
        Median ``cos(DZA)`` of contributing pixels. This is a diagnostic
        summary of the pixel-level correction factors, not an exact factor
        relating the corrected and uncorrected binned medians.
    los_correction : bool
        Whether the pixel-level ``cos(DZA)`` correction was applied.
    shape : tuple
        Shape of the binned data arrays.
    """
    
    def __init__(self,
                 pI: PreImage,
                 grid: CSgrid,
                 target_grid: Optional[CSgrid] = None,
                 inflate_uncertainty: bool = False,
                 correction: Optional[str] = None,
                 los_correction: bool = True
                 ):
        """
        Bin statistics from a PreImage object into a CSgrid.

        Parameters
        ----------
        pI : PreImage
            Input IMAGE data to bin.
        grid : CSgrid
            Cubbed sphere grid to bin onto.
        target_grid : CSgrid, optional
            If provided, interpolate results onto this cubbed sphere grid.
        inflate_uncertainty : bool
            If True, inflates uncertainties using t and chi² statistics.
            Should be used when less than 30 counts per bin.
        correction : {"SH", "DG"} or None
            Select the background-corrected image to bin. ``None`` bins the
            raw image.
        los_correction : bool
            If True, multiply each source pixel by ``cos(DZA)`` before
            calculating the binned image statistics.
        """
        self.grid = dcopy(grid)
        self.correction = correction
        self.los_correction = bool(los_correction)

        time_len, ny, nx = pI.shape[0], grid.shape[0], grid.shape[1]
        self.counts = np.zeros((time_len, ny, nx))
        self.mu = np.full_like(self.counts, np.nan)
        self.sigma = np.full_like(self.counts, np.nan)
        self.w = np.full_like(self.counts, np.nan)
        self.sza = np.full_like(self.counts, np.nan)
        self.dza = np.full_like(self.counts, np.nan)
        self.los_factor = np.full_like(self.counts, np.nan)
        self.shape = self.counts.shape
        self.ssalon = pI.ssalon
        
        for i in range(time_len):
            #lat, lon = pI.get_mcoords(i) # Get magnetic coordinates
            lat, _, mlt, _ = pI.get_mcoords(i) # Get magnetic coordinates
            lon = mlt*15
            if correction == 'SH':
                if los_correction:
                    img = pI.get_shimg_los(i) # Get the SH corrected image
                else:
                    img = pI.get_shimg(i) # Get the SH corrected image
            elif correction == 'DG':
                if los_correction:
                    img = pI.get_dgimg_los(i) # Get the DG corrected image
                else:
                    img = pI.get_dgimg(i) # Get the DG corrected image
            else:
                if los_correction:
                    img = pI.get_img_los(i) # Get the LOS-corrected raw image
                else:
                    img = pI.get_img(i) # Get the raw image
            w = pI.get_dgw(i) * pI.get_shw(i) # Get and combine weights
            sza = pI.get_SZA(i)
            dza = pI.get_DZA(i)

            # ``bin_index`` returns one flattened grid-cell index per source
            # pixel. Sort those indices once, preserving source-pixel order
            # within each cell, and then work only on populated cells. The old
            # implementation scanned every source pixel for every grid cell.
            j, k = grid.bin_index(lon, lat)
            valid_bin = (j >= 0) & (j < ny) & (k >= 0) & (k < nx)
            source_index = np.flatnonzero(valid_bin)
            flat_bin = j[valid_bin] * nx + k[valid_bin]
            order = np.argsort(flat_bin, kind='stable')
            source_index = source_index[order]
            flat_bin = flat_bin[order]

            # Flatten each source field once. A stable bin sort means that the
            # values inside a cell retain their original order, preserving the
            # floating-point arithmetic of the previous implementation.
            img_flat = img.flatten()
            w_flat = w.flatten()
            sza_flat = sza.flatten()
            dza_flat = dza.flatten()

            if flat_bin.size:
                starts = np.r_[0, np.flatnonzero(np.diff(flat_bin)) + 1]
                stops = np.r_[starts[1:], flat_bin.size]
            else:
                starts, stops = [], []

            for start, stop in zip(starts, stops):
                jj, kk = divmod(flat_bin[start], nx)
                id_ = (i, jj, kk)
                indices = source_index[start:stop]

                # Keep the historical count semantics: cells begin with the
                # number of geometrically valid pixels. Counts are reduced for
                # NaN image values only when at least two image values remain.
                self.counts[id_] = stop - start
                if self.counts[id_] < 2:
                    continue

                values = img_flat[indices]
                fnan = np.isnan(values)
                if np.sum(~fnan) < 2:
                    continue
                if np.any(fnan):
                    self.counts[id_] -= np.sum(fnan)

                median_val = np.nanmedian(values)
                self.mu[id_] = max(median_val, 0)  # zero if negative
                self.sigma[id_] = np.nanstd(values)

                # Preserve viewing geometry for the same source pixels that
                # contributed to the image statistics. The median cosine is a
                # diagnostic because correction occurs before image binning.
                valid_image = ~fnan
                sza_values = sza_flat[indices][valid_image]
                dza_values = dza_flat[indices][valid_image]
                finite_sza = np.isfinite(sza_values)
                finite_dza = np.isfinite(dza_values)
                if np.any(finite_sza):
                    self.sza[id_] = np.median(sza_values[finite_sza])
                if np.any(finite_dza):
                    self.dza[id_] = np.median(dza_values[finite_dza])
                    self.los_factor[id_] = np.median(
                        np.cos(np.radians(dza_values[finite_dza]))
                    )

                # Weights retain their historical independent NaN handling;
                # an image NaN does not remove the corresponding weight.
                self.w[id_] = np.nanmedian(w_flat[indices])
                            
        if inflate_uncertainty:
            self._inflate_uncertainty()

        if target_grid is not None:
            self._interpolate(target_grid)
            self.shape = self.mu.shape

    def _inflate_uncertainty(self, 
                             alpha_mean: float = 0.32, 
                             alpha_std: float = 0.32):
        """
        Inflate the uncertainty estimates using Student's t-distribution
        and the chi-squared distribution.

        Parameters
        ----------
        alpha_mean : float
            Confidence level for inflating the uncertainty on the mean. .32 = 68% = 1 std
        alpha_std : float
            Confidence level for inflating the standard deviation. .32 = 68% = 1 std
        """
        # Counts are integers stored in a floating-point array. Most cells
        # share one of only a few counts, so evaluate the expensive probability
        # distributions once per count and reuse the scalar multipliers.
        valid_counts = np.unique(self.counts[self.counts >= 2]).astype(int)
        multipliers = {}
        for count in valid_counts:
            df = count - 1
            multipliers[count] = (
                t.ppf(1 - alpha_mean / 2, df),
                np.sqrt(df / chi2.ppf(alpha_std / 2, df)),
            )

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    count = self.counts[i, j, k]
                    df = count - 1
                    if df < 1:
                        continue
                    t_multiplier, std_multiplier = multipliers[int(count)]
                    mean_unc = t_multiplier * self.sigma[i, j, k] / np.sqrt(count)
                    std_inflation = self.sigma[i, j, k] * std_multiplier
                    self.sigma[i, j, k] = np.sqrt(mean_unc**2 + std_inflation**2)

    def _interpolate(self, 
                     target_grid: CSgrid):
        """
        Interpolate image, uncertainty, weight, and viewing-geometry fields
        from the current grid to a new grid.

        Parameters
        ----------
        target_grid : CSgrid
            The grid to interpolate onto.
        """
        self.mu_    = np.copy(self.mu)
        self.sigma_ = np.copy(self.sigma)
        self.w_     = np.copy(self.w)
        self.sza_   = np.copy(self.sza)
        self.dza_   = np.copy(self.dza)
        self.los_factor_ = np.copy(self.los_factor)

        time_len    = self.shape[0]
        ny, nx      = target_grid.shape

        self.mu     = np.full((time_len, ny, nx), np.nan)
        self.sigma  = np.full((time_len, ny, nx), np.nan)
        self.w      = np.full((time_len, ny, nx), np.nan)
        self.sza    = np.full((time_len, ny, nx), np.nan)
        self.dza    = np.full((time_len, ny, nx), np.nan)
        self.los_factor = np.full((time_len, ny, nx), np.nan)

        source_fields = {
            'mu': self.mu_,
            'sigma': self.sigma_,
            'w': self.w_,
            'sza': self.sza_,
            'dza': self.dza_,
            'los_factor': self.los_factor_,
        }

        for i in range(time_len):
            # Fields can share one triangulation only when they have exactly
            # the same non-NaN source cells. Group equal masks to gain that
            # reuse without discarding valid values from another field. If all
            # masks differ, this naturally reproduces six independent calls.
            mask_groups = {}
            for name, values in source_fields.items():
                mask = ~np.isnan(values[i])
                mask_groups.setdefault(mask.tobytes(), []).append((name, mask))

            for group in mask_groups.values():
                names = [name for name, _ in group]
                mask = group[0][1]
                points = (self.grid.xi[mask], self.grid.eta[mask])

                if len(names) == 1:
                    values = source_fields[names[0]][i][mask]
                else:
                    values = np.column_stack([
                        source_fields[name][i][mask] for name in names
                    ])

                interpolated = griddata(
                    points, values,
                    (target_grid.xi, target_grid.eta), method='linear',
                    fill_value=np.nan
                )

                if len(names) == 1:
                    getattr(self, names[0])[i] = interpolated
                else:
                    for column, name in enumerate(names):
                        getattr(self, name)[i] = interpolated[..., column]

        self.grid = dcopy(target_grid)

    def discard(self, f: Union[list[int], NDArray[np.int_]]):
        """
        Discard all time steps NOT listed in `f`.

        Parameters
        ----------
        f : list[int] or NDArray[np.int_]
            Indices of time steps to retain.
        """
        self.counts = self.counts[f]
        self.mu = self.mu[f]
        self.sigma = self.sigma[f]
        self.w = self.w[f]
        self.sza = self.sza[f]
        self.dza = self.dza[f]
        self.los_factor = self.los_factor[f]
        self.shape = self.mu.shape
