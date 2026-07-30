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
            f = grid.ingrid(lon, lat) # Find data inside the CS grid
            self.counts[i] = grid.count(lon[f], lat[f]) # Count the number of pixels in each bin

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

            j, k = grid.bin_index(lon, lat) # Make bin index
            for jj in range(ny):
                for kk in range(nx):
                    
                    # id
                    id_ = (i, jj, kk)
                    
                    # If less than 2 pixels in bin, continue
                    if self.counts[id_] < 2:
                        continue
                    
                    # Get index of data in a single bin
                    mask = (j == jj) & (k == kk)
                    
                    # Grab values in the (jj, kk) bin
                    values = img.flatten()[mask]
                    
                    # Any NaN?
                    fnan = np.isnan(values)
                    
                    # If less than 2 non-NaN values, continue
                    if np.sum(~fnan) < 2:
                        continue
                    
                    # Correct counts based on NaN values
                    if np.any(fnan):
                        self.counts[id_] -= np.sum(fnan)
                        
                    # Calculate NaN safe statistics
                    median_val = np.nanmedian(values)
                    self.mu[id_] = max(median_val, 0)  # zero if negative
                    self.sigma[id_] = np.nanstd(values)

                    # Preserve viewing geometry for the same source pixels
                    # that contributed to the image statistics. The median
                    # cosine is retained because the correction is applied
                    # pixel by pixel before taking the image median.
                    valid_image = ~fnan
                    sza_values = sza.flatten()[mask][valid_image]
                    dza_values = dza.flatten()[mask][valid_image]
                    finite_sza = np.isfinite(sza_values)
                    finite_dza = np.isfinite(dza_values)
                    if np.any(finite_sza):
                        self.sza[id_] = np.median(sza_values[finite_sza])
                    if np.any(finite_dza):
                        self.dza[id_] = np.median(dza_values[finite_dza])
                        self.los_factor[id_] = np.median(
                            np.cos(np.radians(dza_values[finite_dza]))
                        )
                        
                    # Grab weights
                    values = w.flatten()[mask]
                    self.w[id_] = np.nanmedian(values)
                            
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
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    df = self.counts[i, j, k] - 1
                    if df < 1:
                        continue
                    t_multiplier = t.ppf(1 - alpha_mean / 2, df)
                    mean_unc = t_multiplier * self.sigma[i, j, k] / np.sqrt(self.counts[i, j, k])
                    chi2_lower = chi2.ppf(alpha_std / 2, df)
                    std_inflation = self.sigma[i, j, k] * np.sqrt(df / chi2_lower)
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

        for i in range(time_len):
            # Interpolate mu
            mask = ~np.isnan(self.mu_[i])
            self.mu[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.mu_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate sigma
            mask = ~np.isnan(self.sigma_[i])
            self.sigma[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.sigma_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate w
            mask = ~np.isnan(self.w_[i])
            self.w[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.w_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate SZA
            mask = ~np.isnan(self.sza_[i])
            self.sza[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.sza_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate DZA
            mask = ~np.isnan(self.dza_[i])
            self.dza[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]), self.dza_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear', fill_value=np.nan
            )
            # Interpolate the diagnostic LOS factor
            mask = ~np.isnan(self.los_factor_[i])
            self.los_factor[i] = griddata(
                (self.grid.xi[mask], self.grid.eta[mask]),
                self.los_factor_[i][mask],
                (target_grid.xi, target_grid.eta), method='linear',
                fill_value=np.nan
            )

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
