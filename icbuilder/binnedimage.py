#%% Imports

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid
from copy import deepcopy as dcopy
from scipy.stats import t, chi2
from .preimage import PreImage
from .footprints import overlap_mapping, overlap_mean, overlap_statistics
from netCDF4 import Dataset, date2num
from datetime import datetime

#%% BinnedImage class

class BinnedImage:
    """
    A class to bin IMAGE data onto a CSgrid.

    This class processes a `PreImage` object using either footprint-overlap
    averaging or the former point-centre median calculation.

    Attributes
    ----------
    grid : CSgrid
        The native sensor CSgrid to which data is binned.
    counts : np.ndarray
        Number of source pixels contributing to each grid cell.
    mu : np.ndarray
        Footprint-weighted mean or point-centre median in each grid cell.
    sigma : np.ndarray
        Provisional within-cell spread of the contributing values.
    coverage : np.ndarray
        Fraction of each target cell covered by valid detector footprints.
        This is NaN for the legacy centre-binning method.
    sza : np.ndarray
        Overlap-weighted mean or point-centre median solar zenith angle
        [degrees].
    dza : np.ndarray
        Overlap-weighted mean or point-centre median detector zenith angle
        [degrees].
    los_factor : np.ndarray
        Overlap-weighted mean or point-centre median ``cos(DZA)``. This is a
        diagnostic summary of the pixel-level correction factors.
    los_correction : bool
        Whether the pixel-level ``cos(DZA)`` correction was applied.
    shape : tuple
        Shape of the binned data arrays.
    """
    
    def __init__(self,
                 pI: PreImage,
                 grid: CSgrid,
                 time: Union[NDArray[datetime], list[datetime]],
                 inflate_uncertainty: bool = False,
                 correction: Optional[str] = None,
                 los_correction: bool = True,
                 binning_method: str = "footprint",
                 ):
        """
        Bin statistics from a PreImage object into a CSgrid.

        Parameters
        ----------
        pI : PreImage
            Input IMAGE data to bin.
        grid : CSgrid
            Cubed-sphere grid to bin onto.
        time : np.ndarray or list of datetime
            UTC timestamp for each frame in ``pI``.
        inflate_uncertainty : bool
            If True, inflates uncertainties using t and chi² statistics.
            Should be used when less than 30 counts per bin.
        correction : {"SH", "DG"} or None
            Select the background-corrected image to bin. ``None`` bins the
            raw image.
        los_correction : bool
            If True, multiply each source pixel by ``cos(DZA)`` before
            calculating the binned image statistics.
        binning_method : {"footprint", "centre"}
            ``"footprint"`` distributes each detector pixel according to its
            inferred area overlap with the target cells. ``"centre"`` keeps
            the previous point-centre median calculation.
        """
        self.sensor = pI.sensor
        self.time = np.asarray(time, dtype=object)
        if self.time.ndim != 1 or self.time.size != pI.shape[0]:
            raise ValueError("time must contain one timestamp per image frame")
        if correction not in (None, "SH", "DG"):
            raise ValueError("correction must be None, 'SH', or 'DG'")
        if binning_method not in ("footprint", "centre"):
            raise ValueError("binning_method must be 'footprint' or 'centre'")

        self.grid = dcopy(grid)
        self.correction = correction
        self.los_correction = bool(los_correction)
        self.binning_method = binning_method

        time_len, ny, nx = pI.shape[0], grid.shape[0], grid.shape[1]
        shape = (time_len, ny, nx)
        self.counts = np.zeros(shape, dtype=np.int32)
        self.mu = np.full(shape, np.nan)
        self.sigma = np.full(shape, np.nan)
        self.w = np.full(shape, np.nan)
        self.sza = np.full(shape, np.nan)
        self.dza = np.full(shape, np.nan)
        self.los_factor = np.full(shape, np.nan)
        self.coverage = np.full(shape, np.nan)
        self.shape = self.counts.shape
        self.ssalon = pI.ssalon

        for i in range(time_len):
            lat, _, mlt, _ = pI.get_mcoords(i)
            img, w, sza, dza = self._frame_fields(pI, i)

            if self.binning_method == "footprint":
                self._bin_footprints(i, lat, mlt, img, w, sza, dza)
            else:
                self._bin_centres(i, lat, mlt, img, w, sza, dza)
                            
        if inflate_uncertainty:
            # This is the existing small-sample treatment. Its use with the
            # provisional footprint spread is retained for now so footprint
            # geometry and measurement uncertainty can be treated separately.
            self._inflate_uncertainty()

    def _frame_fields(self, preimage, index):
        """Return the image and diagnostic source fields for one frame."""

        if self.correction == "SH":
            if self.los_correction:
                image = preimage.get_shimg_los(index)
            else:
                image = preimage.get_shimg(index)
        elif self.correction == "DG":
            if self.los_correction:
                image = preimage.get_dgimg_los(index)
            else:
                image = preimage.get_dgimg(index)
        else:
            if self.los_correction:
                image = preimage.get_img_los(index)
            else:
                image = preimage.get_img(index)

        weight = preimage.get_dgw(index) * preimage.get_shw(index)
        return image, weight, preimage.get_SZA(index), preimage.get_DZA(index)

    def _bin_footprints(self, frame, lat, mlt, image, weight, sza, dza):
        """Distribute uniform detector footprints over intersected CS cells."""

        mapping, cell_area = overlap_mapping(lat, mlt, self.grid)
        output_shape = self.grid.shape

        mean, spread, count, coverage = overlap_statistics(
            image, mapping, cell_area, output_shape
        )
        self.mu[frame] = np.maximum(mean, 0)
        self.sigma[frame] = spread
        self.counts[frame] = count
        self.coverage[frame] = coverage

        # Geometry describes the same valid detector pixels as the image.
        valid_image = np.isfinite(image)
        masked_sza = np.where(valid_image, sza, np.nan)
        masked_dza = np.where(valid_image, dza, np.nan)
        masked_los = np.where(valid_image, np.cos(np.radians(dza)), np.nan)

        self.sza[frame], _ = overlap_mean(masked_sza, mapping, output_shape)
        self.dza[frame], _ = overlap_mean(masked_dza, mapping, output_shape)
        self.los_factor[frame], _ = overlap_mean(
            masked_los, mapping, output_shape
        )

        # Retain the historical independent NaN handling for quality weights.
        self.w[frame], _ = overlap_mean(weight, mapping, output_shape)

    def _bin_centres(self, frame, lat, mlt, image, weight, sza, dza):
        """Keep the former point-centre median calculation available."""

        ny, nx = self.grid.shape
        longitude = mlt * 15
        j, k = self.grid.bin_index(longitude, lat)
        valid_bin = (j >= 0) & (j < ny) & (k >= 0) & (k < nx)
        source_index = np.flatnonzero(valid_bin)
        flat_bin = j[valid_bin] * nx + k[valid_bin]
        order = np.argsort(flat_bin, kind="stable")
        source_index = source_index[order]
        flat_bin = flat_bin[order]

        image = image.flatten()
        weight = weight.flatten()
        sza = sza.flatten()
        dza = dza.flatten()

        if flat_bin.size:
            starts = np.r_[0, np.flatnonzero(np.diff(flat_bin)) + 1]
            stops = np.r_[starts[1:], flat_bin.size]
        else:
            starts, stops = [], []

        for start, stop in zip(starts, stops):
            jj, kk = divmod(flat_bin[start], nx)
            output_index = (frame, jj, kk)
            indices = source_index[start:stop]

            values = image[indices]
            finite = np.isfinite(values)
            self.counts[output_index] = np.sum(finite)
            if not np.any(finite):
                continue

            finite_values = values[finite]
            self.mu[output_index] = max(np.median(finite_values), 0)
            self.sigma[output_index] = np.std(finite_values)

            sza_values = sza[indices][finite]
            dza_values = dza[indices][finite]
            if np.any(np.isfinite(sza_values)):
                self.sza[output_index] = np.nanmedian(sza_values)
            if np.any(np.isfinite(dza_values)):
                self.dza[output_index] = np.nanmedian(dza_values)
                self.los_factor[output_index] = np.nanmedian(
                    np.cos(np.radians(dza_values))
                )

            self.w[output_index] = np.nanmedian(weight[indices])

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
        # Most cells share one of only a few integer sample counts, so evaluate
        # the expensive probability distributions once per count and reuse
        # the scalar multipliers.
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
        self.coverage = self.coverage[f]
        self.time = self.time[f]
        self.ssalon = self.ssalon[f]
        self.shape = self.mu.shape

    def to_nc(self, filename: str):
        """Save one sensor's native-grid binned images to NetCDF4.

        Parameters
        ----------
        filename : str
            Full path to output NetCDF file.
        """

        image_fields = (
            "counts", "mu", "sigma", "w", "sza", "dza", "los_factor",
            "coverage",
        )
        for name in image_fields:
            if getattr(self, name).shape != self.shape:
                raise ValueError(f"{name} and binned image dimensions do not match")
        if self.time.size != self.shape[0]:
            raise ValueError("time and binned image lengths do not match")
        if np.asarray(self.ssalon).shape != (self.shape[0],):
            raise ValueError("ssalon must contain one value per image frame")

        grid_coordinates = {
            "xi": (np.asarray(self.grid.xi), "radians"),
            "eta": (np.asarray(self.grid.eta), "radians"),
            "mlat": (np.asarray(self.grid.lat), "degrees"),
            "mlt": (np.mod(np.asarray(self.grid.lon) / 15.0, 24.0), "hours"),
        }
        for name, (values, _) in grid_coordinates.items():
            if values.shape != self.shape[1:]:
                raise ValueError(f"grid {name} dimensions do not match binned images")

        with Dataset(filename, "w", format="NETCDF4") as nc:
            t, y, x = self.shape
            nc.createDimension("time", t)
            nc.createDimension("dim1", y)
            nc.createDimension("dim2", x)

            nc.product_type = "binned_fuv"
            nc.schema_version = 1
            nc.sensor = self.sensor
            nc.image_correction = self.correction or "raw"
            nc.los_correction = np.int8(self.los_correction)
            nc.binning_method = self.binning_method

            variables = {
                "counts": (self.counts.astype(np.int32), "i4", "1"),
                "mu": (self.mu, "f4", "counts"),
                "sigma": (self.sigma, "f4", "counts"),
                "w": (self.w, "f4", "1"),
                "sza": (self.sza, "f4", "degrees"),
                "dza": (self.dza, "f4", "degrees"),
                "los_factor": (self.los_factor, "f4", "1"),
                "coverage": (self.coverage, "f4", "1"),
            }
            for name, (data, dtype, units) in variables.items():
                variable = nc.createVariable(
                    name, dtype, ("time", "dim1", "dim2"), zlib=True
                )
                variable[:] = data
                variable.units = units

            if self.binning_method == "footprint":
                nc.counts_definition = (
                    "Number of valid detector footprints intersecting each cell."
                )
                nc.sigma_definition = (
                    "Provisional overlap-weighted within-cell spread; retained "
                    "for compatibility and not a complete measurement uncertainty."
                )
                nc.los_factor_definition = (
                    "Overlap-weighted mean cos(DZA) of valid detector footprints."
                )
                nc.coverage_definition = (
                    "Fraction of projected CS-cell area covered by valid detector "
                    "footprints under a uniform top-hat response assumption."
                )
            else:
                nc.counts_definition = (
                    "Number of valid detector-pixel centres inside each cell."
                )
                nc.sigma_definition = (
                    "Point-centre within-cell spread; provisional and not a "
                    "complete measurement uncertainty."
                )
                nc.los_factor_definition = (
                    "Median cos(DZA) of the source pixels contributing to each "
                    "binned image value; diagnostic only."
                )
                nc.coverage_definition = (
                    "Unavailable for centre binning; stored as NaN."
                )
            time_units = "seconds since 2000-01-01 00:00:00"
            vtime = nc.createVariable("time", "f8", ("time",))
            vtime[:] = date2num(
                self.time.tolist(), units=time_units, calendar="standard"
            )
            vtime.units = time_units
            vtime.calendar = "standard"
            vtime.time_zone = "UTC"

            ssalon = nc.createVariable("ssalon", "f4", ("time",))
            ssalon[:] = self.ssalon
            ssalon.units = "degrees"

            grid_grp = nc.createGroup("grid")
            grid_grp.position    = self.grid.projection.position.astype(float)
            grid_grp.orientation = self.grid.projection.orientation
            grid_grp.L    = self.grid.L
            grid_grp.W    = self.grid.W
            grid_grp.Lres = self.grid.Lres
            grid_grp.Wres = self.grid.Wres
            grid_grp.R    = self.grid.R

            for name, (data, units) in grid_coordinates.items():
                coordinate = grid_grp.createVariable(
                    name, "f8", ("dim1", "dim2"), zlib=True
                )
                coordinate[:] = data
                coordinate.units = units
