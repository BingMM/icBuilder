#%% Import

import numpy as np
from typing import Union, Optional
from numpy.typing import NDArray
from copy import deepcopy as dcopy
from datetime import datetime
from netCDF4 import Dataset

# External dependencies
from .imagesat_e0_eflux_estimates import E0_eflux_propagated as EF_fun
from .imagesat_e0_eflux_estimates import e0_fe_covariance
from .robinson import ped, hall, peduncertainty, halluncertainty
from .binnedimage import BinnedImage
from .kp import _utc_datetime64
from .zhang_paxton_lookup import load_zhang_paxton_lookup

#%% Conductance Image class

class ConductanceImage:
    """
    Constructs ionospheric conductance from IMAGE satellite binned counts.

    Attributes
    ----------
    time : datetime, optional
        Time associated with the dataset.
    Ep : float
        Mean electron energy [keV].
    dEp : float
        Uncertainty in the mean energy [keV].
    grid : CSgrid
        Cubbed sphere grid from the BinnedImage input.
    shape : tuple
        Shape of the input binned images.

    The following arrays are populated:
        E0, dE0 : Characteristic energy and its uncertainty.
        Fe, dFe : Energy flux and its uncertainty.
        R, dR   : Ratio and its uncertainty.
        P, H    : Pedersen and Hall conductance [mho].
        dP, dH  : Uncertainty based on error propagation [mho].
        dP2, dH2: Alternative uncertainty.
        ssalon : Apex magnetic longitude of subsolar point
        wic_sza, s12_sza, s13_sza : Per-sensor median solar zenith angle.
        wic_dza, s12_dza, s13_dza : Per-sensor median detector zenith angle.
        wic_los_factor, s12_los_factor, s13_los_factor :
            Per-sensor median ``cos(DZA)`` diagnostic.
    """

    def __init__(
        self,
        wic: BinnedImage,
        s12: BinnedImage,
        s13: BinnedImage,
        time: Optional[Union[NDArray[datetime], list[datetime]]] = None,
        Ep: Union[int, float] = 2,
        dEp: Union[int, float] = 0,
        kp=None,
        kp_interval_start=None,
        kp_provenance=None,
        energy_method="zhang_paxton",
        out_fn: Optional[str] = None
    ):
        """
        Initialize a ConductanceImage from three binned images.

        Parameters
        ----------
        wic : BinnedImage
            WIC binned image (counts).
        s12 : BinnedImage
            SI12 binned image (counts).
        s13 : BinnedImage
            SI13 binned image (counts).
        time : array or list of datetime, optional
            Timestamp of the images.
        Ep : float, optional
            Mean electron energy [keV], default 2 keV.
        dEp : float, optional
            Uncertainty in Ep, default 0 keV.
        kp : array
            Original definitive GFZ Kp value for each frame.
        kp_interval_start : array
            UTC start time of each enclosing three-hour Kp interval.
        kp_provenance : dict
            Source information returned by ``load_gfz_kp``.
        energy_method : {"zhang_paxton", "image_ratio"}
            Zhang--Paxton is the production path. ``image_ratio`` keeps the
            rejected WIC/SI13 inversion available only for comparisons.
        out_fn : str, optional
            If given, the resulting conductance image will be saved to this NetCDF file.
        """
        if not (wic.shape == s12.shape == s13.shape):
            raise ValueError('wic, s12, and s13 have to have the same shape.')

        self.time = time
        self.ssalon = dcopy(wic.ssalon)
        self.Ep = Ep
        self.dEp = dEp
        self.grid = dcopy(wic.grid)
        self.shape = wic.shape
        self.energy_method = energy_method
        
        self._store_binned_counts(wic, s12, s13)
        self._store_binned_weights(wic, s12, s13)
        self._store_binned_geometry(wic, s12, s13)
        self._store_electron_energy(kp, kp_interval_start, kp_provenance)
        self._initialize_arrays()
        self._compute_conductance()

        if out_fn:
            self.to_nc(out_fn)

    def _store_binned_counts(self, wic: BinnedImage, s12: BinnedImage, s13: BinnedImage):
        """
        Stores the average and standard deviation from each BinnedImage.
        """
        self.wic_avg = wic.mu
        self.wic_std = wic.sigma
        self.s12_avg = s12.mu
        self.s12_std = s12.sigma
        self.s13_avg = s13.mu
        self.s13_std = s13.sigma
    
    def _store_binned_weights(self, wic: BinnedImage, s12: BinnedImage, s13: BinnedImage):
        """
        Stores the weights from each BinnedImage.
        """
        self.wic_w = wic.w
        self.s12_w = s12.w
        self.s13_w = s13.w

    def _store_binned_geometry(
        self,
        wic: BinnedImage,
        s12: BinnedImage,
        s13: BinnedImage
    ):
        """
        Preserve per-sensor viewing geometry and LOS-processing provenance.

        Geometry remains sensor-specific because the three instruments can
        contribute different source pixels to the same target grid cell.
        """
        for sensor, binned in (("wic", wic), ("s12", s12), ("s13", s13)):
            for field in ("sza", "dza", "los_factor"):
                values = getattr(binned, field)
                if values.shape != self.shape:
                    raise ValueError(
                        f"{sensor}.{field} must have shape {self.shape}, "
                        f"got {values.shape}."
                    )
                setattr(self, f"{sensor}_{field}", np.copy(values))

            setattr(
                self,
                f"{sensor}_los_correction",
                bool(binned.los_correction)
            )
            image_correction = (
                "raw" if binned.correction is None else str(binned.correction)
            )
            setattr(
                self,
                f"{sensor}_image_correction",
                image_correction
            )

    def _store_electron_energy(self, kp, kp_interval_start, kp_provenance):
        """Load one Zhang--Paxton layer for each frame in this orbit."""

        if self.energy_method == "image_ratio":
            self.kp = None
            self.kp_lookup = None
            self.kp_interval_start = None
            self.kp_provenance = None
            self.lookup_E0 = None
            self.lookup_dE0 = None
            self.lookup_provenance = None
            return
        if self.energy_method != "zhang_paxton":
            raise ValueError("energy_method must be 'zhang_paxton' or 'image_ratio'")
        if kp is None or kp_interval_start is None:
            raise ValueError(
                "the Zhang--Paxton production path requires Kp and "
                "Kp interval start for every frame"
            )
        if self.time is None or len(self.time) != self.shape[0]:
            raise ValueError(
                "the Zhang--Paxton production path requires one time per frame"
            )
        required_provenance = {
            "source",
            "status",
            "doi",
            "licence",
            "query",
            "acquired",
            "sha256",
        }
        if kp_provenance is None or not required_provenance.issubset(
            kp_provenance
        ):
            raise ValueError("complete GFZ Kp provenance is required")

        self.kp = np.asarray(kp, dtype=float)
        self.kp_interval_start = np.asarray(
            kp_interval_start,
            dtype="datetime64[s]",
        )
        if self.kp.shape != (self.shape[0],):
            raise ValueError("Kp length must match the number of image frames")
        if self.kp_interval_start.shape != (self.shape[0],):
            raise ValueError(
                "Kp interval-start length must match the number of image frames"
            )

        frame_time = _utc_datetime64(self.time)
        inside_interval = (
            (self.kp_interval_start <= frame_time)
            & (
                frame_time
                < self.kp_interval_start + np.timedelta64(3, "h")
            )
        )
        if not np.all(inside_interval):
            raise ValueError(
                "every frame time must be inside its stated half-open "
                "three-hour Kp interval"
            )

        lookup = load_zhang_paxton_lookup(self.kp)
        if lookup["E0"].shape != self.shape or lookup["dE0"].shape != self.shape:
            raise ValueError("Zhang--Paxton lookup shape does not match image shape")
        if (
            np.asarray(self.grid.xi).shape != lookup["xi"].shape
            or np.asarray(self.grid.eta).shape != lookup["eta"].shape
            or not np.allclose(self.grid.xi, lookup["xi"])
            or not np.allclose(self.grid.eta, lookup["eta"])
        ):
            raise ValueError("Zhang--Paxton lookup grid does not match image grid")

        self.kp_lookup = np.asarray(lookup["kp"])
        self.lookup_E0 = np.asarray(lookup["E0"])
        self.lookup_dE0 = np.asarray(lookup["dE0"])
        self.lookup_provenance = lookup["provenance"]
        self.kp_provenance = dict(kp_provenance)

    def _initialize_arrays(self):
        """
        Initializes empty arrays for all estimated quantities.
        """
        self.E0     = np.full(self.shape, np.nan)
        self.dE0    = np.full(self.shape, np.nan)
        self.Fe     = np.full(self.shape, np.nan)
        self.dFe    = np.full(self.shape, np.nan)
        self.R      = np.full(self.shape, np.nan)
        self.dR     = np.full(self.shape, np.nan)
        self.P      = np.full(self.shape, np.nan)
        self.H      = np.full(self.shape, np.nan)
        self.dP     = np.full(self.shape, np.nan)
        self.dH     = np.full(self.shape, np.nan)
        self.varE0Fe = np.full(self.shape, np.nan)

    def _compute_conductance(self):
        """
        Loops through all pixels and computes conductance estimates.
        """
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    self._compute_pixel(i, j, k)

    def _compute_pixel(self, i: int, j: int, k: int):
        """
        Compute E0, Fe, R, and conductances at a single pixel.

        Parameters
        ----------
        i, j, k : int
            Indices into the 3D array (time, lat, lon).
        """
        W, T, S = self.wic_avg[i, j, k], self.s12_avg[i, j, k], self.s13_avg[i, j, k]
        dW, dT, dS = self.wic_std[i, j, k], self.s12_std[i, j, k], self.s13_std[i, j, k]

        counts = [W, T, S]
        dayglow = [0, 0, 0]  # Assume zero dayglow
        uncertainties = [dW, dT, dS]
        
        if np.all(~np.isnan(counts)) and np.all(~np.isnan(uncertainties)):
            if self.energy_method == "zhang_paxton":
                E0_input = self.lookup_E0[i, j, k]
                dE0_input = self.lookup_dE0[i, j, k]
            else:
                E0_input = None
                dE0_input = None

            E0, Fe, dE0, dFe, R, dR = EF_fun(
                counts,
                dayglow,
                uncertainties,
                self.Ep,
                self.dEp,
                E0=E0_input,
                dE0=dE0_input,
            )

            self.E0[i, j, k], self.dE0[i, j, k] = E0, dE0
            self.Fe[i, j, k], self.dFe[i, j, k] = Fe, dFe
            self.R[i, j, k],  self.dR[i, j, k]  = R,  dR 

            if self.energy_method == "zhang_paxton":
                varE0Fe = e0_fe_covariance(E0, Fe, dE0)
            else:
                varE0Fe = 0
            self.varE0Fe[i, j, k] = varE0Fe
            P, H = ped(E0, Fe), hall(E0, Fe)
            
            dP = peduncertainty( E0, Fe, dE0, dFe, varE0Fe)
            dH = halluncertainty(E0, Fe, dE0, dFe, varE0Fe)
            
            self.P[i, j, k], self.H[i, j, k] = P, H
            self.dP[i, j, k], self.dH[i, j, k] = dP, dH

        self.w = (self.wic_w + self.s12_w + self.s13_w)/3

    def to_nc(self, filename: str):
        """
        Save conductance image to a NetCDF4 file.
        Can be read/rebuilt using the icReader library.
        
        Parameters
        ----------
        filename : str
            Full path to output NetCDF file.
        """
    
        # Create a NetCDF4 file (not netcdf3)
        with Dataset(filename, "w", format="NETCDF4") as nc:
            t, y, x = self.shape
    
            # Root-level dimensions
            nc.createDimension("time", t)
            nc.createDimension("dim1", y)
            nc.createDimension("dim2", x)
    
            # Helper
            def save_var(name, data, units=None, long_name=None):
                v = nc.createVariable(name, "f4", ("time", "dim1", "dim2"), zlib=True)
                v[:] = data
                if units is not None:
                    v.units = units
                if long_name is not None:
                    v.long_name = long_name
                return v
    
            # Save main variables
            for attr in [
                "wic_avg", "s12_avg", "s13_avg", "wic_std", "s12_std", "s13_std",
                "E0", "dE0", "Fe", "dFe", "varE0Fe", "R", "dR",
                "P", "H", "dP", "dH", "w"
            ]:
                save_var(attr, getattr(self, attr))
            nc.variables["varE0Fe"].units = "keV mW m-2"
            nc.variables["varE0Fe"].long_name = (
                "First-order covariance of electron energy and energy flux"
            )

            # Preserve viewing geometry separately for each IMAGE-FUV sensor.
            sensor_names = {
                "wic": "WIC",
                "s12": "SI12",
                "s13": "SI13",
            }
            for sensor, display_name in sensor_names.items():
                save_var(
                    f"{sensor}_sza",
                    getattr(self, f"{sensor}_sza"),
                    units="degrees",
                    long_name=(
                        f"Median {display_name} solar zenith angle of "
                        "contributing pixels"
                    ),
                )
                save_var(
                    f"{sensor}_dza",
                    getattr(self, f"{sensor}_dza"),
                    units="degrees",
                    long_name=(
                        f"Median {display_name} detector zenith angle of "
                        "contributing pixels"
                    ),
                )
                save_var(
                    f"{sensor}_los_factor",
                    getattr(self, f"{sensor}_los_factor"),
                    units="1",
                    long_name=(
                        f"Median {display_name} cos(DZA) of contributing "
                        "pixels"
                    ),
                )
                nc.setncattr(
                    f"{sensor}_los_correction",
                    np.int8(getattr(self, f"{sensor}_los_correction")),
                )
                nc.setncattr(
                    f"{sensor}_image_correction",
                    getattr(self, f"{sensor}_image_correction"),
                )

            nc.los_factor_definition = (
                "Median cos(DZA) of the source pixels contributing to each "
                "binned image value; diagnostic only and not generally an "
                "exact multiplier between corrected and uncorrected binned "
                "medians."
            )
    
            # Scalars
            nc.Ep  = float(self.Ep)
            nc.dEp = float(self.dEp)
            nc.zero_flux_uncertainty_definition = (
                "At Fe=0, dP and dH are one-sided conductance excursions "
                "from Fe=0 to Fe=dFe because the sqrt(Fe) derivative is "
                "singular at zero."
            )
            nc.electron_energy_method = self.energy_method
            if self.energy_method == "zhang_paxton":
                nc.dE0_interpretation = (
                    "Spherical-area-weighted spread of E0 within the selected "
                    "Zhang-Paxton MLAT profile. It is not formal model "
                    "prediction or coefficient uncertainty."
                )
                nc.e0_fe_covariance_definition = (
                    "-Wprime * Wm_prime(E0) / Wm(E0)^2 * dE0^2"
                )
                for name, value in self.lookup_provenance.items():
                    nc.setncattr(f"electron_energy_lookup_{name}", value)
                for name, value in self.kp_provenance.items():
                    nc.setncattr(f"kp_{name}", value)
    
            # Time variable
            if self.time is not None:
                ref_time = datetime(2000, 1, 1)
                time_seconds = np.array(
                    [(t - ref_time).total_seconds() for t in self.time],
                    dtype=np.int32
                )
                vtime = nc.createVariable("time", "i4", ("time",))
                vtime[:] = time_seconds
                nc.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")

            if self.energy_method == "zhang_paxton":
                vkp = nc.createVariable("Kp", "f4", ("time",))
                vkp[:] = self.kp
                vkp.units = "1"
                vkp.long_name = "Original definitive GFZ three-hour Kp"

                vkp_lookup = nc.createVariable("Kp_lookup", "f4", ("time",))
                vkp_lookup[:] = self.kp_lookup
                vkp_lookup.units = "1"
                vkp_lookup.long_name = (
                    "Kp rounded to the nearest hundredth for lookup indexing"
                )

                interval_seconds = (
                    self.kp_interval_start
                    - np.datetime64(ref_time, "s")
                ).astype("timedelta64[s]").astype(np.int64)
                vkp_start = nc.createVariable(
                    "Kp_interval_start",
                    "i8",
                    ("time",),
                )
                vkp_start[:] = interval_seconds
                vkp_start.units = (
                    "seconds since 2000-01-01 00:00:00 UTC"
                )
                vkp_start.long_name = (
                    "UTC start of enclosing half-open three-hour Kp interval"
                )
    
            ssalon = nc.createVariable("ssalon", "f4", ("time",))
            ssalon[:] = self.ssalon
    
            # Grid GROUP
            grid_grp = nc.createGroup("grid")

            # Grid metadata
            grid_grp.position    = self.grid.projection.position.astype(float)
            grid_grp.orientation = self.grid.projection.orientation
            grid_grp.L    = self.grid.L
            grid_grp.W    = self.grid.W
            grid_grp.Lres = self.grid.Lres
            grid_grp.Wres = self.grid.Wres
            grid_grp.R    = self.grid.R
