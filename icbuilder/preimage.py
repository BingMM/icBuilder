#%% Imports

import numpy as np
from netCDF4 import Dataset
from typing import Union, Optional
from numpy.typing import NDArray
from secsy import CSgrid

#%% Class for storing orbit files

class PreImage:
    """
    Container for orbit IMAGE data loaded from a NetCDF file.

    This class extracts and holds data from UV imagers, including geomagnetic and 
    geographic coordinates, measured and modeled images, and projected data for each 
    frame in the file.

    Attributes
    ----------
    mlat : np.ndarray
        Magnetic latitude coordinates [time, y, x].
    mlon : np.ndarray
        Magnetic longitude coordinates [time, y, x].
    mlt : np.ndarray
        Magnetic local times [time, y, x].
    glat : np.ndarray
        Geographic latitude coordinates [time, y, x].
    glon : np.ndarray
        Geographic longitude coordinates [time, y, x].
    dgimg : np.ndarray
        Projected image in detector geometry [time, y, x].
    shimg : np.ndarray
        Projected image in SH (spherical harmonics) geometry [time, y, x].
    dgmodel : np.ndarray
        Model image in detector geometry [time, y, x].
    shweight : np.ndarray
        IRLS weight for SH fit [time, y, x].
    dgweight : np.ndarray
        IRLS weight for dayglow fit [time, y, x].
    shape : tuple
        Shape of the image arrays.
    index : Optional[list[int] or np.ndarray]
        Indices of time frames loaded.
    ssalon : np.ndarray
        Apex longitude of the subsolar point at each time.
    """

    def __init__(self,
                 sensor: str,
                 ncdf: Dataset,
                 index: Optional[Union[list[int], NDArray[np.int_]]] = None):
        """
        Load orbit file data (e.g., WIC/SI13/SI12) from a NetCDF file.

        Parameters
        ----------
        sensor : str
            IMAGE-FUV sensor name: ``"WIC"``, ``"SI12"``, or ``"SI13"``.
        ncdf : NetCDFFile
            NetCDF file containing image and coordinate data.
        index : Optional[list[int] or np.ndarray]
            Frame indices to load. If None, all frames are loaded.
        """

        sensor = sensor.upper()
        if sensor not in ("WIC", "SI12", "SI13"):
            raise ValueError("sensor must be 'WIC', 'SI12', or 'SI13'")

        self.sensor = sensor
        self.index = index

        var_names = ['mlat', 'mlon', 'mlt', 'glat', 'glon', 'img', 'dgimg', 'shimg', 'dgmodel', 'shweight', 'dgweight', 'sza', 'dza']
        for name in var_names:
            var_data = ncdf.variables[name]
            data = var_data[...] if index is None else var_data[index, :, :]
            setattr(self, name, np.copy(data))

        self.shape = self.mlat.shape
        self.ssalon = np.full(self.shape[0], np.nan)

    def get_SZA(self, i: int) -> NDArray[np.float64]:
        """
        Return the SZA for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            SZA.
        """
        return self.sza[i, :, :]
    
    def get_DZA(self, i: int) -> NDArray[np.float64]:
        """
        Return the DZA for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            DZA.
        """
        return self.dza[i, :, :]

    def get_img_los(self, i: int) -> NDArray[np.float64]:
        """
        Return the image with DZA correction for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            Image with DZA correction.
        """
        return self.img[i, :, :]*np.cos(np.radians(self.dza[i, :, :]))

    def get_shimg_los(self, i: int) -> NDArray[np.float64]:
        """
        Return the SH corrected image with DZA correction for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            SH corrected image with DZA correction.
        """
        return self.shimg[i, :, :]*np.cos(np.radians(self.dza[i, :, :]))

    def get_dgimg_los(self, i: int) -> NDArray[np.float64]:
        """
        Return the dayglow subtracted image with DZA correction for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            dayglow subtracted image with DZA correction.
        """
        return self.dgimg[i, :, :]*np.cos(np.radians(self.dza[i, :, :]))

    def get_img(self, i: int) -> NDArray[np.float64]:
        """
        Return the image for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            Image.
        """
        return self.img[i, :, :]

    def get_shimg(self, i: int) -> NDArray[np.float64]:
        """
        Return the SH corrected image for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            SH corrected image.
        """
        return self.shimg[i, :, :]

    def get_dgimg(self, i: int) -> NDArray[np.float64]:
        """
        Return the dayglow subtracted image for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            dayglow subtracted image.
        """
        return self.dgimg[i, :, :]
    
    def get_dgw(self, i: int) -> NDArray[np.float64]:
        """
        Return IRLS weights from dayglow subtraction for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            weights.
        """
        return self.dgweight[i, :, :]

    def get_shw(self, i: int) -> NDArray[np.float64]:
        """
        Return IRLS weights from SH fit for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            weights.
        """
        return self.shweight[i, :, :]

    def get_model(self, i: int) -> NDArray[np.float64]:
        """
        Return the dayglow model for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        np.ndarray
            Dayglow model image.
        """
        return self.dgmodel[i, :, :]

    def get_mcoords(self, i: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the magnetic coordinates for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        tuple of np.ndarray
            (magnetic latitude, magnetic longitude)
        """
        return self.mlat[i, :, :], self.mlon[i, :, :], self.mlt[i, :, :], self.ssalon[i]

    def get_gcoords(self, i: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return the geographic coordinates for frame i.

        Parameters
        ----------
        i : int
            Frame index.

        Returns
        -------
        tuple of np.ndarray
            (geographic latitude, geographic longitude)
        """
        return self.glat[i, :, :], self.glon[i, :, :]

    def discard(self, 
                f: Union[list[int], NDArray[np.int_]]) -> None:
        """
        Keep frame NOT in 'f'.

        Parameters
        ----------
        f : list[int] or np.ndarray
            Indices of frames to retain.
        """
        for name in ['mlat', 'mlon', 'mlt', 'glat', 'glon', 'img', 'dgimg', 'shimg', 'dgmodel', 'shweight', 'dgweight', 'sza', 'dza']:
            setattr(self, name, getattr(self, name)[f, :, :])
        
        self.ssalon = self.ssalon[f]

        self.shape = self.mlat.shape

    def percent_full(self, grid: CSgrid) -> float:
        """
        Compute the fraction of grid cells that contain data at each time step.

        Parameters
        ----------
        grid : CSgrid
            Grid to compare coverage against.

        Returns
        -------
        float
            Fraction of non-empty grid cells for each time frame.
        """
        counts = np.zeros((self.shape[0], grid.shape[0], grid.shape[1]))
        for i in range(self.shape[0]):
            #mlat, mlon = self.get_mcoords(i)
            mlat, _, mlt, _ = self.get_mcoords(i)
            mlon = mlt*15
            f = grid.ingrid(mlon, mlat)
            counts[i] = grid.count(mlon[f], mlat[f])
        f = counts != 0
        return np.sum(f, axis=(1,2)) / grid.size
