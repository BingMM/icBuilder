#%% Import

from typing import Optional
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import kron, vstack, csc_matrix
import scipy.sparse as sp
from icreader import ConductanceImage
from sksparse.cholmod import cholesky
from scipy.io import netcdf_file
from datetime import datetime

#%%
class SplineImage():
    def __init__(self, 
                 cI: ConductanceImage,
                 ncp: Optional[int] = 20,
                 ncpt: Optional[int] = None,
                 k: Optional[int] = 3,
                 kt: Optional[int] = 2,
                 lH: Optional[int] = 0,
                 lP: Optional[int] = 0):
        
        self.cI = cI                 
        
        self.ncp = ncp
        self.k = k
        self._ncpt = ncpt
        self.kt = kt
        self.lH = lH
        self.lP = lP

        self.reset_time()
        self.reset_spline()
        self.reset_model()
        self.reset_ev()

    @property
    def ncpt(self):
        if self._ncpt is None:
            self._ncpt = int((self.time[-1] - self.time[0]).total_seconds() / 60 // 5)
        return self._ncpt

#%% Time
        
    def reset_time(self):
        self._t = None
    
    @property
    def time(self):
        return self.cI.time
    
    @property
    def t(self):
        if self._t is None:
            self._t = np.array([(t - self.time[0]).total_seconds() for t in self.time])
        return self._t
    
    @property
    def nt(self):
        return self.time.size
    
#%% Grid

    @property
    def grid(self):
        return self.cI.grid
    
    @property
    def x(self):
        return self.grid.xi
    
    @property
    def y(self):
        return self.grid.eta
    
    @property
    def shape(self):
        return self.grid.shape
    
    @property
    def n(self):
        return self.shape[0]
    
#%% Conductance

    @property
    def H(self):
        return self.cI.H
   
    @property
    def P(self):
        return self.cI.P
    
    @property
    def dH(self):
        return self.cI.dH
   
    @property
    def dP(self):
        return self.cI.dP
    
#%% Spline

    def reset_spline(self):
        self._G = None
        self._Gt = None

    @property
    def nk(self):
        return self.ncp + self.k + 1
    
    @property
    def nkt(self):
        return self.ncpt + self.kt + 1

    @property
    def knots(self):
        return np.r_[[self.x.min()]*self.k, np.linspace(self.x.min(), self.x.max(), self.nk-2*self.k), [self.x.max()]*self.k]

    @property
    def tknots(self):
        return np.r_[[self.t.min()]*self.kt, np.linspace(self.t.min(), self.t.max(), self.nkt-2*self.kt), [self.t.max()]*self.kt]

    @property
    def G(self):
        if self._G is None:
            self._G = self.generate_G_2d(return_sparse=True)
        return self._G

    @property
    def Gt(self):
        if self._Gt is None:
            self._Gt = self.generate_G_3d()
        return self._Gt
        
    def generate_G_2d(self, return_sparse=False):
        G = np.zeros((self.x.size, self.ncp**2))
        for i, (xi, yi) in enumerate(zip(self.x.flatten(), self.y.flatten())):
            Gx = BSpline.design_matrix(xi, self.knots, self.k).todense()
            Gy = BSpline.design_matrix(yi, self.knots, self.k).todense()
            Gy = np.kron(np.eye(self.ncp), Gy)
            G[i, :] = Gx.dot(Gy)
        if return_sparse:
            return csc_matrix(G)
        return G
        
    def generate_G_3d(self):
        Gt = []
        for i, ti in enumerate(self.t):
            Gt_ = BSpline.design_matrix(ti, self.tknots, self.kt)  # Already sparse
            Gt_ = kron(np.eye(self.G.shape[1]), Gt_, format='csr')  # Ensure sparse output
            Gt.append(self.G @ Gt_)  # Matrix multiplication remains sparse
        return vstack(Gt, format='csr')  # Efficient sparse stacking

#%% Models

    def reset_model(self):
        self._mH = None
        self._mP = None
        self._CpH = None
        self._CpP = None
        self._factorH = None
        self._factorP = None
        
    @property
    def factorH(self):
        if self._factorH is None:
            self._mH, self._CpH, self._factorH = self.make_model(comp='hall')
        return self._factorH
        
    @property
    def mH(self):
        if self._mH is None:
            self._mH, self._CpH, self._factorH = self.make_model(comp='hall')
        return self._mH
    
    @property
    def CpH(self):
        if self._CpH is None:
            self._mH, self._CpH, self._factorH = self.make_model(comp='hall')
        return self._CpH

    @property
    def factorP(self):
        if self._factorP is None:
            self._mP, self._CpP, self._factorP = self.make_model(comp='pedersen')
        return self._factorP    
    
    @property
    def mP(self):
        if self._mP is None:
            self._mP, self._CpP, self._factorP = self.make_model(comp='pedersen')
        return self._mP
    
    @property
    def CpP(self):
        if self._CpP is None:
            self._mP, self._CpP, self._factorP = self.make_model(comp='pedersen')
        return self._CpP    
    
    def make_model(self, comp='hall'):
        if comp == 'hall':
            d, q, l = self.H.flatten(), self.dH.flatten(), self.lH
        else:
            d, q, l = self.P.flatten(), self.dP.flatten(), self.lP
        
        f = ~(np.isnan(d) | np.isinf(d) | np.isnan(q) | np.isinf(q))
    
        return _make_model(self.Gt[f], d[f], q[f], l)
    
#%% Evaluation

    def reset_ev(self):
        self._pH = None
        self._dpH = None
        self._pP = None
        self._dpP = None
   
    @property
    def pH(self):
        if self._pH is None:
            self._pH = self.ev(comp='hall')
        return self._pH
    
    @property
    def pP(self):
        if self._pP is None:
            self._pP = self.ev(comp='pedersen')
        return self._pP
    
    @property
    def dpH(self):
        if self._dpH is None:
            self._dpH = self.ev_uncertainty(comp='hall')
        return self._dpH
            
    @property
    def dpP(self):
        if self._dpP is None:
            self._dpP = self.ev_uncertainty(comp='pedersen')
        return self._dpP

    def ev(self, comp='hall'):
        if comp == 'hall':
            m = self.mH
        else:
            m = self.mP
        return self.Gt.dot(m).reshape((self.nt, self.n, self.n))
    
    def ev_uncertainty(self, comp='hall'):
        if comp == 'hall':
            Cp = self.CpH
        else:
            Cp = self.CpP
        
        Gp = self.Gt @ Cp
        diag_elements = np.array(self.Gt.multiply(Gp).sum(axis=1)).ravel()
        return np.sqrt(diag_elements).reshape((self.nt, self.n, self.n))

#%% Save image
    
    def to_nc(self, filename: str):
            """
            Save spline image to a NetCDF file.
            Can be read/rebuilt using the icReader library.
    
            Parameters
            ----------
            filename : str
                Full path to output NetCDF file.
            """
            with netcdf_file(filename, 'w') as nc:
                t, y, x = self.nt, self.n, self.n
                nc.createDimension('time', t)
                nc.createDimension('dim1', y)
                nc.createDimension('dim2', x)
    
                nc.createVariable('H',  'f8', ('time', 'dim1', 'dim2'))[:] = self.pH
                nc.createVariable('P',  'f8', ('time', 'dim1', 'dim2'))[:] = self.pP
                nc.createVariable('dH', 'f8', ('time', 'dim1', 'dim2'))[:] = self.dpH
                nc.createVariable('dP', 'f8', ('time', 'dim1', 'dim2'))[:] = self.dpH
    
                if self.time is not None:
                    ref_time = datetime(2000, 1, 1)
                    time_seconds = np.array([(t - ref_time).total_seconds() for t in self.time], dtype=np.int32)
                    nc.createVariable("time", np.int32, ("time",))[:] = time_seconds
                    nc.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")
    
                if self.grid and hasattr(self.grid, "projection"):
                    nc.position     = self.grid.projection.position.astype(float)
                    nc.orientation  = self.grid.projection.orientation
                    nc.L    = self.grid.L
                    nc.W    = self.grid.W
                    nc.Lres = self.grid.Lres
                    nc.Wres = self.grid.Wres
                    nc.gridR    = self.grid.R

                nc.createDimension('m', self.mH.size)
                
                nc.createVariable('mH', 'f8', ('m',))[:] = self.mH
                nc.createVariable('mP', 'f8', ('m',))[:] = self.mP
                                
                L = self.factorH.L()
                nc.createDimension('LH_nnz', L.nnz)
                nc.createDimension('ncols_plus_1', self.mH.size+1)
                nc.createVariable("LH_data", "f4", ("LH_nnz",))[:] = L.data
                nc.createVariable("LH_indices", "i4", ("LH_nnz",))[:] = L.indices
                nc.createVariable("LH_indptr", "i4", ("ncols_plus_1",))[:] = L.indptr
                nc.LH_shape = L.shape
                nc.createVariable('PH', "i4", ('m',))[:] = self.factorH.P()
                
                L = self.factorP.L()
                nc.createDimension('LP_nnz', L.nnz)
                nc.createVariable("LP_data", "f4", ("LP_nnz",))[:] = L.data
                nc.createVariable("LP_indices", "i4", ("LP_nnz",))[:] = L.indices
                nc.createVariable("LP_indptr", "i4", ("ncols_plus_1",))[:] = L.indptr
                nc.LP_shape = L.shape
                nc.createVariable('PP', "i4", ('m',))[:] = self.factorP.P()
                
                nc.kt = self.kt
                nc.nkt = self.nkt
                
                nc.createDimension('g1', x*y)
                nc.createDimension('g2', self.G.shape[1])
                nc.createVariable('G', float, ('g1', 'g2'))[:] = self.G.todense()
    
#%%
    
def _make_model(G, d, q, l):
    GTQinv = G.T.multiply(1/q)  # Sparse multiplication

    GTG = GTQinv @ G  # Stays sparse
    reg_term = 10**l * np.median(GTG.diagonal()) * sp.eye(GTG.shape[0], format='csr')
    GTG = GTG + reg_term  # Sparse regularization

    GTd = GTQinv @ d  # Stays dense

    factor = cholesky(GTG.tocsc())       # Performs sparse Cholesky
    m = factor(GTd)              # Solve for m
    Cp = factor.inv()            # Full posterior covariance

    return m, Cp, factor
        
     