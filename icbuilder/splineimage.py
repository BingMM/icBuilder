#%% Import

from typing import Optional
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import kron, vstack, csc_matrix
import scipy.sparse as sp
from icreader import ConductanceImage
from sksparse.cholmod import cholesky 
from datetime import datetime
from tqdm import tqdm
from netCDF4 import Dataset

#%%
class SplineImage():
    def __init__(self, 
                 cI: ConductanceImage,
                 ncp: Optional[int] = 20,
                 ncpt: Optional[int] = None,
                 cpt_step: Optional[int] = None,
                 k: Optional[int] = 3,
                 kt: Optional[int] = 2,
                 lH: Optional[int] = 0,
                 lP: Optional[int] = 0,
                 wscaling: Optional[bool] = False,
                 psamp: Optional[int] = 5000):
        
        self.cI = cI                 
        
        self.ncp = ncp
        self.k = k
        self._ncpt = ncpt
        self._cpt_step = cpt_step
        self.kt = kt
        self.lH = lH
        self.lP = lP
        self.wscaling = wscaling
        self.psamp = psamp

        self.reset_time()
        self.reset_spline()
        self.reset_model()
        self.reset_ev()
        self.reset_reg()

    @property
    def cpt_step(self):
        if self._cpt_step is None:
            self._cpt_step = 5
        return self._cpt_step

    @property
    def ncpt(self):
        if self._ncpt is None:
            self._ncpt = int((self.time[-1] - self.time[0]).total_seconds() / 60 / self.cpt_step)
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
    
    @property
    def w(self):
        return np.where(np.isnan(self.cI.w), 1, self.cI.w)

#%% Regularization scaling
    
    def reset_reg(self):
        self._LTL_diag = None
        self._LTL = None
        self._tLTL = None
        self._xLTL = None
        self._yLTL = None
        self._sLTL = None

    @property
    def LTL_diag(self):
        if self._LTL_diag is None:
            if self.wscaling:
                numerator = self.Gt.T.dot(self.w.flatten())
                denominator = self.Gt.T.dot(np.ones(self.Gt.shape[0]))
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._LTL_diag = numerator / denominator
                self._LTL_diag /= np.nanmax(self._LTL_diag)
                self._LTL_diag[np.isnan(self._LTL_diag)] = 1
            else:
                self._LTL_diag = np.ones(self.Gt.shape[1])
        return self._LTL_diag
            
    @property
    def LTL(self):
        if self._LTL is None:
            self._LTL = sp.diags(self.LTL_diag, format='csr')
        return self._LTL
    
    @property
    def tLTL(self):        
        if self._tLTL is None:
            Gt = []
            BS = BSpline(self.tknots, np.eye(self.ncpt), self.kt)
            for i, ti in enumerate(self.t):
                Gt_ = sp.csr_matrix(BS(ti, nu=1))
                Gt_ = kron(np.eye(self.G.shape[1]), Gt_, format='csr')  # Ensure sparse output
                Gt.append(self.G @ Gt_)  # Matrix multiplication remains sparse
            Gt = vstack(Gt, format='csr')
            self._tLTL = Gt.T@Gt
            self._tLTL /= np.median(self._tLTL.diagonal())
        return self._tLTL
    
    @property
    def xLTL(self):        
        if self._xLTL is None:
            self._xLTL, self._yLTL = self.get_xyLTL()
        return self._xLTL
    
    @property
    def yLTL(self):        
        if self._yLTL is None:
            self._xLTL, self._yLTL = self.get_xyLTL()
        return self._yLTL

    def get_xyLTL(self):
        Gx = np.zeros((self.x.size, self.ncp**2))
        Gy = np.zeros((self.x.size, self.ncp**2))
        for i, (xi, yi) in enumerate(zip(self.x.flatten(), self.y.flatten())):
            Gx[i, :] = self.splx(xi, nu=1).dot(np.kron(np.eye(self.ncp), self.sply(yi)))
            Gy[i, :] = self.splx(xi).dot(np.kron(np.eye(self.ncp), self.sply(yi, nu=1)))            
        Gx, Gy = csc_matrix(Gx), csc_matrix(Gy)
        
        Gtx, Gty = [], []
        for i, ti in enumerate(self.t):
            Gti = kron(np.eye(Gx.shape[1]), self.splt(ti), format='csr')            
            Gtx.append(Gx @ Gti)
            Gty.append(Gy @ Gti)
        Gtx, Gty = vstack(Gtx, format='csr'), vstack(Gty, format='csr')
        
        return Gtx.T@Gtx, Gty.T@Gty
    
    @property
    def sLTL(self):
        if self._sLTL is None:
            self._sLTL = self.xLTL + self.yLTL
            self._sLTL /= np.median(self._sLTL.diagonal())
        return self._sLTL
    
#%% Spline

    def reset_spline(self):
        self._G = None
        self._Gt = None
        self._splx = None
        self._splt = None

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
    def splx(self):
        if self._splx is None:
            self._splx = BSpline(self.knots, np.eye(self.ncp), self.k)
        return self._splx
    
    @property
    def sply(self):
        return self._splx
    
    @property
    def splt(self):
        if self._splt is None:
            self._splt = BSpline(self.tknots, np.eye(self.ncpt), self.kt)
        return self._splt

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
            G[i, :] = self.splx(xi).dot(np.kron(np.eye(self.ncp), self.sply(yi)))
        if return_sparse:
            return csc_matrix(G)
        return G
        
    def generate_G_3d(self):
        Gt = []
        for i, ti in enumerate(self.t):
            Gt.append(self.G @ kron(np.eye(self.G.shape[1]), self.splt(ti), format='csr'))
        return vstack(Gt, format='csr')  # Efficient sparse stacking

#%% Models

    def reset_model(self):
        self._solverH = None
        self._solverP = None
        self._solverdH = None
        self._solverdP = None
        self._factord = None
    
    @property
    def solverH(self):
        if self._solverH is None:
            self._solverH = Solver(self.Gt, d=self.H.flatten(), q=self.dH.flatten(), l=10**self.lH, LTL=self.LTL)
        return self._solverH
    
    @property
    def solverP(self):
        if self._solverP is None:
            self._solverP = Solver(self.Gt, d=self.P.flatten(), q=self.dH.flatten(), l=10**self.lP, LTL=self.LTL)
        return self._solverP
    
    @property
    def factord(self):
        # TODO: This needs to be fixed, maybe
        if self._factord is None:
            self._factord = Solver(self.Gt, l=1e-5).factor
        return self._factord
    
    @property
    def solverdH(self):
        if self._solverdH is None:
            self._solverdH = Solver(self.Gt, d=self.pdH.flatten(), factor=self.factord)
        return self._solverdH
    
    @property
    def solverdP(self):
        if self._solverdP is None:
            self._solverdP = Solver(self.Gt, d=self.pdP.flatten(), factor=self.factord)
        return self._solverdP    
    
    @property
    def factorH(self):
        return self.solverH.factor

    @property
    def factorP(self):
        return self.solverP.factor    
    
    @property
    def mH(self):
        return self.solverH.m
    
    @property
    def mP(self):
        return self.solverP.m
    
    @property
    def CH(self):
        return self.solverH.C

    @property
    def CP(self):
        return self.solverP.C  
        
    @property
    def mdH(self):
        return self.solverdH.m
    
    @property
    def mdP(self):
        return self.solverdP.m
    
#%% Evaluation

    def reset_ev(self):
        self._pH = None        
        self._pP = None
        self._pdH = None
        self._pdP = None
        
        self._pdH_m = None
        self._pdP_m = None
   
    @property
    def pH(self):
        if self._pH is None:
            self._pH = self.ev(self.mH)
        return self._pH
    
    @property
    def pP(self):
        if self._pP is None:
            self._pP = self.ev(self.mP)
        return self._pP
    
    def ev(self, m):
        return self.Gt.dot(m).reshape((self.nt, self.n, self.n))
    
    @property
    def pdH(self):
        if self._pdH is None:
            self._pdH = self.ev_uncertainty(comp='hall')
        return self._pdH
            
    @property
    def pdP(self):
        if self._pdP is None:
            self._pdP = self.ev_uncertainty(comp='pedersen')
        return self._pdP
        
    def ev_uncertainty(self, comp='hall'):
        """
        Fast stochastic estimation of the posterior variance.
        Uses Hutchinson's trace estimator (Rademacher probing).
        
        Parameters
        ----------
        comp : str
            'hall' or 'pedersen'
        n_samples : int
            Number of random probe vectors. 
            30 is fast/rough, 100 is very accurate.
        """
        factor = self.factorH if comp == 'hall' else self.factorP
        
        # Dimensions
        n_data = self.Gt.shape[0]  # Total pixels in movie
        
        # 1. Generate Rademacher vectors (+1 or -1)
        # Shape: (n_data_points, n_samples)
        # We process all samples at once or in batches if memory is tight
        #print(f"Estimating {comp} uncertainty with {n_samples} stochastic probes...")
        rng = np.random.default_rng(seed=1337)
        z = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(n_data, self.psamp))
        
        # 2. Project random vectors to model space: v = G.T @ z
        # This uses the sparse matrix transpose
        v = self.Gt.T.dot(z)
        
        # 3. Solve in model space: u = A^-1 @ v
        # sksparse can solve for multiple RHS vectors at once efficiently
        u = factor.solve_A(v)
        
        # 4. Project back to data space: w = G @ u
        w = self.Gt.dot(u)
        
        # 5. Estimate diagonal: mean(z * w)
        # element-wise multiplication followed by mean across the random samples
        # Since z is +/- 1, dividing by z is the same as multiplying by z
        diag_estimates = np.mean(z * w, axis=1)
        
        # Clip negative values (numerical noise around 0) and sqrt
        uncertainty = np.sqrt(np.maximum(diag_estimates, 0))
        
        return uncertainty.reshape((self.nt, self.n, self.n))

    @property
    def pdH_m(self):
        if self._pdH_m is None:
            self._pdH_m = self.ev(self.mdH)
        return self._pdH_m
    
    @property
    def pdP_m(self):
        if self._pdP_m is None:
            self._pdP_m = self.ev(self.mdP)
        return self._pdP_m
    

#%% Save image
    
    def spline_to_nc(self, filename: str):
            """
            Save spline image to a NetCDF4 file.
            Can be read/rebuilt using the icReader library.
    
            Parameters
            ----------
            filename : str
                Full path to output NetCDF file.
            """
            with Dataset(filename, 'w') as nc:
                                                
                nc.createDimension('time', self.nt)
                nc.createDimension('m', self.mH.size)
    
                if self.time is not None:
                    ref_time = datetime(2000, 1, 1)
                    time_seconds = np.array([(t - ref_time).total_seconds() for t in self.time], dtype=np.int32)
                    nc.createVariable("time", np.int32, ("time",), zlib=True)[:] = time_seconds
                    nc.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")
                
                nc.createVariable("ssalon", 'f8', ('time',), zlib=True)[:] = self.cI.ssalon
                
                if self.grid and hasattr(self.grid, "projection"):
                    nc.position     = self.grid.projection.position.astype(float)
                    nc.orientation  = self.grid.projection.orientation
                    nc.L        = self.grid.L
                    nc.W        = self.grid.W
                    nc.Lres     = self.grid.Lres
                    nc.Wres     = self.grid.Wres
                    nc.gridR    = self.grid.R
                
                nc.createVariable('mH', 'f8', ('m',), zlib=True)[:]     = self.mH
                nc.createVariable('mP', 'f8', ('m',), zlib=True)[:]     = self.mP
                nc.createVariable('mdH', 'f8', ('m',), zlib=True)[:]    = self.mdH
                nc.createVariable('mdP', 'f8', ('m',), zlib=True)[:]    = self.mdP
                
                nc.k = self.k
                nc.nk = self.nk
                nc.kt = self.kt
                nc.nkt = self.nkt
    
    def factor_to_nc(self, filename: str):
            """
            Save spline factor to a NetCDF4 file.
            Can be read/rebuilt using the icReader library.
    
            Parameters
            ----------
            filename : str
                Full path to output NetCDF file.
            """
            with Dataset(filename, 'w') as nc:
                                
                # Model group
                nc.createDimension('m', self.mH.size)
                nc.createDimension('ncols_plus_1', self.mH.size+1)
                
                L = self.factorH.L()
                nc.createDimension('LH_nnz', L.nnz)
                nc.createVariable("LH_data", "f4", ("LH_nnz",), zlib=True)[:] = L.data
                nc.createVariable("LH_indices", "i4", ("LH_nnz",), zlib=True)[:] = L.indices
                nc.createVariable("LH_indptr", "i4", ("ncols_plus_1",), zlib=True)[:] = L.indptr
                nc.LH_shape = L.shape
                nc.createVariable('PH', "i4", ('m',), zlib=True)[:] = self.factorH.P()
                
                L = self.factorP.L()
                nc.createDimension('LP_nnz', L.nnz)
                nc.createVariable("LP_data", "f4", ("LP_nnz",), zlib=True)[:] = L.data
                nc.createVariable("LP_indices", "i4", ("LP_nnz",), zlib=True)[:] = L.indices
                nc.createVariable("LP_indptr", "i4", ("ncols_plus_1",), zlib=True)[:] = L.indptr
                nc.LP_shape = L.shape
                nc.createVariable('PP', "i4", ('m',), zlib=True)[:] = self.factorP.P()


#%% Solver class
    
class Solver():
    def __init__(self, G, d=None, q=None, LTL=None, l=0, factor=None):                 
        
        self.G = G        
        self._d = d
        self._q = q         
        self._LTL = LTL
        
        self.l = l
        self._factor = factor
        self._m = None
        self._C = None
        self._GT = None
        self._A = None
        self._GTG = None
        self._GTd = None
        self._f = None
    
    @property
    def LTL(self):
        if self._LTL is None:
            self._LTL = sp.eye(self.G.shape[1], format='csr')
        return self._LTL
    
    @property
    def d(self):
        if self._d is None:
            self._d = np.ones(self.G.shape[0])
        return self._d
    
    @property
    def q(self):
        if self._q is None:
            self._q = np.ones(self.G.shape[0])
        return self._q
    
    @property
    def f(self):
        if self._f is None:
            self._f = ~(np.isnan(self.d) | np.isinf(self.d) | np.isnan(self.q) | np.isinf(self.q))
        return self._f
    
    @property
    def factor(self):
        if self._factor is None:
            self._factor = cholesky(self.A.tocsc()) # Performs sparse Cholesky
        return self._factor
    
    @property
    def m(self):
        if self._m is None:
            self._m = self.factor(self.GTd) # Solve for m
        return self._m
    
    @property
    def C(self):
        if self._C is None:
            self._C = self.factor.inv() # Full posterior covariance
        return self._C
    
    @property
    def GT(self):
        if self._GT is None:
            self._GT = self.G[self.f].T.multiply(1/(self.q[self.f])**2) # Sparse multiplication
        return self._GT
    
    @property
    def GTG(self):
        if self._GTG is None:
            self._GTG = self.GT @ self.G[self.f] # Stays sparse
        return self._GTG
    
    @property
    def gtg_mag(self):
        return np.median(self.GTG.diagonal()[self.GTG.diagonal() != 0])
    
    @property
    def A(self):
        if self._A is None:
            self._A = self.GTG + self.l * self.gtg_mag * self.LTL # Sparse regularization
        return self._A
    
    @property
    def GTd(self):
        if self._GTd is None:
            self._GTd = self.GT @ self.d[self.f]  # Stays dense
        return self._GTd
    
    
    
    
    
    
    
    
        
    
    

