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
                 wscaling: Optional[bool] = False):
        
        self.cI = cI                 
        
        self.ncp = ncp
        self.k = k
        self._ncpt = ncpt
        self._cpt_step = cpt_step
        self.kt = kt
        self.lH = lH
        self.lP = lP
        self.wscaling = wscaling

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

    @property
    def LTL_diag(self):
        if self._LTL_diag is None:
            if self.wscaling:
                self._LTL_diag = self.Gt.T.dot(self.w.flatten()) / self.Gt.T.dot(np.ones(self.Gt.shape[0]))
                self._LTL_diag /= np.max(self._LTL_diag)
            else:
                self._LTL_diag = np.ones(self.Gt.shape[1])
        return self._LTL_diag
            
    @property
    def LTL(self):
        if self._LTL is None:
            self._LTL = sp.diags(self.LTL_diag, format='csr')
        return self._LTL
    
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
        # Note optimized given the repeated values in x and y
        # TODO: A look-up table should be used
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
    
    def ev_uncertainty_old(self, comp='hall', block=2000):
        C = self.CH if comp == 'hall' else self.CP
        
        diag_elements = np.zeros(self.Gt.shape[0], dtype=float)
    
        blocks = range(0, self.Gt.shape[0], block)
        loop = tqdm(blocks, total=len(blocks), desc=f'Chunked computation of {comp} uncertainty')
        for start in loop:
            end = min(start + block, self.Gt.shape[0])
            Gblock = self.Gt[start:end]                 # (B × n) CSR
            
            V = C @ Gblock.T                     # (n × B) CSC
            diag_elements[start:end] = np.sum(Gblock.multiply(V.T), axis=1).A.ravel()
        
        return np.sqrt(diag_elements).reshape((self.nt, self.n, self.n))
    
    def ev_uncertainty(self, comp='hall', block=2000):
        """
        Compute sqrt(diag(G Cp G^T)) without forming Cp.
        Uses CHOLMOD triangular solves instead of explicit posterior covariance.
        """
        factor = self.factorH if comp == 'hall' else self.factorP    # Cholesky object
    
        diag_elements = np.zeros(self.Gt.shape[0], dtype=float)
    
        blocks = range(0, self.Gt.shape[0], block)
        loop = tqdm(blocks, total=len(blocks), desc=f'Chunked computation of {comp} uncertainty')    
        for start in loop:
            end = min(start + block, self.Gt.shape[0])
    
            # Grab block of rows (B × nmodel)
            Gblock = self.Gt[start:end]
    
            # Solve Λ X = Gblock.T  →  X = Cp @ Gblock.T
            # This gives an (nmodel × B) matrix
            Y = factor.solve_L(Gblock.T)
            X = factor.solve_Lt(Y)
                        
            #X = factor.solve_A(Gblock.T)     # batched triangular solve
    
            # diag(G Cp G^T)_{i} = sum_j G[i,j] * X[j,i]
            # X.T has same shape as Gblock
            diag_elements[start:end] = np.sum(Gblock.multiply(X.T), axis=1).A.ravel()
    
        return np.sqrt(diag_elements).reshape((self.nt, self.n, self.n))


    '''
    def ev_uncertainty(self, comp='hall', block=1000):
        factor = self.solverH.factor if comp == 'hall' else self.solverP.factor
        
        diag_elements = np.zeros(self.Gt.shape[0], dtype=float)
        
        import scipy.sparse
        
        blocks = range(0, self.Gt.shape[0], block)
        loop = tqdm(blocks, total=len(blocks), desc=f'Chunked computation of {comp} uncertainty')
        
        for start in loop:
            end = min(start + block, self.Gt.shape[0])
            
            # Get block of rows as dense (if sparse enough) or keep sparse
            Gblock = self.Gt[start:end]  # (block × n)
            
            # Build RHS matrix: each column corresponds to one row of Gblock
            # We need to solve C^(-1) @ V = Gblock.T, so V = C @ Gblock.T
            if Gblock.nnz / Gblock.size < 0.01:  # Very sparse
                # Convert to dense for batch solving
                RHS = Gblock.T.toarray()  # (n × block)
            else:
                RHS = Gblock.T.tocsc()  # Keep sparse in CSC format
            
            # Solve all systems at once - this is the key optimization
            V = factor.solve_A(RHS)  # (n × block)
            
            # Extract diagonals efficiently
            if scipy.sparse.issparse(V):
                V = V.toarray()
            
            # Compute g_i @ v_i for each i in block
            diag_block = np.sum(Gblock.toarray() * V.T, axis=1)
            diag_elements[start:end] = diag_block
        
        return np.sqrt(diag_elements).reshape((self.nt, self.n, self.n))
    '''

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
    
    def to_nc_old(self, filename: str):
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
                nc.createVariable('dH', 'f8', ('time', 'dim1', 'dim2'))[:] = self.pdH
                nc.createVariable('dP', 'f8', ('time', 'dim1', 'dim2'))[:] = self.pdH
    
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
                            
                nc.kt = self.k
                nc.nkt = self.nk
                nc.kt = self.kt
                nc.nkt = self.nkt
                
                nc.createDimension('g1', x*y)
                nc.createDimension('g2', self.G.shape[1])
                nc.createVariable('G', float, ('g1', 'g2'))[:] = self.G.todense()

    def to_nc(self, filename: str):
            """
            Save spline image to a NetCDF4 file.
            Can be read/rebuilt using the icReader library.
    
            Parameters
            ----------
            filename : str
                Full path to output NetCDF file.
            """
            with Dataset(filename, 'w') as nc:
                
                # Groups
                data_grp = nc.createGroup("data")
                model_grp = nc.createGroup("model")
                spline_grp = nc.createGroup("spline")
                grid_grp = nc.createGroup("grid")
                
                
                # Data space group
                t, y, x = self.nt, self.n, self.n
                data_grp.createDimension('time', t)
                data_grp.createDimension('dim1', y)
                data_grp.createDimension('dim2', x)
    
                data_grp.createVariable('H',  'f8', ('time', 'dim1', 'dim2'))[:] = self.pH
                data_grp.createVariable('P',  'f8', ('time', 'dim1', 'dim2'))[:] = self.pP
                data_grp.createVariable('dH', 'f8', ('time', 'dim1', 'dim2'))[:] = self.pdH
                data_grp.createVariable('dP', 'f8', ('time', 'dim1', 'dim2'))[:] = self.pdH
    
                if self.time is not None:
                    ref_time = datetime(2000, 1, 1)
                    time_seconds = np.array([(t - ref_time).total_seconds() for t in self.time], dtype=np.int32)
                    data_grp.createVariable("time", np.int32, ("time",))[:] = time_seconds
                    data_grp.reference_time = ref_time.strftime("%Y-%m-%dT%H:%M:%S")
    
                # Grid group
                if self.grid and hasattr(self.grid, "projection"):
                    grid_grp.position     = self.grid.projection.position.astype(float)
                    grid_grp.orientation  = self.grid.projection.orientation
                    grid_grp.L    = self.grid.L
                    grid_grp.W    = self.grid.W
                    grid_grp.Lres = self.grid.Lres
                    grid_grp.Wres = self.grid.Wres
                    grid_grp.gridR    = self.grid.R

                
                # Model group
                model_grp.createDimension('m', self.mH.size)
                model_grp.createDimension('ncols_plus_1', self.mH.size+1)
                
                model_grp.createVariable('mH', 'f8', ('m',), zlib=True)[:] = self.mH
                model_grp.createVariable('mP', 'f8', ('m',), zlib=True)[:] = self.mP
                                
                L = self.factorH.L()
                model_grp.createDimension('LH_nnz', L.nnz)
                model_grp.createVariable("LH_data", "f4", ("LH_nnz",), zlib=True)[:] = L.data
                model_grp.createVariable("LH_indices", "i4", ("LH_nnz",), zlib=True)[:] = L.indices
                model_grp.createVariable("LH_indptr", "i4", ("ncols_plus_1",), zlib=True)[:] = L.indptr
                model_grp.LH_shape = L.shape
                model_grp.createVariable('PH', "i4", ('m',), zlib=True)[:] = self.factorH.P()
                
                L = self.factorP.L()
                model_grp.createDimension('LP_nnz', L.nnz)
                model_grp.createVariable("LP_data", "f4", ("LP_nnz",), zlib=True)[:] = L.data
                model_grp.createVariable("LP_indices", "i4", ("LP_nnz",), zlib=True)[:] = L.indices
                model_grp.createVariable("LP_indptr", "i4", ("ncols_plus_1",), zlib=True)[:] = L.indptr
                model_grp.LP_shape = L.shape
                model_grp.createVariable('PP', "i4", ('m',), zlib=True)[:] = self.factorP.P()
                
                
                # Spline group
                spline_grp.kt = self.k
                spline_grp.nkt = self.nk
                spline_grp.kt = self.kt
                spline_grp.nkt = self.nkt

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
            #self._GT = self.G[self.f].T.multiply(1/(self.q[self.f])) # Sparse multiplication
        return self._GT
    
    @property
    def GTG(self):
        if self._GTG is None:
            self._GTG = self.GT @ self.G[self.f] # Stays sparse
        return self._GTG
    
    @property
    def A(self):
        if self._A is None:
            self._A = self.GTG + self.l * np.median(self.GTG.diagonal()) * self.LTL # Sparse regularization
        return self._A
    
    @property
    def GTd(self):
        if self._GTd is None:
            self._GTd = self.GT @ self.d[self.f]  # Stays dense
        return self._GTd
    
    
    
    
    
    
    
    
        
    
    

