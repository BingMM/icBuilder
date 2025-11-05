#%% Import

from typing import Optional

import numpy as np
from secsy import CSgrid, CSprojection
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from copy import deepcopy as dcopy
from tqdm import tqdm
from scipy.io import netcdf_file
from scipy.interpolate import BSpline
import scipy
from scipy.sparse import kron, vstack, csc_matrix
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from scipy.optimize import lsq_linear

from icreader import ConductanceImage

#%%
class splineImage():
    def __init__(self, 
                 cI: ConductanceImage,
                 ncp: Optional[int] = 30,
                 ncpt: Optional[int] = None,
                 k: Optional[int] = 3,
                 kt: Optional[int] = 2):
        
        self.cI = cI                 
        
        self.ncp = ncp
        self.k = k
        self._ncpt = ncpt
        self.kt = kt

        self.reset_time()
        self.reset_spline()

    @property
    def ncpt(self):
        if self._ncpt is None:
            self._ncpt = self.nt//2
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
            self._t = np.array([(t - self.time[0]).total_seconds for t in self.time])
        return self._t
    
    @property
    def nt(self):
        return self.time.size
    
#%% Grid

    @property
    def x(self):
        return self.cI.grid.xi
    
    @property
    def eta(self):
        return self.cI.grid.eta
    
    @property
    def shape(self):
        return self.cI.grid.shape
    
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
        self.tknots = np.r_[[self.t.min()]*self.kt, np.linspace(self.t.min(), self.t.max(), self.nkt-2*self.kt), [self.t.max()]*self.kt]

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
        
    def generate_design_matrix_2d(self, return_sparse=False):
        G = np.zeros((self.x.size, self.ncp**2))
        for i, (xi, yi) in enumerate(zip(self.x.flatten(), self.y.flatten())):
            Gx = BSpline.design_matrix(xi, self.knots, self.k).todense()
            Gy = BSpline.design_matrix(yi, self.knots, self.k).todense()
            Gy = np.kron(np.eye(self.ncp), Gy)
            G[i, :] = Gx.dot(Gy)
        if return_sparse:
            return csc_matrix(G)
        return G
        
    def generate_design_matrix_3d(self):
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
        

#%%

        self.H = np.copy(H)
        self.P = np.copy(P)
        self.shape = self.H.shape        
        
        if Hu is None:
            #self.Hu = np.stack([np.eye(self.shape[1])*self.shape[0]])
            self.Hu = np.ones(self.shape)
        else:
            self.Hu = np.copy(Hu)
        
        if Pu is None:
            #self.Pu = np.stack([np.eye(self.shape[1])*self.shape[0]])
            self.Pu = np.ones(self.shape)
        else:
            self.Pu = np.copy(Pu)
        
        self.grid = dcopy(grid)
        
        # Define dimensions
        self.n = self.shape[1]
        self.nt = self.shape[0] # time dimension
        
        # Grid coordinates
        self.x = np.copy(self.grid.xi)
        self.y = np.copy(self.grid.eta)
        
        if t is None:
            self.t = np.arange(self.nt)
        else:
            self.t = t
        
        self.ncp = ncp
        self.ncpt = ncpt
        self.k = k
        self.kt = kt
        
        # Amount of knots
        self.nk = ncp + k + 1
        self.nkt = ncpt + kt + 1 # time dimension
        
        # Define truncated knot location
        self.knots = np.r_[[self.x.min()]*self.k, np.linspace(self.x.min(), self.x.max(), self.nk-2*self.k), [self.x.max()]*self.k]
        self.tknots = np.r_[[self.t.min()]*self.kt, np.linspace(self.t.min(), self.t.max(), self.nkt-2*self.kt), [self.t.max()]*self.kt]
    
    def generate_design_matrix_2d(self):
        self.Gs = np.zeros((self.x.size, self.ncp**2))
        for i, (xi, yi) in enumerate(zip(self.x.flatten(), self.y.flatten())):
            Gx = BSpline.design_matrix(xi, self.knots, self.k).todense()
            Gy = BSpline.design_matrix(yi, self.knots, self.k).todense()
            Gy = np.kron(np.eye(self.ncp), Gy)
            self.Gs[i, :] = Gx.dot(Gy)
        
    def generate_design_matrix_3d(self):        
        self.Gs_s = csc_matrix(self.Gs)
        G = []
        for i, ti in enumerate(self.t):
            Gt = BSpline.design_matrix(ti, self.tknots, self.kt)  # Already sparse
            Gt = kron(np.eye(self.Gs.shape[1]), Gt, format='csr')  # Ensure sparse output
            G.append(self.Gs_s @ Gt)  # Matrix multiplication remains sparse
        self.G = vstack(G, format='csr')  # Efficient sparse stacking

    
    def make_models(self, lH=0, lP=0):
        self.lH = lH
        self.lP = lP
        
        if not hasattr(self, 'G'):
            self.generate_design_matrix_3d()

        def make_model_sparse(G, d, C, l=0, nnls=False):
            Cinv = 1 / C  # Element-wise inverse
            GTCinv = G.T.multiply(Cinv)  # Sparse multiplication
    
            GTG = GTCinv @ G  # Stays sparse
            reg_term = l * np.median(GTG.diagonal()) * sp.eye(GTG.shape[0], format='csr')
            GTG = GTG + reg_term  # Sparse regularization

            GTd = GTCinv @ d  # Stays dense
            
            # Solve GTG * m = GTd directly (no explicit inverse)
            print('Start solve')
            if nnls:
                m = lsq_linear(GTG, GTd, bounds=(0, np.inf), method='trf').x
            else:
                m = spla.spsolve(GTG, GTd)
            print('End solve')
            
            print('Start post')
            # Compute posterior covariance (C_m)
            #solver = spla.factorized(GTG)  # Factorize once
            #Cp = np.column_stack([solver(np.eye(GTG.shape[0])[:, i]) for i in range(GTG.shape[0])])
            print('End post')

            #return m, Cp
            return m, 0

        def make_model(G, d, C, l=0):
            Cinv = 1 / C  # Element-wise inverse
            GTCinv = G.T.multiply(Cinv)  # Sparse multiplication
    
            GTG = (GTCinv @ G).todense()  # Stays sparse
            #reg_term = l * np.median(GTG.diagonal()) * sp.eye(GTG.shape[0], format='csr')
            reg_term = l * np.median(np.diag(GTG)) * np.eye(GTG.shape[0])
            GTG += reg_term  # Sparse regularization

            GTd = GTCinv @ d  # Stays dense

            # Solve for Cp efficiently (Cp = GTG^{-1})
            Cp = scipy.linalg.lstsq(GTG, np.eye(GTG.shape[0]), lapack_driver='gelsy')[0]  # Uses sparse solver

            # Solve for m (m = Cp @ GTd)
            m = Cp.dot(GTd)

            return m, Cp

        def make_model_nnls(G, d, C, l=0):
            Cinv = 1 / C
            GTCinv = G.T.multiply(Cinv)  # Sparse multiplication
            
            GTG = (GTCinv @ G).todense()  # Stays sparse
            reg_term = l * np.median(np.diag(GTG)) * np.eye(GTG.shape[0])
            GTG += reg_term  # Sparse regularization
            
            GTd = GTCinv @ d  # Stays dense
            
            # Solve NNLS
            m, _ = opt.nnls(GTG, GTd)

            # Identify active parameters
            active_idx = np.where(m > 1e-10)[0]  # Only consider nonzero parameters

            # Compute posterior covariance only for active parameters
            if len(active_idx) > 0:
                GTG_active = GTG[np.ix_(active_idx, active_idx)]
                Cp_active = scipy.linalg.inv(GTG_active)  # Inverse of the active part
            else:
                Cp_active = np.zeros((GTG.shape[0], GTG.shape[0]))

            #return m, Cp_active, active_idx
            return m, Cp_active

        f = ~(np.isnan(self.H) | np.isinf(self.H) | np.isnan(self.Hu) | np.isinf(self.Hu)).flatten()
        #self.mH, self.CpH = make_model(self.G[f, :], self.H.flatten()[f], self.Hu.flatten()[f], self.lH)
        #self.mH, self.CpH = make_model_sparse(self.G[f, :], self.H.flatten()[f], self.Hu.flatten()[f], self.lH) # Remove nans
        H = self.H.flatten()
        Hu = self.Hu.flatten()
        H[~f] = 0
        Hu[~f] = 1
        self.mH, self.CpH = make_model_sparse(self.G, H, Hu, self.lH)
        #self.mP, self.CpP = self.mH, self.CpH
        
        P = self.P.flatten()
        Pu = self.Pu.flatten()
        P[~f] = 0
        Pu[~f] = 1
        self.mP, self.CpP = make_model_sparse(self.G, P, Pu, self.lP)
        
        #f = ~(np.isnan(self.P) | np.isinf(self.P) | np.isnan(self.Pu) | np.isinf(self.Pu)).flatten()
        #self.mP, self.CpP = make_model(self.G[f, :], self.P.flatten()[f], self.Pu.flatten()[f], self.lP)
    
    def eval(self, m, Cp=None):
        pred = self.G.dot(m).reshape(self.shape)
        if Cp is None:
            return pred
        
        sig = np.zeros(self.G.shape[0])
        for i in range(sig.size):
            sig[i] = self.G[i, :].dot(Cp).dot(self.G[i, :].T)
        
        return pred, sig
    
    def eval_Hall(self, uncertainty=False):
        if uncertainty:            
            pred, sig = self.eval(self.mH, self.CpH)
            return pred, sig
        pred = self.eval(self.mH)
        return pred
    
    def eval_Pedersen(self, uncertainty=False):
        if uncertainty:            
            pred, sig = self.eval(self.mP, self.CpP)
            return pred, sig
        pred = self.eval(self.mP)
        return pred
