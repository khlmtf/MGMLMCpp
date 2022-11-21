import numpy as np
import scipy as sp
from math import sqrt, pow
import pyamg
from scipy.sparse import csr_matrix
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
#import png
from numpy.linalg import eigh
from scipy.sparse import identity

from scipy.sparse.linalg import svds,eigsh,eigs
from scipy.sparse import diags

from scipy.linalg import expm
from math import exp
import matplotlib.mlab as mlab


def get_work_v(mg, nr_levels, params):

    A = mg.ml.levels[0].A
    N = A.shape[0]

    RESTRICTIONS = []
    PROLONGATIONS = []
    for i in range(nr_levels-1):
        mg.coarsest_lev_iters[i] = 0
        PROLONGATIONS.append(mg.ml.levels[i].P)
        RESTRICTIONS.append(mg.ml.levels[i].R)

    nnz_Al = np.zeros(nr_levels)
    work_s = np.zeros(nr_levels)        # work performed on a given level within the V-cycle
    work_v = np.zeros(nr_levels)        # work for one V-cycle of multigrid

    if not params['problem_name']=='schwinger':
        no_smoothing = 1
        for j in range(nr_levels):
            nnz_Al[j] = mg.ml.levels[j].A.nnz       
            if j<(nr_levels-1):
                work_s[j] = (2*no_smoothing+2)*nnz_Al[j]      #  %+3 for the comp. of the residual
                work_s[j] = work_s[j] + RESTRICTIONS[j].nnz + PROLONGATIONS[j].nnz
            else:
                work_s[j] = (mg.ml.levels[j].A.shape[0])^2
    else:
        no_smoothing = 2
        kx = 0
        for j in range(nr_levels):
            nnz_Al[j] = mg.ml.levels[j].A.nnz       
            if j<(nr_levels-1):
                if kx ==0:
                    work_s[j] = (2*no_smoothing+3+1)*nnz_Al[j]  
                else: 
                    work_s[j] = (2*no_smoothing+3+0)*nnz_Al[j]  
                work_s[j] = work_s[j] + RESTRICTIONS[j].nnz + PROLONGATIONS[j].nnz
            else:
                work_s[j] = (mg.ml.levels[j].A.shape[0])^2
            kx +=1

    for k in range(nr_levels):
        work_v[k] = sum(work_s[k:nr_levels])

    # compute the work hats 

    RHATS = []
    PHATS = []
    RHATS.append(sp.sparse.identity(N,dtype=A.dtype))
    PHATS.append(sp.sparse.identity(N,dtype=A.dtype))
    
    for i in range(1,nr_levels):
        Rhats = mg.ml.levels[i-1].R*RHATS[i-1]           
        Phats = PHATS[i-1]*mg.ml.levels[i-1].P

        RHATS.append(Rhats)
        PHATS.append(Phats)

    work_hats = np.zeros(nr_levels)       
#    for j in range(nr_levels-1) :
#        work_hats[j] = PHATS[j].nnz +RHATS[j].nnz + PHATS[j+1].nnz + RHATS[j+1].nnz 

    return(work_v,work_hats,PHATS,RHATS)



