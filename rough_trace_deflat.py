import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
#import gamma_3_5
#from gamma_3_5 import gamma3_application,gamma5_application

from utils import flopsV,flopsV_manual
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation
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
import time
import os

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection,symmetric_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation
#import pymatlab
#import matlab.engine
from multigrid import mg_solve
import multigrid as mg
from numpy.linalg import inv



def rough_trace_deflat(N,A,function_tol,function_name,spec_name,params,function,use_Q,trace_tol):


    np.random.seed(123456)

    # pre-compute a rough estimation of the trace, to set then a tolerance
    nr_rough_iters = 5
    ests = np.zeros(nr_rough_iters, dtype=A.dtype)

    start = time.time()

    # main Hutchinson loop
    for i in range(nr_rough_iters):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1

        x = x.astype(A.dtype)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        if params['problem_name']=='schwinger':
            mg.level_nr = 0
            z,num_iters = mg_solve( A,x,function_tol )
        else:
            z,num_iters = function_sparse(A,x,function_tol,function,function_name,spec_name)

        if use_Q:
            if params['problem_name']=='schwinger':
                x = gamma3_application(x)
            elif params['problem_name']=='LQCD':
                x = gamma5_application(x,0,params['dof'])

        e = np.vdot(x,z)
        ests[i] = e

    end = time.time()
    print("Time to compute rough estimation of trace : "+str(end-start)+"\n")

    rough_trace = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    rough_trace_tol = abs(trace_tol*rough_trace)

    return (rough_trace, rough_trace_tol)

