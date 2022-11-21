import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
import gamma_3_5
from gamma_3_5 import gamma3_application,gamma5_application

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
import random

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



def rough_trace(epsilon,N,A,function_tol,function_name,spec_name,params,ml_solvers,level,nr_rough_iters, use_Q):

    function_iters = 0
    rough_trace_tol = 0
    rough_nr_ests = 5
    function_tol = 1e-6
    ests = np.zeros(rough_nr_ests, dtype=A.dtype)

    start = time.time()
    mg.coarsest_lev_iters[0] = 0

    # main Hutchinson loop
#    i =0
    for i in range(nr_rough_iters):
        # generate a Rademacher vector

#        print(i)
        random.seed(1235)
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

    trace_est5 = np.sum(ests[0:nr_rough_iters])/(nr_rough_iters)
    rough_tol = abs(epsilon*trace_est5)


#    end = time.time()
#    print("\nTime to compute the trace with Deflated Hutchinson (excluding rough trace and excluding time for eigenvectors computation) : "+str(end-start)+"\n")

    return (trace_est5, rough_tol)


