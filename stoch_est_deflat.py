import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
#import png
from numpy.linalg import eigh
from scipy.sparse import identity
from scipy.sparse.linalg import svds,eigsh,eigs
from scipy.sparse import diags
import time
import os
import multigrid as mg
from gamma_3_5 import gamma3_application, gamma5_application

def stoch_est_deflat(A,N,trace_max_nr_ests, Vx,use_Q,function,function_name,spec_name,nr_deflat_vctrs,params,rough_trace_tol,rough_trace):

    rough_function_tol = 1e-6
    function_iters = 0
    ests = np.zeros(trace_max_nr_ests, dtype=A.dtype)

    start = time.time()
    mg.coarsest_lev_iters[0] = 0

    # main Hutchinson loop
    for i in range(trace_max_nr_ests):

        # generate a Rademacher vector
        x = np.random.randint(2, size=N)
        x *= 2
        x -= 1
        x = x.astype(A.dtype)

        if nr_deflat_vctrs>0:
            # deflating Vx from x
            x_def = x - np.dot(Vx,np.dot(Vx.transpose().conjugate(),x))
        else:
            x_def = x

        if use_Q:
            if params['problem_name']=='schwinger':
                x_def = gamma3_application(x_def)
            elif params['problem_name']=='LQCD':
                x_def = gamma5_application(x_def,0,params['dof'])

        if params['problem_name']=='schwinger':
            mg.level_nr = 0
            z,num_iters = mg_solve( A,x_def,function_tol )
        else:
            z,num_iters = function_sparse(A,x_def,rough_function_tol,function,function_name,spec_name)

        function_iters += num_iters

        e = np.vdot(x,z)

        ests[i] = e

        # average of estimates
        ests_avg = np.sum(ests[0:(i+1)])/(i+1)
        # and standard deviation
        ests_dev = sqrt(   np.sum(   np.square(np.abs(ests[0:(i+1)]-ests_avg))   )/(i+1)   )
        error_est = ests_dev/sqrt(i+1)

        print(str(i)+" .. "+str(ests_avg)+" .. "+str(rough_trace)+" .. "+str(error_est)+" .. "+str(rough_trace_tol)+" .. "+str(num_iters))

        # break condition
        if i>=5 and error_est<rough_trace_tol:
            break

    end = time.time()
    print("\nTime to compute the trace with Deflated Hutchinson (excluding rough trace and excluding time for eigenvectors computation) : "+str(end-start)+"\n")

    return(ests_avg, ests_dev, i,function_iters)
