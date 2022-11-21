import numpy as np
import scipy
from scipy import linalg
from function import function_sparse
from multigrid import mg_solve
import multigrid as mg
import random
#from rough_trace import rough_trace


def trace_dir_mlmc_opt_PP(mg,epsilon,OPERATORS,PHATS,RHATS,N,dx,kk, level, nr_levels,work_v,ml_solvers,function_name,spec_name,params,use_Q, function,function_tol):

    A = OPERATORS[0]
    level_solver_tol = 1e-6

    from mg_solver import mg_solver
    from mg_eigsolver import mg_eigsolver

    i = level
    Af = mg.ml.levels[level].A
    Ac = mg. ml.levels[level+1].A
    R = mg.ml.levels[level].R
    P = mg.ml.levels[level].P
    n_f = Af.shape[0]
    n_c = Ac.shape[0]
#    d[level] = dx #d0+level

    random.seed(12345)
    S0 = np.random.randint(2, size=(n_f,dx))
    S0 *= 2
    S0 -= 1
    S = RHATS[i]*PHATS[i]*S0
    S = S.astype(Af.dtype)

    inv_AS_f = np.zeros((n_f,dx)).astype(complex)
    inv_AS_c = np.zeros((n_c,dx)).astype(int)#,dtype=np.float) #.astype(Ac.dtype)
    iters_AS_f = np.zeros(dx)
    iters_AS_c = np.zeros(dx)


    total_iters_AS_f = 0.0
    total_iters_AS_c = 0.0
    for k in range(kk):
        iters_AS_f = np.zeros(dx)
        iters_AS_c = np.zeros(dx)
        for ix in range(dx):  
            (inv_AS_f[:,ix],inv_AS_c[:,ix],ee,iters_AS_f[ix],iters_AS_c[ix],ss) = mg_eigsolver(i,nr_levels,mg,S[:,ix], R*S[:,ix],n_f,A,Af,Ac,params,function,function_name,spec_name,ml_solvers,use_Q)
        inv_AS = inv_AS_f - P*inv_AS_c
        total_iters_AS_f += np.sum(iters_AS_f)   #FIXME   modify it to be inside the k loop
        total_iters_AS_c += np.sum(iters_AS_c)     
        S,r = np.linalg.qr(inv_AS) #.astype(complex)       #          % Projection matrix 

    Vx = S
            
    work_hats_x = P.nnz+ R.nnz   
    work_invAS = total_iters_AS_f*work_v[i]+total_iters_AS_c*work_v[i+1] +  work_hats_x*kk
    work_qr_factr = (2*(n_f- (dx/3))*dx**2)*kk 

    inv_AV_f = np.zeros((n_f,dx), dtype=Af.dtype)
    inv_AV_c = np.zeros((n_c,dx), dtype=Ac.dtype)
    iters_AV_f = np.zeros(dx)
    iters_AV_c = np.zeros(dx)
    Vinv_AV = np.zeros(dx, dtype=Af.dtype)

    Vxx = RHATS[i]*PHATS[i]*Vx
    for k in range(dx): 
        (inv_AV_f[:,k],inv_AV_c[:,k],ee,iters_AV_f[k],iters_AV_c[k],ss)= mg_solver(i,nr_levels,mg,Vxx[:,k],R*Vxx[:,k],n_f,A,Af,Ac,params,function,function_name,spec_name,ml_solvers,use_Q)       
        Vinv_AV[k] = np.dot(Vx[:,k].transpose().conjugate(),inv_AV_f[:,k]) -np.dot(Vx[:,k].transpose().conjugate(),P*inv_AV_c[:,k])

    trace_dir = np.sum(Vinv_AV)
    sum_itersAV_f = sum(iters_AV_f)
    sum_itersAV_c = sum(iters_AV_c)     
    work_invAV = sum_itersAV_f*work_v[i]+sum_itersAV_c*work_v[i+1] + work_hats_x                #  ------- w3
    work_trace_dir = n_f*dx	                        #  ------- w4
    work_setup = (work_invAS + work_qr_factr+ work_invAV+ work_trace_dir)*1e-6

    return(trace_dir,work_setup,Vx)
    
