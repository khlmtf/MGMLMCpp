import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
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

import time
import os

from pyamg.gallery import poisson, load_example
from pyamg.strength import classical_strength_of_connection,symmetric_strength_of_connection

from pyamg.classical import split
from pyamg.classical.classical import ruge_stuben_solver
from pyamg.classical.interpolate import direct_interpolation

from multigrid import mg_solve
import multigrid as mg

# specific to LQCD

class LevelML:
    R = 0
    P = 0
    A = 0
    Q = 0

class SimpleML:
    levels = []

    def __str__(self):
        return "For manual aggregation, printing <ml> is under construction"

from gamma_3_5 import gamma3_application, gamma5_application
# ---------------------------------

# compute tr(A^{-1}) via Hutchinson
def hutchinson(A, function, params):

    mg.level_nr = 0
    mg.coarsest_iters = 0
    mg.coarsest_iters_tot = 0
    mg.coarsest_iters_avg = 0
    mg.nr_calls = 0
    mg.coarsest_lev_iters[0] = 0

    max_nr_levels = params['max_nr_levels']
    # function params
    function_name = params['function_params']['function_name']
    spec_name = params['function_params']['spec_name']
    function_tol = params['function_params']['tol']
    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']
    use_Q = params['use_Q']

    # size of the problem
    N = A.shape[0]
    function_tol = 1e-6

    if use_Q:
        print("Constructing sparse Q ...")
        Q = A.copy()
        mat_size = int(Q.shape[0]/2)
        Q[mat_size:,:] = -Q[mat_size:,:]
        print("... done")
        
    nr_deflat_vctrs = params['nr_deflat_vctrs']
    
    from get_svd import get_svd
    (Sx, Vx,Ux) = get_svd(A, params, use_Q)


    # tr( A ) = tr(U1H*U1* S1 ) + tr( A *(I-U1*U1H))
    # tr( exp(-L) ) = tr(U1H*U1* exp(-S1) ) + tr( exp(-L) *(I-U1*U1H))
    # tr( exp(-L) *(I-U1*U1H)) = (1/n)sum^{n} ( ziH * EXPOFIT( -L,(I-U1*U1H)*zi ) )

    print('\n compute the deflated (smallest) trace part')
    if nr_deflat_vctrs>0:
        start = time.time()
        # compute low-rank part of deflation
        if not function_name=="exponential":
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * np.linalg.inv(Sx)
        else:
            small_A = np.dot(Vx.transpose().conjugate(),Ux) * expm(-Sx)
        tr1 = np.trace(small_A)
        end = time.time()
        print("\nTime to compute the small-matrix contribution in Deflated Hutchinson : "+str(end-start))
    else:
        tr1 = 0.0

    if function_name=="exponential":
        #print("Sending A as dense ...")
        function.putvalue('Ax',A.todense())
        #print("... done")


    from rough_trace_deflat import rough_trace_deflat
    (rough_trace, rough_trace_tol) = rough_trace_deflat(N,A,function_tol,function_name,spec_name,params,function,use_Q,trace_tol)
    print("\n rough trace : "+str(rough_trace), "rough tol : "+str(rough_trace_tol) )
    print("")

    from stoch_est_deflat import stoch_est_deflat 
    (ests_avg, ests_dev, i,function_iters) = stoch_est_deflat(A,N,trace_max_nr_ests, Vx,use_Q,function,function_name,spec_name,nr_deflat_vctrs,params,rough_trace_tol,rough_trace)

    # then, set a rough tolerance
    #rough_solver_tol = rough_trace_tol*lambda_min/N
    #rough_solver_tol = abs(rough_trace_tol/N)

    result = dict()
    #print(tr1)
    result['trace'] = ests_avg+tr1
    result['std_dev'] = ests_dev
    result['nr_ests'] = i
    result['function_iters'] = function_iters
    if function_name=="inverse" and  spec_name=='mg':

        if not params['problem_name']=='schwinger':
            result['total_complexity'] = flopsV(len(function.levels), function.levels, 0)*function_iters
        else:
            result['total_complexity'] = flopsV_manual(len(mg.ml.levels), mg.ml.levels, 0)*function_iters
            result['total_complexity'] += mg.ml.levels[len(mg.ml.levels)-1].A.nnz * mg.coarsest_lev_iters[0]

        # add work due to deflation
        # FIXME : the (harcoded) factor of 3 in the following line is due to non-sparse memory accesses
        result['total_complexity'] += result['nr_ests']*(2*N*nr_deflat_vctrs)/3.0
        #print(result['nr_ests']*(2*N*nr_deflat_vctrs))

    return result
# ---------------------------------



# compute tr(A^{-1}) via MLMC
def mlmc(A, function, params):

    # function params
    function_name = params['function_params']['function_name']
    spec_name = params['function_params']['spec_name']
    function_tol = params['function_params']['tol']
    lambda_min = params['function_params']['lambda_min']
    # trace params
    trace_tol = params['tol']
    trace_max_nr_ests = params['max_nr_ests']
    trace_ml_constr = params['multilevel_construction']
    use_Q = params['use_Q']
    # size of the problem
    N = A.shape[0]  

    max_nr_levels = params['max_nr_levels']
    print(' step 1: generate operators----\n')  
    from get_ml_ops import get_ml_ops
    (mg.ml,t_ml) = get_ml_ops(trace_ml_constr,A,max_nr_levels,params)
    print(' step 1 : done ----')  

    # the actual number of levels
    nr_levels = len(mg.ml.levels)
    mg.total_levels = nr_levels

    for i in range(nr_levels):
        mg.coarsest_lev_iters[i] = 0

    from get_ml_solver import get_ml_solver
    ml_solvers = get_ml_solver(A,nr_levels,mg,max_nr_levels,function_name,params)

    function_tol = 1e-6
    mlmc_levels = nr_levels

    if function_name=="exponential":
        #print("Sending A as dense ...")
        function.putvalue('Ax',A.todense())
        #print("... done")

    OPERATORS = []
    for i in range(nr_levels):
        mg.coarsest_lev_iters[i] = 0
        OPERATORS.append(mg.ml.levels[i].A)

    print("\nConstruction of P and A at all levels (from finest level) ...\n")
#    from get_work_v import get_work_v
    op_levels = len(mg.ml.levels)
#    mlmc_levels = op_levels-2

    print(' step 2: compute the work_v  -----\n')  
    #    (work_v,work_hats,PHATS,RHATS) = get_work_v(mg, nr_levels,OPERATORS)
    from get_find import find
    from get_work_v import get_work_v   
    (work_v,work_hats,PHATS,RHATS) =  get_work_v(mg, nr_levels,params)
    print(' work v = ', work_v*1e-6)
    print(' work hats = ', work_hats*1e-6)
    print(' step 2: done -----\n')  

    epsilon = trace_tol
    print('compute rough trace ........\n')    

    from rough_trace import rough_trace 
#    for level in range(mlmc_levels-1):
#        print('level = ', level)
    nr_rough_iters = 5
    trace_est5, rough_tol = rough_trace(epsilon,N,A,function_tol,function_name,spec_name,params,ml_solvers,1, nr_rough_iters, use_Q)

    print(' .......... done\n')    
    print('rough_trace = ', trace_est5, 'rough tol = ', rough_tol)


    print("")

    # FIXME : this has to be generalized slightly to include stochastic coarsest level
    if nr_levels<3 : raise Exception("Number of levels restricted to >2 for now ...")
    if nr_levels==3:
        tol_fraction0 = 0.5
        tol_fraction1 = 0.5
    else:
        tol_fraction0 = 1.0/3.0
        tol_fraction1 = 1.0/3.0

    cummP = sp.sparse.identity(N,dtype=A.dtype)
    cummR = sp.sparse.identity(N,dtype=A.dtype)
    cummP = csr_matrix(cummP)
    cummR = csr_matrix(cummR)

    start = time.time()
    mg.coarsest_lev_iters[0] = 0

    if not (function_name=="exponential"):
        # coarsest-level inverse
        Acc = mg.ml.levels[nr_levels-1].A
        Ncc = Acc.shape[0]
        np_Acc = Acc.todense()
        np_Acc_inv = np.linalg.inv(np_Acc)
        np_Acc_fnctn = np_Acc_inv[:,:]
    else:
        np_Acc_fnctn = expm( -mg.ml.levels[nr_levels-1].A )

    work_level = np.zeros(nr_levels)
#    var_est = np.zeros((nr_levels - 1,trace_max_nr_ests))
    no_ests_l = np.zeros(nr_levels)

#    var_est = np.zeros(nr_levels-1)
#================================================
    from mg_solver import mg_solver
    from trace_dir_mlmc_opt_PP import trace_dir_mlmc_opt_PP
#    print(' step 3: compute the small trace directly and its work .....\n')  


    samples_l1 = []
    samples_l2 = []
    samples_l3 = []

    iters_l1 = []
    iters_l2 = []
    iters_l3 = []
    
    variance_l1 = []
    variance_l2 = []
    variance_l3 = []
    
    work_dir_l1 = []    
    work_dir_l2 = []    
    work_dir_l3 = []    

    work_stoch_l1 = []    
    work_stoch_l2 = []    
    work_stoch_l3 = []    

    work_total_l1 = []    
    work_total_l2 = []    
    work_total_l3 = []    

    work_total_sum = []    


    print('compute the mlmc trace ----\n')
    start = time.time()
    mg.coarsest_lev_iters[0] = 0
    op_levels = len(mg.ml.levels)
    mlmc_levels = op_levels

    max_estimates = 1000000
    print(' mlmc_levels = ', mlmc_levels)  

    level_solver_tol = 1e-6
    counter = 0

    kk = 2       # number of power iterations 
    mlmc_levels = nr_levels
#    df = [14,55,30] #101
    df = [3,50,26] #101

    for dx in range(1): 

        estimates = np.zeros((mlmc_levels - 1,max_estimates)).astype(complex)
        estimates_sq = np.zeros((mlmc_levels - 1,max_estimates))
        var_est = np.zeros((mlmc_levels - 1,max_estimates)).astype(complex)
        no_estimates = np.zeros(mlmc_levels, dtype=A.dtype)   
        nr_ests =  np.zeros(mlmc_levels, dtype=A.dtype) 
        nr_iters = np.zeros(mlmc_levels, dtype=A.dtype)
        variances = np.zeros(mlmc_levels).astype(complex)
        variances_x = np.zeros(mlmc_levels)
        tolerances = np.zeros((mlmc_levels))
        std_dev = np.zeros((mlmc_levels)).astype(complex)
        tolerances1 = np.zeros((mlmc_levels - 1))
        sums = np.zeros((mlmc_levels)).astype(complex)    
        trace_est = np.zeros(mlmc_levels).astype(complex)
        ests_dev = np.zeros(mlmc_levels).astype(complex)
        ests_avg = np.zeros(mlmc_levels).astype(complex)
        ests_avg_ests = np.zeros((mlmc_levels,max_estimates)).astype(complex)
        rms_deviations = np.zeros(mlmc_levels).astype(complex)
        iterations = np.zeros(mlmc_levels)
        work_level = np.zeros(mlmc_levels)
        work_stoch = np.zeros(mlmc_levels)
        work_work_stochx = np.zeros(mlmc_levels)
        average_work = np.zeros((mlmc_levels))
        no_estimates_th = np.zeros(mlmc_levels)
#        vars_est = np.zeros((mlmc_levels,max_estimates)).astype(complex)
#        aver_var = np.zeros(mlmc_levels).astype(complex)
        active = np.ones((mlmc_levels-1))#, dtype=np.int64)

        nr = dx
        nr_defl = [3,50,26]  # nr of deflation vectors

        work_setup = np.zeros(mlmc_levels)
        trace_dir = np.zeros(mlmc_levels).astype(complex)
        nr_rough_iters = 5
        V = []
        d =  np.zeros(mlmc_levels) #[]

        for level in range(nr_levels-1):
            print('\n start setup phase for l = ',level ,'d = ',nr_defl[level])    
            (trace_dirx, work_setupx,Vx) =  trace_dir_mlmc_opt_PP(mg,epsilon,OPERATORS,PHATS,RHATS,N,nr_defl[level],kk, level, nr_levels,work_v,ml_solvers,function_name,spec_name,params,use_Q, function,function_tol) 
#trace_dir_mlmc_opt_PP(mg,epsilon,OPERATORS,PHATS,RHATS,N,dx,nr_levels,work_v,ml_solvers,function_name,spec_name,params,use_Q, function,function_tol)

            V.append(Vx)
            work_setup[level] = work_setupx # .append(work_setupx)
            trace_dir[level] = trace_dirx #.append(trace_dirx)
            d[level] = nr_defl[level] #.append(nr_defl)
        print(' ........ done')    
        j = -1
        ests = np.zeros(max_estimates, dtype=A.dtype)

        print('\n start stochastic phase ....')    
        while (np.sum(active) > 0):   
        #active_levels = np.nonzero(active)[0]
            active_levels = np.flatnonzero(( active)) #find(active)
            j+=1       

            for l in range(0,len(active_levels)):    #(mlmc_levels-1):#  active_levels : #range(0,np.size(active_levels)): #
                level = active_levels[l]
                no_estimates[level] += 1  # no_estimates[level] + 1
                m = no_estimates[level].astype(int)
                N = A.shape[0]
                i = level
                if i==0 : tol_fctr = sqrt(tol_fraction0)
                elif i==1 : tol_fctr = sqrt(tol_fraction1)
        # e.g. sqrt(0.45), sqrt(0.45), sqrt(0.1*(1.0/(nl-2)))
                else :
                    if params['coarsest_level_directly']==True:
                        tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-3)
                    else:
                        tol_fctr = sqrt(1.0-tol_fraction0-tol_fraction1)/sqrt(nr_levels-2)

#            level_trace_tol  = abs(trace_tol*rough_trace*tol_fctr)
        # fine and coarse matrices
                Af = mg.ml.levels[i].A
                Ac =mg. ml.levels[i+1].A
                n_f = Af.shape[0]
                n_c = Ac.shape[0]
        # P and R
                R = mg.ml.levels[i].R
                P = mg.ml.levels[i].P
#            print("Computing for level "+str(i)+"...")
                ests = np.zeros(trace_max_nr_ests, dtype=Af.dtype)
                ests_dev_x = np.zeros(trace_max_nr_ests, dtype=Af.dtype)
            # generate a Rademacher vector
#            x0 = np.random.randint(2, size=N)
                x0 = np.random.randint(2, size=n_f)
                x0 *= 2
                x0 -= 1
                x1 = x0 - np.dot(V[i],np.dot(V[i].transpose().conjugate(),x0))
                x1 = x1.astype(Af.dtype)
                x = RHATS[i]*PHATS[i]*x1
                xc = R*x            
#            x0 = x0 - np.dot(V[level],np.dot(V[level].transpose().conjugate(),x0))
#            x0 = x0.astype(A.dtype)
#            x = RHATS[i]*x0

                (z,y, estimate, num_iters1, num_iters2,np_Acc_fnctn) = mg_solver(i,nr_levels,mg,x,xc,n_f,A,Af,Ac,params,function,function_name,spec_name,ml_solvers,use_Q)

#                nr_iters[i] += num_iters1
#                nr_iters[i+1] += num_iters2

                iters_f = num_iters1
                iters_c = num_iters2
                iterations[level] += iters_f			
                iterations[level + 1] += iters_c           
                work_stoch[level] += iters_f*work_v[level]+iters_c*work_v[level+1]+2*N*d[level] # + work_hats[level]
                average_work[level] = work_stoch[level] / (j+1)   
                e1 = np.vdot(x1,z) #np.vdot(x0,PHATS[i]*z)
#            cummPh = PHATS[i]*P
                e2 = np.vdot(x1,P*y) #np.vdot(x0,PHATS[i+1]*y)
                ests[j] = e1-e2 # np.linalg.norm(e1-e2) #
                estimate = ests[j]
                estimates[level,j] = ests[j] 
                sums[level] += estimate

                ests_avg = np.sum(estimates[level,0:(j+1)])/(j+1)
                ests_avg_ests[level, j] = ests_avg

#                aa = np.abs(estimates[level,0:(j+1)]-ests_avg)
#                bb = np.square(aa)
#                cc = np.sum(bb)
#                ests_dev_xx = sqrt(cc/(j+1))
                ests_dev = sqrt(np.sum(np.square(np.abs(estimates[level,0:(j+1)]-ests_avg)))/(j+1))
                error_est = ests_dev/sqrt(j+1)
                var_est[level,j] = ests_dev*ests_dev 
                variances[level] = ests_dev*ests_dev
                rms_deviations[level] = error_est

#            print("j="+str(j)+ ", variances ="+str(variances[level])+", estimate = "+str(ests[j])+", trace = "+str(ests_avg) )
    
                if j>=5 and rms_deviations[level]<tolerances[level]:
                    active[level] = False
                

                trace_est[level] = ests_avg  #sums[level]/(j)
            rms_deviations_not_active = rms_deviations[ np.flatnonzero(( active == 0))] #  rms_deviations[find(( active == 0))] #
            tol_rest = np.sqrt( np.square(rough_tol)-sum(np.square(rms_deviations_not_active)))
            actives = find(active)
            CV = np.multiply(average_work[actives],variances[actives])
            sqrt_CV = np.sqrt(CV) 
            sum_sqrt_CV = sum(sqrt_CV)
            mu = 1 /tol_rest * np.sqrt(sum_sqrt_CV)
            tolerances[actives] =  abs((1 / mu) * np.sqrt(sqrt_CV))
#            print("j="+str(j)+", err_est ="+str(error_est)+",  tol= "+str(tolerances[actives]))
            print('.',end='',flush=True)

            no_estimates_th[actives] = (mu**2) * np.sqrt(variances[actives] / average_work[actives])


        print('......done \n')    
        
        AXX =mg.ml.levels[mlmc_levels-1].A  
        n_L = AXX.shape[0] #Af[mlmc_levels-1].shape[0]   
#        print('size_n_L', n_L) 
#    work_level[mlmc_levels-1] = 2*n_L**3 + n_L*PHATS[mlmc_levels-1].nnz   
        work_stoch[mlmc_levels-1] = 2*n_L**3 + n_L*PHATS[mlmc_levels-1].nnz
 #       zz = (2*n_L**3 + n_L*PHATS[mlmc_levels-1].nnz )*1e-6
 #       zz1 = (2*n_L**3)*1e-6
  #      print('coarsest work', zz, 'coarsest work_1', zz1) 
    
        no_estimates[mlmc_levels-1] = 1
#    iterations[mlmc_levels-1] = 1
#        print("\ncompute the coarsest trace ----\n")      
        Acc = mg.ml.levels[mlmc_levels-1].A
#    Acc_inv = np.linalg.inv(Acc)
        Ncc = Acc.shape[0]
        np_Acc = Acc.todense()
        np_Acc_inv = np.linalg.inv(np_Acc)
        np_Acc_fnctn = np_Acc_inv[:,:]
        RP = RHATS[mlmc_levels-1]*PHATS[mlmc_levels-1]
        crst_mat = np_Acc_fnctn*RP
#        nr_ests[nr_levels-1] = 0
        std_dev[nr_levels-1] = 0
        tolerances[nr_levels-1] = 0.0
#    std_dev.append(0.0)
        variances[nr_levels-1] = 0.0
#        error_estimate[nr_levels-1] = 0
#    trace_coarsest = np.trace(Acc_inv* (RHATS[mlmc_levels-1]*PHATS[mlmc_levels-1]) )    
        trace_coarsest = np.trace(crst_mat) 
        trace_est[mlmc_levels-1] = trace_coarsest  
        total_trace_est = sum(trace_est)
        total_work = work_setup + work_stoch*1e-6
    
        print("\nNice print for the data ---------------------\n")           
        print('-- tolerances ', tolerances)   
        print("\n---------------------------\n")           
#    print('- matrix type    : ', problem_name)
#    print(" -- matrix : "+matrix_name)
        print(" -- matrix size : "+str(A.shape[0])+"x"+str(A.shape[1]))
        print(" -- solver is  = ", spec_name)
        print(" -- levels for mlmc = ", nr_levels)
#    print(" -- rough trace = ",rough_trace)   
        print(" -- rel tol is", trace_tol, ", we use (abs.) tol =",rough_tol)
        print(" -- tr(A^{-1}) = ", total_trace_est+sum(trace_dir))
#    print(" -- trace dir  = ", sum(trace_dir))
        print(' -- work: setup phase   : ', sum(work_setup), 'MFLOPS' )
        print(' -- work: stoch phase   : ', sum(work_stoch)*1e-6, 'MFLOPS' )
        print(' -- work total          : ', (sum(total_work)), 'MFLOPS' )
       

        print("{:<8} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} ".format("level",'d', 'k' ,"nr_ests", "nr_iters", "trace_est", "trace_dir","std_dev", "variances", "work_stoch","work_setup","work_level", "tolerances","RMSD"))

        for i in range(nr_levels):
            print("{:<8} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} ".format(i,d[i], kk,  no_estimates[i].real,  iterations[i], str(trace_est[i])[:9], str(trace_dir[i])[:9], str(sqrt(variances[i]))[:6], str(variances[i])[:9], str(work_stoch[i]*1e-6)[:6], str(work_setup[i])[:6],str(total_work[i])[:6],str(tolerances[i])[:6], str(rms_deviations[i])[:8]))

            nr_ests[i] = no_estimates[i]
            nr_iters[i] = iterations[i]


        samples_l1.append(nr_ests[0])
        samples_l2.append(nr_ests[1])
        samples_l3.append(nr_ests[2])

        iters_l1.append(nr_iters[0]) 
        iters_l2.append(nr_iters[1]) 
        iters_l3.append(nr_iters[2]) 
        
        variance_l1.append(variances[0])
        variance_l2.append(variances[1])
        variance_l3.append(variances[2])

        work_dir_l1.append(work_setup[0])    
        work_dir_l2.append(work_setup[1]) 
        work_dir_l3.append(work_setup[2]) 

        work_stoch_l1.append(work_level[0]) 
        work_stoch_l2.append(work_level[1]) 
        work_stoch_l3.append(work_level[2])    

        work_total_l1.append(work_level[0] + work_setup[0])
        work_total_l2.append(work_level[1] + work_setup[1])
        work_total_l3.append(work_level[2] + work_setup[2])
        
        work_total_sum.append(total_work)


    from scipy.io import savemat
    mdict =  {"samples_l1": samples_l1, "samples_l2":samples_l2, "samples_l3": samples_l3, "iters_l1": iters_l1, "iters_l2": iters_l2, "iters_l3": iters_l3, "variance_l1": variance_l1, "variance_l2": variance_l2, "variance_l3":variance_l3, "work_dir_l1": work_dir_l1, "work_dir_l2": work_dir_l2, "work_dir_l3": work_dir_l3, "work_stoch_l1": work_stoch_l1, "work_stoch_l2": work_stoch_l2, "work_stoch_l3": work_stoch_l3, "work_total_l1": work_total_l1, "work_total_l2": work_total_l2, "work_total_l3": work_total_l3,"work_total_sum": work_total_sum}

#    savemat("optimal_mlmcpp_k_"+str(kk)+"_ds_0to"+str(df)+".mat", mdict)
    print("......done ")

    exit(0)

    return output_params
    

