import numpy as np
import scipy as sp
from function import function_sparse
from math import sqrt, pow
import pyamg
from pyamg.aggregation.adaptive import adaptive_sa_solver
from aggregation import manual_aggregation
from scipy.sparse import csr_matrix


def get_ml_solver(A,nr_levels,mg,max_nr_levels,function_name,params):

   # np.random.seed(51234)    
   
    if nr_levels<2:
        raise Exception("Use three or more levels.")

    for i in range(nr_levels):
        print("size(A"+str(i)+") = "+str(mg.ml.levels[i].A.shape[0])+"x"+str(mg.ml.levels[i].A.shape[1]))
    for i in range(nr_levels-1):
        print("size(P"+str(i)+") = "+str(mg.ml.levels[i].P.shape[0])+"x"+str(mg.ml.levels[i].P.shape[1]))


    ml_solvers = []
    if not (function_name=="exponential"):
        if not params['problem_name']=='schwinger':
            print("\nCreating solver with PyAMG for each level ...")
#            ml_solvers = list()
            for i in range(nr_levels-1):
                [mlx, work] = adaptive_sa_solver(mg.ml.levels[i].A, num_candidates=2, candidate_iters=5, improvement_iters=8,
                                                 strength='symmetric', aggregate='standard', max_levels=9)
                ml_solvers.append(mlx)
            print("... done")

    return(ml_solvers)


