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
#import single_stoch_diff
#from single_stoch_diff import single_stoch_diff


def get_ml_ops(trace_ml_constr,A,max_nr_levels,params):

    print("\nConstruction of P and A at all levels (from finest level) ...")

    start = time.time()
    if trace_ml_constr=='pyamg':
        if params['aggregation_type']=='SA':
            mg.ml = pyamg.smoothed_aggregation_solver( A,max_levels=max_nr_levels )
        elif params['aggregation_type']=='ASA':
            [mg.ml, work] = adaptive_sa_solver(A, num_candidates=1, candidate_iters=2, improvement_iters=8,
                                            strength='symmetric', aggregate='standard', max_levels=max_nr_levels)
        else:
            raise Exception("Aggregation type not specified for PyAMG")
        #[ml, work] = adaptive_sa_solver(A, num_candidates=5, improvement_iters=5)

    elif trace_ml_constr=='direct_interpolation':

        mg.ml = SimpleML()
        # appending level 0
        mg.ml.levels.append(LevelML())
        mg.ml.levels[0].A = A.copy()

        for i in range(max_nr_levels-1):

            #print(i)

            print(mg.ml.levels[i].A)

            S = symmetric_strength_of_connection(mg.ml.levels[i].A)
            splitting = split.RS(S)

            mg.ml.levels[i].P = 1.0 * direct_interpolation(mg.ml.levels[i].A, S, splitting)
            mg.ml.levels[i].R = 1.0 * mg.ml.levels[i].P.transpose().conjugate().copy()

            print(mg.ml.levels[i].P)
            print(mg.ml.levels[i].P.count_nonzero())
            print(mg.ml.levels[i].P * mg.ml.levels[i].P.transpose().conjugate())

            mg.ml.levels.append(LevelML())
            mg.ml.levels[i+1].A = csr_matrix( mg.ml.levels[i].R * mg.ml.levels[i].A * mg.ml.levels[i].P )

    # specific to Schwinger
    elif trace_ml_constr=='manual_aggregation':

        # TODO : get <aggr_size> from input params
        # TODO : get <dof_size> from input params

        aggrs = params['aggrs']
        dof = params['dof']

        # 128 ---> 64 ---> 8 ---> 2

        mg.ml = manual_aggregation(A, dof=dof, aggrs=aggrs, max_levels=max_nr_levels, dim=2, acc_eigvs=params['accuracy_eigvs'], sys_type=params['problem_name'])

    # specific to LQCD
    elif trace_ml_constr=='from_files':

        import scipy.io as sio

        mg.ml = SimpleML()

        # load A at each level
        for i in range(max_nr_levels):
            mg.ml.levels.append(LevelML())
            if i==0:
                mat_contents = sio.loadmat('LQCD_A'+str(i+1)+'.mat')
                Axx = mat_contents['A'+str(i+1)]
                mg.ml.levels[i].A = Axx.copy()

        # load P at each level
        for i in range(max_nr_levels-1):
            mat_contents = sio.loadmat('LQCD_P'+str(i+1)+'.mat')
            Pxx = mat_contents['P'+str(i+1)]
            mg.ml.levels[i].P = Pxx.copy()
            # construct R from P
            Rxx = Pxx.copy()
            Rxx = Rxx.conjugate()
            Rxx = Rxx.transpose()
            mg.ml.levels[i].R = Rxx.copy()

        # build the other A's
        for i in range(1,max_nr_levels):
            mg.ml.levels[i].A = mg.ml.levels[i-1].R*mg.ml.levels[i-1].A*mg.ml.levels[i-1].P

    else:
        raise Exception("The specified <trace_multilevel_constructor> does not exist.")
    end = time.time()
    t_ml = end
    print("... done")
    print("Elapsed time to compute the multigrid hierarchy = "+str(end-start))
    print("IMPORTANT : this ML hierarchy was computed with 1 core i.e. elapsed time = "+str(end-start)+" cpu seconds")

    print("\nMultilevel information:")
    print(mg.ml)

    return(mg.ml,t_ml)

