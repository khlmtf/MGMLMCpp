import numpy as np
import scipy as sp
import pyamg
from numpy.linalg import norm as npnorm
from scipy.sparse.linalg import norm
from numpy.linalg import eigh
from scipy.sparse.linalg import svds,eigsh,eigs
import time
import os


def get_svd(A, params, use_Q):

    # compute the SVD (for low-rank part of deflation)
    np.random.seed(65432)
    nr_deflat_vctrs = params['nr_deflat_vctrs']

    if params['accuracy_eigvs'] == 'low':
        tolx = tol=1.0e-1
        ncvx = nr_deflat_vctrs+2
    elif params['accuracy_eigvs'] == 'high':
        tolx = tol=1.0e-6
        ncvx = None
    else:
        raise Exception("<accuracy_eigvs> does not have a possible value.")

    # FIXME : hardcoded value for eigensolving tolerance for now
    tolx = 1.0e-14

    if nr_deflat_vctrs>0:
        print("Computing SVD (finest level) ...")
        start = time.time()
        if use_Q:
            #Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0,ncv=ncvx )

            #Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0 )

            Sy,Ux = eigsh( Q,k=nr_deflat_vctrs,which='SM',tol=tolx )

            Vx = np.copy(Ux)
        else:
            diffA = A-A.getH()
            diffA_norm = norm( diffA,ord='fro' )
            if diffA_norm<1.0e-14:
                #Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0,ncv=ncvx )
                Sy,Ux = eigsh( A,k=nr_deflat_vctrs,which='SM',tol=tolx )
                Vx = np.copy(Ux)
            else:
                if params['problem_name']=='schwinger':
                    # extract eigenpairs of Q
                    print("Constructing sparse Q ...")
                    Q = A.copy()
                    mat_size = int(Q.shape[0]/2)
                    Q[mat_size:,:] = -Q[mat_size:,:]
                    print("... done")
                    print("Eigendecomposing Q ...")
                    Sy,Vx = eigsh( Q,k=nr_deflat_vctrs,which='LM',tol=tolx,sigma=0.0 )
                    sgnS = np.ones(Sy.shape[0])
                    for i in range(Sy.shape[0]): sgnS[i]*=(2.0*float(Sy[i]>0)-1.0)
                    Sy = np.multiply(Sy,sgnS)
                    Ux = np.copy(Vx)
                    for idx,sgn in enumerate(sgnS) : Ux[:,idx] *= sgn
                    mat_size = int(Ux.shape[0]/2)
                    Ux[mat_size:,:] = -Ux[mat_size:,:]
                    print("... done")
                else:
                    Ux,Sy,Vy = svds( A,k=nr_deflat_vctrs,which='SM',tol=tolx )
                    Vx = Vy.transpose().conjugate()
        Sx = np.diag(Sy)
        end = time.time()
        print("... done")
        print("Time to compute singular vectors (or eigenvectors) = "+str(end-start))

        try:
            nr_cores = int(os.getenv('OMP_NUM_THREADS'))
            print("IMPORTANT : this SVD decomposition was computed with "+str(nr_cores)+" cores i.e. elapsed time = "+str((end-start)*nr_cores)+" cpu seconds")
        except TypeError:
            raise Exception("Run : << export OMP_NUM_THREADS=32 >>")
    return (Sx, Vx,Ux)
