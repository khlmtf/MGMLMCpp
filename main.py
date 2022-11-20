from gateway import G101,G102,G103,G104,G105,G106,G107,G108,G109,G110,G111
#from gateway import G201,G202,G203,G204,G205,G206,G207,G208,G209,G210,G211

import os

# IMPORTANT ---> to run :

# for the exponential function (PYTHON 2):
#		in gateway.py, set : params['function'] = 'exponential'
#		to run : python2 main.py

# for the inverse function (PYTHON 3):
#		in gateway.py, set : params['function'] = 'inverse'
#		to run : python3 main.py



# main section
if __name__=='__main__':

    # functions starting with G1 execute MLMC

    # Schwinger 16^2
    #os.environ['OMP_NUM_THREADS'] = '1'
    G101()
    #os.environ['OMP_NUM_THREADS'] = '24'
    #G201()

    # Schwinger 128^2
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
#    G102()
    #os.environ['OMP_NUM_THREADS'] = '1'
#    os.environ['OMP_NUM_THREADS'] = '1'
#    os.environ['MKL_NUM_THREADS'] = '1'
#    os.environ['OMP_NUM_THREADS'] = '1'
 #   G202()

    # Gauge Laplacian
#    os.environ['OMP_NUM_THREADS'] = '1'
#    G103()
    #os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
#    G203()

    # Laplace 2D
    #os.environ['OMP_NUM_THREADS'] = '1'
    #G104()
    #os.environ['OMP_NUM_THREADS'] = '24'
    #G204()

    # Linear Elasticity
    #G105()
    #G205()

    # diffusion
    #G106()
    #G206()

    # undirected graph
    #G107()
    #G207()

    # LQCD small
    #G108()
    #G208()

    # LQCD large
    #G109()
    #G209()

    # Laplace 3D
    #G110()
    #G210()

    # Estrada Index
    #G111()
    #G211()
