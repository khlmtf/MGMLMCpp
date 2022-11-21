import numpy as np
import scipy as sp

# ---------------------------------

def gamma3_application(v):
    v_size = int(v.shape[0]/2)
    v[v_size:] = -v[v_size:]
    return v

def gamma5_application(v,l,dof):

    sz = v.shape[0]
    for i in range(int(sz/dof[l])):
        # negate first half
        for j in range(int(dof[l]/2)):
            v[i*dof[l]+j] = -v[i*dof[l]+j]

    return v

