
import numpy as np
from matplotlib import pyplot as plt
from py import opt, read, hypergraph
from py.utils import *

import networkx as nx

from scipy.linalg import eigh


def multiway_spectral(A, A0, k):
    
# implementation(?) of Zhang + Newman on multiway spectral partitioning
    
    B = A - A0
    
    n = B.shape[0]
    m = A.sum()/2
    
    
    eigen = eigh(B, eigvals = (n-k, n-1))

    lam = eigen[0]
    U = eigen[1]

    # vertex vectors (eq (10))

    r = np.sqrt(lam)*U 

    ix = np.random.randint(0, n, k)

    # initialize community vectors
    R = r[ix,:]

    # random assignments
    ix = np.random.randint(0, k, n)
    L = np.zeros((n,k))
    L[np.arange(n), ix] = 1
    L_old = L.copy()
    
    ell_old = np.zeros(n)
    
    for j in range(100):
        # # until convergence

        Rr = np.dot(r, R.T) 
        
        # current assignment
        ell = np.argmax(Rr, axis = 1)
        L = np.zeros((n, k))
        L[np.arange(n), ell] = 1
        ell = L.argmax(axis=1)
        R = np.dot(L.T,r)

        obj = (1 / (2*m))*(R**2).sum()
        
        if (ell == ell_old).mean() == 1:
            break
        ell_old = ell.copy()
        
    return(ell, obj)

def find_best_clustering(A, A0,  k, n_reps = 100):
    
    best_Q = 0
    for i in range(n_reps):
        ell, obj = multiway_spectral(A, A0, k)
        Z = one_hot(ell)
        Q = modularity(A, A0, Z)
        if Q > best_Q:
            best_ell = ell
            best_Q = Q
    return(best_ell, best_Q)


def one_hot(ell):
    n = len(ell)
    Z = np.zeros((n,ell.max() + 1))
    Z[np.arange(n), ell] = 1
    return(Z)

def modularity(A, A0, Z):
    m = .5*A.sum()
    return(1/(2*m)*np.trace(np.dot(Z.T, np.dot(A-A0, Z))))


def MI(X, Y, normalize = False, return_joint = False):
    # assumes X and Y are vectors of integer labels of the same length and on the same alphabet space
    n = len(X)
    k = max(X.max(), Y.max())
    joint = np.zeros((k+1, k+1))
    p_x = np.zeros(k+1)
    p_y = np.zeros(k+1)
    
    for i in range(n):
        joint[X[i], Y[i]] += 1
        p_x[X[i]] += 1
        p_y[Y[i]] += 1
    joint = joint / joint.sum()
    
    p_x = p_x/p_x.sum()
    p_y = p_y/p_y.sum()
    
    mat = (joint*np.log(joint / np.outer(p_x, p_y)))
    mat[np.isnan(mat)] = 0
    
    I_XY = mat.sum()
    
    if normalize: 
        H_x = -p_x*np.log(p_x)
        H_x[np.isnan(H_x)] = 0
        H_x = H_x.sum()
        
        H_y = -p_y*np.log(p_y)
        H_y[np.isnan(H_y)] = 0
        H_y = H_y.sum()
    
        NMI = I_XY/(.5*(H_x + H_y))
        out = NMI
    else:
        out = I_XY
    
    if return_joint:
        return(out, joint)
    else:
        return(out)
    
