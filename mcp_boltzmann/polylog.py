#approximations of polylogarithms on the interval (-1, 1), needed for computation of elastic scattering integrals
import numpy as np

import scipy.special as sps
from mpmath import fp as fp

@np.vectorize
def polylog_vec(s, z):
    return fp.polylog(s, z)

def Li0(x):
    return x/(1-x)
    
def Li1(x):
    return -np.log(1-x)
    
def Li2(x):
    return sps.spence(1-x)

#this is almost an OOM faster that mpmath.fp.polylog(3, x)
def Li3(x):
    return x + x**2/8 + x**3/27 + x**4/64 + x**5/125 + x**6/216

def polylog_fast(n, x):
    '''
    A fast method for computing polylogs in parallel, valid for x in (-1, 1) and n in (0, 1, 2, 3)
    
    n - order of polylogarithm
    x - argument of polylogarithm
    
    returns: (len(n) by len(x)) array of LiN(x))
    '''
    n = np.atleast_1d(n)
    res = np.zeros((len(n), len(x)))
    n0 = np.where(n==0)
    n1 = np.where(n==1)
    n2 = np.where(n==2)
    n3 = np.where(n==3)

    if len(n0[0]) > 0:
        res[n0] = Li0(x)
        
    if len(n1[0]) > 0:
        res[n1] = Li1(x)
        
    if len(n2[0]) > 0:
        res[n2] = Li2(x)       
        
    if len(n3[0]) > 0:
        res[n3] = Li3(x)
        
    return res


def L_n_zeta(a, b, n, zeta):
    #equation C.51 of 2206.13530, used to evaluate elastic scattering integrals
    r = np.arange(0, n + 1, dtype=int)
    return np.sum(
        sps.perm(n, r[:,np.newaxis])*np.power(a, n-r[:,np.newaxis])*(
            -zeta*polylog_fast(r+1, -zeta*np.exp(-a+b))
            +zeta*polylog_fast(r+1, -zeta*np.exp(-a-b))
        ),
        axis=0
    )
