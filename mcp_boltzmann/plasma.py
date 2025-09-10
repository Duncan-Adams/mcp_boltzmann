# Calculation of photon plasma mass
import numpy as np

from scipy.integrate import quad
from mcp_boltzmann.distributions import n_gam

def mgamma_integrand(z, x):
    return np.sqrt(z**2 - x**2)/(np.exp(z)+1)

def mgamma_int(x):
    return quad(mgamma_integrand, x, np.inf, args=(x), epsabs=1e-12, epsrel=1e-12)[0]

@np.vectorize
def m_gam_2(T_sm):
    #squared photon plasma mass as a function of plasma temperature, ignoring hadronic contributions below the QCD phase transition
    alpha = 1.0/137.0
    LQCD = 200 #temperature of QCD phase transition

    x_e = 0.511/T_sm
    x_mu = 105/T_sm
    x_tau = 1776/T_sm

    x_u = 2.2/T_sm
    x_d = 4.7/T_sm
    x_s = 95.0/T_sm
    x_c = 1270.0/T_sm

    q_u = 2/3
    q_d = -1/3
    q_s = -1/3
    q_c = 2/3
    
    #summing over spin states only gives a factor of 2, as the derivation of A.8 in 2206.13530 (which is really from a textbook) 
    #assumes that f_p is the sum of particle anti particle pairs
    qed = 2*(mgamma_int(x_e) + mgamma_int(x_mu) + mgamma_int(x_tau))
    qcd = 3*2*(q_u**2*mgamma_int(x_u) + q_d**2*mgamma_int(x_d) + q_s**2*mgamma_int(x_s) + q_c**2*mgamma_int(x_c))

    return (4*alpha/np.pi)*T_sm**2*(qed + qcd*np.heaviside(T_sm - LQCD, 0))

#plasmon decay collision integral
def C_plasmon(T_sm, T_ds, m_mcp, Q_mcp):
    #it looks like 2206.13530 didnt include the factor of 2 in the definition of n_gamma, otherwise they have a spurious factor
    #of 2 compared to vogel and redondo
    alpha = 1/137
    mg2 = m_gam_2(T_sm)
    
    arg_sqrt = (1 - 4*m_mcp**2/mg2)
    sq = np.sqrt(arg_sqrt*np.heaviside(arg_sqrt, 0))

    return (alpha*Q_mcp**2/3)*(mg2 + 2*m_mcp**2)*sq*(n_gam(T_sm) - n_gam(T_ds))
