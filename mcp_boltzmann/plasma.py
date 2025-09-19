# Calculation of photon plasma mass
import numpy as np

from scipy.integrate import quad
from mcp_boltzmann.distributions import n_gam, n_BE


#fundamental constants
alpha = 1.0/137.0
s2_theta_w = 0.22339 

tan2_theta_w = s2_theta_w/(1 - s2_theta_w)

LQCD = 200 #temperature of QCD phase transition
T_EW = 160*1e3 #temperature of electroweak phase transition

m_e = 0.511
m_mu = 105.0
m_tau = 1776.0

m_u = 2.2
m_d = 4.7
m_s = 95.0
m_c = 1270.0

q_u = 2/3
q_d = -1/3
q_s = -1/3
q_c = 2/3

M_Z = 91188.0 #MeV

def mgamma_integrand(z, x):
    return np.sqrt(z**2 - x**2)/(np.exp(z)+1)

def mgamma_int(x):
    return quad(mgamma_integrand, x, np.inf, args=(x), epsabs=1e-12, epsrel=1e-12)[0]

@np.vectorize
def m_gam_2(T_sm):
    #squared photon plasma mass as a function of plasma temperature, ignoring hadronic contributions below the QCD phase transition
    x_e = m_e/T_sm
    x_mu = m_mu/T_sm
    x_tau = m_tau/T_sm

    x_u = m_u/T_sm
    x_d = m_d/T_sm
    x_s = m_s/T_sm
    x_c = m_c/T_sm

    #summing over spin states only gives a factor of 2, as the derivation of A.8 in 2206.13530 (which is really from a textbook) 
    #assumes that f_p is the sum of particle anti particle pairs
    qed = 2*(mgamma_int(x_e) + mgamma_int(x_mu) + mgamma_int(x_tau))
    qcd = 3*2*(q_u**2*mgamma_int(x_u) + q_d**2*mgamma_int(x_d) + q_s**2*mgamma_int(x_s) + q_c**2*mgamma_int(x_c))

    return (4*alpha/np.pi)*T_sm**2*(qed + qcd*np.heaviside(T_sm - LQCD, 0))

#plasmon decay collision integral
def C_plasmon(T_sm, T_ds, m_mcp, Q_mcp):
    #it looks like 2206.13530 didnt include the factor of 2 in the definition of n_gamma, otherwise they have a spurious factor
    #of 2 compared to vogel and redondo
    mg2 = m_gam_2(T_sm)
    
    arg_sqrt = np.nan_to_num((1 - 4*m_mcp**2/mg2)) #this is fine b/c arg_sqrt will nan wehn mgam=0, but this will cause the whole rate to evaluate to 0, which it should at mgam=0
    sq = np.sqrt(arg_sqrt*np.heaviside(arg_sqrt, 0))

    return ((1/3)*alpha*Q_mcp**2)*(mg2 + 2*m_mcp**2)*sq*(n_gam(T_sm) - n_gam(T_ds))
    

def Gamma_Z_xx(m_mcp, Q_mcp):
    #we probably dont care about running of alpha even though its O(10%) larger at M_Z
    
    pref = Q_mcp**2*alpha*tan2_theta_w/3
    arg_sqrt = np.nan_to_num((1 - 4*m_mcp**2/M_Z**2))
    sq = np.sqrt(arg_sqrt*np.heaviside(arg_sqrt, 0))
    
    return pref*M_Z*sq*(1+ (2*m_mcp**2/M_Z**2))
    

def C_Z_decay(T_sm, T_ds, m_mcp, Q_mcp):
    return Gamma_Z_xx(m_mcp, Q_mcp)*M_Z*(n_BE(T_sm, np.zeros_like(T_sm), M_Z, g=3.0) - n_BE(T_ds, np.zeros_like(T_ds), M_Z, g=3.0))*np.heaviside(T_EW - T_sm, 0)
