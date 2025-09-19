import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import kn

from mcp_boltzmann.polylog import Lie2, Lie3, Lie4


z3 = 1.20206 #zeta of 3
LI2 = (np.pi**2)/12
LI3 = 3*z3/4
LI4 = 7*(np.pi**4)/(720)

tm_ratio_threshold = 300

#Tabulated integrals for distribution functions. 
#Load the tabulated integrands, and define interpolators for each of them. 
#Get the directory of distributions.py
import os
thisdirname = os.path.dirname(__file__)
tabdirname = os.path.join(thisdirname, '../input/tabulated_distributions/')
arr = np.load(os.path.join(tabdirname,'n_FD_integrand.npy'))
n_FD_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'n_BE_integrand.npy'))
n_BE_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'rho_FD_integrand.npy'))
rho_FD_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'rho_BE_integrand.npy'))
rho_BE_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'p_FD_integrand.npy'))
p_FD_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'p_BE_integrand.npy'))
p_BE_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'dn_dT_FD_integrand.npy'))
dn_dT_FD_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'dn_dT_BE_integrand.npy'))
dn_dT_BE_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'drho_dT_FD_integrand.npy'))
drho_dT_FD_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

arr = np.load(os.path.join(tabdirname,'drho_dT_BE_integrand.npy'))
drho_dT_BE_integral = interp1d(arr[0],arr[1],kind='cubic',bounds_error=False,fill_value=0)

#########################
# Thermal distributions #
#########################

def f_MB(E, T, mu, mass):
    argexp = np.atleast_1d(-(E-mu)/T)
    res = np.empty_like(argexp)
    res = np.exp(argexp,dtype=np.longdouble)
    return res

def f_FD(E, T, mu, mass):
    argexp = np.atleast_1d((E-mu)/T)
    mask = (argexp > 20)
    res = np.empty_like(argexp)
    res[mask] = f_MB(E, T, mu, mass)[mask]
    res[~mask] = (np.exp(argexp[~mask])+1)**-1

    return res

def f_FD_prime(E, T, mu, mass = 0):
    E = np.atleast_1d(E)
    argexp = E/T
    mask = (argexp > 20)
    res = np.empty_like(argexp)
    res[mask] = f_MB(E, T, 0, mass)[mask] #as E/T -> infinity, f_FD_prime -> e^(-E/T) = f_MB. 
    res[~mask] = 1/(np.exp(argexp[~mask]) + np.exp(-1*argexp[~mask]) + 2)

    return res

def f_BE(E, T, mu, mass):
    argexp = np.atleast_1d((E-mu)/T)
    mask = (argexp > 20)
    res = np.empty_like(argexp)
    res[mask] = f_MB(E, T, mu, mass)[mask]
    res[~mask] = (np.exp(argexp[~mask])-1)**-1

    return res

####################
# Number Densities #
####################

def n_nu(T, mu):
    x = mu/T
    g = 2

    return -g*((T**3)/(np.pi**2))*Lie3(x)
    
def n_gam(T):
    g = 2
    return g*z3*T**3/np.pi**2

def n_nonrel(T, mu, m):
    x = mu/T
    return (m*T/(2*np.pi))**(1.5) * np.exp(x-m/T,dtype=np.longdouble)


def nDM_FD(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 4.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**3/(2*np.pi**2))*np.exp(x[~mask]) * n_FD_integral(mDM/T[~mask])
    return res

def nDM_BE(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 2.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**3/(2*np.pi**2))*np.exp(x[~mask]) * n_BE_integral(mDM/T[~mask])
    return res
    
def n_MB(T, mu, m):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    mask = T < m/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = n_nonrel(T[mask],mu[mask],m)
    res[~mask] = m**2*T[~mask]*np.exp(mu[~mask]/T[~mask])*kn(2, m/T[~mask])/(2*np.pi**2)
    return res
    
def n_BE(T, mu, m, g=2):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    mask = T < m/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = g * n_nonrel(T[mask],mu[mask],m)
    res[~mask] = g * (T[~mask]**3/(2*np.pi**2))*np.exp(x[~mask]) * n_BE_integral(m/T[~mask])
    return res

####################
# Energy Densities #
####################
def rho_nu(T, mu):
    x = mu/T
    g = 2

    return -g*((3*T**4)/(np.pi**2))*Lie4(x)

def rho_gam(T):
    return 2 * np.pi**2/30. * T**4  

def rho_e(T):
    return rhoDM_FD(T,np.zeros_like(T),0.511)

def p_e(T):
    m_e = 0.511
    return p_DM_FD(T,np.zeros_like(T),m_e)

def rhoDM_FD(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 4
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = mDM * gDM * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**4/(2*np.pi**2))*np.exp(x[~mask]) * rho_FD_integral(mDM/T[~mask])
    return res

def rhoDM_BE(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 2.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * mDM * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**4/(2*np.pi**2))*np.exp(x[~mask]) * rho_BE_integral(mDM/T[~mask])
    return res

def p_DM_FD(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 4
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * T[mask] * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**4/(6*np.pi**2))*np.exp(x[~mask]) * p_FD_integral(mDM/T[~mask])
    return res

def p_DM_BE(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 2.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * T[mask] * n_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = gDM * (T[~mask]**4/(6*np.pi**2))*np.exp(x[~mask]) * p_BE_integral(mDM/T[~mask])
    return res
# # Rho tot
# def rho_tot(T_gam, T_nu, mu_nu, T_DM, mu_DM, mDM):
#     return rho_gam(T_gam) + rho_e(T_gam) + 2*rho_nu(T_nu, mu_nu) + 4*rho_nu(T_nu, mu_nu) + rhoDM_BE(T_DM, mu_DM, mDM) - P_int(T_gam) + T_gam*dP_intdT(T_gam)

##############################
# Number Density Derivatives #
##############################

def dn_nudT(T, mu):
    x = mu/T
    g = 2

    return g*((T**2)/(np.pi**2))*(x*Lie2(x) -3*Lie3(x))

def dn_nudmu(T, mu):
    x = mu/T
    g = 2

    return -g*((T**2)/(np.pi**2))*Lie2(x)

def drho_nudT_LO(T, mu):
    g = 1
    P = g*T**3/(60*np.pi**2)
    A = 7*np.pi**4
    B = 405*z3

    x = mu/T

    return P*(A + B*x)

def drho_nudmu_LO(T, mu):
    g = 1
    P = g*(T**3)/(4*np.pi**2)
    A = 9*z3
    B = np.pi**2

    x = mu/T

    return P*(A + B*x)

def dn_nudT_LO(T, mu):
    g = 1
    A = 27*z3/(np.pi**2)

    return (g/12)*(T**2)*(A + 2*(mu/T)) 

def dn_nudmu_LO(T, mu):
    g = 1
    A = 1
    B = 12*np.log(2)/(np.pi**2)

    x = mu/T

    return (g/12)*(T**2)*(A + B*x)

def dn_dT_nonrel(T,mu,m):
    x = mu/T
    return (m/(2*np.pi))**(3/2) * ((m-mu)/np.sqrt(T) + 1.5*np.sqrt(T)) * np.exp(x-m/T,dtype=np.longdouble)

def dnDM_dT_FD(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 4
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * dn_dT_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = -mu[~mask]*nDM_FD(T[~mask],mu[~mask],mDM)/T[~mask]**2 + np.exp(x[~mask])*gDM*(T[~mask]**2/(2*np.pi**2))*dn_dT_FD_integral(mDM/T[~mask])
    return res

def dnDM_dT_BE(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 2.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = dn_dT_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = -mu[~mask]*nDM_BE(T[~mask],mu[~mask],mDM)/T[~mask]**2 + np.exp(x[~mask])*gDM*(T[~mask]**2/(2*np.pi**2))*dn_dT_BE_integral(mDM/T[~mask])
    return res

def dnDM_dmu_FD(T, mu, mDM):
    return nDM_FD(T, mu, mDM)/T

def dnDM_dmu_BE(T, mu, mDM):
    return nDM_BE(T, mu, mDM)/T

##############################
# Energy Density Derivatives #
##############################

def drho_nudT(T, mu):
    x = mu/T
    g = 2

    return g*((3*T**3)/(np.pi**2))*(x*Lie3(x) -4*Lie4(x))

def drho_gamdT(T):
    return 4*rho_gam(T)/T

def drho_edT(T):
    m_e = 0.511
    return drhoDM_dT_FD(T,np.zeros_like(T),m_e)

def drho_nudmu(T, mu):
    x = mu/T
    g = 2

    return -g*((3*T**3)/(np.pi**2))*Lie3(x)

def drhoDM_dmu_FD(T, mu, mDM):
    return rhoDM_FD(T, mu, mDM)/T

def drhoDM_dmu_BE(T, mu, mDM):
    return rhoDM_BE(T, mu, mDM)/T

def drhoDM_dT_FD(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 4
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * mDM * dn_dT_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = -mu[~mask]*rhoDM_FD(T[~mask],mu[~mask],mDM)/T[~mask]**2 + np.exp(x[~mask])*gDM*(T[~mask]**3/(2*np.pi**2))*drho_dT_FD_integral(mDM/T[~mask])
    return res

def drhoDM_dT_BE(T, mu, mDM):
    T = np.atleast_1d(T)
    mu = np.atleast_1d(mu)
    x = mu/T    
    gDM = 2.0
    mask = T < mDM/tm_ratio_threshold
    res = np.empty_like(T)
    res[mask] = gDM * mDM * dn_dT_nonrel(T[mask],mu[mask],mDM)
    res[~mask] = -mu[~mask]*rhoDM_BE(T[~mask],mu[~mask],mDM)/T[~mask]**2 + np.exp(x[~mask])*gDM*(T[~mask]**3/(2*np.pi**2))*drho_dT_BE_integral(mDM/T[~mask])
    return res
