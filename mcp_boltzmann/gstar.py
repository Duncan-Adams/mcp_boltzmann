import os
import pdg

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad

#defintions of the various integrals used to compute g_star values for different thermodynamic densities
def g_star_n_integrand_fd(u, z):
    return u*np.sqrt(u**2-z**2)/(np.exp(u) + 1)
    
def g_star_E_integrand_fd(u, z):
    return u**2*np.sqrt(u**2-z**2)/(np.exp(u) + 1)

def g_star_p_integrand_fd(u, z):
    return (u**2 - z**2)**(1.5)/(np.exp(u) + 1)
    
def g_star_n_integrand_be(u, z):
    return u*np.sqrt(u**2-z**2)/(np.exp(u) - 1)
    
def g_star_E_integrand_be(u, z):
    return u**2*np.sqrt(u**2-z**2)/(np.exp(u) - 1)

def g_star_p_integrand_be(u, z):
    return (u**2 - z**2)**(1.5)/(np.exp(u) - 1)

def g_star_n_fd(T, m, g):
    z = m/T
    pref = (15*g)/(np.pi**4)
    res = quad(g_star_n_integrand_fd, z, np.inf, args=(z), limit=200)[0]
    
    return pref*res
    
def g_star_E_fd(T, m, g):
    z = m/T
    pref = (15*g)/(np.pi**4)
    res = quad(g_star_E_integrand_fd, z, np.inf, args=(z), limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    
    return pref*res

def g_star_P_fd(T, m, g):
    z = m/T
    pref = (15*g)/(np.pi**4)
    res = quad(g_star_p_integrand_fd, z, np.inf, args=(z), limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    
    return pref*res
    
def g_star_E_be(T, m, g):
    z = m/T
    pref = (15*g)/(np.pi**4)
    res = quad(g_star_E_integrand_be, z, np.inf, args=(z), limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    
    return pref*res

def g_star_P_be(T, m, g):
    z = m/T
    pref = (15*g)/(np.pi**4)
    res = quad(g_star_p_integrand_be, z, np.inf, args=(z), limit=400, epsabs=1e-12, epsrel=1e-12)[0]
    
    return pref*res
########################################################################################################################

#compute nuetrino contribitions to g_star during & after electron position annihilation, so we can subtract out neutrinos from the EM plasma
@np.vectorize
def g_star_E_nu(T):
    T_nd = 3
    m_e = 0.511
    g_star_e_per_nu = 0.875    
    T_nu_over_T = (2+ g_star_E_fd(T, m_e, 4))/5.5 #if we were being very very technical this should be computed via g_star_S, but it g_star_E = g_star_s for T <= 1 MeV
    
    T_nu_over_T_EM = 1.0
    
    if T < T_nd:
        g_star_s_at_dec = 2 + 0.25*(3*g_star_E_fd(T_nd, 0.511, 4) + g_star_P_fd(T_nd, 0.511, 4))
        g_star_s_at_T = 2 + 0.25*(3*g_star_E_fd(T, 0.511, 4) + g_star_P_fd(T, 0.511, 4))
        
        T_nu_over_T_EM = (g_star_s_at_T/g_star_s_at_dec)**(1/3)

    return 2*3.0*g_star_e_per_nu*T_nu_over_T_EM**4
            
    # ~ pref = 3.0*(2*7/8)
    # ~ m_e = 0.511

    # ~ return pref*((4/11)**(4/3) + (1 - (4/11)**(4/3))*(8/(4*7))*g_star_E_fd(T, m_e, 4))

@np.vectorize
def g_star_P_nu(T):
    T_nd = 3
    m_e = 0.511
    g_star_p_per_nu = 0.875
    
    T_nu_over_T_EM = 1.0
    
    if T < T_nd:
        g_star_s_at_dec = 2 + 0.25*(3*g_star_E_fd(T_nd, 0.511, 4) + g_star_P_fd(T_nd, 0.511, 4))
        g_star_s_at_T = 2 + 0.25*(3*g_star_E_fd(T, 0.511, 4) + g_star_P_fd(T, 0.511, 4))
        
        T_nu_over_T_EM = (g_star_s_at_T/g_star_s_at_dec)**(1/3)

    return 2*3.0*g_star_p_per_nu*T_nu_over_T_EM**4
                

#I eventually abandoned using Husdal's results and just computed g_star myself

def _convert_kT_to_MeV(str_kt):
    '''
    Helper function when loading in the data from 1609.04979
    '''
    #when used as a converter for np.genfromtxt it will be fed in as bytes and not a str
    if isinstance(str_kt, bytes):
        str_kt = str_kt.decode('utf-8')
        
    TeV_to_MeV = 1e6
    GeV_to_MeV = 1e3
    MeV_to_MeV = 1
    keV_to_MeV = 1e-3
    eV_to_MeV = 1e-6
    
    numeric = str_kt.split()[0]
    unit = str_kt.split()[1]

    match unit:
        case 'TeV':
            conv = TeV_to_MeV
        case 'GeV':
            conv = GeV_to_MeV
        case 'MeV':
            conv = MeV_to_MeV
        case 'keV':
            conv=keV_to_MeV
        case 'eV':
            conv=eV_to_MeV
        case _:
            print(f'Error: {unit} is not one of TeV, GeV, MeV, keV, or eV (case sensitive!!)')
            return None

    return float(numeric)*conv

#load in the data from 1609.04979
# ~ _base_path = os.path.dirname(os.path.abspath(__file__))
# ~ _gstar_dat_path = os.path.join(_base_path, '../input/gstar_husdal.csv')
def _load_gstar_hudsal(path, include_nuetrinos=True):    
    gstar_dat = np.genfromtxt(_gstar_dat_path, delimiter=',', skip_header=3, converters={0: _convert_kT_to_MeV})
    gstar_dat = list(zip(*gstar_dat))
    
    Temp_in_MeV = np.array(gstar_dat[0])
    g_E = np.array(gstar_dat[4])
    g_P = np.array(gstar_dat[5])

    #subtract out neutrino contributions if you want to model neutrinos as seperate sector in boltzmann equation
    if include_nuetrinos is False:
        g_E = g_E - g_star_E_nu(Temp_in_MeV)
        g_P = g_P - g_star_P_nu(Temp_in_MeV)

    return Temp_in_MeV, g_E, g_P

_base_path = os.path.dirname(os.path.abspath(__file__))
_gstar_dat_path = os.path.join(_base_path, '../input/g_star.npz')

def _load_gstar(path):
    with np.load(path) as f:
        Temp_grid = f['Temp_grid']
        g_star_E_SM = f['g_star_E_EM']
        g_star_P_SM = f['g_star_P_EM']
        
    return Temp_grid, g_star_E_SM, g_star_P_SM 

# ~ _T_grid, _g_E_no_nu, _g_P_no_nu = _load_gstar_hudsal(_gstar_dat_path, include_nuetrinos=False)


_T_grid, _g_E_no_nu, _g_P_no_nu = _load_gstar(_gstar_dat_path)
_g_E_EM_I = UnivariateSpline(_T_grid, _g_E_no_nu, k=3, s=0, ext=3)
_dg_E_EM_I = _g_E_EM_I.derivative()
_g_P_EM_I = UnivariateSpline(_T_grid, _g_P_no_nu, k=3, s=0, ext=3)


def gstar_E_EM(T):
    '''
    compute g* for the pressure of the elctromagnetic plasma
    
    Parameters
    ----------
    T - temperature in MeV
    
    Returns
    -------
    float
        g_star_energy of T
    '''
    return _g_E_EM_I(T)
    
def d_gstar_E_EM_dT(T):
    '''
    compute g* for the pressure of the elctromagnetic plasma
    
    Parameters
    ----------
    T - temperature in MeV
    
    Returns
    -------
    float
        dg_star_energy_dT of T
    '''
    return _dg_E_EM_I(T)

def gstar_P_EM(T):
    '''
    compute g* for the pressure of the electromagnetic plasma
    
    Parameters
    ----------
    T - temperature in MeV
    
    Returns
    -------
    float
        g_star_pressure of T
    '''
    return _g_P_EM_I(T)


def w_EM(T):
    '''
    compute the equation of state, w, for the electromagnetic plasma
    
    Parameters
    ----------
    T - temperature in MeV
    
    Returns
    -------
    float
        equation of state of plasma at T
    '''
    return gstar_P_EM(T)/(3*gstar_E_EM(T))



