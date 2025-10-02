import os
from functools import partial

import numpy as np
import scipy.special as sps
import vegas

from scipy.integrate import quad
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy.special import kn
from mpmath import fp as fp

from . import polylog as plog
from mcp_boltzmann.kinematics import beta_t

#Constants
MeV = 1
GeV = 1e3*MeV
keV = 1e-6 * GeV
M_Planck = 1.22 * 1e19 * GeV
m_e = 0.511 * keV #huh?
Ql = 1
e = 0.302822

Gamma_rhopipi = 149.1 * MeV
m_rho = 775.26 * MeV
m_pi = 139.570 * MeV

s2_theta_w = 0.22339 
c2_theta_w = 1-s2_theta_w


#G function define in equation C.18 of Adshead, Ralengenkar, & Shelton (2206.13530)
_base_path = os.path.dirname(os.path.abspath(__file__))
_G_function_fermi_dat = np.load(os.path.join(_base_path, '../input/Gfun_fermion.npz'))

_G_fermion_interp = interp1d(
    _G_function_fermi_dat['x_range'], 
    _G_function_fermi_dat['G_fermion_table'], 
    bounds_error=False, 
    fill_value=0
)

def load_ann_rate(path):
    with np.load(path) as rate_file:
        return interp1d(rate_file['Temp_grid'], rate_file['rate'], bounds_error=False, fill_value=0)

def _G_fermi_small_x(x):
    return np.log(2)*(np.pi**2)/(6*x**2)

def G_fermion(x):
    x = np.atleast_1d(x)

    mask_small_x = np.where(x < 1e-2)
    mask_large_x = np.where(x > 20)

    res = np.zeros(len(x))

    res = _G_fermion_interp(x)
    res[mask_small_x] = _G_fermi_small_x(x[mask_small_x])
    res[mask_large_x] = kn(2, x[mask_large_x])

    return res

#asymptotic expansion of modified bessel function of second kind
def kn2_asymptotic(z):
    # ~ zf = z.astype(np.float128) #disable for now, ask jared
    zf = z #temporary hack until Jared has been asked
    return np.sqrt(np.pi/(2 * zf)) * np.exp(-zf) * (1 + (4 * 2**2 - 1)/(8 * zf) + (4*2**2 - 1)*(4*2**2 - 9)/(2 * (8*z)**2))

def M2_llff(s,c2, m_mcp, m_l):#Note VR13 say they neglect the smaller of the 2 masses. 
    #s is the mandelstam variable s, center of mass energy squared, and c2 is cos(theta)^2, theta the scattering angle
    return 4 * e**4 * Ql**2 * (1/s**2) * (s**2 * (1 + c2) + 4 * s * (m_mcp**2 + m_l**2)*(1-c2) + 16 * m_mcp**2 * m_l**2 * c2)

#Cross-section:
def sigma_llff(s, m_mcp, m_f, q_f=-1.0):
    #Kinematics
    Ei = np.sqrt(s)/2
    Ef = Ei
    pi = np.sqrt(Ei**2 - m_f**2)
    pf = np.sqrt(Ef**2 - m_mcp**2)
    return 4 * e**4 * q_f**2 * (2 * np.pi) / ((8 * np.pi)**2 ) * (1/s) * (pf/pi) * (8/3 + (32 * m_mcp**2 * m_f**2) / (3 * s**2) + (16 *(m_mcp**2 + m_f**2)) / (3 * s))

# includes contribution from Z boson mediated annihilation, with the on shell Z piece subtracted off  see eq A.3 of 2206.13530
def sigma_llff_Z_boson(s, m_mcp, m_f, q_f, cv, ca):
    #Kinematics
    M_z = 91.1880*1e3
    Ei = np.sqrt(s)/2
    Ef = Ei
    pi = np.sqrt(Ei**2 - m_f**2)
    pf = np.sqrt(Ef**2 - m_mcp**2)
    
    prefactor = 4 * e**4 * (2 * np.pi) / ((8 * np.pi)**2 )
    photon_med = q_f**2*(4/3)*(2*m_mcp**2 + s)*(2*m_f**2 + s)
    Z_med = -np.heaviside(s - M_z**2, 0)*(s + 2*m_mcp**2)*m_f**2*(cv**2 + 3*ca**2)/(2*c2_theta_w**2)
    #I think only the second term in here is actually interference but w/e
    interference = np.heaviside(s - M_z**2, 0)*(4/3)*(2*m_mcp**2 + s)*(2*m_f**2 + s)*( (cv**2 + ca**2)/(4*c2_theta_w) - cv*q_f/c2_theta_w ) 
    
    return prefactor * (1/s**3) * (pf/pi) * (photon_med + Z_med + interference)

#form factor for pion
def F_pi(s):
    return 1.20 * m_rho**2 / (m_rho**2 - s - 1j * m_rho * Gamma_rhopipi)

#Pion annihilation
def M2_pipiff(s,c2, m_f):
    return 2 * e**4 * np.abs(F_pi(s))**2 * (1/s**2) * (s**2 * (1-c2) + 4 * s * (m_f**2 * c2 - m_pi**2 * (1 - c2)) - 16 * m_f**2 * m_pi**2 * c2)

# ~ def sigma_pipiff(s, m_mcp):
    # ~ #other factors in sigma? Check. 
    # ~ return 2 * e**4 * np.abs(F_pi(s))**2 * (4/3 - (32 * m_mcp**2 * m_pi**2)/(3 * s**2) + (8 * m_mcp**2)/(3 * s) - (16 * m_pi**2)/(3 * s))
    
def sigma_pipi_ff(s, m_mcp):
    #other factors in sigma? Check. 
    Ei = np.sqrt(s)/2
    Ef = Ei
    pi = np.sqrt(Ei**2 - m_pi**2)
    pf = np.sqrt(Ef**2 - m_mcp**2)
    return e**4 * np.abs(F_pi(s))**2 * (pf/pi) * (s - 4*m_pi**2) * (s + 2*m_mcp**2) / (24*np.pi * s**3)
        
def I_integrand(lns,sigma_func, m_f, T):#log-space
    s=np.exp(lns)
    dlns = s
    z = np.sqrt(s)/T
    return s * sigma_func(s) * (s - 4 * m_f**2) * dlns * np.where(z<100,kn(2,z),kn2_asymptotic(z))
    
def I_integrand_fermi(lns, sigma_func, m_f, T):#log-space
    s=np.exp(lns)
    dlns = s
    z = np.sqrt(s)/T
    return s * sigma_func(s) * (s - 4 * m_f**2) * G_fermion(z)*dlns

#Annihilation collision term integral
g1 = 2
g2 = 2
#dont use these because the cross sections are already spin-summed

def Ix(sigma_func, m_mcp, m_f, T):
    prefactor = T /(2**5 * np.pi**4) 
    #At high temperature, the upper bound should be past the peak which is around 10 T^2. 
    #At low temperature, this estimate is actually lower than 4m^2, so we need to explicitly go a bit above 4m^2. The peak is very narrow and close to 4m^2 in this regime. 
    # ~ lns_min = np.log(4*m_f**2) #should this be log(max(4m_f^2, 4m_l^2))?
    mass_thresh = max(4*m_mcp**2, 4*m_f**2)
    lns_min = np.log(mass_thresh)
    lns_max = max(np.log(T**2)+10,np.log(mass_thresh)+1)
    
    integral = quad(
        I_integrand,
        lns_min,
        lns_max,
        args=(sigma_func, m_f, T),
        epsabs=0 #huh?
    )
    
    return prefactor * integral[0]


def Ix_fermi(sigma_func, m_mcp, m_f, T):
    prefactor = T /(32 * np.pi**4) 
    #At high temperature, the upper bound should be past the peak which is around 10 T^2. 
    #At low temperature, this estimate is actually lower than 4m^2, so we need to explicitly go a bit above 4m^2. The peak is very narrow and close to 4m^2 in this regime. 
    # ~ lns_min = np.log(4*m_f**2) #should this be log(max(4m_f^2, 4m_l^2))?
    mass_thresh = max(4*m_mcp**2, 4*m_f**2)
    lns_min = np.log(mass_thresh)
    lns_max = max(np.log(T**2)+10,np.log(mass_thresh)+1)
    
    integral = quad(
        I_integrand_fermi,
        lns_min,
        lns_max,
        args=(sigma_func, m_f, T),
        epsabs=0, #huh?
        limit=200
    )
    
    return prefactor * integral[0]
