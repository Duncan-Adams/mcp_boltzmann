import numpy as np
from scipy.integrate import quad

def tabulate_integral(integrand_name, fname):
    x = np.concatenate(([0],np.geomspace(0.05,300,10000))) #Include 0 for relativistic limit. #np.geomspace(0.05,300,10000) #m/T
    res = np.array([quad(integrand_name, x0, np.inf,args=(x0), epsabs=1e-12, epsrel=1e-12)[0] for x0 in x])
    np.save(fname,np.array([x,res]))
    
    
def n_FD_integrand(z,x):
    return (z*np.sqrt(z**2 - (x)**2))/(np.exp(z) + 1)

def n_BE_integrand(z,x):
    return (z*np.sqrt(z**2 - (x)**2))/(np.exp(z) - 1) 

def rho_FD_integrand(z,x):
    return (z**2*np.sqrt(z**2 - (x)**2))/(np.exp(z) + 1) 

def rho_BE_integrand(z,x):
    return (z**2*np.sqrt(z**2 - (x)**2))/(np.exp(z) - 1)

def p_FD_integrand(E, x):
    return (E**2-(x)**2)**1.5/(np.exp(E) + 1.)

def p_BE_integrand(E, x):
    return (E**2-(x)**2)**1.5/(np.exp(E) - 1.)

def dn_dT_FD_integrand(z,x):
    return (0.25*z**2)*np.sqrt(z**2 - (x)**2)/(np.cosh(z/2)**2)

def dn_dT_BE_integrand(z, x):
    return (0.25*z**2)*np.sqrt(z**2 - (x)**2)/(np.sinh(z/2)**2)

def drho_dT_FD_integrand(z,x):
    return (0.25*z**3)*np.sqrt(z**2 - (x)**2)/(np.cosh(z/2)**2)

def drho_dT_BE_integrand(z,x):
    return (0.25*z**3)*np.sqrt(z**2 - (x)**2)/(np.sinh(z/2)**2)


#Only needs to be done once. Has already been done. 
tabulate_integral(n_FD_integrand,'n_FD_integrand')
tabulate_integral(n_BE_integrand,'n_BE_integrand')
tabulate_integral(rho_FD_integrand,'rho_FD_integrand')
tabulate_integral(rho_BE_integrand,'rho_BE_integrand')
tabulate_integral(p_FD_integrand,'p_FD_integrand')
tabulate_integral(p_BE_integrand,'p_BE_integrand')
tabulate_integral(dn_dT_FD_integrand,'dn_dT_FD_integrand')
tabulate_integral(dn_dT_BE_integrand,'dn_dT_BE_integrand')
tabulate_integral(drho_dT_FD_integrand,'drho_dT_FD_integrand')
tabulate_integral(drho_dT_BE_integrand,'drho_dT_BE_integrand')