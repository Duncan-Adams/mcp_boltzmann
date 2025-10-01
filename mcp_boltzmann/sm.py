import os

from scipy.interpolate import interp1d
from scipy.integrate import quad, odeint
import numpy as np


base_dir = os.path.dirname(os.path.realpath(__file__))
QED_dir = os.path.join(base_dir, '../input/QED/')
rates_dir = os.path.join(base_dir, '../input/SM_Rates/')

# Conversion factor to convert MeV^-1 into seconds
FAC = 1./(6.58212e-22)

# All in MeV Units!
GF  = 1.1663787e-5*1e-6 #in MeV^{-2}
me  = 0.511
Mpl = 1.2209e19*1e3

# Left and Right nu-e couplings as relevant for E < 10 MeV. From the EW review of the PDG
geL, geR, gmuL, gmuR = 0.727, 0.233, -0.273, 0.233

P_int      =interp1d(np.loadtxt(QED_dir + "QED_P_int.cvs")[:,0],np.loadtxt(QED_dir + "QED_P_int.cvs")[:,1]+np.loadtxt(QED_dir + "QED_P_int.cvs")[:,2],bounds_error=False,fill_value=0.0,kind='cubic')
dP_intdT   =interp1d(np.loadtxt(QED_dir + "QED_dP_intdT.cvs")[:,0],np.loadtxt(QED_dir + "QED_dP_intdT.cvs")[:,1]+np.loadtxt(QED_dir + "QED_dP_intdT.cvs")[:,2],bounds_error=False,fill_value=0.0,kind='cubic')
d2P_intdT2 =interp1d(np.loadtxt(QED_dir + "QED_d2P_intdT2.cvs")[:,0],np.loadtxt(QED_dir + "QED_d2P_intdT2.cvs")[:,1]+np.loadtxt(QED_dir + "QED_d2P_intdT2.cvs")[:,2],bounds_error=False,fill_value=0.0,kind='cubic')

# ~ P_int, dP_intdT, d2P_intdT2   = lambda x: 0, lambda x: 0, lambda x: 0

# Suppression of the rates as a result of a non-negligible electron mass
f_nue_s  = interp1d(np.loadtxt(rates_dir+"nue_scatt.dat")[:,0],np.loadtxt(rates_dir+"nue_scatt.dat")[:,1],kind='cubic')
f_numu_s = interp1d(np.loadtxt(rates_dir+"numu_scatt.dat")[:,0],np.loadtxt(rates_dir+"numu_scatt.dat")[:,1],kind='cubic')
f_nue_a  = interp1d(np.loadtxt(rates_dir+"nue_ann.dat")[:,0],np.loadtxt(rates_dir+"nue_ann.dat")[:,1],kind='cubic')
f_numu_a = interp1d(np.loadtxt(rates_dir+"numu_ann.dat")[:,0],np.loadtxt(rates_dir+"numu_ann.dat")[:,1],kind='cubic')
f_nue_scan = interp1d(np.loadtxt(rates_dir+"sigmav_nue.dat")[:,0],np.loadtxt(rates_dir+"sigmav_nue.dat")[:,1],kind='cubic')
f_numu_scan = interp1d(np.loadtxt(rates_dir+"sigmav_numu.dat")[:,0],np.loadtxt(rates_dir+"sigmav_numu.dat")[:,1],kind='cubic')

##Uncomment in order to remove the effect of m_e in the rates.
# ~ f_nue_s, f_numu_s, f_nue_a, f_numu_a, f_nue_scan, f_numu_scan = lambda T : 1, lambda T : 1, lambda T : 1, lambda T : 1, lambda T : 1, lambda T : 1

# Pauli Blocking factors. Set to 1 for the MB rates
faFD, fsFD, fnFD = 0.884, 0.829, 0.852
# ~ faFD, fsFD, fnFD = 1, 1, 1

def Ffunc_nue_e(T1,T2):
    return 32* faFD *(T1**9 - T2**9) * f_nue_a(T1) + 56 * fsFD  * f_nue_s(T1)  *T1**4*T2**4*(T1-T2)

def Ffunc_numu_e(T1,T2):
    return 32* faFD *(T1**9 - T2**9) * f_numu_a(T1) + 56 * fsFD * f_numu_s(T1) *T1**4*T2**4*(T1-T2)

def Ffunc(T1,T2):
    return 32* faFD *(T1**9 - T2**9) + 56* fsFD *T1**4*T2**4*(T1-T2)


#equation A.23 of 2001.04466 
def G1(T1, mu1, T2, mu2):
    x1 = mu1/T1
    x2 = mu2/T2
    return 32*(T1**9*np.exp(2*x1) - T2**9*np.exp(2*x2))

def G2(T1, mu1, T2, mu2):
    x1 = mu1/T1
    x2 = mu2/T2
    return 56*(T1**4)*(T2**4)*(T1 - T2)*np.exp(x1)*np.exp(x2)
    # return 56*(T1**4)*(T2**4)*(T1 - T2)*(1 + x1 + x2)

def Gfunc_fo(T1, mu1, T2, mu2):
    return faFD*G1(T1, mu1, T2, mu2) + fsFD*G2(T1, mu1, T2, mu2)

def Gfunc_nue_e(T1, mu1, T2, mu2):
    return faFD*f_nue_a(T1)*G1(T1, mu1, T2, mu2) + fsFD*f_nue_s(T1)*G2(T1, mu1, T2, mu2)

def Gfunc_numu_e(T1, mu1, T2, mu2):
    return faFD*f_numu_a(T1)*G1(T1, mu1, T2, mu2) + fsFD*f_numu_s(T1)*G2(T1, mu1, T2, mu2)

def DeltaRho_nue(T_gam, T_nu, mu_nu):
    pref =  FAC*GF**2/np.pi**5

    drho = 4*(geL**2 + geR**2)*Gfunc_nue_e(T_gam, 0, T_nu, mu_nu) + 2*Gfunc_fo(T_nu, mu_nu, T_nu, mu_nu)

    return pref*drho

def DeltaRho_numu(T_gam, T_nu, mu_nu):
    pref =  FAC*GF**2/np.pi**5

    drho = 4*(gmuL**2 + gmuR**2)*Gfunc_numu_e(T_gam, 0, T_nu, mu_nu) - Gfunc_fo(T_nu, mu_nu, T_nu, mu_nu)

    return pref*drho

def DeltaN_nue(T_gam, T_nu, mu_nu):
    pref =  FAC*8*fnFD*GF**2/np.pi**5

    x_nu = mu_nu/T_nu

    dN = 4*(geL**2 + geR**2)*f_nue_scan(T_gam)*(T_gam**8 - np.exp(2*x_nu)*T_nu**8) + 2*(np.exp(2*x_nu)*T_nu**8 - np.exp(2*x_nu)*T_nu**8)

    return pref*dN

def DeltaN_numu(T_gam, T_nu, mu_nu):
    pref =  FAC*8*fnFD*GF**2/np.pi**5

    x_nu = mu_nu/T_nu

    dN = 4*(gmuL**2 + gmuR**2)*f_numu_scan(T_gam)*(T_gam**8 - np.exp(2*x_nu)*T_nu**8) - (np.exp(2*x_nu)*T_nu**8 - np.exp(2*x_nu)*T_nu**8)

    return pref*dN
