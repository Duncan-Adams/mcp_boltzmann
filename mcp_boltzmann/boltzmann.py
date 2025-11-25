#boltzmann.py - routines and class for solving boltzmann equation
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from mcp_boltzmann.distributions import *
from mcp_boltzmann.gstar import gstar_E_EM, d_gstar_E_EM_dT, w_EM

#Fundametal parameters
MeV = 1
GeV = 1e3
M_Planck = 1.22 * 1e19 * GeV

m_e = 0.511*MeV
m_mu = 105*MeV

mev_to_K = 11.6
MeV_to_K = 1e9*mev_to_K

T_nu_dec = 3.0
    

class MCPBoltzmann:
    def __init__(self, m_mcp, Q, rtol=1e-5, atol=1e-5):
        self.m_mcp = m_mcp
        self.Q = Q
        self.colterms_EM_NU = [] #collision terms between em sector and nuetrino sector
        self.colterms_EM_DS = [] #collision terms between em sector and dark sector
        self.rtol = rtol
        self.atol = atol
        
    def rho_EM(self, T_gam):
        return (np.pi**2/30)*gstar_E_EM(T_gam, T_nu_dec)*T_gam**4
    
    def p_EM(self, T_gam):
        return w_EM(T_gam, T_nu_dec)*self.rho_EM(T_gam)
    
    def drho_EM_dT(self, T_gam):
        return 4*(np.pi**2/30)*gstar_E_EM(T_gam, T_nu_dec)*T_gam**3 + (np.pi**2/30)*d_gstar_E_EM_dT(T_gam, T_nu_dec)*T_gam**4
        
    def rho_neutrino(self, T_nu):
        return 3*rho_nu(T_nu, np.zeros_like(T_nu))
    
    def drho_neutrino_dT(self, T_nu):
        return 3*drho_nudT(T_nu, np.zeros_like(T_nu))
    
    def rho_DS(self, T_ds):
        return rho_gam(T_ds) + rhoDM_FD(T_ds, np.zeros_like(T_ds), self.m_mcp)
    
    def p_DS(self, T_ds):
        return (1/3)*rho_gam(T_ds) + p_DM_FD(T_ds, np.zeros_like(T_ds), self.m_mcp)
    
    def drho_DS_dT(self, T_ds):
        return drho_gamdT(T_ds) + drhoDM_dT_FD(T_ds, np.zeros_like(T_ds), self.m_mcp)
        
    def rho_tot_sm(self, T_gam, T_nu):
        return (
            np.heaviside(T_gam - T_nu_dec, 1.0)*self.rho_EM(T_gam) 
          + np.heaviside(T_nu_dec - T_gam, 0.0)*(self.rho_EM(T_gam) + self.rho_neutrino(T_nu))
        )
        
    def p_tot_sm(self,T_gam, T_nu):
        return (
            np.heaviside(T_gam - T_nu_dec, 1.0)*self.p_EM(T_gam) 
          + np.heaviside(T_nu_dec - T_gam, 0.0)*(self.p_EM(T_gam) + (1/3)*self.rho_neutrino(T_nu))
        )
            
    def rho_tot_bsm(self, T_gam, T_nu, T_ds):
        return (
            np.heaviside(T_gam - T_nu_dec, 1.0)*(self.rho_EM(T_gam) + self.rho_DS(T_ds))
          + np.heaviside(T_nu_dec - T_gam, 0.0)*(self.rho_EM(T_gam) + self.rho_neutrino(T_nu) + self.rho_DS(T_ds))
        )

    def p_tot_bsm(self, T_gam, T_nu, T_ds):
        return (
            np.heaviside(T_gam - T_nu_dec, 1.0)*(self.p_EM(T_gam) + self.p_DS(T_ds)) 
          + np.heaviside(T_nu_dec - T_gam, 0.0)*(self.p_EM(T_gam) + (1/3)*self.rho_neutrino(T_nu) + self.p_DS(T_ds))
        )    
    
    def Hubble_SM(self, T_gam, T_nu):
        return np.sqrt((8 * np.pi)/(3 * M_Planck**2) * (self.rho_tot_sm(T_gam, T_nu)))
            
    def Hubble(self, T_gam, T_nu, T_ds):    
        return np.sqrt((8 * np.pi)/(3 * M_Planck**2) * (self.rho_tot_bsm(T_gam, T_nu, T_ds)))
        

    def add_colterm_EM_NU(self, colterm):
        '''
        Add a collision term representing energy transfer bewteen EM and nuetrino sectors
        colterm should have signature F(T_gam, T_nu), should be positive for T_gam > T_nu
        '''
        self.colterms_EM_NU.append(colterm)
        return

    def add_colterm_EM_DS(self, colterm):
        '''
        Add a collision term representing energy transfer bewteen EM and dark sectors
        colterm should have signature F(T_gam, T_ds, Q), should be positive for T_gam > T_ds
        '''
        self.colterms_EM_DS.append(colterm)
        return
        
    def colterm_EM_NU(self, T_gam, T_nu):
        res = 0
        for C in self.colterms_EM_NU:
            res += C(T_gam, T_nu)
        return res
    
    def colterm_EM_DS(self, T_gam, T_ds):
        res = 0
        for C in self.colterms_EM_DS:
            res += C(T_gam, T_ds, self.Q)
        return res
            
    def dT_EM_dt(self, T_gam, T_nu, T_ds):
        H = self.Hubble(T_gam, T_nu, T_ds)
        
        hub_term = -3*H*(self.rho_EM(T_gam) + self.p_EM(T_gam))
        col_term = -self.colterm_EM_NU(T_gam, T_nu) - self.colterm_EM_DS(T_gam, T_ds)
        
        return (1/self.drho_EM_dT(T_gam))*(hub_term + col_term)
        
    def dT_EM_dt_SM(self, T_gam, T_nu):
        H = self.Hubble_SM(T_gam, T_nu)
        
        hub_term = -3*H*(self.rho_EM(T_gam) + self.p_EM(T_gam))
        col_term = -self.colterm_EM_NU(T_gam, T_nu)
        
        return (1/self.drho_EM_dT(T_gam))*(hub_term + col_term)
       
    def dT_neutrino_dt(self, T_gam, T_nu, T_ds):      
        if T_gam >= T_nu_dec:
            return np.atleast_1d(self.dT_EM_dt(T_gam, T_nu, T_ds))
            
        H = self.Hubble(T_gam, T_nu, T_ds)
        
        hub_term = -4*H*(self.rho_neutrino(T_nu))
        col_term = self.colterm_EM_NU(T_gam, T_nu)

        return (1/self.drho_neutrino_dT(T_nu))*(hub_term + col_term)
        
    def dT_neutrino_dt_SM(self, T_gam, T_nu):        
        if T_gam >= T_nu_dec:
            return self.dT_EM_dt_SM(T_gam, T_nu)
        
        H = self.Hubble_SM(T_gam, T_nu)
        
        hub_term = -4*H*(self.rho_neutrino(T_nu))
        col_term = self.colterm_EM_NU(T_gam, T_nu)
        
        return (1/self.drho_neutrino_dT(T_nu))*(hub_term + col_term)
        
    def dT_DS_dt(self, T_gam, T_nu, T_ds):
        H = self.Hubble(T_gam, T_nu, T_ds)
        
        hub_term = -3*H*(self.rho_DS(T_ds) + self.p_DS(T_ds))
        col_term = self.colterm_EM_DS(T_gam, T_ds)
        
        return (1/self.drho_DS_dT(T_ds))*(hub_term + col_term)
        
    def solve_boltzmann_eq(self, T_gam0, T_nu0, T_ds0):
        '''
        Setup and solve the coupled boltzmann equations governing the evolution of the EM temperature, nuetrino temperature
        and the dark sector temperature, with a massless dark photon and a  millicharged particle of mass self.m_mcp and charge Q
        
        Params
        ------
        T_gam0 - initial electromagnetic plasma temperature
        T_nu0 - initial nuetrino temp, should probably be equl to T_gam0 unless you want to do some funky shit
        T_ds0 - initial dark sector temperature
        Q - EM charge of millicharged particle
        
        Returns
        -------
        
        sol - output of scipy.integrate.solve_ivp containing the temperatures of the 3 sectors as a function of time
        '''
        
        #stops the solver when the photon temp goes below the neutrino decoupling temp
        def nu_dec(t, vec):
            T_gam, T_ds, a = vec
            
            return T_gam - T_nu_dec
        nu_dec.terminal = True
        
        def dT_pre_nudec(t, vec):
            T_gam, T_ds, a = vec
            T_nu = T_gam
            res = np.array([
                self.dT_EM_dt(T_gam, T_nu, T_ds)[0],
                self.dT_DS_dt(T_gam, T_nu, T_ds)[0],
                a*self.Hubble(T_gam, T_nu, T_ds)[0]
            ]
            )
            return res
            
        def dT_post_nudec(t, vec):
            T_gam, T_nu, T_ds, a = vec
            res = np.array([
                self.dT_EM_dt(T_gam, T_nu, T_ds)[0],
                self.dT_neutrino_dt(T_gam, T_nu, T_ds)[0],
                self.dT_DS_dt(T_gam, T_nu, T_ds)[0],
                a*self.Hubble(T_gam, T_nu, T_ds)[0]
            ]
            )
            return res
            
        
        IC_0 = [T_gam0, T_ds0, 1.0]
        t0 = 1./(2 * self.Hubble(T_gam0, T_nu0, T_ds0))[0]
        t_max = 1e29
        t_eval = np.geomspace(t0, t_max, 500)
        
        sol_0 = solve_ivp(dT_pre_nudec, [t0, t_max], IC_0, t_eval=t_eval, events=nu_dec, method='BDF', rtol=self.rtol, atol=self.atol)
        
        T_gam1 = sol_0.y[0][-1]
        T_ds1 = sol_0.y[1][-1]
        a_1 = sol_0.y[2][-1]
        
        t1 = sol_0.t[-1]
        
        IC_1 = [T_gam1, T_gam1, T_ds1, a_1]
        
        sol_1 = solve_ivp(dT_post_nudec, [t1, t_max], IC_1, method='BDF', rtol=self.rtol, atol=self.atol)
        
        # ~ #renormalize scale factors
        scale_factor_end = sol_1.y[3][-1]
        T_gam_end = sol_1.y[0][-1]
        
        rel_scale_factor = 2.7/(T_gam_end*MeV_to_K)
        
        cosmic_time = np.concatenate((sol_0.t, sol_1.t))
        T_gam = np.concatenate((sol_0.y[0], sol_1.y[0]))
        T_nu = np.concatenate((sol_0.y[0], sol_1.y[1]))
        T_ds = np.concatenate((sol_0.y[1], sol_1.y[2]))
        scale_factor = np.concatenate((sol_0.y[2], sol_1.y[3]))*(rel_scale_factor/scale_factor_end)
        
        return cosmic_time, T_gam, T_nu, T_ds, scale_factor

    def solve_boltzmann_eq_L(self, T_gam0, T_nu0, T_ds0):
        '''
        Setup and solve the coupled boltzmann equations governing the evolution of the EM temperature, nuetrino temperature
        and the dark sector temperature, with a massless dark photon and a  millicharged particle of mass self.m_mcp and charge Q
        
        Params
        ------
        T_gam0 - initial electromagnetic plasma temperature
        T_nu0 - initial nuetrino temp, should probably be equl to T_gam0 unless you want to do some funky shit
        T_ds0 - initial dark sector temperature
        Q - EM charge of millicharged particle
        
        Returns
        -------
        
        sol - output of scipy.integrate.solve_ivp containing the temperatures of the 3 sectors as a function of time
        '''
        
        #stops the solver when the photon temp goes below the neutrino decoupling temp
        def nu_dec(t, vec):
            L_gam, L_ds, a = vec
            T_gam = np.exp(L_gam)
            T_ds = np.exp(L_ds)
            
            return T_gam - T_nu_dec
        nu_dec.terminal = True
        
        def dT_pre_nudec(t, vec):
            L_gam, L_ds, a = vec
            T_gam = np.exp(L_gam)
            T_ds = np.exp(L_ds)
            T_nu = T_gam
            res = np.array([
                (1/T_gam)*self.dT_EM_dt(T_gam, T_nu, T_ds)[0],
                (1/T_ds)*self.dT_DS_dt(T_gam, T_nu, T_ds)[0],
                a*self.Hubble(T_gam, T_nu, T_ds)[0]
            ])
            return res
            
        def dT_post_nudec(t, vec):
            T_gam, T_nu, T_ds, a = vec
            res = np.array([
                self.dT_EM_dt(T_gam, T_nu, T_ds)[0],
                self.dT_neutrino_dt(T_gam, T_nu, T_ds)[0],
                self.dT_DS_dt(T_gam, T_nu, T_ds)[0],
                a*self.Hubble(T_gam, T_nu, T_ds)[0]
            ])
            return res
            
        
        IC_0 = [np.log(T_gam0), np.log(T_ds0), 1.0]
        t0 = 1./(2 * self.Hubble(T_gam0, T_nu0, T_ds0))[0]
        t_max = 1e29
        t_eval = np.geomspace(t0, t_max, 500)
        
        sol_0 = solve_ivp(dT_pre_nudec, [t0, t_max], IC_0, t_eval=t_eval, events=nu_dec, method='BDF', rtol=self.rtol, atol=self.atol)
        
        T_gam1 = np.exp(sol_0.y[0][-1])
        T_ds1 = np.exp(sol_0.y[1][-1])
        a_1 = sol_0.y[2][-1]
        
        t1 = sol_0.t[-1]
        
        IC_1 = [T_gam1, T_gam1, T_ds1, a_1]
        
        sol_1 = solve_ivp(dT_post_nudec, [t1, t_max], IC_1, method='BDF', rtol=self.rtol, atol=self.atol)
        
        # ~ #renormalize scale factors
        scale_factor_end = sol_1.y[3][-1]
        T_gam_end = sol_1.y[0][-1]
        
        rel_scale_factor = 2.7/(T_gam_end*MeV_to_K)
        
        cosmic_time = np.concatenate((sol_0.t, sol_1.t))
        T_gam = np.concatenate((np.exp(sol_0.y[0]), sol_1.y[0]))
        T_nu = np.concatenate((np.exp(sol_0.y[0]), sol_1.y[1]))
        T_ds = np.concatenate((np.exp(sol_0.y[1]), sol_1.y[2]))
        scale_factor = np.concatenate((sol_0.y[2], sol_1.y[3]))*(rel_scale_factor/scale_factor_end)
        
        return cosmic_time, T_gam, T_nu, T_ds, scale_factor
        
    def solve_boltzmann_eq_SM(self, T_gam0, T_nu0):
        '''
        Setup and solve the coupled boltzmann equations governing the evolution of the EM temperature and nuetrino temperature
        in the standard model
        
        Params
        ------
        T_gam0 - initial electromagnetic plasma temperature
        T_nu0 - initial nuetrino temp, should probably be equl to T_gam0 unless you want to do some funky shit
        
        Returns
        -------
        
        sol - output of scipy.integrate.solve_ivp containing the temperatures of the 2 sectors as a function of time, as well as the scale factor
        '''
        
        def nu_dec(t, vec):
            T_gam, a = vec
            return T_gam - T_nu_dec
            
        nu_dec.terminal = True
        
        def dT_pre_nudec(t, vec):
            T_gam, a = vec
            res = np.array([
                self.dT_EM_dt_SM(T_gam, T_gam),
                a*self.Hubble_SM(T_gam, T_gam)
            ])
            return res
            
        def dT_post_nudec(t, vec):
            T_gam, T_nu, a = vec
            res = np.array([
                self.dT_EM_dt_SM(T_gam, T_nu),
                self.dT_neutrino_dt_SM(T_gam, T_nu),
                a*self.Hubble_SM(T_gam, T_nu)
            ])
            return res
            
        IC = [T_gam0, 1.0]
        t0 = 1./(2 * self.Hubble_SM(T_gam0, T_nu0))
        t_max = 1e29
        t_eval = np.geomspace(t0, t_max, 500)
        
        sol_0 = solve_ivp(dT_pre_nudec, [t0, t_max], IC, t_eval=t_eval, events=nu_dec, method='BDF', rtol=self.rtol, atol=self.atol)
        
        T_gam1 = sol_0.y[0][-1]
        a_1 = sol_0.y[1][-1]
        t1 = sol_0.t[-1]
        
        IC_1 = [T_gam1, T_gam1, a_1]
        
        sol_1 = solve_ivp(dT_post_nudec, [t1, t_max], IC_1, method='BDF', rtol=self.rtol, atol=self.atol)
        
        #renormalize scale factors
        scale_factor_end = sol_1.y[2][-1]
        T_gam_end = sol_1.y[0][-1]
        
        rel_scale_factor = 2.7/(T_gam_end*MeV_to_K)
        
        cosmic_time = np.concatenate((sol_0.t, sol_1.t))
        T_gam = np.concatenate((sol_0.y[0], sol_1.y[0]))
        T_nu = np.concatenate((sol_0.y[0], sol_1.y[1]))
        scale_factor = np.concatenate((sol_0.y[1], sol_1.y[2]))*(rel_scale_factor/scale_factor_end)

        return cosmic_time, T_gam, T_nu, scale_factor
          
    def guess_initial_dark_temp(self, T_gam0):
        T_dark_test = np.geomspace(1e-4*T_gam0, 1.1*T_gam0)
        R = 4*self.Hubble(T_gam0, T_gam0, T_dark_test)*self.rho_DS(T_dark_test)/np.abs(self.colterm_EM_DS(T_gam0, T_dark_test))
        id_initial_T_dark = (np.abs(np.log10(R))).argmin() #find where R = 1
        
        T_dark_inital = np.minimum(T_dark_test[id_initial_T_dark], T_gam0)
        
        return T_dark_inital
        
    def N_eff(self, T_gam, T_nu, T_ds):
        return (8/7)*(11/4)**(4/3)*((self.rho_neutrino(T_nu) + self.rho_DS(T_ds))/self.rho_EM(T_gam))

    def N_eff_SM(self, T_gam, T_nu):
        return (8/7)*(11/4)**(4/3)*((self.rho_neutrino(T_nu)/self.rho_EM(T_gam)))
    #compute delta Neff only conisdering the temperature of the dark photons and photons
    def Delta_Neff_ds_only(self, T_gam, T_ds):
        return (8/7)*(11/4)**(4/3)*(self.rho_DS(T_ds)/self.rho_EM(T_gam))
        


class ADMBoltzmann(MCPBoltzmann):
    def __init__(self, m_de, m_dp, Q, rtol=1e-5, atol=1e-5):
        self.m_de = m_de
        self.m_dp = m_dp
        self.Q = Q
        self.colterms_EM_NU = [] #collision terms between em sector and nuetrino sector
        self.colterms_EM_DS = [] #collision terms between em sector and dark sector
        self.rtol = rtol
        self.atol = atol
        
    def rho_DS(self, T_ds):
        return rho_gam(T_ds) + rhoDM_FD(T_ds, np.zeros_like(T_ds), self.m_de) + rhoDM_FD(T_ds, np.zeros_like(T_ds), self.m_dp)
    
    def p_DS(self, T_ds):
        return (1/3)*rho_gam(T_ds) + p_DM_FD(T_ds, np.zeros_like(T_ds), self.m_de) + p_DM_FD(T_ds, np.zeros_like(T_ds), self.m_dp)
    
    def drho_DS_dT(self, T_ds):
        return drho_gamdT(T_ds) + drhoDM_dT_FD(T_ds, np.zeros_like(T_ds), self.m_de) + drhoDM_dT_FD(T_ds, np.zeros_like(T_ds), self.m_dp)
