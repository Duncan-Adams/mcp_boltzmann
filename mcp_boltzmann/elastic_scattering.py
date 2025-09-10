import numpy as np
import scipy.special as sps
from mpmath import fp as fp
import vegas
from functools import partial

from scipy.integrate import quad
from . import polylog as plog
from mcp_boltzmann.kinematics import beta_t

class ElasticCollisionIntegral:
    def __init__(self, M_a, M_b, M_phi, zeta_a=0, zeta_b=0):
        self.M_a = M_a
        self.M_b = M_b
        self.zeta_a = zeta_a
        self.zeta_b = zeta_b
        #mphi should be a function of T_a and T_b or constant
        self.M_phi = M_phi

        if not callable(M_phi):
            self.M_phi = lambda T_a, T_b: M_phi

        self.matrix_element_nml = {
            'c000' : None,
            'c001' : None,
            'c002' : None,
            'c201' : None,
            'c021' : None,
            'c202' : None,
            'c022' : None,
            'c222' : None
        }

        return

    def integrand_nml_MB(self, p, p0, T_a, T_b, n, m, l, coeff):
        if coeff is None:
            return np.zeros(p.shape)
        m_a = self.M_a
        m_b = self.M_b
        m_phi = self.M_phi(T_a, T_b)
        
        r1 = np.arange(0, n + 1, dtype=int)[:, np.newaxis]
        r2 = np.arange(0, m + 1, dtype=int)[:, np.newaxis]

        t = p0**2 - p**2

        A = t**(l) / (t - m_phi**2)**2
        B = (2*T_a/p)**n
        C = (2*T_b/p)**m
        D = np.sum(sps.perm(n, r1)*(0.5*p*beta_t(m_a, t)/T_a)**(n-r1), axis=0)
        E = np.sum(sps.perm(m, r2)*(0.5*p*beta_t(m_b, t)/T_b)**(m-r2), axis=0)
        F = np.exp(-0.5*p*beta_t(m_a, t)/T_a)*np.exp(-0.5*p*beta_t(m_b, t)/T_b)

        return coeff*A*B*C*D*E*F

    def integrand_nml_QS(self, p, p0, T_a, T_b, n, m, l, coeff):
        if coeff is None:
            return np.zeros(p.shape)
        m_a = self.M_a
        m_b = self.M_b
        m_phi = self.M_phi(T_a, T_b)
        
        r1 = np.arange(0, n + 1, dtype=int)
        r2 = np.arange(0, m + 1, dtype=int)

        t = p0**2 - p**2

        aa = 0.5*p*beta_t(m_a, t)/T_a
        ab = 0.5*p0/T_a

        ba = 0.5*p*beta_t(m_b, t)/T_b
        bb = 0.5*p0/T_b

        A = t**(l)/(t - m_phi**2)**2
        B = (2*T_a/p)**n
        C = (2*T_b/p)**m
        D = plog.L_n_zeta(aa, ab, n, self.zeta_a)
        E = plog.L_n_zeta(ba, bb, m, self.zeta_b)


        return coeff*A*B*C*D*E

    def integrand_p_MB(self, p, p0, T_a, T_b):
        c000 = self.matrix_element_nml['c000']
        c001 = self.matrix_element_nml['c001']
        c002 = self.matrix_element_nml['c002']
        c201 = self.matrix_element_nml['c201']
        c021 = self.matrix_element_nml['c021']
        c202 = self.matrix_element_nml['c202']
        c022 = self.matrix_element_nml['c022']
        c222 = self.matrix_element_nml['c222']

        return (
             self.integrand_nml_MB(p, p0, T_a, T_b, 0,0,0, c000)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 0,0,1, c001)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 0,0,2, c002)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 2,0,1, c201)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 0,2,1, c021)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 2,0,2, c202)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 0,2,2, c022)
            +self.integrand_nml_MB(p, p0, T_a, T_b, 2,2,2, c222)
        )

    def integrand_p_QS(self, p, p0, T_a, T_b):        
        c000 = self.matrix_element_nml['c000']
        c001 = self.matrix_element_nml['c001']
        c002 = self.matrix_element_nml['c002']
        c201 = self.matrix_element_nml['c201']
        c021 = self.matrix_element_nml['c021']
        c202 = self.matrix_element_nml['c202']
        c022 = self.matrix_element_nml['c022']
        c222 = self.matrix_element_nml['c222']

        return (
             self.integrand_nml_QS(p, p0, T_a, T_b, 0,0,0, c000)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 0,0,1, c001)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 0,0,2, c002)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 2,0,1, c201)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 0,2,1, c021)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 2,0,2, c202)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 0,2,2, c022)
            +self.integrand_nml_QS(p, p0, T_a, T_b, 2,2,2, c222)
        )

    def integrand_p0_MB(self, p0, T_a, T_b):
        return p0*(np.exp(-0.5*p0/T_a)*np.exp(0.5*p0/T_b) - np.exp(0.5*p0/T_a)*np.exp(-0.5*p0/T_b))
        # return 2*p0*np.sinh(0.5*p0*(T_a-T_b)/(T_a*T_b))

    def integrand_p0_QS(self, p0, T_a, T_b):
        return p0*(
            +np.power(np.expm1(p0/T_a), -1)*np.power(-np.expm1(-p0/T_b), -1)
            -np.power(np.expm1(p0/T_b), -1)*np.power(-np.expm1(-p0/T_a), -1)
        )

    def phase_space(self, p, p0):
        #restrict the integration region of p to be greater than p0
        return np.heaviside(p - p0, 0)

    def vegas_phase_space(self, T_a, T_b, z):
        z_p, z_p0 = np.transpose(z)
        
        p = min(T_a, T_b)*(z_p/(1-z_p))
        p0 = min(T_a, T_b)*(z_p0/(1-z_p0))

        return self.phase_space(p, p0)

    def vegas_integrand_helper_MB(self, T_a, T_b, z):
        z_p, z_p0 = np.transpose(z)
        
        p = min(T_a, T_b)*(z_p/(1-z_p))
        p0 = min(T_a, T_b)*(z_p0/(1-z_p0))
        
        J = min(T_a, T_b)**2*(1/(1-z_p))**2*(1/(1-z_p0))**2

        return J*self.integrand_p0_MB(p0, T_a, T_b)*self.integrand_p_MB(p, p0, T_a, T_b)

    def vegas_integrand_helper_QS(self, T_a, T_b, z):
        z_p, z_p0 = np.transpose(z)
        
        p = min(T_a, T_b)*(z_p/(1-z_p))
        p0 = min(T_a, T_b)*(z_p0/(1-z_p0))
        
        J = min(T_a, T_b)**2*(1/(1-z_p))**2*(1/(1-z_p0))**2

        r = J*self.integrand_p0_QS(p0, T_a, T_b)*self.integrand_p_QS(p, p0, T_a, T_b)

        sel = np.where(np.isnan(r))

        return J*self.integrand_p0_QS(p0, T_a, T_b)*self.integrand_p_QS(p, p0, T_a, T_b)

    def vegas_integrand_MB(self, T_a, T_b, z):
        z_p, z_p0 = np.transpose(z)
        
        resvec = np.zeros(len(z))
        ps = self.vegas_phase_space(T_a, T_b, z)
        
        eval_ind = np.where(ps > 0)
        if(len(eval_ind[0]) == 0):
            return resvec

        resvec[eval_ind] = self.vegas_integrand_helper_MB(T_a, T_b, z[eval_ind])

        return resvec

    def vegas_integrand_QS(self, T_a, T_b, z):
        z_p, z_p0 = np.transpose(z)
        
        resvec = np.zeros(len(z))
        ps = self.vegas_phase_space(T_a, T_b, z)

        eval_ind = np.where(ps > 0)
        if(len(eval_ind[0]) == 0):
            return resvec

        resvec[eval_ind] = self.vegas_integrand_helper_QS(T_a, T_b, z[eval_ind])

        return resvec

    def compute_MB(self, T_a, T_b, n_strat=([3] + [3]), neval=1e4, nitn=30):
        integrand = vegas.batchintegrand(partial(self.vegas_integrand_MB, T_a, T_b))

        integ = vegas.Integrator([[0.0, 1.0], [0.0, 1.0]])

        train = integ(integrand, nitn=nitn, nstrat=n_strat, neval=neval)
        int_result = integ(integrand, nitn=10, nstrat=n_strat, neval=neval)

        pref = 4*T_a*T_b*((32*np.pi**2)/(2**7*(2*np.pi)**8))

        res = pref*int_result.mean
        sdev = pref*int_result.sdev

        return (res, sdev, int_result.Q, int_result)

    def compute_QS(self, T_a, T_b, n_strat=([3] + [3]), neval=1e4, nitn=30):
        integrand = vegas.batchintegrand(partial(self.vegas_integrand_QS, T_a, T_b))

        integ = vegas.Integrator([[0.0, 1.0], [0.0, 1.0]])

        train = integ(integrand, nitn=nitn, nstrat=n_strat, neval=neval)
        int_result = integ(integrand, nitn=10, nstrat=n_strat, neval=neval)

        pref = 4*T_a*T_b*((32*np.pi**2)/(2**7*(2*np.pi)**8))

        res = pref*int_result.mean
        sdev = pref*int_result.sdev

        return (res, sdev, int_result.Q, int_result)
