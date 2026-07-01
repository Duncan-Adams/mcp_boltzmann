import os
import argparse
import time
import warnings
from functools import partial

import numpy as np
import schwimmbad
from scipy.interpolate import interp1d

import mcp_boltzmann.annihilation as ann
    
def compute_annihilation_rate_mcp_fermionic_gamma_only(task, n_temps, outdir, maxwell_boltzmann=False, summed=True):
    '''
    Compute forwards photon mediated annhilation collision integral (sm, sm -> mcp, mcp)
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    Tm_min - min temperature in units of mcp_mass
    Tm_max - max temperature in units of mcp_mass
    ntemps - number of geometricall spaced temperatures to compute at
    maxwell_boltzmann - If true, assume MB distribtions for everyone
    summed - If True, the output is the summed rate
    
    Returns
    -------
    (float) sum of collision integrals for stanard model particles annihilating into dirac fermion mcps
    '''
    
    warnings.filterwarnings('ignore')
    
    m_mcp, T_min, T_max = task
    
    MeV = 1
    GeV = 1e3*MeV
    
    m_e = 0.511*MeV
    m_mu = 105*MeV
    m_tau = 1776*MeV
    m_s = 95*MeV
    m_c = 1270*MeV
    m_b = 4180*MeV
    m_t = 172.76*GeV
    m_pi = 139.570 * MeV
    Q = 1.0
    
    outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1.npz'
    if maxwell_boltzmann is True:
        outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1_MB.npz'
    outfile_path = os.path.join(outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        return

    Temp_grid = np.geomspace(T_min, T_max, n_temps)

    #setup cross sections
    sigma_ee_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_e)
    sigma_mumu_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_mu)
    sigma_tautau_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_tau)
    
    sigma_lq_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = 0.0)
    sigma_ss_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_s)
    sigma_cc_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_c)
    sigma_bb_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_b)
    sigma_tt_ff = partial(ann.sigma_ff_xx_fermionic, m_mcp = m_mcp, m_f = m_t)

    sigma_pipi_ff = partial(ann.sigma_pipi_ff, m_mcp = m_mcp)

    if maxwell_boltzmann is True:
        Ix_table_e = np.array(
            [ann.Ix(sigma_ee_ff, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix(sigma_mumu_ff, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix(sigma_tautau_ff, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [ann.Ix(sigma_lq_ff, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix(sigma_ss_ff, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix(sigma_cc_ff, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix(sigma_bb_ff, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix(sigma_tt_ff, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ff, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
    else:
        Ix_table_e = np.array(
            [ann.Ix_fermi(sigma_ee_ff, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix_fermi(sigma_mumu_ff, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix_fermi(sigma_tautau_ff, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [ann.Ix_fermi(sigma_lq_ff, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix_fermi(sigma_ss_ff, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix_fermi(sigma_cc_ff, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix_fermi(sigma_bb_ff, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix_fermi(sigma_tt_ff, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ff, m_mcp, m_pi, T) for T in Temp_grid]
        )
    
    
    LQCD = 200*MeV
    rate_EM = Ix_table_e + Ix_table_mu + Ix_table_tau

    rate_lq = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_lq)
    rate_strange = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_strange)
    rate_charm = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_charm)
    rate_bot = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_bot)
    rate_top = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_top)
    
    rate_qcd = rate_lq + rate_strange + rate_charm + rate_bot + rate_top
    rate_had = np.heaviside(LQCD - Temp_grid, 0)*Ix_table_pipi
    rate_tot = rate_EM + rate_qcd + rate_had
    
    if summed is True:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate=rate_tot
        )
        
    else:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate_e = Ix_table_e,
            rate_mu = Ix_table_mu,
            rate_tau = Ix_table_tau,
            rate_lq = rate_lq,
            rate_strange = rate_strange,
            rate_charm = rate_charm,
            rate_bot = rate_bot,
            rate_top = rate_top,
            rate_pion = rate_had,
            rate_tot = rate_tot
        )
                    
    return 
  
def compute_annihilation_rate_mcp_fermionic(task, n_temps, outdir, maxwell_boltzmann=False, summed=True):
    '''
    Compute forwards annhilation collision integral (sm, sm -> mcp, mcp) including Z boson contributions
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    Tm_min - min temperature in units of mcp_mass
    Tm_max - max temperature in units of mcp_mass
    ntemps - number of geometricall spaced temperatures to compute at
    maxwell_boltzmann - If true, assume MB distribtions for everyone
    summed - If True, the output is the summed rate
    
    Returns
    -------
    (float) sum of collision integrals for stanard model particles annihilating into dirac fermion mcps
    '''
    
    warnings.filterwarnings('ignore')
    
    m_mcp, T_min, T_max = task
    
    MeV = 1
    GeV = 1e3*MeV
    
    m_e = 0.511*MeV
    m_mu = 105*MeV
    m_tau = 1776*MeV
    m_s = 95*MeV
    m_c = 1270*MeV
    m_b = 4180*MeV
    m_t = 172.76*GeV
    m_pi = 139.570 * MeV
    M_W = 80369.2*MeV
    Q = 1.0
    
    outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1.npz'
    if maxwell_boltzmann is True:
        outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1_MB.npz'
    outfile_path = os.path.join(outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        return

    Temp_grid = np.geomspace(T_min, T_max, n_temps)

    #setup cross sections
    sigma_ee_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_e, q_f=-1, cv=ann.cv_e, ca=ann.ca_e)
    sigma_mumu_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_mu, q_f=-1, cv=ann.cv_mu, ca=ann.ca_mu)
    sigma_tautau_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_tau, q_f=-1, cv=ann.cv_tau, ca=ann.ca_tau)
    
    sigma_uu_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=ann.q_u, cv=ann.cv_u, ca=ann.ca_u)
    sigma_dd_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=ann.q_d, cv=ann.cv_d, ca=ann.ca_d)
    sigma_ss_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_s, q_f=ann.q_s, cv=ann.cv_s, ca=ann.ca_s)
    sigma_cc_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_c, q_f=ann.q_c, cv=ann.cv_c, ca=ann.ca_c)
    sigma_bb_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_b, q_f=ann.q_b, cv=ann.cv_b, ca=ann.ca_b)
    sigma_tt_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = m_t, q_f=ann.q_t, cv=ann.cv_t, ca=ann.ca_t)
    
    sigma_nunu_ff_Z = partial(ann.sigma_ff_xx_fermionic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=0.0, cv=ann.cv_nu, ca=ann.ca_nu)

    sigma_pipi_ff = partial(ann.sigma_pipi_ff, m_mcp = m_mcp)
    
    sigma_WW_ff = partial(ann.sigma_WW_xx_fermionic, m_mcp=m_mcp)

    if maxwell_boltzmann is True:
        Ix_table_e = np.array(
            [ann.Ix(sigma_ee_ff_Z, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix(sigma_mumu_ff_Z, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix(sigma_tautau_ff_Z, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [(ann.Ix(sigma_uu_ff_Z, m_mcp, 0.0, T) + ann.Ix(sigma_dd_ff_Z, m_mcp, 0.0, T)) for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix(sigma_ss_ff_Z, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix(sigma_cc_ff_Z, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix(sigma_bb_ff_Z, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix(sigma_tt_ff_Z, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_nu =  np.array(
            [ann.Ix(sigma_nunu_ff_Z, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ff, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
        Ix_table_WW = np.array(
            [ann.Ix(sigma_WW_ff, m_mcp, M_W, T) for T in Temp_grid]
        )
        
    else:
        Ix_table_e = np.array(
            [ann.Ix_fermi(sigma_ee_ff_Z, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix_fermi(sigma_mumu_ff_Z, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix_fermi(sigma_tautau_ff_Z, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [(ann.Ix_fermi(sigma_uu_ff_Z, m_mcp, 0.0, T) + ann.Ix_fermi(sigma_dd_ff_Z, m_mcp, 0.0, T))  for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix_fermi(sigma_ss_ff_Z, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix_fermi(sigma_cc_ff_Z, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix_fermi(sigma_bb_ff_Z, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix_fermi(sigma_tt_ff_Z, m_mcp, m_t, T) for T in Temp_grid]
        )
    
        Ix_table_nu =  np.array(
            [ann.Ix_fermi(sigma_nunu_ff_Z, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix_bose(sigma_pipi_ff, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
        Ix_table_WW = np.array(
            [ann.Ix_bose(sigma_WW_ff, m_mcp, M_W, T) for T in Temp_grid]
        )
    
    
    LQCD = 200*MeV
    rate_EM = Ix_table_e + Ix_table_mu + Ix_table_tau

    rate_lq = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_lq)
    rate_strange = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_strange)
    rate_charm = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_charm)
    rate_bot = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_bot)
    rate_top = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_top)
    
    rate_qcd = rate_lq + rate_strange + rate_charm + rate_bot + rate_top
    rate_had = np.heaviside(LQCD - Temp_grid, 0)*Ix_table_pipi
    
    rate_nu = Ix_table_nu
    
    rate_W = Ix_table_WW
    
    rate_tot = rate_EM + rate_qcd + rate_had + rate_nu + rate_W
    
    if summed is True:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate=rate_tot
        )
        
    else:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate_e = Ix_table_e,
            rate_mu = Ix_table_mu,
            rate_tau = Ix_table_tau,
            rate_lq = rate_lq,
            rate_strange = rate_strange,
            rate_charm = rate_charm,
            rate_bot = rate_bot,
            rate_top = rate_top,
            rate_pion = rate_had,
            rate_nu = rate_nu,
            rate_W = rate_W,
            rate_tot = rate_tot
        )
                    
    return 
 
 
def compute_annihilation_rate_mcp_fermionic_gamma_only(task, n_temps, outdir, maxwell_boltzmann=False, summed=True):
    '''
    Compute forwards photon mediated annhilation collision integral (sm, sm -> mcp, mcp)
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    Tm_min - min temperature in units of mcp_mass
    Tm_max - max temperature in units of mcp_mass
    ntemps - number of geometricall spaced temperatures to compute at
    maxwell_boltzmann - If true, assume MB distribtions for everyone
    summed - If True, the output is the summed rate
    
    Returns
    -------
    (float) sum of collision integrals for stanard model particles annihilating into dirac fermion mcps
    '''
    
    warnings.filterwarnings('ignore')
    
    m_mcp, T_min, T_max = task
    
    MeV = 1
    GeV = 1e3*MeV
    
    m_e = 0.511*MeV
    m_mu = 105*MeV
    m_tau = 1776*MeV
    m_s = 95*MeV
    m_c = 1270*MeV
    m_b = 4180*MeV
    m_t = 172.76*GeV
    m_pi = 139.570 * MeV
    Q = 1.0
    
    outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1.npz'
    if maxwell_boltzmann is True:
        outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1_MB.npz'
    outfile_path = os.path.join(outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        return

    Temp_grid = np.geomspace(T_min, T_max, n_temps)

    #setup cross sections
    sigma_ee_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_e)
    sigma_mumu_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_mu)
    sigma_tautau_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_tau)
    
    sigma_lq_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = 0.0)
    sigma_ss_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_s)
    sigma_cc_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_c)
    sigma_bb_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_b)
    sigma_tt_ss = partial(ann.sigma_ff_xx_bosonic, m_mcp = m_mcp, m_f = m_t)

    sigma_pipi_ss = partial(ann.sigma_pipi_ff, m_mcp = m_mcp)

    if maxwell_boltzmann is True:
        Ix_table_e = np.array(
            [ann.Ix(sigma_ee_ss, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix(sigma_mumu_ss, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix(sigma_tautau_ss, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [ann.Ix(sigma_lq_ss, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        #strange strange to scalar scalar
        Ix_table_strange = np.array(
            [ann.Ix(sigma_ss_ss, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix(sigma_cc_ss, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix(sigma_bb_ss, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix(sigma_tt_ss, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ss, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
    else:
        Ix_table_e = np.array(
            [ann.Ix_fermi(sigma_ee_ss, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix_fermi(sigma_mumu_ss, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix_fermi(sigma_tautau_ss, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [ann.Ix_fermi(sigma_lq_ss, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix_fermi(sigma_ss_ss, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix_fermi(sigma_cc_ss, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix_fermi(sigma_bb_ss, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix_fermi(sigma_tt_ss, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ss, m_mcp, m_pi, T) for T in Temp_grid]
        )
    
    
    LQCD = 200*MeV
    rate_EM = Ix_table_e + Ix_table_mu + Ix_table_tau

    rate_lq = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_lq)
    rate_strange = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_strange)
    rate_charm = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_charm)
    rate_bot = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_bot)
    rate_top = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_top)
    
    rate_qcd = rate_lq + rate_strange + rate_charm + rate_bot + rate_top
    rate_had = np.heaviside(LQCD - Temp_grid, 0)*Ix_table_pipi
    rate_tot = rate_EM + rate_qcd + rate_had
    
    if summed is True:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate=rate_tot
        )
        
    else:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate_e = Ix_table_e,
            rate_mu = Ix_table_mu,
            rate_tau = Ix_table_tau,
            rate_lq = rate_lq,
            rate_strange = rate_strange,
            rate_charm = rate_charm,
            rate_bot = rate_bot,
            rate_top = rate_top,
            rate_pion = rate_had,
            rate_tot = rate_tot
        )
                    
    return 
    
    
def compute_annihilation_rate_mcp_bosonic(task, n_temps, outdir, maxwell_boltzmann=False, summed=True):
    '''
    Compute forwards annhilation collision integral (sm, sm -> mcp, mcp) including Z boson contributions
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    Tm_min - min temperature in units of mcp_mass
    Tm_max - max temperature in units of mcp_mass
    ntemps - number of geometricall spaced temperatures to compute at
    maxwell_boltzmann - If true, assume MB distribtions for everyone
    summed - If True, the output is the summed rate
    
    Returns
    -------
    (float) sum of collision integrals for stanard model particles annihilating into dirac fermion mcps
    '''
    
    warnings.filterwarnings('ignore')
    
    m_mcp, T_min, T_max = task
    
    MeV = 1
    GeV = 1e3*MeV
    
    m_e = 0.511*MeV
    m_mu = 105*MeV
    m_tau = 1776*MeV
    m_s = 95*MeV
    m_c = 1270*MeV
    m_b = 4180*MeV
    m_t = 172.76*GeV
    m_pi = 139.570 * MeV
    M_W = 80369.2*MeV
    Q = 1.0
    
    outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1.npz'
    if maxwell_boltzmann is True:
        outfile_name = f'mcp_annihilation_rate_m_{m_mcp}_Q_1_MB.npz'
    outfile_path = os.path.join(outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        return

    Temp_grid = np.geomspace(T_min, T_max, n_temps)

    #setup cross sections
    sigma_ee_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_e, q_f=-1, cv=ann.cv_e, ca=ann.ca_e)
    sigma_mumu_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_mu, q_f=-1, cv=ann.cv_mu, ca=ann.ca_mu)
    sigma_tautau_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_tau, q_f=-1, cv=ann.cv_tau, ca=ann.ca_tau)
    
    sigma_uu_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=ann.q_u, cv=ann.cv_u, ca=ann.ca_u)
    sigma_dd_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=ann.q_d, cv=ann.cv_d, ca=ann.ca_d)
    sigma_ss_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_s, q_f=ann.q_s, cv=ann.cv_s, ca=ann.ca_s)
    sigma_cc_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_c, q_f=ann.q_c, cv=ann.cv_c, ca=ann.ca_c)
    sigma_bb_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_b, q_f=ann.q_b, cv=ann.cv_b, ca=ann.ca_b)
    sigma_tt_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = m_t, q_f=ann.q_t, cv=ann.cv_t, ca=ann.ca_t)
    
    sigma_nunu_ss_Z = partial(ann.sigma_ff_xx_bosonic_Z_boson, m_mcp = m_mcp, m_f = 0.0, q_f=0.0, cv=ann.cv_nu, ca=ann.ca_nu)

    sigma_pipi_ss = partial(ann.sigma_pipi_ss, m_mcp = m_mcp)
    
    sigma_WW_ss = partial(ann.sigma_WW_xx_bosonic, m_mcp=m_mcp)

    if maxwell_boltzmann is True:
        Ix_table_e = np.array(
            [ann.Ix(sigma_ee_ss_Z, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix(sigma_mumu_ss_Z, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix(sigma_tautau_ss_Z, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [(ann.Ix(sigma_uu_ss_Z, m_mcp, 0.0, T) + ann.Ix(sigma_dd_ss_Z, m_mcp, 0.0, T)) for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix(sigma_ss_ss_Z, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix(sigma_cc_ss_Z, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix(sigma_bb_ss_Z, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix(sigma_tt_ss_Z, m_mcp, m_t, T) for T in Temp_grid]
        )
        
        Ix_table_nu =  np.array(
            [ann.Ix(sigma_nunu_ss_Z, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix(sigma_pipi_ss, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
        Ix_table_WW = np.array(
            [ann.Ix(sigma_WW_ss, m_mcp, M_W, T) for T in Temp_grid]
        )
        
    else:
        Ix_table_e = np.array(
            [ann.Ix_fermi(sigma_ee_ss_Z, m_mcp, m_e, T) for T in Temp_grid]
        )
        
        Ix_table_mu = np.array(
            [ann.Ix_fermi(sigma_mumu_ss_Z, m_mcp, m_mu, T) for T in Temp_grid]
        )
        
        Ix_table_tau = np.array(
            [ann.Ix_fermi(sigma_tautau_ss_Z, m_mcp, m_tau, T) for T in Temp_grid]
        )
        
        Ix_table_lq = np.array(
            [(ann.Ix_fermi(sigma_uu_ss_Z, m_mcp, 0.0, T) + ann.Ix_fermi(sigma_dd_ss_Z, m_mcp, 0.0, T))  for T in Temp_grid]
        )
        
        Ix_table_strange = np.array(
            [ann.Ix_fermi(sigma_ss_ss_Z, m_mcp, m_s, T) for T in Temp_grid]
        )
        
        Ix_table_charm = np.array(
            [ann.Ix_fermi(sigma_cc_ss_Z, m_mcp, m_c, T) for T in Temp_grid]
        )
        
        Ix_table_bot = np.array(
            [ann.Ix_fermi(sigma_bb_ss_Z, m_mcp, m_b, T) for T in Temp_grid]
        )
        
        Ix_table_top = np.array(
            [ann.Ix_fermi(sigma_tt_ss_Z, m_mcp, m_t, T) for T in Temp_grid]
        )
    
        Ix_table_nu =  np.array(
            [ann.Ix_fermi(sigma_nunu_ss_Z, m_mcp, 0.0, T) for T in Temp_grid]
        )
        
        Ix_table_pipi = np.array(
            [ann.Ix_bose(sigma_pipi_ss, m_mcp, m_pi, T) for T in Temp_grid]
        )
        
        Ix_table_WW = np.array(
            [ann.Ix_bose(sigma_WW_ss, m_mcp, M_W, T) for T in Temp_grid]
        )
    
    
    LQCD = 200*MeV
    rate_EM = Ix_table_e + Ix_table_mu + Ix_table_tau

    rate_lq = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_lq)
    rate_strange = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_strange)
    rate_charm = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_charm)
    rate_bot = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_bot)
    rate_top = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_top)
    
    rate_qcd = rate_lq + rate_strange + rate_charm + rate_bot + rate_top
    rate_had = np.heaviside(LQCD - Temp_grid, 0)*Ix_table_pipi
    
    rate_nu = Ix_table_nu
    
    rate_W = Ix_table_WW
    
    rate_tot = rate_EM + rate_qcd + rate_had + rate_nu + rate_W
    
    if summed is True:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate=rate_tot
        )
        
    else:
        np.savez_compressed(
            outfile_path,
            Temp_grid=Temp_grid,
            rate_e = Ix_table_e,
            rate_mu = Ix_table_mu,
            rate_tau = Ix_table_tau,
            rate_lq = rate_lq,
            rate_strange = rate_strange,
            rate_charm = rate_charm,
            rate_bot = rate_bot,
            rate_top = rate_top,
            rate_pion = rate_had,
            rate_nu = rate_nu,
            rate_W = rate_W,
            rate_tot = rate_tot
        )
    return 
     
 
    
if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_annihilation.py',
                    description='tabulate annihilation rates for MCPs')

    parser.add_argument('mass_list', action='store', type=str, help='filename of csv with mass, T_min, T_max in [MeV]')
    parser.add_argument('n_temps', action='store', type=int, help='number of temperatures in temperature grid')
    parser.add_argument('--maxwell_boltzmann', dest='maxwell_boltzmann', action='store_true', help='assume MB distributions')
    parser.add_argument('--summed_only', dest='summed', action='store_true', help='output file contains only sum total rate')
    parser.add_argument('--weak', dest='weak', action='store_true',help='Include Z boson mediated contribution and W boson initial states')
    parser.add_argument('--bose', dest='bose', action='store_true', help='Treat MCPs as complex scalars instead of dirac fermions' )
    parser.add_argument('--outdir', dest='outdir', action='store', default='./', type=str)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    #setup output directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        
    tasks = np.genfromtxt(args.mass_list, delimiter=',', skip_header=0)
    
    if args.weak is True:
        fun = compute_annihilation_rate_mcp_fermionic
    else:
        fun = compute_annihilation_rate_mcp_fermionic_gamma_only
        
    if args.bose is True:
        if args.weak is True:
            fun = compute_annihilation_rate_mcp_bosonic
        else:
            fun = compute_annihilation_rate_mcp_bosonic_gamma_only
        
    fun_loop = partial(
        fun, 
        n_temps= args.n_temps,
        outdir = args.outdir,
        maxwell_boltzmann = args.maxwell_boltzmann,
        summed=args.summed
    )
    
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    

    
    time_start = time.time()
    result_total = pool.map(fun_loop, tasks)
    time_end = time.time()

    pool.close()

    print(f'Time: {(time_end - time_start)/60:.5f} minutes')
    exit()


