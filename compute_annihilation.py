import os
import argparse
import time
import warnings
from functools import partial

import numpy as np
import schwimmbad
from scipy.interpolate import interp1d

import mcp_boltzmann.annihilation as ann

    
def compute_annihilation_rate(m_mcp, T_min, T_max, n_temps, outdir, maxwell_boltzmann=False):
    '''
    Compute forwards annhilation collision integral (sm, sm -> mcp, mcp)
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    Tm_min - min temperature in units of mcp_mass
    Tm_max - max temperature in units of mcp_mass
    ntemps - number of geometricall spaced temperatures to compute at
    maxwell_boltzmann - If true, assume MB distribtions for everyone
    
    Returns
    -------
    (float) sum of collision integrals for stanard model particles annihilating into mcps
    '''
    
    warnings.filterwarnings('ignore')
    
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
    
    outfile_name = f'mcp_annihilation_rate_m_{args.mass}_Q_1.npz'
    if maxwell_boltzmann is True:
        outfile_name = f'mcp_annihilation_rate_m_{args.mass}_Q_1_MB.npz'
    outfile_path = os.path.join(outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        return

    Temp_grid = np.geomspace(T_min, T_max, n_temps)

    #setup cross sections
    sigma_ee_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_e)
    sigma_mumu_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_mu)
    sigma_tautau_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_tau)
    
    sigma_lq_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = 0.0)
    sigma_ss_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_s)
    sigma_cc_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_c)
    sigma_bb_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_b)
    sigma_tt_ff = partial(ann.sigma_llff, m_mcp = m_mcp, m_l = m_t)

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
    rate_qcd = np.heaviside(Temp_grid - LQCD, 0)*(Ix_table_lq + Ix_table_strange + Ix_table_charm + Ix_table_bot + Ix_table_top)
    rate_had = np.heaviside(LQCD - Temp_grid, 0)*Ix_table_pipi
    rate_tot = rate_EM + rate_qcd
    
    np.savez_compressed(
        outfile_path,
        Temp_grid=Temp_grid,
        rate=rate_tot
    )
                    
    return 
    
if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_annihilation.py',
                    description='tabulateannihilation scattering rates for MCPs')

    parser.add_argument('mass', action='store', type=float, help='mass in MeV of mcp')
    parser.add_argument('T_min', action='store', type=float, help='minimum T [MeV] to compute rate at')
    parser.add_argument('T_max', action='store', type=float, help='minimum T [MeV] to compute rate at')
    parser.add_argument('n_temps', action='store', type=int, help='number of temperatures in temperature grid')
    parser.add_argument('--maxwell_boltzmann', dest='maxwell_boltzmann', action='store_true', help='assume MB distributions')
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
        
    fun_loop = partial(
        compute_annihilation_rate, 
        T_min = args.T_min,
        T_max = args.T_max,
        n_temps= args.n_temps,
        outdir = args.outdir,
        maxwell_boltzmann = args.maxwell_boltzmann
    )
    
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    
    mass_array = np.array([args.mass])
    
    time_start = time.time()
    result_total = pool.map(fun_loop, mass_array)
    time_end = time.time()

    pool.close()

    print(f'Time: {(time_end - time_start)/60:.5f} minutes')
    exit()


