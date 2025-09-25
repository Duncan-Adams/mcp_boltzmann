import os
import argparse
import time
import warnings
from functools import partial

import numpy as np
import schwimmbad
from scipy.interpolate import interp1d

import mcp_boltzmann.elastic_scattering as elscat
import mcp_boltzmann.plasma as plas

def setup_coulomb_integrals(mass, mgam):
    m_e = 0.511
    m_mu = 105
    m_tau = 1776
    m_s = 95
    m_c = 1270
    m_mcp = mass
    
    coulomb_e = elscat.ElasticCollisionIntegral(m_e, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_e.matrix_element_nml['c222'] = 0.75
    coulomb_e.matrix_element_nml['c002'] = 0.75
    coulomb_e.matrix_element_nml['c202'] = -0.25
    coulomb_e.matrix_element_nml['c022'] = -0.25
    coulomb_e.matrix_element_nml['c001'] = m_mcp**2 + m_e**2
    coulomb_e.matrix_element_nml['c201'] = m_mcp**2
    coulomb_e.matrix_element_nml['c021'] = m_e**2
    coulomb_e.matrix_element_nml['c000'] = 4*m_e**2*m_mcp**2
    
    coulomb_mu = elscat.ElasticCollisionIntegral(m_mu, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_mu.matrix_element_nml['c222'] = 0.75
    coulomb_mu.matrix_element_nml['c002'] = 0.75
    coulomb_mu.matrix_element_nml['c202'] = -0.25
    coulomb_mu.matrix_element_nml['c022'] = -0.25
    coulomb_mu.matrix_element_nml['c001'] = m_mcp**2 + m_mu**2
    coulomb_mu.matrix_element_nml['c201'] = m_mcp**2
    coulomb_mu.matrix_element_nml['c021'] = m_mu**2
    coulomb_mu.matrix_element_nml['c000'] = 4*m_mu**2*m_mcp**2

    coulomb_tau = elscat.ElasticCollisionIntegral(m_tau, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_tau.matrix_element_nml['c222'] = 0.75
    coulomb_tau.matrix_element_nml['c002'] = 0.75
    coulomb_tau.matrix_element_nml['c202'] = -0.25
    coulomb_tau.matrix_element_nml['c022'] = -0.25
    coulomb_tau.matrix_element_nml['c001'] = m_mcp**2 + m_tau**2
    coulomb_tau.matrix_element_nml['c201'] = m_mcp**2
    coulomb_tau.matrix_element_nml['c021'] = m_tau**2
    coulomb_tau.matrix_element_nml['c000'] = 4*m_tau**2*m_mcp**2

    coulomb_lq = elscat.ElasticCollisionIntegral(0.0, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_lq.matrix_element_nml['c222'] = 0.75
    coulomb_lq.matrix_element_nml['c002'] = 0.75
    coulomb_lq.matrix_element_nml['c202'] = -0.25
    coulomb_lq.matrix_element_nml['c022'] = -0.25
    coulomb_lq.matrix_element_nml['c001'] = m_mcp**2 
    coulomb_lq.matrix_element_nml['c201'] = m_mcp**2
    coulomb_lq.matrix_element_nml['c021'] = 0
    coulomb_lq.matrix_element_nml['c000'] = 0
    
    coulomb_strange = elscat.ElasticCollisionIntegral(m_s, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_strange.matrix_element_nml['c222'] = 0.75
    coulomb_strange.matrix_element_nml['c002'] = 0.75
    coulomb_strange.matrix_element_nml['c202'] = -0.25
    coulomb_strange.matrix_element_nml['c022'] = -0.25
    coulomb_strange.matrix_element_nml['c001'] = m_mcp**2 + m_s**2
    coulomb_strange.matrix_element_nml['c201'] = m_mcp**2
    coulomb_strange.matrix_element_nml['c021'] = m_s**2
    coulomb_strange.matrix_element_nml['c000'] = 4*m_s**2*m_mcp**2
    
    coulomb_charm = elscat.ElasticCollisionIntegral(m_c, m_mcp, mgam, zeta_a=1, zeta_b=1)
    coulomb_charm.matrix_element_nml['c222'] = 0.75
    coulomb_charm.matrix_element_nml['c002'] = 0.75
    coulomb_charm.matrix_element_nml['c202'] = -0.25
    coulomb_charm.matrix_element_nml['c022'] = -0.25
    coulomb_charm.matrix_element_nml['c001'] = m_mcp**2 + m_c**2
    coulomb_charm.matrix_element_nml['c201'] = m_mcp**2
    coulomb_charm.matrix_element_nml['c021'] = m_c**2
    coulomb_charm.matrix_element_nml['c000'] = 4*m_c**2*m_mcp**2
    
    return {
        "rate_e": coulomb_e, 
        "rate_mu": coulomb_mu, 
        "rate_tau": coulomb_tau, 
        "rate_lq": coulomb_lq, 
        "rate_strange": coulomb_strange,
        "rate_charm": coulomb_charm
        } 
    
def compute_coulomb_rate(temps, mass, n_strat, neval, nitn):
    '''
    Compute coulomb scattering collision integral with sm fermions for an mcp of mass M at millicharge = 1.0
    
    Parameters
    ----------
    mass - mass [MeV] of mcp
    temps - tuple of (T_sm, T_ds) [MeV, MeV]
    n_strat - number of stratifications for vegas integrator
    neval - number of evaluations for vegas integrator
    nitn - number of iterations for vegas integrator
    
    Returns
    -------
    (float) sum of collision integrals for coulomb scattering with each sm fermion
    '''
    
    warnings.filterwarnings('ignore')
    
    T_sm = temps[0]
    T_ds = temps[1]
    

    #setup up callable that computes photon plasma mass
    mgamma_thermal = lambda T_sm, T_ds: np.sqrt(plas.m_gam_2(T_sm))
    
    #setup coulomb integrals
    m_e = 0.511
    m_mu = 105
    m_tau = 1776
    m_s = 95
    m_c = 1270
    m_mcp = mass
    
    LQCD = 200
    
    time_start = time.time()
    rate_funs = setup_coulomb_integrals(mass, mgamma_thermal)
    rate_int_e = rate_funs['rate_e']
    rate_int_mu = rate_funs['rate_mu']
    rate_int_tau = rate_funs['rate_tau']
    rate_int_lq = rate_funs['rate_lq']
    rate_int_strange = rate_funs['rate_strange']
    rate_int_charm = rate_funs['rate_charm']

    
    
    result_e = 0.0
    result_mu = 0.0
    result_tau = 0.0
    result_lq = 0.0
    result_strange = 0.0
    result_charm = 0.0
            

    result_e = rate_int_e.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
    result_mu = rate_int_mu.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
    result_tau = rate_int_tau.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
    if T_sm > LQCD:
        result_lq = rate_int_lq.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        result_strange = rate_int_strange.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        result_charm = rate_int_charm.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
    #compute prefactors, dont include millicharge since all processes rescale with the millicharge
    alpha = 1.0/137.0
    e = np.sqrt(4*alpha*np.pi)
    
    q_u = 2/3
    q_d = -1/3
    q_s = -1/3
    q_c = 2/3   
    
    #factor to scale the coeeficients by
    pref_lept = 4*16*np.pi*e**4
    pref_lq = 4*3*16*np.pi*e**4*(q_u**2 + q_d**2)
    pref_strange = 4*3*16*np.pi*e**4*(q_s**2)
    pref_charm = 4*3*16*np.pi*e**4*(q_c**2)
    
    result_total = pref_lept*(result_e + result_mu + result_tau) + pref_lq*(result_lq) + pref_charm*result_charm + pref_strange*result_strange
    time_end = time.time()
    print(f'Time to compute integral: {(time_end - time_start)} seconds')
    
    return result_total
    
if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_coulomb.py',
                    description='tabulate coulomb scattering rates for MCPs')

    parser.add_argument('mass', action='store', type=float, help='mass in MeV of mcp')
    parser.add_argument('Tm_min', action='store', type=float, help='minimum T in units of mass to compute rates at')
    parser.add_argument('Tm_max', action='store', type=float, help='maximum T in units of mass to compute rates at')
    parser.add_argument('n_temps', action='store', type=int, help='number of temperatures in temperature grid')
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
        
    outfile_name = f'mcp_coulomb_rate_m_{args.mass}_Q_1.npz'
    outfile_path = os.path.join(args.outdir, outfile_name)
    
    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        exit()

    fun_loop = partial(
        compute_coulomb_rate, 
        mass=args.mass,
        n_strat=([3]+[3]), 
        neval=1e3, 
        nitn=10
    )
    
    T_min = args.mass*args.Tm_min
    T_max = args.mass*args.Tm_max
    Temp_grid = np.geomspace(T_min, T_max, args.n_temps)
    
    
    temp_grid_2d = []
    for Ti in Temp_grid:
        for Tj in Temp_grid:
            temp_grid_2d.append([Ti,Tj])
            


    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    
    time_start = time.time()
    result_total = pool.map(fun_loop, temp_grid_2d)
    time_end = time.time()

    pool.close()
    
    np.savez_compressed(outfile_path, n_temps = args.n_temps, temp_grid = temp_grid_2d, rate = result_total)  


    print(f'Time: {(time_end - time_start)/60:.5f} minutes')


