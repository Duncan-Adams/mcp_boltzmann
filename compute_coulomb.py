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
    
def compute_coulomb_rate(mass, Tm_min, Tm_max, n_temps, outdir, n_strat, neval, nitn):
    warnings.filterwarnings('ignore')
    T_min = mass*Tm_min
    T_max = mass*Tm_max
    Temp_grid = np.geomspace(T_min, T_max, n_temps)
    
    #setup up callable that computes photon plasma mass
    mgam_grid = np.sqrt(plas.m_gam_2(Temp_grid))
    mgam_interp = interp1d(Temp_grid, mgam_grid, bounds_error=False, fill_value=(0, mgam_grid[-1]))
    mgamma_thermal = lambda T_sm, T_ds: mgam_interp(T_sm)
    
    #setup coulomb integrals
    m_e = 0.511
    m_mu = 105
    m_tau = 1776
    m_s = 95
    m_c = 1270
    m_mcp = mass
    
    LQCD = 200
    
    rate_funs = setup_coulomb_integrals(mass, mgamma_thermal)
    rate_int_e = rate_funs['rate_e']
    rate_int_mu = rate_funs['rate_mu']
    rate_int_tau = rate_funs['rate_tau']
    rate_int_lq = rate_funs['rate_lq']
    rate_int_strange = rate_funs['rate_strange']
    rate_int_charm = rate_funs['rate_charm']
    
    temp_grid_2d = []
    for Ti in Temp_grid:
        for Tj in Temp_grid:
            temp_grid_2d.append([Ti,Tj])
            
    result_e = np.zeros(len(temp_grid_2d))
    result_mu = np.zeros(len(temp_grid_2d))
    result_tau = np.zeros(len(temp_grid_2d))
    result_lq = np.zeros(len(temp_grid_2d))
    result_strange = np.zeros(len(temp_grid_2d))
    result_charm = np.zeros(len(temp_grid_2d))
            
    for i in range(len(temp_grid_2d)):
        T_sm = temp_grid_2d[i][0]
        T_ds = temp_grid_2d[i][1]
        
        result_e[i] = rate_int_e.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        result_mu[i] = rate_int_mu.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        result_tau[i] = rate_int_tau.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
        if T_sm > LQCD:
            result_lq[i] = rate_int_lq.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            result_strange[i] = rate_int_strange.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            result_charm[i] = rate_int_charm.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
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
    name = f'mcp_coulomb_rate_m_{mass}_Q_1.npz'
    
    #save to a npz file
    np.savez_compressed(os.path.join(outdir,name), n_temps = n_temps, temp_grid = temp_grid_2d, rate = result_total)  
        
    
if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_coulomb.py',
                    description='tabulate coulomb scattering rates for MCPs')

    parser.add_argument('mass_list', action='store', type=str, help='file containing masses in MeV of milli-charged particle')
    parser.add_argument('Tm_min', action='store', type=float, help='minimum T in units of mass to compute rates at')
    parser.add_argument('Tm_max', action='store', type=float, help='maximum T in units of mass to compute rates at')
    parser.add_argument('n_temps', action='store', type=int, help='number of temperatures in temperature grid')
    parser.add_argument('--outdir', dest='outdir', action='store', default='./', type=str)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores)
    
    fun_loop = partial(compute_coulomb_rate, x_min=args.x_min, x_max=args.x_max, n_temps=args.n_temps, outdir=args.outdir, n_strat=([3]+[3]), neval=1e3, nitn=10)

    # ~ fun_loop = partial(compute_coulomb_rate, T_min=args.T_min, T_max=args.T_max, n_temps=args.n_temps, outdir=args.outdir, n_strat=([5]+[5]), neval=1e6, nitn=100)

    masses = np.loadtxt(args.mass_list)
    time_start = time.time()
    for p in pool.map(fun_loop, np.atleast_1d(masses)):
        pass
    pool.close()

    time_end = time.time()
    print(f'Time: {(time_end - time_start)/60:.5f} minutes')


