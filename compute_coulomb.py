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

def _setup_int_ff(m_mcp, m_f, mgam):
    elcol = elscat.ElasticCollisionIntegral(m_f, m_mcp, mgam, zeta_a=1, zeta_b=1)
    elcol.matrix_element_nml['c222'] = 0.75
    elcol.matrix_element_nml['c202'] = -0.25
    elcol.matrix_element_nml['c022'] = -0.25
    elcol.matrix_element_nml['c002'] = 0.75
    elcol.matrix_element_nml['c001'] = m_mcp**2 + m_f**2
    elcol.matrix_element_nml['c201'] = m_mcp**2
    elcol.matrix_element_nml['c021'] = m_f**2
    elcol.matrix_element_nml['c000'] = 4*m_f**2*m_mcp**2
    
    return elcol
    
def _setup_int_hh(m_mcp, m_B):
    elcol = elscat.ElasticCollisionIntegral(0.0, m_mcp, m_B, zeta_a=-1, zeta_b=1)
    elcol.matrix_element_nml['c222'] = 0.75
    elcol.matrix_element_nml['c202'] = -0.25
    elcol.matrix_element_nml['c022'] = -0.25
    elcol.matrix_element_nml['c002'] = -0.25
    elcol.matrix_element_nml['c201'] = m_mcp**2
    elcol.matrix_element_nml['c001'] = -m_mcp**2
    
    return elcol

def setup_coulomb_integrals(m_mcp, mgam):
    m_e = 0.511
    m_mu = 105
    m_tau = 1776
    m_s = 95
    m_c = 1270
    m_b = 4180
    m_t = 172.76*1e3
    
    
    coulomb_e = _setup_int_ff(m_mcp, m_e, mgam)
    coulomb_mu = _setup_int_ff(m_mcp, m_mu, mgam)
    coulomb_tau = _setup_int_ff(m_mcp, m_tau, mgam)   
    
    coulomb_lq = _setup_int_ff(m_mcp, 0.0, mgam)   
    coulomb_strange = _setup_int_ff(m_mcp, m_s, mgam)   
    coulomb_charm = _setup_int_ff(m_mcp, m_c, mgam)  
    coulomb_bottom = _setup_int_ff(m_mcp, m_b, mgam)  
    coulomb_top = _setup_int_ff(m_mcp, m_t, mgam)
    
    coulomb_higgs = _setup_int_hh(m_mcp, mgam)
    
    return {
        "rate_e": coulomb_e, 
        "rate_mu": coulomb_mu, 
        "rate_tau": coulomb_tau, 
        "rate_lq": coulomb_lq, 
        "rate_strange": coulomb_strange,
        "rate_charm": coulomb_charm,
        "rate_bottom": coulomb_bottom,
        "rate_top": coulomb_top,
        "rate_higgs": coulomb_higgs
        } 
    
def compute_coulomb_rate(temps, m_mcp, n_strat, neval, nitn):
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
    
    
    #setup coulomb integrals
    m_e = 0.511
    m_mu = 105
    m_tau = 1776
    m_s = 95
    m_c = 1270
    m_b = 4180
    m_t = 172.76*1e3
    
    LQCD = 200
    T_EW = 160*1e3
    
    #setup up callable that computes photon plasma mass
    electroweak = (T_sm > T_EW)
    
    if electroweak:
        mgamma_thermal = lambda T_sm, T_ds: np.sqrt(plas.m_B_2(T_sm))
    else:
        mgamma_thermal = lambda T_sm, T_ds: np.sqrt(plas.m_gam_2(T_sm))
    
    rate_funs = setup_coulomb_integrals(m_mcp, mgamma_thermal)
    rate_int_e = rate_funs['rate_e']
    rate_int_mu = rate_funs['rate_mu']
    rate_int_tau = rate_funs['rate_tau']
    rate_int_lq = rate_funs['rate_lq']
    rate_int_strange = rate_funs['rate_strange']
    rate_int_charm = rate_funs['rate_charm']
    rate_int_bot = rate_funs['rate_bottom']
    rate_int_top = rate_funs['rate_top']
    rate_int_higgs = rate_funs['rate_higgs']
    
    result_e = 0.0
    result_mu = 0.0
    result_tau = 0.0
    result_lq = 0.0
    result_strange = 0.0
    result_charm = 0.0
    result_bot = 0.0
    result_top = 0.0
    result_higgs = 0.0

    result_e = rate_int_e.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
    
    if electroweak:
        result_higgs = rate_int_higgs.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
    
    if T_sm > m_mu/30.0:
        if T_sm > 5*m_mu:
            result_mu = result_e
        else:
            result_mu = rate_int_mu.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
    if T_sm > m_tau/30.0:
        if T_sm > 5*m_tau:
            result_tau = result_e
        else:
            result_tau = rate_int_tau.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
    if T_sm > LQCD:
        result_lq = rate_int_lq.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
        if T_sm > 5*m_s:
            result_strange = result_lq
        else:
            result_strange = rate_int_strange.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
        if T_sm > 5*m_c:
            result_charm = result_lq
        else:
            result_charm = rate_int_charm.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
        if T_sm > m_b/30.0:
            if T_sm > 5*m_b:
                result_bot = result_lq
            else:
                result_bot = rate_int_bot.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
        if T_sm > m_t/30.0:
            if T_sm > 5*m_t:
                result_top = result_lq
            else:
                result_top = rate_int_top.compute_QS(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
            
    #compute prefactors, dont include millicharge since all processes rescale with the millicharge
    alpha = 1.0/137.0
    e = np.sqrt(4*alpha*np.pi)
    
    s2_theta_w = 0.22339 
    c2_theta_w = 1-s2_theta_w
    
    q_e = -1
    q_u = 2/3
    q_d = -1/3
    q_s = -1/3
    q_c = 2/3   
    q_b = -1/3
    q_t = 2/3
    
    #hyper charges, using convention that Q = T3 + Y 
    Y_u_l = 1/6
    Y_u_r = 2/3
    Y_d_l = 1/6
    Y_d_r = -1/3
    Y_H = 1/2
    
    Y_e_l = -1/2
    Y_nu_L = -1/2
    Y_e_R = -1
    
    #factor to scale the coeeficients by
    pref_lept = 4*16*np.pi*e**4
    pref_lq = 4*3*16*np.pi*e**4*(q_u**2 + q_d**2)
    pref_strange = 4*3*16*np.pi*e**4*(q_s**2)
    pref_charm = 4*3*16*np.pi*e**4*(q_c**2)
    pref_bot = 4*3*16*np.pi*e**4*(q_b**2)
    pref_top = 4*3*16*np.pi*e**4*(q_t**2)
    pref_higgs = 0.0
    
    if electroweak:
        #need to sum over each weyl fermion now
        pref_e_l = Y_e_l**2/(2*q_e**2*c2_theta_w**2)
        pref_e_r = Y_e_R**2/(2*q_e**2*c2_theta_w**2)
        
        pref_u_l = Y_u_l**2/(2*q_u**2*c2_theta_w**2)
        pref_u_r = Y_u_r**2/(2*q_u**2*c2_theta_w**2)
        
        pref_d_l = Y_d_l**2/(2*q_d**2*c2_theta_w**2)
        pref_d_r = Y_d_r**2/(2*q_d**2*c2_theta_w**2)
        
        pref_higgs = 4*np.pi*(e**4/c2_theta_w**2)
        
        pref_lept = 2*pref_e_l + 2*pref_e_r 
        pref_u = 2*pref_u_l + 2*pref_u_r
        pref_d = 2*pref_d_l + 2*pref_d_r
        
        pref_lq = 3*(pref_u + pref_d)
        pref_charm = 3*pref_u
        pref_strange = 3*pref_d
        pref_top = 3*pref_u
        pref_bot = 3*pref_d
        
    
    result_total = (
        pref_lept*(result_e + result_mu + result_tau) 
      + pref_lq*(result_lq) 
      + pref_charm*result_charm 
      + pref_strange*result_strange
      + pref_bot*result_bot
      + pref_top*result_top
      + pref_higgs*result_higgs
    )
    
    return result_total


def compute_coulomb_rate_forwards(temps, m_mcp, n_strat, neval, nitn):
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
    m_b = 4180
    m_t = 172.76*1e3
    
    
    LQCD = 200
    
    rate_funs = setup_coulomb_integrals(m_mcp, mgamma_thermal)
    rate_int_e = rate_funs['rate_e']
    rate_int_mu = rate_funs['rate_mu']
    rate_int_tau = rate_funs['rate_tau']
    rate_int_lq = rate_funs['rate_lq']
    rate_int_strange = rate_funs['rate_strange']
    rate_int_charm = rate_funs['rate_charm']
    rate_int_bot = rate_funs['rate_bottom']
    rate_int_top = rate_funs['rate_top']

    
    result_e = 0.0
    result_mu = 0.0
    result_tau = 0.0
    result_lq = 0.0
    result_strange = 0.0
    result_charm = 0.0
    result_bot = 0.0
    result_top = 0.0
            

    result_e = rate_int_e.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
    
    if T_sm > m_mu/30.0:
        if T_sm > 5*m_mu:
            result_mu = result_e
        else:
            result_mu = rate_int_mu.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
    if T_sm > m_tau/30.0:
        if T_sm > 5*m_tau:
            result_tau = result_e
        else:
            result_tau = rate_int_tau.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
    if T_sm > LQCD:
        result_lq = rate_int_lq.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
        if T_sm > 5*m_s:
            result_strange = result_lq
        else:
            result_strange = rate_int_strange.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
        if T_sm > 5*m_c:
            result_charm = result_lq
        else:
            result_charm = rate_int_charm.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
        if T_sm > m_b/30.0:
            if T_sm > 5*m_b:
                result_bot = result_lq
            else:
                result_bot = rate_int_bot.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
            
        if T_sm > m_t/30.0:
            if T_sm > 5*m_t:
                result_top = result_lq
            else:
                result_top = rate_int_top.compute_QS_forwards(T_sm, T_ds, n_strat=n_strat, neval=neval, nitn=nitn)[0]
        
            
    #compute prefactors, dont include millicharge since all processes rescale with the millicharge
    alpha = 1.0/137.0
    e = np.sqrt(4*alpha*np.pi)
    
    q_u = 2/3
    q_d = -1/3
    q_s = -1/3
    q_c = 2/3   
    q_b = -1/3
    q_t = 2/3
    
    
    #factor to scale the coeeficients by
    pref_lept = 4*16*np.pi*e**4
    pref_lq = 4*3*16*np.pi*e**4*(q_u**2 + q_d**2)
    pref_strange = 4*3*16*np.pi*e**4*(q_s**2)
    pref_charm = 4*3*16*np.pi*e**4*(q_c**2)
    pref_bot = 4*3*16*np.pi*e**4*(q_b**2)
    pref_top = 4*3*16*np.pi*e**4*(q_t**2)
    
    result_total = (
        pref_lept*(result_e + result_mu + result_tau) 
      + pref_lq*(result_lq) 
      + pref_charm*result_charm 
      + pref_strange*result_strange
      + pref_bot*result_bot
      + pref_top*result_top
    )

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
    parser.add_argument('--forwards', action='store_true', help='compute forwards rate only')
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
    
    if args.forwards is True:
        outfile_name = f'mcp_coulomb_rate_m_{args.mass}_Q_1_F.npz'
    outfile_path = os.path.join(args.outdir, outfile_name)
    

    if (args.overwrite is False) and os.path.exists(outfile_path):
        print(f'{outfile_path} already exists . . . skipping')
        exit()

    fun_loop = partial(
        compute_coulomb_rate, 
        m_mcp=args.mass,
        n_strat=([3]+[3]), 
        neval=1e3, 
        nitn=10
    )
    
    if args.forwards is True:
        fun_loop = partial(
            compute_coulomb_rate_forwards, 
            m_mcp=args.mass,
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


