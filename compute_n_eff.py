import os
import argparse
import time
import warnings
import traceback

import numpy as np
import matplotlib.pyplot as plt
import schwimmbad
from scipy.interpolate import interp1d

import mcp_boltzmann.elastic_scattering as elscat
import mcp_boltzmann.plasma as plas
from mcp_boltzmann import annihilation as ann
from mcp_boltzmann.boltzmann import MCPBoltzmann


def worker(task):
    m_mcp = task['mass']
    Q = task['Q']
    outdir = task['outdir']
    do_plots = task['do_plots']
    
    try:
        res = compute_neff(m_mcp, Q)
        
    except Exception as e:
        print(traceback.format_exc())
        return None
        
    return (task, res)
        
def save_results(task_result):
    if task_result is None:
        return
    
    task, result = task_result
    
    m_mcp = task['mass']
    Q = task['Q']
    outdir = task['outdir']
    do_plots = task['do_plots']
    
    result_dir = os.path.join(outdir, f'm_{m_mcp}/Q_{Q:.3e}/')
    
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
    
    Boltz, sol_sm, sol_bsm = result
    
    time_sm = sol_sm[0]
    T_gam_sm = sol_sm[1]
    T_nu_sm = sol_sm[2]
    sf_sm = sol_sm[3]
    
    time_bsm = sol_bsm[0]
    T_gam_bsm = sol_bsm[1]
    T_nu_bsm = sol_bsm[2]
    T_dark_bsm = sol_bsm[3]
    sf_bsm = sol_bsm[4]
    
    N_eff_bsm = Boltz.N_eff(T_gam_sm[-1], T_nu_sm[-1], T_dark_bsm[-1])
    N_eff_sm = Boltz.N_eff_SM(T_gam_sm[-1], T_nu_sm[-1])
    Delta_Neff = N_eff_bsm - N_eff_sm 
    
    DNeff_dso = Boltz.Delta_Neff_ds_only(T_gam_sm[-1], T_dark_bsm[-1], m_mcp)
    
    with open(os.path.join(result_dir, 'result.txt'), 'w') as out_txt:
        print(f'{m_mcp=}', file=out_txt)
        print(f'{Q=}', file=out_txt)
        print(f'{N_eff_sm=}', file=out_txt)
        print(f'{N_eff_bsm=}', file=out_txt)
        print(f'{DNeff_dso=}', file=out_txt)
        print(f'{Delta_Neff=}', file=out_txt)
        
    np.savez_compressed(
        os.path.join(result_dir, 'result.npz'),
        m_mcp = m_mcp,
        Q = Q,
        N_eff_sm = N_eff_sm,
        N_eff_bsm = N_eff_bsm[0],
        DNeff_dso = DNeff_dso[0],
        Delta_Neff = Delta_Neff[0]
        )
        
    if do_plots is True:
        plt.plot(time_sm, T_gam_sm, label=r'T$_\gamma$')
        plt.plot(time_sm, T_nu_sm, label=r'T$_\nu$')
        plt.yscale('log')
        plt.xscale('log')
        
        plt.xlabel(r'Time [MeV$^{-1}$]')
        plt.ylabel(r'Temperature [MeV]')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'temp_sm.png'))
        plt.cla()
        ################################################################################################################
        
        plt.plot(time_bsm, T_gam_bsm, label=r'T$_\gamma$')
        plt.plot(time_bsm, T_nu_bsm, label=r'T$_{\nu}$')
        plt.plot(time_bsm, T_dark_bsm, label=r'T$_{\rm ds}$')
        
        plt.plot(time_bsm, T_gam_bsm[0]*np.sqrt(time_bsm[0]/time_bsm), linestyle='dashed', color='black', alpha=0.5)
        plt.plot(time_bsm, T_nu_bsm[-1]*np.sqrt(time_bsm[-1]/time_bsm), linestyle='dashed', color='black', alpha=0.5)
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'Time [MeV$^{-1}$]')
        plt.ylabel(r'Temperature [MeV]')
        
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'temp_bsm.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(time_bsm, T_nu_bsm/T_gam_bsm, label=r'T$_\nu$/T$_\gamma$')
        plt.plot(time_bsm, T_dark_bsm/T_gam_bsm, label=r'T$_{\rm dark}$/T$_\gamma$')
        plt.axhline(T_nu_sm[-1]/T_gam_sm[-1], color='black', linestyle='dashed', alpha=0.7)
        plt.xscale('log')
        
        plt.xlabel('Time [MeV$^{-1}$]')
        plt.title('Temperature Ratio')
        plt.legend()
        
        plt.savefig(os.path.join(result_dir, 'temp_ratio_time.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(T_gam_bsm, T_nu_bsm/T_gam_bsm, label=r'T$_\nu$/T$_\gamma$')
        plt.plot(T_gam_bsm, T_dark_bsm/T_gam_bsm, label=r'T$_{\rm dark}$/T$_\gamma$')
        plt.axhline(T_nu_sm[-1]/T_gam_sm[-1], color='black', linestyle='dashed', alpha=0.7)
        plt.axhline(1.0, color='black', alpha=0.3)
        plt.xscale('log')
        
        plt.xlabel(r'T$_\gamma$ [MeV]')
        plt.title('Temperature Ratio')
        plt.gca().invert_xaxis()
        plt.legend()
        
        plt.savefig(os.path.join(result_dir, 'temp_ratio.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[0](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='annihilation')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[1](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='scattering')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[2](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='Plasmon Decay')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[3](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='Z decay')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(1e-27, 1e-22)
        plt.gca().invert_xaxis()
        
        plt.xlabel('Time [MeV$^{-1}$]')
        plt.title(r'C/$T^6_\gamma$ [MeV$^{-1}$]')
        plt.legend()
        
        plt.savefig(os.path.join(result_dir, 'rates.png'))
        plt.cla()
        
    return 
        
def compute_neff(m_mcp, Q):
    MeV = 1
    GeV = 1e3
    
    #load energy transfter rates
    _CF_ff_xx_I = ann.load_ann_rate(f'./output/rates/annihilation/mcp_annihilation_rate_m_{m_mcp}_Q_1.npz')
    mcp_coulomb_rate = elscat.load_tabulated_rate(f'./output/rates/coulomb/cluster/mcp_coulomb_rate_m_{m_mcp}_Q_1.npz')
    
    def CF_ann(T, Q):
        return Q**2*_CF_ff_xx_I(T)

    def CF_ann_sm_ds(T_sm, T_ds, Q):
        return CF_ann(T_sm, Q) - CF_ann(T_ds, Q)
    
    def CF_scatt_sm_ds(T_sm, T_ds, Q):
        return Q**2*mcp_coulomb_rate(T_sm, T_ds)

    def CF_plas(T_sm, T_ds, Q):
        T_EW = 160*GeV
        pdecay = np.heaviside(T_EW - T_sm, 0)*plas.C_plasmon(T_sm, T_ds, m_mcp, Q) 
        bdecay = np.heaviside(T_sm - T_EW, 1)*plas.C_B_decay(T_sm, T_ds, m_mcp, Q)
        return pdecay + bdecay
        
    def CF_Z_decay(T_sm, T_ds, Q):
        return plas.C_Z_decay(T_sm, T_ds, m_mcp, Q)
        
    Boltz = MCPBoltzmann(m_mcp, Q)
    Boltz.add_colterm_EM_DS(CF_ann_sm_ds)
    Boltz.add_colterm_EM_DS(CF_scatt_sm_ds)
    Boltz.add_colterm_EM_DS(CF_plas)
    Boltz.add_colterm_EM_DS(CF_Z_decay)
    
    #initial conditions
    T_gamma_0 = 100*m_mcp
    T_nu_0 = T_gamma_0
    # ~ T_DS_0 = 1e-4*T_gamma_0 #need to calculate this more inelligently
    T_DS_0 = Boltz.guess_initial_dark_temp(T_gamma_0)
    # ~ print(Boltz.guess_initial_dark_temp(T_gamma_0))
    
    sol_sm = Boltz.solve_boltzmann_eq_SM(T_gamma_0, T_nu_0)
    sol_bsm = Boltz.solve_boltzmann_eq(T_gamma_0, T_nu_0, T_DS_0)
    
    #should return the boltzman solver instance and the sm and bsm solutions
    return Boltz, sol_sm, sol_bsm

if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_coulomb.py',
                    description='tabulate coulomb scattering rates for MCPs')

    parser.add_argument('mass', action='store', type=float, help='mass of millicharge particle in MeV')
    parser.add_argument('Q_min', action='store', type=float, help='minimum value of millicharge to scan over')
    parser.add_argument('Q_max', action='store', type=float, help='maximum value of millicharge to scan over')
    parser.add_argument('num_Q', action='store', type=int, help='number of geometrically spaced millicharges in scan')
    parser.add_argument('--save_plots', dest='do_plots', action='store_true')
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
    
    # read in list of models
    Q_range = np.geomspace(args.Q_min, args.Q_max, args.num_Q)
    tasks = [dict()]*len(Q_range)
    
    for (i, Q) in enumerate(Q_range):
        td = {
            'mass': args.mass,
            'Q': Q,
            'outdir': args.outdir,
            'do_plots': args.do_plots
        }
        tasks[i] = td
        
    
    time_start = time.time()
    for p in pool.map(worker, tasks, callback=save_results):
        pass
    pool.close()

    time_end = time.time()
    print(f'Time: {(time_end - time_start)/60:.5f} minutes')


