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
from mcp_boltzmann.boltzmann import Boltzmann


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
    
    result_dir = os.path.join(outdir, f'm_{m_mcp}_Q_{Q}/')
    
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
    
    Boltz, sol_sm, sol_bsm = result
    
    N_eff_bsm = Boltz.N_eff(sol_bsm.y[0][-1], sol_bsm.y[1][-1], sol_bsm.y[2][-1])
    N_eff_sm = Boltz.N_eff_SM(sol_sm.y[0][-1], sol_sm.y[1][-1])
    Delta_Neff = N_eff_bsm - N_eff_sm 
    
    DNeff_dso = Boltz.Delta_Neff_ds_only(sol_bsm.y[0][-1], sol_bsm.y[2][-1], m_mcp)
    
    with open(os.path.join(result_dir, 'result.txt'), 'w') as out_txt:
        print(f'{m_mcp=}', file=out_txt)
        print(f'{Q=}', file=out_txt)
        print(f'{N_eff_sm=}', file=out_txt)
        print(f'{N_eff_bsm=}', file=out_txt)
        print(f'{DNeff_dso=}', file=out_txt)
        print(f'{Delta_Neff=}', file=out_txt)
        
    if do_plots is True:
        plt.plot(sol_sm.t, sol_sm.y[0], label=r'T$_\gamma$')
        plt.plot(sol_sm.t, sol_sm.y[1], label=r'T$_\nu$')
        plt.yscale('log')
        plt.xscale('log')
        
        plt.xlabel(r'Time [MeV$^{-1}$]')
        plt.ylabel(r'Temperature [MeV]')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'temp_sm.png'))
        plt.cla()
        ################################################################################################################
        
        plt.plot(sol_bsm.t, sol_bsm.y[0], label=r'T$_\gamma$')
        plt.plot(sol_bsm.t, sol_bsm.y[1], label=r'T$_{\nu}$')
        plt.plot(sol_bsm.t, sol_bsm.y[2], label=r'T$_{\rm ds}$')
        
        plt.plot(sol_bsm.t, sol_bsm.y[0][0]*np.sqrt(sol_bsm.t[0]/sol_bsm.t), linestyle='dashed', color='black', alpha=0.5)
        plt.plot(sol_bsm.t, sol_bsm.y[1][-1]*np.sqrt(sol_bsm.t[-1]/sol_bsm.t), linestyle='dashed', color='black', alpha=0.5)
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r'Time [MeV$^{-1}$]')
        plt.ylabel(r'Temperature [MeV]')
        
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'temp_bsm.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(sol_bsm.t, sol_bsm.y[1]/sol_bsm.y[0], label=r'T$_\nu$/T$_\gamma$')
        plt.plot(sol_bsm.t, sol_bsm.y[2]/sol_bsm.y[0], label=r'T$_{\rm dark}$/T$_\gamma$')
        plt.xscale('log')
        
        plt.xlabel('Time [GeV$^{-1}$]')
        plt.title('Temperature Ratio')
        plt.legend()
        
        plt.savefig(os.path.join(result_dir, 'temp_ratio.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(sol_bsm.y[0], Boltz.colterms_EM_DS[0](sol_bsm.y[0], sol_bsm.y[2], Q)/sol_bsm.y[0]**6, label='annihilation')
        plt.plot(sol_bsm.y[0], Boltz.colterms_EM_DS[1](sol_bsm.y[0], sol_bsm.y[2], Q)/sol_bsm.y[0]**6, label='scattering')
        plt.plot(sol_bsm.y[0], Boltz.colterms_EM_DS[2](sol_bsm.y[0], sol_bsm.y[2], Q)/sol_bsm.y[0]**6, label='Plasmon Decay')
        plt.plot(sol_bsm.y[0], Boltz.colterms_EM_DS[3](sol_bsm.y[0], sol_bsm.y[2], Q)/sol_bsm.y[0]**6, label='Z decay')
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
        
        
def load_ann_rate(path):
    with np.load(path) as rate_file:
        return interp1d(rate_file['Temp_grid'], rate_file['rate'], bounds_error=False, fill_value=0)

def compute_neff(m_mcp, Q):
    MeV = 1
    GeV = 1e3
    
    #load energy transfter rates
    _CF_ff_xx_I = load_ann_rate(f'./output/rates/annihilation/ann_m_{m_mcp}_Q_1.npz')
    mcp_coulomb_rate = elscat.load_tabulated_rate(f'./output/rates/coulomb/mcp_coulomb_rate_m_{m_mcp}_Q_1.npz')
    
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
        
    Boltz = Boltzmann(m_mcp)
    Boltz.add_colterm_EM_DS(CF_ann_sm_ds)
    Boltz.add_colterm_EM_DS(CF_scatt_sm_ds)
    Boltz.add_colterm_EM_DS(CF_plas)
    Boltz.add_colterm_EM_DS(CF_Z_decay)
    
    #initial conditions
    T_gamma_0 = 1e6 * MeV
    T_nu_0 = T_gamma_0
    T_DS_0 = 1e-4*T_gamma_0
    
    sol_sm = Boltz.solve_boltzmann_eq_SM(T_gamma_0, T_nu_0)
    sol_bsm = Boltz.solve_boltzmann_eq(T_gamma_0, T_nu_0, T_DS_0, Q)
    
    #should return the boltzman solver instance and the sm and bsm solutions
    return Boltz, sol_sm, sol_bsm

if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_coulomb.py',
                    description='tabulate coulomb scattering rates for MCPs')

    parser.add_argument('model_params', action='store', type=str, help='file containing masses in MeV of milli-charged particle, and millicharges')
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
    param_points = np.genfromtxt(args.model_params, delimiter=',')
    tasks = [dict()]*len(param_points)
    
    for (i, p) in enumerate(param_points):
        td = {
            'mass': p[0],
            'Q': p[1],
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


