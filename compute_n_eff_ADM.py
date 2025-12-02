import os
import argparse
import time
import warnings
import traceback

import numpy as np
import matplotlib.pyplot as plt
import schwimmbad
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')

import mcp_boltzmann.elastic_scattering as elscat
import mcp_boltzmann.plasma as plas
from mcp_boltzmann import annihilation as ann
from mcp_boltzmann.boltzmann import ADMBoltzmann


def worker(task):
    m_de = task['m_de']
    m_dp = task['m_dp']
    Q = task['Q']
    outdir = task['outdir']
    do_plots = task['do_plots']
    overwrite = task['overwrite']
    
    result_dir = os.path.join(outdir, f'mde_{m_de}/mdp_{m_dp}/')
    result_fname = f'result_Q_{Q:.3e}.npz'
    
    if (overwrite is False) and os.path.exists(os.path.join(result_dir, result_fname)):
        print(f'path: {result_fname} exists and overwrite is false... skipping')
        return None 
        
    try:
        res = compute_neff(m_de, m_dp, Q)
        
    except Exception as e:
        print(traceback.format_exc())
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f'FAILED_Q_{Q:.3e}.txt'), 'w') as out_txt:
            print(f'{m_de=}', file=out_txt)
            print(f'{m_dp=}', file=out_txt)
            print(f'{Q=}', file=out_txt)
        return None
        
    return (task, res)
        
def save_results(task_result):
    if task_result is None:
        return
    
    task, result = task_result
    
    m_de = task['m_de']
    m_dp = task['m_dp']
    Q = task['Q']
    outdir = task['outdir']
    do_plots = task['do_plots']
    
    result_dir = os.path.join(outdir, f'mde_{m_de}/mdp_{m_dp}/')
    plot_dir = os.path.join(outdir, f'mde_{m_de}/mdp_{m_dp}/Q_{Q:.3e}/')
    
    if (not os.path.exists(result_dir)):
        os.makedirs(result_dir, exist_ok=True)
        
    if (not os.path.exists(plot_dir)) and do_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    Boltz, sol_sm, sol_bsm, wall_time = result
    
    time_sm = sol_sm[0]
    T_gam_sm = sol_sm[1]
    T_nu_sm = sol_sm[2]
    sf_sm = sol_sm[3]
    
    time_bsm = sol_bsm[0]
    T_gam_bsm = sol_bsm[1]
    T_nu_bsm = sol_bsm[2]
    T_dark_bsm = sol_bsm[3]
    sf_bsm = sol_bsm[4]
    
    N_eff_bsm = Boltz.N_eff(T_gam_bsm[-1], T_nu_bsm[-1], T_dark_bsm[-1])
    N_eff_sm = Boltz.N_eff_SM(T_gam_sm[-1], T_nu_sm[-1])
    Delta_Neff = N_eff_bsm - N_eff_sm 
    
    DNeff_dso = Boltz.Delta_Neff_ds_only(T_gam_bsm[-1], T_dark_bsm[-1])
    
    with open(os.path.join(result_dir, f'result_Q_{Q:.3e}.txt'), 'w') as out_txt:
        print(f'{m_de=}', file=out_txt)
        print(f'{m_dp=}', file=out_txt)
        print(f'{Q=}', file=out_txt)
        print(f'{N_eff_sm=}', file=out_txt)
        print(f'{N_eff_bsm=}', file=out_txt)
        print(f'{DNeff_dso=}', file=out_txt)
        print(f'{Delta_Neff=}', file=out_txt)
        print(f'{T_nu_sm[-1]/T_gam_sm[-1]=}', file=out_txt)
        print(f'{T_nu_bsm[-1]/T_gam_bsm[-1]=}', file=out_txt)
        print(f'{T_dark_bsm[-1]/T_gam_bsm[-1]=}', file=out_txt)
        print(f'{T_gam_sm[-1]/T_gam_bsm[-1]=}', file=out_txt)
        print('', file=out_txt)
        print(f'wall time: {wall_time} seconds', file=out_txt)
        
    np.savez_compressed(
        os.path.join(result_dir, f'result_Q_{Q:.3e}.npz'),
        m_de = m_de,
        m_dp = m_dp,
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
        plt.savefig(os.path.join(plot_dir, 'temp_sm.png'))
        plt.cla()
        ################################################################################################################
        T_gam_SM_I = interp1d(time_sm, T_gam_sm)
        T_gam_BSM_I = interp1d(time_bsm, T_gam_bsm)
        
        plt.plot(time_sm, T_gam_BSM_I(time_sm)/T_gam_SM_I(time_sm))
        plt.ylim(1e-2, 2)
        
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('Time [MeV$^{-1}$]')

        plt.savefig(os.path.join(plot_dir, 'tgam_compare.png'))
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
        plt.savefig(os.path.join(plot_dir, 'temp_bsm.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(time_bsm, T_nu_bsm/T_gam_bsm, label=r'T$_\nu$/T$_\gamma$')
        plt.plot(time_bsm, T_dark_bsm/T_gam_bsm, label=r'T$_{\rm dark}$/T$_\gamma$')
        plt.axhline(T_nu_sm[-1]/T_gam_sm[-1], color='black', linestyle='dashed', alpha=0.7)
        plt.xscale('log')
        
        plt.xlabel('Time [MeV$^{-1}$]')
        plt.title('Temperature Ratio')
        plt.legend()
        
        plt.savefig(os.path.join(plot_dir, 'temp_ratio_time.png'))
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
        
        plt.savefig(os.path.join(plot_dir, 'temp_ratio.png'))
        plt.cla()
        ################################################################################################################
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[0](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='annihilation')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[1](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='scattering')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[2](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='Plasmon Decay')
        plt.plot(T_gam_bsm, Boltz.colterms_EM_DS[3](T_gam_bsm, T_dark_bsm, Q)/T_gam_bsm**6, label='Z decay')
        plt.xscale('log')
        plt.yscale('log')
        
        sum_rate = (
            Boltz.colterms_EM_DS[0](T_gam_bsm, T_dark_bsm, Q)
            +Boltz.colterms_EM_DS[1](T_gam_bsm, T_dark_bsm, Q)
            +Boltz.colterms_EM_DS[2](T_gam_bsm, T_dark_bsm, Q)
            +Boltz.colterms_EM_DS[3](T_gam_bsm, T_dark_bsm, Q)
        )
        
        ymax = np.max(sum_rate/T_gam_bsm**6)
        plt.ylim(1e-5*ymax, 20*ymax)
        
        plt.axvline(m_de, label='Dark Electron', linestyle='dashed', color='darkred')
        plt.axvline(m_dp, label='Dark Proton', linestyle='dashed', color='darkblue')

        plt.gca().invert_xaxis()
        
        plt.xlabel(r'T$_\gamma$ [MeV]')
        plt.title(r'C/$T^6_\gamma$ [MeV$^{-1}$]')
        plt.legend()
        
        plt.savefig(os.path.join(plot_dir, 'rates.png'))
        plt.cla()
        
    return 
        
def compute_neff(m_de, m_dp, Q):
    time_start = time.time()
    MeV = 1
    GeV = 1e3
    
    _CF_ff_de_de_I = ann.load_ann_rate(
        f'./output/rates/annihilation/scan/mcp_annihilation_rate_m_{m_de}_Q_1.npz'
    )
    
    _CF_ff_dp_dp_I = ann.load_ann_rate(
        f'./output/rates/annihilation/scan/mcp_annihilation_rate_m_{m_dp}_Q_1.npz'
    )
        
    mcp_coulomb_rate_de = elscat.load_tabulated_rate(f'./output/rates/coulomb/cluster/scan/mcp_coulomb_rate_m_{m_de}_Q_1.npz')
    mcp_coulomb_rate_dp = elscat.load_tabulated_rate(f'./output/rates/coulomb/cluster/scan/mcp_coulomb_rate_m_{m_dp}_Q_1.npz')
    
    def CF_ann(T, Q):
        return Q**2*(_CF_ff_de_de_I(T) + _CF_ff_dp_dp_I(T))

    def CF_ann_sm_ds(T_sm, T_ds, Q):
        return CF_ann(T_sm, Q) - CF_ann(T_ds, Q)
    
    def CF_scatt_sm_ds(T_sm, T_ds, Q):
        return Q**2*(mcp_coulomb_rate_de(T_sm, T_ds) + mcp_coulomb_rate_dp(T_sm, T_ds))

    def CF_plas(T_sm, T_ds, Q):
        T_EW = 160*GeV
        
        pdecay_de = np.heaviside(T_EW - T_sm, 0)*plas.C_plasmon(T_sm, T_ds, m_de, Q) 
        bdecay_de = np.heaviside(T_sm - T_EW, 1)*plas.C_B_decay(T_sm, T_ds, m_de, Q)
    
        pdecay_dp = np.heaviside(T_EW - T_sm, 0)*plas.C_plasmon(T_sm, T_ds, m_dp, Q) 
        bdecay_dp = np.heaviside(T_sm - T_EW, 1)*plas.C_B_decay(T_sm, T_ds, m_dp, Q)
        
        return pdecay_de + bdecay_de + pdecay_dp + bdecay_dp
        
    def CF_Z_decay(T_sm, T_ds, Q):
        return plas.C_Z_decay(T_sm, T_ds, m_de, Q) + plas.C_Z_decay(T_sm, T_ds, m_dp, Q) 
        
    Boltz = ADMBoltzmann(m_de, m_dp, Q, rtol=1e-8, atol=1e-8)
    Boltz.add_colterm_EM_DS(CF_ann_sm_ds)
    Boltz.add_colterm_EM_DS(CF_scatt_sm_ds)
    Boltz.add_colterm_EM_DS(CF_plas)
    Boltz.add_colterm_EM_DS(CF_Z_decay)
    
    #initial conditions
    T_gamma_0 = 100*max(m_de, m_dp)
    T_nu_0 = T_gamma_0
    T_DS_0 = 1e-2*Boltz.guess_initial_dark_temp(T_gamma_0)
    
    sol_sm = Boltz.solve_boltzmann_eq_SM(T_gamma_0, T_nu_0)
    sol_bsm = Boltz.solve_boltzmann_eq(T_gamma_0, T_nu_0, T_DS_0)
    
    time_end = time.time()
    
    return Boltz, sol_sm, sol_bsm, (time_end - time_start)

if __name__ == "__main__":
    __spec__ = None

    parser = argparse.ArgumentParser(
                    prog='compute_n_eff_ADM.py',
                    description='calculate Neff from mcp freeze-in/freeze-out in adm models')
                    
    subparsers = parser.add_subparsers(dest="subparser")
    
    parser_compute = subparsers.add_parser('compute', help='Compute N_eff at a single point in the (m_de, m_dp, Q) parameter space')
    parser_compute.add_argument('m_de', action='store', type=float, help='mass of dark electron in MeV')
    parser_compute.add_argument('m_dp', action='store', type=float, help='mass of dark proton in MeV')
    parser_compute.add_argument('Q', action='store', type=float, help='millicharge of dark fermions')

    parser_scan = subparsers.add_parser('scan', help='Compute N_eff values for points defined in an input file (see notebooks/adm_scan_list.ipynb)')
    parser_scan.add_argument('scan_list', action='store', type=str, help='filename of text file with scan parameters')
    parser_scan.add_argument('i_start', action='store', type=int, help='index into scan params list to start at')
    parser_scan.add_argument('i_end', action='store', type=int, help='index into scan params list to end at')

    parser.add_argument('--save_plots', dest='do_plots', action='store_true')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--outdir', dest='outdir', action='store', default='./', type=str)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ncores", dest="n_cores", default=1,
                       type=int, help="Number of processes (uses multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")
    args = parser.parse_args()
    

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        
    pool_kwargs = dict()
    if args.mpi is True:
        pool_kwargs = {
        'use_dill': True
        }

    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.n_cores, **pool_kwargs)
    
    if args.subparser == 'compute':
        td = {
            'm_de': args.m_de,
            'm_dp': args.m_dp,
            'Q': args.Q,
            'outdir': args.outdir,
            'do_plots': args.do_plots,
            'overwrite': args.overwrite
        }
        tasks = [td]
    
    if args.subparser == 'scan':
        # read file of model points
        joblist = np.loadtxt(args.scan_list, delimiter=',')[args.i_start:args.i_end]
        tasks = [dict()]*len(joblist)
        
        for (i, job) in enumerate(joblist):
            td = {
                'm_de': job[0],
                'm_dp': job[1],
                'Q': job[2],
                'outdir': args.outdir,
                'do_plots': args.do_plots,
                'overwrite': args.overwrite
            }
            tasks[i] = td
        
    
    time_start = time.time()
    for p in pool.map(worker, tasks, callback=save_results):
        pass
    pool.close()

    time_end = time.time()
    print(f'Time: {(time_end - time_start)/60:.5f} minutes')


