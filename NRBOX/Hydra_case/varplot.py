# load the required python modules
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import time
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from munch import Munch
import h5py 
import os

# import homemade code
#sys.path.append('./engrenage_MSPBH/')
sys.path.append('../')

 
# from source._par_rhsevolution import *  # go here to look at how the evolution works
from source.rhsevolution import *             # go here to look at how the evolution works
from source.initialdata import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

from source.initialdata import idata_params
#from Collapse_evo import params

# from io_data import *


params = Munch(dict())
# cosmology
params.omega = 1./3
params.t_ini = 1.
params.H_ini = 2./(3.*(1.+params.omega))/params.t_ini # alpha/t_ini
params.rho_bkg_ini =  3./(8.*np.pi) *params.H_ini**2
params.a_ini = 1
# grid
params.N_r = 800
params.r_max = 400 * params.H_ini
params.r_is_logarithmic = False
params.sigma_factor = 1
params.dt_multiplier = 0.05
params.dx = params.r_max/params.N_r
params.dt0 = params.dx * params.dt_multiplier







nu = 0.90

tmax = 100000000000
num_points = 20
endplot = -1

plotdir = './{nu}/'.format(nu=int(nu*10**6))
h5_filename = './out_{nu}.hdf5'.format(nu=int(nu*10**6))

print('loading ', h5_filename)

if not os.path.exists(plotdir): os.makedirs(plotdir)




start = time.time()
# for control of time integrator and spatial grid




for key in params.keys():
    print(f" {key} : {params[key]}") 
for key in idata_params.keys():
    print(f" {key} : {idata_params[key]}") 


def create_attrib_h5file(h5_filename): 
    with h5py.File(h5_filename, "w-") as h5file:
        for key in params.keys():
            h5file.attrs[key] = params[key]
        for key in idata_params.keys():
            h5file.attrs[key] = idata_params[key]
        datalevel = h5file.create_group('data')
    return 0
            
            
def save_data_h5file(h5_filename, data_array, time, iteration): 
    with h5py.File(h5_filename, "r+") as h5file:

        t_iteration = f"t_{str(iteration)}"
        datalev = h5file["data"]
        datalev.create_dataset(str(iteration), data=data_array)
        datalev.create_dataset(t_iteration, data=[time])
    return 0

def restart_from_h5file(h5_filename, params, idata_params, start_from = False,\
                        max_iterations=10000, max_t=1000, iter_output=1000):
    with h5py.File(h5_filename, "r") as h5file:
        datalev = h5file["data"] 
        keys = datalev.keys()
        iterations = len(keys)//2 -1 
        start_it = iterations if not start_from else int(start_from) 
        tn = datalev[f't_{start_it}'][0]
        yn = datalev[f'{start_it}'][:]
        
        for key in params.keys():
            params[key] = h5file.attrs[key]
            print(f" {key} : {params[key]}")
        
        for key in idata_params.keys():
            idata_params[key] = h5file.attrs[key]
            print(f" {key} : {idata_params[key]}")
             
    # rerun! 
    rk4(tn, yn, params, h5_filename, \
                max_iterations, max_t, iter_output, cnt = int(start_it)+1)


def load_simdata(h5_filename, params, idata_params):
	
    with h5py.File(h5_filename, "r") as h5file:
        datalev = h5file["data"] 
        keys = datalev.keys()
        iterations = len(keys)//2 -4
        
        times = np.array([ datalev[f't_{i}'][0] for i in range(iterations)])
        solutions = np.array([ datalev[f'{i}'][:] for i in range(iterations)]) 

        for key in params.keys():
            params[key] = h5file.attrs[key]
            print(f" {key} : {params[key]}")
        
        for key in idata_params.keys():
            idata_params[key] = h5file.attrs[key]
            print(f" {key} : {idata_params[key]}")
        
        return times, solutions, params, idata_params
	

t_sol, solution, params, idata_params = load_simdata(h5_filename, params, idata_params)

rm = get_rm(params, idata_params, print_out=1)

r, initial_state = get_initial_state(params, idata_params)

# print("times, ", t_sol)

omega = params.omega
############################################################

# plot Cosmo vars 
if False:
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    N_r = params.N_r
    dx = R_max/params.N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    t = t_sol
    
    
    t = t[t<tmax]
    
    fig, axs = plt.subplots(2,4, figsize=(17,8))
    ln = len(t)
    print("!!! Number of iterations ", ln)
    for i, t_i in enumerate(t) :
        ln = len(t)//num_points
        if not (i%ln ==0) :      continue
        # if t_i < ln/2 : continue
        labelt = "t="+str(round(t_i,2))
        
        vars_vec = solution[i,:]
        chi, a, b, K, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(vars_vec, N_r) 

        rho = get_rho(D, E) 
        scalefactor = get_scalefactor(t_i, params.omega,params.a_ini, params.t_ini)
        R = get_R(r, scalefactor, chi, b)
        dRdr = get_dfdx(R, oneoverdx)
        M = get_M(r, rho, R, dRdr)
        dMdr = get_dfdx(M, oneoverdx)

        U = R*0
        
        # var = idx_R
        # R = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        # var = idx_M
        # M = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]



        print("D  E  = ", np.nanmean(D), np.nanmean(E))


        
        rho_bkg = get_rho_bkg(t_i, params.rho_bkg_ini)
        
        dMdr = get_dfdx(M, oneoverdx)
        dRdr = get_dfdx(R, oneoverdx)
                
        HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5        
        Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)

        C = compact_function(M, R, rho_bkg)
        CSS = get_CompactionSS(chi, dRdr)
        
        # f_t = 2*M/R
        f_t = C
        
        r_over_rm = r/rm
        xmin, xmax = [r_over_rm[3]  ,r_over_rm[-4]]
        mask = (r_over_rm)<8
                
        ln = len(r)
        ax = axs[0,0]
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        ax.set_ylim(-2,3)
        ax.set_ylabel('C')
        ax.set_xlim(xmin,4)
        
        ax = axs[0,1]
        f_t = 2*M/R
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        ax.set_ylim(0,1.5)
        ax.set_xlim(0,1.5)
        ax.legend()
        ax.set_ylabel('2M/R')
        ax.set_xlim(xmin,4)
        
        ax = axs[1,0]
        f_t = R
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        ax.set_ylim(0.1,1e6)
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('R')
        ax.set_xlim(xmin,xmax)
        
        
        ax = axs[1,1]
        f_t = CSS
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        ax.set_ylim(-2,3)
        # ax.set_yscale('log')
        ax.set_ylabel('CSS')
        ax.set_xlim(xmin,5)
        
        ax = axs[0,2]
        f_t = np.abs(M)
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel('M')
        
        ax = axs[1,2]
        f_t = rho
        # rho_check = dMdr/dRdr / (4*np.pi*R**2)
        
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.plot(r_over_rm, rho_check, label=labelt)
        # ax.set_ylim(0.1,1e4)
        ax.set_ylabel('rho')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        
        
        ax = axs[0,3]
        f_t = (rho - rho_bkg)/rho_bkg
        # rho_check = dMdr/dRdr / (4*np.pi*R**2)
        
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.plot(r_over_rm, rho_check, label=labelt)
        # ax.set_ylim(0.1,1e4)
        ax.set_ylabel('rho')
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        
        
        ax = axs[1,3]
        f_t = Ham
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        ax.set_ylabel('Ham')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        
                
        
        
    
    # mlabel = "2M/R"
    mlabel = "C"

    # plt.yscale('log')
    # plt.ylim(-2, 1.5)
    # plt.xlim(0, 3)
    # plt.legend(loc=4)
    # plt.xlabel('r')
    # plt.ylabel('value over time of ' + mlabel)
    # plt.grid()
    plt.tight_layout()
    plt.savefig(plotdir + 'evol_data_example.png', dpi=100)
    # plt.show()    
    
    # print('For nu = ', idata_params.nu,  ' the max value of C at end: ', np.nanmax(C))
    


# plot all evo vars
if True:
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    N_r = params.N_r
    dx = R_max/params.N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    t_ini = params.t_ini
    
    t = t_sol
    
    
    t = t[t<tmax]

    ln = len(t)
    print(f"!!! Number of iterations {ln}, plotted {num_points} iterations")

    figtitle = f"Number of iterations {ln}, plotted {num_points} iterations"
    
    fig, axs = plt.subplots(2,8, figsize=(33,8))
    plt.title(figtitle)

    tlist = t[:endplot]
    for i, t_i in enumerate(tlist) :
        ln = len(tlist)//num_points
        
        skp = 1

        if i > len(t) - skp: continue

        if i > len(t) - skp - 4:
            print('plotting lasts timesteps! ')
            pass
        elif not (i%ln ==0) :      continue
        
        

        # if t_i < ln/2 : continue
        labelt = "t="+str(round(t_i,2))
        
        vars_vec = solution[i,:]
        chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(vars_vec, N_r) 

        r_over_rm = r/rm
        
        ax = axs[0,0]
        f_t = chi
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('chi')
        # ax.set_yscale('log')
        
        ax = axs[0,1]
        f_t = a
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('a')
        
        ax = axs[0,2]
        f_t = b
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('b')

        ax = axs[0,3]
        f_t = np.abs(K)
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('abs K')
        ax.set_yscale('log')

        ax = axs[0,4]
        f_t = AX
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('AX')
        ax.set_yscale('log')
        
        ax = axs[0,5]
        f_t = X
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('X')
        ax.set_yscale('log')

        ax = axs[0,6]
        f_t = Lambda
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('Lambda')
        ax.set_yscale('log')

        ax = axs[1,0]
        f_t = lapse
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('lapse')
        ax.set_yscale('log')

        ax = axs[1,1]
        f_t = beta
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('beta')

        ax = axs[1,2]
        f_t = br
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('br')

        omega = 1./3
        scalefactor =  get_scalefactor(t_i, omega, params.a_ini, params.t_ini)
        zeta_evo = 2*chi - 2*np.log(scalefactor)
        R = get_R(r, scalefactor, zeta_evo, 1)   
        dRdr = get_dfdx(R, oneoverdx)
        # CSS = get_CompactionSS(chi, dRdr, scalefactor=1)
        rho = D+E 
        drho = rho - get_rho_bkg(t_i/t_ini, params.rho_bkg_ini)
        C = get_int_Compaction(r, rho, R, dRdr, drho)

        ax = axs[1,3]
        f_t =  R
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('CSS')
        ax.set_yscale('log')
        # ax.set_ylim(-2,5)
        ax.set_xlim(0,5)

        ax = axs[1,4]
        f_t = E
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('E')
        ax.set_yscale('log')


        ax = axs[1,5]
        f_t = S
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('S')
        ax.set_yscale('log')

        ax = axs[1,6]
        f_t = Aa
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('Aa')
        ax.set_yscale('log')
        


        
    
    plt.tight_layout()
    plt.savefig(plotdir + 'evolvars_data_example.png', dpi=100)
    # plt.show()    
    
    # print('For nu = ', idata_params.nu,  ' the max value of C at end: ', np.nanmax(C))
    




###########











