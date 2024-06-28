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
from initialdata import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

from initialdata import idata_params
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







nu = 0.7

tmax = 8000
num_points = 6

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

if True:
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
        U, R, M, rho = unpack_state(vars_vec, N_r)
        
        # var = idx_R
        # R = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        # var = idx_M
        # M = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        
        rho_bkg = get_rho_bkg(t_i, params.rho_bkg_ini)
        C = compact_function(M, R, rho_bkg)
        
        dMdr = get_dfdx(M, oneoverdx)
        dRdr = get_dfdx(R, oneoverdx)
                
        HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5        
        Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)
        
        
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
        f_t = U
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_yscale('log')
        ax.set_ylabel('U')
        ax.set_xlim(xmin,xmax)
        
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
    
    print('For nu = ', idata_params.nu,  ' the max value of C at end: ', np.nanmax(C))
    
    fig, axs = plt.subplots(2,3, figsize=(17,8))
    
    for i, t_i in enumerate(t) :
        ln = len(t)//num_points
        if not (i%ln ==0) :  continue
        
        # if t_i < t_ini : continue
        labelt = "t="+str(round(t_i/(rm**2/4),2))+r" $t_H$"
        
        vars_vec = solution[i,:]
        U, R, M, rho = unpack_state(vars_vec, N_r)
        
        # var = idx_R
        # R = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        # var = idx_M
        # M = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]

        
        dUdr = get_dfdx(U, oneoverdx)
        drhodr = get_dfdx(rho, oneoverdx)
        dMdr = get_dfdx(M, oneoverdx)
        dRdr = get_dfdx(R, oneoverdx)
        
        # f_t = 2*M/R
        f_t = dUdr
                
        r_over_rm = r/rm
        
        
        
        ln = len(r)
        ax = axs[0,0]
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(-2,3)
        ax.set_ylabel('dUdr')
        
        ax = axs[0,1]
        f_t = dMdr
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        ax.set_xlim(xmin,xmax)
        # ax.set_xlim(0,1.5)
        ax.legend()
        ax.set_ylabel('dMdr')
        
        ax = axs[1,0]
        f_t = dRdr
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_ylabel('dRdr')
        ax.set_xlim(xmin,xmax)
        
        
        ax = axs[1,1]
        f_t = drhodr
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_yscale('log')
        ax.set_ylabel('drhodr')
        ax.set_xlim(xmin,xmax)
        
        ax = axs[1,2]
        f_t = rho
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_yscale('log')
        ax.set_ylabel('rho')
        ax.set_xlim(xmin,xmax)


        rho_bkg = get_rho_bkg(t_i, rho_bkg_ini)
        A = np.array([ get_A(rho[ii], rho_bkg, omega) for ii, rrr in enumerate(rho)]) 
        Gamma =  np.array([ get_Gamma(U[ii], r[ii], M[i]) for ii, rrr in enumerate(U)])  
        rhs_rho  = np.array([ get_rhs_rho(U[ii], R[ii], rho[ii], dUdr[ii], dRdr[ii], A[ii], omega) for ii, rrr in enumerate(U)])  
        
        ax = axs[0,2]
        f_t = rhs_rho
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        # ax.set_yscale('log')
        ax.set_ylabel('rhs rho')
        ax.set_xlim(xmin,xmax)

        
        
        
        
    plt.tight_layout()
    plt.savefig( plotdir + 'evol_derivatives_example.png', dpi=100)
    # plt.show()



# # plot the profile for some variable at a selection of times
# var = idx_u # I suggest looking at the field u, or the lapse to see the gauge evolution

# plt.clf()

# for i, t_i in enumerate(t) :
    # if (i < N_t) and (i % num_out == 0) and (t_i > 0.0):
        # labelt = "t="+str(round(t_i,2))
        # f_t = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        # plt.plot(r, f_t, label=labelt)

# plt.legend(loc=4)
# plt.xlabel('r')
# # plt.xlim(-0.2,35.0)
# plt.ylabel('value over time of ' + variable_names[var])
# plt.grid()
# plt.show()













