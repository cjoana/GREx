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

# import homemade code
sys.path.append('./engrenage_Hydra/')
sys.path.append('../')

 
# from source._par_rhsevolution import *  # go here to look at how the evolution works
from source.rhsevolution import *             # go here to look at how the evolution works
from source.initialdata import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

h5_filename = './out_B.hdf5'


params = Munch(dict())
# cosmology
params.omega = 1./3
params.t_ini = 1.
params.H_ini = 2./(3.*(1.+params.omega))/params.t_ini # alpha/t_ini
params.rho_bkg_ini =  3./(8.*np.pi) *params.H_ini**2
params.a_ini = 1
# grid
params.N_r = 500
params.r_max = 400 * params.H_ini
params.r_is_logarithmic = False
params.sigma_factor = 1
params.dt_multiplier = 0.01
params.dx = params.r_max/params.N_r
params.dt0 = params.dx * params.dt_multiplier





# get intial state
N_r = params.N_r
r, initial_state = get_initial_state(params)

#unpackage the vector for readability
# (initial_U, initial_R , initial_M, initial_rho) = unpack_state(initial_state, N_r)


rm =  get_rm(params, idata_params)   ###
oneoverdx  = 1.0 / params.dx
oneoverdxsquared = oneoverdx * oneoverdx

    
import time

start = time.time()
# for control of time integrator and spatial grid


# t_ini = 1.0 
# dx = R_max/N_r
# dt_multiplier = 0.01
# dt = dx * dt_multiplier
# N_t = 10000
# T  = t_ini + dt * N_t
# sigma = 0./ dt_multiplier     # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R_max/N_r  


# print("initial params:")
# print(f" R_max = {R_max}\n N_r = {N_r}\n dx = {dx}\n dt = {dt}\n N_t ={N_t}\n T = {T}\n rm ={rm}\n rho_bkg_ini = {rho_bkg_ini}\n num Horizons = {n_Horizons}")


# ####################################################



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
        iterations = len(keys)//2 -1 
        
        times = np.array([ datalev[f't_{i}'][0] for i in range(iterations)])
        solutions = np.array([ datalev[f'{i}'][:] for i in range(iterations)]) 

        for key in params.keys():
            params[key] = h5file.attrs[key]
            print(f" {key} : {params[key]}")
        
        for key in idata_params.keys():
            idata_params[key] = h5file.attrs[key]
            print(f" {key} : {idata_params[key]}")
        
        return times, solutions
	

    




# RK-4 method
def rk4(t0, y0, params, h5_filename=None,
        max_iterations=10000, max_t=1000, iter_output=1000, cnt=0):
    
    def f(t_i,current_state, progress_bar, sigma, dt):
         # return rhs coded in rhsevolution.py
         return get_rhs(t_i, current_state, \
                        params,  \
                        sigma, progress_bar, [params.t_ini, dt])
            
    dt0 = params.dt0
    dt_multiplier = params.dt_multiplier
    sigma_factor = params.sigma_factor
            
    solutions = []
    times = []
    # cnt = cnt0
    print_start = True if cnt==0 else False
    
    print('\n Starting RK4 integration...\n')
    
    with tqdm(total=max_t, unit=" ") as progress_bar:
         for i in range(max_iterations):
            
            sigma = sigma_factor/ dt_multiplier / t0**0.5
            
            
            # Save solutions
            if (i%iter_output==0) and h5_filename and print_start:                                     
                save_data_h5file(h5_filename, y0, t0, cnt)
                solutions.append(y0)
                times.append(t0)
                cnt+=1
                # save_solution_hdf5()
            else: 
                progress_bar.update(round(t0,3))
                print_start = True
               
            # if (i%10==0): print(f'   - time = {round(t0,2)},   {i}      ', end='\r')
            
            
            # Calculating step size
            dt = dt0   * t0**0.5
            
            
            # print("here:: ", f(t0, y0, progress_bar))
            
            k1 = dt * f(t0, y0, progress_bar, sigma, dt)
            k2 = dt * f(t0+dt/2, y0+k1/2, progress_bar, sigma, dt)
            k3 = dt * f(t0+dt/2, y0+k2/2, progress_bar, sigma, dt)
            k4 = dt * f(t0+dt, y0+k3, progress_bar, sigma, dt)
            k = (k1+2*k2+2*k3+k4)/6
            yn = y0 + k
            
            y0 = yn
            t0 = t0+dt
            if t0 > max_t: break       
    
    print(f'\n Final time = {t0}. Done.')

    return np.array(solutions), np.array(times)
        


do_simulation = True
restart = False

if not restart and do_simulation: 
	create_attrib_h5file(h5_filename)

	N_t = 100
	N_t = 10000
	T = 8000

	solution, t_sol = rk4(params.t_ini, initial_state, params, h5_filename,  \
						   max_iterations=N_t, max_t=T, iter_output=N_t//10)
elif do_simulation:
	N_t = 100
	N_t = 50000
	T = 15000	
	
	restart_from_h5file(h5_filename, params, idata_params, start_from = False,\
                        max_iterations=N_t, max_t=T, iter_output=N_t//20)
	

end = time.time() 
if do_simulation: print(f"Time needed for evolution {end-start} seconds.") 












t_sol, solution = load_simdata(h5_filename, params, idata_params)

rm = get_rm(params, idata_params, print_out=1)


print("times, ", t_sol)

############################################################

if True:
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    dx = R_max/N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    t = t_sol
    
    
    
    fig, axs = plt.subplots(2,4, figsize=(17,8))
    
    for i, t_i in enumerate(t) :
        # if not (i%5 ==0) : continue
        # if t_i < t_ini : continue
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
        ax.set_ylabel('M/R')
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
    plt.savefig('evol_data_example.png', dpi=100)
    # plt.show()    
    
    
    
    fig, axs = plt.subplots(2,3, figsize=(17,8))
    
    for i, t_i in enumerate(t) :
        # if not (i%5 ==0) : continue
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
    plt.savefig('evol_derivatives_example.png', dpi=100)
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













