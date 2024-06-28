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


def load_simdata(h5_filename, params, idata_params, print_params=False):
	
    with h5py.File(h5_filename, "r") as h5file:
        datalev = h5file["data"] 
        keys = datalev.keys()
        iterations = len(keys)//2 -1 
        
        times = np.array([ datalev[f't_{i}'][0] for i in range(iterations)])
        solutions = np.array([ datalev[f'{i}'][:] for i in range(iterations)]) 

        for key in params.keys():
            params[key] = h5file.attrs[key]
            if print_params: print(f" {key} : {params[key]}")
        
        for key in idata_params.keys():
            idata_params[key] = h5file.attrs[key]
            if print_params:  print(f" {key} : {idata_params[key]}")
        
        return times, solutions, params, idata_params
       
       
################################
##
##     RK-4 procedure
################################
        

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
    
    
    
"""
 
do_simulation = True
restart = False

if not restart and do_simulation: 
	create_attrib_h5file(h5_filename)

	# N_t = 100 *params.t_0
	N_t = int(200000 *params.t_0)
	T = int(20000 *params.t_0)
	t_0 = params.t_ini

	solution, t_sol = rk4(t_0, initial_state, params, h5_filename,  \
						   max_iterations=N_t, max_t=T, iter_output=iter_output)
elif do_simulation:
	N_t = 100
	N_t = 50000
	T = 15000	
	
	restart_from_h5file(h5_filename, params, idata_params, start_from = False,\
                        max_iterations=N_t, max_t=T, iter_output=N_t//20)
	

end = time.time() 
if do_simulation: print(f"Time needed for evolution {end-start} seconds.") 


"""

