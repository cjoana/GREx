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

params = Munch(dict())
# cosmology


# initial time
params.t_0 = idata_params.t_0


# cosmology 
params.t_ini = 1.  
params.a_ini = 1.
params.omega = 1./3
params.H_ini =  2./(3.*(1.+params.omega))  * (params.t_ini/params.t_ini) 
params.rho_bkg_ini =  3./(8.*np.pi) *params.H_ini**2

# grid
params.N_r = 1000
params.r_max = 1000 * params.H_ini
#params.r_max = 200 / params.H_ini
params.r_is_logarithmic = False
params.sigma_factor = 1
params.dt_multiplier = 0.02   
params.dx = params.r_max/params.N_r
params.dt0 = params.dx * params.dt_multiplier


iter_output = 100

plotdir = './{nu}/'.format(nu=int(idata_params.nu*10**6), t0=int(params.t_0*100))
h5_filename = './out_{nu}.hdf5'.format(nu=int(idata_params.nu*10**6), t0=int(params.t_0*100))

if not os.path.exists(plotdir): os.makedirs(plotdir)

#plotting
tmax =30000
num_points = 10



# get intial state
N_r = params.N_r
r, initial_state = get_initial_state(params, idata_params)

#unpackage the vector for readability
(initial_chi, initial_a , initial_b, initial_K, initial_Aa, initial_AX, initial_X,
 initial_Lambda, initial_lapse, initial_beta, initial_br, 
 intial_D, initial_E, initial_S) = unpack_state(initial_state, N_r)


rm =  get_rm(params, idata_params)   ###
oneoverdx  = 1.0 / params.dx
oneoverdxsquared = oneoverdx * oneoverdx


# Plot initial data
if True:
    
    fig, axs = plt.subplots(2,4, figsize=(17,8))

    current_state = initial_state
    chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(current_state, N_r) 
    
    t_here = params.t_ini 
    rho_bkg = get_rho_bkg(t_here, params.rho_bkg_ini) 
    scalefactor = get_scalefactor(t_here, params.omega,params.a_ini, params.t_ini)

    print("scale factor: ", scalefactor, params.a_ini)

    rho = get_rho(D, E) 
    
    R = get_R(r, scalefactor, chi,b)
    dRdr = get_dfdx(R, oneoverdx)
    M = get_M(r, rho, R, dRdr)
    dMdr = get_dfdx(M, oneoverdx)

    C = compact_function(M, R, rho_bkg)
    CSS = get_CompactionSS(chi, dRdr)
            
    HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5        
    # Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr))
    Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)
    
    
    omega = get_omega()
    t_ini = params.t_ini
    a_ini = params.a_ini
    
    # f_t = 2*M/R
    f_t = C
    
    r_over_rm = r /rm
    
    ln = len(r)
    ax = axs[0,0]
    ax.plot(r_over_rm[:], f_t[:],'k')
    #
    ax.plot(r_over_rm[:], CSS, 'g--')
    #
    kstar = get_kstar(params, idata_params)
    Rprime_over_a_ezeta = ( 1 + r * get_dzetadr(r, kstar) ) 
    C_v2 = 3*(1+omega)/(5+3*omega) *( 1 - Rprime_over_a_ezeta**2)
    ax.plot(r_over_rm[:], C_v2, 'r--')
    #
    ax.set_ylim(-2,1.5)
    ax.set_ylabel('C')
    ax.set_xlim(0,3)
    
    ax = axs[0,1]
    f_t = 2*M/R
    ax.plot(r_over_rm[:], f_t[:])
    ax.set_ylim(0,1.5)
    ax.set_xlim(0,1.5)
    ax.legend()
    ax.set_ylabel('2*M/R')
    ax.set_xlim(0,3)
    
    ax = axs[1,0]
    f_t = R
    ax.plot(r_over_rm[:], f_t[:])
    #
    k_star = get_kstar(params, idata_params)
    R_v2 = get_expansion_R(t_ini, r, rm, omega, 0, params)
    zeta = get_zeta(r, kstar)
    R_v2 = a_ini * np.exp(zeta) *r
    ax.plot(r_over_rm[:], R_v2, 'r--')
    #
    ax.set_ylim(0.1,1e4)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('R')
    ax.set_xlim(0,8)
    
    U = np.zeros_like(rho)
    
    ax = axs[1,1]
    f_t = U
    ax.plot(r_over_rm[:], f_t[:])
    # ax.set_ylim(0.1,1e4)
    # ax.set_yscale('log')
    ax.set_ylabel('U')
    
    ax = axs[0,2]
    f_t = M
    ax.plot(r_over_rm[:], f_t[:])
    # ax.set_ylim(0.1,1e4)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('M')
    
    ax = axs[1,2]
    f_t = rho
    ax.plot(r_over_rm[:], f_t[:])
    # rho_check = dMdr/(dRdr* 4*np.pi*R**2 +1e-5)
    # ax.plot(r_over_rm, rho_check)
    # ax.set_ylim(0.1,1e4)
    ax.set_ylabel('rho')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    
    
    ax = axs[0,3]
    f_t = dRdr
    ax.plot(r_over_rm[:], f_t[:])
    #
    k_star = get_kstar(params, idata_params)
    Rprime = ( 1 + r * get_dzetadr(r, kstar) ) * a_ini * np.exp(get_zeta(r, kstar))      # R = a ezeta(r) * r ,   Rprime =  a ezeta r zetaprime + a ezeta = (1 + r zetaprime) * a ezeta
    ax.plot(r_over_rm[:], Rprime, 'r--')
    #
    # ax.set_ylim(0.1,1e4)
    ax.set_ylabel('dRdr')
    ax.set_xlabel('r')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0,3)
        
    ax = axs[1,3]
    f_t = Ham
    ax.plot(r_over_rm[:], f_t[:])
    ax.set_ylabel('Ham')
    ax.set_yscale('log')
        
    plt.tight_layout()
    plt.savefig(plotdir + 'initial_data_example.png', dpi=100)
    plt.clf()
    plt.close()



# Start evolution part 
import time

start = time.time()


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
    
    def f(t_i,current_state, prev_state, progress_bar, sigma, dt):
         # return rhs coded in rhsevolution.py
         return get_rhs(t_i, current_state, prev_state,  \
                        params,  \
                        sigma, progress_bar, [params.t_ini, dt])
            
    dt0 = params.dt0
    dt_multiplier = params.dt_multiplier
    sigma_factor = params.sigma_factor
    yprev = y0

            
    solutions = []
    times = []
    # cnt = cnt0
    print_start = True if cnt==0 else False
    
    print('\n Starting RK4 integration...\n')
    
    with tqdm(total=max_t, unit=" ") as progress_bar:
         for i in range(max_iterations):

            # Save solutions
            if (i%iter_output==0) and h5_filename and print_start:                                     
                save_data_h5file(h5_filename, y0, t0, cnt)
                solutions.append(y0)
                times.append(t0)
                cnt+=1
                # save_solution_hdf5()


                print(f" i saved= {cnt} ")
            else: 
                progress_bar.update(round(t0,3))
                print_start = True
            


            # if (i%10==0): print(f'   - time = {round(t0,2)},   {i}      ', end='\r')
            
            
            # Calculating step size
            dt = dt0   * t0**0.5
            sigma = sigma_factor/ dt_multiplier / t0**0.5

            # print("here:: ", f(t0, y0, progress_bar))        
            k1 = dt * f(t0, y0, yprev, progress_bar, sigma, dt)
            k2 = dt * f(t0+dt/2, y0+k1/2, yprev,  progress_bar, sigma, dt)
            k3 = dt * f(t0+dt/2, y0+k2/2, yprev, progress_bar, sigma, dt)
            k4 = dt * f(t0+dt, y0+k3, yprev, progress_bar, sigma, dt)
            k = (k1+2*k2+2*k3+k4)/6
            yn = y0 + k
            yprev = y0
            
            
            y0 = yn
            t0 = t0+dt
            if t0 > max_t: break       
    
    print(f'\n Final time = {t0}. Done.')

    return np.array(solutions), np.array(times)
        

do_simulation = True
restart = False

if not restart and do_simulation: 
	create_attrib_h5file(h5_filename)

	# N_t = 100 *params.t_0
	N_t = int(2000000 *params.t_0)
	T = int(200000 *params.t_0)
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


t_sol, solution = load_simdata(h5_filename, params, idata_params)

rm = get_rm(params, idata_params, print_out=1)


print("times, ", t_sol)

############################################################


# Plot history evolution (results)
if True:

    omega = get_omega()   
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    dx = R_max/N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    t = t_sol
    Cmax = 0 
    
    t = t[t<tmax]
    
    fig, axs = plt.subplots(2,4, figsize=(17,8))
    ln = len(t)//10
    print(ln)
    for i, t_i in enumerate(t) :
        ln = len(t)//num_points
        if  (i%ln !=0) :      continue
        print(ln, i)
        # if t_i < ln/2 : continue
        labelt = "t="+str(round(t_i,2))
           
        vars_vec = solution[i,:]
        chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(vars_vec, N_r) 

        rho = get_rho(D, E) 
        scalefactor = get_scalefactor(t_i, params.omega,params.a_ini, params.t_ini)
        R = get_R(r, scalefactor, chi, b)
        dRdr = get_dfdx(R, oneoverdx)
        M = get_M(r, rho, R, dRdr)
        dMdr = get_dfdx(M, oneoverdx)


        rho_bkg = get_rho_bkg(t_i, params.rho_bkg_ini)
        C = compact_function(M, R, rho_bkg)
        
        dMdr = get_dfdx(M, oneoverdx)
        dRdr = get_dfdx(R, oneoverdx)
                
        HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5        
        Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)
        
        # f_t = 2*M/R
        f_t = C
        
        Cmax = np.nanmax([Cmax, np.nanmax(C)  ])

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
    plt.savefig(plotdir + 'evol_data_example.png', dpi=100)
    # plt.show()    
    
    print('For nu = ', idata_params.nu,  ' the max value of C at end: ', Cmax)
    


# # 
# # End of file 
# ###################################













