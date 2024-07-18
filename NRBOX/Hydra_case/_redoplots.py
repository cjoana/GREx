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

# from source.initialdata import idata_params
#from Collapse_evo import params

# from io_data import *

from _RUN import idata_params


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







nu = idata_params.nu

tmax = 9800
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


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

from itertools import cycle
cycol = cycle('bgrcmk')



if True:
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    N_r = params.N_r
    dx = R_max/params.N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    t = t_sol
    
    
    t = t[t<tmax]

    # HAM PLOT
    fig, axs = plt.subplots(2,4, figsize=(17,8))
    ln = len(t)
    for i, t_i in enumerate(t) :

        icolor = next(cycol)
        ln = len(t)//num_points

        skp = 1
        if i > len(t) - skp: continue
        if i > len(t) - skp -1:
            print('plotting lasts timesteps! ')
            pass
        elif not (i%ln ==0) :      continue


        # if t_i < ln/2 : continue
        labelt = "t="+str(round(t_i,2))
        
        vars_vec = solution[i,:]
        chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(vars_vec, N_r) 

        rho = get_rho(D, E) 
        rho_bkg = np.mean(rho)  #get_rho_bkg(t_i, params.rho_bkg_ini)
        
        # aezeta = np.exp(2*chi)
        # R = get_R(r, aezeta, 0, b)
        # dRdr = get_dfdx(R, oneoverdx)
        # M = get_M(r, rho, R, dRdr) 
        # dMdr = get_dfdx(M, oneoverdx)
        # U = R*0

        # rho = get_rho(D, E) 
        
        # aezeta = np.exp(2*chi)
        # R = get_R(r, aezeta, 0, b)
        # dRdr = get_dfdx(R, oneoverdx)
        # M = get_M(r, rho, R, dRdr) 
        # dMdr = get_dfdx(M, oneoverdx)
        # U = R*0
      
        
        
        # dMdr = get_dfdx(M, oneoverdx)
        # dRdr = get_dfdx(R, oneoverdx)
        
        rho_ADM = D + E
        Ab = -0.5*Aa
        dadr = get_dfdx(a, oneoverdx) 
        dbdr = get_dfdx(b, oneoverdx) 
        dchidr = get_dfdx(chi, oneoverdx)
        dLambdadr = get_dfdx(Lambda, oneoverdx) 
        d2chidr2    = get_d2fdx2(chi, oneoverdxsquared)
        d2adr2      = get_d2fdx2(a, oneoverdxsquared) 
        d2bdr2      = get_d2fdx2(b, oneoverdxsquared) 
        em4chi = np.exp(-4*chi)
        
        ricci_scalar = get_ricci_scalar(r, a, b, dadr, dbdr, d2adr2, d2bdr2, em4chi, dchidr, d2chidr2, 
                     dLambdadr)
        HamRel = get_constraint_HamRel(ricci_scalar, Aa, Ab, K, rho_ADM)

        Ham = get_constraint_Ham(ricci_scalar, Aa, Ab, K, rho_ADM)


        ##############
        
        r_over_rm = r/rm
        xmin, xmax = [r_over_rm[3]  ,r_over_rm[-4]]
        mask = (r_over_rm)<8
        
        
        f_t = HamRel           
        ln = len(r)
        ax = axs[0,0]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('HamRel')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(1e-6,2)


        f_t = ricci_scalar           
        ln = len(r)
        ax = axs[0,1]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('ricci')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)


        f_t = 2/3*K*K         
        ln = len(r)
        ax = axs[0,2]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('2/3 K*K')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)


        drho = rho_ADM *16*np.pi - 2/3*K*K  
        f_t = drho     
        ln = len(r)
        ax = axs[0,3]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('drho')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)

        f_t = Ham       
        ln = len(r)
        ax = axs[1,0]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('Ham')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)


        f_t = (Aa*Aa +Ab*Ab)        
        ln = len(r)
        ax = axs[1,1]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('AaAa + AbAb')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)


        
        f_t = ricci_scalar - drho
        ln = len(r)
        ax = axs[1,2]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-')
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('ricci - drho')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        # ax.set_ylim(1e-3, 10)



        f_t = dchidr      
        ln = len(r)
        ax = axs[1,3]
        ax.plot(r_over_rm[mask], f_t[mask], color=icolor, ls='-', label=labelt)
        ax.plot(r_over_rm[mask], -f_t[mask], color=icolor, ls='--')
        ax.set_ylabel('dchidr')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        ax.legend()


    
    
    

    plt.tight_layout()
    plt.savefig(plotdir + 'evol_HamConst.png', dpi=100)
    # plt.show()    
        

    


# plot all evo vars
if True:
    R_max = params.r_max
    rho_bkg_ini = params.rho_bkg_ini
    N_r = params.N_r
    dx = R_max/params.N_r
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    t = t_sol
    
    
    t = t[t<tmax]

    ln = len(t)
    print(f"!!! Number of iterations {ln}, plotted {num_points} iterations")

    figtitle = f"Number of iterations {ln}, plotted {num_points} iterations"
    
    fig, axs = plt.subplots(2,7, figsize=(25,8))
    plt.title(figtitle)
    for i, t_i in enumerate(t) :
        ln = len(t)//num_points

        if i > len(t) - 3:
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
        ax.set_yscale('log')
        
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
        f_t = X
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('X')
        ax.set_yscale('log')
        
        ax = axs[0,5]
        f_t = AX
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('AX')
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

        ax = axs[1,3]
        f_t = D
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('D')
        ax.set_yscale('log')

        ax = axs[1,4]
        f_t = E
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('E')
        ax.set_yscale('log')


        ax = axs[1,5]
        f_t = Aa
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('Aa')
        ax.set_yscale('log')


        ax = axs[1,6]
        f_t = S
        ax.plot(r_over_rm, f_t, label=labelt)
        ax.set_ylabel('S')
        ax.set_yscale('log')
        


        
    
    plt.tight_layout()
    plt.savefig(plotdir + 'evolvars_data_example.png', dpi=100)
    # plt.show()    
    
    # print('For nu = ', idata_params.nu,  ' the max value of C at end: ', np.nanmax(C))
    


# plot Cosmo vars 
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
        ln = len(t)//(num_points)


        skp = 1

        if i > len(t) - skp: continue

        if i > len(t) - skp -1:
            print('plotting lasts timesteps! ')
            pass
        elif not (i%ln ==0) :      continue


        # if t_i < ln/2 : continue
        labelt = "t="+str(round(t_i,2))
        
        vars_vec = solution[i,:]
        chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(vars_vec, N_r) 

        rho = get_rho(D, E) 
        
        aezeta = np.exp(2*chi)
        R = get_R(r, aezeta, 0, b)
        dRdr = get_dfdx(R, oneoverdx)
        M = get_M(r, rho, R, dRdr) 
        dMdr = get_dfdx(M, oneoverdx)
        U = R*0
        

        print("D  E  = ", np.nanmean(D), np.nanmean(E))


        
        rho_bkg = (K**2/(24*np.pi))[-30]  #get_rho_bkg(t_i, params.rho_bkg_ini)
        
        dMdr = get_dfdx(M, oneoverdx)
        dRdr = get_dfdx(R, oneoverdx)
        
        rho_ADM = D + E
        Ab = -0.5*Aa
        dadr = get_dfdx(a, oneoverdx)
        dbdr = get_dfdx(b, oneoverdx)
        dchidr = get_dfdx(chi, oneoverdx)
        dLambdadr = get_dfdx(Lambda, oneoverdx)
        d2chidr2    = get_d2fdx2(chi, oneoverdxsquared)
        d2adr2      = get_d2fdx2(a, oneoverdxsquared)
        d2bdr2      = get_d2fdx2(b, oneoverdxsquared)
        em4chi = np.exp(-4*chi)
        
        ricci_scalar = get_ricci_scalar(r, a, b, dadr, dbdr, d2adr2, d2bdr2, em4chi, dchidr, d2chidr2, 
                     dLambdadr)
        Ham = get_constraint_Ham(ricci_scalar, Aa, Ab, K, rho_ADM)
        HamRel = get_constraint_HamRel(ricci_scalar, Aa, Ab, K, rho_ADM)

        C = compact_function(r, M, R, dRdr, rho_bkg)
        # C = get_CompactionSS_altern(r, dchidr, omega=1/3)
        CSS = get_CompactionSS(chi, dRdr)
        # CSS = get_Compaction(chi, dRdr)
        
        
        ##############
        
        r_over_rm = r/rm
        xmin, xmax = [r_over_rm[3]  ,r_over_rm[-4]]
        mask = (r_over_rm)<8
        
        
        
        f_t = C  
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
        f_t = np.abs(HamRel)
        ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
        # ax.set_ylim(0.1,1e4)
        ax.set_ylabel('Ham')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(1e-5,2)
        
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
    





###########











