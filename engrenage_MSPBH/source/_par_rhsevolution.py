#rhsevolution.py

# python modules
import numpy as np
import time

import sys
sys.path.append("/home/cjoana/dev/GREx/engrenage/")

# homemade code
from source.uservariables import *
from source.gridfunctions import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.tensoralgebra import *
from source.misnersharp import *
from source.initialdata import rho_bkg_ini, t_ini
from source.initialdata import * 

# global N_r, R, r_is_logarithmic, eta
# from globals import N_r, R, r_is_logarithmic, eta

# parallelisation tools
import multiprocessing as mp
if not ('DefSimParams' in vars() or 'DefSimParams' in globals()):
    from source._simparams import * # N_r, R, r_is_logarithmic, eta


def par_compute_rhs_idx(ix):
	        
	#t0A = time.time()
	
	# where am I?
	r_here = sh_r[ix]
	
	t_i = sh_t_i[0]
			
	# print("Assign vars done in ", t1A - t0A) 
	# End of: Calculate some useful quantities, now start RHS
	#########################################################
	
	# Get the auxiliary vars        
	omega = get_omega()        
	
	
	rho_bkg = get_rho_bkg(t_i/t_ini, rho_bkg_ini)
	
	A = get_A(sh_rho[ix], rho_bkg, omega)
	Gamma = get_Gamma(sh_U[ix], sh_R[ix], sh_M[ix])
	
	# Get the Misner-sharp rhs 
	sh_rhs_U[ix]     =  get_rhs_U(sh_U[ix], sh_M[ix], sh_R[ix], sh_rho[ix], sh_dRdr[ix], sh_drhodr[ix], A, Gamma, omega)
	
	sh_rhs_R[ix]     =  get_rhs_R(sh_U[ix], A)
	
	sh_rhs_M[ix]     =  get_rhs_M(sh_U[ix], sh_R[ix], sh_rho[ix], A, omega) 
	
	sh_rhs_rho[ix]   =  get_rhs_rho(sh_U[ix], sh_R[ix], sh_rho[ix], sh_dUdr[ix], sh_dRdr[ix], A, omega)
                   
    # end of rhs iteration over grid points   
	
	
	

def get_rhs(t_i, current_state, R, N_r, r_is_logarithmic, sigma, progress_bar, time_state) :      ###CJ!!! remove eta  --> change to params

    # Uncomment for timing and tracking progress
    # start_time = time.time()
    sh_t_i[0] = t_i
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R, N_r, r_is_logarithmic)
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    # this is where the rhs will go
    rhs = np.zeros_like(current_state)                                    ###CJ!!!
    
    ####################################################################################################
    #unpackage the state vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    fill_outer_boundary(current_state, dx, N, r_is_logarithmic)
    
    # Unpack variables from current_state - see uservariables.py
    ### u, v , phi, hrr, htt, hpp, K, arr, att, app, lambdar, shiftr, br, lapse = unpack_state(current_state, N_r) 
    U, R, M, rho = unpack_state(current_state, N_r) 
    
    U[-num_ghosts:] = U[-num_ghosts-1]
    M[-num_ghosts:] = M[-num_ghosts-1]
    R[-num_ghosts:] = R[-num_ghosts-1]
    
    
    rho_bkg = get_rho_bkg(t_i/t_ini, rho_bkg_ini)
    rho[-num_ghosts-1:] = rho_bkg
    if np.sum(rho<0) >0 : print("  WARNING rho become negative, set to min rho.")
    rho[rho<0] = rho_bkg
    
    
    sh_r[:] = r[:]
    sh_U[:] = U[:]
    sh_M[:] = M[:]
    sh_R[:] = R[:]
    sh_rho[:] = rho[:]
    
    sh_rhs_U[:] = np.zeros_like(U)
    sh_rhs_M[:] = np.zeros_like(U)
    sh_rhs_R[:] = np.zeros_like(U)
    sh_rhs_rho[:] = np.zeros_like(U)
    
    # for ix in range(N) : 
		# sh_r[ix] = r[ix]
		# sh_U[ix] = U[ix]
    
        
    # t0 = time.time()
    # print("grid and var setup done in ", t0-start_time)
    
       
    ####################################################################################################

    # get the various derivs that we need to evolve things
    if(r_is_logarithmic) : #take logarithmic derivatives
			
        # # second derivatives
        # d2Udx2     = get_logd2fdx2(U, oneoverlogdr2)
        # d2Rdx2   = get_logd2fdx2(R, oneoverlogdr2)
        # d2Mdx2   = get_logd2fdx2(M, oneoverlogdr2)
        # d2rhodx2   = get_logd2fdx2(rho, oneoverlogdr2)
        
        # # first derivatives        
        # dUdr       = get_logdfdx(U, oneoverlogdr)
        # dRdr       = get_logdfdx(R, oneoverlogdr)
        # dMdr       = get_logdfdx(M, oneoverlogdr)
        # drhodr     = get_logdfdx(rho, oneoverlogdr)
        raise()
    
    else :
        
        # # second derivatives
        # d2Udr2     = get_d2fdx2(U, oneoverdxsquared)
        # d2Mdr2   = get_d2fdx2(M, oneoverdxsquared)
        # d2Rdr2     = get_d2fdx2(R, oneoverdxsquared)
        # d2rhodxr2   = get_d2fdx2(rho, oneoverdxsquared)

        # first derivatives
        dUdr       = get_dfdx(U, oneoverdx)
        dRdr       = get_dfdx(R, oneoverdx)
        dMdr       = get_dfdx(M, oneoverdx)
        drhodr     = get_dfdx(rho, oneoverdx)
        
        # B.C. like A. Escriva does ... 
        drhodr[:num_ghosts+1] = 0
        drhodr[-num_ghosts-1:] = 0
        
    sh_dUdr[:] = dUdr[:]
    sh_dMdr[:] = dMdr[:]
    sh_dRdr[:] = dRdr[:]
    sh_drhodr[:] = drhodr[:]

 

    # t2 = time.time()
    # print("derivs found in ", t2 - t1)
        
    ####################################################################################################
    
    # make containers for rhs values
    # rhs_U   = np.zeros_like(U)
    # rhs_R   = np.zeros_like(R)
    # rhs_M = np.zeros_like(M)
    # rhs_rho = np.zeros_like(rho)
    
    ####################################################################################################    
    
    # now iterate over the grid (vector) and calculate the rhs values
    # note that we do the ghost cells separately below
    mp.Pool().map(par_compute_rhs_idx, range(num_ghosts, N-num_ghosts))
    
    # end of rhs iteration over grid points   
    # t3 = time.time()
    # print("rhs iteration over grid done in ", t3 - t2)
    
    ####################################################################################################

	# B.C. like A. Escriva does ... 
    sh_rhs_U[-num_ghosts:] = np.zeros_like(U[-num_ghosts:])
    sh_rhs_M[-num_ghosts:] = np.zeros_like(U[-num_ghosts:])
    sh_rhs_R[-num_ghosts:] = np.zeros_like(U[-num_ghosts:])
    sh_rhs_rho[-num_ghosts:] = np.zeros_like(U[-num_ghosts:])
    


    #package up the rhs values into a vector rhs (like current_state) for return - see uservariables.py
    pack_state(rhs, N_r, sh_rhs_U, sh_rhs_R , sh_rhs_M, sh_rhs_rho)



    #################################################################################################### 
            
    # finally add Kreiss Oliger dissipation which removed noise at frequency of grid resolution
    # sigma = 10.0 # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R/N_r    ###CJ change to params.sigma
    
    diss = np.zeros_like(current_state) 
    for ivar in range(0, NUM_VARS) :
        ivar_values = current_state[(ivar)*N:(ivar+1)*N]
        ivar_diss = np.zeros_like(ivar_values)
        if(r_is_logarithmic) :
            ivar_diss = get_logdissipation(ivar_values, oneoverlogdr, sigma)
        else : 
            ivar_diss = get_dissipation(ivar_values, oneoverdx, sigma)
        diss[(ivar)*N:(ivar+1)*N] = ivar_diss
    rhs += diss
    
    # t4 = time.time()
    # print("KO diss done in ", t4 - t3)    
    
    #################################################################################################### 
    
    # see gridfunctions for these, or https://github.com/KAClough/BabyGRChombo/wiki/Useful-code-background
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    # fill_outer_boundary(current_state, dx, N, r_is_logarithmic)
    # fill_outer_boundary(rhs, dx, N, r_is_logarithmic)
    
    # rm = get_rm()
    # epsilon = 0.01
    # _rhs_fill_outer_boundary(current_state, dx, N, r_is_logarithmic, t_i, rm, omega, epsilon)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(current_state, dx, N, r_is_logarithmic)
    fill_inner_boundary(rhs, dx, N, r_is_logarithmic)
    
    # t5 = time.time()
    # print("Fill boundaries done in ", t5 - t4) 
                
    #################################################################################################### 
    
    # Some code for checking timing and progress output
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress bar]
    # its values can be carried between function calls throughout the ODE integration
    last_t, deltat = time_state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded number:
    time_state[0] = last_t + deltat * n
    
    # t6 = time.time()
    # print("Check timing and output ", t6 - t5) 
    
    # end_time = time.time()
    # print("total rhs time at t= ", t_i, " is, ", end_time-start_time)
        
    ####################################################################################################
    
    #Finally return the rhs
    return rhs


