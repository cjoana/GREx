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


# import homemade code
sys.path.append('./engrenage_MSPBH/')
sys.path.append('../')

 
# from source._par_rhsevolution import *  # go here to look at how the evolution works
from source.rhsevolution import *             # go here to look at how the evolution works
from source.initialdata import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

# Input parameters for grid and evolution here
N_r = 600 # num points on physical grid
# R_max = 300.0 * 0.5 # Maximum outer radius


omega = 1./3

t_ini = 1
alpha = 2./(3.*(1.+omega))
H_ini = alpha/t_ini
rho_bkg_ini = 3./(8.*np.pi) *H_ini**2
a_ini = 1
# R_max = 150 * H_ini

# R_max = 400 * H_ini

from source.initialdata import R_max

# nu = 0.9
# fNL = -1
# n_Horizons = 10
#k_star = 1./(n_Horizons*H_ini)

# k_star = (n_Horizons/H_ini/2.7471)**-1 



r_is_logarithmic = False
r, initial_state = get_initial_state(R_max, N_r, r_is_logarithmic)

#unpackage the vector for readability
(initial_U, initial_R , initial_M, initial_rho) = unpack_state(initial_state, N_r)

rm =  get_rm()

dx = R_max/N_r
oneoverdx  = 1.0 / dx
oneoverdxsquared = oneoverdx * oneoverdx

if True:
	
	fig, axs = plt.subplots(2,4, figsize=(17,8))

	U, R, M, rho = unpack_state(initial_state, N_r)
		
	# var = idx_R
	# R = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
	# var = idx_M
	# M = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
	
	rho_bkg = get_rho_bkg(t_ini, rho_bkg_ini)
	C = compact_function(M, R, rho_bkg)
	
	dMdr = get_dfdx(M, oneoverdx)
	dRdr = get_dfdx(R, oneoverdx)
			
	HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5		
	# Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr))
	Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)
	
	
	# f_t = 2*M/R
	f_t = C
	
	r_over_rm = r /rm
	
	ln = len(r)
	ax = axs[0,0]
	ax.plot(r_over_rm[:], f_t[:],'k')
	#
	Rprime_over_a_ezeta = ( 1 + r * get_dzetadr(r) ) 
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
	R_v2 = get_expansion_R(t_ini, r, rm, omega, 0, t_ini=1)
	zeta = get_zeta(r)
	R_v2 = a_ini * np.exp(zeta) *r
	ax.plot(r_over_rm[:], R_v2, 'r--')
	#
	ax.set_ylim(0.1,1e4)
	# ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_ylabel('R')
	ax.set_xlim(0,8)
	
	
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
	Rprime = ( 1 + r * get_dzetadr(r) ) * a_ini * np.exp(get_zeta(r))      # R = a ezeta(r) * r ,   Rprime =  a ezeta r zetaprime + a ezeta = (1 + r zetaprime) * a ezeta
	ax.plot(r_over_rm[:], Rprime, 'r--')
	#
	# ax.set_ylim(0.1,1e4)
	ax.set_ylabel('dRdr')
	ax.set_xlabel('r')
	# ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlim(0,8)
		
	ax = axs[1,3]
	f_t = Ham
	ax.plot(r_over_rm[:], f_t[:])
	ax.set_ylabel('Ham')
	ax.set_yscale('log')
		
	# ax = axs[1,3]
	# zeta = get_zeta(r)
	# p_zeta =  get_dzetadr(r) 
	# pp_zeta =  get_d2zetadr2(r) 
	# #
	# ax.plot(r_over_rm[:], zeta, 'b-')
	# ax.plot(r_over_rm[:], p_zeta, 'r-')
	# ax.plot(r_over_rm[:], pp_zeta, 'g-')
	# ax.set_ylabel('zeta, zeta_prime')
	# # ax.set_xlim(0,1)
	# # ax.set_yscale('log')
	
	plt.tight_layout()
	plt.savefig('initial_data_example.png', dpi=100)
	plt.clf()
	plt.close()


if True :
	#plot initial conditions
	plt.xlabel('r')
	plt.plot(r, initial_U, '-o', label='U') # zero, but plot as dots to see the grid
	plt.plot(r, initial_R, label='R')
	plt.plot(r, initial_M, label='M')
	plt.plot(r, initial_rho, label='rho')
	plt.legend(loc='best')
	plt.grid()
	#plt.xlim(-0.25,5.0)
	#plt.ylim(-0.0005,0.0005)
	plt.yscale('log')
	
	plt.savefig('init_data_example.png', dpi=100)
	# plt.show()
	plt.clf()
	
	# initial_M = initial_M /100
	
	
	Gamma = np.sqrt(1 + initial_U**2 - 2*initial_M/initial_R)
	
	AH = 2*initial_M/initial_R
	
	C = compact_function(initial_M, initial_R, rho_bkg_ini)
	
	r_over_rm = r/rm
	
	plt.plot(r_over_rm, C, 'b-', label="C")
	plt.plot(r_over_rm, AH, 'k-', label="AH")
	plt.plot(r_over_rm, Gamma, 'g-', label=r"$\Gamma$")

	# dx, N, r, logarithmic_dr = setup_grid(R_max, N_r, r_is_logarithmic)
	dx = R_max/N_r
	oneoverdx  = 1.0 / dx
	oneoverdxsquared = oneoverdx * oneoverdx

	dRdr = get_dfdx(initial_R, oneoverdx)
	plt.plot(r_over_rm, dRdr, 'r-', label=r"dRdr")

	 
	plt.ylim(-2, 3.0)
	# plt.xlim(0, r_over_rm.max())
	# plt.ylabel('C')
	plt.xlabel('r/rm')
	plt.legend()
	plt.savefig('init_compact_example.png', dpi=100)
	# plt.show()
	plt.clf()
	
	
	dMdr = get_dfdx(initial_M, oneoverdx)

				
	HamAbs = ((dMdr)**2 + (4*np.pi*initial_rho*initial_R**2 * dRdr)**2)**0.5		
	Ham = (dMdr - 4*np.pi*initial_rho*initial_R**2 * dRdr) / HamAbs
	
			
	plt.plot(r_over_rm, np.abs(Ham), 'r-', label=r"Ham Rel")
	plt.xlabel('r/rm')
	plt.legend()
	plt.yscale('log')
	plt.savefig('Ham_initial.png', dpi=100)
	
	




# check the Hamiltonian constraint initially satisfied
# apart from numerical errors
# r, Ham = get_Ham_diagnostic(initial_state, np.array([0]), R_max, N_r, r_is_logarithmic)
# plot the profile for Ham
# plt.plot(r, Ham[0])
# plt.xlabel('r')
# plt.xlim(-4.0,R_max+4.0)
# plt.ylim(-0.01,0.01)
# plt.ylabel('Ham value')
# plt.grid()
# plt.show

import time

start = time.time()
# for control of time integrator and spatial grid
t_ini = 1.0 
dx = R_max/N_r
dt_multiplier = 0.01
dt = dx * dt_multiplier
N_t = 50000
T  = t_ini + dt * N_t
# T = 10*n_Horizons
# T = 2.0 # Maximum evolution time
# N_t = 10 # time resolution (only for outputs, not for integration)

# Work out dt and time spacing of outputs
# dt = T/N

sigma = 1./ dt_multiplier # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R_max/N_r  
# if(max_step > R_max/N_r/sigma): print("WARNING: kreiss oliger condition not satisfied!")


print("initial params:")
print(f" R_max = {R_max}\n N_r = {N_r}\n dx = {dx}\n dt = {dt}\n N_t ={N_t}\n T = {T}\n rm ={rm}\n rho_bkg_ini = {rho_bkg_ini}\n num Horizons = {n_Horizons}")


# raise()

# Solve for the solution using RK45 integration of the ODE
# to make like (older) python odeint method use method='LSODA' instead
# use tqdm package to track progress
with tqdm(total=N_t, unit=" ") as progress_bar:
    dense_solution = solve_ivp(get_rhs, [t_ini,T], initial_state, 
                               args=(R_max, N_r, r_is_logarithmic, sigma, progress_bar, [t_ini, dt]),
                        # atol=1e-8, rtol=1e-6,
                        atol=1e-80, rtol=1e-60,
                        max_step= dt, #for stability and for KO coeff of 10
                        method='RK45',
                        # method='LSODA',
                        dense_output=True)

# Interpolate the solution at the time points defined in myparams.py
num_tslides = 5
t_out = dense_solution.t
t_sol = np.linspace(t_out[0], t_out[-1], num_tslides)

solution = dense_solution.sol(t_sol).T

end = time.time() 

print(f"Time needed for evolution {end-start} seconds.")
print(dense_solution.message)
print("\n status \n ", dense_solution.status)




rm = get_rm(print_out=1)


############################################################

if False: 
	# Plot a single point versus time  at location r_i
	var1 = idx_U 
	var2 = idx_R
	var3 = idx_M
	var4 = idx_rho

	t = t_sol

	idx = num_ghosts+1
	r_i = np.round(r[idx],2)
	var1_of_t = solution[:, var1 * (N_r + 2*num_ghosts) + idx]

	plt.plot(t[:], var1_of_t[:], 'b-', label=variable_names[var1])

	plt.xlabel('t')
	plt.ylabel('value at r is '+str(r_i))
	plt.legend(loc='best')
	plt.grid()
	plt.show()


#######################################################


if True:
	
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
		
		rho_bkg = get_rho_bkg(t_i, rho_bkg_ini)
		C = compact_function(M, R, rho_bkg)
		
		dMdr = get_dfdx(M, oneoverdx)
		dRdr = get_dfdx(R, oneoverdx)
				
		HamAbs = ((dMdr)**2 +  (4*np.pi*rho*R**2 * dRdr)**2)**0.5		
		Ham = np.abs((dMdr - 4*np.pi*rho*R**2 * dRdr) / HamAbs)
		
		
		# f_t = 2*M/R
		f_t = C
		
		r_over_rm = r/rm
		xmin, xmax = [r_over_rm[0]  ,r_over_rm[-4]]
		mask = (r_over_rm)<8
				
		ln = len(r)
		ax = axs[0,0]
		ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
		ax.set_ylim(-2,3)
		ax.set_ylabel('C')
		ax.set_xlim(xmin,xmax)
		
		ax = axs[0,1]
		f_t = 2*M/R
		ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
		ax.set_ylim(0,1.5)
		ax.set_xlim(0,1.5)
		ax.legend()
		ax.set_ylabel('M/R')
		ax.set_xlim(xmin,xmax)
		
		ax = axs[1,0]
		f_t = R
		ax.plot(r_over_rm[mask], f_t[mask], label=labelt)
		ax.set_ylim(0.1,1e4)
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













