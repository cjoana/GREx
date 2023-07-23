# initial data file

import sys
# sys.path.append("/home/admin/git/GREx/engrenage_MSPBH/")
sys.path.append("../")

from source.uservariables import *
from source.tensoralgebra import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.gridfunctions import *
from source.misnersharp import *

import numpy as np
import scipy.optimize as opt
# from scipy.optimize import bisect, root
#from scipy.interpolate import interp1d


# INITIAL PARAMS 

omega = 1./3

t_ini = 1
alpha = 2./(3.*(1.+omega))
H_ini = alpha/t_ini
rho_bkg_ini = 3./(8.*np.pi) *H_ini**2
a_ini = 1
R_max = 100 * H_ini

nu = 0.9
fNL = -0.8
n_Horizons = 30
k_star = 1./(n_Horizons*H_ini)

# print(f'H_ini in {H_ini}')

def get_zeta(r):  # curvature 
	
	zeta =  nu * np.sinc(k_star*r) + 3/5*fNL*nu**2 * (np.sinc(k_star*r))**2
	return zeta
	
def get_dzetadr(r):  # d/dr curvature 
	
	A = nu
	B = 3/5*fNL*nu**2
	k = k_star
	x = r
	
	dr_zeta = ((k*x*np.cos(k*x) - np.sin(k*x)) * \
			  (A + 2*B*np.sinc(k*x)))/(k*x**2)
		
	""" 
	Wolfram alpha: 
	d/dx(A sinc(x k) + B sinc(x k)^2) = 
			((k x cos(k x) - sin(k x)) (A + 2 B sinc(k x)))/(k x^2)
	"""			
	
	return dr_zeta
	
def get_d2zetadr2(r):  # d2/dr2 curvature 
	
	A = nu
	B = 3/5*fNL*nu**2
	k = k_star
	x = r
	
	d2dr2_zeta =  A * k * ((2.* np.sin(k * x))/(k**2 * x**3) - \
				  (2.* np.cos(k*x))/(k*x**2) -  np.sin(k*x)/x) + \
				  B*(2.*k * np.sinc(k*x) * ((2*np.sin(k*x))/(k**2*x**3) - \
				  (2*np.cos(k*x))/(k*x**2) - np.sin(k*x)/x) +  \
				  2.*k**2 * (np.cos(k*x)/(k*x) - np.sin(k*x)/(k**2*x**2))**2)
				  
				  
	""" 
	Wolfram alpha: 
	d^2/dx^2(A sinc(x k) + B sinc(x k)^2) = 
		A k ((2 sin(k x))/(k^2 x^3) 
		- (2 cos(k x))/(k x^2) - sin(k x)/x) 
		+ B (2 k sinc(k x) ((2 sin(k x))/(k^2 x^3)
		- (2 cos(k x))/(k x^2) - sin(k x)/x) 
		+ 2 k^2 (cos(k x)/(k x) - sin(k x)/(k^2 x^2))^2)
	"""
	
	return d2dr2_zeta

	
def get_L_pert(a, rm):
	
	zeta_at_rm = get_zeta(rm)
	L = a * rm * np.exp(zeta_at_rm)
	return L 
	
def get_epsilon(H, L):
	
	epsilon = 1/(H*L)
	return epsilon 

def get_scalefactor(t, omega):

	alpha = 2./(3.*(1.+omega))
	a = a_ini*(t/t_ini)**alpha 
	return a 

def get_Hubble(t, omega):
	
	alpha = 2./(3.*(1.+omega))
	Hubble = alpha/t
	return Hubble
	
def get_rho_bkg(t_over_t_ini, rho_bkg_ini):
	# Assumes FLRW evolution
	rho_bkg = rho_bkg_ini * t_over_t_ini**(-2)
	return rho_bkg
	

# Pertrubative equations (tildes) for initial data


def get_tilde_rho(r, rm, omega):
	
	zeta_rm = get_zeta(rm)
	zeta = get_zeta(r)
	d2dr2_zeta = get_d2zetadr2(r)
	dzetadr = get_dzetadr(r)
	
	trho  = - 2*(1+omega)/(5+3*omega) * \
			np.exp(2*( zeta_rm - zeta )) * ( \
			d2dr2_zeta + dzetadr * (2/r + 0.5*dzetadr)*rm**2
			)
	
	return trho
	
	
def get_tilde_U(r, rm, omega):
	
	zeta_rm = get_zeta(rm)
	zeta = get_zeta(r)
	d2dr2_zeta = get_d2zetadr2(r)
	dzetadr = get_dzetadr(r)
	
	tilde_U  = - 1./(5+3*omega) * \
			np.exp(2*( zeta_rm - zeta )) * dzetadr * rm**2 * \
			(2/r + dzetadr)
	
	return tilde_U
	
	
def get_tilde_M(r, rm, omega):
	
	tilde_U = get_tilde_U(r, rm, omega)
	
	tilde_M = -3*(1+omega)*tilde_U
	
	return tilde_M
	
def get_tilde_R(r, rm, omega):
	
	tilde_rho = get_tilde_rho(r, rm, omega)
	tilde_U = get_tilde_U(r, rm, omega)
	
	tilde_R = - omega/(1+3*omega)/(1+omega) * tilde_rho + \
			  1./(1+3*omega) * tilde_U
			  
	return tilde_R
	
# initial data functions 

def get_expansion_R(t, r, rm, omega, epsilon):
	
	a = get_scalefactor(t, omega)
	zeta = get_zeta(r)
	tilde_R = get_tilde_R(r, rm, omega)
	
	out_R = a*np.exp(zeta)*r*(1 + epsilon**2 * tilde_R)
	return out_R
	
def get_expansion_U(t, r, rm, omega, epsilon):
	
	H = get_Hubble(t, omega)
	tilde_U = get_tilde_U(r, rm, omega)
	R = get_expansion_R(t, r, rm, omega, epsilon)
	
	out_U = H*R * (1 + epsilon**2 * tilde_U)
	return out_U
	
def get_expansion_rho(t, r, rm, omega, epsilon):
	
	t_over_t_ini = t/t_ini
	rho_bkg = get_rho_bkg(t_over_t_ini, rho_bkg_ini)
	tilde_rho = get_tilde_rho(r, rm, omega)
	
	out_rho = rho_bkg * (1 + epsilon**2 * tilde_rho)
	return out_rho


def get_expansion_M(t, r, rm, omega, epsilon):
	
	t_over_t_ini = t/t_ini
	rho_bkg = get_rho_bkg(t_over_t_ini, rho_bkg_ini)

	R = get_expansion_R(t, r, rm, omega, epsilon)
	tilde_M = get_tilde_M(r, rm, omega)
	
	out_M= 4*np.pi/3 * rho_bkg * R**3 * (1 + epsilon**2 * tilde_M)
	return out_M		



def compact_function(M, R, rho_bkg):
	C = (8./3.)*np.pi*rho_bkg * R**2 - 2*M/R  
	return C


def get_rm(print_epsilon=0):
	
	def _root_func(r) :
		dz = get_dzetadr(r)
		ddz = get_d2zetadr2(r)
		
		return dz + r * ddz
		
	a, b = [0.01, 100]
	xs = np.linspace(a, b, 100)
	ys = _root_func(xs)
	sa = np.sign(ys[0])
	idx = np.where( np.sign(ys)*sa < 0)[0] 
	b = xs[idx][0]	
	
	# rm = opt.bisect(_root_func, 1e-3, R_max)
	rm = opt.brentq(_root_func, a, b)
	# print(f"rm is {rm}")
	
	L = get_L_pert(1, rm)
	eps = get_epsilon(H_ini, L)
	if print_epsilon: print(f"epsilon is {eps}")
	
	
	return rm 


def get_expansion_Compaction(r, omega):
	dzeta = get_dzetadr(r)
	C = 3 * (1+omega)/(5+3*omega) * (1 - (1 + r*dzeta)**2) 
	return C
	
	


# Function to get initial state 

def get_initial_state(R_max, N_r, r_is_logarithmic) :
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R_max, N_r, r_is_logarithmic)
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    initial_state = np.zeros(NUM_VARS * N)
    
   
    # fill all positive values of r
    for ix in range(num_ghosts, N) :

        # position on the grid
        r_i = r[ix]
        
        
        rm = get_rm() ####
        omega = get_omega() 
        Hubble = get_Hubble(t_ini, omega) 
        L_pert = get_L_pert(a_ini, rm) 
        epsilon = get_epsilon(Hubble,L_pert)
		

        # initial values
        initial_state[ix + idx_U * N] = get_expansion_U(t_ini, r_i, rm, omega, epsilon)
        initial_state[ix + idx_R * N] = get_expansion_R(t_ini, r_i, rm, omega, epsilon)
        initial_state[ix + idx_M * N] = get_expansion_M(t_ini, r_i, rm, omega, epsilon)
        initial_state[ix + idx_rho * N] = get_expansion_rho(t_ini, r_i, rm, omega, epsilon)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(initial_state, dx, N, r_is_logarithmic)
                           
    return r, initial_state
    
    




#########################################################
#### Test  Plot Zeta, Compfaction (C), and M, R

Do_zeta_C_rm_test = False 
if Do_zeta_C_rm_test: 
	x = np.linspace(0.001, 150, 200)
	y = get_zeta(x)

	rm = 1 # get_rm()

	import matplotlib.pyplot as plt

	def _root_func(r) :
			dz = get_dzetadr(r)
			ddz = get_d2zetadr2(r)
			
			return dz + r * ddz

	y2 = get_expansion_Compaction(x, 1./3.)
	y3 = _root_func(x)

	y  = y/y.max()
	y2 = y2/y2.max()
	y3 = y3/y3.max()

	rm = get_rm()

	plt.plot(x, y, label=r"$\zeta$")
	plt.plot(x, y2, label="C")
	plt.plot(x, y3, label="root func")
	plt.axvline(rm, color="k", ls="--", label="rm")
	plt.legend()
	plt.show()


	rho_bkg = get_rho_bkg(1, rho_bkg_ini)
	M = lambda r:  get_expansion_M(1, r, rm, 1./3., 0.001)
	R = lambda r: get_expansion_R(1, r, rm, 1./3., 0.001)
	C = lambda r : compact_function(M(r), R(r), rho_bkg)

	y  = M(x)/R(x)
	y2 = y2
	y3 = R(x)
	y4 = C(x)

	y  = y/y.max()
	y2 = y2/y2.max()
	y3 = y3/y3.max()
	y4 = y4/y4.max()

	x = x/rm
	nrm = 1 #rm  

	plt.plot(x, y, label=r"$M$")
	plt.plot(x, y2, label="C")
	plt.plot(x, y4, label="C2")
	plt.plot(x, y3, label="R")
	plt.axvline(nrm, color="k", ls="--", label="rm")
	plt.legend()
	plt.show()

########
#  Test get initial state: 

Do_initial_state_test = True
if Do_initial_state_test: 
	
	import matplotlib.pyplot as plt
	r_is_logarithmic = 0
	N_r = 100
	
	r, initial_state = get_initial_state(R_max, N_r, r_is_logarithmic)

	plt.plot(initial_state)
	plt.yscale('log')
	plt.show()


#########################################################
