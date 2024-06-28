import sys
# sys.path.append("/home/admin/git/GREx/engrenage_MSPBH/")
sys.path.append("../")

import numpy as np
import time
from  source.gridfunctions import num_ghosts


print_Gamma_set_zero = 0
print_A_set_one = 0


def get_rho_bkg(t_over_t_ini, rho_bkg_ini):
	# Assumes FLRW evolution
	rho_bkg = rho_bkg_ini * t_over_t_ini**(-2)
	return rho_bkg

def get_A(rho, rho_bkg, omega): 
	
	
	if rho < 0: rho = rho_bkg
	
	A = (rho_bkg/rho)**(omega/(omega+1))
	
	if A!=A:
		if print_A_set_one: print("A imposed 1!! with", rho_bkg, rho, omega  )
		A = 1
	
	return A

def get_B(U, R, M, dRdr):
	
	Gamma = get_Gamma(U, R, M)
	B = dRdr/Gamma
	return B 


def get_Gamma(U, R, M):
	
	disc = 1 + U**2 - 2*M/R
	disc = disc if disc==disc else 0
	
	if disc <0: 
		# print("Gamma imposed zero!! with U,M, R ", U, M, R, "and U**2, 2M/R, diff ", U**2, 2*M/R,  U**2 - 2*M/R,  )
		if print_Gamma_set_zero: print("Gamma imposed zero!! with sqrt of ", 1 + U**2 - 2*M/R,  )
		disc = 0
		Gamma = 0
	else: Gamma = np.sqrt(disc)

	return Gamma
	

# RHS equations 
	
def get_rhs_U(U, M, R, rho, dRdr, drhodr, A, Gamma, omega):
		
	dUdt = - A * ( omega/(omega+1) * (drhodr * Gamma**2)/(rho * dRdr) + M/R**2 + 4*np.pi*R*omega*rho)
	
	img = np.imag(dUdt)
	if img != 0:     ## this happens when rho <0 : now we impsoe rho[rho<0] = np.min(np.abs(rho)) during evolution.
		print( "rhs U got imaginary", dUdt)
		print(f"U = {U}")
		print(f"M = {M}")
		print(f"R = {R}")
		print(f"rho = {rho}")
		print(f"dRdr = {dRdr}")
		print(f"A = {A}")
		print(f"Gamma = {Gamma}")
		print(f"omega = {omega}")
	
	dUdt = np.real(dUdt)
	
	dUdt = dUdt if dUdt==dUdt else 0
	
	return dUdt
	
def get_rhs_R(U, A):
	
	dRdt = A * U
	return dRdt


def get_rhs_rho(U, R, rho, dUdr, dRdr, A, omega):
	
	# drhodt = - A*rho*(1+omega)*(2*U/R + dUdr/dRdr)
	
	fraction = dUdr/ (dRdr) if np.abs(dRdr) >1e-4 else - 2*U/R
	
	drhodt = - A*rho*(1+omega)*(2*U/R + fraction)
	
	drhodt = drhodt if drhodt==drhodt else 0
	
	return drhodt


def get_rhs_M(U, R, rho, A, omega):
	
	dMdt = - 4*np.pi*A*omega*rho*U*R**2
	
	dMdt = dMdt if dMdt==dMdt else 0
	return dMdt



##########################################################################
### FRW and other quantities
############################################


def get_scalefactor(t, omega, a_ini, t_ini):

	alpha = 2./(3.*(1.+omega))
	a = a_ini*(t/t_ini)**alpha 
	return a 

def get_Hubble(t, omega, t_ini=1):
	
	alpha = 2./(3.*(1.+omega))
	Hubble = alpha/(t/t_ini)
	return Hubble
	
def get_rho_bkg(t_over_t_ini, rho_bkg_ini):
	# Assumes FLRW evolution
	rho_bkg = rho_bkg_ini * t_over_t_ini**(-2)
	return rho_bkg
	



##################################################

def compact_function(M, R, rho_bkg):
	C =  2*M/R  - (8./3.)*np.pi*rho_bkg * R**2 
	return C

#END
