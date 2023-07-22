import numpy as np
import time




def get_rho_bkg(t_over_t_ini, rho_bkg_ini):
	# Assumes FLRW evolution
	rho_bkg = rho_bkg_ini * t_over_t_ini**(-2)
	return rho_bkg

def get_A(rho, rho_bkg, omega): 
	
	A = (rho_bkg/rho)**(omega/(omega+1))
	return A

def get_B(U, R, M, dRdr):
	
	Gamma = get_Gamma(U, R, M)
	B = dRdr/Gamma
	return B 


def get_Gamma(U, R, M):
	
	Gamma = np.sqrt(1 + U**2 - 2*M/R)
	return Gamma
	

# RHS equations 
	
def get_rhs_U(U, M, R, rho, drhodr, dRdr, A, omega):
	
	Gamma = get_Gamma(U, R, M)
	dUdt = - A * ( omega/(omega+1) * (drhodr * Gamma**2)/(rho * dRdr) + M/R**2 + 4*np.pi*R*omega*rho)
	return dUdt
	
def get_rhs_R(U, A):
	
	dRdt = A * U
	return dRdt

def get_rhs_rho(U, R, rho, dUdr, dRdr, A, omega):
	
	drhodt = - A*rho*(1+omega)*(2*U/R + dUdr/dRdr)
	return drhodt

def get_rhs_M(U, R, rho, A, omega):
	
	dMdt = -4*np.pi*A*omega*rho*U*R**2
	return dMdt


def get_omega():
	
	omega = 1./3.
	return omega




####  code below by Albert Escriva
#######################################################

# # Evolution of bkg vars
# def solution_FLRW(t):

	# w =1./3.#EQ. of state
	# t_initial = 1.0 #initial time
	# alpha = 2./(3.*(1.+w))
	# #numerical initial conditions(background quantities)
	# H_bI = alpha/(t_initial) #Initial Hubble constant
	# e_bI = (3./(8.*pi))*H_bI**2 #initial energy density of the background
	# a_I = 1. #initial scale factor 
	# RHI = 1/H_bI # initial cosmological horizon 


	# a_FLRW = a_I*(t/t_initial)**alpha 			#scale factor
	# H_FLRW = alpha/t 							#Hubble factor
	# e_FLRW = e_bI*(t_initial/t)**2 				#energy density of the background
	# R_FLRW = a_FLRW*x 							#areal radious 
	# U_FLRW = R_FLRW*H_FLRW						#euler velotity 
	# M_FLRW = (4.*pi/3.)*e_FLRW*R_FLRW**3		#mass of the bakground
	# A_FLRW = 1. 								#lapse function
	# G_FLRW = 1. 								#gamma function
	# return e_FLRW,R_FLRW,U_FLRW,M_FLRW,A_FLRW,G_FLRW
	
	
# def system_static(epc,Mpc,Rpc,Upc,e_FRWc):
	# Aqq = 1.*(e_FRWc/epc)**(w/(w+1.))
	# fraction = Mpc[:-1]/Rpc[:-1]
	# fraction = np.insert(fraction, len(fraction), 0.)
	# Gqq = np.sqrt(1+Upc**2-2.*(fraction))
	# return Aqq,Gqq
	
	
# # The Misner-Sharp equations are set up
# def system_dynamic_RK(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp):
	# #Note that the point r=0 in defined 
	# fraction = Mp[:-1]/Rp[:-1]**2
	# fraction = np.insert(fraction, len(fraction), 0.)
	# Ut = -Ap*( 4.*pi*Rp*w*ep + fraction +  (w/(1.+w))*(devep*Gp**2)/(ep*devRp) )
	# Rt = Up*Ap
	# Mt = -Ap*4.*pi*w*ep*Up*Rp**2
	# derU_R = devUp/devRp
	# ratioUR = Up[:-1]/Rp[:-1]
	# ratioUR = np.insert(ratioUR, len(ratioUR), derU_R[-1])
	# et = -Ap*ep*(1.+w)*(2.*ratioUR+devUp/devRp)
	# return Ut,Rt,Mt,et
	











