

# SPriBHoS code (Spectral Primordial Black Hole Simulator). Code made by Albert Escriva. See webpage https://sites.google.com/fqa.ub.edu/albertescriva/home for more details. The code is based on https://arxiv.org/abs/1907.13065

#We import the numeric libraries needed for the code.
from numpy import *
import numpy as np
import math
from sympy import *
pi = math.pi
import time
import sys

#External modulus
from Dmatrix import chebymatrix
from curvature_profiles import curvature_profile
from curvature_profiles import derivative_curvature_profile
from threshold import A_versus_threshold


start_time = time.time()

#Here we set up the initial variables and magnitudes.
w =1./3.#EQ. of state
t_initial = 1.0 #initial time
alpha = 2./(3.*(1.+w))
#numerical initial conditions(background quantities)
H_bI = alpha/(t_initial) #Initial Hubble constant
e_bI = (3./(8.*pi))*H_bI**2 #initial energy density of the background
a_I = 1. #initial scale factor 
RHI = 1/H_bI # initial cosmological horizon
Nh = 90 #number of initial horizon radious, to set up the final point of the grid


r_initial=0.0 #initial point of the grid
r_final = Nh*RHI #final point of the grid, given by the mass enclosed for the given radious

a = r_initial
b = r_final


dt0 = 10**(-3.)#initial time-step
t_final = 80000. #final time of the simulation. This is used as an "exit" in the bisection method
t = t_initial


#Differentiation matrix pseudospectral method
N_cheb = 400 #Number of chebyshev points used
vector_ones = np.array([1. for l in range(N_cheb+1)],dtype=np.float64)
D,x = chebymatrix(N_cheb,a,b) #we get the chebychev differetiaiton matirx and the grid x


rm_N = 10. #number of initial cosmological horizon that we put the length scale of the perturbtion rk. The long wavelength approximation must be fulfilld! Take rm_N always such that epsilon<0.1

error = 10**(-3)

#Minimum and maximum thresholds allowed. Case for radiation fluid
thresh_min = 2./5.
thresh_max = 2./3.


thresh_limit_yes = thresh_max
thresh_limit_no = thresh_min

vector_ones = np.array([1. for l in range(N_cheb+1)],dtype=np.float64) 

print ("Welcome to the primordial black hole simulator, the simulation is done with the following parameters:")
print ("N_cheb:",N_cheb)
print ("dt0:",dt0)
print ("Nh:",Nh)
print ("rm_N",rm_N)
print ("error allowed in bisection" , error)
print ("The simulation is in process")

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# The evolution of the FRW magnitudes in time
def solution_FRW(t):
	a_FRW = a_I*(t/t_initial)**alpha #scale factor
	H_FRW = alpha/t #Hubble factor
	e_FRW = vector_ones*e_bI*(t_initial/t)**2 #energy density of the background
	R_FRW = a_FRW*x #areal radious 
	U_FRW = R_FRW*H_FRW #euler velotity 
	M_FRW = (4.*pi/3.)*e_FRW*R_FRW**3 #mass of the bakground
	A_FRW = 1.*vector_ones #lapse function
	G_FRW = 1.*vector_ones #gamma function
	return e_FRW,R_FRW,U_FRW,M_FRW,A_FRW,G_FRW
	

def energy_FRW(t):
	e_FRW = vector_ones*e_bI*(t_initial/t)**2
	return e_FRW
# Dynamical magnitudes


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#The main numerical code is set up in the three following functions
#--------------------------------------------------------------------------------
# The Misner-Sharp equations are set up
def system_dynamic_RK(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp):
	#Note that the point r=0 in defined 
	fraction = Mp[:-1]/Rp[:-1]**2
	fraction = np.insert(fraction, len(fraction), 0.)
	Ut = -Ap*( 4.*pi*Rp*w*ep + fraction +  (w/(1.+w))*(devep*Gp**2)/(ep*devRp) )
	Rt = Up*Ap
	Mt = -Ap*4.*pi*w*ep*Up*Rp**2
	derU_R = devUp/devRp
	ratioUR = Up[:-1]/Rp[:-1]
	ratioUR = np.insert(ratioUR, len(ratioUR), derU_R[-1])
	et = -Ap*ep*(1.+w)*(2.*ratioUR+devUp/devRp)
	return Ut,Rt,Mt,et
# Dynamical equations of the Runge-Kutta 4 method. We evolve M,R,U and rho
def system_RK(Upl,Rpl,Mpl,Apl,Gpl,epl,devepl,devRpl,devUpl,dt,t):
	#Note that we set up the boundary conditions in each time step
	k1U,k1R,k1M,k1e = system_dynamic_RK(Upl,Rpl,Mpl,Apl,Gpl,epl,devepl,devRpl,devUpl)

	shifte1 = epl+k1e*0.5*dt
	shiftU1 = Upl+k1U*0.5*dt
	shiftR1 = Rpl+k1R*0.5*dt
	shiftM1 = Mpl+k1M*0.5*dt

	shiftU1[-1] = 0.
	shiftR1[-1] = 0.
	shiftM1[-1] = 0.

	e_FRW1 = energy_FRW(t+0.5*dt-dt)

	devep2,devUp2,devRp2 = compute_derivatives(shifte1  , shiftU1 , shiftR1)
	devep2[0] = 0.
	devep2[-1] = 0.

	Ap2,Gp2 = system_static(shifte1 , shiftM1 , shiftR1 , shiftU1 , e_FRW1)
	k2U,k2R,k2M,k2e = system_dynamic_RK(shiftU1 , shiftR1 , shiftM1 , Ap2 , Gp2 , shifte1 , devep2 , devRp2 , devUp2)

	shifte2 = epl+k2e*0.5*dt
	shiftU2 = Upl+k2U*0.5*dt
	shiftR2 = Rpl+k2R*0.5*dt
	shiftM2 = Mpl+k2M*0.5*dt

	shiftU2[-1] = 0.
	shiftR2[-1] = 0.
	shiftM2[-1] = 0.

	devep3,devUp3,devRp3 = compute_derivatives(shifte2 , shiftU2 , shiftR2)
	devep3[0] = 0.
	devep3[-1] = 0. 

	e_FRW2= energy_FRW(t+0.5*dt-dt)

	Ap3,Gp3 = system_static(shifte2 , shiftM2 , shiftR2 , shiftU2 ,e_FRW2)
	k3U,k3R,k3M,k3e = system_dynamic_RK(shiftU2 , shiftR2 , shiftM2 , Ap3 , Gp3 , shifte2 , devep3 , devRp3 , devUp3)

	shifte3 = epl+k3e*dt
	shiftU3 = Upl+k3U*dt
	shiftR3 = Rpl+k3R*dt
	shiftM3 = Mpl+k3M*dt

	shiftU3[-1] = 0.
	shiftR3[-1] = 0.
	shiftM3[-1] = 0.

	devep4,devUp4,devRp4 = compute_derivatives(shifte3 , shiftU3 , shiftR3)
	devep4[0] = 0.
	devep4[-1] = 0. 

	e_FRW3 = energy_FRW(t+dt-dt)

	Ap4,Gp4 = system_static(shifte3,shiftM3,shiftR3,shiftU3,e_FRW3)
	k4U,k4R,k4M,k4e = system_dynamic_RK(shiftU3,shiftR3,shiftM3,Ap4,Gp4,shifte3,devep4,devRp4,devUp4)

	U = Upl +(1./6.)*dt*(k1U+2.*k2U+2.*k3U+k4U)
	R = Rpl + (1./6.)*dt*(k1R+2.*k2R+2.*k3R+k4R)
	M = Mpl + (1./6.)*dt*(k1M+2.*k2M+2.*k3M+k4M)
	e = epl + (1./6.)*dt*(k1e+2.*k2e+2.*k3e+k4e)
	return U,R,M,e
# We solve the magnitudes A,G using the previous variables rho,U,M,R got from the RK method.
def system_static(epc,Mpc,Rpc,Upc,e_FRWc):
	Aqq = 1.*(e_FRWc/epc)**(w/(w+1.))
	fraction = Mpc[:-1]/Rpc[:-1]
	fraction = np.insert(fraction, len(fraction), 0.)
	Gqq = np.sqrt(1+Upc**2-2.*(fraction))
	return Aqq,Gqq
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# Initial perturbation magnitudes, we only need supply K and its derivative K'
def initial_perturbation_magnitudes(Kk,Kderk,rmk):
	dek = (3.*(1.+w)/(5.+3.*w))*(Kk+(x/3.)*Kderk)*rmk**2
	dUk = -(1./(5.+3.*w))*Kk*rmk**2
	dAk = -(w/(1+w))*dek
	dMk = -3.*(1+w)*dUk
	dRk = -dek*w/((1+3.*w)*(1.+w))+dUk/(1+3.*w)
	return dek,dUk,dAk,dMk,dRk

#We set up the initial conditions of the simulation
def initial_conditions(epskk,dekk,dRkk,dUkk,dAkk,dMkk):
	e_Ikk = e_bI*(1*vector_ones+dekk*(epskk**2))
	R_Ikk = a_I*x*(1*vector_ones+dRkk*(epskk**2))              
	U_Ikk = H_bI*R_Ikk*(1*vector_ones+dUkk*(epskk**2))
	A_Ilkk = 1*vector_ones+dAkk*(epskk**2)	
	M_Ikk = ((4.*pi)/3.)*e_bI*(1*vector_ones+dMkk*(epskk**2))*R_Ikk**3
	fraction = M_Ikk[:-1]/R_Ikk[:-1]
	fraction = np.insert(fraction, len(fraction), 0.)
	G_Ikk = np.sqrt(1.*vector_ones+U_Ikk**2-2.*fraction)
	putaaa = (energy_FRW(1.)/e_Ikk)**(w/(w+1.))
	#G_I = np.sqrt(1-K*x**2)
	return e_Ikk,R_Ikk,U_Ikk,A_Ilkk,M_Ikk,G_Ikk

#The derivatives at each point are computed using Chebyshev differentiation matrix. 
def compute_derivatives(err,Urr,Rrr):
	deve =  np.dot(D,err)
	devU =  np.dot(D,Urr)
	devR =  np.dot(D,Rrr)
	return deve,devU,devR

#we get the epsilon parameter
def epsilon_horizon_crosing(rmNt):
	rm_total = rmNt*RHI #length scale of the perturbation
	epst = 1./(a_I*H_bI*rm_total)
	tHt = t_initial*rmNt**2   #value of the time horizon crossing
	return epst,tHt,rm_total

#we compute the compaction function
def compact_function(Mv,Rv,efrw):
	Cc = 2*(Mv[:-1]-(4./3.)*pi*efrw[:-1]*Rv[:-1]**3)/Rv[:-1]
	Cc = np.insert(Cc, -1, 0.)
	return Cc

# Computation of the variables after each RK cycle
def computation(dt,t,Rpp,Mpp,epp,Upp,App,Gpp):

	e_FRW = energy_FRW(t)

	Rp = Rpp.copy()
	Mp = Mpp.copy()
	ep = epp.copy()
	Up = Upp.copy()
	Ap = App.copy()
	Gp = Gpp.copy()

	devep,devUp,devRp = compute_derivatives(ep,Up,Rp)

	devep[0] = 0. 
	devep[-1] = 0. #we imposs the Neumann boundary conditions

	Upp,Rpp,Mpp,epp = system_RK(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp,dt,t)

	Upp[-1]=0.
	Rpp[-1]=0.
	Mpp[-1]=0.#we impose the Dirichlet boundary conditions
	App,Gpp = system_static(epp,Mpp,Rpp,Upp,e_FRW)

	return Upp,Rpp,Mpp,epp,App,Gpp,e_FRW #we return the final values already computed	


#We check if a BH if formed or not. We give the value of delta to contruct the initial condition
def search(thresh):

	Kbar = curvature_profile(x,rmww)
	Kbarder = derivative_curvature_profile(x,rmww) #we have to add the amplitude of the perturbation

	Am = A_versus_threshold(x,w,thresh,rmww)
#we build the curvature profile \bar{K}. and its derivative, needed to build the initial conditions
	K = Am*Kbar
	Kder = Am*Kbarder

	derr,dUrr,dArr,dMrr,dRrr = initial_perturbation_magnitudes(K,Kder, rmww)
	e_Ie,R_Ie,U_Ie,A_Ile,M_Ie,G_Ie = initial_conditions(epsrr,derr,dRrr,dUrr,dArr,dMrr)
	Rv,Mv,ev,Uv,Av,Gv = R_Ie,M_Ie,e_Ie,U_Ie,A_Ile,G_Ie

	t = t_initial
	while t<t_final: #we set a maximum time (an "exit") to avoid problems

		dt = dt0*np.sqrt(t) #time-step
		t += dt

		Uk,Rk,Mk,ek,Ak,Gk,e_FRW = computation(dt,t,Rv,Mv,ev,Uv,Av,Gv)

		CC = compact_function(Mk,Rk,e_FRW) #we construct the compaction function
		Cmax = max(CC) #we search its maximum


		if t>tH:
		
			if (np.any(np.isnan(Ak)) == True) or (np.any(np.isnan(Uk)) == True) or (np.any(np.isnan(Mk)) == True) or (np.any(np.isnan(ek)) == True) or (np.any(np.isnan(Gk)) == True) or (np.any(np.isnan(Rk)) == True):
				print ("The simulation has broken, provably due to large gradients, at time: ",t)
				return 0
				break
		
			if (Cmax<0.3 and t>tH):
				print ("No BH formation, we proceed with the next bisection iteration")
				return -1
				break
			elif (Cmax>1.0 and t>tH):
				print ("Yes BH formation, we proceed with the next bisection iteration")
				return +1
				break
		elif t<tH:

			if (np.any(np.isnan(Ak)) == True) or (np.any(np.isnan(Uk)) == True) or (np.any(np.isnan(Mk)) == True) or (np.any(np.isnan(ek)) == True) or (np.any(np.isnan(Gk)) == True) or (np.any(np.isnan(Rk)) == True):
				print ("The simulation has broken before the time of horizon crosing, probably because the initial conditions are wrong, the curvature profile is problematic of the stability condition between Ncheb and dt0 is not satisfied, check it")
				sys.exit(0)
				

		
		Rv = Rk.copy()
		Mv = Mk.copy()
		ev = ek.copy()
		Uv = Uk.copy()
		Av = Ak.copy()
		Gv = Gk.copy()


#We stablish the bisection procedure
def bisection(thresh_low, thresh_high, err):
	
	thresh_mid = (thresh_low+thresh_high)/2 #initial guess of the bisection

	comp =0.0

	thresh_limit_yes = thresh_max
	thresh_limit_no = thresh_min

	while(abs(thresh_mid-comp)/2. > err):
		
		A_mid = float(A_versus_threshold(x,w,thresh_mid,rmww))
		print ("We try the value of delta:", thresh_mid)
		value = search(thresh_mid)

		if value > 0:

			thresh_limit_yes = thresh_mid 
			thresh_high = thresh_mid
			comp = thresh_high

		elif value < 0:

			thresh_limit_no = thresh_mid
			thresh_low = thresh_mid
			comp = thresh_low

		elif value == 0: #this shift the interval of bisection to avoid the region of deltas where pressure gradients are strong

			print ("The value of delta has been shifted")
			thresh_low = thresh_low+2*error
			
			thresh_high = thresh_high+2*error
		thresh_mid = (thresh_low+thresh_high)/2
	delta_c = (thresh_limit_yes+thresh_limit_no)/2. #final result of deltac
	print ("The value of the threshold and its resolution is given by:", delta_c ,abs(thresh_limit_yes-delta_c ) )
	return delta_c


#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

epsrr,tH,rmww = epsilon_horizon_crosing(rm_N) #we get the epsilon parametter
bisection(thresh_min, thresh_max , error) #we start with the bisection procedure



print ("Simulation done successfully. The time of the computation was:")
print("--- %s seconds ---"% (time.time()-start_time))


