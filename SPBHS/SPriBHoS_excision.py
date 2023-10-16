

# SPriBHoS code (Spectral Primordial Black Hole Simulator). Code made by Albert Escriva. See webpage https://sites.google.com/fqa.ub.edu/albertescriva/home for more details. The code is based on https://arxiv.org/abs/1907.13065

#THIS IS THE PART OF THE CODE TO COMPUTE THE MASS OF THE PBH. THIS IS AN EXAMPLE TO COMPUTE THE PBH(mass) for the largest allowed mass (in radiation \delta_c =2/3). The simulation takes around 30 minutes.

#We import the numeric libraries needed for the code.
from numpy import *
import numpy as np
import math
from sympy import *
pi = math.pi
import time
import sys


# import interpolation
from scipy import interpolate
from scipy import optimize

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
t_final = 80000. #final time of the simulation. This is used as an "exit" the first part of the code
t = t_initial


#Differentiation matrix pseudospectral method
N_cheb = 1000 #Number of chebyshev points used
vector_ones = np.array([1. for l in range(N_cheb+1)],dtype=np.float64)
D,x = chebymatrix(N_cheb,a,b) #we get the chebychev differetiaiton matirx and the grid x


rm_N = 10. #number of initial cosmological horizon that we put the length scale of the perturbtion rk. The long wavelength approximation must be fulfilld! Take rm_N always such that epsilon<0.1

print ("Welcome to the primordial black hole simulator, the simulation is done with the following parameters:")
print ("N_cheb:",N_cheb)
print ("dt0:",dt0)
print ("Nh:",Nh)
print ("rm_N",rm_N)
print ("The simulation to compute the largest allowed mass for a gaussian profile in radiation is in process")

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

	Rp = Rpp[:]
	Mp = Mpp[:]
	ep = epp[:]
	Up = Upp[:]
	Ap = App[:]
	Gp = Gpp[:]

	devep,devUp,devRp = compute_derivatives(ep,Up,Rp)

	devep[0] = 0. 
	devep[-1] = 0. #we imposs the Neumann boundary conditions

	Upp,Rpp,Mpp,epp = system_RK(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp,dt,t)

	Upp[-1]=0.
	Rpp[-1]=0.
	Mpp[-1]=0.#we impose the Dirichlet boundary conditions
	App,Gpp = system_static(epp,Mpp,Rpp,Upp,e_FRW)

	return Upp,Rpp,Mpp,epp,App,Gpp,e_FRW #we return the final values already computed	






#--------------------------------------------------------------------------------
# In this part of the code, we use the same functions under some modification to proceed with the excision technique once we excise part of the computation domain


#the same difreential eqautions but wihtout impose the boundary conditions
def system_dynamic_RK_excision(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp):
	Ut = -Ap*( 4.*pi*Rp*w*ep + Mp/Rp**2 +  (w/(1.+w))*(devep*Gp**2)/(ep*devRp) )
	Rt = Up*Ap
	Mt = -Ap*4.*pi*w*ep*Up*Rp**2
	et = -Ap*ep*(1.+w)*(2.*Up/Rp+devUp/devRp)

	return Ut,Rt,Mt,et

# Computation of the variables after each RK cycle, for the excision case
def computation_excision(dt,t,Rpp,Mpp,epp,Upp,App,Gpp):

	e_FRW = energy_FRW(t)

	Rp = Rpp[:]
	Mp = Mpp[:]
	ep = epp[:]
	Up = Upp[:]
	Ap = App[:]
	Gp = Gpp[:]

	devep,devUp,devRp = compute_derivatives_excision(ep,Up,Rp)

	devep[0] = 0. 
	devep[-1] = derebc

	Upp,Rpp,Mpp,epp = system_RK_excision(Up,Rp,Mp,Ap,Gp,ep,devep,devRp,devUp,dt,t)

	App,Gpp = system_static_excision(epp,Mpp,Rpp,Upp,e_FRW)

	return Upp,Rpp,Mpp,epp,App,Gpp,e_FRW #we return the final values already computed	


def system_RK_excision(Upl,Rpl,Mpl,Apl,Gpl,epl,devepl,devRpl,devUpl,dt,t):
	#Note that we set up the boundary conditions in each time step
	k1U,k1R,k1M,k1e = system_dynamic_RK_excision(Upl,Rpl,Mpl,Apl,Gpl,epl,devepl,devRpl,devUpl)

	shifte1 = epl+k1e*0.5*dt
	shiftU1 = Upl+k1U*0.5*dt
	shiftR1 = Rpl+k1R*0.5*dt
	shiftM1 = Mpl+k1M*0.5*dt


	e_FRW1 = energy_FRW(t+0.5*dt-dt)

	devep2,devUp2,devRp2 = compute_derivatives_excision(shifte1  , shiftU1 , shiftR1)
	devep2[0] = 0.
	devep2[-1] = derebc

	Ap2,Gp2 = system_static_excision(shifte1 , shiftM1 , shiftR1 , shiftU1 , e_FRW1)
	k2U,k2R,k2M,k2e = system_dynamic_RK_excision(shiftU1 , shiftR1 , shiftM1 , Ap2 , Gp2 , shifte1 , devep2 , devRp2 , devUp2)

	shifte2 = epl+k2e*0.5*dt
	shiftU2 = Upl+k2U*0.5*dt
	shiftR2 = Rpl+k2R*0.5*dt
	shiftM2 = Mpl+k2M*0.5*dt



	devep3,devUp3,devRp3 = compute_derivatives_excision(shifte2 , shiftU2 , shiftR2)
	devep3[0] = 0.
	devep3[-1] = derebc

	e_FRW2= energy_FRW(t+0.5*dt-dt)

	Ap3,Gp3 = system_static_excision(shifte2 , shiftM2 , shiftR2 , shiftU2 ,e_FRW2)
	k3U,k3R,k3M,k3e = system_dynamic_RK_excision(shiftU2 , shiftR2 , shiftM2 , Ap3 , Gp3 , shifte2 , devep3 , devRp3 , devUp3)

	shifte3 = epl+k3e*dt
	shiftU3 = Upl+k3U*dt
	shiftR3 = Rpl+k3R*dt
	shiftM3 = Mpl+k3M*dt


	devep4,devUp4,devRp4 = compute_derivatives_excision(shifte3 , shiftU3 , shiftR3)
	devep4[0] = 0.
	devep4[-1] = derebc

	e_FRW3 = energy_FRW(t+dt-dt)

	Ap4,Gp4 = system_static_excision(shifte3,shiftM3,shiftR3,shiftU3,e_FRW3)
	k4U,k4R,k4M,k4e = system_dynamic_RK_excision(shiftU3,shiftR3,shiftM3,Ap4,Gp4,shifte3,devep4,devRp4,devUp4)

	U = Upl +(1./6.)*dt*(k1U+2.*k2U+2.*k3U+k4U)
	R = Rpl + (1./6.)*dt*(k1R+2.*k2R+2.*k3R+k4R)
	M = Mpl + (1./6.)*dt*(k1M+2.*k2M+2.*k3M+k4M)
	e = epl + (1./6.)*dt*(k1e+2.*k2e+2.*k3e+k4e)
	return U,R,M,e

#The derivatives at each point are computed using the new Chebyshev differentiation matrix perfomes for excision. 
def compute_derivatives_excision(err,Urr,Rrr):
	deve =  np.dot(Dexcision,err)
	devU =  np.dot(Dexcision,Urr)
	devR =  np.dot(Dexcision,Rrr)
	return deve,devU,devR


def system_static_excision(epc,Mpc,Rpc,Upc,e_FRWc):
	Aqq = 1.*(e_FRWc/epc)**(w/(w+1.))
	Gqq = np.sqrt(1+Upc**2-2.*Mpc/Rpc)
	return Aqq,Gqq



#We define the full 2M/R, to locate the position of the AH
def compact_full(Mv,Rv,efrw):
	Cc = 2*(Mv[:-1])/Rv[:-1]
	Cc = np.insert(Cc, -1, 0.)
	return Cc


#We search the position of the horizon, provident the term 2M/R, in the interval from "ac" to "bc"
def search_horizon(C_function,xdom,ac,bc):
	#la funcion estara dada en un determinado dominio
	C_function_regul = C_function-1
	C_function_interpolated = interpolate.interp1d(xdom, C_function_regul ,kind='cubic')
	r_horizon = optimize.ridder(C_function_interpolated, ac, bc, args=(), xtol=1.e-15, rtol=1.e-15 , maxiter=100, full_output=False, disp=True)
	return r_horizon



#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------



#Here is set the part of computation of the mass. The first part is devoted to evolve until an apparent thorizon is formed. The second one (the largest) is devoted to iterate the excision procedure
def search(thresh):
	global subs_boundary
	global ddrr
	global derebc #value of the derivative to freeze
	global N_cheb
	global dt0

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
	while t<t_final: #we set a maximum time

		dt = dt0*np.sqrt(t) #time-step
		t += dt

		Uk,Rk,Mk,ek,Ak,Gk,e_FRW = computation(dt,t,Rv,Mv,ev,Uv,Av,Gv)

		CC = compact_function(Mk,Rk,e_FRW) #we construct the compaction function
		Cmax = max(CC) #we search its maximum		

		
		Rv = Rk[:]
		Mv = Mk[:]
		ev = ek[:]
		Uv = Uk[:]
		Av = Ak[:]
		Gv = Gk[:]

		if Cmax>1.01: #if this is satisfied, an AH will be formed
			pos_max = np.argmax(CC)
			CC0 = compact_full(Mk,Rk,e_FRW)
			new_horizon_position = search_horizon(CC0,x,x[pos_max],x[pos_max]+rmww/2)
			intervM_interp = interpolate.interp1d(x, Mv ,kind='cubic') #interpolates using a cubic spline 
			Mhorizon.append(intervM_interp(new_horizon_position))
			rhorizon.append(new_horizon_position)
			store_time.append(t)
			
		#above a given cut-off for the peak of C, we cut and start the excision method
		if Cmax>Cmaxlimit:
			pos_max = np.argmax(CC)
			rmaxs2 = x[pos_max]
			compactfull = compact_full(Mv,Rv,e_FRW)
			r_horizon = search_horizon(compactfull,x,rmaxs2,rmaxs2+rmww/2) #we search the horizon in a domain
			r_i_excision = r_horizon-subs_boundary
			break
			#It stops and start to run the procedure of the bisection

	global x_excision
	global Dexcision
	Dexcision , x_excision = chebymatrix(N_cheb,r_i_excision,b) #we define the new computational domain excised
	

	Uvinterp = interpolate.interp1d(x, Uv, kind="cubic", fill_value="extrapolate")
	Mvinterp = interpolate.interp1d(x, Mv, kind="cubic", fill_value="extrapolate")
	evinterp = interpolate.interp1d(x, ev, kind="cubic", fill_value="extrapolate")
	Rvinterp = interpolate.interp1d(x, Rv, kind="cubic", fill_value="extrapolate")
	Avinterp = interpolate.interp1d(x, Av, kind="cubic", fill_value="extrapolate")
	Gvinterp = interpolate.interp1d(x, Gv, kind="cubic", fill_value="extrapolate")
	
	
	U_new = Uvinterp(x_excision)
	M_new = Mvinterp(x_excision)
	e_new = evinterp(x_excision)
	R_new = Rvinterp(x_excision)
	A_new = Avinterp(x_excision)
	G_new = Gvinterp(x_excision)

	Rvex = R_new[:]
	Mvex = M_new[:]
	evex = e_new[:]
	Uvex = U_new[:]
	Avex = A_new[:]
	Gvex = G_new[:]
	

	derebc = np.dot(Dexcision,evex)[-1]


	CC = compact_full(Mvex,Rvex,e_FRW)
	past_position_horizon = r_horizon
		
	
	dr_horizonte = 0.
	mas_initial_bh = Mvinterp(r_horizon)

	while t<t_max:

		dt = dt0*np.sqrt(t)
		t += dt

		Ukex,Rkex,Mkex,ekex,Akex,Gkex,e_FRW = computation_excision(dt,t,Rvex,Mvex,evex,Uvex,Avex,Gvex)

		if (np.any(np.isnan(Akex)) == True) or (np.any(np.isnan(Ukex)) == True) or (np.any(np.isnan(Mkex)) == True) or (np.any(np.isnan(ekex)) == True) or (np.any(np.isnan(Gkex)) == True) or (np.any(np.isnan(Rkex)) == True):
			print ("The maximum resolution of the excision procedure has been achieved. Simulation stopped")	
			break


		Rvex = Rkex[:]
		Mvex = Mkex[:]
		evex = ekex[:]
		Uvex = Ukex[:]
		Avex = Akex[:]
		Gvex = Gkex[:]


		intervM_interp = interpolate.interp1d(x_excision, Mvex ,kind='cubic')


		CC = compact_full(Mvex,Rvex,e_FRW)

		new_horizon_position = search_horizon(CC,x_excision,x_excision[-1],x_excision[-1]+rmww/2)
		masa_final_bh = intervM_interp(new_horizon_position)


		Mhorizon.append(masa_final_bh)
		rhorizon.append(new_horizon_position)
		store_time.append(t)

		mas_initial_bh = masa_final_bh
		
		dr = new_horizon_position-past_position_horizon

	
		if dr>ddrr:

			past_position_horizon = new_horizon_position

			#interpolation of the variables
			Uvinterp = interpolate.interp1d( x_excision, Uvex, kind="cubic", fill_value="extrapolate")
			Mvinterp = interpolate.interp1d( x_excision, Mvex, kind="cubic", fill_value="extrapolate")
			evinterp = interpolate.interp1d( x_excision, evex, kind="cubic", fill_value="extrapolate")
			Rvinterp = interpolate.interp1d( x_excision, Rvex, kind="cubic", fill_value="extrapolate")
			Avinterp = interpolate.interp1d( x_excision, Avex, kind="cubic", fill_value="extrapolate")
			Gvinterp = interpolate.interp1d( x_excision, Gvex, kind="cubic", fill_value="extrapolate")
	

			#we set the new position of the excision
			new_x_boundary = new_horizon_position-subs_boundary

			Dexcision , x_excision = chebymatrix(N_cheb,new_x_boundary,b) #we define the new excised domain

			#interpolate to the new domain
			U_new = Uvinterp(x_excision)
			M_new = Mvinterp(x_excision)
			e_new = evinterp(x_excision)
			R_new = Rvinterp(x_excision)
			A_new = Avinterp(x_excision)
			G_new = Gvinterp(x_excision)

			Rvex = R_new[:]
			Mvex = M_new[:]
			evex = e_new[:]
			Uvex = U_new[:]
			Avex = A_new[:]
			Gvex = G_new[:]
			derebc = np.dot(Dexcision,evex)[-1]
			dr = 0.
		

t_max = 10**5
Mhorizon = []
rhorizon = []
store_time = []

Cmaxlimit =1.2


#parameters of the excision. For large perturbations this two parameteres can be leave as constants without modification. 
subs_boundary = 0.06
ddrr = 0.03


epsrr,tH,rmww = epsilon_horizon_crosing(rm_N) #we get the epsilon parametter

search(2./3.) #this starts the simulation


#We generate the data files
filename1 = "results.dat"
f1 = open(filename1, "w")
#we write the daa files: 1-colum (time) , 2-column (PBHmass) , 3-column (horizon position)
for i in range(len(store_time)):
	f1.write(  str(store_time[i])+" "+str(Mhorizon[i])+" "+str(rhorizon[i])  )
	f1.write("\n")
f1.close()


print ("Simulation done successfully. The time of the computation was:")
print("--- %s seconds ---"% (time.time()-start_time))


