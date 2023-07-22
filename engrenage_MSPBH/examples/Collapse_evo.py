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


#from source._parallel_rhsevolution import *  # go here to look at how the evolution works
from source.rhsevolution import *             # go here to look at how the evolution works
from source.initialdata import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

# Input parameters for grid and evolution here
N_r = 100 # num points on physical grid
R_max = 150.0 # Maximum outer radius

r_is_logarithmic = False
r, initial_state = get_initial_state(R_max, N_r, r_is_logarithmic)

#unpackage the vector for readability
(initial_U, initial_R , initial_M, initial_rho) = unpack_state(initial_state, N_r)

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
plt.show()

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
T = 16.0 # Maximum evolution time
N_t = 11 # time resolution (only for outputs, not for integration)

# Work out dt and time spacing of outputs
dt = T/N_t
t = np.linspace(0, T-dt, N_t)
sigma = 10.0 # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R_max/N_r  

if(max_step > R/N_r/sigma): print("WARNING: kreiss oliger condition not satisfied!")


# Solve for the solution using RK45 integration of the ODE
# to make like (older) python odeint method use method='LSODA' instead
# use tqdm package to track progress
with tqdm(total=100, unit="‰") as progress_bar:
    dense_solution = solve_ivp(get_rhs, [0,T], initial_state, 
                               args=(R, N_r, r_is_logarithmic, sigma, progress_bar, [0, T/100]),
                        #atol=1e-5, rtol=1e-5,
                        max_step=(0.1*R_max/N_r), #for stability and for KO coeff of 10
                        method='RK45', dense_output=True)

# Interpolate the solution at the time points defined in myparams.py
solution = dense_solution.sol(t).T

end = time.time() 

print(f"Time needed for evolution {end-start} seconds.  ")

