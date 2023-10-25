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
sys.path.append('./engrenage/')
sys.path.append('../')
#from source._parallel_rhsevolution import *                   # go here to look at how the evolution works
from source.rhsevolution import *                   # go here to look at how the evolution works
from source.oscillatoninitialconditions import *              # go here to change the initial conditions
from source.hamdiagnostic import *  

# Input parameters for grid and evolution here
N_r = 400 # num points on physical grid
R = 200.0 # Maximum outer radius

r_is_logarithmic = False
r, initial_state = get_initial_state(R, N_r, r_is_logarithmic)

#unpackage the vector for readability
(initial_u, initial_v , initial_phi, initial_hrr, initial_htt, initial_hpp, 
 initial_K, initial_arr, initial_att, initial_app, 
 initial_lambdar, initial_shiftr, initial_br, initial_lapse) = unpack_state(initial_state, N_r)

#plot initial conditions
plt.xlabel('r')
plt.plot(r, initial_u, '-o', label='u') # zero, but plot as dots to see the grid
plt.plot(r, initial_v, label='v')
plt.plot(r, initial_phi, label='phi')
plt.plot(r, initial_hrr, label='hrr')
plt.plot(r, initial_htt, label='htt')
plt.plot(r, initial_lambdar, label='lambdar')
#plt.plot(r, initial_shiftr, label='shiftr')
plt.plot(r, initial_lapse-1, label='lapse - 1')
plt.legend(loc='best')
plt.grid()
#plt.xlim(-0.25,5.0)
#plt.ylim(-0.0005,0.0005)


# check the Hamiltonian constraint initially satisfied
# apart from numerical errors
r, Ham = get_Ham_diagnostic(initial_state, np.array([0]), R, N_r, r_is_logarithmic)

# plot the profile for Ham
plt.plot(r, Ham[0])

plt.xlabel('r')
#plt.xlim(-4.0,R+4.0)
#plt.ylim(-0.01,0.01)
plt.ylabel('Ham value')
plt.grid()

















import time

start = time.time()
# for control of time integrator and spatial grid
T = 20.0 # Maximum evolution time
N_t = 100 # time resolution (only for outputs, not for integration)

# Work out dt and time spacing of outputs
dt = T/N_t
t = np.linspace(0, T-dt, N_t)
eta = 2.0 # the 1+log slicing damping coefficient - of order 1/M_adm of spacetime

# Solve for the solution using RK45 integration of the ODE
# to make like (older) python odeint method use method='LSODA' instead
# use tqdm package to track progress
with tqdm(total=100, unit="‰") as progress_bar:
    dense_solution = solve_ivp(get_rhs, [0,T], initial_state, 
                               args=(R, N_r, r_is_logarithmic, eta, progress_bar, [0, T/100]),
                        #atol=1e-5, rtol=1e-5,
                        max_step=(0.1*R/N_r), #for stability and for KO coeff of 10
                        method='RK45', dense_output=True)

# Interpolate the solution at the time points defined in myparams.py
solution = dense_solution.sol(t).T

end = time.time() 

# Save output as binary file
np.save("osc_solution_engrenage.npy", solution)

print(f"Time needed for evolution {end-start} seconds.  ")

