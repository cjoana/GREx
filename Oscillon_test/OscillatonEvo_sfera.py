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

# import homemade code
sys.path.append('./sfera/')
sys.path.append('../')
#from source._parallel_rhsevolution import *                   # go here to look at how the evolution works
from source.rhsevolution import *                   # go here to look at how the evolution works
from source.oscillatoninitialconditions import *              # go here to change the initial conditions
# from source.hamdiagnostic import *  

# Input parameters for grid and evolution here
N_r = 400 # num points on physical grid
R = 200 # Maximum outer radius

r_is_logarithmic = False
r, initial_state = get_initial_state(R, N_r, r_is_logarithmic)

#unpackage the vector for readability
(initial_chi, initial_a, initial_b, initial_K,  initial_Aa, initial_AX, initial_X, 
 initial_Lambda, initial_lapse, initial_beta, initial_br, initial_phi,
 initial_psy, initial_Pi) = unpack_state(initial_state, N_r)

#plot initial conditions
plt.xlabel('r')
plt.plot(r, initial_phi, '-o', label='phi') # zero, but plot as dots to see the grid
plt.plot(r, initial_Pi, label='Pi')
plt.plot(r, initial_chi, label='chi')
plt.plot(r, initial_a, label='a')
plt.plot(r, initial_b, label='b')
plt.plot(r, initial_Lambda, label=r'$Lambda^r$')
#plt.plot(r, initial_beta, label='shift')
plt.plot(r, initial_lapse-1, label='lapse - 1')
plt.legend(loc='best')
plt.grid()
#plt.xlim(-0.25,5.0)
#plt.ylim(-0.0005,0.0005)

# plt.show()

####TODO: code and plot hamdiagnostics
# # check the Hamiltonian constraint initially satisfied
# # apart from numerical errors
# r, Ham = get_Ham_diagnostic(initial_state, np.array([0]), R, N_r, r_is_logarithmic)

# # plot the profile for Ham
# plt.plot(r, Ham[0])

# plt.xlabel('r')
# #plt.xlim(-4.0,R+4.0)
# #plt.ylim(-0.01,0.01)
# plt.ylabel('Ham value')
# plt.grid()







import time

start = time.time()
# for control of time integrator and spatial grid
T = 20.0 # Maximum evolution time
N_t = 100 # time resolution (only for outputs, not for integration)

# Work out dt and time spacing of outputs
dt = T/N_t
t = np.linspace(0, T-dt, N_t)
eta = 2.0 # the 1+log slicing damping coefficient - of order 1/M_adm of spacetime


params = Munch(dict())
params.r_max = R
params.N_r = N_r
params.r_is_logarithmic = r_is_logarithmic
# params.t_ini = 0

sigma = 0.1*R/N_r 


# Solve for the solution using RK45 integration of the ODE
# to make like (older) python odeint method use method='LSODA' instead
# use tqdm package to track progress
with tqdm(total=100, unit="‰") as progress_bar:
    dense_solution = solve_ivp(get_rhs, [0,T], initial_state, 
                               args=(params, sigma, progress_bar, [0, T/100]),
                        #atol=1e-5, rtol=1e-5,
                        max_step=(0.1*R/N_r), #for stability and for KO coeff of 10
                        method='RK45', dense_output=True)

# Interpolate the solution at the time points defined in myparams.py
solution = dense_solution.sol(t).T








end = time.time() 

# Save output as binary file
np.save("osc_solution_sfera.npy", solution)

print(f"Time needed for evolution {end-start} seconds.  ")

