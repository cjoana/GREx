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
import time


# import homemade code
sys.path.append('./engrenage/')
sys.path.append('../')
from source._par_rhsevolution import *                   # go here to look at how the evolution works
#from source.rhsevolution import *                   # go here to look at how the evolution works
from source.oscillatoninitialconditions import *              # go here to change the initial conditions
from source.hamdiagnostic import *  


DefSimParams = True 
# Input parameters for grid and evolution here
N_r = 500 # num points on physical grid
R = 50.0 # Maximum outer radius

dx = R/N_r

r_is_logarithmic = False
r, initial_state = get_initial_state(R, N_r, r_is_logarithmic)

#unpackage the vector for readability
(initial_u, initial_v , initial_phi, initial_hrr, initial_htt, initial_hpp, 
 initial_K, initial_arr, initial_att, initial_app, 
 initial_lambdar, initial_shiftr, initial_br, initial_lapse) = unpack_state(initial_state, N_r)



start = time.time()
# for control of time integrator and spatial grid

N_t = 500 # time resolution (only for outputs, not for integration)
dt = dx*0.05

T = N_t * dt # Maximum evolution time
t_res = T*dt

# Work out dt and time spacing of outputs
# dt = T/N_t
t = np.linspace(0, T-dt, N_t)
eta = 2.0 # the 1+log slicing damping coefficient - of order 1/M_adm of spacetime

# Solve for the solution using RK45 integration of the ODE
# to make like (older) python odeint method use method='LSODA' instead
# use tqdm package to track progress
with tqdm(total=N_t, unit="‰") as progress_bar:
    dense_solution = solve_ivp(get_rhs, [0,T], initial_state, 
                               args=(R, N_r, r_is_logarithmic, eta, progress_bar, [0, T/N_t]),
                        #atol=1e-5, rtol=1e-5,
                        max_step=dt, #for stability and for KO coeff of 10
                        method='RK45', dense_output=True)

# Interpolate the solution at the time points defined in myparams.py
solution = dense_solution.sol(t).T

end = time.time() 

print(f"Time needed for PARALLEL evolution {end-start} seconds ({(end-start)/60}).  ")



# Plot a single point versus time
var1 = idx_u
var2 = idx_v

num_out = 100

idx = num_ghosts+1
r_i = np.round(r[idx],2)
var1_of_t = solution[0:N_t, var1 * (N_r + 2*num_ghosts) + idx]
plt.plot(t[::N_t//num_out], var1_of_t[::N_t//num_out], 'b-', label=variable_names[var1])
var2_of_t = solution[0:N_t, var2 * (N_r + 2*num_ghosts) + idx]
plt.plot(t[::N_t//num_out], var2_of_t[::N_t//num_out], 'g-', label=variable_names[var2])
plt.legend(loc='best')
plt.xlabel('t')
plt.ylabel('value at r is '+str(r_i))
plt.legend(loc='best')
plt.grid()
plt.show()




# plot the profile for some variable at a selection of times
var = idx_u # I suggest looking at the field u, or the lapse to see the gauge evolution

plt.clf()

for i, t_i in enumerate(t) :
    if (i < N_t) and (i % num_out == 0) and (t_i > 0.0):
        labelt = "t="+str(round(t_i,2))
        f_t = solution[i, var * (N_r + 2*num_ghosts): (var + 1) * (N_r + 2*num_ghosts)]
        plt.plot(r, f_t, label=labelt)

plt.legend(loc=4)
plt.xlabel('r')
# plt.xlim(-0.2,35.0)
plt.ylabel('value over time of ' + variable_names[var])
plt.grid()
plt.show()


# calculate the diagnostics, just the Hamiltonian constraint for now
r, Ham = get_Ham_diagnostic(solution, t, R, N_r, r_is_logarithmic)

plt.clf()

for i, t_i in enumerate(t) :
    if (i < N_t) and (i % (N_t//num_out) == 0) :
        labelt = "t="+str(round(t_i,2))
        Ham_t = Ham[i]
        Ham_t = Ham_t[num_ghosts:(N_r + num_ghosts)] # don't plot ghosts for diagnostics
        r_diagnostics = r[num_ghosts:(N_r + num_ghosts)]
        plt.plot(r_diagnostics, np.abs(Ham_t), label=labelt)

plt.legend(loc=4)
plt.xlabel('r')
plt.yscale('log')
#plt.xlim(-1,R+2)
#plt.ylim(-1.0,1.0)
plt.ylabel('Ham value over time')
plt.grid()
plt.show()
