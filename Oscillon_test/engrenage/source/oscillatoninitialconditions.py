# oscillatoninitialconditions.py

# set the initial conditions for all the variables for an oscillaton
# see further details in https://github.com/GRChombo/engrenage/wiki/Running-the-oscillaton-example

from source.uservariables import *
from source.tensoralgebra import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.gridfunctions import *
import numpy as np
from scipy.interpolate import interp1d

projdir = '/Users/cjoana/Desktop/Oscillon_test'

def get_initial_state(R, N_r, r_is_logarithmic) :
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R, N_r, r_is_logarithmic)
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    initial_state = np.zeros(NUM_VARS * N)
    
    # Use oscilloton data to construct functions for the vars
    grr0_data    = np.loadtxt(projdir + "/initial_data/grr0.csv")
    lapse0_data  = np.loadtxt(projdir + "/initial_data/lapse0.csv")
    v0_data      = np.loadtxt(projdir + "/initial_data/v0.csv")
    
    # set up grid in radial direction in areal polar coordinates
    dR = 0.01;
    length = np.size(grr0_data)
    R = np.linspace(0, dR*(length-1), num=length)
    f_grr   = interp1d(R, grr0_data)
    f_lapse = interp1d(R, lapse0_data)
    f_v     = interp1d(R, v0_data)
    
    # fill all positive values of r
    for ix in range(num_ghosts, N) :

        # position on the grid
        r_i = r[ix]

        # scalar field values
        initial_state[ix + idx_u * N] = 0.0 # start at a moment where field is zero
        initial_state[ix + idx_v * N] = f_v(r_i)
 
        # non zero metric variables (note h_rr etc are rescaled difference from flat space so zero
        # and conformal factor is zero for flat space)
        initial_state[ix + idx_lapse * N] = f_lapse(r_i)
        # note that we choose that the determinant \bar{gamma} = \hat{gamma} initially
        grr_here = f_grr(r_i)
        gtt_over_r2 = 1.0
        # The following is required for spherical symmetry
        gpp_over_r2sintheta = gtt_over_r2
        phys_gamma_over_r4sin2theta = grr_here * gtt_over_r2 * gpp_over_r2sintheta
        # Note sign error in Baumgarte eqn (2) 
        phi_here = 1.0/12.0 * np.log(phys_gamma_over_r4sin2theta)
        initial_state[ix + idx_phi * N]   = phi_here
        em4phi = np.exp(-4.0*phi_here)
        initial_state[ix + idx_hrr * N]   = em4phi * grr_here - 1
        initial_state[ix + idx_htt * N]   = em4phi * gtt_over_r2 - 1.0
        initial_state[ix + idx_hpp * N]   = em4phi * gpp_over_r2sintheta - 1.0

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(initial_state, dx, N, r_is_logarithmic)
                
    # needed for lambdar
    hrr    = initial_state[idx_hrr * N : (idx_hrr + 1) * N]
    htt    = initial_state[idx_htt * N : (idx_htt + 1) * N]
    hpp    = initial_state[idx_hpp * N : (idx_hpp + 1) * N]
    
    if(r_is_logarithmic) :
        dhrrdx = get_logdfdx(hrr, oneoverlogdr)
        dhttdx = get_logdfdx(htt, oneoverlogdr)
        dhppdx = get_logdfdx(hpp, oneoverlogdr)
    else :
        dhrrdx     = get_dfdx(hrr, oneoverdx)
        dhttdx     = get_dfdx(htt, oneoverdx)
        dhppdx     = get_dfdx(hpp, oneoverdx)
        
    # assign lambdar values
    for ix in range(num_ghosts, N-num_ghosts) :

        # position on the grid
        r_here = r[ix]
        
        # Assign BSSN vars to local tensors
        h = np.zeros_like(rank_2_spatial_tensor)
        h[i_r][i_r] = hrr[ix]
        h[i_t][i_t] = htt[ix]
        h[i_p][i_p] = hpp[ix]
        
        dhdr = np.zeros_like(rank_2_spatial_tensor)
        dhdr[i_r][i_r] = dhrrdx[ix]
        dhdr[i_t][i_t] = dhttdx[ix]
        dhdr[i_p][i_p] = dhppdx[ix]
        
        # (unscaled) \bar\gamma_ij and \bar\gamma^ij
        bar_gamma_LL = get_metric(r_here, h)
        bar_gamma_UU = get_inverse_metric(r_here, h)
        
        # The connections Delta^i, Delta^i_jk and Delta_ijk
        Delta_U, Delta_ULL, Delta_LLL  = get_connection(r_here, bar_gamma_UU, bar_gamma_LL, h, dhdr)
        initial_state[ix + idx_lambdar * N]   = Delta_U[i_r]

    # Fill boundary cells for lambdar
    fill_outer_boundary_ivar(initial_state, dx, N, r_is_logarithmic, idx_lambdar)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary_ivar(initial_state, dx, N, r_is_logarithmic, idx_lambdar)
            
    return r, initial_state
