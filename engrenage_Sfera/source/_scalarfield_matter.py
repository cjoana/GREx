# mymatter.py
# Calculates the matter rhs and stress energy contributions
# this assumes spherical symmetry

import numpy as np
from source.tensoralgebra import *

# params for matter
scalar_mu = 1.0 # this is an inverse length scale related to the scalar compton wavelength

def get_matter_rhs(phi, Pi, dphidr, d2udr2, r_gamma_UU, em4chi,     #### r_gamma_UU -> a?
                   dchidr, K, lapse, dlapsedr, r_conformal_chris) :
    
    dphidt =  lapse * Pi
    dPidt =  lapse * K * Pi + r_gamma_UU[i_r][i_r] * em4chi * (2.0 * lapse * dchidr * dphidr 
                                                               + lapse * d2udr2
                                                               + dlapsedr * dphidr)
    for i in range(0, SPACEDIM): 
        for j in range(0, SPACEDIM):
            dPidt +=  - em4chi * lapse * r_gamma_UU[i][j] * r_conformal_chris[i_r][i][j] * dphidr
    
    # Add mass term
    dVdu = scalar_mu * scalar_mu * phi
    dPidt += - lapse * dVdu
    
    return dphidt, dPidt

def get_rho(phi, dphidr, Pi, r_gamma_UU, em4chi) :

    # The potential V(phi) = 1/2 mu^2 phi^2
    V_u = 0.5 * scalar_mu * scalar_mu * phi * phi
    rho = 0.5 * Pi*Pi + 0.5 * em4chi * r_gamma_UU[i_r][i_r] * dphidr * dphidr + V_u

    return rho

def get_Si(phi, dphidr, Pi) :
    S_i = np.zeros_like(rank_1_spatial_tensor)
    
    S_i[i_r] = - Pi * dphidr
    
    return S_i

# Get rescaled Sij value (rSij = diag[Srr, S_tt / r^2, S_pp / r^2 sin2theta ])
def get_rescaled_Sij(phi, dphidr, Pi, r_gamma_UU, em4chi, r_gamma_LL) :
    rS_ij = np.zeros_like(rank_2_spatial_tensor)

    # The potential V(phi) = 1/2 mu^2 phi^2
    V_u = 0.5 * scalar_mu * scalar_mu * phi * phi
    
    # Useful quantity Vt
    Vt = - Pi*Pi + em4chi * r_gamma_UU[i_r][i_r] * (dphidr * dphidr)
    for i in range(0, SPACEDIM):    
        rS_ij[i][i] = - (0.5 * Vt  + V_u) * r_gamma_LL[i][i] / em4chi + delta[i][i_r] * dphidr * dphidr
    
    # The trace of S_ij
    S = 0.0
    for i in range(0, SPACEDIM): 
        for j in range(0, SPACEDIM):
            S += rS_ij[i][j] * r_gamma_UU[i][j] * em4chi
    return S, rS_ij
