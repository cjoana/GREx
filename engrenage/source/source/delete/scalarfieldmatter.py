# mymatter.py
# Calculates the matter rhs and stress energy contributions
# this assumes spherical symmetry

import numpy as np
from source.tensoralgebra import *

# params for matter
scalar_mu = 1.0 # this is an inverse length scale related to the scalar compton wavelength

def get_matter_rhs(D, E, S, V, P,
                   dDdx, dEdx, dSdx, dVdx, dPdx, 
                    K_rr, K, em4phi, dphidx 
                    lapse, dlapsedx):
    # Assuming that S = S_i  with down indices 
    
    
    chi = em4phi   #check
    dchidx =  -4*dphidx* emp4phi  
    S_dot_chi = S * dchidx * hrr_UU; 
    
    # Covariant derivatives D_X       ####  can be simplified if using D_r * S_r,   currently D_i * S_i  
    covS = dSdx - Chris_ULL_rrr * S +  
            (0.5/chi)*(S*S*dchidx*dchidx - h_rr * S_dot_chi )    
    covV = dVdx - 3/2/chi * dchidx*V                                         # check
    
    
    """
    data_t covdV = 0;    // D_m V^m
    FOR1(m) { covdV += - 3/(2*vars.chi)* d1.chi[m]*vars.V[m]  +  d1.V[m][m]; }


    Tensor<2, data_t> covdZ;     // D_k Z_l
    FOR2(k, l)
    {
        covdtildeZ[k][l] = d1.Z[k][l];
        FOR1(m) { covdtildeZ[k][l] -= chris.ULL[m][k][l] * vars.Z[m]; }
        covdZ[k][l] =
            covdtildeZ[k][l] +
            (0.5/vars.chi) * (vars.Z[k] * d1.chi[l] + d1.chi[k] * vars.Z[l] -
                   vars.h[k][l] * Z_dot_dchi);
    }


    """ 
                            
    dDdt = lapse * K * D - (dlapsedx*D*V + dDdx*lapse*V + covV*lapse*D)  
    
    dEdt = lapse*K*(E+P) +  (D+E+P)*(lapse*V*V*K_rr - V*dlapsedx)           # Assuming no spin:   V_theta = V_phi = 0
    
    Bracket = (S*V* + P)
    dBracket = covS * V + S *covV + dPdx 
    dSdt = lapse*K*S - (E+D)*dlapsedx 
        - dlapsedx * Bracket - lapse * dBracket

    
    return dDdt, dEdt, dSdt

def get_rho(D, U) :
    # rho_ADM 
    rho = D+E
    return rho

def get_Si(S) :
    S_i = S
    return S_i

# Get rescaled Sij value (rSij = diag[Srr, S_tt / r^2, S_pp / r^2 sin2theta ])
def get_rescaled_Sij(u, dudr, v, r_gamma_UU, em4phi, r_gamma_LL) :
    rS_ij = np.zeros_like(rank_2_spatial_tensor)

    # The potential V(u) = 1/2 mu^2 u^2
    V_u = 0.5 * scalar_mu * scalar_mu * u * u
    
    # Useful quantity Vt
    Vt = - v*v + em4phi * r_gamma_UU[i_r][i_r] * (dudr * dudr)
    for i in range(0, SPACEDIM):    
        rS_ij[i][i] = - (0.5 * Vt  + V_u) * r_gamma_LL[i][i] / em4phi + delta[i][i_r] * dudr * dudr
    
    # The trace of S_ij
    S = 0.0
    for i in range(0, SPACEDIM): 
        for j in range(0, SPACEDIM):
            S += rS_ij[i][j] * r_gamma_UU[i][j] * em4phi
    return S, rS_ij
