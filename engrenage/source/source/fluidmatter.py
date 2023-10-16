# mymatter.py
# Calculates the matter rhs and stress energy contributions
# this assumes spherical symmetry

import numpy as np
from source.tensoralgebra import *


def get_matter_rhs(r, D, E, S, V, P,
                   dDdr, dEdr, dSdr, dVdr, dPdr, 
                   a, b, dadr, dbdr, dchidr,
                   em4chi, K, Aa, lapse, dlapsedr):
    # Assuming that S = S_i  with down indices 
    # Assuming no spin:   V_theta = V_phi = 0

    # Covariant derivatives :  Dk_(S_r)  and Dk_(V^k)  where k goes from r, theta, phi.
    covS = dSdr - S*(2*dchidr + 0.5*dadr/a)                   # contains Chris_Rrr.   (Chris_Trt, Chris_Prp are zero) 
    covV = dVdr - V * (6*dchidr + 0.5*dadr/a + dbdr/b + 2/r)  # contains Chris_Rrr, Chris_Trt, Chris_Prp
   
    K_rr = a/em4chi *(Aa + K/3)

    #  rhs equation like in Alcubierre's book
    dDdt = lapse * K * D - (dlapsedr*D*V + dDdr*lapse*V + covV*lapse*D)  
    
    dEdt = lapse*K*(E+P) +  (D+E+P)*(lapse*V*V*K_rr - V*dlapsedr) + \
              - (dlapsedr*V*(E+P) + lapse*covV*(E+P) + lapse*V*(dEdr+dPdr) )
    
    Bracket = (S*V* + P)
    dBracket = covS * V + S *covV + dPdr 
    dSdt = lapse*K*S - (E+D)*dlapsedr - dlapsedr * Bracket - lapse * dBracket
     
    return dDdt, dEdt, dSdt


def get_rho(D, E) :
    # rho_ADM 
    rho = D+E
    return rho

def get_Si(S) :
    # getting S_i  or Jr with lower indices
    S_i = S
    return S_i

def get_diag_Sij(r, a, b, D, E, V, P, em4chi):
    # getting S_ij with lower indices 

    rhohW2 = E + D + P
    hrr = a/em4chi
    htt = r*r*b/em4chi
    # hpp = r*r*b # assuming sin\theta = 1
    V_r = hrr * V

    Srr = rhohW2 *V_r*V_r + P*hrr
    Stt = P*htt
    Spp = Srr

    return Srr, Stt, Spp 


def get_Sa_Sb(r, a, b, D, E, V, P, em4chi):
    # getting S_ij with lower indices 

    rhohW2 = E + D + P
    hrr = a/em4chi
    htt = r*r*b/em4chi
    # hpp = r*r*b # assuming sin\theta = 1
    V_r = hrr * V

    Srr = rhohW2 *V_r*V_r + P*hrr
    Stt = P*htt

    Sa = Srr/hrr
    Sb = Stt/htt

    return Sa, Sb


def get_lorentz(V):
    W = (1 - V*V)**0.5
    return W 

def get_velocity(D, E, P, S, a, em4chi):
    rhohW2 = D + E + P
    hRR = em4chi/a
    V_R = S * hRR / rhohW2
    return V_R 

# P = get_pressure()


def get_rhofluid(D, E, S, a, em4chi, omega):
    hRR = em4chi/a
    S2 = S * S * hRR 
    # use algebraic solution from second order equation
    in_sqrt = (omega-1)*(omega-1)*(E + D)*(E + D) - 4*omega*S2 + 4*omega*(E + D)*(E + D)
    if np.sum(in_sqrt < 0) > 0 : 
        print('Warning in_sqrt < 0')
    in_sqrt[in_sqrt < 0] = 0                   
    fl_dens = ((omega -1)*(E+D) + (in_sqrt)**0.5 )/(2*omega)
    return fl_dens

def get_rhofluid_pressure_W_velocity(D, E, S, a, em4chi, omega):

    # Assumtion for solving second order equation alebraically 
    if omega <= 0 or omega>=1: raise  

    hRR = em4chi/a
    S2 = S * S * hRR 
    # use algebraic solution from second order equation
    in_sqrt = (omega-1)*(omega-1)*(E + D)*(E + D) - 4*omega*S2 + 4*omega*(E + D)*(E + D)

    if np.sum(in_sqrt < 0) > 0 : 
        print('Warning in_sqrt < 0')
    in_sqrt[in_sqrt < 0] = 0                                    
    
    fl_dens = ((omega -1)*(E+D) + (in_sqrt)**0.5 )/(2*omega)
    pressure = fl_dens*omega
    
    Lorentz= (fl_dens + pressure)/(E+D+pressure)
    W = 1/Lorentz

    rhohW2 = D + E + pressure
    hRR = em4chi/a
    V_R = S * hRR / rhohW2


    return fl_dens, pressure, W,  V_R


