# mymatter.py
# Calculates the matter rhs and stress energy contributions using
# prescription in Alcubierre's book.
# this assumes perfect fluid in spherical symmetry.

# Indices: 
# V is upper : V^r 
# S is lower : S_r
# Other tensors varnames are like hRR = h^{rr}, hrr = h_{rr} , CRuv = C^r_{uv}

import numpy as np
from source.tensoralgebra import *


def get_matter_rhs(r, D, E, S, V, P,
                   dDdr, dEdr, dSdr, dVdr, dPdr, 
                   a, b, dadr, dbdr, dchidr,
                   em4chi, K, Aa, lapse, dlapsedr):
    # Assuming that S = S_i  with down indices 
    # Assuming no spin:   V_theta = V_phi = 0

    # Covariant derivatives :  Dk_(S_r)  and Dk_(V^k)  where k goes from r, theta, phi. 
    covS = dSdr - S*(2*dchidr + 0.5*dadr/a)                   # contains Chris_Rrr.   
    #                                                         #  (terms with Chris_Trt, Chris_Prp are zero due to Si=Sr) 
    covV = dVdr + V * (6*dchidr + 0.5*dadr/a + dbdr/b + 2/r)  # contains Chris_Rrr, Chris_Ttr, Chris_Ppr
   
    K_rr = a/em4chi *(Aa + K/3)

    #  rhs equation like in Alcubierre's book
    dDdt = lapse * K * D - (dlapsedr*D*V + dDdr*lapse*V + covV*lapse*D)  
    
    dEdt = lapse*K*(E+P) +  (D+E+P)*(lapse*V*V*K_rr - V*dlapsedr) + \
              - (dlapsedr*V*(E+P) + lapse*covV*(E+P) + lapse*V*(dEdr+dPdr) )
    
    Bracket = (S*V + P)
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

def get_Jr_U(S, a, em4chi):
    hRR = em4chi/a
    Jr = hRR * S
    return Jr

def get_diag_Sij(r, a, b, D, E, V, P, em4chi):
    # getting S_ij with lower indices 

    rhohW2 = E + D + P
    hrr = a/em4chi
    htt = r*r*b/em4chi
    # hpp = r*r*b # assuming sin\theta = 1
    V_r = hrr * V

    Srr = rhohW2 *V_r*V_r + P*hrr
    Stt = P*htt
    Spp = Stt

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


# def get_lorentz(V):
#     W = (1 - V*V)**0.5
#     return W 

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
    # if np.sum(in_sqrt < 0) > 0 : 
    #     print('Warning in_sqrt < 0')
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

    # if np.sum(in_sqrt < 0) > 0 : 
    #     print(f'Warning in_sqrt < 0,   = {in_sqrt}')
    in_sqrt[in_sqrt < 0] = 0                                    
    
    fl_dens = ((omega -1)*(E+D) + (in_sqrt)**0.5 )/(2*omega)
    pressure = fl_dens*omega
    
    floor = 1e-16
    
    Lorentz= (fl_dens + pressure)/(E+D+pressure + floor)
    W = 1/Lorentz

    rhohW2 = D + E + pressure
    V_R = S * hRR / (rhohW2 + floor)


    return fl_dens, pressure, W,  V_R



######

def get_R(r, scalefactor, zeta, b=1):  ## Areal Radius
    # mean = np.nanmean(chi[N//4:N*3//4])
    # print(f'scalefactor is {scalefactor} and exp(2*meanZ) is {np.exp(2*mean)}')
    # return  scalefactor * r * np.exp(2*chi-2*mean)
    return  r * np.exp(zeta) * scalefactor # * np.sqrt(b)


def get_M(r, rho, R, dRdr):
    
    # # Use cumsum
    dr = np.diff(r)
    integrant =  R*R*rho*dRdr
    Mass = np.cumsum(integrant) * 4*np.pi *dr[0]

    # # Use integrate
    # from scipy import integrate
    # from scipy import interpolate
    
    # dr = np.diff(r)[0]
    # integrant_r =  R*R*rho*dRdr* 4*np.pi 
    # x = r
    # y = integrant_r
    # # integrant_R = rho*R*R* 4*np.pi       # problematic for PBH_2 (non-monotonic R)
    # # x = R
    # # y = integrant_R
    # f = interpolate.interp1d(x, y, kind='quadratic')
    # Mass = np.array([integrate.quad(f, 0., np.abs(x_i), limit=10000)[0] for x_i in x])

    return Mass 


def compact_function(r, M, R, dRdr, rho_bkg):

    # # Use cumsum
    dr = np.diff(r)
    integrant =  R*R*rho_bkg*dRdr
    Mbkg = np.cumsum(integrant) * 4*np.pi *dr[0]

    C =  2*(M-Mbkg)/R

    # C =  2*M/R  - (8./3.)*np.pi*rho_bkg * R**2   
    return C  #  *2/3

def get_CompactionSS(chi, dRdr, b=1, scalefactor=1, omega=1./3):
    fomega = 3*(omega+1)/(5+3*omega)   * np.ones_like(chi)   # sometimes fomega is set to 0.5 (arxiv:2401.06329)
    # fomega = 0.5
    CSS = fomega*(1 - (np.exp(-2*chi) * dRdr)**2 )    # divde by b? :  dRdr)**2/b

    return CSS

def get_Compaction(r, dRdr,  omega=1./3):

    Cl = -6*(omega+1)/(5+3*omega) * r * dRdr
    CSS = Cl * (1 - (5+3*omega)/(12*(1+omega))*Cl )

    return CSS

def get_CompactionSS_altern(r, dchidr, omega=1/3):
    CSS = 0.5*(1-( 1 + dchidr*r )**2)
    return CSS


def get_int_Compaction(r, rho, R, dRdr, drho):
    
    # # Use cumsum
    # dr = np.zeros_like(r)
    # dr[:-1] = np.diff(r)
    # integrant =  R*R*rho*dRdr
    # Mass = np.cumsum(integrant) * 4*np.pi *dr[0]

    # # Use integrate
    # from scipy import integrate
    # from scipy import interpolate
    
    dr = np.diff(r)[0]

    # integrant_r =  4*np.pi *  R*R*dRdr*drho
    # x = r
    # y = integrant_r
    # # integrant_R = rho*R*R* 4*np.pi       # problematic for PBH_2 (non-monotonic R)
    # # x = R
    # # y = integrant_R
    # f = interpolate.interp1d(x, y, kind='linear')
    # dMass = np.array([integrate.quad(f, 0., np.abs(x_i), limit=10000)[0] for x_i in x])

    dMass = np.sum(4*np.pi *  R*R* dRdr* drho *dr)

    return dMass/R



def get_beta_comoving(r, K, lapse):
    beta =   K/3 * r*lapse
    # beta[beta**2>0.8] = -0.8
    return beta  
