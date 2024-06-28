# mymatter.py
# Calculates the matter rhs and stress energy contributions
# this assumes spherical symmetry

import numpy as np
# from source.tensoralgebra import *

# params for matter
scalar_mu = 1.0 # this is an inverse length scale related to the scalar compton wavelength
#####TODO # scalar field potential V(phi) is hard-coded in equations

def get_matter_rhs(r, phi, psy, Pi, dpsydr, a, b, dadr, dbdr, em4chi,
                   dchidr, K, lapse, dlapsedr, dr_lapsePi) :
    
    dphidt = lapse * Pi
    dpsydt = dr_lapsePi
    dPidt  = lapse * em4chi /a * (dpsydr +  psy * (2/r - 0.5*dadr/a + dbdr/b + 2*dchidr)) + \
             psy * em4chi/a * dlapsedr + lapse * K * Pi
    

    # Add potential
    dVdphi = scalar_mu * scalar_mu * phi
    dPidt += - lapse * dVdphi
    
    return dphidt, dpsydt, dPidt

def get_rho(phi, psy, Pi, a, em4chi) :

    # The potential V(phi) = 1/2 mu^2 phi^2
    V_phi = 0.5 * scalar_mu * scalar_mu * phi * phi

    rho = 0.5 * Pi*Pi + 0.5 * em4chi/a  * psy * psy + V_phi

    return rho

def get_Sr_U(phi, psy, Pi) :       
    Sr = - Pi * psy
    return Sr


def get_Sa_Sb(phi, psy, Pi, a, em4chi):
    V_phi = 0.5 * scalar_mu * scalar_mu * phi * phi

    Sa = 0.5 * (Pi * Pi + psy * psy * em4chi /a) - V_phi
    Sb = 0.5 * (Pi * Pi - psy * psy * em4chi /a) - V_phi


    # S = 1.5 * Pi*Pi  -psy*psy *emp4chi/a - 3*V_u

    return Sa, Sb


