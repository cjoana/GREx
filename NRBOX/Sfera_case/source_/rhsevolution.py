#rhsevolution.py

####################
#  This is the code related to the rhs evolution equations. It evolves BSSN with scalar fields in
#  spherical symmetry.  The code constructed using the structure of theengranage code as a reference,
#  but optimized for fast/spherical simulations. For the pedagogical engrenage code, please visit
#  https://github.com/GRChombo/engrenage develope by Katy Clough + others. 
# 
#  The codes below have been develoved by Cristian Joana. 
#  CJ deeply thanks K. Clough et. al. for inspiring these lines of code.
############################################


# python modules
import numpy as np
import time

# homemade code
from source.uservariables import *
from source.gridfunctions import *
from source.fourthorderderivatives import *
from source.logderivatives import *
# from source.tensoralgebra import *
from source.scalarfield_matter import *
from source.bssnsphsym import *
    
# function that returns the rhs for each of the field vars
# see further details in https://github.com/GRChombo/engrenage/wiki/Useful-code-background
def get_rhs(t_i, current_state, params, sigma, progress_bar, time_state) :     
 
    R_max = params.r_max
    N_r = params.N_r
    r_is_logarithmic = params.r_is_logarithmic
    t_ini = params.t_ini
    # rho_bkg_ini = params.rho_bkg_ini

    # some params  (hardcoded)
    sigma_frame = 1
    t_ini = 1. 

    # omega = 0.33333333333333333
    t_over_t_ini = t_i/t_ini
    # H_ini = 2./(3.*(1.+omega))/t_ini # alpha/t_ini
    # rho_bkg_ini =  3./(8.*np.pi) *H_ini**2
    


    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R_max, N_r, r_is_logarithmic)

    
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    # this is where the rhs will go
    rhs = np.zeros_like(current_state) 
    
    ####################################################################################################
    #unpackage the state vector for readability - these are the vectors of values across r values at time t_i
    # see uservariables.py for naming conventions
    
    # Unpack variables from current_state - see uservariables.py
    chi, a, b, K, AX, X, Lambda, lapse, beta, br, phi, psy, Pi = unpack_state(current_state, N_r) 

    em4chi = np.exp(-4*chi)
    Aa, Ab = get_Aa_Ab(r, AX)
    
    
     # create mask to filter out ghosts
    mask = np.zeros_like(D, dtype=bool)
    mask[num_ghosts:-num_ghosts] = 1
                
    # Matter sources - see mymatter.py                                                                       
    matter_rho      = get_rho(phi[mask], psy[mask], Pi[mask], a[mask], em4chi[mask]) 
    matter_Si       = get_Sr_U(phi[mask], psy[mask], Pi[mask])   #upper index
    matter_Jr       = matter_Si     #upper index 
    matter_Sa, matter_Sb   = get_Sa_Sb(psy[mask], Pi[mask], a[mask], em4chi[mask])

    # t0 = time.time()
    # print("grid and var setup done in ", t0-start_time)

    ####################################################################################################

    # get the various derivs that we need to evolve things
    if(r_is_logarithmic) : #take logarithmic derivatives

        ###### TODO
        raise()
      
    else: 

        # second derivatives
        # d2phidx2     = get_d2fdx2(phi, oneoverdxsquared)
        # d2Pidx2     = get_d2fdx2(Pi, oneoverdxsquared)
        d2chidr2    = get_d2fdx2(chi, oneoverdxsquared)
        d2adr2      = get_d2fdx2(a, oneoverdxsquared)
        d2bdr2      = get_d2fdx2(b, oneoverdxsquared)
        d2lapsedr2  = get_d2fdx2(lapse, oneoverdxsquared)
        d2betadr2   = get_d2fdx2(beta, oneoverdxsquared)
        d2Xdr2      = get_d2fdx2(X, oneoverdxsquared)
    
        # first derivatives
        dphidr       = get_dfdx(phi, oneoverdx)
        dpsydr       = get_dfdx(psy, oneoverdx)
        dPidr       = get_dfdx(Pi, oneoverdx)
        dchidr     = get_dfdx(chi, oneoverdx)
        dadr       = get_dfdx(a, oneoverdx)
        dbdr       = get_dfdx(b, oneoverdx)
        dKdr       = get_dfdx(K, oneoverdx)
        dAXdr      = get_dfdx(AX, oneoverdx)
        dAadr      = get_dfdx(Aa, oneoverdx)
        dXdr       = get_dfdx(X, oneoverdx)
        dLambdadr = get_dfdx(Lambda, oneoverdx)
        dlapsedr   = get_dfdx(lapse, oneoverdx)
        dbetadr   = get_dfdx(beta, oneoverdx)

        # Extra derivatives
        cov_beta = get_covbeta(r, a, b, dadr, dbdr, beta, dbetadr)
        dr_cov_beta = get_dfdx(cov_beta, oneoverdx)
        dr_beta_over_r = get_dfdx(beta/r, oneoverdx)
        dr_dlapsedr_over_r = get_dfdx(dlapsedr/r, oneoverdx)
        dr_dchidr_over_r = get_dfdx(dchidr/r, oneoverdx)
        dr_Lambda_over_r = get_dfdx(Lambda/r, oneoverdx)
        dr_lapsePi = get_dfdx(lapsePi, oneoverdx)
    
        # first derivatives - advec left and right
        dphidr_advec_L       = get_dfdx_advec_L(phi, oneoverdx)
        dpsydr_advec_L       = get_dfdx_advec_L(psy, oneoverdx)
        dPidr_advec_L       = get_dfdx_advec_L(Pi, oneoverdx)
        dchidr_advec_L     = get_dfdx_advec_L(chi, oneoverdx)
        dadr_advec_L       = get_dfdx_advec_L(a, oneoverdx)
        dbdr_advec_L       = get_dfdx_advec_L(b, oneoverdx)
        dKdr_advec_L       = get_dfdx_advec_L(K, oneoverdx)
        dAXdr_advec_L      = get_dfdx_advec_L(AX, oneoverdx)
        dXdr_advec_L       = get_dfdx_advec_L(X, oneoverdx)
        dLambdadr_advec_L = get_dfdx_advec_L(Lambda, oneoverdx)
        dbetadr_advec_L  = get_dfdx_advec_L(beta, oneoverdx)
        dlapsedr_advec_L   = get_dfdx_advec_L(lapse, oneoverdx)
    
        dphidr_advec_R      = get_dfdx_advec_R(phi, oneoverdx)
        dpsydr_advec_R      = get_dfdx_advec_R(psy, oneoverdx)
        dPidr_advec_R       = get_dfdx_advec_R(Pi, oneoverdx)

        dchidr_advec_R     = get_dfdx_advec_R(chi, oneoverdx)
        dadr_advec_R       = get_dfdx_advec_R(a, oneoverdx)
        dbdr_advec_R       = get_dfdx_advec_R(b, oneoverdx)
        dKdr_advec_R       = get_dfdx_advec_R(K, oneoverdx)
        dAXdr_advec_R      = get_dfdx_advec_R(AX, oneoverdx)
        dXdr_advec_R       = get_dfdx_advec_R(X, oneoverdx)
        dLambdadr_advec_R  = get_dfdx_advec_R(Lambda, oneoverdx)
        dbetadr_advec_R    = get_dfdx_advec_R(beta, oneoverdx)
        dlapsedr_advec_R   = get_dfdx_advec_L(lapse, oneoverdx) 

    # t2 = time.time()
    # print("derivs found in ", t2 - t1)
        
    ####################################################################################################
    
    # make containers for rhs values
    rhs_phi   = np.zeros_like(phi)
    rhs_psy   = np.zeros_like(psy)
    rhs_Pi    = np.zeros_like(Pi)

    rhs_chi = np.zeros_like(chi)
    rhs_a = np.zeros_like(a)
    rhs_b = np.zeros_like(b)
    rhs_K   = np.zeros_like(K)
    rhs_X = np.zeros_like(X)
    rhs_AX = np.zeros_like(AX)   
    rhs_Lambda = np.zeros_like(Lambda)
    rhs_beta  = np.zeros_like(beta)
    rhs_lapse   = np.zeros_like(lapse)
    rhs_br   = np.zeros_like(br)
    
    ####################################################################################################    


   

    # get masked rhs

    m_rhs_chi = get_rhs_chi(lapse[mask], K[mask], cov_beta[mask], sigma_frame)

    m_rhs_a = get_rhs_a(a[mask], Aa[mask], lapse[mask], dbetadr[mask], cov_beta[mask], sigma_frame)     

    m_rhs_b = get_rhs_b(r[mask], b[mask], Ab[mask], lapse[mask], beta[mask], cov_beta[mask], sigma_frame) 

    m_rhs_K =  get_rhs_K(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], em4chi[mask], dchidr[mask], K[mask], dKdr[mask],
                         Aa[mask], Ab[mask], lapse[mask], d2lapsedr2[mask], dlapsedr[mask],
                         matter_rho, matter_Sa, matter_Sb)
    
    m_rhs_AX = get_rhs_AX(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], X[mask], dXdr[mask], d2Xdr2[mask], em4chi[mask], dchidr[mask],
                          lapse[mask], dlapsedr[mask],  beta[mask], Lambda[mask], AX[mask], K[mask], matter_Sa, matter_Sb,
                          dr_dlapsedr_over_r[mask], dr_dchidr_over_r[mask], dr_Lambda_over_r[mask])
     
    m_rhs_X = get_rhs_X(r[mask], a[mask], b[mask], AX[mask], lapse[mask], X[mask], beta[mask], dr_beta_over_r[mask])

    m_rhs_Lambda =  get_rhs_Lambda(r[mask], a[mask], b[mask], dbdr[mask], dchidr[mask], dKdr[mask],  Aa[mask], Ab[mask], dAadr[mask], 
                                    Lambda[mask], lapse[mask], dlapsedr[mask], matter_Jr, sigma_frame,
                                   d2betadr2[mask], cov_beta[mask], dr_beta_over_r[mask], dr_cov_beta[mask])
    
    m_rhs_phi, m_rhs_psy m_rhs_Pi = get_matter_rhs(phi[mask], psy[mask], Pi[mask], dpsydr[mask], a[mask], b[mask], dadr[mask], dbdr[mask], em4chi[mask],
                                    dchidr[mask], K[mask], lapse[mask], dlapsedr[mask], dr_lapsePi[mask]) 
    
    
    #### TODO: choose gauge evolution
    ####### rhs Gauge vars
    # rhs_br[ix]     = 0.75 * rhs_Lambda[ix] - eta * br[ix]
    # rhs_beta[ix] = br[ix]
    # rhs_lapse[ix]  = - 2.0 * lapse[ix] * K[ix] 
    

    # Write the RHS into the final arrays    
    rhs_chi[mask] = m_rhs_chi
    rhs_a[mask] = m_rhs_a
    rhs_b[mask] = m_rhs_b
    rhs_K[mask] = m_rhs_K
    rhs_X[mask] = m_rhs_X
    rhs_AX[mask] = m_rhs_AX
    rhs_Lambda[mask] = m_rhs_Lambda
    rhs_phi[mask] = m_rhs_phi
    rhs_psy[mask] = m_rhs_psy
    rhs_Pi[mask] = m_rhs_Pi

        
    # Add advection to time derivatives (this is the bit coming from the Lie derivative   
    # RIGHT side advec (upwind)
    maskR = (beta>0)
    rhs_phi[maskR]      += beta[maskR] * dphidr_advec_R[maskR]
    rhs_psy[maskR]      += beta[maskR] * dpsydr_advec_R[maskR] + psy[maskR]* dbetadr_advec_R[maskR]
    rhs_Pi[maskR]       += beta[maskR] * dPidr_advec_R[maskR]
    rhs_chi[maskR]     += beta[maskR] * dchidr_advec_R[maskR]
    rhs_a[maskR]       += beta[maskR] * dadr_advec_R[maskR] + 2*a[maskR]*dbetadr_advec_R[maskR]
    rhs_b[maskR]       += beta[maskR] * dbdr_advec_R[maskR]
    rhs_K[maskR]       += beta[maskR] * dKdr_advec_R[maskR]
    rhs_Lambda[maskR]  += beta[maskR] * dLambdadr_advec_R[maskR] + Lambda[maskR] * dbetadr_advec_R[maskR]
    rhs_X[maskR]       += beta[maskR] * dXdr_advec_R[maskR]
    rhs_AX[maskR]      += beta[maskR] * dAXdr_advec_R[maskR]

    # NB optional to add advection to lapse and shift vars
    # rhs_lapse       += beta[maskR] * dlapsedr_advec_R[maskR]
    # rhs_br[maskR]      += 0.0
    # rhs_beta[maskR]  += 0.0


    # LEFT side advec (downwind)
    maskL = (beta<0)
    rhs_phi[maskL]      += beta[maskL] * dphidr_advec_L[maskL]
    rhs_psy[maskL]      += beta[maskL] * dpsydr_advec_L[maskL] + psy[maskL]* dbetadr_advec_L[maskL]
    rhs_Pi[maskL]       += beta[maskL] * dPidr_advec_L[maskL]
    rhs_chi[maskL]     += beta[maskL] * dchidr_advec_L[maskL]
    rhs_a[maskL]       += beta[maskL] * dadr_advec_L[maskL] + 2*a[maskL]*dbetadr_advec_L[maskL]
    rhs_b[maskL]       += beta[maskL] * dbdr_advec_L[maskL]
    rhs_K[maskL]       += beta[maskL] * dKdr_advec_L[maskL]
    rhs_Lambda[maskL]  += beta[maskL] * dLambdadr_advec_L[maskL] + Lambda[maskL] * dbetadr_advec_L[maskL]
    rhs_X[maskL]       += beta[maskL] * dXdr_advec_L[maskL]
    rhs_AX[maskL]      += beta[maskL] * dAXdr_advec_L[maskL]

    # NB optional to add advection to lapse and shift vars
    # rhs_lapse       += beta[maskL] * dlapsedr_advec_L[maskL]            
    # rhs_br[maskL]      += 0.0
    # rhs_beta[maskL]  += 0.0
    

    
            
    # end of rhs iteration over grid points   
    # t3 = time.time()
    # print("rhs iteration over grid done in ", t3 - t2)
    
    ####################################################################################################

    #package up the rhs values into a vector rhs (like current_state) for return - see uservariables.py                     
    pack_state(rhs, N_r, rhs_chi, rhs_a, rhs_b, rhs_K, rhs_AX, rhs_X, rhs_Lambda, rhs_lapse, rhs_beta, rhs_br,
                     rhs_phi, rhs_Pi, rhs_S)

    #################################################################################################### 
            
    # finally add Kreiss Oliger dissipation which removed noise at frequency of grid resolution
    # sigma = 10.0 # kreiss-oliger damping coefficient, max_step should be limited to 0.1 R/N_r
    
    diss = np.zeros_like(current_state) 
    for ivar in range(0, NUM_VARS) :
        ivar_values = current_state[(ivar)*N:(ivar+1)*N]
        ivar_diss = np.zeros_like(ivar_values)
        if(r_is_logarithmic) :
            ivar_diss = get_logdissipation(ivar_values, oneoverlogdr, sigma)
        else : 
            ivar_diss = get_dissipation(ivar_values, oneoverdx, sigma)
        diss[(ivar)*N:(ivar+1)*N] = ivar_diss
    rhs += diss
    
    # t4 = time.time()
    # print("KO diss done in ", t4 - t3)    
    
    #################################################################################################### 
    
    # see gridfunctions for these, or https://github.com/GRChombo/engrenage/wiki/Useful-code-background
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    fill_outer_boundary(current_state, dx, N, r_is_logarithmic)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(rhs, dx, N, r_is_logarithmic)
    
    # t5 = time.time()
    # print("Fill boundaries done in ", t5 - t4) 
                
    #################################################################################################### 
    
    # Some code for checking timing and progress output
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress bar]
    # its values can be carried between function calls throughout the ODE integration
    last_t, deltat = time_state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded number:
    time_state[0] = last_t + deltat * n
    
    # t6 = time.time()
    # print("Check timing and output ", t6 - t5) 
    
    # end_time = time.time()
    # print("total rhs time at t= ", t_i, " is, ", end_time-start_time)
        
    ####################################################################################################
    





    ##### DEBUG ########
    """
    
    msk = np.ones_like(a, dtype=bool)
    msk[:num_ghosts] = 0
    msk[-num_ghosts:] = 0
    point = 10

    # deb_state = [chi, a, b, K, AX, X, Lambda, lapse, beta, br, phi, psy, Pi ]
    # print(f"\n\nstate at time {t_i}:")
    # for iv, var in enumerate(deb_state):
    #     print(f'{variable_names[iv]}  ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )


    #Finally return the rhs
    return rhs
    """
    