#rhsevolution.py

####################
#  This is the code related to the rhs evolution equations. It evolves BSSN with GR-Hydrody. in
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
from source.tensoralgebra import *
from source.fluidmatter import *
from source.bssnsphsym import *

from scipy.interpolate import interp1d
    
# function that returns the rhs for each of the field vars
# see further details in https://githuarr.com/GRChomarro/engrenage/wiki/Useful-code-arrackground
def get_rhs(t_i, current_state, prev_state, params, sigma, progress_bar, time_state) :     
 
    R_max = params.r_max
    N_r = params.N_r
    r_is_logarithmic = params.r_is_logarithmic
    t_ini = params.t_ini
    rho_bkg_ini = params.rho_bkg_ini
    a_ini = params.a_ini

    # some params  (hardcoded)
    sigma_frame = 0
    t_ini = 1. 

    omega = 0.33333333333333333
    t_over_t_ini = t_i/t_ini
    H_ini = 2./(3.*(1.+omega))/t_ini # alpha/t_ini
    rho_bkg_ini =  3./(8.*np.pi) *H_ini**2
    
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R_max, N_r, r_is_logarithmic)

    """  CAREFUL """
    # fill_inner_boundary(current_state, dx, N, r_is_logarithmic)
    # fill_reflective_outer_boundary(current_state, dx, N, r_is_logarithmic)

    
    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    # this is where the rhs will go
    rhs = np.zeros_like(current_state) 
    
    ####################################################################################################
    #unpackage the state vector for readaarrility - these are the vectors of values across r values at time t_i
    # see uservariaarrles.py for naming conventions



    # Exit if too many NANs!!! 
    mask_restore = ~(current_state == current_state)
    current_state[mask_restore] = prev_state[mask_restore]
    Num_nans = np.sum(mask_restore)
    if Num_nans:
        print(f" Number of Nans {Num_nans}/ {len(current_state)}")
        if Num_nans > len(current_state)//2 :
            raise # Too many nans
        

    # Unpack variaarrles from current_state - see uservariaarrles.py
    chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S = unpack_state(current_state, N_r) 




    if True:  # Asymtotics 

        
        asym_chi =  np.log(get_scalefactor(t_i, omega, a_ini, t_ini))*0.5  
        asym_a = 1
        asym_b = 1
        asym_Aa = 0
        asym_AX = 0 
        asym_X = 0
        asym_Lambda = 0
        asym_K =  -3 * get_Hubble(t_i, omega, t_ini=t_ini)
        asym_lapse = 1.
        asym_beta = 0 
        asym_br = 0
        asym_E =   get_rho_bkg(t_i/t_ini, rho_bkg_ini)
        asym_D = 0
        asym_S = 0 

        scalefactor = get_scalefactor(t_i, omega, a_ini, t_ini)


        # some fixes with boundaries
        maskFIX = np.zeros_like(a, dtype=bool)
        idx = -num_ghosts-1  - 30
        maskFIX[idx:] = 1


        chi[maskFIX] =  np.log(scalefactor)*0.5  

        a[maskFIX] =   asym_a
        b[maskFIX] =   asym_b
        Aa[maskFIX] =  asym_Aa
        AX[maskFIX] =  asym_AX
        X[maskFIX] =  asym_X
        Lambda[maskFIX] = asym_Lambda
        K[maskFIX] =  asym_K
        lapse[maskFIX] = asym_lapse
        beta[maskFIX] = asym_beta
        br[maskFIX] = asym_br
        E[maskFIX] = asym_E
        D[maskFIX] = asym_D
        S[maskFIX] = asym_S

        # pack_state(current_state, N_r, chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S)



    floor = 1e-20
    em4chi = np.exp(-4*chi)
    # print("em4chimax = ", np.max(em4chi), np.min(em4chi))
    em4chi = np.clip(em4chi, floor, 10000.)
    em4chi[em4chi<=0] = floor
    em4chi[~(em4chi==em4chi)] = floor

    # Aa, Ab = get_Aa_Ab(r, AX)
    Ab = -0.5*Aa
    
    # convert conserved variables to fluid variables 
    rhofluid, P, W, V = get_rhofluid_pressure_W_velocity(D, E, S, a, em4chi, omega)
    # rhofluid = get_rhofluid(D, E, S, a, em4chi, omega)
    
     # create mask to filter out ghosts
    mask = np.zeros_like(D, dtype=bool)
    mask[num_ghosts:-num_ghosts] = 1
                
    # Matter sources - see mymatter.py                                                                       
    matter_rho      = get_rho( D[mask], E[mask] )
    matter_Si       = get_Si(S[mask])
    matter_Jr       = get_Jr_U(S[mask], a[mask], em4chi[mask])     #upper index 
    matter_Sa, matter_Sb  = get_Sa_Sb(r[mask], a[mask], b[mask], D[mask], E[mask], V[mask], P[mask], em4chi[mask])



    ### Set GAUGE static vals
    # beta = beta*0
    # br = br*0
    # lapse = lapse*0  +1 #+ (rho_bkg/(D+E))**(omega/(omega+1))
    # evolve_gauge = False 
    evolve_gauge = True

    # beta =  get_beta_comoving(r, K, lapse) #
    # lapse = lapse*0 + 1.


    fill_outer_boundary(current_state, dx, N, r_is_logarithmic)


    # lapse = get_lapse(rhofluid, rho_bkg, omega)  ######### CJ check matter_rho or rho_fluid ??? TODO

    # t0 = time.time()
    # print("grid and var setup done in ", t0-start_time)

    ####################################################################################################

    # get the various derivs that we need to evolve things
    if(r_is_logarithmic) : #take logarithmic derivatives

        # second derivatives
        d2Ddr2   = get_logd2fdx2(D, oneoverlogdr2)
        
        # first derivatives        
        dDdr       = get_logdfdx(D, oneoverlogdr)
        
        # first derivatives - advec left and right
        dDdr_advec_L       = get_logdfdx_advec_L(D, oneoverlogdr)
        #
        dDdr_advec_R       = get_logdfdx_advec_R(D, oneoverlogdr)         
    else: 

        # second derivatives
        # d2Ddx2     = get_d2fdx2(D, oneoverdxsquared)
        d2chidr2    = get_d2fdx2(chi, oneoverdxsquared)
        d2adr2      = get_d2fdx2(a, oneoverdxsquared)
        d2bdr2      = get_d2fdx2(b, oneoverdxsquared)
        d2lapsedr2  = get_d2fdx2(lapse, oneoverdxsquared)
        d2betadr2   = get_d2fdx2(beta, oneoverdxsquared)
        d2Xdr2      = get_d2fdx2(X, oneoverdxsquared)
    
        # first derivatives
        dDdr       = get_dfdx(D, oneoverdx)
        dEdr       = get_dfdx(E, oneoverdx)
        dSdr       = get_dfdx(S, oneoverdx)
        dVdr       = get_dfdx(V, oneoverdx)
        dPdr       = get_dfdx(P, oneoverdx)
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
        dbrdr   = get_dfdx(br, oneoverdx)

        # Extra derivatives
        cov_beta = get_covbeta(r, a, b, dadr, dbdr, beta, dbetadr)
        dr_cov_beta = get_dfdx(cov_beta, oneoverdx)
        dr_beta_over_r = get_dfdx(beta/r, oneoverdx)
        dr_dlapsedr_over_r = get_dfdx(dlapsedr/r, oneoverdx)
        dr_dchidr_over_r = get_dfdx(dchidr/r, oneoverdx)
        dr_Lambda_over_r = get_dfdx(Lambda/r, oneoverdx)
    
        # first derivatives - advec left and right
        dDdr_advec_L        = get_dfdx_advec_L(D, oneoverdx)
        dEdr_advec_L        = get_dfdx_advec_L(E, oneoverdx)
        dSdr_advec_L        = get_dfdx_advec_L(S, oneoverdx)
        dchidr_advec_L     = get_dfdx_advec_L(chi, oneoverdx)
        dadr_advec_L       = get_dfdx_advec_L(a, oneoverdx)
        dbdr_advec_L       = get_dfdx_advec_L(b, oneoverdx)
        dKdr_advec_L       = get_dfdx_advec_L(K, oneoverdx)
        dAadr_advec_L      = get_dfdx_advec_L(Aa, oneoverdx)
        dAXdr_advec_L      = get_dfdx_advec_L(AX, oneoverdx)
        dXdr_advec_L       = get_dfdx_advec_L(X, oneoverdx)
        dLambdadr_advec_L  = get_dfdx_advec_L(Lambda, oneoverdx)
        dbetadr_advec_L    = get_dfdx_advec_L(beta, oneoverdx)
        dlapsedr_advec_L   = get_dfdx_advec_L(lapse, oneoverdx)
    
        dDdr_advec_R        = get_dfdx_advec_R(D, oneoverdx)
        dEdr_advec_R        = get_dfdx_advec_R(E, oneoverdx)
        dSdr_advec_R        = get_dfdx_advec_R(S, oneoverdx)
        dchidr_advec_R     = get_dfdx_advec_R(chi, oneoverdx)
        dadr_advec_R       = get_dfdx_advec_R(a, oneoverdx)
        dbdr_advec_R       = get_dfdx_advec_R(b, oneoverdx)
        dKdr_advec_R       = get_dfdx_advec_R(K, oneoverdx)
        dAadr_advec_R      = get_dfdx_advec_R(Aa, oneoverdx)
        dAXdr_advec_R      = get_dfdx_advec_R(AX, oneoverdx)
        dXdr_advec_R       = get_dfdx_advec_R(X, oneoverdx)
        dLambdadr_advec_R  = get_dfdx_advec_R(Lambda, oneoverdx)
        dbetadr_advec_R    = get_dfdx_advec_R(beta, oneoverdx)
        dlapsedr_advec_R   = get_dfdx_advec_L(lapse, oneoverdx) 

    # t2 = time.time()
    # print("derivs found in ", t2 - t1)
        
    ####################################################################################################
    
    # make containers for rhs values
    rhs_D   = np.zeros_like(D)
    rhs_E   = np.zeros_like(E)
    rhs_S   = np.zeros_like(S)
    rhs_V   = np.zeros_like(V)
    rhs_chi = np.zeros_like(chi)
    rhs_a = np.zeros_like(a)
    rhs_b = np.zeros_like(b)
    rhs_K   = np.zeros_like(K)
    rhs_X = np.zeros_like(X)
    rhs_Aa = np.zeros_like(Aa)  
    rhs_AX = np.zeros_like(AX)   
    rhs_Lambda = np.zeros_like(Lambda)
    rhs_beta  = np.zeros_like(beta)
    rhs_lapse   = np.zeros_like(lapse)
    rhs_br   = np.zeros_like(br)
    
    ####################################################################################################    


    ## get ricci tensors
    ricci_scalar =  get_ricci_scalar(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], d2adr2[mask], 
                                      d2bdr2[mask], em4chi[mask], dchidr[mask], d2chidr2[mask], dLambdadr[mask])
    
    ricci_tensor_Rr = get_ricci_tensor_Rr(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], d2adr2[mask], em4chi[mask],
                                     dchidr[mask], d2chidr2[mask], Lambda[mask], dLambdadr[mask])

   

    # get masked rhs

    m_rhs_chi = get_rhs_chi(lapse[mask], K[mask], cov_beta[mask], sigma_frame)

    m_rhs_a = get_rhs_a(a[mask], Aa[mask], lapse[mask], dbetadr[mask], cov_beta[mask], sigma_frame)     

    m_rhs_b = get_rhs_b(r[mask], b[mask], Ab[mask], lapse[mask], beta[mask], cov_beta[mask], sigma_frame) 

    m_rhs_K =  get_rhs_K(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], em4chi[mask], dchidr[mask], K[mask], dKdr[mask],
                         Aa[mask], Ab[mask], lapse[mask], d2lapsedr2[mask], dlapsedr[mask],
                         matter_rho, matter_Sa, matter_Sb)
    
    #TODO: set on/off regularization
    m_rhs_Aa = get_rhs_Aa(r[mask], a[mask], b[mask], dadr[mask],dbdr[mask], em4chi[mask], dchidr[mask], lapse[mask], dlapsedr[mask],
                          d2lapsedr2[mask], K[mask], Aa[mask], matter_Sa, matter_Sb, ricci_scalar, ricci_tensor_Rr)
    
    m_rhs_AX = get_rhs_AX(r[mask], a[mask], b[mask], dadr[mask], dbdr[mask], X[mask], dXdr[mask], d2Xdr2[mask], em4chi[mask], dchidr[mask],
                          lapse[mask], dlapsedr[mask],  beta[mask], Lambda[mask], AX[mask], K[mask], matter_Sa, matter_Sb,
                          dr_dlapsedr_over_r[mask], dr_dchidr_over_r[mask], dr_Lambda_over_r[mask])
     
    m_rhs_X = get_rhs_X(r[mask], a[mask], b[mask], AX[mask], lapse[mask], X[mask], beta[mask], dr_beta_over_r[mask])

    m_rhs_Lambda =  get_rhs_Lambda(r[mask], a[mask], b[mask], dbdr[mask], dchidr[mask], dKdr[mask],  Aa[mask], Ab[mask], dAadr[mask], 
                                    Lambda[mask], lapse[mask], dlapsedr[mask], matter_Jr, sigma_frame, dbetadr[mask],
                                   d2betadr2[mask], cov_beta[mask], dr_beta_over_r[mask], dr_cov_beta[mask])
    
    m_rhs_D, m_rhs_E, m_rhs_S = get_matter_rhs(r[mask], D[mask], E[mask], S[mask], V[mask], P[mask],
                                                dDdr[mask], dEdr[mask], dSdr[mask], dVdr[mask], dPdr[mask], 
                                                a[mask], b[mask], dadr[mask], dbdr[mask], dchidr[mask],
                                                em4chi[mask], K[mask], Aa[mask], lapse[mask], dlapsedr[mask])

    ####### rhs Gauge vars
    ld = np.zeros_like(E)
    if evolve_gauge: 
        eta_gauge = 0.01  #### !!! put 1/M here?? 
        lambda_driver = -0.01 * ld[mask]
        ld[-100:]= 0
        rhs_br[mask]     = lambda_driver * m_rhs_Lambda - eta_gauge * br[mask]
        rhs_beta[mask] = br[mask]
        # Kmean_gauge = np.nanmean(K[mask])
        rhs_lapse[mask]  = - 2.0 * lapse[mask] * (K[mask] - asym_K)
    

    # Write the RHS into the final arrays    
    rhs_chi[mask] = m_rhs_chi
    rhs_a[mask] = m_rhs_a
    rhs_b[mask] = m_rhs_b
    rhs_K[mask] = m_rhs_K
    rhs_X[mask] = m_rhs_X
    rhs_Aa[mask] = m_rhs_Aa
    rhs_AX[mask] = m_rhs_AX
    rhs_Lambda[mask] = m_rhs_Lambda
    rhs_D[mask] = m_rhs_D
    rhs_E[mask] = m_rhs_E
    rhs_S[mask] = m_rhs_S

        
    # Add advection to time derivatives (this is the arrit coming from the Lie derivative   
    # RIGHT side advec (upwind)
    maskR = (beta>0)
    rhs_D[maskR]       += beta[maskR] * dDdr_advec_R[maskR]
    rhs_E[maskR]       += beta[maskR] * dEdr_advec_R[maskR]
    rhs_S[maskR]       += beta[maskR] * dSdr_advec_R[maskR]
    rhs_chi[maskR]     += beta[maskR] * dchidr_advec_R[maskR]
    rhs_a[maskR]       += beta[maskR] * dadr_advec_R[maskR] 
    rhs_b[maskR]       += beta[maskR] * dbdr_advec_R[maskR]
    rhs_K[maskR]       += beta[maskR] * dKdr_advec_R[maskR]
    rhs_Lambda[maskR]  += beta[maskR] * dLambdadr_advec_R[maskR] 
    rhs_X[maskR]       += beta[maskR] * dXdr_advec_R[maskR]
    rhs_AX[maskR]      += beta[maskR] * dAXdr_advec_R[maskR]
    rhs_Aa[maskR]      += beta[maskR] * dAadr_advec_R[maskR]

    # NB optional to add advection to lapse and shift vars
    if evolve_gauge: 
        rhs_lapse[maskR]        += beta[maskR] * dlapsedr_advec_R[maskR]
        rhs_br[maskR]    += 0.0
        rhs_beta[maskR]  += 0.0

        
    # LEFT side advec (downwind)
    maskL = (beta<0)
    rhs_D[maskL]       += beta[maskL] * dDdr_advec_L[maskL]
    rhs_E[maskL]       += beta[maskL] * dEdr_advec_L[maskL]
    rhs_S[maskL]       += beta[maskL] * dSdr_advec_L[maskL]
    rhs_chi[maskL]     += beta[maskL] * dchidr_advec_L[maskL]
    rhs_a[maskL]       += beta[maskL] * dadr_advec_L[maskL]
    rhs_b[maskL]       += beta[maskL] * dbdr_advec_L[maskL]
    rhs_K[maskL]       += beta[maskL] * dKdr_advec_L[maskL]
    rhs_Lambda[maskL]  += beta[maskL] * dLambdadr_advec_L[maskL] 
    rhs_X[maskL]       += beta[maskL] * dXdr_advec_L[maskL]
    rhs_AX[maskL]      += beta[maskL] * dAXdr_advec_L[maskL]
    rhs_Aa[maskL]      += beta[maskL] * dAadr_advec_L[maskL]

    # NB optional to add advection to lapse and shift vars
    if evolve_gauge: 
        rhs_lapse[maskL]        += beta[maskL] * dlapsedr_advec_L[maskL]            
        rhs_br[maskL]    += 0.0
        rhs_beta[maskL]  += 0.0
    

    
            
    # end of rhs iteration over grid points   
    # t3 = time.time()
    # print("rhs iteration over grid done in ", t3 - t2)
    


    ####################################################################################################

    # outfix = "Reflective" #   "None", "asymtotic" 
    outfix = "asymtotic"

# def get_scalefactor(t, omega, a_ini, t_ini):
# def get_Hubble(t, omega, t_ini=1):
# def get_rho_bkg(t_over_t_ini, rho_bkg_ini):


    def get_asym_evo_var(r, F, Fbkg, dtFbkg, dFdr, v=0):
        dFdt = dtFbkg  - v*dFdr - v *(F-Fbkg)/r
        return dFdt 

    if outfix=="asymtotic":  # Zero 

        # asym_chi =  np.log(get_scalefactor(t_i, omega, a_ini, t_ini))*0.5  
        # asym_a = 1
        # asym_b = 1
        # asym_Aa = 0
        # asym_AX = 0 
        # asym_X = 0
        # asym_Lambda = 0
        # asym_K =  -3 * get_Hubble(t_i, omega, t_ini=t_ini)
        # asym_lapse = 1.
        # asym_beta = 0 
        # asym_br = 0
        # asym_E =   get_rho_bkg(t_i/t_ini, rho_bkg_ini)
        # asym_D = 0
        # asym_S = 0 

        # scalefactor = get_scalefactor(t_i, omega, a_ini, t_ini)




        # # some fixes with boundaries
        # maskFIX = np.zeros_like(a, dtype=bool)
        # idx = -num_ghosts-1  - 30
        # maskFIX[idx:] = 1


        # chi[maskFIX] =  np.log(scalefactor)*0.5  

        # a[maskFIX] =   asym_a
        # b[maskFIX] =   asym_b
        # Aa[maskFIX] =  asym_Aa
        # AX[maskFIX] =  asym_AX
        # X[maskFIX] =  asym_X
        # Lambda[maskFIX] = asym_Lambda
        # K[maskFIX] =  asym_K
        # lapse[maskFIX] = asym_lapse
        # beta[maskFIX] = asym_beta
        # br[maskFIX] = asym_br
        # E[maskFIX] = asym_E
        # D[maskFIX] = asym_D
        # S[maskFIX] = asym_S

        dt_scalefactor_bkg = np.sqrt(8*np.pi*asym_E/3) * scalefactor
        dt_K_bkg = 4*np.pi*asym_lapse*(asym_E + asym_E/3) + asym_lapse*(asym_K**2)/3   #4*np.pi/6 * (asym_E + asym_E/3)
        dt_E_bkg = asym_lapse * asym_K*(asym_E + asym_E/3)

        maskFIX = np.zeros_like(a, dtype=bool)
        idx_e =  -1 # idx 
        idx_i = idx_e - 1  - 30
        maskFIX[idx_i:idx_e] = 1
        


        #rhs 
        rhs_chi[maskFIX] = get_asym_evo_var(r[maskFIX], chi[maskFIX], asym_chi, dt_scalefactor_bkg, dchidr[maskFIX], v=1)
        rhs_a[maskFIX] = get_asym_evo_var(r[maskFIX], a[maskFIX], asym_a, 0, dadr[maskFIX], v=1)
        rhs_b[maskFIX] = get_asym_evo_var(r[maskFIX], b[maskFIX], asym_b, 0, dbdr[maskFIX], v=1)
        rhs_Aa[maskFIX] = get_asym_evo_var(r[maskFIX], Aa[maskFIX], asym_Aa, 0, dAadr[maskFIX], v=1)
        rhs_AX[maskFIX] = get_asym_evo_var(r[maskFIX], AX[maskFIX], asym_AX, 0, dAXdr[maskFIX], v=1)
        rhs_X[maskFIX] = get_asym_evo_var(r[maskFIX], X[maskFIX], asym_X, 0, dXdr[maskFIX], v=1)
        rhs_AX[:]=0
        rhs_X[:]=0
        rhs_Lambda[maskFIX] = get_asym_evo_var(r[maskFIX], Lambda[maskFIX], asym_Lambda, 0, dLambdadr[maskFIX], v=np.sqrt(2))
        rhs_K[maskFIX] = get_asym_evo_var(r[maskFIX], K[maskFIX], asym_K, dt_K_bkg, dKdr[maskFIX], v=1)
        rhs_lapse[maskFIX] = get_asym_evo_var(r[maskFIX], Aa[maskFIX], asym_Aa, 0, dAadr[maskFIX], v=np.sqrt(2)/scalefactor)
        rhs_beta[maskFIX] = 0 # get_asym_evo_var(r[maskFIX], beta[maskFIX], asym_beta, 0, dbetadr[maskFIX], v=1)
        
        # print("bf, " , rhs_br[maskFIX])
        rhs_br[maskFIX] =   0 # get_asym_evo_var(r[maskFIX], br[maskFIX], asym_br, 0, dbrdr[maskFIX], v=1)
        # print("af, " , rhs_br[maskFIX])
        rhs_br[:num_ghosts+1]=0 
        rhs_br[idx_i-4:]=0 

        rhs_E[maskFIX] = get_asym_evo_var(r[maskFIX], E[maskFIX], asym_E, dt_E_bkg, dEdr[maskFIX], v=1)
        rhs_D[maskFIX] = get_asym_evo_var(r[maskFIX], D[maskFIX], asym_D, 0, dDdr[maskFIX], v=1)
        rhs_S[maskFIX] = get_asym_evo_var(r[maskFIX], S[maskFIX], asym_S, 0, dSdr[maskFIX], v=1)

        # lapse[:] = 1.
        # rhs_lapse[:] = 0

        # o =  rhs_K[maskFIX]
        # print("bf, " , rhs_K[maskFIX])
        # rhs_K[maskFIX] = get_asym_evo_var(r[maskFIX], K[maskFIX], asym_K, dt_K_bkg, dKdr[maskFIX], v=1)
        # print("af, " , o/rhs_K[maskFIX])

        pack_state(current_state, N_r, chi, a, b, K, Aa, AX, X, Lambda, lapse, beta, br, D, E, S)

    ####################################################################################################

    #package up the rhs values into a vector rhs (like current_state) for return - see uservariaarrles.py                     
    pack_state(rhs, N_r, rhs_chi, rhs_a, rhs_b, rhs_K, rhs_Aa, rhs_AX, rhs_X, rhs_Lambda, rhs_lapse, rhs_beta, rhs_br,
                     rhs_D, rhs_E, rhs_S)

    #################################################################################################### 


    # finally add Kreiss Oliger dissipation which removed noise at frequency of grid resolution
    # sigma = 10.0 # kreiss-oliger damping coefficient, max_step should arre limited to 0.1 R/N_r
    
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
    
    # see gridfunctions for these, or https://githuarr.com/KAClough/BaarryGRChomarro/wiki/Useful-code-arrackground
    
    # overwrite outer arroundaries with extrapolation (order specified in uservariaarrles.py)
    # fill_outer_boundary(current_state, dx, N, r_is_logarithmic)
    # fill_reflective_outer_boundary(current_state, dx, N, r_is_logarithmic)
    # fill_outer_boundary_ivar(current_state, dx, N, r_is_logarithmic, 0)
    # fill_outer_boundary_ivar(current_state, dx, N, r_is_logarithmic, 3)
    # fill_outer_boundary(current_state, dx, N, r_is_logarithmic)

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(rhs, dx, N, r_is_logarithmic)
    
    # t5 = time.time()
    # print("Fill arroundaries done in ", t5 - t4) 
                
    #################################################################################################### 
    
    # Some code for checking timing and progress output
    
    # state is a list containing last updated time t:
    # state = [last_t, dt for progress arrar]
    # its values can arre carried arretween function calls throughout the ODE integration
    last_t, deltat = time_state
    
    # call update(n) here where n = (t - last_t) / dt
    n = int((t_i - last_t)/deltat)
    progress_bar.update(n)
    # we need this to take into account that n is a rounded numarrer:
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

    # deb_state = [chi, a, b, K, AX, X, Lambda, lapse, beta, br, D, E, S ]
    # print(f"\n\nstate at time {t_i}:")
    # for iv, var in enumerate(deb_state):
    #     print(f'{variable_names[iv]}  ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )
    
    # print(f'rho, Sa, Sb  -> {matter_rho[point]}  {matter_Sa[point]}  {matter_Sb[point]} ' ) 
    
    # print(f'P, Jr  -> {P[point]}  {matter_Jr[point]}  {matter_Si[point]} ' )    


    
    

    # derivs = [dadr, dbdr, d2adr2, d2bdr2, dchidr, d2chidr2, dLambdadr]
    # for iv, var in enumerate(derivs):
    #     print(f'deriv {iv}  ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )



    ricci_scalar = get_ricci_scalar(r, a, b, dadr, dbdr, d2adr2, d2bdr2, em4chi, dchidr, d2chidr2, 
                     dLambdadr)

    Ham = get_constraint_HamRel(ricci_scalar[mask], Aa[mask], Ab[mask], K[mask], matter_rho)
    var = Ham
    print(f'   Ham  ->  {Ham[point]}  : ', np.mean(var), np.std(var), np.min(var), np.max(var) )
    var = ricci_scalar
    print(f'ricci ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )
    var = Aa
    print(f'Aa ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )
    var = K * K *2/3
    print(f'K term ->  {var[point]}  : ', np.mean(var[msk]), np.std(var[msk]), np.min(var[msk]), np.max(var[msk]) )
    var = matter_rho *16*np.pi
    print(f'rho term ->  {var[point]}  : ', np.mean(var), np.std(var), np.min(var), np.max(var) )



    # print('\n\n t, chi: ', t_i, chi, K )
    rho = D+E
    if np.sum(rho!=rho) > 50 : raise()
    if t_i > 5.000 : raise()

    """



    # nans = ~(rhs==rhs)
    # rhs[nans] = 0

    # nans = (rhs==np.infty)
    # rhs[nans] = 0



    #Finally return the rhs
    return rhs
