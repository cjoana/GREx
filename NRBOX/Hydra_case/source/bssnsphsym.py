# bssnrhs.py
# as in Alcubierre https://arxiv.org/pdf/1010.4013.pdf
# see also Fufza https://arxiv.org/pdf/1809.02127.pdf

# Spherical symmetry case
# IMPORTANT: terms like beta*dXdr * X*dbetadr are added separety as advec terms (with up/down-wind derivatives)

import numpy as np
from source.tensoralgebra import *

print_A_set_one = False

# chi is the (exponential) conformal factor, that is \gamma_ij = e^{4\chi) \bar gamma_ij
def get_rhs_chi(lapse, K, cov_beta, sigma_frame) :

    # Calculate rhs
    dchidt = (- one_sixth * lapse * K 
              + sigma_frame * one_sixth * cov_beta)    
    
    return dchidt

# this 'a' is the gamma_rr (down indices)
def get_rhs_a(a, Aa, lapse, dbetadr, cov_beta, sigma_frame) :

    dadt = 2*a*dbetadr - 2*one_third*sigma_frame*a*cov_beta - 2*lapse*a*Aa
    
    return dadt

# this 'b' is the   gamma_\theta\theta / r^2   (down indices);  \gamma_{\theta\theta} r^2 = b
def get_rhs_b(r, b, Ab, lapse, beta, cov_beta, sigma_frame) :
    
    dbdt = 2*b*beta/r -2*one_third*sigma_frame*b*cov_beta - 2*lapse*b*Ab 
    
    return dbdt

# K is the trace of the extrinsic curvature 
# that is K_ij = A_ij + 1/3 \gamma_ij K
def get_rhs_K(r, a, b, dadr, dbdr, em4chi, dchidr, K, dKdr,
              Aa, Ab, lapse, d2lapsedr2, dlapsedr,
              rho, Sa, Sb) :
    
    # Calculate D^k D_k lapse
    D2_lapse = em4chi/a  * \
               (d2lapsedr2 -  dlapsedr * (dadr/(2*a) - dbdr/b - 2*dchidr -2/r))

    # Calculate rhs    
    dKdt = - D2_lapse \
            + lapse * one_third * K * K  +  lapse*(Aa*Aa + 2*Ab*Ab)  \
            + 0.5 * eight_pi_G * lapse * (rho + Sa + 2*Sb)



    # DEBUG CJ TODO
    # print("D2_lapse:",  - D2_lapse[1], "  at r", r[1])
    # geo = lapse * one_third * K * K  +  lapse*(Aa*Aa + 2*Ab*Ab) 
    # print('  geo',  geo[1])
    # mat =  + 0.5 * eight_pi_G * lapse * (rho + Sa + 2*Sb)
    # print('  mat', mat[1])
    # print("")


    return dKdt

def get_rhs_Aa(r, a, b, dadr,dbdr, em4chi, dchidr, lapse, dlapsedr, d2lapsedr2, K, Aa, Sa, Sb,
                ricci_scalar, ricci_tensor_Rr):

    # Calculate D^k D_k lapse
    D2_lapse = em4chi/a  * \
               (d2lapsedr2 -  dlapsedr * (dadr/(2*a) - dbdr/b - 2*dchidr -2/r))   
     # Calculate D^k D_k lapse
    DrDR_lapse = em4chi/a * \
                     (d2lapsedr2 - dlapsedr*(dadr/(2*a) + 2*dchidr)  )  
                    
    dAdt = - (DrDR_lapse - D2_lapse/3) \
           + lapse * (ricci_tensor_Rr - ricci_scalar/3) \
           + lapse * K * Aa -  2*eight_pi_G * lapse *( Sa - Sb) * one_third  # factor 1/3 missing in Alcubierre, but in Fuzfa

    # testing
    # dAdt = - (DrDR_lapse - D2_lapse/3) + lapse*K*Aa  - 2*eight_pi_G*lapse*(Sa - Sb)  

    # DEBUG CJ TODO
    # print("dAadt:",   dAdt[1] *a[1] )
    # geo = lapse * one_third * K * K  +  lapse*(Aa*Aa + 2*Ab*Ab) 
    # print('  geo',  geo[1])
    # mat =  + 0.5 * eight_pi_G * lapse * (rho + Sa + 2*Sb)
    # print('  mat', mat[1])
    # print("")

    return dAdt

# AX is the A_lambda from Alcubierre's paper
def get_rhs_AX(r, a, b, dadr, dbdr, X, dXdr, d2Xdr2, em4chi, dchidr,
             lapse, dlapsedr,  beta, Lambda, AX, K, Sa, Sb,
             dr_dlapsedr_over_r, dr_dchidr_over_r, dr_Lambda_over_r): 
    
    one_over_rae4chi = em4chi/(r*a)
     
    dAXdt = 2*AX*beta/r    

    dAXdt += - one_over_rae4chi * \
      (dr_dlapsedr_over_r - dlapsedr/(2*r) * (dadr/a + dbdr/b + 8*dchidr))  

    dAXdt += - lapse * one_over_rae4chi * \
      (2*dr_dchidr_over_r - dchidr/r * (dadr/a + dbdr/b + 4*dchidr))  

    dAXdt += + lapse *  em4chi/a * ( \
                          0.5*b/a*d2Xdr2 \
                        + a/r*dr_Lambda_over_r \
                        + dXdr/r*(1 + 2*b/a - 0.5*r*b*Lambda) \
                        + dadr/(a*r*r)*(0.75*dadr/a - dbdr/b) \
                        - X/r*(b*Lambda + 2*dbdr/b) \
                        + b/a*X*X  \
                        )    

    dAXdt += + lapse*K*AX - lapse*eight_pi_G*(Sa - Sb)/r/r    
    
            
    return dAXdt 

# X is the lambda function in Alcubierre's paper
def get_rhs_X(r, a, b, AX, lapse, X, beta, dr_beta_over_r) :
    
    dXdt = 2.0/r *(beta*X - a/b * dr_beta_over_r) + 2*lapse*a/b*AX
    
    return dXdt


# Lambda is \hat Lambda^r in Alcubierre
def get_rhs_Lambda(r, a, b, dbdr, dchidr, dKdr,  Aa, Ab, dAadr, 
              Lambda, lapse, dlapsedr, Jr, sigma_frame, dbetadr,
              d2betadr2, cov_beta, dr_beta_over_r, dr_cov_beta) :
    
    Xi = 2

    dLambdadt = - Lambda* dbetadr + d2betadr2/a + 2/b*dr_beta_over_r  \
              + sigma_frame *one_third * (dr_cov_beta/a + 2*Lambda*cov_beta) \
              - 2/a*(Aa*dlapsedr + lapse*dAadr) \
              + 2*lapse*(Aa*Lambda -2/(r*b)*(Aa-Ab)) \
              + lapse*Xi/a*(dAadr -2*one_third*dKdr + 6*Aa*dchidr
                            + (Aa-Ab)*(2/r + dbdr/b) - eight_pi_G*Jr)

    return dLambdadt



###### Utils 

def get_covbeta(r, a, b, dadr, dbdr, beta,dbetadr ):
    
    covbeta = dbetadr + beta * (0.5*dadr/a + dbdr/b + 2/r)
    return covbeta 

def get_AX(r,Aa):
    AX = 3*Aa*one_third/(r*r)
    return AX

def get_ricci_tensor_Rr(r, a, b, dadr, dbdr, d2adr2, em4chi, dchidr, d2chidr2, Lambda,
                     dLambdadr):
    
    ricci_tensor = -em4chi/a *(0.5*d2adr2/a - a*dLambdadr - 0.75*(dadr/a)**2 \
                                    + 0.5*(dbdr/b)**2 - 0.5*Lambda*dadr + dadr/(r*b) \
                                    + 2/(r*r)*(1 - a/b)*(1 + r*dbdr/b) \
                                    + 4*d2chidr2 - 2*dchidr*(dadr/a - dbdr/b - 2/r) \
                                )
                            
    return ricci_tensor


def get_ricci_scalar(r, a, b, dadr, dbdr, d2adr2, d2bdr2, em4chi, dchidr, d2chidr2, 
                     dLambdadr):
    
    ricci_scalar = -em4chi/a *(0.5*d2adr2/a + d2bdr2/b - a*dLambdadr - (dadr/a)**2 \
                               + 0.5*(dbdr/b)**2 + 2/(r*b)*(3-a/b)*dbdr \
                               + 4/(r*r)*(1-a/b) + 8*(d2chidr2 + dchidr*dchidr) \
                                 -8*dchidr*(0.5*dadr/a - dbdr/b - 2/r))
    return ricci_scalar


def get_constraint_Lambda(r, a,b, dadr, dbdr, Lambda, X):
    # LambC should be close to 0
    LambC = Lambda - (0.5*dadr/a - dbdr/b - 2*r*X)/a
    return LambC


def get_constraint_Ham(ricci_scalar, Aa, Ab, K, rho_ADM):
    # Ham should be close to zero
    Ham = ricci_scalar - (Aa*Aa +Ab*Ab) + 2*one_third*K*K - 2*eight_pi_G*rho_ADM    
    return Ham

def get_constraint_HamRel(ricci_scalar, Aa, Ab, K, rho_ADM):
    # Ham should be close to zero
    Ham = ricci_scalar - (Aa*Aa +Ab*Ab) + 2*one_third*K*K - 2*eight_pi_G*rho_ADM 
    Ham_abs =    np.abs(ricci_scalar) + (Aa*Aa +Ab*Ab) + 2*one_third*np.abs(K*K) + np.abs(2*eight_pi_G*rho_ADM) 

    HamRel = Ham/Ham_abs
    return HamRel

def get_constraint_Mom(r, b, dbdr, dchidr, dKdr, Aa, dAadr, AX, Jr):
    # Mom should be close to zero
    Mom = dAadr - 2*one_third*dKdr + 6*Aa*dchidr + \
        + AX*(2*r + r*r*dbdr/b) - eight_pi_G*Jr
    return Mom


def get_Aa_Ab(r, AX):
    # Use definition of A_lambda in Alcubierre's paper 
    Aa = two_thirds*r*r*AX
    Ab = -0.5*Aa
    return Aa, Ab


def get_lapse(rho, rho_bkg, omega): 
	 
    rho[rho <= 0] = rho_bkg

    A = (rho_bkg/rho)**(omega/(omega+1))

    if np.sum(A!=A)>0:
        if print_A_set_one: print("In get_lapse()  --> A imposed 1!! with", rho_bkg, rho, omega  )
        A[A!=A]=1

    return A


##########################################################################
###    FLRW and other quantities
##########################################################################


def get_scalefactor(t, omega, a_ini, t_ini):

	alpha = 2./(3.*(1.+omega))
	a = a_ini*(t/t_ini)**alpha 
	return a 

def get_Hubble(t, omega, t_ini=1):
	
	alpha = 2./(3.*(1.+omega))
	Hubble = alpha/(t/t_ini)
	return Hubble
	
def get_rho_bkg(t_over_t_ini, rho_bkg_ini):
	# Assumes FLRW evolution
	rho_bkg = rho_bkg_ini * t_over_t_ini**(-2)
	return rho_bkg



##################################################



#END
