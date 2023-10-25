# initial data file

import sys
# sys.path.append("/home/admin/git/GREx/engrenage_MSPBH/")
sys.path.append("../")


from source.uservariables import *
from source.tensoralgebra import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.gridfunctions import *
from source.bssnsphsym import *
from source.fluidmatter import *
from munch import Munch

import numpy as np
import scipy.optimize as opt
# from scipy.optimize import bisect, root
#from scipy.interpolate import interp1d


# INITIAL PARAMS 

# Curv. pert. 
idata_params = Munch(dict())
idata_params.use_fNL_expansion=True
idata_params.nu = 0.9
idata_params.fNL = -1
idata_params.n_Horizons=20
idata_params.omega = 1./3



def get_kstar(params, idata_params):
    return (idata_params.n_Horizons/params.H_ini/2.7471)**-1


##########################################################
### Curvature perturb.  Long wave limit 

def MySinc(x): 
    return np.sin(x)/x

def get_zeta(r, k_star):  # curvature 
    
    # Load curv. params
    nu = idata_params.nu
    fNL = idata_params.fNL
    
    # load simplifications
    x = r    
    k = k_star    
    A = nu
    B = 3./5*fNL*nu**2
    
    # off = (nu * np.pi/k_star) 
    off = 0
    off = 0.76395   # excpected value sinc()^2        
    
    zeta =  nu * MySinc(k*x) + B * ((MySinc(k*x))**2  - off**2) 
    if off: print("offset in zeta is : ", off, end='\r')
    
    return zeta
    
def get_dzetadr(r, k_star):  # d/dr curvature 
    
    # Load curv. params
    nu = idata_params.nu
    fNL = idata_params.fNL
    
    # load simplifications
    x = r    
    k = k_star    
    A = nu
    B = 3./5*fNL*nu**2    
    
    dr_zeta =  (k*x*np.cos(k*x) - np.sin(k*x)) * (A + 2*B*MySinc(k*x)) /  (k*x**2)
        
    """ 
    Wolfram alpha: 
    d/dx(A MySinc(x k) + B MySinc(x k)^2) = 
            ((k x cos(k x) - sin(k x)) (A + 2 B MySinc(k x)))/(k x^2)
    """            
    
    return dr_zeta
    
def get_d2zetadr2(r, k_star):  # d2/dr2 curvature 
    
    # Load curv. params
    nu = idata_params.nu
    fNL = idata_params.fNL
    
    # load simplifications
    x = r    
    k = k_star    
    A = nu
    B = 3./5*fNL*nu**2
    
    d2dr2_zeta =  A * k * ((2.* np.sin(k * x))/(k**2 * x**3) - \
                  (2.* np.cos(k*x))/(k*x**2) -  np.sin(k*x)/x) + \
                  B*(2.*k * MySinc(k*x) * ((2*np.sin(k*x))/(k**2*x**3) - \
                  (2*np.cos(k*x))/(k*x**2) - np.sin(k*x)/x) +  \
                  2.*k**2 * (np.cos(k*x)/(k*x) - np.sin(k*x)/(k**2*x**2))**2)
                  
                  
    """ 
    Wolfram alpha: 
    d^2/dx^2(A MySinc(x k) + B MySinc(x k)^2) = 
        A k ((2 sin(k x))/(k^2 x^3) 
        - (2 cos(k x))/(k x^2) - sin(k x)/x) 
        + B (2 k MySinc(k x) ((2 sin(k x))/(k^2 x^3)
        - (2 cos(k x))/(k x^2) - sin(k x)/x) 
        + 2 k^2 (cos(k x)/(k x) - sin(k x)/(k^2 x^2))^2)
        

    """
    
    return d2dr2_zeta


#####################################################################
############### Gradient expansion pert. ~  tildes
# Pertrubative equations (tildes) for initial data


def get_tilde_rho(r, rm, kstar, omega):
    
    zeta_rm = get_zeta(rm, kstar)
    zeta = get_zeta(r, kstar)
    d2dr2_zeta = get_d2zetadr2(r, kstar)
    dzetadr = get_dzetadr(r, kstar)
    exp_ratio = np.exp(2*zeta_rm) / np.exp(2*zeta )
    
    trho  = - 2*(1+omega)/(5+3*omega) * exp_ratio * ( \
            # d2dr2_zeta + dzetadr * (2/r + 0.5*dzetadr) *rm**2 )        ## the position of rm^2 varies in Escriva and Musco papers  # A. Escriva 2202.01028.pdf
            d2dr2_zeta + dzetadr * (2/r  + 0.5*dzetadr) ) *rm**2         ##   Musco  1809.02127.pdf
    
    return trho 
    
def get_tilde_rho_altern(r, rm, kstar,  omega, tilde_U, dr_tildeU):
    
    zeta_rm = get_zeta(rm, kstar)
    zeta = get_zeta(r,kstar)
    dzetadr = get_dzetadr(r,kstar)

    func =  ( 3*tilde_U*dzetadr + dr_tildeU)
    
    trho = -(1+omega) / (1+r*dzetadr) * func
    

    
    return trho
    
    
def get_tilde_U(r, rm, kstar, omega):
    
    zeta_rm = get_zeta(rm, kstar)
    zeta = get_zeta(r, kstar)
    d2dr2_zeta = get_d2zetadr2(r, kstar)
    dzetadr = get_dzetadr(r, kstar)
    
    exp_ratio = np.exp(2*zeta_rm) / np.exp(2*zeta)
    
    tilde_U  =  1./(5+3*omega) * \
            exp_ratio * dzetadr * rm**2 * \
            (2/r + dzetadr)
    
    return tilde_U
    
    
def get_tilde_M(r, rm, kstar, omega):
    
    tilde_U = get_tilde_U(r, rm, kstar, omega)
    
    tilde_M = -3*(1+omega)*tilde_U
    
    return tilde_M
    
def get_tilde_R(r, rm, kstar, omega):
        
    tilde_rho = get_tilde_rho(r, rm, kstar, omega)
    tilde_U = get_tilde_U(r, rm, kstar, omega)
    
    tilde_R = - omega/(1+3*omega)/(1+omega) * tilde_rho + \
              1./(1+3*omega) * tilde_U
              
    return tilde_R
    
#####################################################################
############### Gradient expansion total     
# initial data functions  

def get_expansion_R(t, r, rm, omega, epsilon, params):
    
    a_ini = params.a_ini
    t_ini = params.t_ini    
    kstar = get_kstar(params, idata_params)
    
    
    a = get_scalefactor(t, omega, a_ini, t_ini)
    
    tilde_R = get_tilde_R(r, rm, kstar, omega)     
    zeta = get_zeta(r, kstar) 
    
    out_R =  a * np.exp(zeta) * r   * (1 + epsilon**2 * tilde_R)                      
    
    # print("a ini" , a)
    return out_R  ##
    
def get_expansion_U(t, r, rm, omega, epsilon, params):
    
    a_ini = params.a_ini
    t_ini = params.t_ini    
    kstar = get_kstar(params, idata_params)
    
    H = get_Hubble(t, omega, t_ini)
    tilde_U = get_tilde_U(r, rm, kstar, omega)
    R = get_expansion_R(t, r, rm, omega, epsilon, params)
    
    out_U = H*R * (1 + epsilon**2 * tilde_U)
    
    # print("H ini" , H)
    return out_U
    
def get_expansion_rho(t, r, rm, omega, epsilon, params):
    
    t_ini = params.t_ini
    rho_bkg_ini = params.rho_bkg_ini
    kstar = get_kstar(params, idata_params)
    
    t_over_t_ini = t/t_ini
    rho_bkg = get_rho_bkg(t_over_t_ini, rho_bkg_ini)
    tilde_rho = get_tilde_rho(r, rm, kstar, omega)
    
    out_rho = rho_bkg * (1 + epsilon**2 * tilde_rho)
    return out_rho


# def get_expansion_rho_altern(t, r, rm, omega, epsilon, U, dU, params):
    
    # t_ini = params.t_ini
    # rho_bkg_ini = params.rho_bkg_ini
    
    # t_over_t_ini = t/t_ini
    # rho_bkg = get_rho_bkg(t_over_t_ini, rho_bkg_ini)
    # tilde_rho = get_tilde_rho_altern(r, rm, kstar, omega, U, dU)
    
    # out_rho = rho_bkg * (1 + epsilon**2 * tilde_rho)
    # return out_rho


def get_expansion_M(t, r, rm, omega, epsilon, params):
    
    t_ini = params.t_ini
    rho_bkg_ini = params.rho_bkg_ini
    kstar = get_kstar(params, idata_params)
    
    t_over_t_ini = t/t_ini
    rho_bkg = get_rho_bkg(t_over_t_ini, rho_bkg_ini)

    tilde_M = get_tilde_M(r, rm, kstar, omega)
    R = get_expansion_R(t, r, rm, omega, epsilon, params)
    
    
    # rho_bkg = 1
    
    out_M= 4*np.pi/3 * rho_bkg * R**3  * (1 + epsilon**2 * tilde_M)
    return out_M        


################################################################


def get_rm(params, idata_params, print_out=0):
        
    def _root_func(r) :
        dz = get_dzetadr(r)
        ddz = get_d2zetadr2(r)        
        return dz + r * ddz
    
    kstar = get_kstar(params, idata_params)
    a, b = [idata_params.n_Horizons/params.H_ini, 100]
    try:
        xs = np.linspace(a, b, 100)
        ys = _root_func(xs)
        sa = np.sign(ys[0])
        idx = np.where( np.sign(ys)*sa < 0)[0] 
        b = xs[idx][0]    

        rm = opt.brentq(_root_func, a, b)
    except:
        rm = a
        
    if print_out: 
        L = get_L_pert(1, rm, kstar)
        eps = get_epsilon(params.H_ini, L)
        print(f"epsilon is {eps}, rm is {rm}")

    return rm 


def get_expansion_Compaction(r, omega):
    
    dzeta = get_dzetadr(r)
    C = 3 * (1+omega)/(5+3*omega) * (1 - (1 + r*dzeta)**2) 
    return C
    
    
def get_L_pert(a, rm, kstar):
	
    zeta_at_rm = get_zeta(rm, kstar)
    L = a * rm * np.exp(zeta_at_rm)
    return L 
    
def get_epsilon(H, L):
    
    epsilon = 1/(H*L)
    return epsilon 


def get_omega():
    
    omega = idata_params.omega
    return omega

    
##############################################################


# Function to get initial state 

def get_initial_state(params) :
    
    R_max = params.r_max 
    N_r = params.N_r
    r_is_logarithmic=params.r_is_logarithmic
    t_ini=params.t_ini
    a_ini = params.a_ini
    kstar = get_kstar(params, idata_params)
    
    # Set up grid values
    dx, N, r, logarithmic_dr = setup_grid(R_max, N_r, r_is_logarithmic)

    
    # predefine some userful quantities
    oneoverlogdr = 1.0 / logarithmic_dr
    oneoverlogdr2 = oneoverlogdr * oneoverlogdr
    oneoverdx  = 1.0 / dx
    oneoverdxsquared = oneoverdx * oneoverdx
    
    initial_state = np.zeros(NUM_VARS * N)
    initial_var = np.zeros(N)
    
    rm = get_rm(params, idata_params) 
    omega = get_omega() 
    Hubble = get_Hubble(t_ini, omega) 
    L_pert = get_L_pert(a_ini, rm, kstar) 
    epsilon = get_epsilon(Hubble,L_pert)      

    rho_fluid = Hubble**2/eight_pi_G *3 

    # scalefactor = a_ini * np.exp(2*chi) # so   chi = 0.5 * zeta(r)
    zeta = get_zeta(r, kstar) 
    phi =  np.exp(0.5*zeta)
    dphidr = get_dfdx(phi, oneoverdx)
    d2phidr2 = get_d2fdx2(phi, oneoverdxsquared)
    myrho = (- phi**-5 * (d2phidr2 + 2*dphidr/r) + 0.75*Hubble**2) / (2*np.pi)

    myrho = Hubble**2/eight_pi_G *3


    chi = 0.5*zeta   #initial_var + 1
    a   = initial_var*0 + 1
    b   = initial_var*0 + 1
    K   = initial_var*0 - 3*Hubble
    AX  = initial_var*0 + 0 
    X   = initial_var*0 + 0 
    Lambda = initial_var + 0 
    lapse  = initial_var + 1
    beta   = initial_var + 0 

    br = initial_var + 0 

    D = initial_var + 0
    E = myrho   # initial_var + rho_fluid
    S = initial_var + 0 


    pack_state(initial_state, N_r, chi, a, b, K, AX, X, Lambda, lapse, beta, br, D, E, S)

        
    print(f' epsilon is {epsilon}')

    # overwrite inner cells using parity under r -> - r
    fill_inner_boundary(initial_state, dx, N, r_is_logarithmic)
    
    # overwrite outer boundaries with extrapolation (order specified in uservariables.py)
    fill_outer_boundary(initial_state, dx, N, r_is_logarithmic)
    
                           
    return r, initial_state
    
    



if __name__ == "__main__":
    
    #########################################################
    #### Test  Plot Zeta, Compfaction (C), and M, R
    
    
    params = Munch(dict())
    # cosmology
    params.omega = 1./3
    params.t_ini = 1.
    params.H_ini = 2./(3.*(1.+params.omega))/params.t_ini # alpha/t_ini
    params.rho_bkg_ini =  3./(8.*np.pi) *params.H_ini**2
    params.a_ini = 1
    # grid
    params.N_r = 500
    params.R_max = 400 * params.H_ini
    params.r_is_logarithmic = False
    params.sigma_factor = 1
    params.dt_multiplier = 0.01
    params.dx = params.r_max/params.N_r
    params.dt0 = params.dx * params.dt_multiplier
    params.n_Horizons = 10
    params.kstar = (params.n_Horizons/params.H_ini/2.7471)**-1

    Do_zeta_C_rm_test = False
    if Do_zeta_C_rm_test: 
        x = np.linspace(0.001, params.R_max, 200)
        y = get_zeta(x, params.kstar)

        import matplotlib.pyplot as plt

        def _root_func(r) :
                dz = get_dzetadr(r)
                ddz = get_d2zetadr2(r)
                
                return dz + r * ddz

        y2 = get_expansion_Compaction(x, 1./3.)
        y3 = _root_func(x)

        y  = y/y.max()
        y2 = y2/y2.max()
        y3 = y3/y3.max()

        rm = get_rm(params, idata_params)

        plt.plot(x, y, label=r"$\zeta$")
        plt.plot(x, y2, label="C")
        plt.plot(x, y3, label="root func")
        plt.axvline(rm, color="k", ls="--", label="rm")
        plt.legend()
        plt.show()


        rho_bkg = get_rho_bkg(1, params.rho_bkg_ini)
        U = lambda r:  get_expansion_U(1, r, rm, 1./3., 0.001)
        R = lambda r: get_expansion_R(1, r, rm, 1./3., 0.001)
        M = lambda r:  get_expansion_M(1, r, rm, 1./3., 0.001)
        rho = lambda r: get_expansion_rho(1, r, rm, 1./3., 0.001)
        # C = lambda r : compact_function(M(r), R(r), rho_bkg)

        y  = U(x)
        y2 = R(x)
        y3 = M(x)
        y4 = rho(x)

        # y  = y/y.max()
        # y2 = y2/y2.max()
        # y3 = y3/y3.max()
        # y4 = y4/y4.max()

        x = x/rm
        nrm = 1 #rm  

        plt.plot(x, y, label=r"$U$")
        plt.plot(x, y2, label="R")
        plt.plot(x, y3, label="M")
        plt.plot(x, y4, label="rho")

        plt.axvline(nrm, color="k", ls="--", label="rm")
        plt.yscale('log')
        plt.legend()
        plt.show()

    ########
    #  Test get initial state: 

    Do_initial_state_test = True
    if Do_initial_state_test: 
        
        import matplotlib.pyplot as plt
        r_is_logarithmic = 0
        N_r = 200
        
        
        r, initial_state = get_initial_state(params.R_max, N_r, r_is_logarithmic)

        # plt.plot(initial_state)
        # plt.yscale('log')
        # plt.show()
        
        
        #unpackage the vector for readability
        (initial_U, initial_R , initial_M, initial_rho) = unpack_state(initial_state, N_r)
        
        #plot initial conditions
        plt.xlabel('r')
        plt.plot(r, initial_U, '-o', label='U') # zero, but plot as dots to see the grid
        plt.plot(r, initial_R, label='R')
        plt.plot(r, initial_M, label='M')
        plt.plot(r, initial_rho, label='rho')
        plt.legend(loc='best')
        plt.grid()
        #plt.xlim(-0.25,5.0)
        #plt.ylim(-0.0005,0.0005)
        # plt.xlim(0, 3)
        plt.yscale('log')
        plt.show()
        plt.clf()
        
        C = compact_function(initial_M, initial_R, params.rho_bkg_ini)
        C = C/C.max()
        
        Gamma = np.sqrt(1 + initial_U**2 - 2*initial_M/initial_R)
        
        plt.plot(r, C) 
        plt.plot(r, Gamma) 
        # plt.ylim(-1, 1.5)
        # plt.xlim(0, 3)
        plt.ylabel('C')
        plt.ylabel('r')
        plt.show()


    #########################################################
