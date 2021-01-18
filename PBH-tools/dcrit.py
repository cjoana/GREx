import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.optimize as opt
import matplotlib.pyplot as plt

def TransferFunction(k, t):
    sq = 1./np.sqrt(3)
    arg = k*t*sq
    return 3 * (np.sin(arg) - arg*np.cos(arg) ) / arg**3

def GaussianPowerSpectrum(k, P0=1,kstar=1, sigma=1):

    return P0 * np.exp( -(k -kstar)**2/(2*sigma**2))


def LogNormalPowerSpectrum(k, P0=1, kstar=1, sigma=1):
    return P0 * np.exp(- np.log(k/kstar) ** 2 / (2 * sigma**2))


def PowerSpectrum(k,t):

    return GaussianPowerSpectrum(k) * TransferFunction(k,t)

def ShapeRHS(t, rm=1, print_errors=False):

    cos = lambda k: (k ** 4 * np.cos(k * rm) * PowerSpectrum(k, t))
    sin = lambda k: (k ** 4 * np.sin(k * rm) * PowerSpectrum(k, t))
    cosint = integrate.quad(cos, 0, np.inf)
    sinint = integrate.quad(sin, 0, np.inf)

    coserr = cosint[1]
    sinerr = sinint[1]

    if print_errors:
        print("errs = ", coserr, sinerr)

    result = -0.5 * (1 + rm * cosint[0]/sinint[0])
    return result

def F_alpha(a):
    arg = 5/(2*a)
    diff = (special.gamma(arg) - special.gammainc(arg, 1/a) )
    return np.sqrt( 1 - 2/5 * np.exp(-1/a)* a**(1-arg)/diff )


def get_rm(t, guess=1, method='root'):

    def func(rm):
        integrand = lambda k: k**2*(  (k**2*rm**2-1)*np.sin(k*rm)/(k*rm) +
                                 np.cos(k*rm))*PowerSpectrum(k,t)
        integ = integrate.quad(integrand, 0, np.inf)
        return integ[0]

    if method=='root':
        sol = opt.root(func, x0=guess)
        root = sol.x
        success = sol.success

        if success:
            return float(root)
        else:
            raise Exception("failed to converge in get_rm iteration")





def ShapeValue(t, rm=1, guess=0.1, method='root'):

    def func(a):
        return F_alpha(a)*(1+F_alpha(a))*a - ShapeRHS(t, rm=rm)

    if method=='root':
        sol = opt.root(func, x0=guess)
        root = sol.x
        success = sol.success

        if success:
            return float(root)
        else:
            raise Exception("failed to converge in ShapeValue iteration")

    if method=='newton':
        root = opt.newton(func, x0=guess)
        # test =root
        # print(func(test), F_alpha(test), ShapeRHS(test,rm) ,  func(root))

        return root


def dcrit(a):

    if ( a>=0.1 and a<=3):
        return a**0.125 - 0.05
    if ( a>3 and a<=8):
        return a**0.06+0.025
    if ( a>8 and a<=30):
        return 1.15
    # print("  !!! the value of alpha is out of the allowed window (0.1, 30), alpha = {}".format(a))
    raise Exception("  !!! the value of alpha is out of the allowed window (0.1, 30),\n alpha = {}".format(a))




eta=0.1
rm = get_rm(eta)
alpha = ShapeValue(eta, rm=rm)

print("rm = {}, alpha = {}". format(rm, alpha))
print("dcrit = ", dcrit(alpha))