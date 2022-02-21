import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import scipy.optimize as opt
import matplotlib.pyplot as plt


kp = 1.0  # 
# kp = 2.e6 

eta = 1.0

P0 = 1.0
# P0 = 0.0205 #* 0.8602150

def ddf(x, a):
	
	if x == a:
		return np.inf
	else:
		return 0.0

def TransferFunction(k, t):
    sq = 1./np.sqrt(3)
    arg = k*t*sq
    return 3 * (np.sin(arg) - arg*np.cos(arg) ) / arg**3

def GaussianPowerSpectrum(k, P0=P0,kp=kp, sigma=0.5):

    return P0 * np.exp( -(k - kp)**2/(2*sigma**2 * kp**2))
    

def LogNormalPowerSpectrum(k, P0=P0, kp=kp, sigma=0.5):
    return P0 * np.exp(- (np.log(k/kp))** 2 / (2 * sigma**2))


def PowerSpectrum(k,t, sigma=0.5, m = 'error'):
	if m=='g': 	P = GaussianPowerSpectrum(k * kp, sigma=sigma)
	elif m=='l': 	P = LogNormalPowerSpectrum(k *kp, sigma=sigma)
	# elif m=='p': 	P = PeakPowerSpectrum(k)
	else:
		raise
		
	out =  2.*np.pi**2 /(k**3) * P * TransferFunction(k,t)**2
	return out

def get_rm(t, guess=1, method='root', sigma=0.5, m ='g'):

    def func(rm):
        integrand = lambda k: k**2*(  (k**2*rm**2-1)*np.sin(k*rm)/(k*rm) +
                                 np.cos(k*rm))*PowerSpectrum(k,t, sigma=sigma, m=m)
        integ = integrate.quad(integrand, 0, np.inf, limit=100000, limlst=10000)
        return integ[0]

    if method=='root':
        sol = opt.root(func, x0=guess)
        root = sol.x
        success = sol.success

        if success:
            return float(root)
        else:
            raise Exception("failed to converge in get_rm iteration")



def ShapeRHS(t, rm=1, print_errors=False, m ='error'):

    cos = lambda k: (k**4 * np.cos(k * rm) * PowerSpectrum(k, t, m=m))
    sin = lambda k: (k**3 * np.sin(k * rm) * PowerSpectrum(k, t, m=m))
    cosint = integrate.quad(cos, 0, np.inf, limit=100000, limlst=10000)
    sinint = integrate.quad(sin, 0, np.inf, limit=100000, limlst=10000)

    coserr = cosint[1]
    sinerr = sinint[1]

    # print("RHS ints:", cosint[0], sinint[0], cosint[0]/sinint[0])

    if print_errors:
        print("errs = ", coserr, sinerr)

    result = -0.25 * (1 + rm * cosint[0]/sinint[0])
    return result

def F_alpha(a):
	arg = 5./(2*a)

	# gammainc  ==  inc lower gammar   Normalized  (1/gamma(arg))
	# gammaincc ==  inc upper gammar  Normalized (1/gamma(arg))
	diff = (special.gamma(arg) * special.gammainc(arg, 1/a) )

	return np.sqrt( 1 - (2./5) * np.exp(-1/a) * a**(1. - arg) /diff )

def ShapeValue(t, rm=1, guess=1.0, method='root', m='error'):

    def func(a):
        return F_alpha(a)*(1+F_alpha(a))*a - 2*ShapeRHS(t, rm=rm, m=m)
        
    # method = 'newton'    
    guess = 2 * ShapeRHS(t, rm=rm, m=m)

    if method=='root':
        sol = opt.root(func, x0=guess)
        root = sol.x
        success = sol.success

        if success:
            A = float(root)
        else:
            print("  guess used = ", guess)

            xvals = 10**np.linspace(-20, 1, 1000)
            plt.plot(xvals, func(xvals))
            plt.xscale('log')
            plt.show()

            raise Exception("failed to converge in ShapeValue iteration")

    if method=='newton':
        root = opt.newton(func, x0=guess, maxiter=1000)
        # test =root
        # print(func(test), F_alpha(test), ShapeRHS(test,rm) ,  func(root))
        A = root
    
    A_G = ShapeRHS(t, rm=rm, m=m)
    return A, A_G




def dcrit(a):

	if ( a>=0.1 and a<=7):
		return a**0.047 - 0.5
	elif ( a>7 and a<=13):
		return a**0.035-0.475
	elif ( a>13 and a<=30):
		return a**0.026 - 0.45
	else:
		return 0.
		
		
	# if ( a>=0.1 and a<=3):
		# return a**0.125 - 0.05
	# elif ( a>3 and a<=8):
		# return a**0.06+0.025
	# elif ( a>8 and a<=30):
		# return 1.15
	# else:
		# return 0.
		
		
	# print("  !!! the value of alpha is out of the allowed window (0.1, 30), alpha = {}".format(a))
	raise Exception("  !!! the value of alpha is out of the allowed window (0.1, 30),\n alpha = {}".format(a))
    

def delta(a):

	arg = 5/(2*a)

	# gammainc  ==  inc lower gammar   Normalized  (1/gamma(arg))
	# gammaincc ==  inc upper gammar  Normalized (1/gamma(arg))

	# diff = (special.gamma(arg) - special.gammainc(arg, 1/a) )
	diff = (special.gamma(arg) * special.gammainc(arg, 1/a) )

	return 4./15 *np.exp(-1/a) * a**(1-arg) /diff






# A = np.linspace(0.0001,1,1000)
# F = F_alpha(A)
# indx = (F >= 0)
# A0 = A[indx][0]
# plt.plot(A, F)
# # plt.axvline(A0)
# plt.show()


# A = np.linspace(1e1,1e15,1000)
# F = F_alpha(A)
# # indx = (F >= 0)
# # A0 = A[indx][0]
# plt.plot(A, F)
# # plt.axvline(A0)
# plt.yscale('log')
# # plt.xscale('log')
# plt.show()


# raise


# eta=0.1
# # rm = get_rm(eta)
# rm = 2.74
# alpha , alphaG = ShapeValue(eta, rm=rm, m='g')

# print("rm = {}, alpha = {},  {}". format(rm, alpha, alphaG))
# print("dcrit = ", dcrit(alpha)  , delta(alpha) )
# print("dcritG = ", dcrit(alphaG) , delta(alphaG) )

# print("\n\n ++++++++  = ")


# raise

guess = 2.0

ss = np.linspace(0.1, 1.2, 50) 
krm = []
krm2 = []
dc = []
dc2 = []
ddc = []
dddc = []
ddddc = []
ddc2 = []
dddc2 = []
ddddc2 = []
for sigma in ss:
	rm = get_rm(eta, guess=guess, sigma=sigma,m='g')
	krm.append(rm)
	a, ag = ShapeValue(eta, rm=rm, guess=0.1, method='root', m='g')
	# d = dcrit(a)
	d = delta(a)
	dc.append(d)
	d = delta(ag)
	ddc.append(d)
	d = dcrit(a)
	dddc.append(d)
	d = dcrit(ag)
	ddddc.append(d)
	
	rm = get_rm(eta, guess=guess, sigma=sigma,m='l')
	krm2.append(rm)
	a, ag = ShapeValue(eta, rm=rm, guess=0.1, method='root', m='l')
	d = delta(a)
	dc2.append(d)
	d = delta(ag)
	ddc2.append(d)
	d = dcrit(a)
	dddc2.append(d)
	d = dcrit(ag)
	ddddc2.append(d)
	# print(sigma, rm)


plt.plot(ss, krm, label='gaussian')
plt.plot(ss, krm2, label='lognormal')
plt.ylabel(r"$k_p r_m$")
plt.xlabel(r"$\sigma $")
plt.legend()
plt.show()
	

plt.plot(ss, dc, 'k-', label='gaussian')
plt.plot(ss, ddc, 'k--')
plt.plot(ss, dc2, 'b-', label='lognormal')
plt.plot(ss, ddc2, 'b--')

plt.ylabel(r"$d_c$")
plt.xlabel(r"$\sigma $")
plt.legend()
plt.show()

plt.plot(ss, dddc, 'k-', label='gaussian')
plt.plot(ss, ddddc, 'k--')
plt.plot(ss, dddc2, 'b-', label='lognormal')
plt.plot(ss, ddddc2, 'b--')
plt.ylabel(r"$d_c$")
plt.xlabel(r"$\sigma $")
plt.legend()
plt.show()

