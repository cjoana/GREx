import numpy as np

params = dict()
all_attrb = dict()
base_attrb = dict()
chombogloba_attrb = dict()
levels_attrb = dict()
boxes = dict()
data_attributes = dict()

# basic params  (MANUAL)
params['N'] = 256
params['L'] = 10
params['dt_multiplier'] = 0.01
params['is_periodic'] = [1, 1, 1]
params['ghosts'] = [0, 0, 0]

# Set components  (MANUAL)
components = np.array([
    "chi",
    "h11", "h12", "h13", "h22", "h23", "h33",
    "K",
    "A11", "A12", "A13", "A22", "A23", "A33",
    "Theta",
    "Gamma1", "Gamma2", "Gamma3",
    "lapse",
    "shift1", "shift2", "shift3",
    "B1", "B2", "B3",
    "density", "energy", "pressure", "enthalpy",
    "D", "E", "W",
    "Z1", "Z2", "Z3",
    "V1", "V2", "V3",
])

# Set boxes, for each level (MANUAL)
boxes["level_0"] = np.array([
    [0,0, 0, 255, 255, 255],
])

# boxes["level_0"] = np.array([
#     [0, 0, 0, 63, 63, 63],
# ])

boxes["level_1"] = np.array([
#    [40, 40, 40, 87, 87, 87],
])
boxes["level_2"] = np.array([
#    [104, 104, 104, 151, 151, 151],
])

# set base attibutes (MANUAL)
base_attrb['time'] = 0.0 # float!
base_attrb['iteration'] = 0
base_attrb['max_level'] = 0
base_attrb['num_levels'] = 1
base_attrb['num_components'] = components.size
base_attrb['regrid_interval_0'] = 2
base_attrb['steps_since_regrid_0'] = 0
for comp, name in enumerate(components):
    key = 'component_' + str(comp)
    tt = 'S' + str(len(name))
    base_attrb[key] = np.array(name, dtype=tt)

# def Chombo_global attributes (AUTO)
chombogloba_attrb['testReal'] = 0.0
chombogloba_attrb['SpaceDim'] = 3

# set level attributes and boxes (AUTO)
for il in range(base_attrb['num_levels']):
    levels_attrb['level_{}'.format(il)] = dict()
    ldict = levels_attrb['level_{}'.format(il)]
    ldict['ref_ratio'] = 2
    ldict['dt'] = float(params['L']) / params['N'] * params['dt_multiplier'] / (float(ldict['ref_ratio']) ** il)
    ldict['dx'] = float(params['L']) / params['N'] / (float(ldict['ref_ratio']) ** il)
    ldict['time'] = base_attrb['time']
    ldict['is_periodic_0'] = params['is_periodic'][0]
    ldict['is_periodic_1'] = params['is_periodic'][1]
    ldict['is_periodic_2'] = params['is_periodic'][2]
    ldict['tag_buffer_size'] = 3
    Nlev = int(params['N'] * (int(ldict['ref_ratio']) ** il))
    prob_dom = (0, 0, 0, Nlev - 1, Nlev - 1, Nlev - 1)
    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    ldict['prob_domain'] = np.array(prob_dom, dtype=prob_dt)

    prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                        ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
    lev_box = np.array([tuple(elm) for elm in boxes["level_{}".format(il)]], dtype=prob_dt)
    boxes["level_{}".format(il)] = lev_box

# set "data attributes" directory in levels, always the same.  (AUTO)
dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
data_attributes['ghost'] = np.array(tuple(params['ghosts']), dtype=dadt)
data_attributes['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
data_attributes['comps'] = base_attrb['num_components']
data_attributes['objectType'] = np.array('FArrayBox', dtype='S9')


###################################
###   DATA TEMPLATE        ########
###################################


import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import units as nu
from math import pi
import h5py as h5
from scipy.integrate import odeint, solve_ivp, quad
from scipy.interpolate import interp1d
import math

class eos:
	# monotropic EoS
	# transition continuity constant
	a = 0.0
	c2 = nu.c**2
	min_press = 1e-12

	def __init__(self, K, G, atm=1e-18):
		self.K = K / self.c2
		self.G = G
		self.n = 1.0 / (G - 1)
		self.atm = atm

	# pressure P(density)
	def pressure_from_density(self, density):
		return self.K * density ** self.G

	# energy density_adm = density * (1 + energy)
	def rho_from_density(self, density):
		press = self.K * density ** self.G
		energy = press/(density*(self.G-1))
		return density + density * energy
	
	# energy from dens
	def energy_from_density(self, density):
		press = self.K * density ** self.G
		return press/(density*(self.G-1))

	# for inverse functions lets define  density (P)
	def density_from_pressure(self, pressure):
		if pressure < self.min_press:
			return self.min_press
		return (pressure / self.K) ** (1 / self.G)

	def densities_from_pressure(self, pressure):
		mask = np.array(pressure < self.min_press, dtype=bool)
		out = (pressure / self.c2 / self.K) ** (1 / self.G)
		out[mask] = self.min_press
		return out


class tov:

	def __init__(self, peos, r0=1e-8, rf=1e10, verbose=0, atm=1e-8):
		self.physical_eos = peos
		self.r0 = r0
		self.rf = rf
		self.verbose = verbose
		self.atm = atm


	def tov(self, y, r):
		P, mass, lna = y
		rho = self.physical_eos.density_from_pressure(P)

		dPdr = -nu.G * (rho + P / nu.c ** 2) * (mass + 4.0 * pi * r ** 3 * P / nu.c ** 2)
		dPdr = dPdr / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
		dmdr = 4.0 * pi * r ** 2 * rho

		dlnadr = (mass + 4*pi*r**3*P)/ (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
		
		sing = (r - 2.0 * mass)**0.5
		dlnchdr =  (r**0.5 - sing)/(r * sing)

		return [dPdr, dmdr, dlnadr]

	def tov_ivp(self, r, y, crit_sr = -0.0001):		
		small_r = False if r >= crit_sr else True
		p_min = self.atm * (G - 1)
		
		P, mass, lna, lch = y
		P = np.max([P, p_min])
		dens = self.physical_eos.density_from_pressure(P)
		rho = self.physical_eos.rho_from_density(dens)
			
		if small_r:
			dPdr = -(rho + P) * (4.0 * pi * r *  ( rho/3 + P))
			dPdr = dPdr / ( 1 - 8.0 * pi * r**2 * rho/3)
			dmdr = 4.0 * pi * r ** 2 * rho

			dlnadr = (4.0 * pi *  r * ( rho/3 + P) ) / ( 1 - 8.0 * pi * r**2 * rho/3)
		else:		
			dPdr = -(rho + P ) * (mass + 4.0 * pi * r**3 * P )
			dPdr = dPdr / (r * (r - 2.0 * mass))
			dmdr = 4.0 * pi * r**2 * rho

			dlnadr = (mass + 4*pi*r**3 * P)/ (r * (r - 2.0 * mass))
		
		sing = (r - 2.0 * mass)**0.5
		dlnchdr =  (r**0.5 - sing)/(r * sing)
		
		
		if not sing==sing:
			print("  !! ", dlnchdr, sing, r, 2*mass)

		return [dPdr, dmdr, dlnadr, dlnchdr]

	def ode_radius(self, r, y):
		lapse0, psi0 = y
		P = self.pressure_from_radius(r)
		mass = self.mass_from_radius(r)
		dlnadr = (mass + 4 * pi * r ** 3 * P) / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
		lamb = (r**0.5 - (r - 2.0 * nu.G * mass / nu.c ** 2)**0.5) / (r * (r - 2.0 * nu.G * mass / nu.c ** 2)**0.5)

		return [dlnadr, lamb]

	def tovsolve(self, density_central):

		N = 1e6
		r0, rf = [self.r0, self.rf]
		r = np.linspace(r0, rf, int(N))
		P_0 = self.physical_eos.pressure_from_density(density_central)
		rho_0 = self.physical_eos.density_from_pressure(P_0)
		mass_0 = 0. # 4.0 * pi * r[0] ** 3 * rho_0    # See arxiv:1212.1421
		lna_0 = 1  # it doesn't matter
		lnch_0 = 0

		# intmethod = "odeint"
		intmethod = "ivp"  # does not work for sys of diff.eqs ???

		if intmethod == "odeint":
			raise
			return "error"			
			# psol = odeint(self.tov, [P_0, mass_0, lna_0], r, t)  # , rtol=1.0e-4, atol=1.0e-4)
			# press = psol[:, 0]
			# mass = psol[:, 1]
			# lna_old = psol[:, 2]
		elif intmethod == "ivp":
			sol = solve_ivp(self.tov_ivp, [r0, rf], [P_0, mass_0, lna_0, lnch_0])
			r = sol.t
			rrange = 10.0 ** np.linspace(np.log10(r[0]), np.log10(r[-1]), int(N))
			sol = solve_ivp(self.tov_ivp, [r0, rf], [P_0, mass_0, lna_0, lnch_0], t_eval=rrange)
			r = sol.t
			press = sol.y[0]
			mass = sol.y[1]
			lna_old = sol.y[2]
			lnch = sol.y[3]
		else:
			raise
			return "error"

		return r, press, mass, lna_old, lnch

	def solve_with_atm(self, density_central):

		rad, press, mass, lna, lnch = self.tovsolve(density_central)

		# add atmosphere
		p_min = self.atm * (G - 1)
		atm_indx = np.squeeze(np.argwhere(press <= p_min))
		press[atm_indx] = p_min 
		
		print("p d atm", p_min, np.min(press) )
		# def radius and mass of star
		star_indx = atm_indx[0] 
		rad_star = rad[star_indx+1]
		mass_star = mass[star_indx+1]

		density = physical_eos.densities_from_pressure(press)
		rho = physical_eos.rho_from_density(density)
		energy = (K * density ** (G - 1)) / (G - 1)
		
		# a_fact = (1 - 2 * mass / rad) **-0.5  # IN SHPHERICAL COORDS		
		# chi = (1 - 2 * mass / rad)   # gamma_rr **-1    # IN SPHERICAL COORDS
		
		# passing to conformal radius
		chi = np.exp(lnch)
		a_fact = chi**-0.5
		rad_fix = rad.copy()
		rad_comoving = rad/a_fact   # rad in comoving coords
		
		if False:
			# Analitical method
			#Exterior
			r = rad_fix.copy()
			r_ext = 0.5*( np.sqrt(r**2 - 2*mass_star*r) + r - mass_star)
			a_ext = (1 + mass_star/(2*r_ext))
			
			#Interior
			m_of_r = interp1d(r, mass, bounds_error=False, fill_value='extrapolate')
			
			def r_integral(r):
				num = 1 - (1 - 2*m_of_r(r)/r)*0.5
				dem = r*(1 - 2*m_of_r(r)/r)*0.5
				return num/dem
			
			r_int = []
			r_space = np.linspace(1e-20, rad[star_indx+1], 100)
			for rint in r_space:
				C =  quad(r_integral, 0, rint)[0]
				# print(C)
				val =  r * np.exp( C )
				r_int.append(val)
			r_int = np.array(r_int)
			C =  (np.sqrt(rad_star**2 - 2*mass_star*rad_star) + rad_star - mass_star)/(2*rad_star) * np.exp(-C)
			r_int = C * r_int
			
			rr_of_r = interp1d(r, mass, bounds_error=False, fill_value='extrapolate')
			
			r = r_ext
			r[:star_indx+1] = rr_of_r(r[:star_indx+1])   # rad in comoving coords
			rad = r
		
			
		lapse = np.exp(lna)
		i_max = np.argmax(lapse)
		a_max = a_fact[-1]
		# lapse[i_max:-1] = lapse[i_max]
		lapse = lapse / lapse[i_max]
		lapse = lapse / a_max
		
		# print("amax, lmax", a_max, lapse[-1])
		# print(a_fact)
		# print(np.exp(lna))

		out = dict()
		out['pressure'] = press
		out['density'] = density
		out['energy'] = energy
		out['lapse'] = lapse # np.exp(lna)
		out['rho'] = rho
		out['chi'] = chi
		out['mass'] = mass
		out['rad'] = rad_comoving

		other = dict()
		other['rad_star'] = rad_star
		other['r_star'] = rad_comoving[star_indx]
		other['mass_star'] = mass_star

		return out, other


plt.rcParams.update({'font.size': 22})
do_plot = True

K = 1
G = 2
omega = G - 1
#omega = 1
#G = omega + 1
density_central = 0.42
atm = 1e-16

rest_mass = atm
K_in = -np.sqrt(24*np.pi*rest_mass)


physical_eos = eos(K, G)
t = tov(physical_eos, r0=1e-2, rf=1e2, atm=atm)  # integration limits r_ini, r_end

out, other = t.solve_with_atm(density_central)

rad_star = other['rad_star']
putL = int(np.log10(rad_star))+1

print("R star less than", 10**putL, "  fix/com r", other['rad_star'], other['r_star'])
print("M star is  ", other['mass_star'])




if do_plot:
	nplots = 7
	fig, axs = plt.subplots(nplots, 1, figsize=(20,15))

	keys = list(out.keys())
	keys.remove('rad')

	for i in range(nplots):
		var = keys[i]

		if var == 'chi':
			init = np.where( 0.01 <=  out['rad'])[0][0]
			end = np.where(out['rad'] >= 10**putL)[0][0]
			ys = out[var][init:end] **-0.5    # a = chi**-0.5
			axs[i].plot(out['rad'][init:end], ys, label='a', lw =4 )
			# axs[i].plot(out['rad'][init:end], out[var][init:end], label=var, lw =4 )

			
		else:

			init = np.where( 0.01 <=  out['rad'])[0][0]
			end = np.where(out['rad'] >= 10**putL)[0][0]

			axs[i].plot(out['rad'][init:end], out[var][init:end], label=var, lw =4 )
			# axs[i].plot(out['rad'], out[var], label=var)


	for ax in axs[:]:
		# ax.set_yscale('log')
		# ax.set_xscale('log')
		xmin = out['rad'][0] if  out['rad'][0] > 0.01 else 0.01
		# ax.set_xlim( xmin, 10**putL)
		ax.set_xlim( 0 , 10**putL)
		# ax.set_xlim( 0 , 4)
		# ax.set_xlim(xmin, 1.2)
		ax.axvline( other['r_star'] , color='r')
		ax.legend()

	plt.tight_layout()
	plt.savefig("idata.png")


radius = out["rad"]
f_density = interp1d(radius, out['density'], bounds_error=False, fill_value='extrapolate')
f_energy = interp1d(radius, out['energy'], bounds_error=False, fill_value='extrapolate')
f_pressure = interp1d(radius, out['pressure'], bounds_error=False, fill_value='extrapolate')
f_chi = interp1d(radius, out['chi'], bounds_error=False, fill_value='extrapolate')
f_lapse = interp1d(radius, out['lapse'], bounds_error=False, fill_value='extrapolate')

def _radius(x, y, z):
	"""
	x,y,z : These are the cell-centered conformal physical coordinates  ( grid-cords-centered * N_lev/ L )
			usually they are given as 3D arrays. size :(Dim, Nx_box, Ny_box, Nz_box)
	"""
	L = params['L']
	vec = np.array([x, y, z])
	rc = np.zeros_like(vec) + L / 2
	cvec = vec - rc

	rad = cvec[0, :] ** 2  + cvec[1, :] ** 2  + cvec[2, :] ** 2

	return np.sqrt(rad)


def _chi(x, y, z):
	r = _radius(x, y, z)
	out = f_chi(r)
	return out


def _K(x, y, z):
	#out =  - np.sqrt(24*np.pi* rho_mean - _lap(x, y, z))
	out =  np.zeros_like(x) - np.sqrt(24*np.pi* rho_mean) 
	#out =  - np.sqrt(24*np.pi*_D(x, y, z) )
	return out


def _D(x, y, z):
    out = _density(x, y, z) 
    return out


def _E(x, y, z):
    out = _density(x, y, z) * _energy(x, y, z)
    return out 
    

def _density(x,y,z):
	#out = _drho_th(x, y, z) + rho_mean
	r = _radius(x, y, z)
	out = f_density(r)
	return out

def _energy(x,y,z):
	r = _radius(x, y, z)
	out = f_energy(r)
	return out
	
def _pressure(x,y,z):
	r = _radius(x, y, z)
	out = f_pressure(r)
	return out

def _lapse(x,y,z):
	r = _radius(x, y, z)
	out = f_lapse(r)
	return out
	


components_vals = [
    ['chi', _chi],
    ['h11', 1], ['h22', 1], ['h33', 1],
    ['h12', 0], ['h13', 0], ['h23', 0],
    ['K', 0],
    ['A11', 0], ['A22', 0], ['A33', 0],
    ['A12', 0], ['A13', 0], ['A23', 0],
    ['Theta', 0],
    ['Gamma1', 0], ['Gamma2', 0], ['Gamma3', 0],
    ['lapse', _lapse],
    ['shift1', 0], ['shift2', 0], ['shift3', 0],
    ['B1', 0], ['B2', 0], ['B3', 0],
    ['density', _density], ['energy', _energy], ['pressure', _pressure], ['enthalpy', 0],
    ['D', _D], ['E', _E], ['W', 1],
    ['Z1', 0], ['Z2', 0], ['Z3', 0],
    ['V1', 0], ['V2', 0], ['V3', 0],
]
components_vals = np.array(components_vals)


