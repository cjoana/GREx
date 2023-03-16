##### ######   ##
#		 #	   ##   Params and function 
#	  #  #	   ##  
##### ####     ##

import numpy as np

# dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
# h5_filename = './data/{exp}_test.hdf5'
dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
h5_filename = './h5_data/{exp}_test.hdf5'

prefx = "run1p_"  
exps = ["asym04", ]


############################################  Set vars 

global_vars = [
	['Ham', False],
	['Ham_abs_terms', False], 
	['Mom', False],
	['Mom_abs_terms', False],
	['K', 'all'],
	['N', 'all'],
	['ricci_scalar', 'all'],
	['trA2', 'all'],
	['rho', 'all'],
	['omega', 'all'],
	['W', 'all'],
	['lapse', 'all'],
	['delta_rho', 'all'],
	#
]

special_global_vars = [
	'time',  'Vol', 'dset',
]


############################################## functions

def dfn(path, num):
	return path.format(str(num).zfill(6))

units_override = {"length_unit": (1.0, "l_pl"),
				  "time_unit": (1.0, "t_pl"),
				  "mass_unit": (1.0, "m_pl")}
unit_system = 'planck'

#######################################  add derives cells

def _N(field, data):
	return np.log(data['chi'] ** -0.5) 

def _volcell(field, data):
	var = data["dx"] ** 3 * data["chi"] ** (-3 / 2)
	return var

def _omega(field, data):
	var = data["S"] / data["rho"] / 3
	return var

def _rPsi4(field, data):
    cntarg = np.argmin(data["lapse"].d)
    # cnt = [30, 30, 30]
    xx = data["x"].d -  data["x"][cntarg].d
    yy = data["y"].d -  data["x"][cntarg].d
    zz = data["z"].d -  data["x"][cntarg].d
    rad = np.sqrt(xx*xx +  yy*yy + zz*zz)
    return rad * data["Weyl4_Re"]

def _deltarho(field, data):
	rho = data["rho"].d
	mean = (data["K"].d)**2
	return rho/mean/24/np.pi

