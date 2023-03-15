import numpy as np

def make_boxes(cords, box_size=8):
	Cx = np.arange(cords[0], cords[3]+1, box_size)
	Cy = np.arange(cords[0], cords[3]+1, box_size)
	Cz = np.arange(cords[0], cords[3]+1, box_size)
	itv = box_size -1
	boxes = []
	for x in Cx:
		for y in Cy:
			for z in Cz: 
				box = [x,y,z,x+itv,y+itv,z+itv]
				boxes.append(box)
	return boxes

params = dict()
all_attrb = dict()
base_attrb = dict()
chombogloba_attrb = dict()
levels_attrb = dict()
boxes = dict()
data_attributes = dict()

Nc = 512

# basic params  (MANUAL)
params['N'] = Nc
params['L'] = 60
params['dt_multiplier'] = 0.1
params['is_periodic'] = [1, 1, 1]
params['ghosts'] = [0, 0, 0]         # No ghosts possible yet

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

# set base attibutes (MANUAL)
base_attrb['time'] = 0.0 # float!
base_attrb['iteration'] = 0
base_attrb['max_level'] = 4
base_attrb['num_levels'] = 1  # min = 1
base_attrb['num_components'] = components.size
base_attrb['regrid_interval_0'] = 2
base_attrb['steps_since_regrid_0'] = 0
for comp, name in enumerate(components):
    key = 'component_' + str(comp)
    tt = 'S' + str(len(name))
    base_attrb[key] = np.array(name, dtype=tt)

# Set boxes, for each level (MANUAL)
bend = params['N']-1
boxes["level_0"] = make_boxes([0,0,0,bend,bend,bend],box_size=12)

bini = Nc - 3*12
bend = bini + 6*12 -1
#boxes["level_1"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=12)

bini = 128 - 2*8
bend = bini +4*8-1
#boxes["level_2"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)

bini = 256 - 2*8
bend = bini +  4*8-1
#boxes["level_3"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)



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
####      DATA TEMPLATE        ####
###################################

dic_curv = dict()
HR = 1.
rho_mean = HR**-2 *3./(8*np.pi)
dic_curv['A_gauss'] =   1
dic_curv['S_gauss'] =   5  #  20./np.sqrt(2)   # r_BH = 2
omega = 1./3
rest_mass = 1e-18


def _curv(x, y, z, set_rc=False, rc = False, A=False, B=False):
	"""
	x,y,z : These are the cell-centered conformal physical coordinates  ( grid-cords-centered * N_lev/ L )
			usually they are given as 3D arrays. size :(Dim, Nx_box, Ny_box, Nz_box)
	"""
	L = params['L']
	vec = np.array([x, y, z])
	if not set_rc :    rc = np.zeros_like(vec) + L/2

	cvec = vec - rc

	if not A: A_gauss = dic_curv['A_gauss']
	if not B: B = dic_curv['S_gauss']
	S_gauss_x = dic_curv['S_gauss']
	S_gauss_y = dic_curv['S_gauss'] # + 0.1
	S_gauss_z = dic_curv['S_gauss'] # - 0.1

	# dot_prod = cvec[0, :] ** 2 / S_gauss_x ** 2 + cvec[1, :] ** 2 / S_gauss_y ** 2 + cvec[2, :] ** 2 / S_gauss_z ** 2
	r2 = cvec[0, :] ** 2 + cvec[1, :] ** 2  + cvec[2, :] ** 2 

	return A_gauss * np.exp(- 0.5 * r2/B**2)
	#return A_gauss * np.exp(- 0.5 * dot_prod)


def _drho_th(x, y, z):

	out = 1.0 * _ricci_scalar(x, y, z)/ (16 * np.pi)
	return out
	

def get_center(x,y,z, cnt):
	crd = np.zeros_like(x)
	center = np.array([crd+cnt[0], crd+cnt[1], crd+cnt[2]])
	
	return center
	


def _psi(x, y, z):
	
	c1 = np.array([29.5,30,22])
	c2 = np.array([26,30,31])
	c3 = np.array([30,32,31])
	
	cn1 = get_center(x,y,z, c1)
	cn2 = get_center(x,y,z, c2)
	cn3 = get_center(x,y,z, c3)

	f=1.639
	a1 = 0.31 *f
	a2 = 0.33  *f
	a3 = 0.35 *f


	out = np.exp( 0.5 * (
                 a1 *  _curv(x, y, z, set_rc=True, rc=cn1)  \
			   + a2 *  _curv(x, y, z, set_rc=True, rc=cn2)    \
			   + a3 *  _curv(x, y, z, set_rc=True, rc=cn3)    \
						 ) )
	return out

def _chi(x, y, z):
    out = _psi(x, y, z)**-4    # Here psi as in Baumgarte book p. 56
    return out

def _ricci_scalar(x, y, z):
	
	A = dic_curv['A_gauss']
	B = dic_curv['S_gauss']	
	G = _curv(x, y, z) / A
	L = params['L']
	vec = np.array([x, y, z])
	rc = np.zeros_like(vec) + L / 2
	cvec = vec - rc
	r2 = cvec[0, :] ** 2 + cvec[1, :] ** 2  + cvec[2, :] ** 2 
	
	
	Omega = (A * np.exp(0.5*A*G - r2/(2*B**2)))/(2*B**2)		
	def dderv(cord):
		cc2 = cord**2 
		return (A * cc2* G/(2*B**2) + cc2/B**2) * Omega - Omega
	
	coord = cvec[0]
	ddx = dderv(coord)
	coord = cvec[1]
	ddy = dderv(coord)
	coord = cvec[2]
	ddz = dderv(coord)	
	
	out =  - 8 * _psi(x, y, z)**-5 * ( ddx + ddy + ddz)
	return out

def _rho_ADM(x, y, z):
	out =  _D(x, y, z) +  _E(x, y, z)
	#out =  - np.sqrt(24*np.pi*_D(x, y, z) )
	return out

def _K(x, y, z):
	#out =  - np.sqrt(24*np.pi* rho_mean - _ricci_scalar(x, y, z))
	out =  np.zeros_like(x) - np.sqrt(24*np.pi* rho_mean) 
	#out =  - np.sqrt(24*np.pi*_D(x, y, z) )
	return out


def _D(x, y, z):
    #out = _drho_th(x, y, z) + rho_mean
    out = np.zeros_like(x) + rest_mass
    return out


def _E(x, y, z):
    #out = np.zeros_like(x)
    out = _drho_th(x, y, z) + rho_mean
    return out + 0
    
def _density(x,y,z):
	#out = _drho_th(x, y, z) + rho_mean
	out = np.zeros_like(x) + rest_mass
	return out

def _energy(x,y,z):
	out = _E(x, y, z) / rest_mass
	#out = np.zeros_like(x) + rest_mass
	return out
	
def _pressure(x,y,z):
	out = _E(x, y, z) * omega
	return out



components_vals = [
    ['chi', _chi],
    ['h11', 1], ['h22', 1], ['h33', 1],
    ['h12', 0], ['h13', 0], ['h23', 0],
    ['K', _K],
    ['A11', 0], ['A22', 0], ['A33', 0],
    ['A12', 0], ['A13', 0], ['A23', 0],
    ['Theta', 0],
    ['Gamma1', 0], ['Gamma2', 0], ['Gamma3', 0],
    ['lapse', 1],
    ['shift1', 0], ['shift2', 0], ['shift3', 0],
    ['B1', 0], ['B2', 0], ['B3', 0],
    ['density', rest_mass], ['energy', _energy], ['pressure', _pressure], ['enthalpy', _ricci_scalar],
    ['D', rest_mass], ['E', _E], ['W', 1],
    ['Z1', 0], ['Z2', 0], ['Z3', 0],
    ['V1', 0], ['V2', 0], ['V3', 0],
]
components_vals = np.array(components_vals)

