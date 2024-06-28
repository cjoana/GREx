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

# basic params  (MANUAL)
params['N'] = 64 #changed
params['L'] = 1e6
params['dt_multiplier'] = 0.01
params['is_periodic'] = [1, 1, 1]
params['ghosts'] = [0, 0, 0]         # No ghosts possible yet

# Set components  (MANUAL)
components = np.array([
    "chi",

    "h11",    "h12",    "h13",    "h22", "h23", "h33",

    "K",

    "A11",    "A12",    "A13",    "A22", "A23", "A33",

    "Theta",

    "Gamma1", "Gamma2", "Gamma3",

    "lapse",

    "shift1", "shift2", "shift3",

    "B1",     "B2",     "B3",

    "phi", "Pi",
  
    "Avec0",  "Avec1",  "Avec2",  "Avec3",
    
    "Evec1",  "Evec2",  "Evec3",
    
    "Zvec", 
])

# set base attibutes (MANUAL)
base_attrb['time'] = 0.0 # float!
base_attrb['iteration'] = 0
base_attrb['max_level'] = 4
base_attrb['num_levels'] = 2  # min = 1
base_attrb['num_components'] = components.size
base_attrb['regrid_interval_0'] = 2
base_attrb['steps_since_regrid_0'] = 0
for comp, name in enumerate(components):
    key = 'component_' + str(comp)
    tt = 'S' + str(len(name))
    base_attrb[key] = np.array(name, dtype=tt)


# Set boxes, for each level (MANUAL)
bend = params['N']-1
boxes["level_0"] = make_boxes([0,0,0,bend,bend,bend],box_size=32)  #changed

bini = 64 - 2*8
bend = bini + 4*8 -1
boxes["level_1"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)

# bini = 128 - 2*8
# bend = bini +4*8-1
# boxes["level_2"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)

# bini = 256 - 2*8
# bend = bini +  4*8-1
# boxes["level_3"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)



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
mp = 1.0
Mp =  1 / np.sqrt(8.0*np.pi)
adf = 1 / np.sqrt(8.0*np.pi)
phi_ini =  0.5   #   in mp 

# Potetnial pars      V(x) = 0.5 a^2 x^2  + 0.25 b^4 x^4 
a = 1e-6   # mass     
b = 1e-3   # lambda

print("a, b =", a, b)

# def SI(phi):
	# mp = 1.0
	# Mp =  1 / np.sqrt(8.0*np.pi)
	# mass = (3.1e-3  * Mp )**4
	# sqf = (np.sqrt(2.0/3.0)/Mp)
	# V = mass *  (1.0 - np.exp(- sqf * (phi)) )**2.0
	# return V

# def dSI(phi):
	# mp = 1.0
	# Mp =  1 / np.sqrt(8.0*np.pi)
	# mass = (3.1e-3  * Mp )**4 # / mp**4
	# sqf = (np.sqrt(2.0/3.0)/Mp)
	# dV = 2.0 * mass * sqf * \
		# np.exp(- 2 *  sqf * phi ) * \
		# (np.exp( sqf * phi) - 1)
	# return dV

def V(phi, a=a, b=b):
    return  0.5* (a * phi)**2 + 0.25 * (b * phi) **4 

def dV_dphi(phi, a=a, b=b, v=0):
	dVdphi =  a**2 * phi +  b**4 * phi**3
	return dVdphi

def get_Pi_init(phi, V=V ,dV = dV_dphi):
    # assuming slowroll
    eta =  - dV(phi) * np.sqrt(3/V(phi) ) * Mp
    return  eta

def epsilon(phi, V=V, dV = dV_dphi):
      return 0.5* Mp**2 * (dV(phi)/V(phi))**2


Pi_ini =  get_Pi_init(phi_ini)

# print("Pi, Pi/Mp2 , phi, phi/Mp :  ", Pi_ini, Pi_ini/Mp**2, phi_ini, phi_ini/Mp)


# p_phis = np.linspace(0, 2, 1000)  # * adf
# import matplotlib.pyplot as plt 
# plt.plot(p_phis, epsilon(p_phis))
# plt.yscale('log')
# plt.show()



# rest initial values
rho_ini =  0.5*Pi_ini**2 + V(phi_ini)
K_ini = - np.sqrt(24*np.pi* rho_ini)

print(' rho, K, V inits ', rho_ini, K_ini, V(phi_ini))

# parms for phi2
np.random.seed(124569)
modes = 8
phase = np.random.rand(modes)
phase2 = np.random.rand(modes)
phaseA = np.random.rand(modes,3)
phaseE = np.random.rand(modes,3)


def _fluc(x, y, z, phase):
    fluc = 0
    for ik in range(modes):
        kk = ik + 1
        LL  = (2*np.pi) / params['L'] * kk    # Assuming L is like R_H, it needs to be L because of BC. 
        RH = np.abs(3/K_ini)
        AmpP2 =  0.5*LL/kk**2
        # AmpP2 =  0.5*RH/kk   #  RH ~ L 
        fluc += AmpP2 * \
                        (np.sin(x *  (LL) + phase[ik]*2*np.pi ) + \
                         np.sin(y *  (LL) +  phase[ik]*2*np.pi ) + \
                         np.sin(z  *  (LL) + phase[ik]*2*np.pi  ))
    return fluc


def _phi2(x, y, z):
        
    fluc = _fluc(x,y,z, phase2)
    phi2 = fluc * Mp
    return phi2


def _phi(x, y, z):
    fluc = _fluc(x,y,z, phase)
    phi = fluc + phi_ini

    return phi


print('exporting phi : ',  phi_ini,  '   and Pi : ', Pi_ini)


def _A1(x, y, z):
    f = _fluc(x,y,z, phaseA[:,0])
    return f
def _A2(x, y, z):
    f = _fluc(x,y,z, phaseA[:,1])
    return f
def _A3(x, y, z):
    f = _fluc(x,y,z, phaseA[:,2])
    return f
def _E1(x, y, z, phase=phaseE[:,0]):
    f = _fluc(x,y,z, phase)
    return f
def _E2(x, y, z, phase=phaseE[:,1]):
    f = _fluc(x,y,z, phase)
    return f
def _E3(x, y, z, phase=phaseE[:,2]):
    f = _fluc(x,y,z, phase)
    return f



components_vals = [
    ['chi', 1],
    ['h11', 1], ['h22', 1], ['h33', 1],
    ['h12', 0], ['h13', 0], ['h23', 0],
    ['K', K_ini],
    ['A11', 0], ['A22', 0], ['A33', 0],
    ['A12', 0], ['A13', 0], ['A23', 0],
    ['Theta',0],
    ['Gamma1', 0], ['Gamma2', 0], ['Gamma3', 0],
    ['lapse', 1],
    ['shift1', 0], ['shift2', 0], ['shift3', 0],
    ['B1', 0], ['B2', 0], ['B3', 0],
    ['phi', _phi], ['Pi', Pi_ini],
    ['Avec0', 0], ['Avec1', _A1], ['Avec2', _A2], ['Avec3',_A3],
    ['Evec1', _E1], ['Evec2', _E2], ['Evec3', _E3],
    ['Zvec', 0]
]
components_vals = np.array(components_vals)
