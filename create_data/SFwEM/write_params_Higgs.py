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
params['N'] = 120  #changed
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

    # "phi2", "Pi2",
    
    "Avec0",  "Avec1",  "Avec2",  "Avec3",
    
    "Evec1",  "Evec2",  "Evec3",
    
    "Zvec", 
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
boxes["level_0"] = make_boxes([0,0,0,bend,bend,bend],box_size=20)  #changed

# bini = 64 - 2*8
# bend = bini + 4*8 -1
# boxes["level_1"] = make_boxes([bini,bini,bini,bend,bend,bend],box_size=8)

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
Mp =  1 / np.sqrt(8.0*np.pi)
phi_ini =  2.0   * Mp

out_h = False  

# Higgs pars
mass_SI = (3.1302e-3)**4   # 9.6e-11
b = 0.13
a = (b/4/mass_SI)**0.5

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

def V_HI(phi, a=a, b=b, v=0):
    h, iters  = solve_h(phi)
    Mp = 1/np.sqrt(8*np.pi)
    F = 1.0 + a*h**2
    U =  0.25*b*(h**2 - v**2)**2

    return   U/F**2  *  Mp**4

def dV_HI(phi, a=a, b=b, v=0):
	h, iters = solve_h(phi)
	mp = 1.0
	Mp =  1 / np.sqrt(8.0*np.pi)
	F = 1.0 + a*h**2

	dVdphi =  Mp**3 * (b*h*(-a*v**4 + v*v*(a*h*h - 1) + h*h))/F**3  * F / np.sqrt(F +  6*a*a*h*h)
                   # * F**2 / (F + 6*a*a*h*h)
	return dVdphi

def phi_of_h(h, a = a):
        sf = np.sqrt((1+6*a)/(2*a)) * np.arcsinh(h*np.sqrt(a*(1+6*a)))  - np.sqrt(3)*np.arctanh(a*np.sqrt(6)*h/np.sqrt(1+a*(1+6*a)*h*h) )
        return sf*np.sqrt(2) / Mp

def solve_h(phi, h0 = 0, epsilon = 1e-8,  max_iter=300, a = a):
    # if out_h:
        # return phi, 0
    sf = phi/np.sqrt(2)   / Mp
    def x_of_h(h, a = a):
        sf = np.sqrt((1+6*a)/(2*a)) * np.arcsinh(h*np.sqrt(a*(1+6*a)))  - np.sqrt(3)*np.arctanh(a*np.sqrt(6)*h/np.sqrt(1+a*(1+6*a)*h*h) )
        return sf

    def dxdh(h, a = a):
        der = np.sqrt(1+ a*(1+6*a)*h**2)/(np.sqrt(2)*(1+a*h**2))
        return der

    def func(h):
        return x_of_h(h) - sf

    def d_func(h):
        return dxdh(h)

    xn = h0
    nit = 0
    for n in range(0,int(max_iter) ):
        nit +=1

        fxn = func(xn)
        if np.all(abs(fxn/sf) < epsilon):
            return xn, nit
        Dfxn = d_func(xn)
        if np.all(Dfxn == 0):
            print('Zero derivative. No solution found.')
            return None, nit
        xn = xn - fxn/Dfxn *0.3

    print('Exceeded maximum iterations. No solution found.')
    return xn, nit



def get_Pi_init(phi, V=V_HI ,dV = dV_HI):
    # assuming slowroll
    Mp =  1 / np.sqrt(8.0*np.pi)
    H = (V(phi)/3/Mp**2)**0.5
    return  -dV(phi)/(3*H)


Pi_ini =  get_Pi_init(phi_ini)

print("Pi, Pi/Mp2 , phi, phi/Mp :  ", Pi_ini, Pi_ini/Mp**2, phi_ini, phi_ini/Mp)



h , _ = solve_h(phi_ini)
dphi_dh = np.sqrt(1 + a*(1+6*a)*h*h)/(1+a*h*h)
hdot = Pi_ini / dphi_dh
print("h_dot, h ", hdot, h)


def F(h):
	return 0.5*(1.0 + a*h**2) #* Mp**2
def Ghh(h):
	F_here = F(h)
	return (2*F_here + 6*a*a*h*h)/(4.*F_here*F_here)
def Chris(h):
	C = 2*F(h) + 6*a*a*h*h
	return a*(1+6*a)*h/C - a*h/F(h)

def chi_of_s(h, s):
    return s/np.sqrt(F(h))


# rest initial values
h, _ = solve_h(phi_ini)
G_hh = Ghh(h)
# rho_ini =  0.5*Pi_ini**2 + V_HI(phi_ini)
rho_ini =  0.5*G_hh * hdot**2 + V_HI(phi_ini)
K_ini = - np.sqrt(24*np.pi* rho_ini)

print("kinetiks phi, h:", 0.5*Pi_ini**2 , 0.5*G_hh * hdot**2 )
print("G_hh = ", G_hh)
print(' rho, K, V inits ', rho_ini, K_ini, V_HI(phi_ini))

# parms for phi2
np.random.seed(124569)
modes = 5
phase = np.random.rand(modes)
phase2 = np.random.rand(modes)
phaseA = np.random.rand(modes)
phaseE = np.random.rand(modes)


def _fluc(x, y, z, phase):
    fluc = 0
    for ik in range(modes):
        kk = ik + 1
        LL  = (2*np.pi) / params['L'] * kk
        AmpP2 =  0.5*LL/kk**2
        fluc += AmpP2 * \
                        (np.sin(x *  (LL) + phase[ik]*2*np.pi ) + \
                         np.sin(y *  (LL) +  phase[ik]*2*np.pi ) + \
                         np.sin(z  *  (LL) + phase[ik]*2*np.pi  ))
    return fluc


def _phi2(x, y, z):
        
    fluc = _fluc(x,y,z, phase2)
	
    if out_h:
        phi2 = fluc * Mp
    else:
        phi2 = chi_of_s(h, fluc) * Mp
    return phi2


def _phi(x, y, z):
    phi = _fluc(x,y,z, phase)

    if out_h:
        phi = phi + phi_ini
        phi = solve_h(phi) * Mp
    else:
        phi = (phi_ini + phi) * Mp
    return phi


if out_h:
        Pi_ini = hdot


print('exporting phi : ',  phi_ini,  '   and Pi : ', Pi_ini)



components_vals = [
    ['chi', 1],
    ['h11', 1], ['h22', 1], ['h33', 1],
    ['h12', 0], ['h13', 0], ['h23', 0],
    ['K', K_ini],
    ['A11', 0], ['A22', 0], ['A33', 0],
    ['A12', 0], ['A13', 0], ['A23', 0],
    ['Theta', 0],
    ['Gamma1', 0], ['Gamma2', 0], ['Gamma3', 0],
    ['lapse', 1],
    ['shift1', 0], ['shift2', 0], ['shift3', 0],
    ['B1', 0], ['B2', 0], ['B3', 0],
    ['phi', _phi], ['Pi', Pi_ini],
    # ['phi2', _phi2], ['Pi2', 0],
    ['Avec0', 0], ['Avec1', 0], ['Avec2', 0], ['Avec3', 0],
    ['Evec1', 0], ['Evec2', 0], ['Evec3', 0],
    ['Zvec', 0]
]
components_vals = np.array(components_vals)
