import yt

yt.funcs.mylog.setLevel(50)  # or 1: full-log  and 50: no-log
from yt import YTQuantity

print(yt.__version__)

import h5py
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

units_override = {"length_unit": (1.0, "l_pl"),
				  "time_unit": (1.0, "t_pl"),
				  "mass_unit": (1.0, "m_pl")}
unit_system = 'planck'


def dfn(path, num):
	return path.format(str(num).zfill(6))

def _N(field, data):
	return np.log(data['chi'] ** -0.5) 

def _volcell(field, data):
	var = data["dx"] ** 3 * data["chi"] ** (-3 / 2)
	return var
	
def _omega(field, data):
	var = data["S"] / data["rho"] / 3
	return var

def _rPsi4(field, data):
        cnt = [30, 30, 30]
        xx = data["x"].d - cnt[0] 
        yy = data["y"].d - cnt[1]
        zz = data["z"].d - cnt[2]
        rad = np.sqrt(xx*xx +  yy*yy + zz*zz)
        return rad * data["Weyl4_Re"]

""" ############################################################# ignore /!

def _kin(field, data):
	var = 0.5 * (data["Pi"] ** 2) / data["V"]
	return var


def _T_FRW(field, data):
	var = 2* (data["Pi"] ** 2) - 2*data["V"]
	return var

def _Tdiff(field, data):
	var =  (data["rho"] + data["S"]) - data["T_FRW"]
	return var


def _Tdev(field, data):
	var = (data["rho"] + data["S"]) / data["T_FRW"]
	return var


def _A2rho(field, data):
	var = data["trA2"] / data["rho"]
	return var
	

def _adotdot(field, data):
	var = -1. * (data["trA2"] +  4*np.pi* (data["rho"] + data["S"]))/3
	return var


def _grad(field, data):
	var = (data["rho"] - data["V"] - data["kin"] * data["V"]) / data["V"]
	return var


def _V_SI(field, data):
	# Mp = 1.0 * YTQuantity(1, 'm_pl') / np.sqrt(8.0 * np.pi)
	# Mp = data.ds.arr(1.0, "m_pl")/ np.sqrt(8.0 * np.pi)
	Mp = 1.0 / np.sqrt(8.0 * np.pi)
	mass = 9.6e-11
	V = mass * (Mp ** 4.0) * (1.0 - np.exp(-np.sqrt(2.0 / 3.0) * (data["phi"]) / Mp)) ** 2.0
	return V
	

def _HamRelCorr(field, data):
	var = data["HamRel"] 
	o = data["S"] / data["rho"] / 3
	var[o<-0.33] = var[o <-0.33] * 0.1
	return var


def _av_phi(field, data):
	var = data["phi"] - np.mean(np.ndarray.flatten(data['phi']))
	return np.array(var)


def _logK(field, data):
	K = np.array(data['K'])
	K[(K > 1)] = 1
	K[(K < 1e-10) & (K > 0)] = 1e-10
	K[(K < -1)] = -1
	K[(K > -1e-10) & (K < 0)] = -1e-10
	return np.log10(np.abs(K))


def _logHamRel(field, data):	
	var = np.array(data['HamRel'])
	# o = data["S"] / data["rho"] / 3	
	#var[o <-0.33] = var[o <-0.33] * 0.1
	var = np.abs(var)
	var[(var > 1)] = 1
	var[(var == 0)] = 1e-10
	var[(var < 1e-10)] = 1e-10
	return np.log10(np.abs(var))


def V_SI(phi):
	Mp = 1.0 / np.sqrt(8.0 * np.pi)
	mass = 9.6e-11
	V = mass * (Mp ** 4.0) * (1.0 - np.exp(-np.sqrt(2.0 / 3.0) * (phi) / Mp)) ** 2.0
	return V


def hist2D(ax, dd, var, kargs={}):
	kwords = kargs.keys()

	xx = dd['global_values']['time'][:]
	yy = dd['dist_bins'][var][:-1]
	X, Y = np.meshgrid(xx, yy)
	Z = np.array(dd['dist_values'][var][:-1]).T

	clog = False if not 'clog' in kwords else kargs['clog']
	num_xticks = 10 if not 'num_xticks' in kwords else kargs['num_xticks']
	colmap = cm.inferno if not 'colmap' in kwords else kargs['colmap']
	zmax = Z.max() / 10 if not 'zmax' in kwords else kargs['zmax']
	zmin = Z.min() if not 'zmin' in kwords else kargs['zmin']

	if clog:
		zf = Z.flatten()
		zfm = np.min(zf[zf >= 0]) * 0.1
		zfm = zfm if zfm > 0 else 0.1
		Z = np.log10(Z + zfm)
		zmin = np.min(Z)
		zmax = np.mean(Z)

	im = ax.pcolormesh(X, Y, Z.reshape(X.shape), cmap=colmap,
					   vmin=zmin, vmax=zmax)

	return im


def global_plot(ax, gvar):
	ax.plot(gvar[0], 'k', label="mean")
	ax.plot(gvar[0] + gvar[1], 'k--', label="mean + std")
	ax.plot(gvar[2], 'r', label="min")
	ax.plot(gvar[3], 'b', label="max")
	ax.legend()


def hist2D(ax, dd, var, kargs={}):
    kwords = kargs.keys()

    xvar = 'time' if not "xaxis" in kwords else kargs['xaxis'] 
    
    xx = dd['global_values'][xvar][:] if not xvar == 'dsets' else dd['dsets']
    yy = dd['dist_bins'][var][:-1]
    X, Y = np.meshgrid(xx, yy)
    Z = np.array(dd['dist_values'][var][:-1]).T

    clog = False if not 'clog' in kwords else kargs['clog']
    num_xticks = 10 if not 'num_xticks' in kwords else kargs['num_xticks']
    colmap = cm.gist_heat_r if not 'colmap' in kwords else kargs['colmap']
    zmax = Z.max() / 10 if not 'zmax' in kwords else kargs['zmax']
    zmin = Z.min() if not 'zmin' in kwords else kargs['zmin']

    if clog:
        zf = Z.flatten()
        zfm = np.min(zf[zf >= 0]) * 0.1
        zfm = zfm if zfm > 0 else 0.1
        Z = np.log10(Z + zfm)
        zmin = np.min(Z)
        zmax = np.mean(Z)
        
    Z[Z==0] = np.nan    
    im = ax.pcolormesh(X, Y, Z.reshape(X.shape), cmap=colmap,
                       vmin=zmin, vmax=zmax)
    
    def running_mean(x, N):
        return np.convolve(x, np.ones((N,))/N, mode='same')


    vl = list(dd['global_values'].keys() )
    if var in vl:
    # if True:
        vals = dd['global_values'][var][0]
        # vals = np.sin(xx*2*np.pi*10/xx.max())
        #N = len(xx)//5
        #vals = running_mean(vals[0], N)
        #ax.plot(xx[:], vals, "b")
        #vals[:N//2] = vals[N//2]
        #vals[-N//2:] = vals[-N//2]        
        ax.plot(xx[:], vals[:], "b")
    

    return im

""" ##################################################################### !/

test = False

# Load data
exps = [
	"asym01",
	"asym02",
	"asym03",
	"asym04",
	"pancake",
	# "pancake02",
	"Yoo81",
]

dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
h5_filename = './data/{exp}_BasicAnalysis.hdf5'
recompute = False 
refill = True
do_plot = False
do_extraction = True
reduce_step = False

pars = dict()

pars['fint']  = 100
pars['fini'] = pars['fint'] 
pars['fend']  = 1000000  # loop breaks when the dset does not exist.

pars['iteration'] = 100
pars['num_points'] = 400

if test: recompute = True
if test: runs = ['test', ]

# exps = ['PBH_AHFinder',] 
prefx = "run0p_"  


dist_vars = [  # var, bins, weitghen
	#['omega', np.linspace(-1, 1.1, 101, True), True],
	#['logK', np.linspace(-10, 1, 22, True), True],
	# ['logHamRel', np.linspace(-10, 1, 22, True), True],
	#['ricci_scalar_tilde', np.array([-1e-8, -1e-9, -1e-10, -1e-11, -1e-12, -1e-13,  1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]), True],
	#['ricci_scalar', np.array([-1e-8, -1e-9, -1e-10, -1e-11, -1e-12, -1e-13,  1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8]), True],
]
special_dist_vars = [
	#['logK+', np.linspace(-10, 1, 50, True), True],
	#['logK-', np.linspace(-10, 1, 50, True), True],
]
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
	#['omega', 'all'],
	#['pressure', 'all'],
	['W', 'all'],
	['lapse', 'all'],

	# ['trA2_ER', 'ER'],
	# ['A2rho_ER', 'ER'],
	# ['K_ER', 'ER'],
	# ['rho_ER', 'ER'],
	# ['HamRel_ER', 'ER'],
	# ['ricci_scalar_ER', 'ER'],	
	#['Ham', True],
	#['Mom1',True],
	
]
special_global_vars = [
	'L', 'time',  'Vol',  #'Vol_CR', 'Vol_acc', 'Vol_KD', 'Vol_VD', 
]

plot_global_vars = [
	# 'HamRel',
	# 'phi',
	# 'K',
	# 'N',
	# 'ricci_scalar',
	# 'trA2',
	# 'rho',
	# 'av_phi',
]





 # num of x_points in hist/dist_values plots
if test: num_points=6
# files = np.concatenate([[[i, dfn(pfiles, i)] for i in range(0, 500, 20)],
#                         [[i, dfn(pfiles, i)] for i in range(500, 6000, 500)]])

print("Starting loops")

for exp in exps:
#	for run in runs:
		
		fini = pars['fint']
		fend = pars['fend'] # loop breaks when the dset does not exist.
		fint = pars['fint']
		fini = fint
		iteration = pars['iteration']
		num_points = pars['num_points']

		print('Executing ', exp) 
		
		h5_fn = h5_filename.format(exp=exp)
		print('Loading of ', h5_fn)
		
		if not os.path.exists(h5_fn) or recompute:
			out = h5py.File(h5_fn, "w")

			# dist_values = out.create_group("dist_values")
			# dist_bins = out.create_group("dist_bins")
			# for var, bins, weighted in dist_vars:
			# 	dist_values.create_dataset(var, data=np.array([np.zeros_like(bins[:-1])]), maxshape=(None, None,))
			# 	dist_bins.create_dataset(var, data=bins)
			# for var, bins, weighted in special_dist_vars:
			# 	dist_values.create_dataset(var, data=np.array([np.zeros_like(bins[:-1])]), maxshape=(None, None,))
			# 	dist_bins.create_dataset(var, data=bins)
			global_values = out.create_group("global_values")
			for var, weighted in global_vars:
				global_values.create_dataset(var, data=[[], [], [], []], maxshape=(None, None,))
			for var in special_global_vars:
				global_values.create_dataset(var, data=[], maxshape=(None,))
			prev_dsets = []
			dsets = out.create_dataset('dsets', data=[], dtype=int, maxshape=(None,))
		elif refill:
			try:
				out = h5py.File(h5_fn, "r+")
				# dist_values = out['dist_values']
				# dist_bins = out["dist_bins"]
				global_values = out["global_values"]
				prev_dsets = list(np.array(out['dsets'][:], dtype=str))				
				dsets = out["dsets"]
				fini = np.max(dsets)
				num_points = (1000 - len(dsets))//4
				if len(dsets) >= 900 : num_points = 100
				if num_points <= 0: num_points = pars['num_points']
			except Exception as e:
				os.remove(h5_fn)
				continue

		else:
			if not (do_plot or do_extraction):
				continue	
		
		
		if do_extraction:
		
			# data files
			data_path = dsets_path.format(exp=exp)
			pfiles = data_path + prefx + '_{0}.3d.hdf5'

			# if reduce_step and os.path.exists(dfn(pfiles, 20000)):
			#	fint = 100
			# elif reduce_step and os.path.exists(dfn(pfiles, 10000)):
			#	fint = 50
			# else: fint = 10

			files = np.array([[i, dfn(pfiles, i)] for i in range(fini, fend, fint)])
			mask_files = np.array([1 if os.path.exists(f) else 0 for i, f in files], dtype=bool)
			files = files[mask_files]
			files = files[:-1] # Exclude last dataset for precaution

			if len(files) == 0:
				print(' --> exp skipped: files don`t exist')
				print(data_path)
				continue

			# Reduce number of data points to `num_points`
			if reduce_step:
				print('checking times' )
				new_files = []
				times = np.array([h5py.File(f, "r").attrs["time"] for i, f in files], dtype=float)
				sel_times = np.linspace(times[0], times[-1], num_points, True)
				print(times)
				print(sel_times)
				for t in sel_times:
					new_files.append(files[times>=t][0])
				files = np.array(new_files)

			first_file = files[0,1]	

			if not os.path.exists(first_file):
				print(' --> exp skipped: first file doesn`t exist', first_file)
				continue
			

			dat = {'dist_values': dict(), 'global_values': dict(), 'dsets': []}
			for var, bins, weighted in dist_vars:
				dat['dist_values'][var] = []
			for var, bins, weighted in special_dist_vars:
				dat['dist_values'][var] = []
			for var, weighted in global_vars:
				dat['global_values'][var] = []
			for var in special_global_vars:
				dat['global_values'][var] = []
			itr = 0

			for i_ds, dset in files:

				print("Loading ", i_ds, dset)

				if (str(i_ds) in prev_dsets) or (str(i_ds) in dat['dsets']):
					print(" -->skip")
					continue
				if not os.path.exists(dset):
					print("  break!! dset does not exist", dset)
					break
				try:
					ds = yt.frontends.chombo.ChomboDataset(dset, unit_system=unit_system, units_override=units_override)
					# ds.add_field(('chombo', 'omega'), function=_omega, units="", take_log=False, display_name='omega')
					ds.add_field(('chombo', 'volcell'), function=_volcell, units="l_pl**3", take_log=False,
								 display_name='volcell')
					# ds.add_field(('chombo', 'logK'), function=_logK, units="", take_log=False, display_name='logK')
					ds.add_field(('chombo', 'N'), function=_N, units="", take_log=False, display_name='N')
					# ds.add_field(('chombo', 'A2rho'), function=_A2rho, units="", take_log=False, display_name='A2rho')
					# ds.add_field("rPsi4", _rPsi4, units="")
				# TODO: Add all needed fields

				except Exception as e:
					print("  break!! error in adding fields in ds", dset)
					print(e)
					break

				# res = 100j
				# fd = np.ndarray.flatten( ds.r[::res, ::res, ::res]['omega']   )
				reg = ds.r[:, :, :]
				vol_cell = np.ndarray.flatten(reg['volcell'])
				vol = np.sum(vol_cell)*1.0
				weights = vol_cell / vol
				#mask_kpos = np.ndarray.flatten(reg['K']) > 0
				# vol_CR = np.sum(vol_cell[mask_kpos])
				# mask_accpos = np.ndarray.flatten(reg['adotdot']) > 0
				# vol_acc = np.sum(vol_cell[mask_accpos])
				# mask_KD = np.ndarray.flatten(reg['omega']) > 0.66
				# vol_KD = np.sum(vol_cell[mask_KD])
				# mask_VD = np.ndarray.flatten(reg['omega']) < -0.66
				# vol_VD = np.sum(vol_cell[mask_VD])
				# mask_ER = np.array( np.sign(np.ndarray.flatten(reg['K'])) < 0, dtype=bool)
				# weights_ER = reg['volcell'][mask_ER]

				# Special global/dist vars
				dat['global_values']['L'].append(vol ** (1 / 3))
				dat['global_values']['time'].append(ds.current_time)
				dat['global_values']['Vol'].append(vol)
				# dat['global_values']['Vol_CR'].append(vol_CR)
				# dat['global_values']['Vol_acc'].append(vol_acc)
				# dat['global_values']['Vol_KD'].append(vol_KD)
				# dat['global_values']['Vol_VD'].append(vol_VD)	
				

				# logK+ logK-
				bins = out["dist_bins"]["logK+"][:]
				vals = np.ndarray.flatten(reg['logK'])
				mask_pos = np.array( np.sign(np.ndarray.flatten(reg['K'])) > 0, dtype=bool)
				prevals = vals.copy()
				prevals[mask_pos] = +10
				h, b = np.histogram(prevals, bins, weights=weights)
				dat['dist_values']['logK-'].append(h)			
				prevals = vals.copy()
				prevals[~mask_pos] = +10
				h, b = np.histogram(prevals, bins, weights=weights)
				dat['dist_values']['logK+'].append(h)	


				# dist vars
				for var, bins, weighted in dist_vars:
					# Compute histogram
					fd = np.ndarray.flatten(reg[var])
					w = weights if weighted else None
					h, b = np.histogram(fd, bins, weights=w)
					# Append values to existing dataset
					dat['dist_values'][var].append(h)

				# global vars
				for var, weighted in global_vars:
					_var = var
					if not weighted:
						w=np.ones_like(weights)
						fd = np.ndarray.flatten(reg[_var])
					# elif weighted == 'ER':
					# 	w = weights_ER 
					# 	_var=var[:-3]
					# 	fd = np.ndarray.flatten(reg[_var])
					# 	fd = fd[mask_ER]
					elif weighted == 'all':
						w = weights 
						fd = np.ndarray.flatten(reg[_var])				
					std = np.sqrt(np.cov(fd, aweights=w))
					vals = [np.average(fd, weights=w), std, np.min(fd), np.max(fd)]
					# Append values to existing dataset
					dat['global_values'][var].append(vals)
				dat['dsets'].append(i_ds)
				prev_dsets.append(i_ds)

				if itr == iteration - 1 or i_ds == files[-1][0]:
					for var, weighted in global_vars:
						gv = global_values[var]
						m, n = gv.shape
						vals = np.array(dat['global_values'][var]).T
						mm, nn = np.shape(vals)
						global_values[var].resize((m, n + nn))
						global_values[var][:, -nn:] = vals

					# for var, bins, weighted in dist_vars:
					# 	m, n = dist_values[var].shape
					# 	vals = np.array(dat['dist_values'][var])
					# 	mm, nn = np.shape(vals)
					# 	dist_values[var].resize((m + mm, n))
					# 	dist_values[var][-mm:] = vals

					# for var, bins, weighted in special_dist_vars:
					# 	m, n = dist_values[var].shape
					# 	vals = np.array(dat['dist_values'][var])
					# 	mm, nn = np.shape(vals)
					# 	dist_values[var].resize((m + mm, n))
					# 	dist_values[var][-mm:] = vals

					# Special vars
					vals = dat['global_values']['L']
					ln = len(vals)
					global_values['L'].resize((dsets.shape[0] + ln), axis=0)
					global_values['L'][-ln:] = np.array(vals, dtype=int)
					vals = dat['global_values']['time']
					ln = len(vals)
					global_values['time'].resize((dsets.shape[0] + ln), axis=0)
					global_values['time'][-ln:] = np.array(vals)
					vals = dat['global_values']['Vol']
					ln = len(vals)
					# global_values['Vol'].resize((dsets.shape[0] + ln), axis=0)
					# global_values['Vol'][-ln:] = np.array(vals)
					# vals = dat['global_values']['Vol_CR']
					# ln = len(vals)
					# global_values['Vol_CR'].resize((dsets.shape[0] + ln), axis=0)
					# global_values['Vol_CR'][-ln:] = np.array(vals)
					# vals = dat['global_values']['Vol_acc']
					# ln = len(vals)
					# global_values['Vol_acc'].resize((dsets.shape[0] + ln), axis=0)
					# global_values['Vol_acc'][-ln:] = np.array(vals)
					# vals = dat['global_values']['Vol_VD']
					# ln = len(vals)
					# global_values['Vol_VD'].resize((dsets.shape[0] + ln), axis=0)
					# global_values['Vol_VD'][-ln:] = np.array(vals)
					# vals = dat['global_values']['Vol_KD']
					# ln = len(vals)
					# global_values['Vol_KD'].resize((dsets.shape[0] + ln), axis=0)
					# global_values['Vol_KD'][-ln:] = np.array(vals)

					# Append last cycle into dataset
					vals = dat['dsets']
					ln = len(vals)
					dsets.resize((dsets.shape[0] + ln), axis=0)
					dsets[-ln:] = np.array(vals, dtype=int)

					# Reset
					dat = {'dist_values': dict(), 'global_values': dict(), 'dsets': []}
					for var, bins, weighted in dist_vars:
						dat['dist_values'][var] = []
					for var, bins, weighted in special_dist_vars:
						dat['dist_values'][var] = []
					for var, weighted in global_vars:
						dat['global_values'][var] = []
					for var in special_global_vars:
						dat['global_values'][var] = []
					itr = 0
				else:
					itr += 1
				
				del ds, reg

			out.close()

		##############################
		# Starts Plotting part
		##############################

		# if (do_plot and refill) or (do_plot and recompute):
		if do_plot:
			# Plotting settings
			left = 0.03
			right = 0.22
			hsep = 0.02
			bottom = 0.06
			top = 0.20 / 2
			vsep = 0.03
			wspace = 0.05  # Not used
			fsx, fsy = [20, 18]
			sac_amp_thres = 4.

			# Loading and chrono ordering the dset
			dd = h5py.File(h5_fn, 'r+')
			dsets = np.array(dd['dsets'])
			argsort = np.argsort(dsets)
			largsort = list(argsort)
			largsort.append(len(largsort))
			largsort = np.array(largsort, dtype=int)			
			dsets = dsets[argsort]
		
			for var, w  in global_vars: 
				try: dd['global_values'][var][:] = np.array(dd['global_values'][var])[:, argsort]
				except: pass
			for var in special_global_vars: 
				try: dd['global_values'][var][:] = np.array(dd['global_values'][var])[argsort]
				except: pass
			for var in dd['dist_values']: 
				try: dd['dist_values'][var][:] = np.array(dd['dist_values'][var])[largsort]
				except: pass
					
			dd['dsets'][:] = np.array(dd['dsets'])[argsort]			
			time = dd['global_values']['time'][:]
			xvals = time

			fig = plt.figure(0, figsize=(35, 20))
			title_fig = "{} {}".format(exp, run)
			plt.suptitle(title_fig, fontsize=24)

			axs = np.empty((7, 7), dtype=object)
			# """ Skeleton """

			x, y = [0, 6]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'omega'
			im = hist2D(axs[x, y], dd, var, kargs={})
			cbar = plt.colorbar(im, ax=axs[x, y])
			axs[x, y].set_title(var)

			# ---------------------------------

			x, y = [1, 6]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'omega'
			im = hist2D(axs[x, y], dd, var, kargs={'clog': True})
			cbar = plt.colorbar(im, ax=axs[x, y])
			axs[x, y].set_title(var)
			# # ---------------------------------

			# x, y = [2, 6]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'logK-'
			# im = hist2D(axs[x, y], dd, var, kargs={'clog': True})
			# cbar = plt.colorbar(im, ax=axs[x, y])
			# axs[x, y].set_title(var)
			# # ---------------------------------

			# x, y = [3, 6]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'logK+'
			# im = hist2D(axs[x, y], dd, var, kargs={'clog': True, "xaxis":"dsets"})
			# cbar = plt.colorbar(im, ax=axs[x, y])
			# axs[x, y].set_title(var)
			

			# # ---------------------------------
			# x, y = [0, 5]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'logHamRel'
			# im = hist2D(axs[x, y], dd, var, kargs={'clog': False})
			# cbar = plt.colorbar(im, ax=axs[x, y])
			# axs[x, y].set_title(var)

			# # ---------------------------------

			# x, y = [1, 5]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'logHamRel'
			# im = hist2D(axs[x, y], dd, var, kargs={'clog': True})
			# cbar = plt.colorbar(im, ax=axs[x, y])
			# axs[x, y].set_title(var)
			# # ---------------------------------

			x, y = [2, 5]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'W'
			im = hist2D(axs[x, y], dd, var, kargs={'clog': False})
			cbar = plt.colorbar(im, ax=axs[x, y])
			axs[x, y].set_title(var)
			# ---------------------------------

			x, y = [3, 5]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'pressure'
			im = hist2D(axs[x, y], dd, var, kargs={'clog': True})
			cbar = plt.colorbar(im, ax=axs[x, y])
			axs[x, y].set_title(var)
			# ---------------------------------

			x, y = [0, 4]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'N'
			axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			axs[x, y].set_title(var)
			# # ---------------------------------

			# x, y = [1, 4]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'av_phi'
			# axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   # dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# axs[x, y].set_title(var)
			# # ---------------------------------

			# x, y = [2, 4]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'HamRel'
			# axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   # dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# axs[x, y].set_title(var)
			# # # ---------------------------------

			# x, y = [3, 4]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'L'
			# axs[x, y].plot(xvals, dd['global_values'][var], 'k', label='mean')
			# axs[x, y].set_title(var)
			# # ---------------------------------

			x, y = [1, 3]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'K'
			axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			axs[x, y].set_title(var)
			# ---------------------------------

			x, y = [0, 3]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'ricci_scalar'
			axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			axs[x, y].set_title(var)
			# ---------------------------------

			x, y = [0, 2]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'rho'
			axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			axs[x, y].set_title(var)
			# ---------------------------------

			x, y = [1, 2]
			gs = gridspec.GridSpec(1, 1)
			gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  wspace=wspace)
			axs[x, y] = plt.subplot(gs[:, :])
			var = 'trA2'
			axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			axs[x, y].set_title(var)
			# ---------------------------------

			# # ---------------------------------

			# x, y = [1, 1]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'grad'
			# axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   # dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# axs[x, y].set_title(var)

			# # ---------------------------------

			# x, y = [1, 0]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'kin'
			# axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   # dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# axs[x, y].set_title(var)

			# ---------------------------------

			# x, y = [0, 0]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# var = 'V'
			# axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
								   # dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# axs[x, y].set_title(var)

			# ---------------------------------

			# x, y = [0, 1]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=left + x * (right + hsep) + right,
					  # bottom=bottom + y * (top + vsep), top=bottom + y * (top + vsep) + top,
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# # var = 'adotdot'
			# # axs[x, y].plot(xvals, dd['global_values'][var][0], 'k', label='mean')
			# # #axs[x, y].fill_between(xvals, dd['global_values'][var][0] - dd['global_values'][var][1],
			# # #					   dd['global_values'][var][0] + dd['global_values'][var][1], color='g', alpha=0.2)
			# # axs[x, y].set_title(var)

			# # # ---------------------------------

			# x, y = [2, 0]
			# gs = gridspec.GridSpec(1, 1)
			# gs.update(left=left + x * (right + hsep), right=2 * left + (x * (right + hsep)) + 2 * right,
					  # bottom=bottom + y * (top + vsep), top=3.5 * (bottom + y * (top + vsep) + top),
					  # wspace=wspace)
			# axs[x, y] = plt.subplot(gs[:, :])
			# axs[x, y].plot(dd['global_values']['grad'][0], dd['global_values']['kin'][0], 'k', label='traj')
			# axs[x, y].set_xlabel('grad/V')
			# axs[x, y].set_ylabel('kin/V')

			# save
			dd.close()
			pfn = "./plots/{exp}_{run}.png".format(run=run, exp=exp)
			
			print("saving figure at ", pfn)
			
			plt.savefig(pfn)
			# plt.show()
			plt.clf()

