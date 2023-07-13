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

import sys, os
# FILEPATH = os.getcwd()  #  gives *working-directory* which is not necess. the path of file. 
FILEPATH = os.path.dirname(os.path.realpath(__file__))   # os.path.realpath(__file__)[:-21]
sys.path.append(FILEPATH)
print(FILEPATH)

from analysis_functions import load_dataset, get_prefixes_in_files, get_files_in_path, get_ids_dsets_in_filelist
import analysis_functions as af 

import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
from scipy.ndimage import map_coordinates
from scipy.interpolate import interpn, griddata

def dens_plot(ax, sdata, mycmap, **kargs):
    p = ax.imshow(sdata, interpolation='spline16', 
            cmap=mycmap, **kargs)
    cbar = fig.colorbar(p,ax=ax, aspect=8, shrink=0.7)

    cbar.ax.tick_params(labelsize=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return p, cbar


mpl.rcParams.update({'font.size': 10,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)

#mpl.rcParams['legend.edgecolor'] = 'inherit'

mpl.rcParams.update(mpl.rcParamsDefault)



verbose = 2

## input dirs
dir_dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
h5_filename = './data/{exp}_summary.hdf5'
# dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
# h5_filename = h5_filepath + '{exp}_test.hdf5'

## output dirs
dir_plots = FILEPATH + '/plots/'
dir_output = dir_plots

prefix = None #  "runXXX"  ,  None =  found automatically if uniquely. 
exps = [
    # "asym01",
    #"asym02","asym03", "asym04",
    # "pancake", 
    #"pancake02",
    "asym05"
    ]


############################################  Set vars 

lst_plot_vars = [
    #'Ham',
    #'Ham_abs_terms', 
    #'Mom',
    #'Mom_abs_terms',
    #'K',
    #'ricci_scalar',
    #'trA2',
    #'omega', 
    #'W', 
    #'lapse', 
 
    #
    'deltaN',    
    'rho',     
    'deltarho',
]



for exp in exps:
    if verbose > 1 : print('Initiating collection of ', exp) 

	# check existence of output dir
    if not os.path.exists(dir_plots): os.makedirs(dir_plots)
        
    # Search h5-files in dirpath, get IDs dsets 
    if verbose > 1 : print("Searching datasets - hdf5 files ")    
    dirpath = dir_dsets_path.format(exp=exp)
    files = get_files_in_path(dirpath)
    if not prefix: 
        prefixes = get_prefixes_in_files(files)
        if len(prefixes)>1 : 
            print(f" !!! SKIP exp {exp} because too many prefixes (e.g. {prefixes})")
            continue
        prefix = prefixes[0]    
    lst_dsets = get_ids_dsets_in_filelist(files, prefix=prefix)
    lst_dsets = np.sort(lst_dsets)


    # Loop through datasets and extract data
    if verbose > 1 : print("Loading simulated data ")    
    for idd, id_dset in enumerate(lst_dsets):
		
		if idd >3: break
		
		# Reset options
		recalculate_slice = 1

        # Load data
        ds = load_dataset(dirpath, prefix, id_dset)    
        reg = ds.r[:]

        ## METADATA
        vol_cell = np.ndarray.flatten(reg['volcell'])
        vol = np.sum(vol_cell)*1.0
        weights = vol_cell / vol
        L = vol ** (1 / 3)
        time = ds.current_time   
        m_dict = dict()
        m_dict['Vol'] = vol 
        m_dict['L'] = L
        m_dict['dset'] = id_dset
        m_dict['time'] = time
        if verbose>2: print(f"Volume of box is {vol.d},  effective L = {vol.d **(1/3)}")
        
		## create slice
		if recalculate_slice:
			recalculate_slice = 0
			
			X, Y, Z = (	dd["x"] , 
						dd["y"] ,
						dd["z"] )
			try: 
				rho = dd['D'] + dd["E"]
			except:
				rho = dd["rho"]
			Hmean = dd["K"]**2/24/np.pi
			pdata = rho/Hmean
			idx = np.argmax(pdata)
			zpos = Z[idx]
			dN = np.max(dd["dx"])
			mask = (Z > zpos - dN) & (Z < zpos + dN)
			
			g1 = np.linspace(X.min(), X.max(), 21)
			g2 = np.linspace(X.max()*1/3, X.max()*2/3, 60)
			pos = np.sort(np.hstack([g1,g2]))
			xi, yi = np.meshgrid(pos, pos)
			zi = np.ones_like(xi) * zpos
	
        
        
        for ivar, var in enumerate(lst_plot_vars):
			
			## Set outputdirs
			dir_output = dir_plots + f"{exp}/{var}/"
			if not os.path.exists(dir_output): os.makedirs(dir_output)
			prefx = ""
			sufx = ""
			outfile = "{prefx}{id_dset}{sufx}.png"
			
			
			if (var =="deltaN"): 
				w = weights
				fd = dd["N"]
				avg = np.average(fd, weights=w)
				
				pdata = fd - avg
				
			else:		
			    pdata  = dd[var] 		 

			idata = griddata( (X[mask],Y[mask],Z[mask]), pdata[mask], 
					(xi , yi, zi ) , method='nearest' )
			sdata = idata
				

			## Plotting
			
			mymap1 = mpl.cm.magma
			mymap2 = mpl.cm.seismic
			mymap3 = mpl.cm.CMRmap_r
			mymap4 = mpl.cm.gist_gray_r


			# p1
			fig, ax = plt.subplots(1,1, figsize=(8,8))
			p1, cb1 = dens_plot(ax, sdata, mymap1)
		
			prefx = ""
			sufx = ""
			outfile = f"{prefx}{id_dset}{sufx}.png"
			dir_output_p = dir_output + "/p1/"
			plt.tight_layout()
			if not os.path.exists(dir_output_p): os.makedirs(dir_output_p)
			plt.savefig(dir_output_p + outfile, dpi=300 )


			# p2
			fig, ax = plt.subplots(1,1, figsize=(8,8))
			p2, cb2 = dens_plot(ax, sdata, mymap2)

			prefx = ""
			sufx = ""
			outfile = f"{prefx}{id_dset}{sufx}.png"
			dir_output_p = dir_output + "/p2/"
			plt.tight_layout()
			if not os.path.exists(dir_output_p): os.makedirs(dir_output_p)
			plt.savefig(dir_output_p + outfile, dpi=300 )


			# logplots
			sdata = np.abs(sdata)


			# p3
			fig, ax = plt.subplots(1,1, figsize=(8,8))
			p3, cb3  = dens_plot(ax, sdata, mymap3 ,
				norm=LogNorm(vmin=sdata.min(), vmax= 10**(np.log10( sdata.max())//1 +1 ) )
				)

			prefx = "log_"
			sufx = ""
			outfile = f"{prefx}{id_dset}{sufx}.png"
			dir_output_p = dir_output + "/p3/"
			plt.tight_layout()
			if not os.path.exists(dir_output_p): os.makedirs(dir_output_p)
			plt.savefig(dir_output_p + outfile, dpi=300 )


			# p4
			fig, ax = plt.subplots(1,1, figsize=(8,8))
			p4, cb4  = dens_plot(ax, sdata, mymap2,
				norm=LogNorm(vmin=sdata.min(), vmax= 10**(np.log10( sdata.max())//1 +1 ) )
				)

			prefx = "log_"
			sufx = ""
			outfile = f"{prefx}{id_dset}{sufx}.png"
			dir_output_p = dir_output + "/p4/"
			plt.tight_layout()
			if not os.path.exists(dir_output_p): os.makedirs(dir_output_p)
			plt.savefig(dir_output_p + outfile, dpi=300 )
		




        if verbose >1 : print(f"Collecting data from {exp}, dset {id_dset} is DONE.")
    if verbose >1 : print(f"Collecting data from {exp} is DONE.")
