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

verbose = 2

h5_filepath =  FILEPATH + '/h5_data/'

dir_dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
h5_filename = './data/{exp}_summary.hdf5'
# dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
# h5_filename = h5_filepath + '{exp}_test.hdf5'

prefix = None #  "runXXX"  ,  None =  found automatically if uniquely. 
exps = [
    # "asym01",
    #"asym02","asym03", "asym04",
     "pancake", 
    #"pancake02",
    ]
recompute = True

############################################  Set vars 

lst_simdata = [
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
    ['deltarho', 'all'],
    #
]

  #   (x, y z) of @lapsemin and @lapsemax

struct_simdata = [[], [], [], [], [], []]  #  var = [ mean, std, min, max, @lapsemin, @lapsemax ]
struct_metadata =[]
lst_metadata = [
    'time',  'Vol', 'dset',
    "x_lapsemin",  "y_lapsemin",  "z_lapsemin", 
    "x_lapsemax",  "y_lapsemax",  "z_lapsemax", 
]

for exp in exps:
    if verbose > 1 : print('Initiating collection of ', exp) 

    # prepare h5 sumary
    h5_fn = h5_filename.format(exp=exp)
    if recompute: 
        if not os.path.exists(h5_filepath): os.mkdir(h5_filepath)
        out = h5py.File(h5_fn, "w")
        if verbose > 1 : print('Creating h5-analysis ', h5_fn)
        simdata = out.create_group("simulated_data")
        for item in lst_simdata:
            simdata.create_dataset(item[0], data=struct_simdata, maxshape=(None, None,))
        metadata = out.create_group("metadata")
        for item in lst_metadata:
            _dtype = int if item=="dset" else None
            metadata.create_dataset(item, data=struct_metadata, maxshape=(None,), dtype=_dtype)
        out.close()
        if verbose > 2 : print('Creating h5-analysis ', h5_fn, "  DONE. ")
        
        
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

    #TODO: reduce dataset with only not existen ones. (refill only)
    if not recompute:
        raise(" !! STOP !!  'recompute' is False. Refill method has not been coded yet. ")


    # Loop through datasets and extract data
    if verbose > 1 : print("Loading simulated data ")    
    for idd, id_dset in enumerate(lst_dsets):
        #if idd > 2:  break

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

        ## Find coordinates 
        vlap = np.ndarray.flatten(reg["lapse"])
        arglapmin = np.argmin(vlap)
        arglapmax = np.argmax(vlap)
        if isinstance(arglapmin, list): arglapmin=arglapmin[0]
        if isinstance(arglapmax, list): arglapmax=arglapmax[0]
        
        xfd = np.ndarray.flatten(reg["x"])
        yfd = np.ndarray.flatten(reg["y"])
        zfd = np.ndarray.flatten(reg["z"])
        
        m_dict['x_lapsemin'] = xfd[arglapmin]
        m_dict['y_lapsemin'] = yfd[arglapmin]
        m_dict['z_lapsemin'] = zfd[arglapmin]
        m_dict['x_lapsemax'] = xfd[arglapmax]
        m_dict['y_lapsemax'] = yfd[arglapmax]
        m_dict['z_lapsemax'] = zfd[arglapmax]

        # Save metada
        out = h5py.File(h5_fn, "r+")
        for var in lst_metadata:
            metavar = out["metadata"][var]
            metavar.resize( (metavar.shape[0] + 1), axis = 0)
            metavar[-1] = m_dict[var]
        out.close()


        ## SIM DATA
        #  Structure:   var : [ mean, std, min, max, @lapsemin, @lapsemax ]    
        for var, weighted in lst_simdata:
            fd = np.ndarray.flatten(reg[var])
            if not weighted:
                w = np.ones_like(weights)
            elif weighted == 'all':
                w = weights 
            else:
                raise(f"weights not defined, {weighted} ?")			
            avg = np.average(fd, weights=w)
            std = np.sqrt(np.cov(fd, aweights=w))
            vmin, vmax = [np.min(fd), np.max(fd)]
            vals = [avg, std, vmin, vmax, fd[arglapmin], fd[arglapmax]]
            #nvs = len(vals)

            out = h5py.File(h5_fn, "r+")
            datavar = out["simulated_data"][var]
            m, n = datavar.shape
            datavar.resize((m , n +1))
            datavar[:,-1] = vals
            out.close()
            
            if verbose >2: print(f"  > Collected {var} from {exp}.")

        if verbose >1 : print(f"Collecting data from {exp}, dset {id_dset} is DONE.")
    if verbose >1 : print(f"Collecting data from {exp} is DONE.")
