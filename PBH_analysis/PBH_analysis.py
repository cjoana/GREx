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

from analysis_functions import load_dataset
import analysis_functions as af 

# dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
# h5_filename = './data/{exp}_test.hdf5'
dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
h5_filepath =  FILEPATH + '/h5_data/'
h5_filename = h5_filepath + '{exp}_test.hdf5'

prefx = "run1p_"  
exps = ["asym04", ]
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
    ['delta_rho', 'all'],
    #
]

lst_metadata = [
    'time',  'Vol', 'dset',
]



for exp in exps:
    print('Initiating collection of ', exp) 

    # prepare h5 sumary
    h5_fn = h5_filename.format(exp=exp)
    if recompute: 
        if not os.path.exists(h5_filepath): os.mkdir(h5_filepath)
        out = h5py.File(h5_fn, "w")
        print('Creating h5-analysis ', h5_fn)
        simdata = out.create_group("simulated_data")
        for item in lst_simdata:
            simdata.create_dataset(item[0], data=[[], [], [], []], maxshape=(None, None,))
        metadata = out.create_group("metadata")
        for item in lst_metadata:
            _dtype = int if item=="dset" else None
            simdata.create_dataset(item, data=[], maxshape=(None,), dtype=_dtype)
        out.close()
        print('Creating h5-analysis ', h5_fn, "  DONE. ")
        
        



    # Load data

    i_dset = 1000
    
    print("Loading simulated data ")    
    ds = load_dataset(num=i_dset, exp=exp)    



    # Read & extract Sim data

    reg = ds.r[:]
    vol_cell = np.ndarray.flatten(reg['volcell'])
    vol = np.sum(vol_cell)*1.0
    print(f"Volume of box is {vol.d},  effective L = {vol.d **(1/3)}")
    
    
    # Save data
    
    


    # load BH data


    ##  lapse, rho, trA2, ricci_scalar, N,  K, S, W, omega, Ham, Ham_abs_terms, Mom, Mom_abs_terms
    ##  delta_rho, 
    ##  Volume,
    ##                          __>>  mean, std, min, max, "center = minlapse"