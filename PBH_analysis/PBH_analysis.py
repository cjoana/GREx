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
FILEPATH = os.path.dirname(os.path.realpath(__file__)) # os.path.realpath(__file__)[:-21]
sys.path.append(FILEPATH)
print(FILEPATH)

# from analysis_functions import exps, dsets_path, prefx
import analysis_functions as af 



def _add_fields(ds): 
    ds.add_field(('chombo', 'volcell'), function=af._volcell, # units="l_pl**3", 
                take_log=False, sampling_type="local",  display_name='volcell')
    ds.add_field(('chombo', 'N'), function=af._N, # units="", 
                sampling_type="local", take_log=False, display_name='N')
    
    return ds


for exp in af.exps:
    print('Initiating collection of ', exp) 

    # Load data

    i_dset = 1000
    
    print("Loading simulated data ")
    data_path = af.dsets_path.format(exp=exp)
    pfiles = data_path + af.prefx + '{0}.3d.hdf5'
    dset_fn = af.dfn(pfiles, i_dset)
    ds = yt.frontends.chombo.ChomboDataset(dset_fn, unit_system=af.unit_system, units_override=af.units_override)
    ds = _add_fields(ds)
    





    # Read & extract Sim data

    reg = ds.r[:]
    vol_cell = np.ndarray.flatten(reg['volcell'])
    vol = np.sum(vol_cell)*1.0
    print(f"Volume of box is {vol.d},  effective L = {vol.d **(1/3)}")
    
    
    # Save data

    h5_fn = af.h5_filename.format(exp=exp)
    print('Creating h5-analysis ', h5_fn)


    # load BH data


    ##  lapse, rho, trA2, ricci_scalar, N,  K, S, W, omega, Ham, Ham_abs_terms, Mom, Mom_abs_terms
    ##  delta_rho, 
    ##  Volume,
    ##                          __>>  mean, std, min, max, "center = minlapse"