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

from analysis_functions import *
import analysis_functions as af 

units_override = {"length_unit": (1.0, "l_pl"),
				  "time_unit": (1.0, "t_pl"),
				  "mass_unit": (1.0, "m_pl")}
unit_system = 'planck'


# dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
# h5_filename = './data/{exp}_test.hdf5'
dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
h5_filename = './h5_data/{exp}_test.hdf5'


prefx = "run1p_"  
exps = [
    "asym04"
]

for exp in exps:
    print('Initiating collection of ', exp) 
		
    i_dset = 1000
    
    print("Loading simulated data ")
    data_path = dsets_path.format(exp=exp)
    pfiles = data_path + prefx + '{0}.3d.hdf5'
    dset_fn = af.dfn(pfiles, i_dset)
    ds = yt.frontends.chombo.ChomboDataset(dset_fn, unit_system=unit_system, units_override=units_override)
    ds.add_field(('chombo', 'volcell'), function=af._volcell, # units="l_pl**3", 
                 take_log=False, sampling_type="local",  display_name='volcell')
    ds.add_field(('chombo', 'N'), function=af._N, # units="", 
                 sampling_type="local", take_log=False, display_name='N')
    

    reg = ds.r[:]
    vol_cell = np.ndarray.flatten(reg['volcell'])
    vol = np.sum(vol_cell)*1.0
    print(f"Volume of box is {vol.d},  effective L = {vol.d **(1/3)}")
    
    
    h5_fn = h5_filename.format(exp=exp)
    print('Creating h5-analysis ', h5_fn)
