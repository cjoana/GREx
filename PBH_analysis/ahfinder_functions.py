##### ######   ##
#        #     ##   Params and function 
#     #  #     ##  
##### ####     ##

import numpy as np
import yt 
import glob
import re

import sys, os
# FILEPATH = os.getcwd()  #  gives *working-directory* which is not necess. the path of file. 
FILEPATH = os.path.dirname(os.path.realpath(__file__))   # os.path.realpath(__file__)[:-21]
sys.path.append(FILEPATH)
print(FILEPATH)

# h5_filepath =  FILEPATH + '/h5_data/'

# dir_dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5AH/'
# h5_filename = './data/{exp}_ahfinder.hdf5'
# ahdata_path = '/public/home/cjoana/outpbh/{exp}/data_PP/'
# dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5AH/'
# h5_filename = h5_filepath + '{exp}_test.hdf5'
# ahdata_path = '/Volumes/Expansion/data/{exp}/data_PP/'


############################################## functions

def get_files_in_path(path, extension ="*.hdf5"):
    fn = path + extension
    fs = glob.glob(fn)
    return fs

def get_prefixes_in_files(file_list):
    fs = file_list
    re_plotfile = 'run\d+p_'
    prefixes =  np.unique(np.hstack([re.findall(re_plotfile, f)  for f in fs ]))
    return prefixes

def get_ids_dsets_in_filelist(file_list, prefix=""): 
    fs = file_list
    re_dsets = prefix + '(\d+)'+'.3d.hdf5'
    id_dsets = np.hstack([re.findall(re_dsets, f)  for f in fs ])
    id_dsets = np.array(id_dsets, dtype=int)
    return id_dsets

def dfn(path, num):
    return path.format(str(num).zfill(6))

def load_dataset(dirpath, prefix, id_dset):
    pfiles = dirpath + prefix + '{0}.3d.hdf5'
    dset_fn = dfn(pfiles, id_dset)
    ds = yt.frontends.chombo.ChomboDataset(dset_fn, unit_system=unit_system, units_override=units_override)
    ds = _add_fields(ds)
    return ds


units_override = {"length_unit": (1.0, "l_pl"),
                  "time_unit": (1.0, "t_pl"),
                  "mass_unit": (1.0, "m_pl")}
unit_system = 'planck'

#######################################  add derives cells and functions

def _add_fields(ds): 
    ds.add_field(('chombo', 'volcell'), function=_volcell, # units="l_pl**3", 
                take_log=False, sampling_type="local",  display_name='volcell')
    ds.add_field(('chombo', 'N'), function=_N, # units="", 
                sampling_type="local", take_log=False, display_name='N') 
    #ds.add_field(('chombo', 'rPsi4'), function=_rPsi4, # units="", 
    #            sampling_type="local", take_log=False, display_name='rPsi4')
  
    return ds

def _N(field, data):
    return np.log(data['chi'] ** -0.5) 

def _volcell(field, data):
    var = data["dx"] ** 3 * data["chi"] ** (-3 / 2)
    return var

def _omega(field, data):
    var = data["S"] / data["rho"] / 3
    return var

def _rPsi4(field, data):
    # TODO does not work ,  cntarg out of bounds
    #cntarg = np.argmin(data["lapse"].d)
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

