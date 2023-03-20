##### ######   ##
#        #     ##   Params and function 
#     #  #     ##  
##### ####     ##

import numpy as np
import yt 


############################################## functions

def dfn(path, num):
    return path.format(str(num).zfill(6))

def load_dataset(num, exp=None, dirpath=None, fullpath=None):
    if fullpath: 
        dset_fn = fullpath
    elif dirpath:
        dir_path = dirpath 
        pfiles = dir_path + prefx + '{0}.3d.hdf5'
        dset_fn = dfn(pfiles, num)
    else: 
        dir_path = dir_dsets_path.format(exp=exp)
        pfiles = dir_path + prefx + '{0}.3d.hdf5'
        dset_fn = dfn(pfiles, num)
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
    cntarg = np.argmin(data["lapse"].d)
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

