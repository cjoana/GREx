import sys
sys.path.insert(1, '/home/cjoana/git')

import yt 
print(yt.__version__)
import h5py as h5
import numpy as np
import os

# load the packages required
import matplotlib.pyplot as plt;
import numpy as np
import time
from matplotlib import pylab
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from yt import derived_field
import matplotlib.gridspec as gridspec

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d
np.random.seed(777777)

def HG(phi):
    Mp = 1.0/np.sqrt(8.0*np.pi)
    mass = 9.6e-11
    V = mass * (Mp ** 4.0) *  (1.0 - np.exp(-np.sqrt(2.0/3.0) * phi/Mp) ) **2.0   
    return V

path = "./"
fn = path + "InitialConditionsFinal.3d.hdf5"
ds = yt.frontends.grchombo.GRChomboDataset(fn)  


comps = np.array( ds._get_components(), dtype=str)
domain = ds._get_domain()
data = ds._get_data(comps)


sf = data['phi'] * 0 + 10
pot = HG( sf )
kin = np.sqrt( 2* (data['rho'] - pot) + 1e-20 )

kin[ np.isnan(kin) ] = 0.0
print("Kin max/min = " , kin.min(), kin.max() )

data['phi'] = data['phi'] * 0 + 10
data['Pi'] = kin


ds.create_dataset_level0(path + "InitialConditionsFinal_kin.3d.hdf5", data=data, overwrite=True)


print("Done.")

exit()
