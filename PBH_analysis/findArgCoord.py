##
#
##
import yt

#
import numpy as np
import sys, os
# FILEPATH = os.getcwd()  #  gives *working-directory* which is not necess. the path of file. 
FILEPATH = os.path.dirname(os.path.realpath(__file__))   # os.path.realpath(__file__)[:-21]
sys.path.append(FILEPATH)
print(FILEPATH)



#h5_filepath =  FILEPATH + '/h5_data/'
#dir_dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5/'
#h5_filename = './data/{exp}_test.hdf5'
dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5/'
#h5_filename = h5_filepath + '{exp}_test.hdf5'

prefx = "run1p_"  
exps = ["pancake", ]
exp = exps[0]

exp_path = dir_dsets_path.format(exp=exp)  + prefx + "000100.3d.hdf5"

print(exp_path, os.path.exists(exp_path))

ds = yt.load(exp_path)

# find arg for: 
wC =  [15, 15, 15]

def find_argcord(ds, coord):
    reg = ds.r[:]

    X , Y, Z = [reg['x'], reg['y'] , reg['z'] ]
    rad2 = X**2 + Y**2 + Z**2
    mask = (X>=coord[0]) * (Y>=coord[1]) * (Z>=coord[2])
    warg = np.where(mask)[0]
    iarg = np.argmin(rad2[warg])
    arg = warg[iarg]

    return arg





