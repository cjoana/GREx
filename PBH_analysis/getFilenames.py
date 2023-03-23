##
#
##

import glob
import re

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

exp_path = dir_dsets_path.format(exp=exp) 


print(f"Reading from : {exp_path}")

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

### Example

files = get_files_in_path(exp_path)
prefixes = get_prefixes_in_files(files)

# print(files)
# print(prefixes)


for prefx in prefixes: 
    ids_dsets = get_ids_dsets_in_filelist(files, prefix=prefx)
    
    print(f"with prefix {prefx} found {len(ids_dsets)} files : {ids_dsets[0:2]} ... {ids_dsets[-2:]} ")
