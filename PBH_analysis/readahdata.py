import numpy as np

dir_dsets_path = '/public/home/cjoana/outpbh/{exp}/hdf5AH/'
h5_filename = './data/{exp}_test.hdf5'
ahdata_path = '/public/home/cjoana/outpbh/{exp}/data_PP/'
# dir_dsets_path = '/Volumes/Expansion/data/{exp}/hdf5AH/'
# h5_filename = h5_filepath + '{exp}_test.hdf5'
ahdata_path = '/Volumes/Expansion/data/{exp}/data_PP/'

exps = ["asym01","asym02","asym03", "asym04", "pancake", "pancake02" ]
exp = "asym04"

lst_ahdata = [ 'time', 'dset', 'mass', 'spin', 'spin_x', 'spin_y', 'spin_y', 'center_x', 'center_y', 'center_x' ]
ind_ahdata = [ 0, 1, 4, 5, 6, 7, 8, -3, -2, -1]

#time                   file                area                mass                    irreducible mass                
# spin                  dimless spin-x      dimless spin-y      dimless spin-z          dimless spin-z-alt     
# linear mom. |P|       linear mom. Px      linear mom. Py      linear mom. Pz          origin_x            
# origin_y              origin_z            center_x            center_y                center_z
ah_dir_path = ahdata_path.format(exp=exp)
f1 = ah_dir_path + "stats_AH1.dat"
f2 = ah_dir_path + "stats_AH2.dat"

dat = np.loadtxt(f1)

out = h5py.File(h5_fn, "r+")
for iv, var in enumerate(lst_ahdata):
    ahvar = out["ahdata"][var]
    ahvar.resize( (ahvar.shape[0] + 1), axis = 0)
    ahvar[-1] = dat[:, ind_ahdata[iv]]
out.close()