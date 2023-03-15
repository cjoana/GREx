import h5py
import numpy as np


path = ""

last_dset = 1e11 #


if True: ## security
    
    with h5py.File(path,  "a") as h5:
        # h5 = h5py.File(path, 'a')
        print(h5.keys())

        dsets = h5['dsets'][:]
        gv =  h5['global_values']
        dv =  h5['dist_values']

        mask = (dsets < last_dset)

        print("mask ones: ", np.sum(mask))

        k_gv = gv.keys()
        k_dv = dv.keys()
        print(len(dsets))

        for k in k_gv:
            print(k, len( gv[k] ))
            if len( gv[k] ) == 4:
                if len( gv[k][0]) == np.sum(mask):
                    print('skip  ', k )
                    continue
                dat = gv[k][:]        
                dat= dat[:, mask]
                del gv[k]            
                gv[k] = dat 
                # print(dat.shape)
            else:
                dat = gv[k][:]  
                dat= dat[mask]
                del gv[k]            
                gv[k] = dat

        for k in k_dv:
            print(k,  dv[k].shape )
            n_mask = np.hstack([1, mask])
            dat = dv[k][:]
            dat = dat[n_mask, :]
            del dv[k]            
            dv[k] = dat

        dat = dsets[mask]
        del h5['dsets']
        h5['dsets'] = dat

