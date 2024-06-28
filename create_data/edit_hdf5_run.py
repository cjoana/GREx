import h5py
import numpy as np

verbose = 2 
path = "./run0_rerun.3d.hdf5"

if True: ## security
    
    with h5py.File(path,  "a") as h5:
        # h5 = h5py.File(path, 'a')
        if verbose >1: print(h5.keys())
        if verbose >1: print(h5.attrs.keys()  )
        
        
        h_keys = h5.keys()

        for k in h_keys:
            
            sbh5 = h5[k]
            sbh5_att =sbh5.attrs.keys()
            if verbose >1: print("atts in ", k, " are ", sbh5_att)

        
        h5.attrs["iteration"] = 0
        
        print(" edited ", path) 
        print( f"  >> MODIFIED:  attr. 'iteration' to {h5.attrs['iteration']}")
        
