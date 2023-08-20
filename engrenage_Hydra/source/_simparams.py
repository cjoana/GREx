import multiprocessing as mp

# homemade code
from source.uservariables import *
from source.gridfunctions import *
from source.fourthorderderivatives import *
from source.logderivatives import *
from source.tensoralgebra import *
from source.mymatter import *
from source.bssnrhs import *

DefSimParams = True 

N_r = 100   # Number of points 
# R = 150.0   # Maximum outer radius
eta = 2.0

# r_is_logarithmic = False





"""
Shared Vars
"""


sh_hrr = mp.Array('d', N_r + 2*num_ghosts)
sh_htt = mp.Array('d', N_r + 2*num_ghosts)
sh_hpp = mp.Array('d', N_r + 2*num_ghosts)
sh_arr = mp.Array('d', N_r + 2*num_ghosts)
sh_att = mp.Array('d', N_r + 2*num_ghosts)
sh_app = mp.Array('d', N_r + 2*num_ghosts)
sh_r = mp.Array('d', N_r + 2*num_ghosts)
sh_phi = mp.Array('d', N_r + 2*num_ghosts)
sh_u = mp.Array('d', N_r + 2*num_ghosts)
sh_v = mp.Array('d', N_r + 2*num_ghosts)
sh_K = mp.Array('d', N_r + 2*num_ghosts)
sh_lapse = mp.Array('d', N_r + 2*num_ghosts)
sh_shiftr = mp.Array('d', N_r + 2*num_ghosts)
sh_br = mp.Array('d', N_r + 2*num_ghosts)
sh_lambdar = mp.Array('d', N_r + 2*num_ghosts)

# RHS 
rhs_hrr = mp.Array('d', N_r + 2*num_ghosts)
rhs_htt = mp.Array('d', N_r + 2*num_ghosts)
rhs_hpp = mp.Array('d', N_r + 2*num_ghosts)
rhs_arr = mp.Array('d', N_r + 2*num_ghosts)
rhs_att = mp.Array('d', N_r + 2*num_ghosts)
rhs_app = mp.Array('d', N_r + 2*num_ghosts)
rhs_r = mp.Array('d', N_r + 2*num_ghosts)
rhs_phi = mp.Array('d', N_r + 2*num_ghosts)
rhs_u = mp.Array('d', N_r + 2*num_ghosts)
rhs_v = mp.Array('d', N_r + 2*num_ghosts)
rhs_K = mp.Array('d', N_r + 2*num_ghosts)
rhs_lapse = mp.Array('d', N_r + 2*num_ghosts)
rhs_shiftr = mp.Array('d', N_r + 2*num_ghosts)
rhs_br = mp.Array('d', N_r + 2*num_ghosts)
rhs_lambdar = mp.Array('d', N_r + 2*num_ghosts)


# Derivs
# second derivatives
sh_d2udx2     = mp.Array('d', N_r + 2*num_ghosts)
sh_d2phidx2   = mp.Array('d', N_r + 2*num_ghosts)
sh_d2hrrdx2   = mp.Array('d', N_r + 2*num_ghosts)
sh_d2httdx2   = mp.Array('d', N_r + 2*num_ghosts)
sh_d2hppdx2   = mp.Array('d', N_r + 2*num_ghosts)    
sh_d2lapsedx2   = mp.Array('d', N_r + 2*num_ghosts) 
sh_d2shiftrdx2   =mp.Array('d', N_r + 2*num_ghosts) 
# first derivatives        
sh_dudx       = mp.Array('d', N_r + 2*num_ghosts)
sh_dvdx       = mp.Array('d', N_r + 2*num_ghosts)
sh_dphidx     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhrrdx     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhttdx     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhppdx     = mp.Array('d', N_r + 2*num_ghosts)
sh_darrdx     = mp.Array('d', N_r + 2*num_ghosts) #
sh_dattdx     = mp.Array('d', N_r + 2*num_ghosts) #
sh_dappdx     = mp.Array('d', N_r + 2*num_ghosts) #
sh_dKdx       = mp.Array('d', N_r + 2*num_ghosts)
sh_dlambdardx = mp.Array('d', N_r + 2*num_ghosts)
sh_dbrdx    = mp.Array('d', N_r + 2*num_ghosts)   # 
sh_dshiftrdx  = mp.Array('d', N_r + 2*num_ghosts)
sh_dlapsedx   = mp.Array('d', N_r + 2*num_ghosts)
# first derivatives - advec left and right
sh_dudx_advec_L       = mp.Array('d', N_r + 2*num_ghosts)
sh_dvdx_advec_L       = mp.Array('d', N_r + 2*num_ghosts)
sh_dphidx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhrrdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhttdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dhppdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_darrdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dattdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dappdx_advec_L     = mp.Array('d', N_r + 2*num_ghosts)
sh_dKdx_advec_L       = mp.Array('d', N_r + 2*num_ghosts)
sh_dlambdardx_advec_L = mp.Array('d', N_r + 2*num_ghosts)
sh_dshiftrdx_advec_L  = mp.Array('d', N_r + 2*num_ghosts) #
sh_dbrdx_advec_L      = mp.Array('d', N_r + 2*num_ghosts) #
sh_dlapsedx_advec_L   = mp.Array('d', N_r + 2*num_ghosts) #
sh_dudx_advec_R         = mp.Array('d', N_r + 2*num_ghosts)
sh_dvdx_advec_R         = mp.Array('d', N_r + 2*num_ghosts)
sh_dphidx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dhrrdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dhttdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dhppdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_darrdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dattdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dappdx_advec_R       = mp.Array('d', N_r + 2*num_ghosts)
sh_dKdx_advec_R         = mp.Array('d', N_r + 2*num_ghosts)
sh_dlambdardx_advec_R   = mp.Array('d', N_r + 2*num_ghosts)
sh_dshiftrdx_advec_R  = mp.Array('d', N_r + 2*num_ghosts) #
sh_dbrdx_advec_R      = mp.Array('d', N_r + 2*num_ghosts) #
sh_dlapsedx_advec_R   = mp.Array('d', N_r + 2*num_ghosts) # 
