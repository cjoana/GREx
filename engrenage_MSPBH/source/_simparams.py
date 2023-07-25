import multiprocessing as mp

import sys
sys.path.append("/home/cjoana/dev/GREx/engrenage/")

# homemade code
from source.uservariables import *
from source.gridfunctions import *
# from source.fourthorderderivatives import *
# from source.logderivatives import *
# from source.tensoralgebra import *
# from source.mymatter import *
# from source.bssnrhs import *

DefSimParams = True 

N_r = 1000   # Number of points 
# R = 150.0   # Maximum outer radius
# eta = 2.0

# r_is_logarithmic = False





"""
Shared Vars
"""

sh_r = mp.Array('d', N_r + 2*num_ghosts)

sh_t_i = mp.Array('d', 1)

sh_U = mp.Array('d', N_r + 2*num_ghosts)
sh_M = mp.Array('d', N_r + 2*num_ghosts)
sh_R = mp.Array('d', N_r + 2*num_ghosts)
sh_rho = mp.Array('d', N_r + 2*num_ghosts)

sh_rhs_U = mp.Array('d', N_r + 2*num_ghosts)
sh_rhs_M = mp.Array('d', N_r + 2*num_ghosts)
sh_rhs_R = mp.Array('d', N_r + 2*num_ghosts)
sh_rhs_rho = mp.Array('d', N_r + 2*num_ghosts)

sh_dUdr = mp.Array('d', N_r + 2*num_ghosts)
sh_dMdr = mp.Array('d', N_r + 2*num_ghosts)
sh_dRdr = mp.Array('d', N_r + 2*num_ghosts)
sh_drhodr = mp.Array('d', N_r + 2*num_ghosts)

 
