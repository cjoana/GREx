#uservariables.py

# This file provides the list of (rescaled) variables to be evolved and
# assigns each one an index and its parity
# For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background


#  chi, a, b, K, AX, X, Lambda, lapse, beta, D, E, S

idx_chi     = 0    # conformal factor of metric
idx_a       = 1    # 
idx_b       = 2    # 
idx_K       = 3    # mean curvature K
idx_AX      = 4    # A_lambda in alcubierre's paper
idx_X       = 5    # lambda in alcubierre's paper 
idx_Lambda  = 6    # Delta^r in alcubierre's papaer
idx_lapse   = 7    # lapse
idx_beta    = 8   # shift^r
idx_br      = 9   # br for evolution of beta

idx_D       = 10    # fluid cons. var for rest energy
idx_E       = 11    # fluid cons. var for intrinsic energy
idx_S       = 12    # fluid cons. var for radial momenta

NUM_VARS = idx_lapse + 1

variable_names = [ "phi", "hrr", "htt", "hpp", 
                   "K", "arr", "att", "app", 
                   "lambdar", "shiftr", "br", "lapse",
                   "D", "E", "V"]

# parity under r -> -r
parity = [
          1, 1, 1, 1,     # phi, h
          1, 1, 1, 1,     # K, a
          -1, -1, -1, 1,  # lambda^r, shift^r, b^r, lapse
          1, 1, -1,       # D, E, V
          ]  


#### very much dependent on gauge choice!!!
# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
# currently only for linear r
asymptotic_power =   [
                      -1., -1., -1., -1.,    # phi, h
                      -2., -2., -2., -2.,    # K, a
                      -2., -1., -1., 0.,     # lambda^r, shift^r, b^r, lapse
                      0., 0.,  -1.,          # D, E, V
                      ]             
