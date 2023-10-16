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
idx_beta    = 8    # shift^r
idx_br      = 9    # br for evolution of beta

idx_phi     = 10    # phi
idx_Pi      = 11    # Pi


NUM_VARS = 12  ## (idx_[last] + 1)

variable_names = [ "chi", "a", "b", "K", 
                   "AX", "X", "Lambda", 
                   "lapse", "beta", "br", 
                   "phi", "Pi"]

# parity under r -> -r
parity = [
          1, 1, 1, 1,   # chi, a, b, K 
          1, 1, -1,       # AX, X,  Lambda
          1, -1, -1,      # lapse, beta, br
          1, 1,           # phi, Pi
          ]  


#### very much dependent on gauge choice!!!
# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
# currently only for linear r
asymptotic_power =   [          ### TODO check
                       0., 0., 0., 0.,    # chi, a, b, K 
                       0., 0., 0.,         # AX, X, Lambda, 
                       0., 0,  0,         # lapse, beta, br
                       0., 0.,  0.,          # D, E, S
                     ]             
