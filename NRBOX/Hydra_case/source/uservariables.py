#uservariables.py

# This file provides the list of (rescaled) variables to be evolved and
# assigns each one an index and its parity
# For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background


#  chi, a, b, K, AX, X, Lambda, lapse, beta, D, E, S

idx_chi     = 0    # conformal factor of metric
idx_a       = 1    # 
idx_b       = 2    # 
idx_K       = 3    # mean curvature K
idx_Aa      = 4
idx_AX      = 5    # A_lambda in alcubierre's paper
idx_X       = 6    # lambda in alcubierre's paper 
idx_Lambda  = 7    # Delta^r in alcubierre's papaer
idx_lapse   = 8    # lapse
idx_beta    = 9    # shift^r
idx_br      = 10    # br for evolution of beta

idx_D       = 11    # fluid cons. var for rest energy
idx_E       = 12    # fluid cons. var for intrinsic energy
idx_S       = 13    # fluid cons. var for radial momenta

NUM_VARS = idx_S + 1

variable_names = [ "chi", "a", "b", "K", 
                   "Aa", "AX", "X", "Lambda", 
                   "lapse", "beta", "br", 
                   "D", "E", "S"]

# parity under r -> -r
parity = [
          1, 1, 1, 1,     # chi, a, b, K 
          1, 1, 1, -1,    # Aa, AX, X,  Lambda
          1, -1, -1,      # lapse, beta, br
          1, 1, -1,       # D, E, S
          ]  


#### very much dependent on gauge choice!!!
# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
# currently only for linear r
asymptotic_power =   [          ### TODO check
                       0, 0., 0., 0,      # chi, a, b, K 
                       -2., -2., -2., -2.,    # Aa, AX, X, Lambda, 
                       0., 1.,  1.,         # lapse, beta, br
                       0., 0.,  -1.,          # D, E, S
                     ]        



# asymptotic_power =   [0., 0.,                # u, v
#                       -1., -1., -1., -1.,    # phi, h
#                       -2., -2., -2., -2.,    # K, a
#                       -2., -1., -1., 0.]     # lambda^r, shift^r, b^r, lapse
