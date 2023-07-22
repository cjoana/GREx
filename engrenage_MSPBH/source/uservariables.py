#uservariables.py

# This file provides the list of (rescaled) variables to be evolved and
# assigns each one an index and its parity
# For description of the data structure see https://github.com/GRChombo/engrenage/wiki/Useful-code-background

idx_U       = 0    # Eulerian velocity
idx_R       = 1    # areal radius
idx_M	    = 2    # Misner-Sharp mass
idx_rho     = 3    # energy density
# idx_A       = 4    # lapse
# idx_B       = 5    # spatial radial line-element term
# idx_omega   = 6    # EoS


NUM_VARS = idx_rho + 1    # BE CAREFUL if YOU ADD vars

variable_names = ["U", "R", "M", "rho"]

# parity under r -> -r
parity = [ 
          1, 1, 1, 1,      # U, R, M, rho, 
         ]  

# scaling at larger r as power of r, i.e. var ~ r^asymptotic_power
# currently only for linear r
asymptotic_power =   [
					   1., 1.,                # U, R, 
                       3., 0, 		     	  # M, rho 
					 ]    
