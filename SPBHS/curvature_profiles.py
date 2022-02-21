
# This modulus is used to build the curvature profile and its derivative K,K'. This two things are used to build the initial perturbations. In this example is used the Gausian curvature profile.
import numpy as np

def curvature_profile(xf,rmf):
	K = np.exp(-(xf/rmf)**2)
	return K

def derivative_curvature_profile(xf,rmf):
	Kd = -2.*(xf/rmf**2)*np.exp(-(xf/rmf)**2)
	return Kd
	





