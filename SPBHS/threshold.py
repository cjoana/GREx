

#In this modulus is computed the value of the amplitude of the perturbation for a given value of delta.
import numpy as np
from curvature_profiles import curvature_profile


def A_versus_threshold(xf,w,thresh,rmf):
	Krm = curvature_profile(rmf,rmf)
	fw = 3.*(1.+w)/(5.+3.*w)
	Amp = thresh/(fw*Krm*rmf**2)
	return Amp










