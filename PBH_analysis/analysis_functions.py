##### ######   ##
#		 #	   ##   Params and function 
#	  #  #	   ##  
##### ####     ##

import numpy as np


def dfn(path, num):
	return path.format(str(num).zfill(6))

####  add derives cells
def _N(field, data):
	return np.log(data['chi'] ** -0.5) 

def _volcell(field, data):
	var = data["dx"] ** 3 * data["chi"] ** (-3 / 2)
	return var
	