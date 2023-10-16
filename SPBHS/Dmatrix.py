
# Method based on L. N. Trefethen,Spectral  Methods  in  MATLAB(SIAM,2000) and http://blue.math.buffalo.edu/438/trefethen_spectral/all_py_files/
import numpy as np
import math
pi = math.pi
#It builds the Chebyshev grid and a differentiation matrix in a general domain (a, b)
def chebymatrix(Ncheb,a,b):

	range_cheb = np.arange(0,Ncheb+1)
	x = np.cos(pi*range_cheb/Ncheb)
	t = (a+b)/2.-((a-b)/2.)*x
	carray = np.hstack([2, np.ones(Ncheb-1), 2])*(-1)**np.arange(0,Ncheb+1)
	X = np.tile(x,(Ncheb+1,1))
	dX = X.T - X
	Dp = (carray[:,np.newaxis]*(1.0/carray)[np.newaxis,:])/(dX+(np.identity(Ncheb+1)))       
	Dp = Dp - np.diag(Dp.sum(axis=1))            
	Dcheb =(2./(b-a))*Dp

	return Dcheb, t
