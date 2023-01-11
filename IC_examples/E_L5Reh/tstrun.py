#!/usr/bin/python

import numpy as np
import yt 
import sys, getopt

#for opt, arg in opts:
#    if opt == "--file":
#        print(arg) 


if len(sys.argv) >1:
    num = sys.argv[1]
    fn = "./vcPoissonOut.3d_{num}.hdf5".format(num=str(num))
    #print("fn is", fn)

    ds = yt.load(fn)
    dd = ds.all_data()


    chi = dd["psi"]
    #Pi2 = dd["pii2_0"]**2
    #Pi = dd["pi_0"]**2
    phi = dd["phi_0"]
    phi2 = dd["phi2_0"]
    rho = dd["rho_0"]

    print("rho = ", np.mean(rho), np.min(rho), np.max(rho))
    print("psi = ", np.mean(chi) , np.min(chi), np.max(chi))

    #print("Pis^2 = ", np.max(Pi), np.max(Pi2))
    print("sfs = ", np.max(phi), np.max(phi2), " means :", np.mean(phi), np.mean(phi2))

    #raise
else: 
    fn = "./InitialConditionsFinal.3d.hdf5"

    print("fn is", fn)

    ds = yt.load(fn)
    dd = ds.all_data()

    #Ham = dd["Ham_abs_terms"]
    #rho = dd["rho"]
    #S = dd["S"]

    chi = dd["chi"]
    N = -0.5*np.log(chi)
    #Pi2 = dd["Pi2"]**2
    #Pi = dd["Pi"]**2
    phi = dd["phi"]
    phi2 = dd["phi2"]


    #print("Ham :", np.mean(np.abs(Ham)), np.max(Ham), np.min(Ham) )
    #print("w = ", np.mean(S/rho/3))
    #print("rho = ", np.mean(rho), np.min(rho), np.max(rho))
    print("chi = ", np.mean(chi) , np.min(chi), np.max(chi))

    #print("Pis^2 = ", np.max(Pi), np.max(Pi2))
    print("sfs = ", np.max(phi), np.max(phi2), " means :", np.mean(phi), np.mean(phi2)) 


