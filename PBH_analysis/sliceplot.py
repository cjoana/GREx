import numpy as np
import yt
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
from scipy.ndimage import map_coordinates
from scipy.interpolate import interpn, griddata

import sys, getopt


if len(sys.argv) ==2:
    num = sys.argv[1]
    run = 'run0p'
    var = 'drho'
    dirh = "hdf5"
elif len(sys.argv) >2:
    dirh = sys.argv[1]
    run = sys.argv[2]
    num = sys.argv[3]
    var = sys.argv[4]

#print("fn is", fn)
else:
    print('you didnt specified file!')

fn = "./{dir}/{run}_{num}.3d.hdf5".format(dir=dirh, run=str(run),
        num=str(num).zfill(6))

mpl.rcParams.update({'font.size': 10,'font.family':'serif'})
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

mpl.rcParams['legend.edgecolor'] = 'inherit'

mpl.rcParams.update(mpl.rcParamsDefault)

def dens_plot(ax, sdata, mycmap, **kargs):
    p = ax.imshow(sdata, interpolation='spline16', 
            cmap=mycmap, **kargs)
    cbar = fig.colorbar(p,ax=ax, aspect=8, shrink=0.7)

    cbar.ax.tick_params(labelsize=40)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return p, cbar



################################################# 



#prefx = "AH0p"
#num = "006000"

#var = 'drho'


#fn = f"./{prefx}_{num}.3d.hdf5"
outfile = "./plots/{num}_{var}.png".format(num=num, var=var)

print(fn, outfile)


ds = yt.load(fn)
L, _,_ = ds.domain_width

res = 1000j
slc = ds.r[::res, ::res, L/2]
dd = ds.r[::]


if var=="drho":
    X, Y, Z = (dd["x"] - dd["dx"]/2, 
            dd["y"] - dd["dy"]/2,
            dd["z"] - dd["dz"]/2)

    rho = dd['D'] + dd["E"]
    Hmean = dd["K"]**2/24/np.pi
    pdata = rho/Hmean

    idx = np.argmax(pdata)
    zpos = Z[idx]
    dN = np.max(dd["dx"])

    mask = (Z > zpos - dN) & (Z < zpos + dN)

    pos = np.linspace(X.min(), X.max(), 200)
    xi, yi = np.meshgrid(pos, pos)
    zi = np.ones_like(xi) * zpos

    idata = griddata( (X[mask],Y[mask],Z[mask]), pdata[mask], 
            (xi , yi, zi ) , method='nearest' )

    #idata = griddata( (X[mask],Y[mask]), pdata[mask], 
            #        (xi , yi ) )

    sdata = idata

elif var=="W":
    X, Y, Z = (dd["x"] - dd["dx"]/2, 
            dd["y"] - dd["dy"]/2,
            dd["z"] - dd["dz"]/2)


    rho = dd['D'] + dd["E"]
    Hmean = dd["K"]**2/24/np.pi
    pdata = rho/Hmean

    idx = np.argmax(pdata)
    zpos = Z[idx]
    dN = np.max(dd["dx"])

    mask = (Z > zpos - dN) & (Z < zpos + dN)




    W  = dd['W'] 
    pdata = W - 1


    pos = np.linspace(X.min(), X.max(), 200)
    xi, yi = np.meshgrid(pos, pos)
    zi = np.ones_like(xi) * zpos

    idata = griddata( (X[mask],Y[mask],Z[mask]), pdata[mask], 
            (xi , yi, zi ) , method='nearest' )

    #idata = griddata( (X[mask],Y[mask]), pdata[mask], 
            #        (xi , yi ) )

    sdata = idata

else: 
    X, Y, Z = (dd["x"] - dd["dx"]/2, 
            dd["y"] - dd["dy"]/2,
            dd["z"] - dd["dz"]/2)


    rho = dd['D'] + dd["E"]
    Hmean = dd["K"]**2/24/np.pi
    pdata = rho/Hmean

    idx = np.argmax(pdata)
    zpos = Z[idx]
    dN = np.max(dd["dx"])

    mask = (Z > zpos - dN) & (Z < zpos + dN)

    pos = np.linspace(X.min(), X.max(), 200)
    xi, yi = np.meshgrid(pos, pos)
    zi = np.ones_like(xi) * zpos



    v  = dd[var] 
    pdata = v



    idata = griddata( (X[mask],Y[mask],Z[mask]), pdata[mask], 
            (xi , yi, zi ) , method='nearest' )

    #idata = griddata( (X[mask],Y[mask]), pdata[mask], 
            #        (xi , yi ) )

    sdata = idata





mymap1 = mpl.cm.magma
mymap2 = mpl.cm.seismic
mymap3 = mpl.cm.CMRmap_r
mymap4 = mpl.cm.gist_gray_r

fig, axs = plt.subplots(2,2, figsize=(25,25))

ax =axs[0,0]

p1, cb1 = dens_plot(ax, sdata, mymap1)

ax =axs[0,1]
p2, cb2 = dens_plot(ax, sdata, mymap2)

sdata = np.abs(sdata)

ax = axs[1,0]
p3, cb3  = dens_plot(ax, sdata, mymap3 ,
    norm=LogNorm(vmin=sdata.min(), vmax= 10**(np.log10( sdata.max())//1 +1 ) )
    )


ax = axs[1,1]
p4, cb4  = dens_plot(ax, sdata, mymap2,
    norm=LogNorm(vmin=sdata.min(), vmax= 10**(np.log10( sdata.max())//1 +1 ) )
    )

plt.tight_layout()
plt.savefig(str(outfile), dpi=300 )

