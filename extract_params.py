import yt
import numpy as np
from sklearn.mixture import GaussianMixture
# import scikit

dfile = "GMM/init.3d.hdf5"
ds = yt.load(dfile)

dd = ds.all_data()

x, y, z = [dd['x'], dd['y'], dd['z']]
chi = dd['chi']
psi = chi**(-1/4)

sumGauss = 2 * np.log(psi)

factor = 1./np.min(sumGauss)
f2 = 1

vals = np.array(sumGauss * factor *f2, dtype=int) -f2

# print( np.unique(vals, return_counts=True))

data = []
for i, v in enumerate(x):
    # if (i%2==0): continue
    # print(vals[i])
    for times in range(vals[i]):
        data.append([x[i], y[i], z[i]])

data = np.array(data)


# print( np.unique(data, return_counts=True))
# print(data)


print("\n Starting GMM ")

n_comp = 3
gm = GaussianMixture(n_components=n_comp, covariance_type='spherical',
                      random_state=0).fit(data)

for i in range(n_comp):

    print(" mean, var,  weight:  " , gm.means_[i], gm.covariances_[i], gm.weights_[i]    )


