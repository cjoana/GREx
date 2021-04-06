import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import units as nu
from math import pi
import h5py as h5
# from polytropes import monotrope, polytrope
# from crust import SLyCrust
# from eoslib import get_eos, glue_crust_and_core, eosLib
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
# from label_line import label_line

import math

# from matplotlib import cm
#import palettable as pal

#cmap = pal.colorbrewer.qualitative.Set1_6.mpl_colormap


# cmap = pal.cmocean.sequential.Matter_8.mpl_colormap #best so far
# cmap = pal.wesanderson.Zissou_5.mpl_colormap

# --------------------------------------------------

class eos:
    # monotropic EoS
    # transition continuity constant
    a = 0.0
    c2 = nu.c**2
    min_press = 1e-12

    def __init__(self, K, G):
        self.K = K / self.c2
        self.G = G
        self.n = 1.0 / (G - 1)

    # pressure P(density)
    def pressure_from_density(self, density):
        return self.c2 * self.K * density ** self.G

    # energy density_adm = density * (1 + energy)
    def rho_from_density(self, density):
        # return (1.0 + self.a) * density + (self.K / (self.G - 1)) * density ** self.G
        return density + (self.K / (self.G - 1)) * density ** self.G

    # for inverse functions lets define  density (P)
    def density_from_pressure(self, pressure):
        if pressure < self.min_press:
            return self.min_press
        return (pressure / self.c2 / self.K) ** (1 / self.G)

    def densities_from_pressure(self, pressure):
        mask = np.array(pressure < self.min_press, dtype=bool)
        out = (pressure / self.c2 / self.K) ** (1 / self.G)
        out[mask] = self.min_press
        return out


class tov:

    def __init__(self, peos, r0=1e-8, rf=1e10, verbose=0):
        self.physical_eos = peos
        self.r0 = r0
        self.rf = rf
        self.verbose = verbose


    def tov(self, y, r):
        P, mass, lna = y
        rho = self.physical_eos.density_from_pressure(P)

        dPdr = -nu.G * (rho + P / nu.c ** 2) * (mass + 4.0 * pi * r ** 3 * P / nu.c ** 2)
        dPdr = dPdr / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
        dmdr = 4.0 * pi * r ** 2 * rho

        dlnadr = (mass + 4*pi*r**3*P)/ (r * (r - 2.0 * nu.G * mass / nu.c ** 2))

        return [dPdr, dmdr, dlnadr]

    def tov_ivp(self, r, y):
        P, mass, lna = y
        rho = self.physical_eos.density_from_pressure(P)

        dPdr = -nu.G * (rho + P / nu.c ** 2) * (mass + 4.0 * pi * r ** 3 * P / nu.c ** 2)
        dPdr = dPdr / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
        dmdr = 4.0 * pi * r ** 2 * rho

        dlnadr = (mass + 4*pi*r**3*P)/ (r * (r - 2.0 * nu.G * mass / nu.c ** 2))

        return [dPdr, dmdr, dlnadr]

    def ode_radius(self, r, y):
        lapse0, psi0 = y
        P = self.pressure_from_radius(r)
        mass = self.mass_from_radius(r)
        dlnadr = (mass + 4 * pi * r ** 3 * P) / (r * (r - 2.0 * nu.G * mass / nu.c ** 2))
        lamb = (r**0.5 - (r - 2.0 * nu.G * mass / nu.c ** 2)**0.5) / (r * (r - 2.0 * nu.G * mass / nu.c ** 2)**0.5)

        # print(dlnadr)

        return [dlnadr, lamb]

    def tovsolve(self, density_central):

        N = 1e4
        r0, rf = [self.r0, self.rf]
        r = np.linspace(r0, rf, int(N))
        P_0 = self.physical_eos.pressure_from_density(density_central)
        rho_0 = self.physical_eos.density_from_pressure(P_0)
        mass_0 = 4.0 * pi * r[0] ** 3 * rho_0
        lna_0 = 1

        # intmethod = "odeint"
        intmethod = "ivp"  # does not work for sys of diff.eqs ???

        if intmethod == "odeint":
            psol = odeint(self.tov, [P_0, mass_0, lna_0], r, t)  # , rtol=1.0e-4, atol=1.0e-4)
            press = psol[:, 0]
            mass = psol[:, 1]
            lna_old = psol[:, 2]
        elif intmethod == "ivp":
            sol = solve_ivp(self.tov_ivp, [r0, rf], [P_0, mass_0, lna_0])
            r = sol.t
            rrange = 10.0 ** np.linspace(np.log10(r[0]), np.log10(r[-1]), int(N))
            sol = solve_ivp(self.tov_ivp, [r0, rf], [P_0, mass_0, lna_0], t_eval=rrange)
            r = sol.t
            press = sol.y[0]
            mass = sol.y[1]
            lna_old = sol.y[2]
        else:
            return "error"

        return r, press, mass, lna_old

    def mass_radius(self): #TODO: not useful
        N = 10
        mcurve = np.zeros(N)
        rcurve = np.zeros(N)
        rhocs = np.logspace(14.0, 16.0, N)

        print(rhocs)

        mass_max = 0.0
        j = 0
        for rhoc in rhocs:
            rad, press, mass, lna = self.tovsolve(rhoc)

            # rad /= 1.0e5  # cm to km
            # mass /= nu.Msun

            mstar = mass[-1]
            rstar = rad[-1]
            atm_indx = []
            for i, p in enumerate(press):
                if p > 0.0:
                    mstar = mass[i]
                    rstar = rad[i]
                else:
                    atm_indx.append(i)
            mcurve[j] = mstar
            rcurve[j] = rstar

            j += 1
            if mass_max < mstar:
                mass_max = mstar
            else:
                pass
                # break

        return mcurve[:j], rcurve[:j], rhocs[:j]

    def solve_with_atm(self, density_central):

        rad, press, mass, lna = self.tovsolve(density_central)

        atm_indx = []
        p_min = np.max(press)
        for i, p in enumerate(press):
            if p > 0.0:
                if self.verbose: print(p)
                p_min = p if p < p_min else p_min
            else:
                if self.verbose: print("! ", p)
                atm_indx.append(i)

        # add atmosphere
        # press_atm = p_min * 0.1 if p_min * 0.1 >= 1e-8 else 1e-8
        press[atm_indx] = p_min * 0.1
        print("p d atm", p_min * 0.1)
        # def radius and mass of star
        star_lim = atm_indx[0]-1
        rad_star = rad[star_lim]
        mass_star = mass[star_lim+1]

        density = physical_eos.densities_from_pressure(press)
        rho = physical_eos.rho_from_density(density)
        energy = (K * density ** (G - 1)) / (G - 1)
        chi = (1 - 2 * mass / rad)   # gamma_rr **-1

        # # chi_old = (1 - 2*mass/rad)
        # # chi = np.exp(lnpsi)**2
        # # chi = chi_old
        #
        # pressure_from_radius = interp1d(rad, press, bounds_error=False, fill_value='extrapolate')
        # mass_from_radius = interp1d(rad, mass, bounds_error=False, fill_value='extrapolate')
        # self.pressure_from_radius = pressure_from_radius
        # self.mass_from_radius = mass_from_radius
        #
        # lna_0 =  np.log(1 - 2 * mass_star/rad[-1])
        # lamb_0 = - 0.5*lna_0
        #
        # r0, rf = [self.r0, self.rf]
        # sol = solve_ivp(self.ode_radius, [rf, r0], [lna_0, -lamb_0], t_eval=rad[::-1])
        # lna = sol.y[0][::-1]
        # Lamb = - sol.y[1][::-1]  # rh = r * exp(- Lamb) ;; chi = exp(-2*Lamb)
        #
        # # lna = -2*Lamb
        #
        # print(np.exp(lna_0),  np.exp(lna))
        #
        # if not np.all(sol.t[::-1] == rad):
        #     print(sol.t)
        #     print(rad)
        #     raise ValueError("radius do not match")
        #
        # # lna = lna_old
        #
        # # chi = np.exp(-2 * Lamb)
        # chi = (1 - 2 * mass/rad)
        # rad_iso = rad * np.exp(- Lamb)
        # rad = rad_iso  ## convert to isotropic cordinates,  radius


        out = dict()
        out['pressure'] = press
        out['density'] = density
        out['energy'] = energy
        out['lapse'] = np.exp(lna)
        out['rho'] = rho
        out['chi'] = chi
        out['mass'] = mass
        out['rad'] = rad

        other = dict()
        other['rad_star'] = rad_star
        other['mass_star'] = mass_star

        return out, other

###########################################################

if __name__ == "__main__":
    # main(sys.argv)
    # plt.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.95, wspace=0.1, hspace=0.1)
    # plt.savefig('mr.pdf')

    plt.rcParams.update({'font.size': 22})

    K = 100
    # G = 2
    # omega = G - 1
    omega = 0.33333333333333333
    G = omega + 1
    density_central = 0.00128

    create_data = True
    #create_data = False

    physical_eos = eos(K, G)
    # eos = glue_crust_and_core(SLyCrust, dense_eos)
    t = tov(physical_eos, r0=1e-2, rf=1e15)

    out, other = t.solve_with_atm(density_central)

    rad_star = other['rad_star']
    putL = int(np.log10(rad_star))+1

    print("R star less than", 10**putL, "  app.", other['rad_star'])
    print("M star is  ", other['mass_star'])

    nplots = 7
    fig, axs = plt.subplots(nplots, 1, figsize=(20,15))

    keys = list(out.keys())
    keys.remove('rad')

    for i in range(nplots):
        var = keys[i]

        init = np.where( 0.01 <=  out['rad'])[0][0]
        end = np.where(out['rad'] >= 10**putL)[0][0]

        axs[i].plot(out['rad'][init:end], out[var][init:end], label=var, lw =4 )
        # axs[i].plot(out['rad'], out[var], label=var)


    for ax in axs[:]:
        ax.set_yscale('log')
        ax.set_xscale('log')
        xmin = out['rad'][0] if  out['rad'][0] > 0.01 else 0.01
        # ax.set_xlim( xmin, 10**putL)
        ax.set_xlim( 0.01, 10**putL)
        ax.legend()

    plt.tight_layout()
    plt.savefig("idata.png")

    # print(out['lapse']**2  * out['chi']**-1)


    ##### Create data
    if create_data:

        N = 128
        L = 10**putL
        dt_multiplier = 0.0001
        periodic_BCs = True

        print("creating data with domain: L = {},  N = {}".format(L, N))

        radius = out["rad"]
        f_density = interp1d(radius, out['density'], bounds_error=False, fill_value='extrapolate')
        f_energy = interp1d(radius, out['energy'], bounds_error=False, fill_value='extrapolate')
        f_pressure = interp1d(radius, out['pressure'], bounds_error=False, fill_value='extrapolate')
        f_chi = interp1d(radius, out['chi'], bounds_error=False, fill_value='extrapolate')
        f_lapse = interp1d(radius, out['lapse'], bounds_error=False, fill_value='extrapolate')

        # HDF5 file
        save_data = True
        path = "./"
        filename = "init_tov.hdf5"  # Name of the new file to create
        component_names = [  # The order is important: component_0 ... component_(nth-1)
            "chi",

            "h11", "h12", "h13", "h22", "h23", "h33",

            "K",

            "A11", "A12", "A13", "A22", "A23", "A33",

            "Theta",

            "Gamma1", "Gamma2", "Gamma3",

            "lapse",

            "shift1", "shift2", "shift3",

            "B1", "B2", "B3",

            "density", "energy", "pressure", "enthalpy",

            "D", "E", "W",

            "Z1", "Z2", "Z3",

            "V1", "V2", "V3",

        ]

        temp_comp = np.zeros((N, N, N))  # template for components: array [Nx, Ny. Nz]
        dset = dict()
        dset['chi'] = temp_comp.copy() + 1.
        dset['Ham'] = temp_comp.copy()
        dset['h11'] = temp_comp.copy() + 1.
        dset['h22'] = temp_comp.copy() + 1.
        dset['h33'] = temp_comp.copy() + 1.
        dset['lapse'] = temp_comp.copy() + 1.

        dset['D'] = temp_comp.copy()
        dset['E'] = temp_comp.copy()
        dset['density'] = temp_comp.copy()
        dset['energy'] = temp_comp.copy()
        dset['pressure'] = temp_comp.copy()
        dset['enthalpy'] = temp_comp.copy()
        dset['V1'] = temp_comp.copy()
        dset['V2'] = temp_comp.copy()
        dset['V3'] = temp_comp.copy()
        dset['Z1'] = temp_comp.copy()
        dset['Z2'] = temp_comp.copy()
        dset['Z3'] = temp_comp.copy()
        dset['W'] = temp_comp.copy() + 1.
        dset['K'] = temp_comp.copy()

        # ## Constructing variables (example for SF)
        indices = []
        xcord = temp_comp.copy()
        ycord = temp_comp.copy()
        zcord = temp_comp.copy()
        for z in range(N):
            for y in range(N):
                for x in range(N):
                    # wvl = 2 * np.pi * 4 / L
                    ind = x + y * N + z * N ** 2
                    dd = L / N

                    xi, yi, zi = np.array([x,y,z]) * dd + dd/2

                    xcord[x][y][z] = xi
                    ycord[x][y][z] = yi
                    zcord[x][y][z] = zi

                    r = (xi**2 + yi**2 + zi**2)**0.5

                    dset['chi'][x][y][z] = f_chi(r)
                    dset['density'][x][y][z] = f_density(r)
                    dset['pressure'][x][y][z] = f_pressure(r)
                    dset['energy'][x][y][z] = f_energy(r)
                    dset['lapse'][x][y][z] = f_lapse(r)


                    indices.append(ind)

        dset['D'] = dset['density']
        dset['E'] = dset['density'] * dset['energy']
        dset['enthalpy'] = 1 + dset['energy'] + omega
        dset['rho'] = dset['D'] + dset['E']
        dset['S'] = dset['pressure'] * 3
        # dset['W'] = dset['density']/dset['density']   #  1 = 1 / ( 1 - V^2) = u0 * lapse


        ## Save data

        if not save_data:
            print("!!!\n\nYou have chosen not to save the data")
        else:

            if not os.path.exists(path):
                os.mkdir(path)
            print(" ! > new mkdir: ", path)

            """
            Mesh and Other Params
            """
            # def base attributes
            base_attrb = dict()
            base_attrb['time'] = 0.0
            base_attrb['iteration'] = 0
            base_attrb['max_level'] = 0
            base_attrb['num_components'] = len(component_names)
            base_attrb['num_levels'] = 1
            base_attrb['regrid_interval_0'] = 1
            base_attrb['steps_since_regrid_0'] = 0
            for comp, name in enumerate(component_names):
                key = 'component_' + str(comp)
                tt = 'S' + str(len(name))
                base_attrb[key] = np.array(name, dtype=tt)

            # def Chombo_global attributes
            chombogloba_attrb = dict()
            chombogloba_attrb['testReal'] = 0.0
            chombogloba_attrb['SpaceDim'] = 3

            # def level0 attributes
            level_attrb = dict()
            level_attrb['dt'] = float(L) / N * dt_multiplier
            level_attrb['dx'] = float(L) / N
            level_attrb['time'] = 0.0
            if periodic_BCs:
                level_attrb['is_periodic_0'] = 1
                level_attrb['is_periodic_1'] = 1
                level_attrb['is_periodic_2'] = 1
            else:
                level_attrb['is_periodic_0'] = 0
                level_attrb['is_periodic_1'] = 0
                level_attrb['is_periodic_2'] = 0
            level_attrb['ref_ratio'] = 2
            level_attrb['tag_buffer_size'] = 3
            prob_dom = (0, 0, 0, N - 1, N - 1, N - 1)
            prob_dt = np.dtype([('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'),
                                ('hi_i', '<i4'), ('hi_j', '<i4'), ('hi_k', '<i4')])
            level_attrb['prob_domain'] = np.array(prob_dom, dtype=prob_dt)
            boxes = np.array([(0, 0, 0, N - 1, N - 1, N - 1)],
                             dtype=[('lo_i', '<i4'), ('lo_j', '<i4'), ('lo_k', '<i4'), ('hi_i', '<i4'), ('hi_j', '<i4'),
                                    ('hi_k', '<i4')])

            """"
            CREATE HDF5
            """

            # TODO: if overwrite:   [...] else: raise()
            if os.path.exists(filename):
                os.remove(filename)

            h5file = h5.File(filename, 'w')  # New hdf5 file I want to create

            # base attributes
            for key in base_attrb.keys():
                h5file.attrs[key] = base_attrb[key]

            # group: Chombo_global
            chg = h5file.create_group('Chombo_global')
            for key in chombogloba_attrb.keys():
                chg.attrs[key] = chombogloba_attrb[key]

            # group: levels
            l0 = h5file.create_group('level_0')
            for key in level_attrb.keys():
                l0.attrs[key] = level_attrb[key]
            sl0 = l0.create_group('data_attributes')
            dadt = np.dtype([('intvecti', '<i4'), ('intvectj', '<i4'), ('intvectk', '<i4')])
            sl0.attrs['ghost'] = np.array((3, 3, 3), dtype=dadt)
            sl0.attrs['outputGhost'] = np.array((0, 0, 0), dtype=dadt)
            sl0.attrs['comps'] = base_attrb['num_components']
            sl0.attrs['objectType'] = np.array('FArrayBox', dtype='S10')

            # level datasets
            dataset = np.zeros((base_attrb['num_components'], N, N, N))
            for i, comp in enumerate(component_names):
                if comp in dset.keys():
                    dataset[i] = dset[comp].T
            fdset = []
            for c in range(base_attrb['num_components']):
                fc = dataset[c].T.flatten()
                fdset.extend(fc)
            fdset = np.array(fdset)

            l0.create_dataset("Processors", data=np.array([0]))
            l0.create_dataset("boxes", data=boxes)
            l0.create_dataset("data:offsets=0", data=np.array([0, (base_attrb['num_components']) * N ** 3]))
            l0.create_dataset("data:datatype=0", data=fdset)

            h5file.close()

            print("Done!")


