/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYMATTERFUNCTIONS_HPP
#define MYMATTERFUNCTIONS_HPP

#include "SetPsiAndAij.H"

Real my_phi_function_reh(RealVect loc, const PoissonParameters &a_params,
                         const RealVect &a_dx)
{

    RealVect L = a_params.domainLength;

    Real xx = loc[0] + L[0] / 2;
    Real yy = loc[1] + L[1] / 2;
    Real zz = loc[2] + L[2] / 2;

    // Periodic BCs:
    if (xx < 0)
    {
        xx += L[0];
    }
    if (yy < 0)
    {
        yy += L[1];
    }
    if (zz < 0)
    {
        zz += L[2];
    }

    if (xx > L[0])
    {
        xx += -L[0];
    }
    if (yy > L[1])
    {
        yy += -L[1];
    }
    if (zz > L[2])
    {
        zz += -L[2];
    }

    int x_H = 0;
    int y_H = 0;
    int z_H = 0;
    if (xx > L[0] - a_dx[0] && xx < L[0])
    {
        // xx -= L[0];
        x_H = a_params.lines;
    }
    if (yy > L[1] - a_dx[1] && xx < L[1])
    {
        // yy -= L[1];
        y_H = a_params.lines;
    }
    if (zz > L[2] - a_dx[2] && xx < L[2])
    {
        // zz -= L[2];
        z_H = a_params.lines;
    }

    // https://www.wikiwand.com/en/Trilinear_interpolation#/Method

    int i_L = static_cast<int>(floor(xx / a_params.spacing));
    int i_H = static_cast<int>(ceil(xx / a_params.spacing));
    int j_L = static_cast<int>(floor(yy / a_params.spacing));
    int j_H = static_cast<int>(ceil(yy / a_params.spacing));
    int k_L = static_cast<int>(floor(zz / a_params.spacing));
    int k_H = static_cast<int>(ceil(zz / a_params.spacing));

    Real x0 = i_L * a_params.spacing;
    Real x1 = i_H * a_params.spacing;
    Real y0 = j_L * a_params.spacing;
    Real y1 = j_H * a_params.spacing;
    Real z0 = k_L * a_params.spacing;
    Real z1 = k_H * a_params.spacing;

    Real xd = (xx - x0) / (x1 - x0);
    Real yd = (yy - y0) / (y1 - y0);
    Real zd = (zz - z0) / (z1 - z0);

    if (xx > L[0] - a_dx[0] && xx < L[0])
    {
        i_H += -a_params.lines;
    }
    if (yy > L[1] - a_dx[1] && xx < L[1])
    {
        j_H += -a_params.lines;
    }
    if (zz > L[2] - a_dx[2] && xx < L[2])
    {
        k_H += -a_params.lines;
    }
    Real c000 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_L + k_L);
    Real c100 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_L + k_L);
    Real c010 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_H + k_L);
    Real c110 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_H + k_L);
    Real c001 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_L + k_H);
    Real c101 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_L + k_H);
    Real c011 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_H + k_H);
    Real c111 = *(a_params.input_dphi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_H + k_H);

    // Interpolate in x
    Real c00 = c000 * (1 - xd) + c100 * xd;
    Real c01 = c001 * (1 - xd) + c101 * xd;
    Real c10 = c010 * (1 - xd) + c110 * xd;
    Real c11 = c011 * (1 - xd) + c111 * xd;

    // Interpolate in y
    Real c0 = c00 * (1 - yd) + c10 * yd;
    Real c1 = c01 * (1 - yd) + c11 * yd;

    // Interpolate in z
    Real dphi_value = c0 * (1 - zd) + c1 * zd;

    return a_params.phi_0 + a_params.phi_amplitude * dphi_value;
}

Real my_pi_function_reh(RealVect loc, const PoissonParameters &a_params,
                        const RealVect &a_dx)
{

    RealVect L = a_params.domainLength;

    Real xx = loc[0] + L[0] / 2;
    Real yy = loc[1] + L[1] / 2;
    Real zz = loc[2] + L[2] / 2;

    // Periodic BCs:
    if (xx < 0)
    {
        xx += L[0];
    }
    if (yy < 0)
    {
        yy += L[1];
    }
    if (zz < 0)
    {
        zz += L[2];
    }

    if (xx > L[0])
    {
        xx += -L[0];
    }
    if (yy > L[1])
    {
        yy += -L[1];
    }
    if (zz > L[2])
    {
        zz += -L[2];
    }

    int x_H = 0;
    int y_H = 0;
    int z_H = 0;
    if (xx > L[0] - a_dx[0])
    {
        // xx -= L[0];
        x_H = a_params.lines - 1;
    }
    if (yy > L[1] - a_dx[1])
    {
        // yy -= L[1];
        y_H = a_params.lines - 1;
    }
    if (zz > L[2] - a_dx[2])
    {
        // zz -= L[2];
        z_H = a_params.lines - 1;
    }

    // https://www.wikiwand.com/en/Trilinear_interpolation#/Method

    int i_L = static_cast<int>(floor(xx / a_params.spacing));
    int i_H = static_cast<int>(ceil(xx / a_params.spacing));
    int j_L = static_cast<int>(floor(yy / a_params.spacing));
    int j_H = static_cast<int>(ceil(yy / a_params.spacing));
    int k_L = static_cast<int>(floor(zz / a_params.spacing));
    int k_H = static_cast<int>(ceil(zz / a_params.spacing));

    Real x0 = i_L * a_params.spacing;
    Real x1 = i_H * a_params.spacing;
    Real y0 = j_L * a_params.spacing;
    Real y1 = j_H * a_params.spacing;
    Real z0 = k_L * a_params.spacing;
    Real z1 = k_H * a_params.spacing;

    Real xd = (xx - x0) / (x1 - x0);
    Real yd = (yy - y0) / (y1 - y0);
    Real zd = (zz - z0) / (z1 - z0);

    if (xx > L[0] - a_dx[0] && xx < L[0])
    {
        i_H += -a_params.lines;
    }
    if (yy > L[1] - a_dx[1] && xx < L[1])
    {
        j_H += -a_params.lines;
    }
    if (zz > L[2] - a_dx[2] && xx < L[2])
    {
        k_H += -a_params.lines;
    }
    Real c000 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_L + k_L);
    Real c100 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_L + k_L);
    Real c010 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_H + k_L);
    Real c110 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_H + k_L);
    Real c001 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_L + k_H);
    Real c101 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_L + k_H);
    Real c011 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_L +
                  a_params.lines * j_H + k_H);
    Real c111 = *(a_params.input_dpi + a_params.lines * a_params.lines * i_H +
                  a_params.lines * j_H + k_H);

    // Interpolate in x
    Real c00 = c000 * (1 - xd) + c100 * xd;
    Real c01 = c001 * (1 - xd) + c101 * xd;
    Real c10 = c010 * (1 - xd) + c110 * xd;
    Real c11 = c011 * (1 - xd) + c111 * xd;

    // Interpolate in y
    Real c0 = c00 * (1 - yd) + c10 * yd;
    Real c1 = c01 * (1 - yd) + c11 * yd;

    // Interpolate in z
    Real dpi_value = c0 * (1 - zd) + c1 * zd;

    return a_params.pi_0 + a_params.pi_amplitude * dpi_value;
}

Real my_phi_function(const RealVect loc, const PoissonParameters &a_params,
                     const RealVect &a_dx)
{

    RealVect L = a_params.domainLength;

    // gaussian - use with Dirichlet BCs
    if (!a_params.periodic_directions_exist)
    {
        Real rr =
            sqrt(D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]));
        Real rrho = sqrt(loc[0] * loc[0] + loc[1] * loc[1]);
        Real cosph = loc[0] / rrho;
        Real sinph = loc[1] / rrho;
        Real costh = loc[2] / rr;
        Real sinth = rrho / rr;
        Real phi = atan2(loc[1], loc[0]);
        Real sigmoid = rr * rr *
                       exp(-0.5 * pow(rr / a_params.phi_wavelength,
                                      2.0)); // 0.5 * (1.0 - tanh((rr -
                                             // a_params.phi_wavelength)/10.));
        return a_params.phi_0 +
               a_params.phi_amplitude *
                   sin(6 * cos(0.15 * rrho) - a_params.n_swirl_phi * phi) *
                   sigmoid; // * loc[0] / rrho;
    }
    else
    {
        return my_phi_function_reh(loc, a_params, a_dx);
    }
}

Real my_Pi_function(const RealVect loc, const PoissonParameters &a_params,
                    const RealVect &a_dx)
{

    RealVect L = a_params.domainLength;

    // gaussian - use with Dirichlet BCs
    if (!a_params.periodic_directions_exist)
    {
        Real rr =
            sqrt(D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]));
        Real rrho = sqrt(loc[0] * loc[0] + loc[1] * loc[1]);
        Real cosph = loc[0] / rrho;
        Real sinph = loc[1] / rrho;
        Real costh = loc[2] / rr;
        Real sinth = rrho / rr;
        Real phi = atan2(loc[1], loc[0]);
        Real sigmoid =
            rr * rr * exp(-0.5 * pow(rr / a_params.pi_wavelength, 2.0));
        return a_params.pi_0 +
               a_params.pi_amplitude *

                   cos(6 * cos(0.15 * rrho) - a_params.n_swirl_pi * phi) *
                   sigmoid;
    }
    else
    {
        return my_pi_function_reh(loc, a_params, a_dx);
    }
}

Real my_potential_function(const Real &phi_here,
                           const PoissonParameters &a_params)
{
    return 0.5 * pow(a_params.pot_Lambda, 2.0) * pow(phi_here, 2.0);
}

Real my_rho_function(const IntVect a_iv, const RealVect &a_dx,
                     FArrayBox &a_multigrid_vars_box,
                     const PoissonParameters &a_params)
{

    RealVect loc;
    get_loc(loc, a_iv, a_dx, a_params);

    Real psi_reg = a_multigrid_vars_box(a_iv, c_psi_reg);
    Real psi_bh = set_binary_bh_psi(loc, a_params);
    Real psi_0 = psi_reg + psi_bh;

    Real V_of_phi =
        my_potential_function(a_multigrid_vars_box(a_iv, c_phi_0), a_params);
    Real Pi_0 = a_multigrid_vars_box(a_iv, c_Pi_0);
    Tensor<1, Real, SpaceDim> d1_phi =
        get_d1(a_iv, a_multigrid_vars_box, a_dx, c_phi_0);
    Real d1_phi_squared = 0;
    FOR1(i) { d1_phi_squared += d1_phi[i] * d1_phi[i]; }

    return 0.5 * Pi_0 * Pi_0 + V_of_phi +
           0.5 * d1_phi_squared * pow(psi_0, -4.0);
}

Real my_Si_function(Tensor<1, Real> &Si, const IntVect a_iv,
                    const RealVect &a_dx, FArrayBox &a_multigrid_vars_box,
                    const PoissonParameters &a_params)
{

    RealVect loc;
    get_loc(loc, a_iv, a_dx, a_params);

    Real Pi_0 = a_multigrid_vars_box(a_iv, c_Pi_0);
    Tensor<1, Real, SpaceDim> d1_phi =
        get_d1(a_iv, a_multigrid_vars_box, a_dx, c_phi_0);

    FOR1(i) { Si[i] = -Pi_0 * d1_phi[i]; }
}
#endif /* MYMATTERFUNCTIONS_HPP */
