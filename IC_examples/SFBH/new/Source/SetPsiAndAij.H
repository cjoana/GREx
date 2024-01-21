/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SETPSIANDAIJ_HPP
#define SETPSIANDAIJ_HPP

#include "PoissonParameters.H"
#include "SetLevelDataGradients.H"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"

// Get the bh radius
void get_bh_coords(Real &bh_radius, RealVect &loc_bh, const RealVect &loc,
                   const RealVect &bh_offset)
{
    // set coords
    loc_bh = loc - bh_offset;

    // set radius
    Real bh_radius_squared = 0.0;
    FOR1(i) { bh_radius_squared += loc_bh[i] * loc_bh[i]; }
    bh_radius = sqrt(bh_radius_squared);
}

// The Brill Lindquist part of psi
Real set_binary_bh_psi(const RealVect &loc, const PoissonParameters &a_params)
{
    // the Bowen York params
    Real m1 = a_params.bh1_bare_mass;
    Real m2 = a_params.bh2_bare_mass;

    // set the BH values - location
    RealVect loc_bh1;
    Real rbh1;
    get_bh_coords(rbh1, loc_bh1, loc, a_params.bh1_offset);

    RealVect loc_bh2;
    Real rbh2;
    get_bh_coords(rbh2, loc_bh2, loc, a_params.bh2_offset);

    return 0.5 * (m1 / rbh1 + m2 / rbh2);
}

// Set Aij Bowen York data
// see Alcubierre pg 110 eqn (3.4.22)
void set_Aij_bh(Tensor<2, Real> &Aij, // const IntVect &iv,
                const RealVect &loc, const PoissonParameters &a_params)
{
    // set the BH values - location
    RealVect loc_bh1;
    Real rbh1;
    get_bh_coords(rbh1, loc_bh1, loc, a_params.bh1_offset);

    RealVect loc_bh2;
    Real rbh2;
    get_bh_coords(rbh2, loc_bh2, loc, a_params.bh2_offset);

    RealVect n1 = {loc_bh1[0] / rbh1, loc_bh1[1] / rbh1, loc_bh1[2] / rbh1};
    RealVect n2 = {loc_bh2[0] / rbh2, loc_bh2[1] / rbh2, loc_bh2[2] / rbh2};

    // the Bowen York params
    RealVect J1 = a_params.bh1_spin;
    RealVect J2 = a_params.bh2_spin;
    RealVect P1 = a_params.bh1_momentum;
    RealVect P2 = a_params.bh2_momentum;

    using namespace TensorAlgebra;
    Tensor<3, Real> epsilon = TensorAlgebra::epsilon();

    FOR2(i, j)
    {
        Aij[i][j] = 1.5 / rbh1 / rbh1 * (n1[i] * P1[j] + n1[j] * P1[i]) +
                    1.5 / rbh2 / rbh2 * (n2[i] * P2[j] + n2[j] * P2[i]);

        FOR1(k)
        {
            Aij[i][j] += 1.5 / rbh1 / rbh1 * (n1[i] * n1[j] - delta(i, j)) *
                             P1[k] * n1[k] +
                         1.5 / rbh2 / rbh2 * (n2[i] * n2[j] - delta(i, j)) *
                             P2[k] * n2[k];

            FOR1(l)
            {
                Aij[i][j] +=
                    -3.0 / rbh1 / rbh1 / rbh1 *
                        (epsilon[i][l][k] * n1[j] + epsilon[j][l][k] * n1[i]) *
                        n1[l] * J1[k] -
                    3.0 / rbh2 / rbh2 / rbh2 *
                        (epsilon[i][l][k] * n2[j] + epsilon[j][l][k] * n2[i]) *
                        n2[l] * J2[k];
            }
        }
    }
}

// The part of Aij excluding the Brill Lindquist BH Aij
// Using ansatz in B&S Appendix B Eq B.5
void set_Aij_reg(Tensor<2, Real> &Aij, const FArrayBox &multigrid_vars_box,
                 const IntVect &iv, const RealVect &a_dx,
                 const PoissonParameters &a_params)
{

    // work out location on the grid
    RealVect loc;
    get_loc(loc, iv, a_dx, a_params);

    // get the derivs
    Tensor<2, Real, SpaceDim> d2_U =
        get_d2(iv, multigrid_vars_box, a_dx, c_U_0);
    Tensor<1, Real, SpaceDim> d1_V1 =
        get_d1(iv, multigrid_vars_box, a_dx, c_V1_0);
    Tensor<1, Real, SpaceDim> d1_V2 =
        get_d1(iv, multigrid_vars_box, a_dx, c_V2_0);
    Tensor<1, Real, SpaceDim> d1_V3 =
        get_d1(iv, multigrid_vars_box, a_dx, c_V3_0);
    Tensor<2, Real, SpaceDim> d2_V1 =
        get_d2(iv, multigrid_vars_box, a_dx, c_V1_0);
    Tensor<2, Real, SpaceDim> d2_V2 =
        get_d2(iv, multigrid_vars_box, a_dx, c_V2_0);
    Tensor<2, Real, SpaceDim> d2_V3 =
        get_d2(iv, multigrid_vars_box, a_dx, c_V3_0);

    // note derivative index j is the last one
    using namespace TensorAlgebra;
    Tensor<2, Real, SpaceDim> d1_Vi;
    FOR2(i, j)
    {
        d1_Vi[i][j] = delta(i, 0) * d1_V1[j] + delta(i, 1) * d1_V2[j] +
                      delta(i, 2) * d1_V3[j];
    }

    // Periodic: Use ansatz B.3 in B&S (p547) JCA TODO: We are not using this U
    // when constructing Aij. Non-periodic: Compact ansatz B.7 in B&S (p547)
    Real trace = 0.0;
    if (!a_params.method_compact)
    {
        FOR1(i)
        {
            trace += d1_Vi[i][i];
            //+ d2_U[i][i];
        }

        // set the values of Aij
        FOR2(i, j)
        {
            Aij[i][j] = 0.75 * d1_Vi[i][j]   //+ d2_U[i][j]
                        + 0.75 * d1_Vi[j][i] //+ d2_U[j][i]
                        //- 2.0 / 3.0 * delta(i,j) * trace;
                        - 0.5 * delta(i, j) * trace;
        }
    }
    else
    {
        FOR1(i)
        {
            trace += 0.75 * d1_Vi[i][i] -
                     0.125 * (d2_U[i][i] + loc[0] * d2_V1[i][i] +
                              loc[1] * d2_V2[i][i] + loc[2] * d2_V3[i][i]);
        }
        // set the values of Aij
        FOR2(i, j)
        {
            Aij[i][j] = 0.75 * d1_Vi[i][j] - 0.125 * d2_U[i][j] +
                        0.75 * d1_Vi[j][i] - 0.125 * d2_U[j][i] -
                        0.125 * (loc[0] * (d2_V1[i][j] + d2_V1[j][i]) +
                                 loc[1] * (d2_V2[i][j] + d2_V2[j][i]) +
                                 loc[2] * (d2_V3[i][j] + d2_V3[j][i])) -
                        2.0 / 3.0 * delta(i, j) * trace;
        }
    }
}

#endif /* SETPSIANDAIJ_HPP */
