/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MULTIGRIDUSERVARIABLES_HPP
#define MULTIGRIDUSERVARIABLES_HPP

// assign an enum to each variable
enum
{
    c_psi,

    c_A11_0,
    c_A12_0,
    c_A13_0,
    c_A22_0,
    c_A23_0,
    c_A33_0,

    c_phi_0, // matter field
    c_pi_0, // matter field
    c_rho_0, // matter field

    NUM_MULTIGRID_VARS
};

namespace MultigridUserVariables
{
static constexpr char const *variable_names[NUM_MULTIGRID_VARS] = {
    "psi",

    "A11_0", "A12_0", "A13_0", "A22_0", "A23_0", "A33_0",

    "phi_0", "pi_0", "rho_0"};
}

#endif /* MULTIGRIDUSERVARIABLES_HPP */
