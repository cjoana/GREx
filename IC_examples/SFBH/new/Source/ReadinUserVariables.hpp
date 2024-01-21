/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef READINUSERVARIABLES_HPP
#define READINUSERVARIABLES_HPP

// assign an enum to each variable
enum
{
    c_phi_in, // matter field added
    c_Pi_in,  //(minus) conjugate momentum

    NUM_READIN_VARS
};

namespace ReadinUserVariables
{
static constexpr char const *variable_names[NUM_READIN_VARS] = {"phi", "Pi"};
}

#endif /* READINUSERVARIABLES_HPP */
