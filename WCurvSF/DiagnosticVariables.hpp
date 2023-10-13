/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef DIAGNOSTICVARIABLES_HPP
#define DIAGNOSTICVARIABLES_HPP

// assign an enum to each variable
enum
{
    c_Ham,

    c_Mom,

    c_Ham_abs_terms,

    c_Mom_abs_terms,

    c_trA2, c_ricci_scalar,

    c_rho, c_S,

    c_ricci_scalar_tilde,

    c_Weyl_curv, 
    c_ChP_inv, 

    c_Weyl4_Re, c_Weyl4_Im,

    NUM_DIAGNOSTIC_VARS
};

namespace DiagnosticVariables
{
static const std::array<std::string, NUM_DIAGNOSTIC_VARS> variable_names = {
    "Ham",

    "Mom",

    "Ham_abs_terms",

    "Mom_abs_terms",

    "trA2", "ricci_scalar",

    "rho", "S",

    "ricci_scalar_tilde", 

    "Weyl_curv", "ChP_inv",

    "Weyl4_Re", "Weyl4_Im"

  };
}
#endif /* DIAGNOSTICVARIABLES_HPP */
