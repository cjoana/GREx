/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef USERVARIABLES_HPP
#define USERVARIABLES_HPP

#include "EmptyDiagnosticVariables.hpp"
#include <array>
#include <string>

// assign an enum to each variable
// if restarting from plot files you should put the plot vars here in the order
// in which you added them to the plot files, so this may not match your ACTUAL
// UserVariables.hpp file. For the purposes of this tool, all vars are treated
// as evolution ones since we use the checkpoint restart feature
enum
{
    c_chi,
    c_rho,
    NUM_VARS
};

namespace UserVariables
{
static const std::array<std::string, NUM_VARS> variable_names = {"chi", "rho"};
}

#include "UserVariables.inc.hpp"

#endif /* USERVARIABLES_HPP */
