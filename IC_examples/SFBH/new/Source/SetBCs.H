#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _SETBCS_H_
#define _SETBCS_H_

#include <iostream>
using std::cerr;

#include "AMRMultiGrid.H"
#include "BoundaryConditions.hpp"
#include "DimensionDefinitions.hpp"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "MultigridUserVariables.hpp"
#include "ParmParse.H"
#include "VariableCoeffPoissonOperator.H"

// BC class for the linear solver steps
class GlobalBCRS
{
  public:
    static BoundaryConditions::params_t
        s_boundary_params; // set boundaries in each dir
    static bool s_areBCsParsed;
};

// This is the function that the solver looks for to apply the BCs to the
// variables in a_state
extern void ParseBC(FArrayBox &a_state, const Box &a_valid,
                    const ProblemDomain &a_domain, Real a_dx,
                    bool a_homogeneous);

#endif
