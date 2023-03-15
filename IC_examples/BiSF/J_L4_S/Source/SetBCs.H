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
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "ParmParse.H"
#include "VariableCoeffPoissonOperator.H"

// BC class
class GlobalBCRS
{
  public:
    static std::vector<bool> s_printedThatLo, s_printedThatHi;
    static std::vector<int> s_bcLo, s_bcHi;
    static RealVect s_trigvec;
    static bool s_areBCsParsed, s_valueParsed, s_trigParsed;
};

extern void ParseValue(Real *pos, int *dir, Side::LoHiSide *side,
                       Real *a_values);

extern void ParseBC(FArrayBox &a_state, const Box &a_valid,
                    const ProblemDomain &a_domain, Real a_dx,
                    bool a_homogeneous);

#endif
