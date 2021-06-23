#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _POISSONPARAMETERS_H_
#define _POISSONPARAMETERS_H_

#include <iostream>
using std::cerr;

#include "AMRMultiGrid.H"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "ParmParse.H"
#include "VariableCoeffPoissonOperator.H"

/// Class for solver input parameters
/**
 */
class PoissonParameters
{
  public:
    IntVect nCells;
    int maxGridSize;
    int blockFactor;
    int bufferSize;
    Real fillRatio;
    Real refineThresh;
    int coefficient_average_type;
    int verbosity;
    Vector<int> periodic;
    int domBcType;
    int maxLevel;
    int numLevels;
    Vector<int> refRatio;
    ProblemDomain coarsestDomain;
    Real coarsestDx;
    RealVect domainLength;
    RealVect probLo;
    RealVect probHi;
    Real alpha;
    Real beta;
    Real G_Newton;
    Real phi_background;  // CJ added
    Real phi_amplitude;   // CJ added
    Real phi_wavelength;   // CJ added
    Real pi_background;  // CJ added
    Real pi_amplitude;   // CJ added
    Real pi_wavelength;   // CJ added
    Real bh1_bare_mass;
    Real bh2_bare_mass;
    Real bh1_spin;
    Real bh2_spin;
    Real bh1_momentum;
    Real bh2_momentum;
    Real bh1_offset;
    Real bh2_offset;
    Real g_coupling;
    Real sf1_a_coupling;
    Real sf1_b_coupling;
    Real sf2_a_coupling;
    Real sf2_b_coupling;
    
    
};

/// get stuff from input file
extern void getPoissonParameters(PoissonParameters &a_params);

#endif
