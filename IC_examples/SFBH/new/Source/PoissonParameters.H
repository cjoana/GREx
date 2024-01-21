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
#include "BoundaryConditions.hpp"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "GRParmParse.hpp"
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
    Real regrid_radius;
    int coefficient_average_type;
    int verbosity;
    Vector<int> periodic;
    bool periodic_directions_exist;
    int domBcType;
    int maxLevel;
    int numLevels;
    int sign_of_K;
    Vector<int> refRatio;
    ProblemDomain coarsestDomain;
    Real coarsestDx;
    std::array<double, SpaceDim> center;          // grid center
    BoundaryConditions::params_t boundary_params; // set boundaries in each dir
    RealVect domainLength;
    RealVect probLo;
    RealVect probHi;
    Real alpha;
    Real beta;
    Real G_Newton;
    Real phi_0;
    Real phi_amplitude;
    Real phi_wavelength;
    Real pi_0;
    Real pi_amplitude;
    Real pi_wavelength;
    Real n_swirl_phi;
    Real n_swirl_pi;
    Real bh1_bare_mass;
    Real bh2_bare_mass;
    RealVect bh1_spin;
    RealVect bh2_spin;
    RealVect bh1_momentum;
    RealVect bh2_momentum;
    RealVect bh1_offset;
    RealVect bh2_offset;
    Real pot_Lambda;
    bool include_A2;
    bool method_compact;
    bool deactivate_zero_mode;
    Real psi_reg;
    Real tolerance;
    bool readin_matter_data;
    std::string input_filename;
    std::string output_filename;
    std::string output_path;
    std::string pout_filename;
    std::string pout_path;
    std::string read_from_data_dphi;
    std::string read_from_data_dpi;
    Real *input_dphi;
    Real *input_dpi;
    int lines;
    Real spacing;
};

/// get stuff from input file
extern void getPoissonParameters(PoissonParameters &a_params);

#endif
