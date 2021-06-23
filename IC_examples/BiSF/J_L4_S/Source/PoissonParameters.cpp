#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "PoissonParameters.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

// function to read in the key params for solver
void getPoissonParameters(PoissonParameters &a_params)
{
    ParmParse pp;

    // problem specific params
    pp.get("alpha", a_params.alpha);
    pp.get("beta", a_params.beta);
    // print out the overall coeffs just to be sure we have selected them
    // correctly
    pout() << "alpha, beta = " << a_params.alpha << ", " << a_params.beta
           << endl;

    // Initial conditions for the scalar field
    pp.get("G_Newton", a_params.G_Newton);
    pp.get("phi_amplitude", a_params.phi_amplitude);
    pp.get("phi_wavelength", a_params.phi_wavelength);
    pp.get("phi_background", a_params.phi_background);

    pp.get("pi_amplitude", a_params.pi_amplitude);
    pp.get("pi_wavelength", a_params.pi_wavelength);
    pp.get("pi_background", a_params.pi_background);

    pp.get("g_coupling", a_params.g_coupling);
    pp.get("sf1_a_coupling", a_params.sf1_a_coupling);
    pp.get("sf1_b_coupling", a_params.sf1_b_coupling);
    pp.get("sf2_a_coupling", a_params.sf2_a_coupling);
    pp.get("sf2_b_coupling", a_params.sf2_b_coupling);
    
    
    
    

    if (abs(a_params.phi_amplitude) > 0.0 || abs(a_params.phi_background) > 0.0 )
    {
        pout() << "Spacetime contains scalar field of amplitude "
               << a_params.phi_amplitude
               << "\n with wavelenght of "
               << a_params.phi_wavelength
               << "\n and a background of "
               << a_params.phi_background
               << endl;
    }

    // Initial conditions for the black holes
    pp.get("bh1_bare_mass", a_params.bh1_bare_mass);
    pp.get("bh2_bare_mass", a_params.bh2_bare_mass);
    pp.get("bh1_spin", a_params.bh1_spin);
    pp.get("bh2_spin", a_params.bh2_spin);
    pp.get("bh1_offset", a_params.bh1_offset);
    pp.get("bh2_offset", a_params.bh2_offset);
    pp.get("bh1_momentum", a_params.bh1_momentum);
    pp.get("bh2_momentum", a_params.bh2_momentum);

    if (abs(a_params.bh1_bare_mass) > 0.0 || abs(a_params.bh2_bare_mass) > 0.0)
    {
        pout() << "Spacetime contains black holes with bare masses "
               << a_params.bh1_bare_mass << " and " << a_params.bh2_bare_mass
               << endl;
    }

    // Set verbosity
    a_params.verbosity = 3;
    pp.query("verbosity", a_params.verbosity);

    // Chombo grid params
    pp.get("max_level", a_params.maxLevel);
    a_params.numLevels = a_params.maxLevel + 1;
    std::vector<int> nCellsArray(SpaceDim);
    pp.getarr("N", nCellsArray, 0, SpaceDim);
    for (int idir = 0; idir < SpaceDim; idir++)
    {
        a_params.nCells[idir] = nCellsArray[idir];
    }

    // Enforce that dx is same in every directions
    // and that ref_ratio = 2 always as these conditions
    // are required in several places in our code
    a_params.refRatio.resize(a_params.numLevels);
    a_params.refRatio.assign(2);
    Real domain_length;
    pp.get("L", domain_length);
    a_params.coarsestDx = domain_length / a_params.nCells[0];
    for (int idir = 0; idir < SpaceDim; idir++)
    {
        a_params.domainLength[idir] =
            a_params.coarsestDx * a_params.nCells[idir];
    }

    // Chombo refinement and load balancing criteria
    pp.get("refine_threshold", a_params.refineThresh);
    pp.get("block_factor", a_params.blockFactor);
    pp.get("max_grid_size", a_params.maxGridSize);
    pp.get("fill_ratio", a_params.fillRatio);
    pp.get("buffer_size", a_params.bufferSize);

    // set average type -
    // set to a bogus default value, so we only break from solver
    // default if it's set to something real
    a_params.coefficient_average_type = -1;
    if (pp.contains("coefficient_average_type"))
    {
        std::string tempString;
        pp.get("coefficient_average_type", tempString);
        if (tempString == "arithmetic")
        {
            a_params.coefficient_average_type = CoarseAverage::arithmetic;
        }
        else if (tempString == "harmonic")
        {
            a_params.coefficient_average_type = CoarseAverage::harmonic;
        }
        else
        {
            MayDay::Error("bad coefficient_average_type in input");
        }
    } // end if an average_type is present in inputs

    // set up coarse domain box
    IntVect lo = IntVect::Zero;
    IntVect hi = a_params.nCells;
    hi -= IntVect::Unit;
    Box crseDomBox(lo, hi);
    a_params.probLo = RealVect::Zero;
    a_params.probHi = RealVect::Zero;
    a_params.probHi += a_params.domainLength;

    // Periodicity - for the moment enforce same in all directions
    ProblemDomain crseDom(crseDomBox);
    int is_periodic;
    pp.get("is_periodic", is_periodic);
    a_params.periodic.resize(SpaceDim);
    a_params.periodic.assign(is_periodic);
    for (int dir = 0; dir < SpaceDim; dir++)
    {
        crseDom.setPeriodic(dir, is_periodic);
    }
    a_params.coarsestDomain = crseDom;

    pout() << "periodicity = " << is_periodic << endl;
}
