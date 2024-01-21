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
#include "FilesystemTools.hpp"

// function to read in the key params for solver
void getPoissonParameters(PoissonParameters &a_params)
{
    GRParmParse pp;
    
#ifdef CH_MPI
    // setPoutBaseName must be called before pout is used
    if (pp.contains("pout_path"))
    {
        pp.get("pout_path", a_params.pout_path);
        // Create pout directory
        if (!FilesystemTools::directory_exists(a_params.pout_path))
            FilesystemTools::mkdir_recursive(a_params.pout_path);
        setPoutBaseName(a_params.pout_path + "pout");
    }
    else
    {
        a_params.pout_path = "";
    }
#endif
    
    // problem specific params
    pp.get("alpha", a_params.alpha);
    pp.get("beta", a_params.beta);
    // print out the overall coeffs just to be sure we have selected them
    // correctly
    pout() << "alpha, beta = " << a_params.alpha << ", " << a_params.beta
           << endl;

    // Read from hdf5 file
    if (pp.contains("input_filename"))
    {
        pp.get("input_filename", a_params.input_filename);
        a_params.readin_matter_data = true;
    }
    else
    {
        a_params.input_filename = "";
        a_params.readin_matter_data = false;
    }
    if (pp.contains("output_path"))
    {
        pp.get("output_path", a_params.output_path);
        if (!FilesystemTools::directory_exists(a_params.output_path))
            FilesystemTools::mkdir_recursive(a_params.output_path);
    }
    else
    {
        a_params.output_path = "";
    }

    if (pp.contains("output_filename"))
    {
        string filename;
        pp.get("output_filename", filename);
        a_params.output_filename = a_params.output_path + filename;
    }
    else
    {
        a_params.output_filename =
            a_params.output_path + "OutputDataFinal.3d.hdf5";
    }

    // Initial conditions for the scalar field
    pp.get("G_Newton", a_params.G_Newton);
    pp.get("phi_0", a_params.phi_0);
    pp.get("phi_amplitude", a_params.phi_amplitude);
    pp.get("phi_wavelength", a_params.phi_wavelength);
    pp.get("pi_0", a_params.pi_0);
    pp.get("pi_amplitude", a_params.pi_amplitude);
    pp.get("pi_wavelength", a_params.pi_wavelength);
    pp.get("n_swirl_phi", a_params.n_swirl_phi);
    pp.get("n_swirl_pi", a_params.n_swirl_pi);

    // Potential parameters
    pp.get("pot_Lambda", a_params.pot_Lambda);

    if (abs(a_params.phi_amplitude) > 0.0)
    {
        pout() << "Spacetime contains scalar field of amplitude "
               << a_params.phi_amplitude << endl;
    }

    pp.get("psi_reg", a_params.psi_reg);
    pp.get("sign_of_K", a_params.sign_of_K);

    a_params.include_A2 = false;
    pp.get("include_A2", a_params.include_A2);
    a_params.method_compact = false;
    pp.get("method_compact", a_params.method_compact);
    a_params.deactivate_zero_mode = false;
    pp.get("deactivate_zero_mode", a_params.deactivate_zero_mode);

    // Initial conditions for the black holes
    pp.get("bh1_bare_mass", a_params.bh1_bare_mass);
    pp.get("bh2_bare_mass", a_params.bh2_bare_mass);
    std::vector<double> temp_spin1(SpaceDim);
    std::vector<double> temp_spin2(SpaceDim);
    std::vector<double> temp_offset1(SpaceDim);
    std::vector<double> temp_offset2(SpaceDim);
    std::vector<double> temp_mom1(SpaceDim);
    std::vector<double> temp_mom2(SpaceDim);
    pp.getarr("bh1_spin", temp_spin1, 0, SpaceDim);
    pp.getarr("bh2_spin", temp_spin2, 0, SpaceDim);
    pp.getarr("bh1_offset", temp_offset1, 0, SpaceDim);
    pp.getarr("bh2_offset", temp_offset2, 0, SpaceDim);
    pp.getarr("bh1_momentum", temp_mom1, 0, SpaceDim);
    pp.getarr("bh2_momentum", temp_mom2, 0, SpaceDim);
    for (int idir = 0; idir < SpaceDim; idir++)
    {
        a_params.bh1_spin[idir] = temp_spin1[idir];
        a_params.bh2_spin[idir] = temp_spin2[idir];
        a_params.bh1_offset[idir] = temp_offset1[idir];
        a_params.bh2_offset[idir] = temp_offset2[idir];
        a_params.bh1_momentum[idir] = temp_mom1[idir];
        a_params.bh2_momentum[idir] = temp_mom2[idir];
    }

    if (abs(a_params.bh1_bare_mass) > 0.0 || abs(a_params.bh2_bare_mass) > 0.0)
    {
        pout() << "Spacetime contains black holes with bare masses "
               << a_params.bh1_bare_mass << " and " << a_params.bh2_bare_mass
               << endl;
    }

    // read from set of data
    if (pp.contains("read_from_data_dphi"))
    {
        pp.get("read_from_data_dphi", a_params.read_from_data_dphi);
        pp.get("read_from_data_dpi", a_params.read_from_data_dpi);
        pp.get("data_lines", a_params.lines);
        pp.get("data_spacing", a_params.spacing);
#define num 128
        static Real inputdata_dphi[num * num * num];
        static Real inputdata_dpi[num * num * num];
        Real tmp_dphi = 0.0;
        Real tmp_dpi = 0.0;
        ifstream ifspsi_dphi(a_params.read_from_data_dphi);
        ifstream ifspsi_dpi(a_params.read_from_data_dpi);
        for (int i = 0; i < a_params.lines * a_params.lines * a_params.lines;
             ++i)
        {
            ifspsi_dphi >> tmp_dphi;
            ifspsi_dpi >> tmp_dpi;
            inputdata_dphi[i] = tmp_dphi;
            inputdata_dpi[i] = tmp_dpi;
        }
        a_params.input_dphi = inputdata_dphi;
        a_params.input_dpi = inputdata_dpi;
    }
    else
    {
        a_params.read_from_data_dphi = "none";
        a_params.read_from_data_dpi = "none";
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
    int max_cells = max(a_params.nCells[0], a_params.nCells[1]);
    max_cells = max(a_params.nCells[2], max_cells);
    a_params.coarsestDx = domain_length / max_cells;
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
    pp.get("regrid_radius", a_params.regrid_radius, 0);

    // Get the tolerance - used to set a_coeff
    pp.get("tolerance", a_params.tolerance);

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

    // Periodicity
    ProblemDomain crseDom(crseDomBox);
    a_params.periodic.resize(SpaceDim);
    pp.getarr("is_periodic", a_params.periodic, 0, SpaceDim);
    a_params.periodic_directions_exist = false;
    for (int dir = 0; dir < SpaceDim; dir++)
    {
        crseDom.setPeriodic(dir, a_params.periodic[dir]);
        if (a_params.periodic[dir])
        {
            a_params.periodic_directions_exist = true;
        }
    }
    a_params.coarsestDomain = crseDom;

    // now the boundary read in
    a_params.boundary_params.read_params(pp);

    FOR1(idir)
    {
        // default center to center of grid, may reset below
        // depending on boundaries
        a_params.center[idir] = domain_length / 2.0;

        if ((a_params.boundary_params.lo_boundary[idir] ==
             BoundaryConditions::REFLECTIVE_BC) &&
            (a_params.boundary_params.hi_boundary[idir] !=
             BoundaryConditions::REFLECTIVE_BC))
            a_params.center[idir] = 0.;
        else if ((a_params.boundary_params.hi_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC) &&
                 (a_params.boundary_params.lo_boundary[idir] !=
                  BoundaryConditions::REFLECTIVE_BC))
            a_params.center[idir] = domain_length;
        else if ((a_params.boundary_params.hi_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC) &&
                 (a_params.boundary_params.lo_boundary[idir] ==
                  BoundaryConditions::REFLECTIVE_BC))
            a_params.center[idir] = 0.;
    }

    pout() << "grid center set to " << a_params.center[0] << " "
           << a_params.center[1] << " " << a_params.center[2] << endl;
}
