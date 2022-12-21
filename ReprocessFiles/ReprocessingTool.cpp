/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to LICENSE, in Chombo's root directory.
 */
#endif

// General includes:
#include "parstream.H" //Gives us pout()
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sys/time.h>
using std::endl;

// Problem specific includes:
#include "AMRInterpolator.hpp"
#include "DefaultLevelFactory.hpp"
#include "GRAMR.hpp"
#include "InterpolationQuery.hpp"
#include "Lagrange.hpp"
#include "ReprocessingLevel.hpp"
#include "SetupFunctions.hpp"
#include "SimulationParameters.hpp"
#include "UserVariables.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

int runReprocessingTool(int argc, char *argv[])
{
    // Load the parameter file and construct the SimulationParameter class
    // To add more parameters edit the SimulationParameters file.
    std::string in_string = argv[argc - 1];
    pout() << in_string << std::endl;
    char const *in_file = argv[argc - 1];
    GRParmParse pp(0, argv + argc, NULL, in_file);
    SimulationParameters sim_params(pp);

    // Setup the initial object (from restart_file checkpoint)
    GRAMR gr_amr;
    DefaultLevelFactory<ReprocessingLevel> empty_level_fact(gr_amr, sim_params);
    setupAMRObject(gr_amr, empty_level_fact);

    // now loop over chk files
    for (int ifile = sim_params.start_file; ifile < sim_params.num_files;
         ifile++)
    {
        // set up the file from next checkpoint
        std::ostringstream current_file;
        current_file << std::setw(6) << std::setfill('0')
                     << ifile * sim_params.checkpoint_interval;
        std::string restart_file(sim_params.checkpoint_prefix +
                                 current_file.str() + ".3d.hdf5");
        HDF5Handle handle(restart_file, HDF5Handle::OPEN_RDONLY);
        gr_amr.setupForRestart(handle);
        handle.close();
    }

    gr_amr.conclude();

    return 0;
}

int main(int argc, char *argv[])
{
    mainSetup(argc, argv);

    int status = runReprocessingTool(argc, argv);

    mainFinalize();
    return status;
}
