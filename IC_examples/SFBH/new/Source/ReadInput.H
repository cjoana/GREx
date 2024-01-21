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
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _READINPUT_H_
#define _READINPUT_H_

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "CH_HDF5.H"
#include "DebugDump.H"
#include "FABView.H"
#include "FArrayBox.H"
#include "LevelData.H"
#include "LoadBalance.H"
#include "MultigridUserVariables.hpp"
#include "MultilevelLinearOp.H"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "ReadinUserVariables.hpp"
#include "SetLevelData.H"
#include <iostream>

#ifdef CH_Linux
// Should be undefined by default
//#define TRAP_FPE
#undef TRAP_FPE
#endif

#ifdef TRAP_FPE
static void enableFpExceptions();
#endif

using std::cerr;

/// This function reads in the grid data
void read_grids(Vector<DisjointBoxLayout> &a_grids,
                const PoissonParameters &a_params,
                const string filename = "SourceData.3d.hdf5")
{
#ifdef CH_USE_HDF5

    // set up a temp data structure for the source data
    Vector<LevelData<FArrayBox> *> temp_data;
    Vector<string> variable_names;
    int temp_num_levels;
    Real dx, dt, time;
    Box temp_domain_box;
    Vector<int> temp_ref_ratio;

    // update directly the input grids
    ReadAMRHierarchyHDF5(filename, a_grids, temp_data, variable_names,
                         temp_domain_box, dx, dt, time, temp_ref_ratio,
                         temp_num_levels);

    int max_temp_level = temp_data.size();
    // clean up temporary storage
    for (int level = 0; level < max_temp_level; level++)
    {
        delete temp_data[level];
        temp_data[level] = NULL;
    }
#endif
}

/// This function reads in the source data
void read_matter_data(Vector<LevelData<FArrayBox> *> &a_multigrid_vars,
                      const Vector<DisjointBoxLayout> &a_grids,
                      const PoissonParameters &a_params,
                      const string filename = "SourceData.3d.hdf5")
{
#ifdef CH_USE_HDF5

    // set up a temp data structure for the source data
    Vector<LevelData<FArrayBox> *> temp_data;
    Vector<string> variable_names;
    int temp_num_levels;
    Real dx, dt, time;
    Box temp_domain_box;
    Vector<int> temp_ref_ratio;
    Vector<DisjointBoxLayout> temp_grids;
    // set up a temporary structure and then transfer data to multigrid_vars
    ReadAMRHierarchyHDF5(filename, temp_grids, temp_data, variable_names,
                         temp_domain_box, dx, dt, time, temp_ref_ratio,
                         temp_num_levels);

    // check the read in data matches what we expect
    int max_level_multigrid = a_multigrid_vars.size();
    CH_assert(temp_num_levels == max_level_multigrid);
    for (int iname = 0; iname < NUM_READIN_VARS; iname++)
    {
        CH_assert(variable_names[iname] ==
                  ReadinUserVariables::variable_names[iname]);
    }
    // TODO: Could add more checks here

    // assign values to the source vars in multigrid_vars
    for (int level = 0; level < max_level_multigrid; level++)
    {
        // note no need to account for ghosts as these will be filled elsewhere
        // TODO: Could make this general for any matter type
        Interval sourceComps(c_phi_in, c_Pi_in);
        Interval multigridComps(c_phi_0, c_Pi_0);
        temp_data[level]->copyTo(sourceComps, *a_multigrid_vars[level],
                                 multigridComps);
    }

    int max_temp_level = temp_data.size();
    // clean up temporary storage
    for (int level = 0; level < max_temp_level; level++)
    {
        delete temp_data[level];
        temp_data[level] = NULL;
    }
#endif
}

#endif
