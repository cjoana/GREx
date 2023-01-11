
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

#ifndef _READHDF5_H_
#define _READHDF5_H_

#include <iostream>

#include "BoxIterator.H"
#include "CH_HDF5.H"
#include "DataIterator.H"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "LevelData.H"
#include "ParmParse.H"
#include "ProblemDomain.H"
#include <sys/time.h>

#include "PoissonParameters.H"
#include "SetLevelData.H"

void defineExchangeCopier(const DisjointBoxLayout &a_level_grids,
                          DisjointBoxLayout grown_grids, Copier exchange_copier)
{
    IntVect iv_ghosts = 3 * IntVect::Unit;
    exchange_copier.exchangeDefine(grown_grids, iv_ghosts);
}

void exchange_function(LevelData<FArrayBox> &a_multigrid_vars,
                     const Copier &a_exchange_copier)
{

    a_multigrid_vars.exchange(a_multigrid_vars.interval(), a_exchange_copier);
}

void read_level_data(LevelData<FArrayBox> &lev_dat_1,
                     LevelData<FArrayBox> &lev_dat_2, int ilev)
{
    CH_assert(lev_dat_1.disjointBoxLayout() == lev_dat_2.disjointBoxLayout());

    DataIterator dit = lev_dat_1.dataIterator();
    const DisjointBoxLayout &grids = lev_dat_1.disjointBoxLayout();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &fab_1 = lev_dat_1[dit()];
        FArrayBox &fab_2 = lev_dat_2[dit()];
        // fab_1.setVal(0.0, c_h11);
        // fab2.setVal(0.0, c_h11_0);
        Box b = fab_2.box();
        Box b_no_ghosts = grids[dit()];
        BoxIterator bit(b_no_ghosts);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            //           for (int copy_comp = 0; copy_comp < 1; ++copy_comp)
            //           {
            fab_2(iv, c_phi_0) = fab_1(iv, c_phi);
            fab_2(iv, c_h11_0) = fab_1(iv, c_h11);
            fab_2(iv, c_h12_0) = fab_1(iv, c_h12);
            fab_2(iv, c_h13_0) = fab_1(iv, c_h13);
            fab_2(iv, c_h22_0) = fab_1(iv, c_h22);
            fab_2(iv, c_h23_0) = fab_1(iv, c_h23);
            fab_2(iv, c_h33_0) = fab_1(iv, c_h33);
            fab_2(iv, c_psi) = pow(fab_1(iv, c_chi), -0.25);
            // pout() << iv << fab_1(iv, c_phi) << " level " << ilev << endl;
            //           }
        }
    }
}

void readgrids(Vector<DisjointBoxLayout> &aa_grids,
               const PoissonParameters &a_params)
{
    HDF5Handle handle(a_params.read_from_file, HDF5Handle::OPEN_RDONLY);
    HDF5HeaderData header;
    header.readFromFile(handle);
    int num_levels = header.m_int["num_levels"];
    aa_grids.resize(num_levels);
    for (int level = 0; level < aa_grids.size(); level++)
    {
        HDF5HeaderData level_header;
        // Set the HDF5 group and load level header
        char level_str[20];
        sprintf(level_str, "%d", level);
        const std::string label = std::string("level_") + level_str;
        handle.setGroup(label);
        level_header.readFromFile(handle);

        // Check and write periodicity of file
        bool periodic_bool[3];
        for (int dir = 0; dir < SpaceDim; ++dir)
        {
            char dir_str[20];
            sprintf(dir_str, "%d", dir);
            const std::string periodic_label =
                std::string("is_periodic_") + dir_str;
            if (level_header.m_int[periodic_label] == true)
            {
                periodic_bool[dir] = true;
                pout() << periodic_bool[dir] << endl;
            }
        }
        // For now we need the loop below because the final output bdry
        // conditions are always periodic.
        if (a_params.periodic[0] == 0)
        {
            for (int dir = 0; dir < SpaceDim; ++dir)
            {
                periodic_bool[dir] = false;
            }
        }

        // Read boxes and assign processors to create DisjointBoxLayout
        // for this level
        ProblemDomain a_domain(level_header.m_box["prob_domain"],
                               periodic_bool);
        Vector<Box> boxes;
        int grid_error = read(handle, boxes);
        Vector<int> procIDs;
        LoadBalance(procIDs, boxes);
        DisjointBoxLayout layout(boxes, procIDs, a_domain);
        aa_grids[level] = layout;
    }

    handle.close();
}

void readHDF5(LevelData<FArrayBox> &a_multigrid_vars,
                const Vector<DisjointBoxLayout> aa_grids,
              const PoissonParameters &a_params, int ilev,
                              const IntVect a_ghosts)
{
#ifdef CH_USE_HDF5

    LevelData<FArrayBox> *gr_chombo_vars;
    gr_chombo_vars = new LevelData<FArrayBox>(aa_grids[ilev], NUM_GRCHOMBO_VARS, a_ghosts);

    HDF5Handle handle(a_params.read_from_file, HDF5Handle::OPEN_RDONLY);
    HDF5HeaderData level_header;
    // Set the HDF5 group and load level header
    char level_str[20];
    sprintf(level_str, "%d", ilev);
    const std::string label = std::string("level_") + level_str;
    handle.setGroup(label);
    level_header.readFromFile(handle);

    int data_error = read(handle, *gr_chombo_vars, "data", aa_grids[ilev], Interval(), false);
    
    pout() << "Reading data from " << a_params.read_from_file << endl;

    // HDF5HeaderData header;
    // header.readFromFile(handle);

    // // Read file
    // const DisjointBoxLayout &level_domain =
    //     a_multigrid_vars.disjointBoxLayout();
    // Real dx, dt, time;
    // Box box;
    // int ref_ratio;

    // Copier exchange_copier;
    // LevelData<FArrayBox> level_data_noghost(level_domain, NUM_GRCHOMBO_VARS,
    //                                         0 * IntVect::Unit);
    // LevelData<FArrayBox> level_data_ghost(level_domain, NUM_GRCHOMBO_VARS,
    //                                       3 * IntVect::Unit);

    // readLevel(handle, ilev, level_data_noghost, dx, dt, time, box, ref_ratio,
    //           level_data_noghost.interval(), false);

    // level_data_noghost.copyTo(level_data_ghost);
    // //    defineExchangeCopier(level_domain,
    // //    level_data_noghost.disjointBoxLayout(), exchange_copier);
    // //    level_data_ghost.exchange(exchange_copier);

    // LevelData<FArrayBox> level_data_readin(a_multigrid_vars.disjointBoxLayout(),
    //                                        level_data_ghost.nComp(),
    //                                        level_data_ghost.ghostVect());

    // level_data_ghost.copyTo(level_data_readin);

    read_level_data(*gr_chombo_vars, a_multigrid_vars, ilev);

    // Cleaning storage
    delete gr_chombo_vars;
    gr_chombo_vars = NULL;

    handle.close();

#endif
}
#endif
