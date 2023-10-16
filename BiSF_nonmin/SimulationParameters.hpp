/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "GRParmParse.hpp"
#include "SimulationParametersBase.hpp"

// Problem specific includes:
#include "CCZ4.hpp"
#include "Potential.hpp"
//#include "ScalarBubble.hpp"

class SimulationParameters : public SimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        // read the problem specific params
        readParams(pp);
    }

    void readParams(GRParmParse &pp)
    {
        // for regridding
        pp.load("regrid_threshold_chi", regrid_threshold_chi);
        pp.load("regrid_threshold_phi", regrid_threshold_phi);

        pp.load("G_Newton", G_Newton, 1.0);
        pp.load("scalar_mass", potential_params.scalar_mass);
        pp.load("g_coupling", potential_params.g_coupling);
        pp.load("sf1_a_coupling", potential_params.sf1_a_coupling);
        pp.load("sf2_a_coupling", potential_params.sf2_a_coupling);
        pp.load("sf1_b_coupling", potential_params.sf1_b_coupling);
        pp.load("sf2_b_coupling", potential_params.sf2_b_coupling);
        pp.load("K_mean", ccz4_params.K_mean, 0.0);

        // Initial and SF data
        // initial_params.centerSF =  center; // already read in SimulationParametersBase
        // pp.load("amplitudeSF", initial_params.amplitudeSF);
        // pp.load("widthSF", initial_params.widthSF);
        // pp.load("r_zero", initial_params.r_zero);

        // Relaxation params
        pp.load("relaxtime", relaxtime);
        pp.load("relaxspeed", relaxspeed);
    }

    // Problem specific parameters
    Real regrid_threshold_chi, regrid_threshold_phi;

    // Initial data for matter and potential
    double G_Newton;
    //ScalarBubble::params_t initial_params;
    Potential::params_t potential_params;
    // Relaxation params
    Real relaxtime, relaxspeed;
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
