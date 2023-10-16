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
#include "EquationOfState.hpp"

// For BH at initio
// #include "KerrBH.hpp"


// #include "ScalarGauss.hpp" FIXME: needed?

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
        // Cosmological parameters
        pp.load("G_Newton", G_Newton, 1.0);

        // Predefine ideal Fluids
        pp.load("omega", eos_params.omega, 0.0);
        pp.load("mass", eos_params.mass, 1.0);

        pp.load("omega", omega, 0.0);
        
        // Grid setup
        pp.get("num_files", num_files);
        pp.get("start_file", start_file);
        pp.get("pp_chk_interval", pp_chk_interval);
        pp.get("pp_chk_prefix", pp_chk_prefix);

#ifdef USE_AHFINDER
        double AH_guess = 0.5;
        pp.load("AH_1_initial_guess", AH_1_initial_guess, AH_guess);
        pp.load("AH_2_initial_guess", AH_2_initial_guess, AH_guess);
#endif

    }

    // Not sure needed here?   TODO: check this!
    Real omega;

    // loading params
    int num_files, start_file;
    int pp_chk_interval;
    std::string pp_chk_prefix; 

    // Initial data for matter and potential
    double G_Newton;
    EquationOfState::params_t eos_params;

#ifdef USE_AHFINDER
    double AH_1_initial_guess;
    double AH_2_initial_guess;
#endif

};

#endif /* SIMULATIONPARAMETERS_HPP_ */
