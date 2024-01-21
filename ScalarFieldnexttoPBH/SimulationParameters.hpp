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
#include "InitialScalarData.hpp"
#include "PBHwithScalarMetric.hpp"
#include "Potential.hpp"

class SimulationParameters : public SimulationParametersBase
{
  public:
    SimulationParameters(GRParmParse &pp) : SimulationParametersBase(pp)
    {
        // read the problem specific params
        read_params(pp);
        check_params();
    }

    void read_params(GRParmParse &pp)
    {
        pp.load("G_Newton", G_Newton, 1.0);

        // Potential data
        pp.load("false_vacuum_potential",
                potential_params.false_vacuum_potential, 0.1);
        pp.load("true_vacuum_potential", potential_params.true_vacuum_potential,
                0.0);
        pp.load("barrier_amplitude",
                potential_params.barrier_amplitude_parameter, 10.0);
        pp.load("true_vacuum_expectation_value", potential_params.phi_VEV, 1.0);

        // Initial scalar field data
        initial_params.center =
            center; // already read in SimulationParametersBase
        pp.load("true_vacuum_expectation_value", initial_params.amplitude, 1.0);
        pp.load("bubble_width", initial_params.width, 1.25);
        pp.load("bubble_radius", initial_params.radius, 8.0);
        initial_params.velocity = (potential_params.false_vacuum_potential -
                                   potential_params.true_vacuum_potential) /
                                  4.0 / acos(-1.0); // where arccos(-1)=pi

        // Initial black hole data
        PBH_params.Bubble_center =  center;
        std::array<double, GR_SPACEDIM> BH_default_center = {(double)(3 * L / 2), 0., 0.};
        pp.load("BH_mass", PBH_params.mass, 0.1);
        pp.load("false_vacuum_potential", PBH_params.vp, 0.1);
        pp.load("BH_center", PBH_params.BH_center, BH_default_center);

        // By now the code can only deal with Schwarzschild BH
        // with no spin, the spin parameters are left as future work
        // pp.load("BH_spin", PBH_params.spin, 0.0);
        // pp.load("BH_spin_dir", PBH_params.spin_direction);
 

#ifdef USE_AHFINDER
        double AH_guess = 0.5 * PBH_parames.mass;
        pp.load("AH_initial_guess", AH_initial_guess, AH_guess);
#endif
    }

    void check_params()
    {
        warn_parameter("bubble_width", initial_params.width,
                       initial_params.width < 0.25 * L,
                       "too large. Scalar bubble is too close to the Black Hole");
        warn_parameter("BH_mass", PBH_params.mass, PBH_params.mass >= 0.0,
                       "should be >= 0.0");

        FOR(idir)
        {
            std::string name = "BH_center[" + std::to_string(idir) + "]";
            warn_parameter(
                name, PBH_params.BH_center[idir],
                (PBH_params.BH_center[idir] >= 0) &&
                    (PBH_params.BH_center[idir] <= (ivN[idir] + 1) * coarsest_dx),
                "should be within the computational domain");
        }
    }

    // Initial data for matter and potential and BH
    double G_Newton;
    
    InitialScalarData::params_t initial_params;
    Potential::params_t potential_params;
    PBHwithScalarMetric::params_t PBH_params;

#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
