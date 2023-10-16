/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

 /* Modifications from base_lab (example with Gauss)


 */

#include "parstream.H" //Gives us pout()
#include <iostream>

#include "BHAMR.hpp"
#include "DefaultLevelFactory.hpp"
//#include "GRAMR.hpp"
#include "GRParmParse.hpp"
#include "SetupFunctions.hpp"
#include "SimulationParameters.hpp"

// Problem specific includes:
#include "PerfectFluidLevel.hpp"

int runGRChombo(int argc, char *argv[])
{
	using Clock = std::chrono::steady_clock;
    using Minutes = std::chrono::duration<double, std::ratio<60, 1>>;

    std::chrono::time_point<Clock> start_time = Clock::now();

	
    // Load the parameter file and construct the SimulationParameter class
    // To add more parameters edit the SimulationParameters file.
    char *in_file = argv[1];
    GRParmParse pp(argc - 2, argv + 2, NULL, in_file);
    SimulationParameters sim_params(pp);
    
    // Rerun! 
        // now loop over chk files
    for (int ifile = 0; ifile < sim_params.num_files;
         ifile++)
    {
		
        // set up the file from next checkpoint
        std::ostringstream current_file;
        current_file << std::setw(6) << std::setfill('0')
                     << sim_params.start_file + ifile * sim_params.pp_chk_interval;
        std::string restart_file(sim_params.pp_chk_prefix +
                                 current_file.str() + ".3d.hdf5");
        // HDF5Handle handle(restart_file, HDF5Handle::OPEN_RDONLY);

      
        sim_params.restart_file = restart_file;  // Unnecessary ?!
		pp.setStr("restart_file", restart_file);


		//pout() << "Loading from restart_file " << sim_params.restart_file << ".\n";
        //std::cout << "Loading from restart_file " << sim_params.restart_file << ".\n";

// ################################################################# 
		{
		// The line below selects the problem that is simulated
		// (To simulate a different problem, define a new child of AMRLevel
		// and an associated LevelFactory)
		//GRAMR gr_amr;
		BHAMR bh_amr;
		DefaultLevelFactory<PerfectFluidLevel> fluid_level_fact(bh_amr,
															sim_params);
		setupAMRObject(bh_amr, fluid_level_fact);


		// call this after amr object setup so grids known
		// and need it to stay in scope throughout run
		AMRInterpolator<Lagrange<4>> interpolator(
			bh_amr, sim_params.origin, sim_params.dx, sim_params.boundary_params,
			sim_params.verbosity);
		bh_amr.set_interpolator(&interpolator);

	#ifdef USE_AHFINDER
		if (sim_params.AH_activate)
		{
		   
			AHSphericalGeometry sph(sim_params.center);
			AHSphericalGeometry sph2(sim_params.center);
		   
		bh_amr.m_ah_finder.add_ah(sph, sim_params.AH_1_initial_guess,
									  sim_params.AH_params);
		
		bh_amr.m_ah_finder.add_ah(sph2, sim_params.AH_2_initial_guess,
									  sim_params.AH_params);

		}
	#endif
  
// #################################################################     

        sim_params.stop_time = 1e99;     // arbitrary large num. > current time
        sim_params.max_steps = 1;        // do not "evolve" further than once.  
        bh_amr.run(sim_params.stop_time, sim_params.max_steps);
        bh_amr.conclude();
		}
    }
    
    std::cout << "Ending RERUN!! " <<  ".\n";

    auto now = Clock::now();
    auto duration = std::chrono::duration_cast<Minutes>(now - start_time);
    pout() << "Total simulation time (mins): " << duration.count() << ".\n";

    

    CH_TIMER_REPORT(); // Report results when running with Chombo timers.

    return 0;
}


int main(int argc, char *argv[])
{
    mainSetup(argc, argv);

    int status = runGRChombo(argc, argv);

    if (status == 0)
        pout() << "GRChombo finished." << std::endl;
    else
        pout() << "GRChombo failed with return code " << status << std::endl;

    mainFinalize();
    return status;
}
