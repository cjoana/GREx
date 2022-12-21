/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "PerfectFluidLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "AMRReductions.hpp"
#include "MyGauge.hpp"
#include "MatterCCZ4RHS.hpp"

// For constraints calculation
//#include "MatterConstraints.hpp"
// #include "ExtendedMatterConstraints.hpp"
#include "MyNewMatterConstraints.hpp"

// Problem specific includes
// #include "ChiRelaxation.hpp"   // $GRCHOMBO/Source/Matter/
#include "FluidInitData.hpp"
//#include "KTaggingCriterion.hpp"
#include "KandWTaggingCriterion.hpp"
#include "ComputePack.hpp"
#include "SetValue.hpp"
#include "PerfectFluid.hpp"     // $GRCHOMBO/Source/Matter/

// Things to do at each advance step, after the RK4 is calculated
void PerfectFluidLevel::specificAdvance()
{


}

// Initial data for field and metric variables                                    
void PerfectFluidLevel::initialData()
{

}


// Things to do before outputting a checkpoint file
void PerfectFluidLevel::prePlotLevel()
{

    // Compute diganostics  if needed. 
        // fillAllGhosts();
        // EquationOfState eos(m_p.eos_params);                                          
        // PerfectFluidWithEOS perfect_fluid(eos);
        // BoxLoops::loop(MatterConstraints<PerfectFluidWithEOS>(
        //                     perfect_fluid, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom),
        //                     c_Ham_abs_terms,  Interval(c_Mom_abs_terms, c_Mom_abs_terms),
        //                     c_ricci_scalar, c_trA2,
        //                     c_rho, c_S, c_ricci_scalar_tilde),
        //                m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        // fillAllGhosts();


    // Print out diagnostic, if needed.
        // AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
        // double Ham = amr_reductions.norm(c_Ham_abs_terms, 2, true);
        // double rho = amr_reductions.norm(c_rho, 2, true);
        // pout() << "Ham_Abs_tems " << Ham <<  "  rho  " <<   rho   << endl;
}




// Things to do in RHS update, at each RK4 step
void PerfectFluidLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{
    // Do nothing. 
    SetValue set_other_values_zero(0.0, Interval(c_chi, NUM_VARS - 1));
       BoxLoops::loop( set_other_values_zero,
        //    make_compute_pack(TraceARemoval(), set_other_values_zero)
       a_soln,  a_rhs, INCLUDE_GHOST_CELLS);
    
    
}

// Things to do at ODE update, after soln + rhs
void PerfectFluidLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Evaluate fluid vars (density, energy, etc) 
    EquationOfState eos(m_p.eos_params);
    PerfectFluidWithEOS perfect_fluid(eos);
    BoxLoops::loop(perfect_fluid, a_soln, a_soln,
                     EXCLUDE_GHOST_CELLS); 

    fillAllGhosts();

}

void PerfectFluidLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                               const FArrayBox &current_state)
{

}

void PerfectFluidLevel::specificPostTimeStep()
{
    // Search for BHs. 
#ifdef USE_AHFINDER
    if (m_p.AH_activate && m_level == m_p.AH_params.level_to_run)
        m_bh_amr.m_ah_finder.solve(m_dt, m_time, m_restart_time);
#endif

    // You might include other stuff, like extraction of GWs. 

}
