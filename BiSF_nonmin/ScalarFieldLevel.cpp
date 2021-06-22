/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "ScalarFieldLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "AMRReductions.hpp"
#include "MatterCCZ4RHS.hpp"

// For constraints calculation
#include "MyNewMatterConstraints.hpp"

// For tag cells
#include "ChiAndBiPhiTaggingCriterion.hpp"

// Problem specific includes
//#include "ChiRelaxation.hpp"
#include "ComputePack.hpp"
#include "Potential.hpp"
//#include "ScalarBubble.hpp"
#include "BiScalarField.hpp"
#include "SetValue.hpp"

// Things to do at each advance step, after the RK4 is calculated
void ScalarFieldLevel::specificAdvance()
{
    // Enforce trace free A_ij and positive chi and alpha
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(),
                          PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck(), m_state_new, m_state_new,
                       EXCLUDE_GHOST_CELLS, disable_simd());

}

// Initial data for field and metric variables
void ScalarFieldLevel::initialData()
{
    CH_TIME("ScalarFieldLevel::initialData");
    if (m_verbosity)
        pout() << "ScalarFieldLevel::initialData " << m_level << endl;

    // First set everything to zero then initial conditions for scalar field -
    // here a bubble
    //BoxLoops::loop(make_compute_pack(SetValue(0.0),
    //                                 ScalarBubble(m_p.initial_params, m_dx)),
    //               m_state_new, m_state_new, INCLUDE_GHOST_CELLS);
}

// Things to do before outputting a checkpoint file
void ScalarFieldLevel::prePlotLevel()
{
    fillAllGhosts();
    Potential potential(m_p.potential_params);
    ScalarFieldWithPotential scalar_field(potential);
    BoxLoops::loop(
        MatterConstraints<ScalarFieldWithPotential>(
            //scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom)),
            scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom),
            c_Ham_abs_terms,  Interval(c_Mom_abs_terms, c_Mom_abs_terms),
            c_ricci_scalar, c_trA2,
            c_rho, c_S, c_ricci_scalar_tilde),
        m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
}

// Things to do in RHS update, at each RK4 step
void ScalarFieldLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{

    // Relaxation function for chi - this will eventually be done separately
    // with hdf5 as input
    if (m_time < m_p.relaxtime)
    {
        // Calculate chi relaxation right hand side
        // Note this assumes conformal chi and Mom constraint trivially
        // satisfied  No evolution in other variables, which are assumed to
        // satisfy constraints per initial conditions
        // Potential potential(m_p.potential_params);
        // ScalarFieldWithPotential scalar_field(potential);
        // ChiRelaxation<ScalarFieldWithPotential> relaxation(
        //     scalar_field, m_dx, m_p.relaxspeed, m_p.G_Newton);
        // SetValue set_other_values_zero(0.0, Interval(c_h11, NUM_VARS - 1));
        // auto compute_pack1 =
        //     make_compute_pack(relaxation, set_other_values_zero);
        // BoxLoops::loop(compute_pack1, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else
    {

        // Enforce trace free A_ij and positive chi and alpha
        BoxLoops::loop(
            make_compute_pack(TraceARemoval(),
                              PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
            a_soln, a_soln, INCLUDE_GHOST_CELLS);

        //AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
        AMRReductions<VariableType::evolution> amr_reductions(m_gr_amr);
        double abs_K_mean = amr_reductions.norm(c_K, 2, true);
        m_p.ccz4_params.K_mean = - abs_K_mean;


        // Calculate MatterCCZ4 right hand side with matter_t = ScalarField
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        MatterCCZ4RHS<ScalarFieldWithPotential> my_ccz4_matter(
            scalar_field, m_p.ccz4_params, m_dx, m_p.sigma, m_p.formulation,
            m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

// Things to do at ODE update, after soln + rhs
void ScalarFieldLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{
    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void ScalarFieldLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                               const FArrayBox &current_state)
{
    BoxLoops::loop(ChiAndPhiTaggingCriterion(m_dx, m_p.regrid_threshold_chi,
                                             m_p.regrid_threshold_phi),
                   current_state, tagging_criterion);
}

// compute dt
Real ScalarFieldLevel::computeDt()
{
    if (m_verbosity)
        pout() << "GRAMRLevel::computeDt " << m_level << endl;

    //AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    double rho_mean = amr_reductions.norm(c_rho, 2, true);
    double S_mean = amr_reductions.norm(c_rho, 2, true);
    double omega = S_mean/rho_mean/3.;

    double new_dt = 0;

    // // AMRLevel
    const Vector<AMRLevel *> &levels =
             const_cast<GRAMR &>(m_gr_amr).getAMRLevels();
    const int num_levels = levels.size();
    AMRLevel &level = *levels[0];

    new_dt = level.computeInitialDt();

    if (omega > 0.9 ||  omega < -0.1 ) {
      new_dt =  new_dt /10;
    }

    // pout() << "dt is  "  << new_dt  << "  initial_dt is "<< m_dt << endl;

    return new_dt;
}
