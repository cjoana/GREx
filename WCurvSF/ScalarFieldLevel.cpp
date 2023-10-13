/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

// General includes common to most GR problems
#include "ScalarFieldLevel.hpp"
#include "BoxLoops.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
#include "SixthOrderDerivatives.hpp"
#include "TraceARemoval.hpp"

// For RHS update
#include "AMRReductions.hpp"
#include "MatterCCZ4RHS.hpp"

// For constraints calculation
//#include "NewMatterConstraints.hpp"
#include "MyNewMatterConstraints.hpp"

// For tag cells
#include "FixedGridsTaggingCriterion.hpp"

// Problem specific includes
#include "ComputePack.hpp"
#include "GammaCalculator.hpp"
#include "ScalarInitData.hpp"
#include "InitialScalarData.hpp"
#include "InitialMetricData.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SetValue.hpp"
//#include "KerrBH.hpp"

// For calculation of Weyl and CP curvature invariants
#include "MatterWeyl4.hpp"





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
        BoxLoops::loop(
            NanCheck(),
            m_state_new, m_state_new, EXCLUDE_GHOST_CELLS, disable_simd());
}

// Initial data for field and metric variables
void ScalarFieldLevel::initialData()
{

    std::cout << "ScalarFieldLevel::initialData " << m_level << endl;


    CH_TIME("ScalarFieldLevel::initialData");
    if (m_verbosity)
        pout() << "ScalarFieldLevel::initialData " << m_level << endl;

    // First set everything to zero then initial conditions for scalar field -
    // here a Kerr BH and a scalar field profile
    BoxLoops::loop(make_compute_pack(SetValue(0.), InitialMetricData(m_p.initial_params, m_dx)),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    //BoxLoops::loop(InitialScalarData(m_p.initial_params, m_dx),
    //               m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);


    // // Compute and print Init Curvatures
    // Potential potential(m_p.potential_params);
    // ScalarFieldWithPotential scalar_field(potential);
    // fillAllGhosts();

    // BoxLoops::loop(
    //     MatterConstraints<ScalarFieldWithPotential>(
    //         scalar_field, m_dx, m_p.G_Newton, c_Ham, Interval(c_Mom, c_Mom),
    //         c_Ham_abs_terms,  Interval(c_Mom_abs_terms, c_Mom_abs_terms),
    //         c_ricci_scalar, c_trA2,
    //         c_rho, c_S, c_ricci_scalar_tilde),
    //     m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

    // BoxLoops::loop(
    //             MatterWeyl4(m_p.extraction_params.center, m_dx),
    //             m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);

    // AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    // AMRReductions<VariableType::evolution> camr_reductions(m_gr_amr);
    // double Ham = amr_reductions.norm(c_Ham_abs_terms, 2, true);
    // double RicciScalar = amr_reductions.norm(c_ricci_scalar, 2, true);
    // double phi = camr_reductions.norm(c_phi, 2, true);

    // double wcurv = amr_reductions.norm(c_Weyl_curv, 2, true);
    // double ChP = amr_reductions.norm(c_ChP_inv, 2, true);
    // pout() << "Ham_Abs_tems " << Ham <<  "  Phi " << phi   << endl;
    // pout() << "In initialdata: WeylCurv " << wcurv <<  "  C-P " << ChP <<  "  rho_R " << RicciScalar/16/3.1415   << endl;

   

}

// #ifdef CH_USE_HDF5
// Things to do before outputting a checkpoint file
void ScalarFieldLevel::prePlotLevel()
{
    std::cout << "PrePlotLevel " <<  endl;

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


    BoxLoops::loop(
                MatterWeyl4<ScalarFieldWithPotential>(scalar_field, m_p.extraction_params.center, m_dx),
            m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);



    AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
    AMRReductions<VariableType::evolution> camr_reductions(m_gr_amr);
    double Ham = amr_reductions.norm(c_Ham_abs_terms, 2, true);
    double RicciScalar = amr_reductions.norm(c_ricci_scalar, 2, true);
    double phi = camr_reductions.norm(c_phi, 2, true);

    double wcurv = amr_reductions.norm(c_Weyl_curv, 2, true);
    double ChP = amr_reductions.norm(c_ChP_inv, 2, true);
    pout() << "Ham_Abs_tems " << Ham <<  "  Phi " << phi   << endl;
    pout() << "In preplot: WeylCurv " << wcurv <<  "  C-P " << ChP <<  "  rho_R " << RicciScalar/16/3.1415   << endl;

}
// #endif

// Things to do in RHS update, at each RK4 step
void ScalarFieldLevel::specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                       const double a_time)
{

    BoxLoops::loop(
    make_compute_pack(TraceARemoval(),
                    PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
    a_soln, a_soln, INCLUDE_GHOST_CELLS);
    

    SetValue set_other_values_zero(0.0, Interval(c_chi, NUM_VARS - 1));
    BoxLoops::loop(
        make_compute_pack(TraceARemoval(), set_other_values_zero)
        , a_soln,  a_rhs, INCLUDE_GHOST_CELLS);

    
    if (m_time > m_p.relaxtime)
    {
        // Enforce trace free A_ij and positive chi and alpha
        BoxLoops::loop(
            make_compute_pack(TraceARemoval(),
                            PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
            a_soln, a_soln, INCLUDE_GHOST_CELLS);

        // Calculate MatterCCZ4 right hand side with matter_t = ScalarField
        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
        MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
                    FourthOrderDerivatives>
            my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
                        m_p.formulation, m_p.G_Newton);
        BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
        

    }
    // else 
    // {
    //     BoxLoops::loop(
    //         make_compute_pack(TraceARemoval(),
    //                         PositiveChiAndAlpha(m_p.min_chi, m_p.min_lapse)),
    //         a_soln, a_soln, INCLUDE_GHOST_CELLS);            
        
    //     // Potential potential(m_p.potential_params);
    //     // ScalarFieldWithPotential scalar_field(potential);
        
    //     // MatterCCZ4RHS<ScalarFieldWithPotential, MovingPunctureGauge,
    //     //                 FourthOrderDerivatives>
    //     // my_ccz4_matter(scalar_field, m_p.ccz4_params, m_dx, m_p.sigma,
    //     //                     m_p.formulation, m_p.G_Newton);
    //     // BoxLoops::loop(my_ccz4_matter, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    // }
}

// Things to do at ODE update, after soln + rhs
void ScalarFieldLevel::specificUpdateODE(GRLevelData &a_soln,
                                         const GRLevelData &a_rhs, Real a_dt)
{

    

    // Enforce trace free A_ij
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);


    if (m_time < m_p.relaxtime)
    {
        std::cout << "CREATING INITIAL DATA: (Pi, K) " <<  endl;

        AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
        double RicciScalar = amr_reductions.norm(c_ricci_scalar, 2, true);
        double rho_mean = amr_reductions.norm(c_rho, 2, true);

        double K_mean = m_p.K_mean; // - RicciScalar ;

        Potential potential(m_p.potential_params);
        ScalarFieldWithPotential scalar_field(potential);
    
        ScalarInitData<ScalarFieldWithPotential> initialisation(
            scalar_field, m_dx, K_mean, rho_mean, m_p.G_Newton);

        BoxLoops::loop(initialisation, a_soln, a_soln, EXCLUDE_GHOST_CELLS, disable_simd());
        fillAllGhosts();

    }

}

void ScalarFieldLevel::preTagCells()
{
    // we don't need any ghosts filled for the fixed grids tagging criterion
    // used here so don't fill any
}

void ScalarFieldLevel::computeTaggingCriterion(FArrayBox &tagging_criterion,
                                               const FArrayBox &current_state)
{
    // BoxLoops::loop(
    //     FixedGridsTaggingCriterion(m_dx, m_level, 2.0 * m_p.L, m_p.center),
    //     current_state, tagging_criterion);
}

void ScalarFieldLevel::specificPostTimeStep()
{
// #ifdef USE_AHFINDER
//     if (m_p.AH_activate && m_level == m_p.AH_params.level_to_run)
//         m_bh_amr.m_ah_finder.solve(m_dt, m_time, m_restart_time);
// #endif
}
