/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef REPROCESSINGLEVEL_HPP_
#define REPROCESSINGLEVEL_HPP_

#include "CustomExtraction.hpp"
#include "GRAMRLevel.hpp"

class ReprocessingLevel : public GRAMRLevel
{
    friend class DefaultLevelFactory<ReprocessingLevel>;
    // Inherit the contructors from GRAMRLevel
    using GRAMRLevel::GRAMRLevel;

    // initialize data
    virtual void initialData() { m_state_new.setVal(0.); }

    void postRestart()
    {
        // Add code here to do what you need it to do on each level
        // Note that if you want the AMRInterpolator you need to define it
        // and set it here (currently it is just a nullptr)
        pout() << "The time is " << m_time << " on level " << m_level
               << ". Your wish is my command." << endl;

        // as an example, on the coarsest level do a simple extraction
        // (NB will still take data from finest level which covers the points)
        if (m_level == 0)
        {
            // set up an interpolator
            // pass the boundary params so that we can use symmetries if
            // applicable
            AMRInterpolator<Lagrange<4>> interpolator(
                m_gr_amr, m_p.origin, m_p.dx, m_p.boundary_params,
                m_p.verbosity);

            // this should fill all ghosts including the boundary ones according
            // to the conditions set in params.txt
            interpolator.refresh();

            // set up the query and execute it
            int num_points = 4;
            CustomExtraction extraction(c_chi, num_points, m_p.L, m_p.center,
                                        m_dt, m_time);
            extraction.execute_query(&interpolator, "outputs");
        }
    }

    virtual void specificEvalRHS(GRLevelData &a_soln, GRLevelData &a_rhs,
                                 const double a_time)
    {
    }

    virtual void computeTaggingCriterion(FArrayBox &tagging_criterion,
                                         const FArrayBox &current_state){};
};

#endif /* REPROCESSINGLEVEL_HPP_ */
