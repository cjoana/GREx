#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "SetGrids.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "PoissonParameters.H"
#include "SetLevelData.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

// Setup grids
// Note that there is also an option to read in grids, but here we use tagging
// for the refinement
int set_grids(Vector<DisjointBoxLayout> &vectGrids, PoissonParameters &a_params)
{
    Vector<ProblemDomain> vectDomain;
    Vector<Real> vectDx;
    set_domains_and_dx(vectDomain, vectDx, a_params);

    int numlevels = a_params.numLevels;

    ParmParse pp;

    // grid generation parameters
    vectGrids.resize(numlevels);

    int maxLevel = numlevels - 1;
    Vector<Vector<Box>> newBoxes(numlevels);
    Vector<Vector<Box>> oldBoxes(numlevels);

    // determine grids dynamically, based on the tagging criteria
    Vector<LevelData<FArrayBox> *> vect_tagging_criterion(maxLevel + 1, NULL);

    // define base level first
    Vector<Vector<int>> procAssign(maxLevel + 1);
    domainSplit(vectDomain[0], oldBoxes[0], a_params.maxGridSize,
                a_params.blockFactor);
    procAssign[0].resize(oldBoxes[0].size());
    LoadBalance(procAssign[0], oldBoxes[0]);
    vectGrids[0].define(oldBoxes[0], procAssign[0], vectDomain[0]);
    vect_tagging_criterion[0] = new LevelData<FArrayBox>(
        vectGrids[0], 1, // only one value in this array
        IntVect::Zero);

    int topLevel = 0;
    bool moreLevels = (maxLevel > 0);

    int nesting_radius = 2;
    // create grid generation object
    BRMeshRefine meshrefine(vectDomain[0], a_params.refRatio,
                            a_params.fillRatio, a_params.blockFactor,
                            nesting_radius, a_params.maxGridSize);

    while (moreLevels)
    {
        // default is moreLevels = false
        // (only repeat loop in the case where a new level
        // is generated which is still less than maxLevel)
        moreLevels = false;

        int baseLevel = 0;
        int oldTopLevel = topLevel;

        // now initialize tagging criterion for this existing hierarchy
        for (int level = 0; level <= topLevel; level++)
        {
            RealVect dxLevel = vectDx[level] * RealVect::Unit;

            LevelData<FArrayBox> *temp_multigrid_vars;
            LevelData<FArrayBox> *temp_dpsi;

            // KC TODO: Make this an input
            IntVect ghosts = 1 * IntVect::Unit;
            temp_multigrid_vars = new LevelData<FArrayBox>(
                vectGrids[level], NUM_MULTIGRID_VARS, ghosts);
            temp_dpsi = new LevelData<FArrayBox>(vectGrids[level],
                                                 NUM_CONSTRAINT_VARS, ghosts);

            set_initial_conditions(*temp_multigrid_vars, *temp_dpsi, dxLevel,
                                   a_params, true);

            // set condition for regrid - use the integrability condition
            // integral
            set_regrid_condition(*vect_tagging_criterion[level],
                                 *temp_multigrid_vars, dxLevel, a_params);

            if (temp_multigrid_vars != NULL)
            {
                delete temp_multigrid_vars;
                temp_multigrid_vars = NULL;
            }
            if (temp_dpsi != NULL)
            {
                delete temp_dpsi;
                temp_dpsi = NULL;
            }
        }

        Vector<IntVectSet> tagVect(topLevel + 1);
        int tags_grow = 2;
        set_tag_cells(vect_tagging_criterion, tagVect, vectDx, vectDomain,
                      a_params.refineThresh, tags_grow, baseLevel,
                      topLevel + 1);

        int new_finest =
            meshrefine.regrid(newBoxes, tagVect, baseLevel, topLevel, oldBoxes);

        if (new_finest > topLevel)
        {
            topLevel++;
        }

        oldBoxes = newBoxes;

        //  no need to do this for the base level (already done)
        for (int lev = 1; lev <= topLevel; lev++)
        {
            // do load balancing
            procAssign[lev].resize(newBoxes[lev].size());
            LoadBalance(procAssign[lev], newBoxes[lev]);
            const DisjointBoxLayout newDBL(newBoxes[lev], procAssign[lev],
                                           vectDomain[lev]);
            vectGrids[lev] = newDBL;
            delete vect_tagging_criterion[lev];
            vect_tagging_criterion[lev] = new LevelData<FArrayBox>(
                vectGrids[lev], 1, IntVect::Zero); // again only one entry
        } // end loop over levels for initialization

        // figure out whether we need another pass through grid generation
        if ((topLevel < maxLevel) && (topLevel > oldTopLevel))
        {
            moreLevels = true;
        }
        // doesn't break anything but redundant I think?
        else
        {
            break;
        }
    } // end while moreLevels loop

    // clean up temp storage
    for (int ilev = 0; ilev < vect_tagging_criterion.size(); ilev++)
    {
        if (vect_tagging_criterion[ilev] != NULL)
        {
            delete vect_tagging_criterion[ilev];
            vect_tagging_criterion[ilev] = NULL;
        }
    }

    return 0;
}

// Set grid hierarchy from input file
void set_domains_and_dx(Vector<ProblemDomain> &vectDomain, Vector<Real> &vectDx,
                        PoissonParameters &a_params)
{

    vectDomain.resize(a_params.numLevels);
    vectDx.resize(a_params.numLevels);
    vectDx[0] = a_params.coarsestDx;
    for (int ilev = 1; ilev < a_params.numLevels; ilev++)
    {
        vectDx[ilev] = vectDx[ilev - 1] / a_params.refRatio[ilev - 1];
    }

    vectDomain[0] = a_params.coarsestDomain;
    for (int ilev = 1; ilev < a_params.numLevels; ilev++)
    {
        vectDomain[ilev] =
            refine(vectDomain[ilev - 1], a_params.refRatio[ilev - 1]);
    }
}

/*
  tag cells for refinement based on magnitude(tagging_criterion)
*/
void set_tag_cells(Vector<LevelData<FArrayBox> *> &vect_tagging_criterion,
                   Vector<IntVectSet> &tagVect, Vector<Real> &vectDx,
                   Vector<ProblemDomain> &vectDomain, const Real refine_thresh,
                   const int tags_grow, const int baseLevel, int numLevels)
{
    for (int lev = baseLevel; lev != numLevels; lev++)
    {
        IntVectSet local_tags;
        LevelData<FArrayBox> &level_tagging_criterion =
            *vect_tagging_criterion[lev];
        DisjointBoxLayout level_domain = level_tagging_criterion.getBoxes();
        DataIterator dit = level_tagging_criterion.dataIterator();

        // KC: this seems an odd way to refine - would expect threshold to
        // decrease with higher levels. It seems to work ok so leave it for now.
        Real max_tagging_criterion = 0;
        max_tagging_criterion = norm(level_tagging_criterion,
                                     level_tagging_criterion.interval(), 0);
        Real tagVal = max_tagging_criterion * refine_thresh;

        // now loop through grids and tag cells where tagging crierion > tagVal
        for (dit.reset(); dit.ok(); ++dit)
        {
            const Box thisBox = level_domain.get(dit());
            const FArrayBox &this_tagging_criterion =
                level_tagging_criterion[dit()];
            BoxIterator bit(thisBox);
            for (bit.begin(); bit.ok(); ++bit)
            {
                const IntVect &iv = bit();
                if (abs(this_tagging_criterion(iv)) >= tagVal)
                    local_tags |= iv;
            }
        } // end loop over grids on this level

        local_tags.grow(tags_grow);
        const Box &domainBox = vectDomain[lev].domainBox();
        local_tags &= domainBox;

        tagVect[lev] = local_tags;

    } // end loop over levels
}
