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

    // determine grids dynamically, based on grad(RHS)
    // will need temp storage for RHS
    Vector<LevelData<FArrayBox> *> vectRHS(maxLevel + 1, NULL);

    // define base level first
    Vector<Vector<int>> procAssign(maxLevel + 1);
    domainSplit(vectDomain[0], oldBoxes[0], a_params.maxGridSize,
                a_params.blockFactor);
    procAssign[0].resize(oldBoxes[0].size());
    LoadBalance(procAssign[0], oldBoxes[0]);
    vectGrids[0].define(oldBoxes[0], procAssign[0], vectDomain[0]);
    vectRHS[0] = new LevelData<FArrayBox>(vectGrids[0], 1, IntVect::Zero);

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

        // now initialize RHS for this existing hierarchy
        for (int level = 0; level <= topLevel; level++)
        {
            RealVect dxLevel = vectDx[level] * RealVect::Unit;

            LevelData<FArrayBox> *temp_multigrid_vars;
            LevelData<FArrayBox> *temp_dpsi;

            temp_multigrid_vars = new LevelData<FArrayBox>(
                vectGrids[level], NUM_MULTIGRID_VARS, 3 * IntVect::Unit);
            temp_dpsi = new LevelData<FArrayBox>(vectGrids[level], 1,
                                                 3 * IntVect::Unit);

            set_initial_conditions(*temp_multigrid_vars, *temp_dpsi, dxLevel,
                                   a_params);

            // set condition for regrid - use the integrability condition
            // integral
            set_regrid_condition(*vectRHS[level], *temp_multigrid_vars, dxLevel,
                                 a_params);

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
        set_tag_cells(vectRHS, tagVect, vectDx, vectDomain,
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
            delete vectRHS[lev];
            vectRHS[lev] =
                new LevelData<FArrayBox>(vectGrids[lev], 1, IntVect::Zero);
        } // end loop over levels for initialization

        // figure out whether we need another pass through grid generation
        if ((topLevel < maxLevel) && (topLevel > oldTopLevel))
            moreLevels = true;

    } // end while moreLevels loop

    // clean up temp storage
    for (int ilev = 0; ilev < vectRHS.size(); ilev++)
    {
        if (vectRHS[ilev] != NULL)
        {
            delete vectRHS[ilev];
            vectRHS[ilev] = NULL;
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
  tag cells for refinement based on magnitude(RHS)
*/
void set_tag_cells(Vector<LevelData<FArrayBox> *> &vectRHS,
                   Vector<IntVectSet> &tagVect, Vector<Real> &vectDx,
                   Vector<ProblemDomain> &vectDomain, const Real refine_thresh,
                   const int tags_grow, const int baseLevel, int numLevels)
{
    for (int lev = baseLevel; lev != numLevels; lev++)
    {
        IntVectSet local_tags;
        LevelData<FArrayBox> &levelRhs = *vectRHS[lev];
        DisjointBoxLayout level_domain = levelRhs.getBoxes();
        DataIterator dit = levelRhs.dataIterator();

        Real maxRHS = 0;

        maxRHS = norm(levelRhs, levelRhs.interval(), 0);

        Real tagVal = maxRHS * refine_thresh;

        // now loop through grids and tag cells where RHS > tagVal
        for (dit.reset(); dit.ok(); ++dit)
        {
            const Box thisBox = level_domain.get(dit());
            const FArrayBox &thisRhs = levelRhs[dit()];
            BoxIterator bit(thisBox);
            for (bit.begin(); bit.ok(); ++bit)
            {
                const IntVect &iv = bit();
                if (abs(thisRhs(iv)) >= tagVal)
                    local_tags |= iv;
            }
        } // end loop over grids on this level

        local_tags.grow(tags_grow);
        const Box &domainBox = vectDomain[lev].domainBox();
        local_tags &= domainBox;

        tagVect[lev] = local_tags;

    } // end loop over levels
}
