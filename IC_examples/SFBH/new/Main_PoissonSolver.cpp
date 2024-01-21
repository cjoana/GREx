/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoundaryConditions.hpp"
#include "CoarseAverage.H"
#include "DebugDump.H"
#include "FABView.H"
#include "FArrayBox.H"
#include "FourthOrderCFInterp.H"
#include "LevelData.H"
#include "LoadBalance.H"
#include "MultilevelLinearOp.H"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "ReadInput.H"
#include "SetBCs.H"
#include "SetGrids.H"
#include "SetLevelData.H"
#include "UsingNamespace.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "WriteOutput.H"
#include "computeNorm.H"
#include "computeSum.H"
#include <iostream>

#ifdef CH_Linux
// Should be undefined by default
//#define TRAP_FPE
#undef TRAP_FPE
#endif

#ifdef TRAP_FPE
static void enableFpExceptions();
#endif

using std::cerr;

// Useful for writing convergence
int writeFile(std::string name, Real value1, Real value2)
{
    ofstream myfile;
    myfile.open(name + ".txt", std::ios_base::app);
    myfile << value1 << "	" << value2 << endl;
    return 0;
}

// Sets up and runs the solver
// The equation solved is: [aCoef*I + bCoef*Laplacian](dpsi) = rhs
// We assume conformal flatness, K=const and Momentum constraint satisfied
// by chosen Aij (for now sourced by Bowen York data),
// lapse = 1 shift = 0, phi is the scalar field and is used to
// calculate the rhs, Pi = dphidt = 0.
int poissonSolve(const Vector<DisjointBoxLayout> &a_grids,
                 const PoissonParameters &a_params)
{
    // the params reader
    ParmParse pp;

    // create the necessary hierarchy of data components
    int nlevels = a_params.numLevels;

    // the user set initial conditions - currently including psi, phi, Pi, K_ij
    Vector<LevelData<FArrayBox> *> multigrid_vars(nlevels, NULL);

    // the correction to the conformal factor and V^i -
    // what the solver solves for
    Vector<LevelData<FArrayBox> *> dpsi(nlevels, NULL);

    // the solver vars:
    // the rhs source term
    Vector<LevelData<FArrayBox> *> rhs(nlevels, NULL);
    // the coeff for the I term
    Vector<RefCountedPtr<LevelData<FArrayBox>>> aCoef(nlevels);
    // the coeff for the Laplacian
    Vector<RefCountedPtr<LevelData<FArrayBox>>> bCoef(nlevels);

    // the error function
    Vector<LevelData<FArrayBox> *> error(nlevels, NULL);

    // Grid params
    Vector<ProblemDomain> vectDomains(nlevels); // the domains
    Vector<RealVect> vectDx(nlevels); // the grid spacings on each level

    // Set temp vars at coarsest level values
    RealVect dxLev = RealVect::Unit;
    dxLev *= a_params.coarsestDx;
    ProblemDomain domLev(a_params.coarsestDomain);

    // currently only one ghost needed for 2nd order stencils
    int num_ghosts = 1;
    IntVect ghosts = num_ghosts * IntVect::Unit;
    IntVect no_ghosts = IntVect::Zero;

    // Declare variables here, with num comps, and ghosts for all
    for (int ilev = 0; ilev < nlevels; ilev++)
    {
        multigrid_vars[ilev] =
            new LevelData<FArrayBox>(a_grids[ilev], NUM_MULTIGRID_VARS, ghosts);
        dpsi[ilev] = new LevelData<FArrayBox>(a_grids[ilev],
                                              NUM_CONSTRAINT_VARS, ghosts);
        rhs[ilev] = new LevelData<FArrayBox>(a_grids[ilev], NUM_CONSTRAINT_VARS,
                                             no_ghosts);
        error[ilev] = new LevelData<FArrayBox>(a_grids[ilev],
                                               NUM_CONSTRAINT_VARS, no_ghosts);
        aCoef[ilev] =
            RefCountedPtr<LevelData<FArrayBox>>(new LevelData<FArrayBox>(
                a_grids[ilev], NUM_CONSTRAINT_VARS, no_ghosts));
        bCoef[ilev] =
            RefCountedPtr<LevelData<FArrayBox>>(new LevelData<FArrayBox>(
                a_grids[ilev], NUM_CONSTRAINT_VARS, no_ghosts));
        vectDomains[ilev] = domLev;
        vectDx[ilev] = dxLev;

        bool set_matter = true;
        if (a_params.readin_matter_data)
        {
            set_matter = false;
            // just set initial guess for psi and zero dpsi
            set_initial_conditions(*multigrid_vars[ilev], *dpsi[ilev],
                                   vectDx[ilev], a_params, set_matter);
        }
        else
        {
            pout() << "Set matter data using analytic data level " << ilev
                   << endl;
            // set initial guess for psi and zero dpsi
            // and values for other multigrid matter sources (Kij set below)
            set_initial_conditions(*multigrid_vars[ilev], *dpsi[ilev],
                                   vectDx[ilev], a_params, set_matter);
        }

        // prepare temp dx, domain vars for next level
        dxLev /= a_params.refRatio[ilev];
        domLev.refine(a_params.refRatio[ilev]);
    }

    // read in the data and make sure boundary cells filled
    if (a_params.readin_matter_data)
    {
        pout() << "Set matter data using read in from file "
               << a_params.input_filename << endl;
        read_matter_data(multigrid_vars, a_grids, a_params,
                         a_params.input_filename);

        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            // fill the boundary cells in case needed (eg, because no ghosts in
            // input)
            BoundaryConditions solver_boundaries;
            solver_boundaries.define(vectDx[ilev][0], a_params.boundary_params,
                                     vectDomains[ilev], num_ghosts);

            // this will populate the multigrid boundaries according to the BCs
            // set some will still just be zeros but this should be ok for now
            solver_boundaries.fill_multigrid_boundaries(Side::Lo,
                                                        *multigrid_vars[ilev]);

            solver_boundaries.fill_multigrid_boundaries(Side::Hi,
                                                        *multigrid_vars[ilev]);

            // For interlevel ghosts
            if (ilev > 0)
            {
                // Fill using 4th order method which fills corner ghosts too
                int num_ghosts = 1;
                FourthOrderCFInterp m_patcher;
                m_patcher.define(a_grids[ilev], a_grids[ilev - 1],
                                 NUM_MULTIGRID_VARS, vectDomains[ilev - 1],
                                 a_params.refRatio[ilev], num_ghosts);
                m_patcher.coarseFineInterp(*multigrid_vars[ilev],
                                           *multigrid_vars[ilev - 1], 0, 0,
                                           NUM_MULTIGRID_VARS);
            }

            // For intralevel ghosts
            DisjointBoxLayout grown_grids;
            solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                         a_grids[ilev]);

            Copier exchange_copier;
            exchange_copier.exchangeDefine(grown_grids, ghosts);
            multigrid_vars[ilev]->exchange(multigrid_vars[ilev]->interval(),
                                           exchange_copier);
        }
    }

    // set up linear operator
    int lBase = 0;
    MultilevelLinearOp<FArrayBox> mlOp;
    BiCGStabSolver<Vector<LevelData<FArrayBox> *>>
        solver; // define solver object

    // default or read in solver params
    int numMGIter = 1;
    pp.query("numMGIterations", numMGIter);
    mlOp.m_num_mg_iterations = numMGIter;

    int numMGSmooth = 4;
    pp.query("numMGsmooth", numMGSmooth);
    mlOp.m_num_mg_smooth = numMGSmooth;

    int preCondSolverDepth = -1;
    pp.query("preCondSolverDepth", preCondSolverDepth);
    mlOp.m_preCondSolverDepth = preCondSolverDepth;

    Real tolerance = 1.0e-7;
    pp.query("tolerance", tolerance);

    int max_iter = 10;
    pp.query("max_iterations", max_iter);

    int max_NL_iter = 4;
    pp.query("max_NL_iterations", max_NL_iter);

    // Iterate linearised Poisson eqn for NL solution
    Real dpsi_norm0 = 0.0;
    Real dpsi_norm1 = 1.0;
    for (int NL_iter = 0; NL_iter < max_NL_iter; NL_iter++)
    {

        pout() << "Main Loop Iteration " << (NL_iter + 1) << " out of "
               << max_NL_iter << endl;

        // This function sets K per our ansatz depending on periodicity
        // It also assigns and updates the values of \bar Aij in the domain
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            set_update_Kij(*multigrid_vars[ilev], *rhs[ilev], vectDx[ilev],
                           a_params);
        }

        // need to fill interlevel and intralevel ghosts in multigrid_vars
        // so that derivatives can be computed within the grid
        // also want to correct coarser levels for finer data
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            // fill the boundary cells and ghosts
            BoundaryConditions solver_boundaries;
            solver_boundaries.define(vectDx[ilev][0], a_params.boundary_params,
                                     vectDomains[ilev], num_ghosts);

            // this will populate the multigrid boundaries according to the BCs
            // in particular it will fill cells for Aij, and updated K
            solver_boundaries.fill_multigrid_boundaries(
                Side::Lo, *multigrid_vars[ilev]); //, Interval(c_K_0, c_A33_0));
            solver_boundaries.fill_multigrid_boundaries(
                Side::Hi, *multigrid_vars[ilev]); //, Interval(c_K_0, c_A33_0));

            // To define an exchange copier to cover the outer ghosts
            DisjointBoxLayout grown_grids;
            solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                         a_grids[ilev]);

            // Fill the interlevel ghosts from a coarser level
            if (ilev > 0)
            {
                // Fill using 4th order method which fills corner ghosts too
                int num_ghosts = 1;
                FourthOrderCFInterp m_patcher;
                m_patcher.define(a_grids[ilev], a_grids[ilev - 1],
                                 NUM_MULTIGRID_VARS, vectDomains[ilev - 1],
                                 a_params.refRatio[ilev], num_ghosts);
                m_patcher.coarseFineInterp(*multigrid_vars[ilev],
                                           *multigrid_vars[ilev - 1], 0, 0,
                                           NUM_MULTIGRID_VARS);
            }

            // exchange the interior ghosts
            Copier exchange_copier;
            exchange_copier.exchangeDefine(grown_grids, ghosts);
            multigrid_vars[ilev]->exchange(multigrid_vars[ilev]->interval(),
                                           exchange_copier);
        }

        // Calculate values for coefficients here - see SetLevelData.cpp
        // for details
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            set_a_coef(*aCoef[ilev], *multigrid_vars[ilev], a_params,
                       vectDx[ilev]);
            set_b_coef(*bCoef[ilev], a_params, vectDx[ilev]);
            set_rhs(*rhs[ilev], *multigrid_vars[ilev], vectDx[ilev], a_params);
            set_error(*error[ilev], *rhs[ilev], *multigrid_vars[ilev],
                      vectDx[ilev], a_params);
        }

        // Check integrability conditions on rhs if periodic
        if (a_params.periodic_directions_exist)
        {
            // Calculate values for integrand here
            pout() << "Computing integrability of rhs for periodic domain... "
                   << endl;
            for (int icomp = 0; icomp < NUM_CONSTRAINT_VARS; icomp++)
            {
                Real integral =
                    computeSum(rhs, a_params.refRatio, a_params.coarsestDx,
                               Interval(icomp, icomp));

                pout() << "Integral of rhs " << icomp << " is " << integral
                       << endl;
                pout() << "(This should be a small number)" << endl;
            }
        }

        // check at this point if converged or diverging and if so exit NL
        // iteration
        dpsi_norm0 = computeNorm(error, a_params.refRatio, a_params.coarsestDx,
                                 Interval(0, 0));
        dpsi_norm1 =
            computeNorm(error, a_params.refRatio, a_params.coarsestDx,
                        Interval(c_V1, c_V3)); // NUM_CONSTRAINT_VARS - 1));
        pout() << "The norm of error Ham before step " << NL_iter << " is "
               << dpsi_norm0 << endl;
        pout() << "The norm of error Mom before step " << NL_iter << " is "
               << dpsi_norm1 << endl;

        int rank;
        MPI_Comm_rank(Chombo_MPI::comm, &rank);
        if (rank == 0)
        {
            writeFile("convergence", dpsi_norm0, dpsi_norm1);
        }

        if ((dpsi_norm0 < tolerance && dpsi_norm1 < tolerance) ||
            dpsi_norm0 > 1e5 || dpsi_norm1 > 1e5)
        {
            break;
        }

        // set up solver factory
        RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>> opFactory =
            RefCountedPtr<AMRLevelOpFactory<LevelData<FArrayBox>>>(
                defineOperatorFactory(a_grids, vectDomains, aCoef, bCoef,
                                      a_params));

        // define the multi level operator
        mlOp.define(a_grids, a_params.refRatio, vectDomains, vectDx, opFactory,
                    lBase);
        // set the solver params
        bool homogeneousBC = false;
        solver.define(&mlOp, homogeneousBC);
        solver.m_verbosity = a_params.verbosity;
        solver.m_normType = 2;
        solver.m_eps = tolerance;
        solver.m_imax = max_iter;

        // output on first timestep pre solver to check initial data
        if (NL_iter == 0)
        {
            // output the data before the solver acts to check starting
            // conditions
            output_solver_data(dpsi, error, multigrid_vars, a_grids, a_params,
                               NL_iter);
        }

        // Engage!
        solver.solve(dpsi, rhs);

        // Add the solution to the linearised eqn to the previous iteration
        // ie psi -> psi + dpsi
        // need to fill interlevel and intralevel ghosts first in dpsi
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            // For interlevel ghosts in dpsi
            if (ilev > 0)
            {
                // Fill using 4th order method which fills corner ghosts too
                int num_ghosts = 1;
                FourthOrderCFInterp m_patcher;
                m_patcher.define(a_grids[ilev], a_grids[ilev - 1],
                                 NUM_CONSTRAINT_VARS, vectDomains[ilev - 1],
                                 a_params.refRatio[ilev], num_ghosts);
                m_patcher.coarseFineInterp(*dpsi[ilev], *dpsi[ilev - 1], 0, 0,
                                           NUM_CONSTRAINT_VARS);
            }

            // For intralevel ghosts - this is done in set_update_phi0
            // but need the exchange copier object to do this
            BoundaryConditions solver_boundaries;
            solver_boundaries.define(vectDx[ilev][0], a_params.boundary_params,
                                     vectDomains[ilev], num_ghosts);
            // For intralevel ghosts
            DisjointBoxLayout grown_grids;
            solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                         a_grids[ilev]);
            Copier exchange_copier;
            exchange_copier.exchangeDefine(grown_grids, ghosts);

            // now the update
            set_update_psi0(*multigrid_vars[ilev], *dpsi[ilev], exchange_copier,
                            a_params);
        }

        // Again update the multigrid ghost values for calculation
        // of Kij at next step
        for (int ilev = 0; ilev < nlevels; ilev++)
        {
            // For interlevel ghosts
            if (ilev > 0)
            {
                // Fill using 4th order method which fills corner ghosts too
                int num_ghosts = 1;
                FourthOrderCFInterp m_patcher;
                m_patcher.define(a_grids[ilev], a_grids[ilev - 1],
                                 NUM_MULTIGRID_VARS, vectDomains[ilev - 1],
                                 a_params.refRatio[ilev], num_ghosts);
                m_patcher.coarseFineInterp(*multigrid_vars[ilev],
                                           *multigrid_vars[ilev - 1], 0, 0,
                                           NUM_MULTIGRID_VARS);
            }

            BoundaryConditions solver_boundaries;
            solver_boundaries.define(vectDx[ilev][0], a_params.boundary_params,
                                     vectDomains[ilev], num_ghosts);

            // Update the solver vars in the boundary cells, esp
            // for reflective since dpsi boundary cells output as zero
            bool filling_solver_vars = true;
            solver_boundaries.fill_multigrid_boundaries(
                Side::Lo, *multigrid_vars[ilev], Interval(c_psi_reg, c_U_0),
                filling_solver_vars);
            solver_boundaries.fill_multigrid_boundaries(
                Side::Hi, *multigrid_vars[ilev], Interval(c_psi_reg, c_U_0),
                filling_solver_vars);

            // For intralevel ghosts
            DisjointBoxLayout grown_grids;
            solver_boundaries.expand_grids_to_boundaries(grown_grids,
                                                         a_grids[ilev]);
            Copier exchange_copier;
            exchange_copier.exchangeDefine(grown_grids, ghosts);
            multigrid_vars[ilev]->exchange(multigrid_vars[ilev]->interval(),
                                           exchange_copier);
        }

        // output the data after the solver acts to check updates
                output_solver_data(dpsi, error, multigrid_vars, a_grids,
                a_params, NL_iter + 1);

    } // end NL iteration loop

    pout() << "The norm of error Ham at the final step was " << dpsi_norm0
           << endl;
    pout() << "The norm of error Mom at the final step was " << dpsi_norm1
           << endl;

    // Mayday if result not converged at all - using a fairly generous threshold
    // for this as usually non convergence means everything goes nuts
    if (dpsi_norm0 > 1e-1 || dpsi_norm1 > 1e-1)
    {
        MayDay::Error(
            "NL iterations did not converge - may need a better initial guess");
    }

    // now output final data in a form which can be read as a checkpoint file
    // for the GRChombo AMR time dependent runs
    output_final_data(multigrid_vars, a_grids, vectDx, vectDomains, a_params,
                      a_params.output_filename);

    // clean up data
    for (int level = 0; level < multigrid_vars.size(); level++)
    {
        if (multigrid_vars[level] != NULL)
        {
            delete multigrid_vars[level];
            multigrid_vars[level] = NULL;
        }
        if (rhs[level] != NULL)
        {
            delete rhs[level];
            rhs[level] = NULL;
        }
        if (error[level] != NULL)
        {
            delete error[level];
            error[level] = NULL;
        }
        if (dpsi[level] != NULL)
        {
            delete dpsi[level];
            dpsi[level] = NULL;
        }
    }

    int exitStatus = solver.m_exitStatus;
    // note that for AMRMultiGrid, success = 1.
    exitStatus -= 1;
    return exitStatus;
}

// Main function - keep this simple with just setup and read params
int main(int argc, char *argv[])
{
    int status = 0;
#ifdef CH_MPI
    MPI_Init(&argc, &argv);
#endif
    // Scoping trick
    {
        if (argc < 2)
        {
            cerr << " usage " << argv[0] << " <input_file_name> " << endl;
            exit(0);
        }

        char *inFile = argv[1];
        ParmParse pp(argc - 2, argv + 2, NULL, inFile);

        PoissonParameters params;
        Vector<DisjointBoxLayout> grids;

        // read params from file
        getPoissonParameters(params);

        // set up the grids, using the rhs for tagging to decide
        // where needs additional levels
        if (params.readin_matter_data)
        {
            read_grids(grids, params, params.input_filename);
        }
        else
        {
            set_grids(grids, params);
        }

        // Solve the equations!
        status = poissonSolve(grids, params);

    } // End scoping trick

#ifdef CH_MPI
    MPI_Finalize();
#endif
    return status;
}
