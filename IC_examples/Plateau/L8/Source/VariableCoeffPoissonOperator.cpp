#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "VariableCoeffPoissonOperator.H"
#include "AMRMultiGrid.H"
#include "AMRPoissonOpF_F.H"
#include "AverageF_F.H"
#include "BoxIterator.H"
#include "CoarseAverage.H"
#include "CoarseAverageFace.H"
#include "DebugOut.H"
#include "FORT_PROTO.H"
#include "FineInterp.H"
#include "InterpF_F.H"
#include "LayoutIterator.H"
#include "Misc.H"
#include "VariableCoeffPoissonOperatorF_F.H"

#include "NamespaceHeader.H"

// This file implements the key functions for the multi grid methods

void VariableCoeffPoissonOperator::residualI(LevelData<FArrayBox> &a_lhs,
                                             const LevelData<FArrayBox> &a_dpsi,
                                             const LevelData<FArrayBox> &a_rhs,
                                             bool a_homogeneous)
{
    CH_TIME("VariableCoeffPoissonOperator::residualI");

    LevelData<FArrayBox> &dpsi = (LevelData<FArrayBox> &)a_dpsi;
    Real dx = m_dx;
    const DisjointBoxLayout &dbl = a_lhs.disjointBoxLayout();
    DataIterator dit = dpsi.dataIterator();
    {
        CH_TIME("VariableCoeffPoissonOperator::residualIBC");

        for (dit.begin(); dit.ok(); ++dit)
        {
            m_bc(dpsi[dit], dbl[dit()], m_domain, dx, a_homogeneous);
        }
    }

    dpsi.exchange(dpsi.interval(), m_exchangeCopier);

    for (dit.begin(); dit.ok(); ++dit)
    {
        const Box &region = dbl[dit()];

#if CH_SPACEDIM == 1
        FORT_VCCOMPUTERES1D
#elif CH_SPACEDIM == 2
        FORT_VCCOMPUTERES2D
#elif CH_SPACEDIM == 3
        FORT_VCCOMPUTERES3D
#else
        This_will_not_compile !
#endif
            (CHF_FRA(a_lhs[dit]), CHF_CONST_FRA(dpsi[dit]),
             CHF_CONST_FRA(a_rhs[dit]), CHF_CONST_REAL(m_alpha),
             CHF_CONST_FRA((*m_aCoef)[dit]), CHF_CONST_REAL(m_beta),
             CHF_CONST_FRA((*m_bCoef)[dit]), CHF_BOX(region),
             CHF_CONST_REAL(m_dx));
    } // end loop over boxes
}

// this preconditioner first initializes dpsihat to (IA)dpsihat = rhshat
// (diagonization of L -- A is the matrix version of L)
// then smooths with a couple of passes of levelGSRB
void VariableCoeffPoissonOperator::preCond(LevelData<FArrayBox> &a_dpsi,
                                           const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::preCond");

    // diagonal term of this operator in:
    //
    //       alpha * a(i)
    //     + beta  * sum_over_dir (b(i-1/2*e_dir) + b(i+1/2*e_dir)) / (dx*dx)
    //
    // The inverse of this is our initial multiplier.

    int ncomp = a_dpsi.nComp();

    CH_assert(m_lambda.isDefined());
    CH_assert(a_rhs.nComp() == ncomp);
    CH_assert(m_bCoef->nComp() == ncomp);

    // Recompute the relaxation coefficient if needed.
    resetLambda();

    // don't need to use a Copier -- plain copy will do
    DataIterator dit = a_dpsi.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        // also need to average and sum face-centered bCoefs to cell-centers
        Box gridBox = a_rhs[dit].box();

        // approximate inverse
        a_dpsi[dit].copy(a_rhs[dit]);
        a_dpsi[dit].mult(m_lambda[dit], gridBox, 0, 0, ncomp);
    }

    relax(a_dpsi, a_rhs, 2);
}

void VariableCoeffPoissonOperator::applyOpI(LevelData<FArrayBox> &a_lhs,
                                            const LevelData<FArrayBox> &a_dpsi,
                                            bool a_homogeneous)
{
    CH_TIME("VariableCoeffPoissonOperator::applyOpI");
    LevelData<FArrayBox> &dpsi = (LevelData<FArrayBox> &)a_dpsi;
    Real dx = m_dx;
    const DisjointBoxLayout &dbl = a_lhs.disjointBoxLayout();
    DataIterator dit = dpsi.dataIterator();

    for (dit.begin(); dit.ok(); ++dit)
    {
        m_bc(dpsi[dit], dbl[dit()], m_domain, dx, a_homogeneous);
    }

    applyOpNoBoundary(a_lhs, a_dpsi);
}

void VariableCoeffPoissonOperator::applyOpNoBoundary(
    LevelData<FArrayBox> &a_lhs, const LevelData<FArrayBox> &a_dpsi)
{
    CH_TIME("VariableCoeffPoissonOperator::applyOpNoBoundary");

    LevelData<FArrayBox> &dpsi = (LevelData<FArrayBox> &)a_dpsi;

    const DisjointBoxLayout &dbl = a_lhs.disjointBoxLayout();
    DataIterator dit = dpsi.dataIterator();

    dpsi.exchange(dpsi.interval(), m_exchangeCopier);

    for (dit.begin(); dit.ok(); ++dit)
    {
        const Box &region = dbl[dit()];

#if CH_SPACEDIM == 1
        FORT_VCCOMPUTEOP1D
#elif CH_SPACEDIM == 2
        FORT_VCCOMPUTEOP2D
#elif CH_SPACEDIM == 3
        FORT_VCCOMPUTEOP3D
#else
        This_will_not_compile !
#endif
            (CHF_FRA(a_lhs[dit]), CHF_CONST_FRA(dpsi[dit]),
             CHF_CONST_REAL(m_alpha), CHF_CONST_FRA((*m_aCoef)[dit]),
             CHF_CONST_REAL(m_beta), CHF_CONST_FRA((*m_bCoef)[dit]),
             CHF_BOX(region), CHF_CONST_REAL(m_dx));
    } // end loop over boxes
}

void VariableCoeffPoissonOperator::restrictResidual(
    LevelData<FArrayBox> &a_resCoarse, LevelData<FArrayBox> &a_dpsiFine,
    const LevelData<FArrayBox> &a_rhsFine)
{
    CH_TIME("VariableCoeffPoissonOperator::restrictResidual");

    homogeneousCFInterp(a_dpsiFine);
    const DisjointBoxLayout &dblFine = a_dpsiFine.disjointBoxLayout();
    for (DataIterator dit = a_dpsiFine.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &dpsi = a_dpsiFine[dit];
        m_bc(dpsi, dblFine[dit()], m_domain, m_dx, true);
    }

    a_dpsiFine.exchange(a_dpsiFine.interval(), m_exchangeCopier);

    for (DataIterator dit = a_dpsiFine.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &dpsi = a_dpsiFine[dit];
        const FArrayBox &rhs = a_rhsFine[dit];
        FArrayBox &res = a_resCoarse[dit];

        const FArrayBox &thisACoef = (*m_aCoef)[dit];
        const FArrayBox &thisBCoef = (*m_bCoef)[dit];

        Box region = dblFine.get(dit());
        const IntVect &iv = region.smallEnd();
        IntVect civ = coarsen(iv, 2);

        res.setVal(0.0);

#if CH_SPACEDIM == 1
        FORT_RESTRICTRESVC1D
#elif CH_SPACEDIM == 2
        FORT_RESTRICTRESVC2D
#elif CH_SPACEDIM == 3
        FORT_RESTRICTRESVC3D
#else
        This_will_not_compile !
#endif
            (CHF_FRA_SHIFT(res, civ), CHF_CONST_FRA_SHIFT(dpsi, iv),
             CHF_CONST_FRA_SHIFT(rhs, iv), CHF_CONST_REAL(m_alpha),
             CHF_CONST_FRA_SHIFT(thisACoef, iv), CHF_CONST_REAL(m_beta),
             CHF_CONST_FRA_SHIFT(thisBCoef, iv), CHF_BOX_SHIFT(region, iv),
             CHF_CONST_REAL(m_dx));
    }
}

void VariableCoeffPoissonOperator::setAlphaAndBeta(const Real &a_alpha,
                                                   const Real &a_beta)
{
    m_alpha = a_alpha;
    m_beta = a_beta;

    // Our relaxation parameter is officially out of date!
    m_lambdaNeedsResetting = true;
}

void VariableCoeffPoissonOperator::setCoefs(
    const RefCountedPtr<LevelData<FArrayBox>> &a_aCoef,
    const RefCountedPtr<LevelData<FArrayBox>> &a_bCoef, const Real &a_alpha,
    const Real &a_beta)
{

    m_alpha = a_alpha;
    m_beta = a_beta;

    m_aCoef = a_aCoef;
    m_bCoef = a_bCoef;

    // Our relaxation parameter is officially out of date!
    m_lambdaNeedsResetting = true;
}

void VariableCoeffPoissonOperator::resetLambda()
{

    if (m_lambdaNeedsResetting)
    {

        Real scale = 1.0 / (m_dx * m_dx);

        // Compute it box by box, point by point
        for (DataIterator dit = m_lambda.dataIterator(); dit.ok(); ++dit)
        {
            FArrayBox &lambdaFab = m_lambda[dit];
            const FArrayBox &aCoefFab = (*m_aCoef)[dit];
            const FArrayBox &bCoefFab = (*m_bCoef)[dit];
            const Box &curBox = lambdaFab.box();

            // Compute the diagonal term
            lambdaFab.copy(aCoefFab);
            lambdaFab.mult(m_alpha);

            // Add in the Laplacian term 6.0*m_beta/(m_dx*m_dx)
            // KC TODO: Should implement other adjustments for NL terms,
            // but appears to converge without
            lambdaFab.plus(2.0 * SpaceDim * m_beta / (m_dx * m_dx));

            // Take its reciprocal
            lambdaFab.invert(1.0);
        }

        // Lambda is reset.
        m_lambdaNeedsResetting = false;
    }
}

// Compute the reciprocal of the diagonal entry of the operator matrix
void VariableCoeffPoissonOperator::computeLambda()
{
    CH_TIME("VariableCoeffPoissonOperator::computeLambda");

    CH_assert(!m_lambda.isDefined());

    // Define lambda
    m_lambda.define(m_aCoef->disjointBoxLayout(), m_aCoef->nComp());
    resetLambda();
}

// NB This was removed as we do not need it - may want to reinstate, if so see
// MG examples for reflux operator
void VariableCoeffPoissonOperator::reflux(
    const LevelData<FArrayBox> &a_dpsiFine, const LevelData<FArrayBox> &a_dpsi,
    LevelData<FArrayBox> &a_residual,
    AMRLevelOp<LevelData<FArrayBox>> *a_finerOp)
{

    // pout() << "Warning :: VariableCoeffPoissonOperator::reflux - called but
    // not implemented" << endl;
}

void VariableCoeffPoissonOperator::levelGSRB(LevelData<FArrayBox> &a_dpsi,
                                             const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::levelGSRB");

    CH_assert(a_dpsi.isDefined());
    CH_assert(a_rhs.isDefined());
    CH_assert(a_dpsi.ghostVect() >= IntVect::Unit);
    CH_assert(a_dpsi.nComp() == a_rhs.nComp());

    // Recompute the relaxation coefficient if needed.
    resetLambda();

    const DisjointBoxLayout &dbl = a_dpsi.disjointBoxLayout();

    DataIterator dit = a_dpsi.dataIterator();

    // do first red, then black passes
    for (int whichPass = 0; whichPass <= 1; whichPass++)
    {
        CH_TIMERS("VariableCoeffPoissonOperator::levelGSRB::Compute");

        // fill in intersection of ghostcells and a_dpsi's boxes
        {
            CH_TIME(
                "VariableCoeffPoissonOperator::levelGSRB::homogeneousCFInterp");
            homogeneousCFInterp(a_dpsi);
        }

        {
            CH_TIME("VariableCoeffPoissonOperator::levelGSRB::exchange");
            a_dpsi.exchange(a_dpsi.interval(), m_exchangeCopier);
        }

        {
            CH_TIME("VariableCoeffPoissonOperator::levelGSRB::BCs");
            // now step through grids...
            for (dit.begin(); dit.ok(); ++dit)
            {
                // invoke physical BC's where necessary
                m_bc(a_dpsi[dit], dbl[dit()], m_domain, m_dx, true);
            }
        }

        for (dit.begin(); dit.ok(); ++dit)
        {
            const Box &region = dbl.get(dit());

#if CH_SPACEDIM == 1
            FORT_GSRBHELMHOLTZVC1D
#elif CH_SPACEDIM == 2
            FORT_GSRBHELMHOLTZVC2D
#elif CH_SPACEDIM == 3
            FORT_GSRBHELMHOLTZVC3D
#else
            This_will_not_compile !
#endif
                (CHF_FRA(a_dpsi[dit]), CHF_CONST_FRA(a_rhs[dit]),
                 CHF_BOX(region), CHF_CONST_REAL(m_dx), CHF_CONST_REAL(m_alpha),
                 CHF_CONST_FRA((*m_aCoef)[dit]), CHF_CONST_REAL(m_beta),
                 CHF_CONST_FRA((*m_bCoef)[dit]), CHF_CONST_FRA(m_lambda[dit]),
                 CHF_CONST_INT(whichPass));
        } // end loop through grids
    }     // end loop through red-black
}

void VariableCoeffPoissonOperator::levelMultiColor(
    LevelData<FArrayBox> &a_dpsi, const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::levelMultiColor");
    MayDay::Abort(
        "VariableCoeffPoissonOperator::levelMultiColor - Not implemented");
}

void VariableCoeffPoissonOperator::looseGSRB(LevelData<FArrayBox> &a_dpsi,
                                             const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::looseGSRB");
    MayDay::Abort("VariableCoeffPoissonOperator::looseGSRB - Not implemented");
}

void VariableCoeffPoissonOperator::overlapGSRB(
    LevelData<FArrayBox> &a_dpsi, const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::overlapGSRB");
    MayDay::Abort(
        "VariableCoeffPoissonOperator::overlapGSRB - Not implemented");
}

void VariableCoeffPoissonOperator::levelGSRBLazy(
    LevelData<FArrayBox> &a_dpsi, const LevelData<FArrayBox> &a_rhs)
{
    CH_TIME("VariableCoeffPoissonOperator::levelGSRBLazy");
    MayDay::Abort(
        "VariableCoeffPoissonOperator::levelGSRBLazy - Not implemented");
}

void VariableCoeffPoissonOperator::levelJacobi(
    LevelData<FArrayBox> &a_dpsi, const LevelData<FArrayBox> &a_rhs)
{

    CH_TIME("VariableCoeffPoissonOperator::levelJacobi");

    // Recompute the relaxation coefficient if needed.
    resetLambda();

    LevelData<FArrayBox> resid;
    create(resid, a_rhs);

    // Get the residual
    residual(resid, a_dpsi, a_rhs, true);

    // Multiply by the weights
    DataIterator dit = m_lambda.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        resid[dit].mult(m_lambda[dit]);
    }

    // Do the Jacobi relaxation
    incr(a_dpsi, resid, 0.5);

    // exchange ghost cells
    a_dpsi.exchange(a_dpsi.interval(), m_exchangeCopier);
}

// Removed as only needed for fluxes, may need to reinstate in future,
// if so see MG examples
void VariableCoeffPoissonOperator::getFlux(FArrayBox &a_flux,
                                           const FArrayBox &a_data,
                                           const FArrayBox &b_data,
                                           const Box &a_facebox, int a_dir,
                                           int a_ref) const
{

    //  pout() << "Warning :: VariableCoeffPoissonOperator::getFlux - called but
    //  not implemented" << endl;
}

// set the time
void VariableCoeffPoissonOperator::setTime(Real a_time)
{
    // Jot down the time.
    m_time = a_time;

    // Interpolate the b coefficient data if necessary / possible. If
    // the B coefficient depends upon the solution, the operator is nonlinear
    // and the integrator must decide how to treat it.
    if (!m_bCoefInterpolator.isNull() &&
        !m_bCoefInterpolator->dependsUponSolution())
        m_bCoefInterpolator->interpolate(*m_bCoef, a_time);

    // Our relaxation parameter is officially out of date!
    m_lambdaNeedsResetting = true;

    // Set the time on the boundary holder.
    m_bc.setTime(a_time);

    // Notify our observers that the time has been set.
    // FIXME: Must implement response of multigrid operators!
    notifyObserversOfChange();
}

#include "NamespaceFooter.H"
