#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "VariableCoeffPoissonOperatorFactory.H"
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
#include "NamespaceHeader.H"
#include "VariableCoeffPoissonOperator.H"
#include "VariableCoeffPoissonOperatorF_F.H"

// function to define op factory
AMRLevelOpFactory<LevelData<FArrayBox>> *
defineOperatorFactory(const Vector<DisjointBoxLayout> &a_grids,
                      const Vector<ProblemDomain> &a_vectDomain,
                      Vector<RefCountedPtr<LevelData<FArrayBox>>> &a_aCoef,
                      Vector<RefCountedPtr<LevelData<FArrayBox>>> &a_bCoef,
                      const PoissonParameters &a_params)
{
    ParmParse pp2;

    VariableCoeffPoissonOperatorFactory *opFactory =
        new VariableCoeffPoissonOperatorFactory;

    opFactory->define(a_params.coarsestDomain, a_grids, a_params.refRatio,
                      a_params.coarsestDx, &ParseBC, a_params.alpha, a_aCoef,
                      a_params.beta, a_bCoef);

    if (a_params.coefficient_average_type >= 0)
    {
        opFactory->m_coefficient_average_type =
            a_params.coefficient_average_type;
    }

    return (AMRLevelOpFactory<LevelData<FArrayBox>> *)opFactory;
}

//-----------------------------------------------------------------------

// Default constructor
VariableCoeffPoissonOperatorFactory::VariableCoeffPoissonOperatorFactory()
{
    setDefaultValues();
}

//  AMR Factory define function
void VariableCoeffPoissonOperatorFactory::define(
    const ProblemDomain &a_coarseDomain,
    const Vector<DisjointBoxLayout> &a_grids, const Vector<int> &a_refRatios,
    const Real &a_coarsedx, BCHolder a_bc, const Real &a_alpha,
    Vector<RefCountedPtr<LevelData<FArrayBox>>> &a_aCoef, const Real &a_beta,
    Vector<RefCountedPtr<LevelData<FArrayBox>>> &a_bCoef)
{

    CH_TIME("VariableCoeffPoissonOperatorFactory::define");

    setDefaultValues();

    m_boxes = a_grids;

    m_refRatios = a_refRatios;

    m_bc = a_bc;

    CH_assert(a_aCoef[0]->nComp() == a_bCoef[0]->nComp());
    m_nComp = a_aCoef[0]->nComp();

    m_dx.resize(a_grids.size());
    m_dx[0] = a_coarsedx;

    m_domains.resize(a_grids.size());
    m_domains[0] = a_coarseDomain;

    int num_ghosts = 1;
    IntVect ghosts = num_ghosts * IntVect::Unit;
    m_exchangeCopiers.resize(a_grids.size());
    m_exchangeCopiers[0].exchangeDefine(a_grids[0], ghosts);
    m_exchangeCopiers[0].trimEdges(a_grids[0], ghosts);

    m_cfregion.resize(a_grids.size());
    m_cfregion[0].define(a_grids[0], m_domains[0]);

    for (int i = 1; i < a_grids.size(); i++)
    {
        m_dx[i] = m_dx[i - 1] / m_refRatios[i - 1];

        m_domains[i] = m_domains[i - 1];
        m_domains[i].refine(m_refRatios[i - 1]);

        m_exchangeCopiers[i].exchangeDefine(a_grids[i], ghosts);
        m_exchangeCopiers[i].trimEdges(a_grids[i], ghosts);

        m_cfregion[i].define(a_grids[i], m_domains[i]);
    }

    m_alpha = a_alpha;
    m_aCoef = a_aCoef;

    m_beta = a_beta;
    m_bCoef = a_bCoef;
}
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// AMR Factory define function, with coefficient data allocated automagically
// for operators.
void VariableCoeffPoissonOperatorFactory::define(
    const ProblemDomain &a_coarseDomain,
    const Vector<DisjointBoxLayout> &a_grids, const Vector<int> &a_refRatios,
    const Real &a_coarsedx, BCHolder a_bc, const IntVect &a_ghostVect)
{
    // This just allocates coefficient data, sets alpha = beta = 1, and calls
    // the other define() method.
    Vector<RefCountedPtr<LevelData<FArrayBox>>> aCoef(a_grids.size());
    Vector<RefCountedPtr<LevelData<FArrayBox>>> bCoef(a_grids.size());

    for (int i = 0; i < a_grids.size(); ++i)
    {
        aCoef[i] = RefCountedPtr<LevelData<FArrayBox>>(
            new LevelData<FArrayBox>(a_grids[i], 1, a_ghostVect));
        bCoef[i] = RefCountedPtr<LevelData<FArrayBox>>(
            new LevelData<FArrayBox>(a_grids[i], 1, a_ghostVect));

        // Initialize the a and b coefficients to 1 for starters.
        for (DataIterator dit = aCoef[i]->dataIterator(); dit.ok(); ++dit)
        {
            (*aCoef[i])[dit()].setVal(1.0);
            (*bCoef[i])[dit()].setVal(1.0);
        }
    }
    Real alpha = 1.0, beta = 1.0;
    define(a_coarseDomain, a_grids, a_refRatios, a_coarsedx, a_bc, alpha, aCoef,
           beta, bCoef);
}
//-----------------------------------------------------------------------

MGLevelOp<LevelData<FArrayBox>> *
VariableCoeffPoissonOperatorFactory::MGnewOp(const ProblemDomain &a_indexSpace,
                                             int a_depth, bool a_homoOnly)
{

    CH_TIME("VariableCoeffPoissonOperatorFactory::MGnewOp");

    Real dxCrse = -1.0;

    int ref;
    for (ref = 0; ref < m_domains.size(); ref++)
    {
        if (a_indexSpace.domainBox() == m_domains[ref].domainBox())
        {
            break;
        }
    }
    CH_assert(ref != m_domains.size()); // didn't find domain

    if (ref > 0)
    {
        dxCrse = m_dx[ref - 1];
    }

    ProblemDomain domain(m_domains[ref]);
    Real dx = m_dx[ref];
    int coarsening = 1;

    for (int i = 0; i < a_depth; i++)
    {
        coarsening *= 2;
        domain.coarsen(2);
    }

    if (coarsening > 1 &&
        !m_boxes[ref].coarsenable(coarsening *
                                  VariableCoeffPoissonOperator::s_maxCoarse))
    {
        return NULL;
    }

    dx *= coarsening;

    DisjointBoxLayout layout;
    coarsen_dbl(layout, m_boxes[ref], coarsening);

    Copier ex = m_exchangeCopiers[ref];
    CFRegion cfregion = m_cfregion[ref];

    if (coarsening > 1)
    {
        ex.coarsen(coarsening);
        cfregion.coarsen(coarsening);
    }

    VariableCoeffPoissonOperator *newOp = new VariableCoeffPoissonOperator;

    newOp->define(layout, dx, domain, m_bc, ex, cfregion);

    newOp->m_alpha = m_alpha;
    newOp->m_beta = m_beta;

    if (a_depth == 0)
    {
        // don't need to coarsen anything for this
        newOp->m_aCoef = m_aCoef[ref];
        newOp->m_bCoef = m_bCoef[ref];
    }
    else
    {
        // need to coarsen coefficients
        RefCountedPtr<LevelData<FArrayBox>> aCoef(new LevelData<FArrayBox>);
        RefCountedPtr<LevelData<FArrayBox>> bCoef(new LevelData<FArrayBox>);
        aCoef->define(layout, m_aCoef[ref]->nComp(), m_aCoef[ref]->ghostVect());
        bCoef->define(layout, m_bCoef[ref]->nComp(), m_bCoef[ref]->ghostVect());

        // average coefficients to coarser level
        // for now, do this with a CoarseAverage --
        // may want to switch to harmonic averaging at some point
        CoarseAverage averager_a(m_aCoef[ref]->getBoxes(), layout,
                                 aCoef->nComp(), coarsening);

        CoarseAverage averager_b(m_bCoef[ref]->getBoxes(), layout,
                                 bCoef->nComp(), coarsening);

        if (m_coefficient_average_type == CoarseAverage::arithmetic)
        {
            averager_a.averageToCoarse(*aCoef, *(m_aCoef[ref]));
            averager_b.averageToCoarse(*bCoef, *(m_bCoef[ref]));
        }
        else if (m_coefficient_average_type == CoarseAverage::harmonic)
        {
            averager_a.averageToCoarseHarmonic(*aCoef, *(m_aCoef[ref]));
            averager_b.averageToCoarseHarmonic(*bCoef, *(m_bCoef[ref]));
        }
        else
        {
            MayDay::Abort("VariableCoeffPoissonOperatorFactory::MGNewOp -- bad "
                          "averagetype");
        }

        newOp->m_aCoef = aCoef;
        newOp->m_bCoef = bCoef;
    }

    newOp->computeLambda();

    newOp->m_dxCrse = dxCrse;

    return (MGLevelOp<LevelData<FArrayBox>> *)newOp;
}

AMRLevelOp<LevelData<FArrayBox>> *
VariableCoeffPoissonOperatorFactory::AMRnewOp(const ProblemDomain &a_indexSpace)
{
    CH_TIME("VariableCoeffPoissonOperatorFactory::AMRnewOp");

    VariableCoeffPoissonOperator *newOp = new VariableCoeffPoissonOperator;
    Real dxCrse = -1.0;

    int ref;

    for (ref = 0; ref < m_domains.size(); ref++)
    {
        if (a_indexSpace.domainBox() == m_domains[ref].domainBox())
        {
            break;
        }
    }

    if (ref == 0)
    {
        // coarsest AMR level
        if (m_domains.size() == 1)
        {
            // no finer level
            newOp->define(m_boxes[0], m_dx[0], a_indexSpace, m_bc,
                          m_exchangeCopiers[0], m_cfregion[0]);
        }
        else
        {
            // finer level exists but no coarser
            int dummyRat = 1; // argument so compiler can find right function
            int refToFiner = m_refRatios[0]; // actual refinement ratio
            newOp->define(m_boxes[0], m_boxes[1], m_dx[0], dummyRat, refToFiner,
                          a_indexSpace, m_bc, m_exchangeCopiers[0],
                          m_cfregion[0], m_nComp);
        }
    }
    else if (ref == m_domains.size() - 1)
    {
        dxCrse = m_dx[ref - 1];

        // finest AMR level
        newOp->define(m_boxes[ref], m_boxes[ref - 1], m_dx[ref],
                      m_refRatios[ref - 1], a_indexSpace, m_bc,
                      m_exchangeCopiers[ref], m_cfregion[ref], m_nComp);
    }
    else if (ref == m_domains.size())
    {
        MayDay::Abort("Did not find a domain to match AMRnewOp(const "
                      "ProblemDomain& a_indexSpace)");
    }
    else
    {
        dxCrse = m_dx[ref - 1];

        // intermediate AMR level, full define
        newOp->define(m_boxes[ref], m_boxes[ref + 1], m_boxes[ref - 1],
                      m_dx[ref], m_refRatios[ref - 1], m_refRatios[ref],
                      a_indexSpace, m_bc, m_exchangeCopiers[ref],
                      m_cfregion[ref], m_nComp);
    }

    newOp->m_alpha = m_alpha;
    newOp->m_beta = m_beta;

    newOp->m_aCoef = m_aCoef[ref];
    newOp->m_bCoef = m_bCoef[ref];

    newOp->computeLambda();

    newOp->m_dxCrse = dxCrse;

    return (AMRLevelOp<LevelData<FArrayBox>> *)newOp;
}

int VariableCoeffPoissonOperatorFactory::refToFiner(
    const ProblemDomain &a_domain) const
{
    int retval = -1;
    bool found = false;

    for (int ilev = 0; ilev < m_domains.size(); ilev++)
    {
        if (m_domains[ilev].domainBox() == a_domain.domainBox())
        {
            retval = m_refRatios[ilev];
            found = true;
        }
    }

    if (!found)
    {
        MayDay::Abort("Domain not found in AMR hierarchy");
    }

    return retval;
}

//-----------------------------------------------------------------------
void VariableCoeffPoissonOperatorFactory::setDefaultValues()
{
    // Default to Laplacian operator
    m_alpha = 0.0;
    m_beta = -1.0;
    m_coefficient_average_type = CoarseAverage::arithmetic;
}
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
void VariableCoeffPoissonOperator::finerOperatorChanged(
    const MGLevelOp<LevelData<FArrayBox>> &a_operator, int a_coarseningFactor)
{
    const VariableCoeffPoissonOperator &op =
        dynamic_cast<const VariableCoeffPoissonOperator &>(a_operator);

    // Perform multigrid coarsening on the operator data.
    LevelData<FArrayBox> &acoefCoar = *m_aCoef;
    const LevelData<FArrayBox> &acoefFine = *(op.m_aCoef);
    LevelData<FArrayBox> &bcoefCoar = *m_bCoef;
    const LevelData<FArrayBox> &bcoefFine = *(op.m_bCoef);
    if (a_coarseningFactor != 1)
    {
        CoarseAverage cellAverage_a(acoefFine.disjointBoxLayout(),
                                    acoefCoar.disjointBoxLayout(), 1,
                                    a_coarseningFactor);
        for (DataIterator dit = acoefCoar.disjointBoxLayout().dataIterator();
             dit.ok(); ++dit)
            acoefCoar[dit()].setVal(0.);
        cellAverage_a.averageToCoarse(acoefCoar, acoefFine);

        CoarseAverage cellAverage_b(bcoefFine.disjointBoxLayout(),
                                    bcoefCoar.disjointBoxLayout(), 1,
                                    a_coarseningFactor);
        for (DataIterator dit = bcoefCoar.disjointBoxLayout().dataIterator();
             dit.ok(); ++dit)
            bcoefCoar[dit()].setVal(0.);
        cellAverage_b.averageToCoarse(bcoefCoar, bcoefFine);
    }

    // Handle inter-box ghost cells.
    acoefCoar.exchange();
    bcoefCoar.exchange();

    // Mark the relaxation coefficient dirty.
    m_lambdaNeedsResetting = true;

    // Notify any observers of this change.
    notifyObserversOfChange();
}
//-----------------------------------------------------------------------

#include "NamespaceFooter.H"
