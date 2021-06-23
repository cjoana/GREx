#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _VARIABLECOEFFPOISSONOPERATOR_H_
#define _VARIABLECOEFFPOISSONOPERATOR_H_

#include "AMRPoissonOp.H"
#include "CoefficientInterpolator.H"

#include "NamespaceHeader.H"

///
/**
   Operator for solving variable-coefficient
   (alpha * aCoef(x) * I - beta * cCoef(x) * laplacian ) dpsi = rhs
   over an AMR hierarchy.
*/
class VariableCoeffPoissonOperator : public AMRPoissonOp
{
  public:
    /// default constructor
    VariableCoeffPoissonOperator() { m_lambdaNeedsResetting = true; }

    /// destructor
    virtual ~VariableCoeffPoissonOperator() {}

    /// calculate residual
    virtual void residualI(LevelData<FArrayBox> &a_lhs,
                           const LevelData<FArrayBox> &a_dpsi,
                           const LevelData<FArrayBox> &a_rhs,
                           bool a_homogeneous = false);

    ///
    virtual void preCond(LevelData<FArrayBox> &a_correction,
                         const LevelData<FArrayBox> &a_residual);

    ///
    virtual void applyOpI(LevelData<FArrayBox> &a_lhs,
                          const LevelData<FArrayBox> &a_dpsi,
                          bool a_homogeneous = false);

    virtual void applyOpNoBoundary(LevelData<FArrayBox> &a_lhs,
                                   const LevelData<FArrayBox> &a_dpsi);

    /**
       calculate restricted residual
       a_resCoarse[2h] = I[h->2h] (rhsFine[h] - L[h](dpsiFine[h])
    */
    virtual void restrictResidual(LevelData<FArrayBox> &a_resCoarse,
                                  LevelData<FArrayBox> &a_dpsiFine,
                                  const LevelData<FArrayBox> &a_rhsFine);

    /// For tga stuff
    virtual void setAlphaAndBeta(const Real &a_alpha, const Real &a_beta);

    /// Also calls reset lambda
    virtual void setCoefs(const RefCountedPtr<LevelData<FArrayBox>> &a_aCoef,
                          const RefCountedPtr<LevelData<FArrayBox>> &a_bCoef,
                          const Real &a_alpha, const Real &a_beta);

    /// Should be called before the relaxation parameter is needed.
    virtual void resetLambda();

    /// Compute lambda once alpha, aCoef, beta, bCoef are defined
    virtual void computeLambda();

    /// refluxing - not implemented
    virtual void reflux(const LevelData<FArrayBox> &a_dpsiFine,
                        const LevelData<FArrayBox> &a_dpsi,
                        LevelData<FArrayBox> &a_residual,
                        AMRLevelOp<LevelData<FArrayBox>> *a_finerOp);

    //! This is called on multigrid operators when their AMR operators
    //! are altered.
    void finerOperatorChanged(const MGLevelOp<LevelData<FArrayBox>> &a_operator,
                              int a_coarseningFactor);

    //! Returns identity coefficient data.
    LevelData<FArrayBox> &identityCoef() { return *m_aCoef; }

    //! Sets up a model that modifies b coefficient data when the operator's
    //! time is set.
    //! \param a_bCoefInterpolator A CoefficientInterpolator that will be used
    //!                            to compute the b coefficient at specific
    //!                            times.
    void setBCoefInterpolator(
        RefCountedPtr<
            CoefficientInterpolator<LevelData<FArrayBox>, LevelData<FArrayBox>>>
            &a_bCoefInterpolator)
    {
        m_bCoefInterpolator = a_bCoefInterpolator;
    }

    //! Returns the B coefficient.
    LevelData<FArrayBox> &BCoef() { return *m_bCoef; }

    // Allows access to the B coefficient interpolator.
    RefCountedPtr<
        CoefficientInterpolator<LevelData<FArrayBox>, LevelData<FArrayBox>>>
    BCoefInterpolator()
    {
        return m_bCoefInterpolator;
    }

    // get flux - not implemented
    void getFlux(FArrayBox &a_flux, const FArrayBox &a_data,
                 const FArrayBox &b_data, const Box &a_facebox, int a_dir,
                 int a_ref) const;

    //! Sets the time centering of the operator. This interpolates b coefficient
    //! data at the given time if an interpolator is set.
    void setTime(Real a_time);

    /// Identity operator spatially varying coefficient storage (cell-centered)
    /// --- if you change this call resetLambda()
    RefCountedPtr<LevelData<FArrayBox>> m_aCoef;

    /// Laplacian operator spatially varying coefficient storage (face-centered)
    /// --- if you change this call resetLambda()
    RefCountedPtr<LevelData<FArrayBox>> m_bCoef;

    /// Reciprocal of the diagonal entry of the operator matrix
    LevelData<FArrayBox> m_lambda;

  protected:
    LayoutData<CFIVS> m_loCFIVS[SpaceDim];
    LayoutData<CFIVS> m_hiCFIVS[SpaceDim];

    // Interpolator for b coefficient data.
    RefCountedPtr<
        CoefficientInterpolator<LevelData<FArrayBox>, LevelData<FArrayBox>>>
        m_bCoefInterpolator;

    // Current time.
    Real m_time;

    // Does the relaxation coefficient need to be reset?
    bool m_lambdaNeedsResetting;

    virtual void levelGSRB(LevelData<FArrayBox> &a_dpsi,
                           const LevelData<FArrayBox> &a_rhs);

    virtual void levelMultiColor(LevelData<FArrayBox> &a_dpsi,
                                 const LevelData<FArrayBox> &a_rhs);

    virtual void looseGSRB(LevelData<FArrayBox> &a_dpsi,
                           const LevelData<FArrayBox> &a_rhs);

    virtual void overlapGSRB(LevelData<FArrayBox> &a_dpsi,
                             const LevelData<FArrayBox> &a_rhs);

    virtual void levelGSRBLazy(LevelData<FArrayBox> &a_dpsi,
                               const LevelData<FArrayBox> &a_rhs);

    virtual void levelJacobi(LevelData<FArrayBox> &a_dpsi,
                             const LevelData<FArrayBox> &a_rhs);
};

#include "NamespaceFooter.H"
#endif
