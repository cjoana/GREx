/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef _SETLEVELDATA_H_
#define _SETLEVELDATA_H_

#include <iostream>
using std::cerr;

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "DimensionDefinitions.hpp"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "GRChomboUserVariables.hpp"
#include "MultigridUserVariables.hpp"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "SetLevelDataGradients.H"
#include "Tensor.hpp"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

// various functions operating on grids to set up data for initial
// conditions, spatially dependent coefficients and rhs etc

extern void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars,
                                   LevelData<FArrayBox> &a_dpsi,
                                   const RealVect &a_dx,
                                   const PoissonParameters &a_params,
                                   const bool set_matter = true);

extern void set_update_Kij(LevelData<FArrayBox> &a_multigrid_vars,
                           LevelData<FArrayBox> &a_rhs, const RealVect &a_dx,
                           const PoissonParameters &a_params);

extern void set_rhs(LevelData<FArrayBox> &a_rhs,
                    LevelData<FArrayBox> &a_multigrid_vars,
                    const RealVect &a_dx, const PoissonParameters &a_params);

extern void set_error(LevelData<FArrayBox> &a_error,
                      LevelData<FArrayBox> &a_rhs,
                      LevelData<FArrayBox> &a_multigrid_vars,
                      const RealVect &a_dx, const PoissonParameters &a_params);

extern void set_regrid_condition(LevelData<FArrayBox> &a_condition,
                                 LevelData<FArrayBox> &a_multigrid_vars,
                                 const RealVect &a_dx,
                                 const PoissonParameters &a_params);

extern void set_a_coef(LevelData<FArrayBox> &a_aCoef,
                       LevelData<FArrayBox> &a_multigrid_vars,
                       const PoissonParameters &a_params, const RealVect &a_dx);

extern void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                       const PoissonParameters &a_params, const RealVect &a_dx);

extern void set_update_psi0(LevelData<FArrayBox> &a_multigrid_vars,
                            LevelData<FArrayBox> &a_dpsi,
                            const Copier &a_exchange_copier,
                            const PoissonParameters &a_params);

extern void set_output_data(LevelData<FArrayBox> &a_vars,
                            LevelData<FArrayBox> &a_multigrid_vars,
                            const PoissonParameters &a_params,
                            const RealVect &a_dx);

#endif
