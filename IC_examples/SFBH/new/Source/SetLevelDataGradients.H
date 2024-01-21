#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _SETLEVELDATAGRADIENTS_H_
#define _SETLEVELDATAGRADIENTS_H_

#include <iostream>
using std::cerr;

#include "DimensionDefinitions.hpp"
#include "DisjointBoxLayout.H"
#include "FArrayBox.H"
#include "MultigridUserVariables.hpp"
#include "Tensor.hpp"

// Functions for taking gradients at points etc

// get location:
// This takes an IntVect and writes the physical coordinates to a RealVect
inline void get_loc(RealVect &a_out_loc, const IntVect &a_iv,
                    const RealVect &a_dx, const PoissonParameters &a_params)
{
    a_out_loc = a_iv + 0.5 * RealVect::Unit;
    a_out_loc *= a_dx;
    FOR1(i) { a_out_loc[i] -= a_params.center[i]; }
}

// computes the Laplacian of component a_comp at a point a_iv in a box of data
inline Real get_laplacian(const IntVect &a_iv, const FArrayBox &a_vars_box,
                          const RealVect &a_dx, const int a_comp)
{
    Real laplacian = 0.0;
    FOR1(idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        // 2nd order stencil for now
        Real d2comp_dxdx = 1.0 / (a_dx[idir] * a_dx[idir]) *
                           (1.0 * a_vars_box(iv_offset2, a_comp) -
                            2.0 * a_vars_box(a_iv, a_comp) +
                            1.0 * a_vars_box(iv_offset1, a_comp));
        laplacian += d2comp_dxdx;
    }
    return laplacian;
} // end get_laplacian

inline Tensor<1, Real, SpaceDim> get_d1(const IntVect &a_iv,
                                        const FArrayBox &multigrid_vars_box,
                                        const RealVect &a_dx, const int icomp)
{
    Tensor<1, Real, SpaceDim> d1;
    FOR1(idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        d1[idir] = 0.5 *
                   (multigrid_vars_box(iv_offset2, icomp) -
                    multigrid_vars_box(iv_offset1, icomp)) /
                   a_dx[idir];
    }
    return d1;
} // end get_d1

inline Tensor<2, Real, SpaceDim> get_d2(const IntVect &a_iv,
                                        const FArrayBox &multigrid_vars_box,
                                        const RealVect &a_dx, const int icomp)
{
    Tensor<2, Real, SpaceDim> d2;
    FOR2(idir1, idir2)
    {
        if (idir1 != idir2)
        {
            IntVect iv_offset1 = a_iv;
            IntVect iv_offset2 = a_iv;
            IntVect iv_offset3 = a_iv;
            IntVect iv_offset4 = a_iv;
            iv_offset1[idir1] -= 1;
            iv_offset1[idir2] -= 1;
            iv_offset2[idir1] += 1;
            iv_offset2[idir2] += 1;
            iv_offset3[idir1] += 1;
            iv_offset3[idir2] -= 1;
            iv_offset4[idir1] -= 1;
            iv_offset4[idir2] += 1;

            d2[idir1][idir2] = (multigrid_vars_box(iv_offset1, icomp) +
                                multigrid_vars_box(iv_offset2, icomp) -
                                multigrid_vars_box(iv_offset3, icomp) -
                                multigrid_vars_box(iv_offset4, icomp)) /
                               (4.0 * a_dx[idir1] * a_dx[idir2]);
        }
        else
        {
            IntVect iv_offset1 = a_iv;
            IntVect iv_offset2 = a_iv;
            iv_offset1[idir1] -= 1;
            iv_offset2[idir1] += 1;

            d2[idir1][idir1] = (multigrid_vars_box(iv_offset1, icomp) -
                                2.0 * multigrid_vars_box(a_iv, icomp) +
                                multigrid_vars_box(iv_offset2, icomp)) /
                               (a_dx[idir1] * a_dx[idir1]);
        }
    }
    return d2;

} // end get_d2

#endif
