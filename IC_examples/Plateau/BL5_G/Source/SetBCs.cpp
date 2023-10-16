#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#include "SetBCs.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

// Global BCRS definitions
std::vector<bool> GlobalBCRS::s_printedThatLo =
    std::vector<bool>(SpaceDim, false);
std::vector<bool> GlobalBCRS::s_printedThatHi =
    std::vector<bool>(SpaceDim, false);
std::vector<int> GlobalBCRS::s_bcLo = std::vector<int>();
std::vector<int> GlobalBCRS::s_bcHi = std::vector<int>();
RealVect GlobalBCRS::s_trigvec = RealVect::Zero;
bool GlobalBCRS::s_areBCsParsed = false;
bool GlobalBCRS::s_valueParsed = false;
bool GlobalBCRS::s_trigParsed = false;

// BCValueHolder class, which is a pointer to a void-type function with the 4
// arguements given pos [x,y,z] position on center of cell edge int dir
// direction, x being 0 int side -1 for low, +1 = high, fill in the a_values
// array

void ParseValue(Real *pos, int *dir, Side::LoHiSide *side, Real *a_values)
{
    ParmParse pp;
    Real bcVal;
    pp.get("bc_value", bcVal);
    a_values[0] = bcVal;
}

void ParseBC(FArrayBox &a_state, const Box &a_valid,
             const ProblemDomain &a_domain, Real a_dx, bool a_homogeneous)
{
    if (!a_domain.domainBox().contains(a_state.box()))
    {

        if (!GlobalBCRS::s_areBCsParsed)
        {
            ParmParse pp;
            pp.getarr("bc_lo", GlobalBCRS::s_bcLo, 0, SpaceDim);
            pp.getarr("bc_hi", GlobalBCRS::s_bcHi, 0, SpaceDim);
            GlobalBCRS::s_areBCsParsed = true;
        }

        Box valid = a_valid;

        for (int i = 0; i < CH_SPACEDIM; ++i)
        {

            // periodic? If not, check if Dirichlet or Neumann
            if (!a_domain.isPeriodic(i))
            {
                Box ghostBoxLo = adjCellBox(valid, i, Side::Lo, 1);
                Box ghostBoxHi = adjCellBox(valid, i, Side::Hi, 1);
                if (!a_domain.domainBox().contains(ghostBoxLo))
                {
                    if (GlobalBCRS::s_bcLo[i] == 1)
                    {
                        if (!GlobalBCRS::s_printedThatLo[i])
                        {
                            GlobalBCRS::s_printedThatLo[i] = true;
                            pout() << "Constant Neumann bcs imposed for low "
                                      "side direction "
                                   << i << endl;
                        }
                        NeumBC(a_state, valid, a_dx, a_homogeneous,
                               ParseValue, // BCValueHolder class
                               i, Side::Lo);
                    }
                    else if (GlobalBCRS::s_bcLo[i] == 0)
                    {
                        if (!GlobalBCRS::s_printedThatLo[i])
                        {
                            GlobalBCRS::s_printedThatLo[i] = true;
                            pout() << "Constant Dirichlet bcs imposed for low "
                                      "side direction "
                                   << i << endl;
                        }
                        DiriBC(a_state, valid, a_dx, a_homogeneous, ParseValue,
                               i, Side::Lo);
                    }
                    else if (GlobalBCRS::s_bcLo[i] == 2)
                    {
                        if (!GlobalBCRS::s_printedThatLo[i])
                        {
                            GlobalBCRS::s_printedThatLo[i] = true;
                            pout() << "Periodic bcs imposed for low side "
                                      "direction "
                                   << i << endl;
                        }
                    }
                    else
                    {
                        MayDay::Error("bogus bc flag low side");
                    }
                }

                if (!a_domain.domainBox().contains(ghostBoxHi))
                {
                    if (GlobalBCRS::s_bcHi[i] == 1)
                    {
                        if (!GlobalBCRS::s_printedThatHi[i])
                        {
                            GlobalBCRS::s_printedThatHi[i] = true;
                            pout() << "Constant Neumann bcs imposed for high "
                                      "side direction "
                                   << i << endl;
                        }
                        NeumBC(a_state, valid, a_dx, a_homogeneous, ParseValue,
                               i, Side::Hi);
                    }
                    else if (GlobalBCRS::s_bcHi[i] == 0)
                    {
                        if (!GlobalBCRS::s_printedThatHi[i])
                        {
                            GlobalBCRS::s_printedThatHi[i] = true;
                            pout() << "Constant Dirichlet bcs imposed for high "
                                      "side direction "
                                   << i << endl;
                        }
                        DiriBC(a_state, valid, a_dx, a_homogeneous, ParseValue,
                               i, Side::Hi);
                    }
                    else if (GlobalBCRS::s_bcHi[i] == 2)
                    {
                        if (!GlobalBCRS::s_printedThatHi[i])
                        {
                            GlobalBCRS::s_printedThatHi[i] = true;
                            pout() << "Periodic bcs imposed for high side "
                                      "direction "
                                   << i << endl;
                        }
                    }
                    else
                    {
                        MayDay::Error("bogus bc flag high side");
                    }
                }

            } // else - is periodic

        } // close for idir
    }
}
