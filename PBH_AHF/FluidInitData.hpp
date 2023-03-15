/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef FLUIDINITDATA_HPP_
#define FLUIDINITDATA_HPP_

#include "CCZ4Geometry.hpp"
#include "Cell.hpp"
#include "FourthOrderDerivatives.hpp"
#include "MatterCCZ4RHS.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "VarsTools.hpp"
#include "simd.hpp"

//!  Calculates  initial data assuming comoving coordinates (W = 1)
/*!
    ...
*/

template <class matter_t> class FluidInitData
{

    // Use the variable definitions in MatterCCZ4RHS
    template <class data_t>
    using Vars = typename MatterCCZ4RHS<matter_t>::template Vars<data_t>;

    template <class data_t>
    using Diff2Vars = typename MatterCCZ4RHS<matter_t>::template Diff2Vars<data_t>;

  public:
    //! Constructor of class FluidInitData
    /*!
        Takes in the grid spacing, plus the relaxation speed, a matter object
        and the value of Newton's constant, which is set to one by default.
    */
    FluidInitData(matter_t a_matter, double dx, double omega,
                  double G_Newton = 1.0);

    //! The compute member which calculates the RHS at each point in the box \sa
    //! rhs_equation()
    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    matter_t my_matter;         //!< The matter object, e.g. a scalar field.
    const double m_omega;       //!< The EoS of the fluid.
    const double m_G_Newton;    //!< Newton's constant, set to one by default.
    const FourthOrderDerivatives
        m_deriv; //!< An object for calculating derivatives of the variables

    //! The function which calculates the RHS, given the vars and derivatives
    //! \sa compute()
    template <class data_t>
    void rhs_equation(
        Vars<data_t> &up_vars, //!< the RHS data for each variable at that point.
        const Vars<data_t> &vars, //!< the value of the variables at the point.
        const Vars<Tensor<1, data_t>>
            &d1, //!< the value of the first derivatives of the variables.
        const Diff2Vars<Tensor<2, data_t>>
            &d2, //!< the value of the second derivatives of the variables.
        const Vars<data_t>
            &advec //!< advec the value of the advection terms beta^i d_i(var)
    ) const;
};

#include "FluidInitData.impl.hpp"

#endif /* FLUIDINITDATA_HPP_ */
