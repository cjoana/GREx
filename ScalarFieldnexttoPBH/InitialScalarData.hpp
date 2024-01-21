/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef INITIALSCALARDATA_HPP_
#define INITIALSCALARDATA_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "MatterCCZ4RHS.hpp"
#include "ScalarField.hpp"
#include "Tensor.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total no. components
#include "VarsTools.hpp"
#include "simd.hpp"

//! Class which sets the initial scalar field matter config
class InitialScalarData
{
  public:
    //! A structure for the input params for scalar field properties and initial
    //! conditions
    struct params_t
    {
        std::array<double, CH_SPACEDIM>
            center; //!< Centre of perturbation in initial bubble

        double amplitude;
        //!< Amplitude of bump in initial bubble, equal to the phi_VEV        
        double width;  //!< Width of bump in initial bubble
        double radius; //!< Radius of the initial bubble
        double velocity;  //!< Initial velocity of the bubble, 
        // which is determined by the vacuum energy difference
        // v = (VF - VT) / (4pi)
    };

    //! The constructor
    InitialScalarData(params_t a_params, double a_dx)
        : m_dx(a_dx), m_params(a_params)
    {
    }

    //! Function to compute the value of all the initial vars on the grid
    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // where am i?
        Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        data_t rr = coords.get_radius();

        // calculate the field value
        data_t phi = m_params.amplitude / 2.0 *
                    (1.0 - tanh((rr - m_params.radius) / m_params.width));

        // data_t Pi = m_params.amplitude / 2.0 * m_params.velocity /
        //             m_params.width *
        //             pow(tanh((rr - m_params.radius) / m_params.width) / sinh((rr - m_params.radius) / m_params.width), 2.0);
        // store the vars
        current_cell.store_vars(phi, c_phi);
        current_cell.store_vars(0.0, c_Pi);
    }

  protected:
    double m_dx;
    const params_t m_params; //!< The matter initial condition params
};

#endif /* INITIALSCALARDATA_HPP_ */
