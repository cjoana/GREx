/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef INITIALMETRICDATA_HPP_
#define INITIALMETRICDATA_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "MatterCCZ4RHS.hpp"
#include "ScalarField.hpp"
#include "Tensor.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total no. components
#include "VarsTools.hpp"
#include "simd.hpp"

//! Class which sets the initial scalar field matter config
class InitialMetricData
{
  public:
    //! A structure for the input params for scalar field properties and initial
    //! conditions
    struct params_t
    {
        double amplitude; //!< Amplitude of bump in initial SF bubble
        std::array<double, CH_SPACEDIM>
            center;   //!< Centre of perturbation in initial SF bubble
        double width; //!< Width of bump in initial SF bubble

        //CJ adds
        double L; // size of grid (again)
    };

    //! The constructor
    InitialMetricData(params_t a_params, double a_dx)
        : m_dx(a_dx), m_params(a_params)
    {
    }

    //! Function to compute the value of all the initial vars on the grid
    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        // where am i?
        Coordinates<data_t> coords(current_cell, m_dx, m_params.center);
        data_t rr = coords.get_radius();
        
        data_t amplitude = 0.3;


        data_t h11 = 1 + amplitude * sin( coords.x * 2*M_PI/m_params.L   +  M_PI*0.3)
                       + amplitude * sin( coords.y * 2*M_PI/m_params.L   +  M_PI*0.3)
                       + amplitude * sin( coords.z * 2*M_PI/m_params.L   +  M_PI*0.3);
        data_t h22 = 1 + amplitude * sin( coords.x * 2*M_PI/m_params.L   +  M_PI*0.3) 
                       + amplitude * sin( coords.y * 2*M_PI/m_params.L   +  M_PI*0.3)
                       + amplitude * sin( coords.z * 2*M_PI/m_params.L   +  M_PI*0.3);
        data_t h33 = 3 - h22 + h33 ;  
        
        data_t h12 = 0 + amplitude * sin( coords.x * 2*M_PI/m_params.L   +  M_PI*0.3)
                       + amplitude * sin( coords.y * 2*M_PI/m_params.L   +  M_PI*0.3) 
                       + amplitude * sin( coords.z * 2*M_PI/m_params.L   +  M_PI*0.3);
        data_t h13 = 0 + 0.1 * sin( coords.x * 2*M_PI/m_params.L   +  M_PI*0.5) 
                       + 0.1 * sin( coords.y * 2*M_PI/m_params.L   +  M_PI*0.5) 
                       + 0.1 * sin( coords.z * 2*M_PI/m_params.L   +  M_PI*0.5); 
        data_t h23 = 0 + 0.3 * sin( coords.x * 2*M_PI/m_params.L   +  M_PI*0.9)
                       + 0.3 * sin( coords.y * 2*M_PI/m_params.L   +  M_PI*0.9) 
                       + 0.3 * sin( coords.z * 2*M_PI/m_params.L   +  M_PI*0.9);
     
        // determinant of h_ij
        data_t detH = h11*h22*h33 - (h11*h23*h23 + h22*h13*h13 + h33*h12*h12) + 2*h12*h13*h23;
        
        // store the vars
        current_cell.store_vars(h11/detH, c_h11);
        current_cell.store_vars(h11/detH, c_h12);
        current_cell.store_vars(h11/detH, c_h13);
        current_cell.store_vars(h11/detH, c_h22);
        current_cell.store_vars(h11/detH, c_h23);
        current_cell.store_vars(h11/detH, c_h33);
        current_cell.store_vars(1.0, c_chi);
        current_cell.store_vars(0.0, c_phi); // for now set to 0
        current_cell.store_vars(0.0, c_Pi);
    }

  protected:
    double m_dx;
    const params_t m_params; //!< The matter initial condition params
};

#endif /* INITIALMETRICDATA_HPP_ */
