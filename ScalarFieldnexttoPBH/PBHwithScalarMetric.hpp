/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef PBHWITHSCALARMETRIC_HPP_
#define PBHWITHSCALARMETRIC_HPP_

#include "ADMConformalVars.hpp"
#include "Cell.hpp"
#include "CoordinateTransformations.hpp"
#include "Coordinates.hpp"
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include "UserVariables.hpp" //This files needs NUM_VARS - total number of components
#include "VarsTools.hpp"
#include "simd.hpp"


// #include <iostream>
// #include <string>
// #include <vector>
// #include <fstream>
// #include <sstream>

const int intp_list_len = 6001;

//! Class which computes the initial conditions for the metric components
class PBHwithScalarMetric
{
    // Use the variable definition in CCZ4
    template <class data_t>
    using Vars = ADMConformalVars::VarsWithGauge<data_t>;

  public:
    //! Stuct for the params of the system
    struct params_t
    {
        double mass;                            //!< The mass of the BH
        double vp;                              //!< The vacuum energy outside the BH 

        std::array<double, intp_list_len> initial_chi_dot; //!< initial data for chi_dot
        std::array<double, intp_list_len> initial_h_dot;   //!< initial data for h_dot

        //!< The center of the vacuum bubble is initialized as the center of the grids
        std::array<double, CH_SPACEDIM> Bubble_center;
        std::array<double, CH_SPACEDIM> BH_center; //!< The center of the BH
        // By now the code can only deal with Schwarzschild BH
        // with no spin, the spin parameters are left as future work
        /*
            double spin; //!< The spin param a = J/M, so 0 <= |a| <= M
            std::array<double, CH_SPACEDIM> spin_direction = {
                0., 0., 1.}; // default to 'z' axis; doesn't need to be normalized
        */
    };

  protected:
    double m_dx;
    params_t m_params;
    
  public:
    //! The constructor
    PBHwithScalarMetric(params_t a_params, double a_dx)
        : m_dx(a_dx), m_params(a_params)
    {
        // check this spin param is sensible
        if (m_params.vp * m_params.mass > 0.19)
        {
            MayDay::Error("the constraint M H < ( 1 / sqrt(3) )^3 should be "
                          "satisfied, otherwise the Black Hole horizon would "
                          "be greater than the cosmological horizon");
        }

        // read chi_dot and h_dot initial data
        char chidot_file_name[] = "chidotlist.csv";
        char hdot_file_name[] = "hdotlist.csv";
        // char r_list_name[] = "rlist.csv";
        m_params.initial_chi_dot = read_csv(chidot_file_name);
        m_params.initial_h_dot = read_csv(hdot_file_name);
        // m_params.r_list = read_csv(r_list_name);
    }

    template <class data_t> void compute(Cell<data_t> current_cell) const;

  protected:
    //! Function which computes the components of the BH metric in spherical coords
    template <class data_t>
    void compute_BH(
        Tensor<2, data_t>
            &spherical_g, //!<< The spatial metric in spherical coords
        Tensor<2, data_t>
            &spherical_K, //!<< The extrinsic curvature in spherical coords
        Tensor<1, data_t>
            &spherical_shift, //!<< The spherical components of the shift
        data_t &BH_lapse,   //!<< The lapse for the Black Hole solution
        const Tensor<1, data_t> &coords //!<< Coords of current cell
    ) const;

    //! Function which computes the components of the metirc
    //  with a vacuum scalar bubble in spherical coords
    template <class data_t>
    void compute_Scalar(
        Tensor<2, data_t>
            &spherical_g, //!<< The spatial metric in spherical coords
        Tensor<2, data_t>
            &spherical_K, //!<< The extrinsic curvature in spherical coords
        Tensor<1, data_t>
            &spherical_shift, //!<< The spherical components of the shift
        data_t &BH_lapse,   //!<< The lapse for the Black Hole solution
        const Tensor<1, data_t> &coords //!<< Coords of current cell
    ) const;


    // Functions to read the initial data written in csv files
    std::array<double, intp_list_len>  read_csv(char file_name[]) const;

    // Intepolations of the initial data
    template <typename T>
    T interpolation(T &x, std::array<double, intp_list_len> fun) const;
    double interpolation(double &x, std::array<double, intp_list_len> fun) const;
    simd<double> interpolation(simd<double> &x, std::array<double, intp_list_len> fun) const;

    template <class data_t>
    data_t initial_chi_dot_scalar(data_t &r) const;

    template <class data_t>
    data_t initial_h_dot_scalar(data_t &r) const;
};

#include "PBHwithScalarMetric.impl.hpp"

#endif /* PBHWITHSCALARMETRIC_HPP_ */
