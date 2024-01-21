/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(PBHWITHSCALARMETRIC_HPP_)
#error "This file should only be included through PBHwithScalarMetric.hpp"
#endif

#ifndef PBHWITHSCALARMETRIC_IMPL_HPP_
#define PBHWITHSCALARMETRIC_IMPL_HPP_

#include "DimensionDefinitions.hpp"

// Computes isotropic Schwarzschild solution together with a scalar bubble near by
template <class data_t>
void PBHwithScalarMetric::compute(Cell<data_t> current_cell) const
{
    using namespace CoordinateTransformations;
    using namespace TensorAlgebra;

    // set up vars for the metric and extrinsic curvature, shift and lapse in
    // spherical coords with the origion at BH center
    Tensor<2, data_t> spherical_g;
    Tensor<2, data_t> spherical_K;
    Tensor<1, data_t> spherical_shift;
    data_t lapse;

    // The cartesian variables and coords
    // work out where we are on the grid
    Vars<data_t> vars;
    Coordinates<data_t> coords(current_cell, m_dx, m_params.Bubble_center);
    data_t x = coords.x;
    data_t y = coords.y;
    data_t z = coords.z;
    Tensor<1, data_t> xyz = {x, y, z};
    
    // Compute the components in spherical coords
    // This gives the initial data for the metric and extrinsic curvature
    // with a spherical scalar bubble
    compute_Scalar(spherical_g, spherical_K, spherical_shift, lapse, xyz);



    // For  x >= 160.  the following function should be used
    // to overwrite the initial data, which describes a Schwarzschild BH
    // centered at  xyz = {196, 0, 0}. So x=160 is the junction surface
    // of the black hole and the spacetime with a vacuum bubble, and is
    // almost flat such that the constraints are 'almost' not violated

    // compute_BH(spherical_g, spherical_K, spherical_shift, lapse, xyz);



    // Convert spherical components to cartesian components using coordinate
    // transform_tensor_UU
    Tensor<2, data_t> cartesian_h =
        spherical_to_cartesian_LL(spherical_g, x, y, z);
    Tensor<2, data_t> cartesian_K =
        spherical_to_cartesian_LL(spherical_K, x, y, z);
    Tensor<1, data_t> cartesian_shift =
        spherical_to_cartesian_U(spherical_shift, x, y, z);

    vars.h = cartesian_h;
    vars.A = cartesian_K;
    vars.shift = cartesian_shift;

    // Convert to BSSN vars
    data_t deth = compute_determinant(vars.h);
    auto h_UU = compute_inverse_sym(vars.h);
    vars.chi = pow(deth, -1. / 3.);

    // transform extrinsic curvature into A and TrK - note h is still non
    // conformal version which is what we need here
    vars.K = compute_trace(vars.A, h_UU);
    make_trace_free(vars.A, vars.h, h_UU);

    // Make conformal
    FOR(i, j)
    {
        vars.h[i][j] *= vars.chi;
        vars.A[i][j] *= vars.chi;
    }

    vars.lapse = lapse;
    // vars.lapse = 1.0; // WIll it be possible to just set lapse=1?


    // Populate the variables on the grid
    // NB We stil need to set Gamma^i which is NON ZERO
    // but we do this via a separate class/compute function
    // as we need the gradients of the metric which are not yet available
    current_cell.store_vars(vars);
}


template <class data_t>
void PBHwithScalarMetric::compute_BH(Tensor<2, data_t> &spherical_g,
                          Tensor<2, data_t> &spherical_K,
                          Tensor<1, data_t> &spherical_shift,
                          data_t &BH_lapse,
                          const Tensor<1, data_t> &coords) const
{
    // Black hole params - mass M and vacuum energy Vp
    double M = m_params.mass;
    double Vp = m_params.vp;

    // work out where we are on the grid
    data_t x = coords[0];
    data_t y = coords[1];
    data_t z = coords[2];

    // the radius, subject to a floor
    data_t r = sqrt(D_TERM(x * x, +y * y, +z * z));
    r = simd_max(r, 1e-6);
    data_t r2 = r * r;

    // the radius in xy plane, subject to a floor
    data_t rho2 = simd_max(x * x + y * y, 1e-12);
    data_t rho = sqrt(rho2);

    // calculate useful position quantities
    data_t cos_theta = z / r;
    data_t sin_theta = rho / r;
    data_t sin_theta2 = sin_theta * sin_theta;

    // set the metric components
    FOR(i, j) { spherical_g[i][j] = 0.0; }
    spherical_g[0][0] = pow(1.0 + M / 2.0 / r, 4.0);    // gamma_rr
    spherical_g[1][1] = spherical_g[0][0] * r2;         // gamma_tt
    spherical_g[2][2] = spherical_g[1][1] * sin_theta2; // gamma_pp

    // set the extrinsic curvature components
    FOR(i, j) { spherical_K[i][j] = 0.0; }
    spherical_K[0][0] = - sqrt( 8.0 * acos(-1.0) * Vp / 3.0) * spherical_g[0][0];   // K_rr
    spherical_K[1][1] = - sqrt( 8.0 * acos(-1.0) * Vp / 3.0) * spherical_g[1][1];   // K_tt
    spherical_K[2][2] = - sqrt( 8.0 * acos(-1.0) * Vp / 3.0) * spherical_g[2][2];   // K_pp

    // set the analytic lapse
    BH_lapse = (1 - M / 2.0 / r) / (1 + M / 2.0 / r);

    // set the shift (all zero)
    spherical_shift[0] = 0.0;
    spherical_shift[1] = 0.0;
    spherical_shift[2] = 0.0;
}

template <class data_t>
void PBHwithScalarMetric::compute_Scalar(Tensor<2, data_t> &spherical_g,
                          Tensor<2, data_t> &spherical_K,
                          Tensor<1, data_t> &spherical_shift,
                          data_t &lapse,
                          const Tensor<1, data_t> &coords) const
{
    // Black hole params - mass M and vacuum energy Vp
    double M = m_params.mass;
    double Vp = m_params.vp;

    // work out where we are on the grid
    data_t x = coords[0];
    data_t y = coords[1];
    data_t z = coords[2];

    // the radius, subject to a floor
    data_t r = sqrt(D_TERM(x * x, +y * y, +z * z));
    r = simd_max(r, 1e-6);
    data_t r2 = r * r;

    // the radius in xy plane, subject to a floor
    data_t rho2 = simd_max(x * x + y * y, 1e-12);
    data_t rho = sqrt(rho2);

    // calculate useful position quantities
    data_t cos_theta = z / r;
    data_t sin_theta = rho / r;
    data_t sin_theta2 = sin_theta * sin_theta;

    data_t chi_dot = initial_chi_dot_scalar(r);
    data_t h_dot = initial_h_dot_scalar(r);

    // set the metric components
    FOR(i, j) { spherical_g[i][j] = 0.0; }
    spherical_g[0][0] = 1.0;    // gamma_rr
    spherical_g[1][1] = r2;         // gamma_tt
    spherical_g[2][2] = spherical_g[1][1] * sin_theta2; // gamma_pp

    // set the extrinsic curvature components
    FOR(i, j) { spherical_K[i][j] = 0.0; }
    spherical_K[0][0] = (chi_dot - h_dot) / 2.0;  // K_rr
    spherical_K[1][1] = (2.0 * chi_dot + h_dot) / 4.0 * r2;   // K_tt
    spherical_K[2][2] = spherical_K[1][1] * sin_theta2;   // K_pp


    // set the analytic lapse
    lapse = 1.0;

    // set the shift (all zero)
    spherical_shift[0] = 0.0;
    spherical_shift[1] = 0.0;
    spherical_shift[2] = 0.0;
}

inline std::array<double, intp_list_len>  PBHwithScalarMetric::read_csv(char file_name[]) const
    {
        using namespace std;

        array<double, intp_list_len> res;
        int index = 0;

        ifstream data(file_name, ios::in);
        string line;

        if(!data.is_open())
        {
            cout << "Error: opening file fail" << endl;
            exit(1);
        }

        while(getline(data, line)){
            istringstream sin(line); 
            string field;
            while (std::getline(sin, field, ',')) 
            {
                double str_double;
                istringstream istr(field);
                istr >> str_double;
                res[index] = str_double;
                index++;
            }
        }

        data.close();
        return res;
    }

template <typename T>
T PBHwithScalarMetric::interpolation(T &x, std::array<double, intp_list_len> fun) const
{
    std::cout << "The data type of r is " << typeid(x).name() << endl;
    std::cout << "no fitness for data type 'double' or 'simd<double>', will return with an error." << endl;
    exit(1);
}

inline double PBHwithScalarMetric::interpolation(double &x, std::array<double, intp_list_len> fun) const
{
    if(x > (double)((intp_list_len - 1) / 100)) return fun[intp_list_len - 1];
    else{
        int index = (int)( x / 0.01);
        return fun[index] + (fun[index + 1] - fun[index]) * (x / 0.01 - (double)(index));
    }
}

inline simd<double> PBHwithScalarMetric::interpolation(simd<double> &x, std::array<double, intp_list_len> fun) const
{
    const int N = sizeof(x) / sizeof(double);
    simd<double> r = x;
    double y[N]; 

    for(int i = 0; i < N; i++)
    {
        if(r[i] > (double)((intp_list_len - 1) / 100)) y[i] = fun[intp_list_len - 1];
        else{
            int j = (int)(r[i] / 0.01);
            y[i] = fun[j] + (fun[j + 1] - fun[j]) * (r[i] / 0.01 - j);
        }
    }

    simd<double> res = r - r;
    res = _mm256_load_pd(y);
    return res;
}

template <class data_t>
data_t PBHwithScalarMetric::initial_h_dot_scalar(data_t &r) const
{
    return interpolation(r , m_params.initial_h_dot);
}

template <class data_t>
data_t PBHwithScalarMetric::initial_chi_dot_scalar(data_t &r) const
{
    return interpolation(r , m_params.initial_chi_dot);
}




#endif /* PBHWITHSCALARMETRIC_IMPL_HPP_ */
