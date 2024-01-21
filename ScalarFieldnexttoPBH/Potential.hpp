/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef POTENTIAL_HPP_
#define POTENTIAL_HPP_

#include "simd.hpp"

class Potential
{
  public:
    struct params_t
    {
        double false_vacuum_potential;
        double true_vacuum_potential;
        double barrier_amplitude_parameter;
        double phi_VEV;
    };

  private:
    params_t m_params;

  public:
    //! The constructor
    Potential(params_t a_params) : m_params(a_params) {}

    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_potential(data_t &V_of_phi, data_t &dVdphi,
                           const vars_t<data_t> &vars) const
    {
        double a = m_params.barrier_amplitude_parameter;
        // The potential value at phi
        // VF (1 +  a (phi/phi0)^2 - (2a+4) (phi/phi0)^3 + (a+3) (phi/phi0)^4) + VT
        V_of_phi =
            m_params.false_vacuum_potential *
                (1.0 +
                 m_params.barrier_amplitude_parameter *
                     pow((vars.phi / m_params.phi_VEV), 2.0) -
                 (2.0 * a + 4.0) * pow((vars.phi / m_params.phi_VEV), 3.0) +
                 (a + 3.0) * pow((vars.phi / m_params.phi_VEV), 4.0)) +
            m_params.true_vacuum_potential;

        // The potential gradient at phi
        // VF/phi0 (2a (phi/phi0) - 3(2a+4) (phi/phi0)^2 + 4(a+3) (phi/phi0)^3)
        dVdphi =
            m_params.false_vacuum_potential / m_params.phi_VEV *
            (2.0 * m_params.barrier_amplitude_parameter *
                 (vars.phi / m_params.phi_VEV) -
             3.0 * (2.0 * a + 4.0) * pow((vars.phi / m_params.phi_VEV), 2.0) +
             4.0 * (a + 3.0) * pow((vars.phi / m_params.phi_VEV), 3.0));
    }
};

#endif /* POTENTIAL_HPP_ */
