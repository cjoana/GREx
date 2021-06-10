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
        double scalar_mass;
        double sf_coupling;
    };

    double MATH_PI = 3.14159265359; // CJ
    double Mp= 1.0 /sqrt(8.0*MATH_PI);  // CJ

  private:
    params_t m_params;

  public:
    //! The constructor
    Potential(params_t a_params) : m_params(a_params) {}

    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_potential(data_t &V_of_phi, data_t &dVdphi,
                           data_t &V_of_phi2, data_t &dVdphi2,
                           const vars_t<data_t> &vars) const
    {
      // // The potential value at phi
      // // 1/2 m^2 phi^2
      // V_of_phi = 0.5 * pow(m_params.scalar_mass * vars.phi, 2.0);
      //
      // // The potential gradient at phi
      // // m^2 phi
      // dVdphi = pow(m_params.scalar_mass, 2.0) * vars.phi;


      // Non-minimal Starobinski Inflation  // CJ
      //

     double g = m_params.sf_coupling;

     // std::cout << " sf_coupling  " <<  g << " " << '\n';

     double SI_mass = pow( 3.1e-3*Mp , 4);  // 1.462066e-13

     V_of_phi =  SI_mass *
                    pow(1.0 - exp(-sqrt(2.0/3.0) * vars.phi / Mp), 2.0) +
	               g * vars.phi*vars.phi * vars.phi2*vars.phi2; // CJ

     dVdphi =   2.0 * SI_mass *
                      sqrt(2.0/3.0) * exp(- sqrt(2.0/3.0) * vars.phi / Mp ) *
                      (1.0 - exp(- sqrt(2.0/3.0) * vars.phi  /Mp)) / Mp +
		            2 * g * vars.phi * vars.phi2*vars.phi2; // CJ

     V_of_phi2 = 0.0;

     dVdphi2 =   2 * g * vars.phi*vars.phi * vars.phi2; // CJ



    //V_of_phi2 = 0.0;
    //dVdphi2 = 0.0;

    }
};

#endif /* POTENTIAL_HPP_ */
