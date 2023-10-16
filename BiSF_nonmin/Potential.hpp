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
        double g_coupling;
        double sf1_a_coupling;
        double sf2_a_coupling;
        double sf1_b_coupling;
        double sf2_b_coupling;
    };

    double MATH_PI = 3.14159265359; // CJ
    // double Mp= 1.0 /sqrt(8.0*MATH_PI);  // CJ
    double Mp= 1.0;  // CJ

  private:
    params_t m_params;

  public:
    //! The constructor
    Potential(params_t a_params) : m_params(a_params) {}

    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_sfmetric( Tensor<2, data_t, 2> &G,
                           Tensor<2, data_t, 2> &G_UU,
                           const vars_t<data_t> &vars) const
    {
        double cp_phi = m_params.sf1_a_coupling;
        double cp_phi2 = m_params.sf2_a_coupling;
        data_t f_coeff = 0.5 *(Mp*Mp + cp_phi * vars.phi*vars.phi
                                     + cp_phi2 * vars.phi2*vars.phi2 );
        data_t C_coeff = 2*f_coeff + 6*cp_phi*cp_phi*vars.phi*vars.phi
                                   + 6*cp_phi2*cp_phi2*vars.phi2*vars.phi2;
        data_t G_coeff = Mp*Mp / (4*f_coeff*f_coeff);
        data_t Ginv_coeff = 2*f_coeff/(Mp*Mp *C_coeff);

        G[0][0] = G_coeff *(2*f_coeff + 6* cp_phi*cp_phi*vars.phi*vars.phi);
        G[0][1] = G_coeff *(6* cp_phi*cp_phi2*vars.phi*vars.phi2);
        G[1][0] = G[0][1];
        G[1][1] = G_coeff *(2*f_coeff + 6* cp_phi2*cp_phi2*vars.phi2*vars.phi2);

        G_UU[0][0] = Ginv_coeff *(2*f_coeff
                                + 6* cp_phi2*cp_phi2*vars.phi2*vars.phi2);
        G_UU[0][1] = Ginv_coeff *(-6* cp_phi*cp_phi2*vars.phi*vars.phi2);
        G_UU[1][0] = G_UU[0][1];
        G_UU[1][1] = Ginv_coeff *(2*f_coeff
                                + 6* cp_phi*cp_phi*vars.phi*vars.phi);

    }

    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_sfchris( Tensor<2, data_t, 2> &sf1_chris,
                           Tensor<2, data_t, 2> &sf2_chris,
                           const vars_t<data_t> &vars) const
    {
      double cp_phi = m_params.sf1_a_coupling;
      double cp_phi2 = m_params.sf2_a_coupling;
      data_t f_coeff = 0.5 *(Mp*Mp + cp_phi * vars.phi*vars.phi
                                   + cp_phi2 * vars.phi2*vars.phi2 );
      data_t C_coeff = 2*f_coeff + 6*cp_phi*cp_phi*vars.phi*vars.phi
                                 + 6*cp_phi2*cp_phi2*vars.phi2*vars.phi2;
      data_t G_coeff = Mp*Mp / (4*f_coeff*f_coeff);
      data_t Ginv_coeff = 2*f_coeff/(Mp*Mp *C_coeff);

      sf1_chris[0][0] = cp_phi*(1+6*cp_phi)*vars.phi/C_coeff
                        - cp_phi*vars.phi/f_coeff;
      sf1_chris[0][1] = - cp_phi2 * vars.phi2/(2*f_coeff);
      sf1_chris[1][0] = - cp_phi2 * vars.phi2/(2*f_coeff);
      sf1_chris[1][1] = cp_phi*(1+6*cp_phi2)*vars.phi/C_coeff;

      sf2_chris[0][0] = cp_phi2*(1+6*cp_phi)*vars.phi2/C_coeff;
      sf2_chris[0][1] = - cp_phi * vars.phi/(2*f_coeff);
      sf2_chris[1][0] = - cp_phi * vars.phi/(2*f_coeff);
      sf2_chris[1][1] = cp_phi2*(1+6*cp_phi2)*vars.phi2/C_coeff
                        - cp_phi2*vars.phi2/f_coeff;




    }



    //! Set the potential function for the scalar field here
    template <class data_t, template <typename> class vars_t>
    void compute_potential(data_t &V_of_phi, data_t &dVdphi,
                           data_t &V_of_phi2, data_t &dVdphi2,
                           const vars_t<data_t> &vars) const
    {

     double g = m_params.g_coupling;
     data_t mass_SI = 9.6e-11;  // M = (3.13e-3)^4
     data_t b = 125./175;  // lambda parameter in HI inflation
     data_t a = sqrt(b/4/mass_SI);  // zi parameter in HI inflation
     data_t v = 1.016*1e-16;  // expectation value / Mp



     // convert phi --> h (Jordan Frame)
     /*
     int max_iter = 120;
     data_t func = 0;
     data_t d_func = 0;
     data_t sf = vars.phi / sqrt(2) / Mp;  // dimensionless scalar field
     data_t h = sf;   // first guess for h
     for (int i = 0; i < max_iter; i++) {
        func = sqrt((1+6*a)/(2*a)) * asinh(h*sqrt(a*(1+6*a)))
              - sqrt(3)*atanh(a*sqrt(6)*h/sqrt(1+a*(1+6*a)*h*h) ) - sf;

        d_func =  sqrt(1+ a*(1+6*a)*h*h)/(sqrt(2)*(1+a*h*h));

        h = h - func/d_func *0.3;
      }

      // // Check convergence
      // h = (func/sf < 1e-4) ? h : sqrt(-1);
      */
      data_t h = vars.phi / Mp;



      // Calculate Higgs Potential from h
      data_t F = 1 +a*h*h;
      data_t U = 0.25*b*(h*h - v*v)*(h*h - v*v);
      data_t V_HI = pow(Mp , 4) * U/F/F;


      V_of_phi =  V_HI +
	               g *h*h* vars.phi2*vars.phi2; // CJ



      dVdphi =   pow(Mp , 3) * b*h*(h*h - v*v) * (1+a*v*v) *
                    //1. /((1+a*h*h)*(1+a*h*h) * sqrt(1+a*(1+6*a)*h*h)) +
                    1. /((1+a*h*h)*(1+a*h*h)*(1+a*h*h)) +
                    2 * g * h * vars.phi2*vars.phi2; // CJ


      dVdphi2 =   2 * g * h*h * vars.phi2; // CJ


     // double SI_mass = pow( 3.1e-3*Mp , 4);  // 1.462066e-13
     // dVdphi =   2.0 * SI_mass *
     //                  sqrt(2.0/3.0) * exp(- sqrt(2.0/3.0) * vars.phi / Mp ) *
     //                  (1.0 - exp(- sqrt(2.0/3.0) * vars.phi  /Mp)) / Mp +
		 //            2 * g * vars.phi * vars.phi2*vars.phi2; // CJ
     //
     // V_of_phi2 = 0.0;
     //
     // dVdphi2 =   2 * g * vars.phi*vars.phi * vars.phi2; // CJ



    //V_of_phi2 = 0.0;
    //dVdphi2 = 0.0;

    }
};

#endif /* POTENTIAL_HPP_ */
