/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYPOTENTIALFUNCTION_HPP
#define MYPOTENTIALFUNCTION_HPP

Real my_potential_function(const Real &phi_here, const Real &phi2_here,
                 const PoissonParameters &a_params)
{

	Real g =  a_params.g_coupling;
  Real mass_SI = 9.6e-11;  // M = (3.13e-3)^4
  Real b = 125./175;  // lambda parameter in HI inflation
  Real a = sqrt(b/4/mass_SI);  // zi parameter in HI inflation
  Real v = 1.016*1e-16;  // expectation value / Mp



  // convert phi --> h (Jordan Frame)
  int max_iter = 120;
  Real func = 0;
  Real d_func = 0;
  Real sf = phi_here / sqrt(2);  // / Mp;  // dimensionless scalar field
  Real h = sf;   // first guess for h
  for (int i = 0; i < max_iter; i++) {
 		func = sqrt((1+6*a)/(2*a)) * asinh(h*sqrt(a*(1+6*a)))
 					- sqrt(3)*atanh(a*sqrt(6)*h/sqrt(1+a*(1+6*a)*h*h) ) - sf;

 		d_func =  sqrt(1+ a*(1+6*a)*h*h)/(sqrt(2)*(1+a*h*h));

 		h = h - func/d_func *0.3;
 	}



 	Real F = 1 +a*h*h;
 	Real U = 0.25*b*(h*h - v*v)*(h*h - v*v);
 	Real V_HI =  U/F/F;  // in mass planck units  (mp^4)
 	Real V_of_phi = 0.0;


 	Real sf2 = phi2_here * sqrt(F);


 	V_of_phi =  V_HI +
 									 g * h*h * sf2 *sf2; // CJ


    //return V_of_phi;
}

#endif /* MYPOTENTIALFUNCTION_HPP */
