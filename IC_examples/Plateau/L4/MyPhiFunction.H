/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYPHIFUNCTION_HPP
#define MYPHIFUNCTION_HPP

Real my_phi_function(RealVect loc, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
    Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
    return amplitude * exp(-r2 / wavelength);

    // sine waves in 3 directions - use with periodic BCs
    // return amplitude * (sin(2*M_PI * loc[0] * wavelength / L[0])
    //                       + sin(2*M_PI * loc[1] * wavelength / L[1])
    //                       + sin(2*M_PI * loc[2] * wavelength / L[2]));
}


Real my_phi_function(RealVect loc, Real background,  Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
    Real jitter = 0.0;
    // Real r1 = D_TERM((loc[0]+jitter) * (loc[0]+jitter), +loc[1] * loc[1], +loc[2] * loc[2]);
    Real r2 = D_TERM((loc[0]-jitter) * (loc[0]-jitter), +loc[1] * loc[1], +loc[2] * loc[2]);

    Real wavesin = 1.;
    Real ampsin = 0.1;
    Real arg = 1. / (  L[0] * L[0] / wavelength / wavelength);

    
    return background + amplitude * exp( -r2 * arg ) 
		      +
                      (
		      // + 1.2 * ampsin * sin(2*M_PI * loc[0] * wavesin / L[0]  + 0.33 * M_PI)
		      // + 1.3 * ampsin * sin(1*M_PI * loc[0] * wavesin / L[0]  + 0.69 * M_PI)
		      + amplitude * ampsin * sin(2*M_PI * loc[0] * wavesin / L[0]  + 1.5 * M_PI)
		      // + 0.3 * ampsin * sin(5*M_PI * loc[0] * wavesin / L[0]  + 0.1 * M_PI)

                      + amplitude * ampsin * sin(2*M_PI * loc[1] * wavesin / L[1]  + 0.39 * M_PI)
                      // + 1.3 * ampsin * sin(2*M_PI * loc[1] * wavesin / L[1]  + 0.2  * M_PI)
                      // + 0.7 * ampsin * sin(4*M_PI * loc[1] * wavesin / L[1]  + 0.88 * M_PI)
                      // + 0.8 * ampsin * sin(1*M_PI * loc[1] * wavesin / L[1]  + 1.42 * M_PI)

           	      + amplitude * ampsin * sin(2*M_PI * loc[2] * wavesin / L[2]  + 0.56 * M_PI)
		      // + 0.2 * ampsin * sin(8*M_PI * loc[2] * wavesin / L[2]  + 0.1 * M_PI)
		      // + 1.2 * ampsin * sin(3*M_PI * loc[2] * wavesin / L[2]  + 1.7 * M_PI)
		      // + 1.0 * ampsin * sin(2*M_PI * loc[2] * wavesin / L[2]  + 1.2 * M_PI)
                      );
    

     // return background + amplitude * exp( -r2 / (  L[0] * L[0] / wavelength / wavelength ) );
     //	           - amplitude * exp( -r1 / (  L[0] * L[0] / wavelength / wavelength ) );
}



Real my_pi_function(RealVect loc, Real background,  Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
    Real jitter = 0.0;
    // Real r1 = D_TERM((loc[0]+jitter) * (loc[0]+jitter), +loc[1] * loc[1], +loc[2] * loc[2]);
    Real r2 = D_TERM((loc[0]-jitter) * (loc[0]-jitter), +loc[1] * loc[1], +loc[2] * loc[2]);

    Real wavesin = 1.;
    Real ampsin = 0.1;
    Real arg = 1. / (  L[0] * L[0] / wavelength / wavelength);

    
    return background + amplitude * exp( -r2 * arg ) 
		      +
                      (
		      // + 1.2 * ampsin * sin(2*M_PI * loc[0] * wavesin / L[0]  + 0.33 * M_PI)
		      // + 1.3 * ampsin * sin(1*M_PI * loc[0] * wavesin / L[0]  + 0.69 * M_PI)
		      + amplitude * ampsin * sin(2*M_PI * loc[0] * wavesin / L[0]  + 1.5 * M_PI)
		      // + 0.3 * ampsin * sin(5*M_PI * loc[0] * wavesin / L[0]  + 0.1 * M_PI)

                      + amplitude * ampsin * sin(2*M_PI * loc[1] * wavesin / L[1]  + 0.39 * M_PI)
                      // + 1.3 * ampsin * sin(2*M_PI * loc[1] * wavesin / L[1]  + 0.2  * M_PI)
                      // + 0.7 * ampsin * sin(4*M_PI * loc[1] * wavesin / L[1]  + 0.88 * M_PI)
                      // + 0.8 * ampsin * sin(1*M_PI * loc[1] * wavesin / L[1]  + 1.42 * M_PI)

           	      + amplitude * ampsin * sin(2*M_PI * loc[2] * wavesin / L[2]  + 0.56 * M_PI)
		      // + 0.2 * ampsin * sin(8*M_PI * loc[2] * wavesin / L[2]  + 0.1 * M_PI)
		      // + 1.2 * ampsin * sin(3*M_PI * loc[2] * wavesin / L[2]  + 1.7 * M_PI)
		      // + 1.0 * ampsin * sin(2*M_PI * loc[2] * wavesin / L[2]  + 1.2 * M_PI)
                      );


     
     // return background + amplitude * exp( - r2 * arg ) * (4 * arg * r2 - 2);    // dd of exp(- r2 *arg)
     // return background + amplitude * exp( - r2 * arg ) * (arg * r2 );    // dd2
     	           // - amplitude * exp( -r1 / (  L[0] * L[0] / wavelength / wavelength ) );
}


// m(K, rho) = 2/3K^2 - 16piG rho
void set_m_value(Real &m, const Real &phi_here, const Real &pi_here,
                 const PoissonParameters &a_params, const Real constant_K)
{

    // KC TODO:
    // For now rho is just the gradient term which is kept separate
    // ... may want to add V(phi) and phidot/Pi here later though
    Real V_of_phi = 0.0;
    Real Mp = 1.0/sqrt(8.0*M_PI);  // CJ
    Real scalar_mass = 9.6e-11;    // CJ  1801.04948 p.31

    V_of_phi =  scalar_mass * pow(1.0*Mp, 4.0) *
         pow(1.0 - exp(-sqrt(2.0/3.0) * abs(phi_here) / Mp), 2.0); // CJ Higgs

    // V_of_phi =  scalar_mass * pow(1.0*Mp, 4.0) *
    //     pow(1.0 - exp(-sqrt(2.0/3.0) * phi_here / Mp), 2.0); // CJ Starobinski



    Real rho = 0.5 * pi_here * pi_here + V_of_phi;

    m = (2.0 / 3.0) * (constant_K * constant_K) -
        16.0 * M_PI * a_params.G_Newton * rho;
}


#endif /* MYPHIFUNCTION_HPP */
