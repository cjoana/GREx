/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYPHIFUNCTION_HPP
#define MYPHIFUNCTION_HPP


Real my_rho_function(RealVect loc, Real background,  Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
    Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);

    Real wavesin = 1.;
    Real ampsin = amplitude;

    return background
		     + amplitude * exp( -r2 / (1. * L[0] * L[0] / wavelength ) );
                      // + ampsin * sin(2*M_PI * loc[0] * wavesin / L[0])
                      // + ampsin * sin(2*M_PI * loc[1] * wavesin / L[1])
                      // + ampsin * sin(2*M_PI * loc[2] * wavesin / L[2]);

    // return background + amplitude * exp( -r2 / (wavelength * L[0] * L[0] / 4 ) );
}

#endif /* MYPHIFUNCTION_HPP */
