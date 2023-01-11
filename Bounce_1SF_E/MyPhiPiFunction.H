/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef MYPHIFUNCTION_HPP
#define MYPHIFUNCTION_HPP

Real my_phi_function(RealVect loc, Real background, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
//        Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
//        return amplitude * exp(-r2 / wavelength);

    Real perturb = 0;
    int nmodes = 10;
    
    double fact = 1;
    double rand = 0.85698235; 

    for(int i=3; i<nmodes; i++){

	    rand += rand;
	    perturb += amplitude * (sin(2 * M_PI * loc[0] * i / L[0] + rand) +
                        sin(2 * M_PI * loc[1] * i / L[1] + rand) +
                        sin(2 * M_PI * loc[2] * i / L[2] + rand)
			); 
    };


    fact = 0.01;
    for(int i=1; i<3; i++){

            rand += rand;

            perturb += amplitude *  fact* (sin(2 * M_PI * loc[0] * i / L[0] + rand) +
                        sin(2 * M_PI * loc[1] * i / L[1] + rand) +
                        sin(2 * M_PI * loc[2] * i / L[2] + rand)
                        );
    };
    

    // sine waves in 3 directions - use with periodic BCs
    return background + perturb;
    			// amplitude * (sin(2 * M_PI * loc[0] * wavelength / L[0]) +
                        // sin(2 * M_PI * loc[1] * wavelength / L[1]) +
                        // sin(2 * M_PI * loc[2] * wavelength / L[2]));
}

Real my_pi_function(RealVect loc, Real background, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
//        Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
//        return amplitude * exp(-r2 / wavelength);

    // sine waves in 3 directions - use with periodic BCs
    return background + amplitude * (sin(2 * M_PI * loc[0] * wavelength / L[0]) +
                        sin(2 * M_PI * loc[1] * wavelength / L[1]) +
                        sin(2 * M_PI * loc[2] * wavelength / L[2]));
}


Real my_phi2_function(RealVect loc, Real background, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
//        Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
//        return amplitude * exp(-r2 / wavelength);


    Real perturb = 0;
    int nmodes = 10;

    double rand = 1.698235;

    for(int i=1; i<nmodes; i++){

            rand += rand;

            perturb += amplitude * (sin(2 * M_PI * loc[0] * i / L[0] + rand) +
                        sin(2 * M_PI * loc[1] * i / L[1] + rand) +
                        sin(2 * M_PI * loc[2] * i / L[2] + rand)
			);  
    };

    // sine waves in 3 directions - use with periodic BCs
    return background + perturb;

    // sine waves in 3 directions - use with periodic BCs
//    return background + amplitude * (sin(2 * M_PI * loc[0] * wavelength / L[0]) +
//                        sin(2 * M_PI * loc[1] * wavelength / L[1]) +
//                        sin(2 * M_PI * loc[2] * wavelength / L[2]));
}

Real my_pi2_function(RealVect loc, Real background, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
//        Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
//        return amplitude * exp(-r2 / wavelength);

    // sine waves in 3 directions - use with periodic BCs
    return background + amplitude * (sin(2 * M_PI * loc[0] * wavelength / L[0]) +
                        sin(2 * M_PI * loc[1] * wavelength / L[1]) +
                        sin(2 * M_PI * loc[2] * wavelength / L[2]));
}

#endif /* MYPHIFUNCTION_HPP */
