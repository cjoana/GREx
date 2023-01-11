/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "SetLevelData.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "MyPhiPiFunction.H"
#include "MyPotentialFunction.H"
#include "PoissonParameters.H"
#include "SetBinaryBH.H"
#include "SetLevelDataCpp.H"
#include "SetLevelDataF_F.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>

//#include "ReadHDF5.H"

// Set various LevelData functions across the grid

// set initial guess value for the conformal factor psi
// defined by \gamma_ij = \psi^4 \tilde \gamma_ij, scalar field phi
// and \bar Aij = psi^2 A_ij.
// For now the default setup is 2 Bowen York BHs plus a scalar field
// with some initial user specified configuration
void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars,
                            LevelData<FArrayBox> &a_dpsi, const RealVect &a_dx,
                            const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);



    //   readHDF5(a_multigrid_vars, a_params);
    DataIterator dit = a_multigrid_vars.dataIterator();
    const DisjointBoxLayout &grids = a_multigrid_vars.disjointBoxLayout();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];
        Box b = multigrid_vars_box.box();
        Box b_no_ghosts = grids[dit()];
        BoxIterator bit(b);
        for (bit.begin(); bit.ok(); ++bit)
        {

            // work out location on the grid
            IntVect iv = bit();

            // set psi to 1.0 and zero dpsi
            // note that we don't include the singular part of psi
            // for the BHs - this is added at the output data stage
            // and when we calculate psi_0 in the rhs etc
            // as it already satisfies Laplacian(psi) = 0
            multigrid_vars_box(iv, c_psi_0) = 1.0;
            if (a_params.read_from_data != "none")
            {
                RealVect loc(iv + 0.5 * RealVect::Unit);
                loc *= a_dx;
                loc -= a_params.domainLength / 2.0;
                Real r = sqrt(D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]));

                // interpolate
                int indxL = static_cast<int>(floor(r / a_params.spacing));
                int indxH = static_cast<int>(ceil(r / a_params.spacing));
                double inputpsiL = *(a_params.psi + indxL);
                double inputpsiH = *(a_params.psi + indxH);
                Real psivals;
                psivals = inputpsiL + (r / a_params.spacing - indxL) *
                                          (inputpsiH - inputpsiL);
                multigrid_vars_box(iv, c_psi_0) = psivals;
            }
            dpsi_box(iv, c_psi) = 0.0;

            // JCAurre: initialized the new variables and the linear
            // solutions
            dpsi_box(iv, c_V0) = 0.0;
            dpsi_box(iv, c_V1) = 0.0;
            dpsi_box(iv, c_V2) = 0.0;

            multigrid_vars_box(iv, c_V0_0) = 0.0;
            multigrid_vars_box(iv, c_V1_0) = 0.0;
            multigrid_vars_box(iv, c_V2_0) = 0.0;
        }

        // JCAurre: out of the box loop so that there are no race condition
        // problems
        FArrayBox grad_multigrid(b_no_ghosts, 3 * NUM_MULTIGRID_VARS);

pout() << " !!! Flag0  ... " << endl;
        get_grad(b_no_ghosts, multigrid_vars_box, Interval(c_V0_0, c_V2_0),
                 a_dx, grad_multigrid, a_params);

pout() << " !!! Flag1  ... " << endl;

        BoxIterator bit_no_ghosts(b_no_ghosts);
        for (bit_no_ghosts.begin(); bit_no_ghosts.ok(); ++bit_no_ghosts)
        {

            // work out location on the grid
            IntVect iv = bit_no_ghosts();

            // set the phi value - need the distance from centre
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // JCAurre: set Aij from components of vector W
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
        }

        // reopen the loop
        for (bit.begin(); bit.ok(); ++bit)
        {

            // work out location on the grid
            IntVect iv = bit();

            // set the phi value - need the distance from centre
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // set phi and pi according to user defined function
            multigrid_vars_box(iv, c_phi_0) =
                my_phi_function(loc, a_params.phi_background, a_params.phi_amplitude,
                                a_params.phi_wavelength, a_params.domainLength);

            multigrid_vars_box(iv, c_pi_0) =
                my_pi_function(loc, a_params.pi_background, a_params.pi_amplitude,
                               a_params.pi_wavelength, a_params.domainLength);

#ifdef SET_2SF 
             multigrid_vars_box(iv, c_phi2_0) =
                 my_phi2_function(loc, a_params.phi2_background, a_params.phi2_amplitude,
                                 a_params.phi2_wavelength, a_params.domainLength);

             multigrid_vars_box(iv, c_pi2_0) =
                 my_pi2_function(loc, a_params.pi2_background, a_params.pi2_amplitude,
                                a_params.pi2_wavelength, a_params.domainLength);
#endif


            multigrid_vars_box(iv, c_h11_0) = 1.0;
            multigrid_vars_box(iv, c_h12_0) = 0.0;
            multigrid_vars_box(iv, c_h13_0) = 0.0;
            multigrid_vars_box(iv, c_h22_0) = 1.0;
            multigrid_vars_box(iv, c_h23_0) = 0.0;
            multigrid_vars_box(iv, c_h33_0) = 1.0;
        }
    }
} // end set_initial_conditions

// set the rhs source for the poisson eqn
void set_rhs(LevelData<FArrayBox> &a_rhs,
             LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        rhs_box.setVal(0.0, 0);
        Box this_box = rhs_box.box(); // no ghost cells
        Box this_box_ghosts = multigrid_vars_box.box();

        // calculate gradients for constructing rho and Aij
        FArrayBox grad_multigrid(this_box, 3 * NUM_MULTIGRID_VARS);
        get_grad(this_box, multigrid_vars_box, Interval(c_psi_0, c_h33_0), a_dx,
                 grad_multigrid, a_params);

        // calculate the laplacian of Psi across the box
        FArrayBox laplace_multigrid(this_box, NUM_CONSTRAINTS_VARS);
        get_laplacian(this_box, multigrid_vars_box, Interval(c_psi, c_V2), a_dx,
                      laplace_multigrid, a_params);

        // calculate the rho contribution from gradients of phi
        FArrayBox rho_gradient(this_box, 1);
        // FORT_GETRHOGRADPHIF(CHF_FRA1(rho_gradient, 0),
        //                     CHF_CONST_FRA1(multigrid_vars_box, c_phi_0),
        //                     CHF_CONST_REAL(a_dx[0]), CHF_BOX(this_box));
        get_rhograd(this_box, multigrid_vars_box, a_dx, rho_gradient,
                    a_params); // new way to calculate it using hij

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // rhs = m/8 psi_0^5 - 2 pi rho_grad psi_0  - laplacian(psi_0)
            Real m = 0;
            set_m_value(m,
                        multigrid_vars_box(iv, c_phi_0),
#ifdef SET_2SF 
                        multigrid_vars_box(iv, c_phi2_0),
#endif               
                        a_params,
                        constant_K);

            // JCAurre: new variables for mom
            Real pi_0 = multigrid_vars_box(iv, c_pi_0);
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
            set_binary_bh_Aij(multigrid_vars_box, iv, loc, a_params);

            // Ricci term
            Real ricci = multigrid_vars_box(iv, c_R_0);

            // pout() << "ricci term is " << ricci << endl;

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi) + psi_bh;

            rhs_box(iv, c_psi) =
                0.125 * ricci * psi_0 +
                0.125 * (m - 8.0 * M_PI * a_params.G_Newton * pow(pi_0, 2.0)) *
                    pow(psi_0, 5.0) -
                0.125 * A2 * pow(psi_0, -7.0) -
                2.0 * M_PI * a_params.G_Newton * rho_gradient(iv, 0) * psi_0 -
                laplace_multigrid(iv, c_psi);

            // JCAurre: Added rhs for new constraint variables.
            rhs_box(iv, c_V0) = -8.0 * M_PI * pow(psi_0, 6.0) * pi_0 *
                                    grad_multigrid(iv, 3 * c_phi_0 + 0) -
                                laplace_multigrid(iv, c_V0);
            rhs_box(iv, c_V1) = -8.0 * M_PI * pow(psi_0, 6.0) * pi_0 *
                                    grad_multigrid(iv, 3 * c_phi_0 + 1) -
                                laplace_multigrid(iv, c_V1);
            rhs_box(iv, c_V2) = -8.0 * M_PI * pow(psi_0, 6.0) * pi_0 *
                                    grad_multigrid(iv, 3 * c_phi_0 + 2) -
                                laplace_multigrid(iv, c_V2);
        }
    }
} // end set_rhs

// Set the integrand for the integrability condition for constant K
// when periodic BCs set
void set_constant_K_integrand(LevelData<FArrayBox> &a_integrand,
                              LevelData<FArrayBox> &a_multigrid_vars,
                              const RealVect &a_dx,
                              const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_integrand.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &integrand_box = a_integrand[dit()];
        integrand_box.setVal(0.0, 0);
        Box this_box = integrand_box.box(); // no ghost cells
        Box this_box_ghosts = multigrid_vars_box.box();

        // calculate gradients for constructing rho and Aij
        FArrayBox grad_multigrid(this_box, 3 * NUM_MULTIGRID_VARS);
        get_grad(this_box, multigrid_vars_box, Interval(c_psi_0, c_h33_0), a_dx,
                 grad_multigrid, a_params);

        // calculate the laplacian of Psi across the box
        FArrayBox laplace_multigrid(this_box, NUM_CONSTRAINTS_VARS);
        get_laplacian(this_box, multigrid_vars_box, Interval(c_psi, c_V2), a_dx,
                      laplace_multigrid, a_params);

        // calculate the rho contribution from gradients of phi
        FArrayBox rho_gradient(this_box, 1);
        // FORT_GETRHOGRADPHIF(CHF_FRA1(rho_gradient, 0),
        //                     CHF_CONST_FRA1(multigrid_vars_box, c_phi_0),
        //                     CHF_CONST_REAL(a_dx[0]), CHF_BOX(this_box));
        get_rhograd(this_box, multigrid_vars_box, a_dx, rho_gradient,
                    a_params); // new way to calculate it using hij

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // integrand = -1.5*m + 1.5 * \bar A_ij \bar A^ij psi_0^-12 +
            // 24 pi rho_grad psi_0^-4  + 12*laplacian(psi_0)*psi^-5
            Real m = 0;
            set_m_value(m,
                        multigrid_vars_box(iv, c_phi_0),
#ifdef SET_2SF 
                        multigrid_vars_box(iv, c_phi2_0),
#endif               
                        a_params,
                        0);

            // JCAurre: new variables for mom
            Real pi_0 = multigrid_vars_box(iv, c_pi_0);
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
            set_binary_bh_Aij(multigrid_vars_box, iv, loc, a_params);

            // Ricci term
            Real ricci = multigrid_vars_box(iv, c_R_0);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi) + psi_bh;

            integrand_box(iv, 0) =
                -1.5 * ricci * pow(psi_0, -4.0) -
                1.5 * (m - 8.0 * M_PI * a_params.G_Newton * pow(pi_0, 2.0)) +
                1.5 * A2 * pow(psi_0, -12.0) +
                24.0 * M_PI * a_params.G_Newton * rho_gradient(iv, 0) *
                    pow(psi_0, -4.0) +
                12.0 * laplace_multigrid(iv, c_psi) * pow(psi_0, -5.0);

            integrand_box(iv, c_V0) = 0;
            integrand_box(iv, c_V1) = 0;
            integrand_box(iv, c_V2) = 0;
        }
    }
} // end set_constant_K_integrand

// set the regrid condition - abs value of this drives AMR regrid
void set_regrid_condition(LevelData<FArrayBox> &a_condition,
                          LevelData<FArrayBox> &a_multigrid_vars,
                          const RealVect &a_dx,
                          const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_condition.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &condition_box = a_condition[dit()];
        condition_box.setVal(0.0, 0);
        Box this_box = condition_box.box(); // no ghost cells
        Box this_box_ghosts = multigrid_vars_box.box();

        // calculate gradients for constructing rho and Aij
        FArrayBox grad_multigrid(this_box, 3 * NUM_MULTIGRID_VARS);
        get_grad(this_box, multigrid_vars_box, Interval(c_psi_0, c_h33_0), a_dx,
                 grad_multigrid, a_params);

        // calculate the laplacian of Psi across the box
        FArrayBox laplace_multigrid(this_box, NUM_CONSTRAINTS_VARS);
        get_laplacian(this_box, multigrid_vars_box, Interval(c_psi, c_V2), a_dx,
                      laplace_multigrid, a_params);

        // calculate the rho contribution from gradients of phi
        FArrayBox rho_gradient(this_box, 1);
        // FORT_GETRHOGRADPHIF(CHF_FRA1(rho_gradient, 0),
        //                     CHF_CONST_FRA1(multigrid_vars_box, c_phi_0),
        //                     CHF_CONST_REAL(a_dx[0]), CHF_BOX(this_box));
        get_rhograd(this_box, multigrid_vars_box, a_dx, rho_gradient,
                    a_params); // new way to calculate it using hij

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // calculate contributions
            Real m = 0;
            set_m_value(m,
                        multigrid_vars_box(iv, c_phi_0),
#ifdef SET_2SF 
                        multigrid_vars_box(iv, c_phi2_0),
#endif               
                        a_params,
                        0);

            // JCAurre: new variables for mom
            Real pi_0 = multigrid_vars_box(iv, c_pi_0);
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
            set_binary_bh_Aij(multigrid_vars_box, iv, loc, a_params);

            // Ricci term
            Real ricci = multigrid_vars_box(iv, c_R_0);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            // the condition is similar to the rhs but we take abs
            // value of the contributions and add in BH criteria
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi) + psi_bh;

            condition_box(iv, 0) = // Need to think how to add here Ricci
                1.5 *
                    abs((m - 8.0 * M_PI * a_params.G_Newton * pow(pi_0, 2.0))) +
                1.5 * A2 * pow(psi_0, -7.0) +
                24.0 * M_PI * a_params.G_Newton * abs(rho_gradient(iv, 0)) *
                    pow(psi_0, 1.0) +
                log(psi_0);

            condition_box(iv, c_V0) = 0.;
            condition_box(iv, c_V1) = 0.;
            condition_box(iv, c_V2) = 0.;
        }
    }
} // end set_regrid_condition

// Add the correction to psi0 after the solver operates
void set_update_psi0(LevelData<FArrayBox> &a_multigrid_vars,
                     LevelData<FArrayBox> &a_dpsi,
                     const Copier &a_exchange_copier)
{

    // first exchange ghost cells for dpsi so they are filled with the
    // correct values
    a_dpsi.exchange(a_dpsi.interval(), a_exchange_copier);

    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];

        Box this_box = multigrid_vars_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            multigrid_vars_box(iv, c_psi_0) += dpsi_box(
                iv,
                c_psi); // JCAurre changed c_psi to c_psi_0 and 0 to c_psi

            // JCAurre: update constraint variables
            multigrid_vars_box(iv, c_V0_0) += dpsi_box(iv, c_V0);
            multigrid_vars_box(iv, c_V1_0) += dpsi_box(iv, c_V1);
            multigrid_vars_box(iv, c_V2_0) += dpsi_box(iv, c_V2);
        }
    }
}

// m(K, rho) = 2/3K^2 - 16piG rho

void set_m_value(Real &m, const Real &phi_here,
                 const PoissonParameters &a_params, const Real constant_K)
{

    // KC TODO:
    // For now rho is just the gradient term which is kept separate
    // ... may want to add V(phi) and phidot/Pi here later though
    Real V_of_phi = my_potential_function(phi_here, a_params);
    Real rho = V_of_phi;

    m = (2.0 / 3.0) * (constant_K * constant_K) -
        16.0 * M_PI * a_params.G_Newton * rho;
}

#ifdef SET_2SF
void set_m_value(Real &m, const Real &phi_here, const Real &phi2_here,
                 const PoissonParameters &a_params, const Real constant_K)
{

    Real V_of_phi = my_potential_function(phi_here, phi2_here, a_params);
    Real rho = V_of_phi;

    m = (2.0 / 3.0) * (constant_K * constant_K) -
        16.0 * M_PI * a_params.G_Newton * rho;
}
#endif

// The coefficient of the I operator on dpsi
void set_a_coef(LevelData<FArrayBox> &a_aCoef,
                LevelData<FArrayBox> &a_multigrid_vars,
                const PoissonParameters &a_params, const RealVect &a_dx,
                const Real constant_K)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_aCoef.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &aCoef_box = a_aCoef[dit()];
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        Box this_box = aCoef_box.box();
        Box this_box_ghosts = multigrid_vars_box.box();

        // calculate gradients for constructing rho and Aij
        FArrayBox grad_multigrid(this_box, 3 * NUM_MULTIGRID_VARS);
        get_grad(this_box, multigrid_vars_box, Interval(c_psi_0, c_h33_0), a_dx,
                 grad_multigrid, a_params);

        // calculate the laplacian of Psi across the box
        FArrayBox laplace_multigrid(this_box, NUM_CONSTRAINTS_VARS);
        get_laplacian(this_box, multigrid_vars_box, Interval(c_psi, c_V2), a_dx,
                      laplace_multigrid, a_params);

        // calculate the rho contribution from gradients of phi
        FArrayBox rho_gradient(this_box, 1);
        // FORT_GETRHOGRADPHIF(CHF_FRA1(rho_gradient, 0),
        //                     CHF_CONST_FRA1(multigrid_vars_box, c_phi_0),
        //                     CHF_CONST_REAL(a_dx[0]), CHF_BOX(this_box));
        get_rhograd(this_box, multigrid_vars_box, a_dx, rho_gradient,
                    a_params); // new way to calculate it using hij

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;
            // m(K, phi) = 2/3 K^2 - 16 pi G rho
            Real m;
            set_m_value(m,
                        multigrid_vars_box(iv, c_phi_0),
                        #ifdef SET_2SF 
                        multigrid_vars_box(iv, c_phi2_0),
                        #endif
                        a_params,
                        constant_K);

            // JCAurre: new variables for mom
            Real pi_0 = multigrid_vars_box(iv, c_pi_0);
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
            set_binary_bh_Aij(multigrid_vars_box, iv, loc, a_params);

            // Ricci term
            Real ricci = multigrid_vars_box(iv, c_R_0);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi) + psi_bh;

            aCoef_box(iv, c_psi) =
                -0.125 * ricci -
                0.625 * (m - 8.0 * M_PI * a_params.G_Newton * pow(pi_0, 2.0)) *
                    pow(psi_0, 4.0) -
                0.875 * A2 * pow(psi_0, -8.0) +
                2.0 * M_PI * a_params.G_Newton * rho_gradient(iv, 0);

            // JCAurre: eq is linear so 0 should be fine
            aCoef_box(iv, c_V0) = 0.0;
            aCoef_box(iv, c_V1) = 0.0;
            aCoef_box(iv, c_V2) = 0.0;
        }
    }
}

// The coefficient of the Laplacian operator, for now set to constant 1
// Note that beta = -1 so this sets the sign
// the rhs source of the Poisson eqn
void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_bCoef.nComp() == NUM_CONSTRAINTS_VARS);
    int comp_number = 0;

    for (DataIterator dit = a_bCoef.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &bCoef_box = a_bCoef[dit()];
        // JCAurre: Loop to set bCoef=1 for all constraint variables
        for (int iconstraint = 0; iconstraint < NUM_CONSTRAINTS_VARS;
             iconstraint++)
        {
            bCoef_box.setVal(1.0, iconstraint);
        }
    }
}

// used to set output data for all ADM Vars for GRChombo restart
void set_output_data(LevelData<FArrayBox> &a_grchombo_vars,
                     LevelData<FArrayBox> &a_multigrid_vars,
                     const PoissonParameters &a_params, const RealVect &a_dx,
                     const Real &constant_K)
{

    CH_assert(a_grchombo_vars.nComp() == NUM_GRCHOMBO_VARS);
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    const DisjointBoxLayout &grids = a_grchombo_vars.disjointBoxLayout();
    DataIterator dit = a_grchombo_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &grchombo_vars_box = a_grchombo_vars[dit()];
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];

        // first set everything to zero
        for (int comp = 0; comp < NUM_GRCHOMBO_VARS; comp++)
        {
            grchombo_vars_box.setVal(0.0, comp);
        }

        // now set non zero terms - const across whole box
        // Conformally flat, and lapse = 1
        // Comment for non conformally flat data
        // grchombo_vars_box.setVal(1.0, c_h11);
        // grchombo_vars_box.setVal(1.0, c_h22);
        // grchombo_vars_box.setVal(1.0, c_h33);
        grchombo_vars_box.setVal(1.0, c_lapse);

        // constant K
        grchombo_vars_box.setVal(constant_K, c_K);

        // now non constant terms by location
        Box this_box = grchombo_vars_box.box();
        BoxIterator bit(this_box);
        Box this_box_ng = grids[dit()];

        // calculate gradients for constructing Aij
        FArrayBox grad_multigrid(this_box_ng, 3 * NUM_MULTIGRID_VARS);
        get_grad(this_box_ng, multigrid_vars_box, Interval(c_V0_0, c_V2_0),
                 a_dx, grad_multigrid, a_params);

        // Aij is defined to be zero in the ghost cells, so be careful with
        // fixed BCs
        BoxIterator bit_no_ghosts(this_box_ng);
        for (bit_no_ghosts.begin(); bit_no_ghosts.ok(); ++bit_no_ghosts)
        {

            // work out location on the grid
            IntVect iv = bit_no_ghosts();

            // set the phi value - need the distance from centre
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // JCAurre: set Aij from components of vector W
            set_Aij_0(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid);
        }

        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // GRChombo conformal factor chi = psi^-4
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real chi = pow(multigrid_vars_box(iv, c_psi) + psi_bh, -4.0);
            grchombo_vars_box(iv, c_chi) = chi;
            Real factor = pow(chi, 1.5);

            // Copy phi and Aij across - note this is now \tilde Aij not
            // \bar Aij
            grchombo_vars_box(iv, c_phi) = multigrid_vars_box(iv, c_phi_0);
            grchombo_vars_box(iv, c_Pi) = multigrid_vars_box(iv, c_pi_0);
            grchombo_vars_box(iv, c_A11) =
                multigrid_vars_box(iv, c_A11_0) * factor;
            grchombo_vars_box(iv, c_A12) =
                multigrid_vars_box(iv, c_A12_0) * factor;
            grchombo_vars_box(iv, c_A13) =
                multigrid_vars_box(iv, c_A13_0) * factor;
            grchombo_vars_box(iv, c_A22) =
                multigrid_vars_box(iv, c_A22_0) * factor;
            grchombo_vars_box(iv, c_A23) =
                multigrid_vars_box(iv, c_A23_0) * factor;
            grchombo_vars_box(iv, c_A33) =
                multigrid_vars_box(iv, c_A33_0) * factor;

            grchombo_vars_box(iv, c_h11) = multigrid_vars_box(iv, c_h11_0);
            grchombo_vars_box(iv, c_h12) = multigrid_vars_box(iv, c_h12_0);
            grchombo_vars_box(iv, c_h13) = multigrid_vars_box(iv, c_h13_0);
            grchombo_vars_box(iv, c_h22) = multigrid_vars_box(iv, c_h22_0);
            grchombo_vars_box(iv, c_h23) = multigrid_vars_box(iv, c_h23_0);
            grchombo_vars_box(iv, c_h33) = multigrid_vars_box(iv, c_h33_0);
        }
    }
}
