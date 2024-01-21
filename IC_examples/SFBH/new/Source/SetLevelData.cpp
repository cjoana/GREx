/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "SetLevelData.H"
#include "MyMatterFunctions.H"
#include "SetPsiAndAij.H"

// Set various LevelData functions across the grid

// set initial guess value for the conformal factor psi
// defined by \gamma_ij = \psi^4 \tilde \gamma_ij, scalar field phi,
// its conjugate momentum Pi and the V_i comps which compose Aij
// For now the default setup is 2 Bowen York BHs plus a scalar field
// with some initial user specified configuration
void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars,
                            LevelData<FArrayBox> &a_dpsi, const RealVect &a_dx,
                            const PoissonParameters &a_params,
                            const bool set_matter)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        // These contain the vars in the boxes, set them all to zero
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];
        for (int comp = 0; comp < NUM_MULTIGRID_VARS; comp++)
        {
            multigrid_vars_box.setVal(0.0, comp);
        }
        for (int comp = 0; comp < NUM_CONSTRAINT_VARS; comp++)
        {
            dpsi_box.setVal(0.0, comp);
        }

        // Iterate over the box and set non zero comps
        Box ghosted_box = multigrid_vars_box.box();
        BoxIterator bit(ghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // note that we don't include the singular part of psi
            // for the BHs - this is added at the output data stage
            // and when we calculate psi_reg in the rhs etc
            // as it already satisfies Laplacian(psi) = 0
            multigrid_vars_box(iv, c_psi_reg) = a_params.psi_reg;

            if (set_matter)
            {
                // set phi and pi according to user defined function
                // KC TODO pass whole box here and set matter in
                // MyMatterFunction?
                multigrid_vars_box(iv, c_phi_0) =
                    my_phi_function(loc, a_params, a_dx);
                multigrid_vars_box(iv, c_Pi_0) =
                    my_Pi_function(loc, a_params, a_dx);
            }
        }
    }
} // end set_initial_conditions

// Set values of K_ij - \bar Aij and K, based on current values
void set_update_Kij(LevelData<FArrayBox> &a_multigrid_vars,
                    LevelData<FArrayBox> &a_rhs, const RealVect &a_dx,
                    const PoissonParameters &a_params)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    // Iterate through the boxes in turn
    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        Box unghosted_box = rhs_box.box();

        // Iterate through the interior of boxes
        // (ghosts need to be filled later due to gradient terms)
        BoxIterator bit(unghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // Calculate the actual value of psi including BH part
            Real psi_reg = multigrid_vars_box(iv, c_psi_reg);
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = psi_reg + psi_bh;
            Real laplacian_psi_reg =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_psi_reg);

            // Assign values of Aij
            Tensor<2, Real> Aij_reg;
            set_Aij_reg(Aij_reg, multigrid_vars_box, iv, a_dx, a_params);
            Tensor<2, Real> Aij_bh;
            set_Aij_bh(Aij_bh, loc, a_params);
            // This is \bar  A_ij \bar A^ij
            Real A2_0 = 0.0;
            FOR2(i, j)
            {
                A2_0 += (Aij_reg[i][j] + Aij_bh[i][j]) *
                        (Aij_reg[i][j] + Aij_bh[i][j]);
            }

            // Assign values of useful matter quantities
            Real rho_matter =
                my_rho_function(iv, a_dx, multigrid_vars_box, a_params);

            // Now work out K using ansatz which sets it to (roughly)
            // the FRW value based on the local densities
            Real K_0_squared = 0.0;
            if (a_params.include_A2)
            {
                K_0_squared = 24.0 * M_PI * a_params.G_Newton * rho_matter +
                              1.5 * A2_0 * pow(psi_0, -12.0) +
                              0.0 * 12.0 * laplacian_psi_reg * pow(psi_0, -5.0);
            }
            else // in non periodic case exclude Aij terms
            {
                K_0_squared = 24.0 * M_PI * a_params.G_Newton * rho_matter;
            }

            // Set value for K
            // be careful if at a point K = 0, may have discontinuity
            multigrid_vars_box(iv, c_K_0) =
                a_params.sign_of_K * sqrt(K_0_squared);

            // set values for \bar Aij_0
            multigrid_vars_box(iv, c_A11_0) = Aij_reg[0][0] + Aij_bh[0][0];
            multigrid_vars_box(iv, c_A22_0) = Aij_reg[1][1] + Aij_bh[1][1];
            multigrid_vars_box(iv, c_A33_0) = Aij_reg[2][2] + Aij_bh[2][2];
            multigrid_vars_box(iv, c_A12_0) = Aij_reg[0][1] + Aij_bh[0][1];
            multigrid_vars_box(iv, c_A13_0) = Aij_reg[0][2] + Aij_bh[0][2];
            multigrid_vars_box(iv, c_A23_0) = Aij_reg[1][2] + Aij_bh[1][2];
        }
    }
} // end set_update_Kij

// Set rhs for the laplacian
void set_rhs(LevelData<FArrayBox> &a_rhs,
             LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
             const PoissonParameters &a_params)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        for (int comp = 0; comp < NUM_CONSTRAINT_VARS; comp++)
        {
            rhs_box.setVal(0.0, comp);
        }
        Box unghosted_box = rhs_box.box();
        BoxIterator bit(unghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // Calculate the actual value of psi including BH part
            Real psi_reg = multigrid_vars_box(iv, c_psi_reg);
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = psi_reg + psi_bh;
            Real laplacian_psi_reg =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_psi_reg);

            // Get values of Aij
            Tensor<2, Real> Aij_reg;
            set_Aij_reg(Aij_reg, multigrid_vars_box, iv, a_dx, a_params);
            Tensor<2, Real> Aij_bh;
            set_Aij_bh(Aij_bh, loc, a_params);
            // This is \bar  A_ij \bar A^ij
            Real A2_0 = 0.0;
            FOR2(i, j)
            {
                A2_0 += (Aij_reg[i][j] + Aij_bh[i][j]) *
                        (Aij_reg[i][j] + Aij_bh[i][j]);
            }

            // Momentum matter comps
            // TODO: Make matter function for S_i
            Tensor<1, Real> Si;
            my_Si_function(Si, iv, a_dx, multigrid_vars_box, a_params);
            Tensor<1, Real, SpaceDim> d1_K =
                get_d1(iv, multigrid_vars_box, a_dx, c_K_0);

            // rhs terms
            if (a_params.include_A2)
            {
                // K is set so that it cancels all rhs terms
                rhs_box(iv, c_psi) = 0.0;
            }
            else
            {
                // K is set to cancel matter terms only
                rhs_box(iv, c_psi) =
                    -0.125 * A2_0 * pow(psi_0, -7.0) - laplacian_psi_reg;
            }
            // Get d_i V_i and laplacians
            Tensor<1, Real, SpaceDim> d1_V1 =
                get_d1(iv, multigrid_vars_box, a_dx, c_V1_0);
            Tensor<1, Real, SpaceDim> d1_V2 =
                get_d1(iv, multigrid_vars_box, a_dx, c_V2_0);
            Tensor<1, Real, SpaceDim> d1_V3 =
                get_d1(iv, multigrid_vars_box, a_dx, c_V3_0);

            Real laplacian_V1 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V1_0);
            Real laplacian_V2 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V2_0);
            Real laplacian_V3 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V3_0);
            Real laplacian_U =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_U_0);

            // now set the values in the box
            rhs_box(iv, c_V1) =
                pow(psi_0, 6.0) *
                (2.0 / 3.0 * d1_K[0] + 8.0 * M_PI * a_params.G_Newton * Si[0]);
            rhs_box(iv, c_V2) =
                pow(psi_0, 6.0) *
                (2.0 / 3.0 * d1_K[1] + 8.0 * M_PI * a_params.G_Newton * Si[1]);
            rhs_box(iv, c_V3) =
                pow(psi_0, 6.0) *
                (2.0 / 3.0 * d1_K[2] + 8.0 * M_PI * a_params.G_Newton * Si[2]);

            // Periodic: Use ansatz B.3 in B&S (p547) JCA TODO: We are not using
            // this U when constructing Aij Non-periodic: Compact ansatz B.7 in
            // B&S (p547)
            if (!a_params.method_compact)
            {
                rhs_box(iv, c_U) = -0.25 * (d1_V1[0] + d1_V2[1] + d1_V3[2]);
            }
            else
            {
                rhs_box(iv, c_U) = 0.0;
                FOR1(i)
                {
                    rhs_box(iv, c_U) +=
                        -loc[i] * (pow(psi_0, 6.0) *
                                   (2.0 / 3.0 * d1_K[i] +
                                    8.0 * M_PI * a_params.G_Newton * Si[i]));
                }
            }

            if (a_params.deactivate_zero_mode)
            {
                rhs_box(iv, c_V1) += -laplacian_V1;
                rhs_box(iv, c_V2) += -laplacian_V2;
                rhs_box(iv, c_V3) += -laplacian_V3;
                rhs_box(iv, c_U) += -laplacian_U;
            }
        }
    }
} // end set_rhs

// Set error function
void set_error(LevelData<FArrayBox> &a_error, LevelData<FArrayBox> &a_rhs,
               LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
               const PoissonParameters &a_params)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        FArrayBox &error_box = a_error[dit()];
        for (int comp = 0; comp < NUM_CONSTRAINT_VARS; comp++)
        {
            error_box.setVal(0.0, comp);
        }
        Box unghosted_box = error_box.box();
        BoxIterator bit(unghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();

            Real laplacian_V1 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V1_0);
            Real laplacian_V2 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V2_0);
            Real laplacian_V3 =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_V3_0);
            Real laplacian_U =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_U_0);

            // now set the values in the box
            error_box(iv, c_psi) = rhs_box(iv, c_psi);
            error_box(iv, c_V1) = rhs_box(iv, c_V1);
            error_box(iv, c_V2) = rhs_box(iv, c_V2);
            error_box(iv, c_V3) = rhs_box(iv, c_V3);
            error_box(iv, c_U) = rhs_box(iv, c_U);
            if (!a_params.deactivate_zero_mode)
            {
                error_box(iv, c_V1) += -laplacian_V1;
                error_box(iv, c_V2) += -laplacian_V2;
                error_box(iv, c_V3) += -laplacian_V3;
                error_box(iv, c_U) += -laplacian_U;
            }
        }
    }
} // end set_error

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

        Box unghosted_box = condition_box.box();
        BoxIterator bit(unghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // Calculate the actual value of psi including BH part
            Real psi_reg = multigrid_vars_box(iv, c_psi_reg);
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = psi_reg + psi_bh;
            Real laplacian_psi_reg =
                get_laplacian(iv, multigrid_vars_box, a_dx, c_psi_reg);

            // Get values of Aij
            Tensor<2, Real> Aij_reg;
            set_Aij_reg(Aij_reg, multigrid_vars_box, iv, a_dx, a_params);
            Tensor<2, Real> Aij_bh;
            set_Aij_bh(Aij_bh, loc, a_params);
            // This is \bar  A_ij \bar A^ij
            Real A2_0 = 0.0;
            FOR2(i, j)
            {
                A2_0 += (Aij_reg[i][j] + Aij_bh[i][j]) *
                        (Aij_reg[i][j] + Aij_bh[i][j]);
            }

            // Assign values of useful matter quantities
            Real rho_matter =
                my_rho_function(iv, a_dx, multigrid_vars_box, a_params);
            Tensor<1, Real> Si;
            my_Si_function(Si, iv, a_dx, multigrid_vars_box, a_params);

            if (a_params.regrid_radius > 0)
            {
                Real rr = sqrt(D_TERM(loc[0] * loc[0], +loc[1] * loc[1],
                                      +loc[2] * loc[2]));
                if (rr < a_params.regrid_radius)
                {
                    condition_box(iv, 0) = 1.0;
                }
            }
            else
            {
                // the condition is similar to the rhs but we take abs
                // value of the contributions and add in effect of psi_0 via log
                condition_box(iv, 0) =
                    2.0 * M_PI * a_params.G_Newton * rho_matter +
                    abs(0.125 * A2_0) + log(psi_0) + laplacian_psi_reg +
                    8.0 * M_PI * a_params.G_Newton *
                        (abs(Si[0]) + abs(Si[1]) + abs(Si[2]));
            }
        }
    }
} // end set_regrid_condition

// Add the correction to psi0 after the solver operates
void set_update_psi0(LevelData<FArrayBox> &a_multigrid_vars,
                     LevelData<FArrayBox> &a_dpsi,
                     const Copier &a_exchange_copier,
                     const PoissonParameters &a_params)
{
    // first exchange ghost cells for dpsi so they are filled with the correct
    // values
    a_dpsi.exchange(a_dpsi.interval(), a_exchange_copier);

    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];

        Box ghosted_box = multigrid_vars_box.box();
        BoxIterator bit(ghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();

            // Update constraint variables for the linear step
            multigrid_vars_box(iv, c_psi_reg) += dpsi_box(iv, c_psi);
            if (a_params.deactivate_zero_mode)
            {
                multigrid_vars_box(iv, c_V1_0) += dpsi_box(iv, c_V1);
                multigrid_vars_box(iv, c_V2_0) += dpsi_box(iv, c_V2);
                multigrid_vars_box(iv, c_V3_0) += dpsi_box(iv, c_V3);
                multigrid_vars_box(iv, c_U_0) += dpsi_box(iv, c_U);
            }
            else
            {
                multigrid_vars_box(iv, c_V1_0) = dpsi_box(iv, c_V1);
                multigrid_vars_box(iv, c_V2_0) = dpsi_box(iv, c_V2);
                multigrid_vars_box(iv, c_V3_0) = dpsi_box(iv, c_V3);
                multigrid_vars_box(iv, c_U_0) = dpsi_box(iv, c_U);
            }
        }
    }
}

// The coefficient of the I operator on dpsi
void set_a_coef(LevelData<FArrayBox> &a_aCoef,
                LevelData<FArrayBox> &a_multigrid_vars,
                const PoissonParameters &a_params, const RealVect &a_dx)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_aCoef.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &aCoef_box = a_aCoef[dit()];
        for (int iconstraint = 0; iconstraint < NUM_CONSTRAINT_VARS;
             iconstraint++)
        {
            aCoef_box.setVal(0.0, iconstraint);

            // this prevents small amounts of noise in the sources
            // activating the zero modes - (Garfinkle trick)
            if (a_params.deactivate_zero_mode)
            {
                // Seems to work best to set this relative to the tolerance
                aCoef_box.setVal(-1e4 * a_params.tolerance, iconstraint);
            }
        }

        // add back some non trivial psi for the Aij part
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        Box unghosted_box = aCoef_box.box();
        BoxIterator bit(unghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // Calculate the actual value of psi including BH part
            Real psi_reg = multigrid_vars_box(iv, c_psi_reg);
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real psi_0 = psi_reg + psi_bh;

            // Get values of Aij
            Tensor<2, Real> Aij_reg;
            set_Aij_reg(Aij_reg, multigrid_vars_box, iv, a_dx, a_params);
            Tensor<2, Real> Aij_bh;
            set_Aij_bh(Aij_bh, loc, a_params);
            // This is \bar  A_ij \bar A^ij
            Real A2_0 = 0.0;
            FOR2(i, j)
            {
                A2_0 += (Aij_reg[i][j] + Aij_bh[i][j]) *
                        (Aij_reg[i][j] + Aij_bh[i][j]);
            }

            if (a_params.include_A2)
            {
                aCoef_box(iv, c_psi) += 0.0;
            }
            else
            {
                // add back the terms in A2
                aCoef_box(iv, c_psi) += -0.875 * A2_0 * pow(psi_0, -8.0);
            }
        }
    }
}
// The coefficient of the Laplacian operator, for now set to constant 1
// Note that beta = -1 so this sets the sign
// the rhs source of the Poisson eqn
void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_bCoef.nComp() == NUM_CONSTRAINT_VARS);

    for (DataIterator dit = a_bCoef.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &bCoef_box = a_bCoef[dit()];
        // JCAurre: Loop to set bCoef=1 for all constraint variables
        for (int iconstraint = 0; iconstraint < NUM_CONSTRAINT_VARS;
             iconstraint++)
        {
            bCoef_box.setVal(1.0, iconstraint);
        }
    }
}

// used to set output data for all ADM Vars for GRChombo restart
void set_output_data(LevelData<FArrayBox> &a_grchombo_vars,
                     LevelData<FArrayBox> &a_multigrid_vars,
                     const PoissonParameters &a_params, const RealVect &a_dx)
{
    CH_assert(a_grchombo_vars.nComp() == NUM_GRCHOMBO_VARS);
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

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
        grchombo_vars_box.setVal(1.0, c_h11);
        grchombo_vars_box.setVal(1.0, c_h22);
        grchombo_vars_box.setVal(1.0, c_h33);
        grchombo_vars_box.setVal(1.0, c_lapse);

        // now non constant terms by location
        Box solver_ghosted_box = multigrid_vars_box.box();
        BoxIterator bit(solver_ghosted_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            // work out location on the grid
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // GRChombo conformal factor chi = psi^-4
            Real psi_bh = set_binary_bh_psi(loc, a_params);
            Real chi = pow(multigrid_vars_box(iv, c_psi_reg) + psi_bh, -4.0);
            grchombo_vars_box(iv, c_chi) = chi;
            Real factor = pow(chi, 1.5);

            // Copy phi and Aij across - note this is now \tilde Aij not
            // \bar Aij
            grchombo_vars_box(iv, c_phi) = multigrid_vars_box(iv, c_phi_0);
            grchombo_vars_box(iv, c_Pi) = multigrid_vars_box(iv, c_Pi_0);

            grchombo_vars_box(iv, c_K) = multigrid_vars_box(iv, c_K_0);
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
        }
    }
}
