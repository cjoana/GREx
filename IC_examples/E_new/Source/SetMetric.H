/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SETMETRICBH_HPP
#define SETMETRICBH_HPP

#include "PoissonParameters.H"
#include "SetLevelDataCpp.H"

void set_ricci(FArrayBox &multigrid_vars_box, const IntVect &iv,
               const RealVect &loc, const RealVect &a_dx,
               const PoissonParameters &a_params,
               const FArrayBox &a_grad_multigrid,
               const FArrayBox &a_grad2_multigrid,
               const FArrayBox &a_mixed_grad2_multigrid,
               const FArrayBox &a_grad_h_UU)
{

    Real h[3][3];
    h[0][0] = multigrid_vars_box(iv, c_h11_0);
    h[1][1] = multigrid_vars_box(iv, c_h22_0);
    h[2][2] = multigrid_vars_box(iv, c_h33_0);
    h[0][1] = multigrid_vars_box(iv, c_h12_0);
    h[0][2] = multigrid_vars_box(iv, c_h13_0);
    h[1][2] = multigrid_vars_box(iv, c_h23_0);
    h[1][0] = h[0][1];
    h[2][0] = h[0][2];
    h[2][1] = h[1][2];

    Real det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
               h[0][1] * (h[2][2] * h[1][0] - h[1][2] * h[2][0]) +
               h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);

    Real h_UU[3][3] = {0};
    Real det_inverse = 1. / det;
    h_UU[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * det_inverse;
    h_UU[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * det_inverse;
    h_UU[2][2] = (h[0][0] * h[1][1] - h[1][0] * h[0][1]) * det_inverse;
    h_UU[0][1] = (h[2][0] * h[1][2] - h[1][0] * h[2][2]) * det_inverse;
    h_UU[1][0] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * det_inverse;
    h_UU[0][2] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * det_inverse;
    h_UU[2][0] = (h[0][1] * h[1][2] - h[1][1] * h[0][2]) * det_inverse;
    h_UU[1][2] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * det_inverse;
    h_UU[2][1] = (h[1][0] * h[0][2] - h[0][0] * h[1][2]) * det_inverse;

    Real d1_h[3][3][3] = {0};
    Real d1_hUU[3][3][3] = {0};
    Real d2_h[3][3][3][3] = {0};

    for (int i = 0; i < SpaceDim; i++)
    {
        d1_h[0][0][i] = a_grad_multigrid(iv, 3 * c_h11_0 + i);
        d1_h[1][1][i] = a_grad_multigrid(iv, 3 * c_h22_0 + i);
        d1_h[2][2][i] = a_grad_multigrid(iv, 3 * c_h33_0 + i);
        d1_h[0][1][i] = a_grad_multigrid(iv, 3 * c_h12_0 + i);
        d1_h[0][2][i] = a_grad_multigrid(iv, 3 * c_h13_0 + i);
        d1_h[1][2][i] = a_grad_multigrid(iv, 3 * c_h23_0 + i);
        d1_h[1][0][i] = d1_h[0][1][i];
        d1_h[2][0][i] = d1_h[0][2][i];
        d1_h[2][1][i] = d1_h[1][2][i];

        d2_h[0][0][i][i] = a_grad2_multigrid(iv, 3 * c_h11_0 + i);
        d2_h[1][1][i][i] = a_grad2_multigrid(iv, 3 * c_h22_0 + i);
        d2_h[2][2][i][i] = a_grad2_multigrid(iv, 3 * c_h33_0 + i);
        d2_h[0][1][i][i] = a_grad2_multigrid(iv, 3 * c_h12_0 + i);
        d2_h[0][2][i][i] = a_grad2_multigrid(iv, 3 * c_h13_0 + i);
        d2_h[1][2][i][i] = a_grad2_multigrid(iv, 3 * c_h23_0 + i);
        d2_h[1][0][i][i] = d2_h[0][1][i][i];
        d2_h[2][0][i][i] = d2_h[0][2][i][i];
        d2_h[2][1][i][i] = d2_h[1][2][i][i];

        d1_hUU[0][0][i] = a_grad_h_UU(iv, 3 * 0 + i);
        d1_hUU[1][1][i] = a_grad_h_UU(iv, 3 * 1 + i);
        d1_hUU[2][2][i] = a_grad_h_UU(iv, 3 * 2 + i);
        d1_hUU[0][1][i] = a_grad_h_UU(iv, 3 * 3 + i);
        d1_hUU[0][2][i] = a_grad_h_UU(iv, 3 * 4 + i);
        d1_hUU[1][2][i] = a_grad_h_UU(iv, 3 * 5 + i);
        d1_hUU[1][0][i] = d1_hUU[0][1][i];
        d1_hUU[2][0][i] = d1_hUU[0][2][i];
        d1_hUU[2][1][i] = d1_hUU[1][2][i];

        for (int j = i + 1; j < SpaceDim; j++)
        {
            d2_h[0][0][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h11_0 + i + j - 1);
            d2_h[1][1][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h22_0 + i + j - 1);
            d2_h[2][2][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h33_0 + i + j - 1);
            d2_h[0][1][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h12_0 + i + j - 1);
            d2_h[0][2][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h13_0 + i + j - 1);
            d2_h[1][2][i][j] =
                a_mixed_grad2_multigrid(iv, 3 * c_h23_0 + i + j - 1);
            d2_h[1][0][i][j] = d2_h[0][1][i][j];
            d2_h[2][0][i][j] = d2_h[0][2][i][j];
            d2_h[2][1][i][j] = d2_h[1][2][i][j];

            d2_h[0][0][j][i] = d2_h[0][0][i][j];
            d2_h[1][1][j][i] = d2_h[1][1][i][j];
            d2_h[2][2][j][i] = d2_h[2][2][i][j];
            d2_h[0][1][j][i] = d2_h[0][1][i][j];
            d2_h[0][2][j][i] = d2_h[0][2][i][j];
            d2_h[1][2][j][i] = d2_h[1][2][i][j];
            d2_h[1][0][j][i] = d2_h[1][0][i][j];
            d2_h[2][0][j][i] = d2_h[2][0][i][j];
            d2_h[2][1][j][i] = d2_h[2][1][i][j];
        }
    }

    Real chris[3][3][3] = {0};
    Real d1_chris[3][3][3][3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                for (int l = 0; l < SpaceDim; l++)
                {
                    chris[i][j][k] +=
                        0.5 * h_UU[k][l] *
                        (d1_h[i][l][j] + d1_h[j][l][i] - d1_h[i][j][l]);

                    for (int m = 0; m < SpaceDim; m++)
                    {
                        d1_chris[i][j][k][m] +=
                            0.5 * d1_hUU[k][l][m] *
                                (d1_h[i][l][j] + d1_h[j][l][i] -
                                 d1_h[i][j][l]) +
                            0.5 * h_UU[k][l] *
                                (d2_h[i][l][j][m] + d2_h[j][l][i][m] -
                                 d2_h[i][j][l][m]);
                    }
                }
            }
        }
    }

    Real riemann_tensor[3][3][3][3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                for (int l = 0; l < SpaceDim; l++)
                {
                    riemann_tensor[i][j][k][l] +=
                        d1_chris[j][l][i][k] - d1_chris[j][k][i][l];

                        for (int m = 0; m < SpaceDim; m++)
                    {
                        riemann_tensor[i][j][k][l] +=
                            chris[m][k][i] * chris[j][l][m] -
                            chris[m][l][i] * chris[j][k][m];
                    }
                }
            }
        }
    }

    Real ricci_tensor[3][3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                ricci_tensor[i][j] += riemann_tensor[k][i][k][j];
            }
        }
    }

    Real ricci_tensor2[3][3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                for (int l = 0; l < SpaceDim; l++)
                {
                    ricci_tensor2[i][j] +=
                        0.5 * d1_hUU[k][l][k] *
                            (d1_h[j][l][i] + d1_h[i][l][j] - d1_h[i][j][l]) +
                        0.5 * h_UU[k][l] *
                            (d2_h[i][l][j][k] + d2_h[j][l][i][k] -
                             d2_h[i][j][k][l] - d2_h[k][l][i][j]) -
                        0.5 * d1_hUU[k][l][j] * d1_h[k][l][i];

                    for (int m = 0; m < SpaceDim; m++)
                    {
                        for (int n = 0; n < SpaceDim; n++)
                        {
                            ricci_tensor2[i][j] +=
                                0.25 * h_UU[m][n] * h_UU[k][l] *
                                (d1_h[k][l][n] *
                                     (d1_h[i][m][j] + d1_h[j][m][i] -
                                      d1_h[i][j][m]) -
                                 d1_h[m][k][i] * d1_h[n][l][j] -
                                 2 * d1_h[i][k][m] * d1_h[j][n][l] +
                                 2 * d1_h[i][k][m] * d1_h[j][l][n]);
                        }
                    }
                }
            }
        }
    }

    Real out_ricci = 0;
    Real out_ricci2 = 0;
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            out_ricci += h_UU[i][j] * ricci_tensor[i][j];
            out_ricci2 += h_UU[i][j] * ricci_tensor2[i][j];
        }
    }

    // pout() << "difference: " << out_ricci - out_ricci2 << endl;
    multigrid_vars_box(iv, c_R_0) = out_ricci;
    // multigrid_vars_box(iv, c_R_0_2) = out_ricci2;
}

void set_christoffel(FArrayBox &multigrid_vars_box, const IntVect &iv,
                     const RealVect &loc, const RealVect &a_dx,
                     const PoissonParameters &a_params,
                     const FArrayBox &a_grad_multigrid)
{
    Real ricci_tensor[3][3] = {0};
    Real h[3][3];
    h[0][0] = multigrid_vars_box(iv, c_h11_0);
    h[1][1] = multigrid_vars_box(iv, c_h22_0);
    h[2][2] = multigrid_vars_box(iv, c_h33_0);
    h[0][1] = multigrid_vars_box(iv, c_h12_0);
    h[0][2] = multigrid_vars_box(iv, c_h13_0);
    h[1][2] = multigrid_vars_box(iv, c_h23_0);
    h[1][0] = h[0][1];
    h[2][0] = h[0][2];
    h[2][1] = h[1][2];

    Real det = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1]) -
               h[0][1] * (h[2][2] * h[1][0] - h[1][2] * h[2][0]) +
               h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);

    Real h_UU[3][3] = {0};
    Real det_inverse = 1. / det;
    h_UU[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * det_inverse;
    h_UU[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * det_inverse;
    h_UU[2][2] = (h[0][0] * h[1][1] - h[1][0] * h[0][1]) * det_inverse;
    h_UU[0][1] = (h[2][0] * h[1][2] - h[1][0] * h[2][2]) * det_inverse;
    h_UU[1][0] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * det_inverse;
    h_UU[0][2] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * det_inverse;
    h_UU[2][0] = (h[0][1] * h[1][2] - h[1][1] * h[0][2]) * det_inverse;
    h_UU[1][2] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * det_inverse;
    h_UU[2][1] = (h[1][0] * h[0][2] - h[0][0] * h[1][2]) * det_inverse;

    Real d1_h[3][3][3] = {0};
    Real d1_hUU[3][3][3] = {0};
    Real d2_h[3][3][3][3] = {0};

    for (int i = 0; i < SpaceDim; i++)
    {
        d1_h[0][0][i] = a_grad_multigrid(iv, 3 * c_h11_0 + i);
        d1_h[1][1][i] = a_grad_multigrid(iv, 3 * c_h22_0 + i);
        d1_h[2][2][i] = a_grad_multigrid(iv, 3 * c_h33_0 + i);
        d1_h[0][1][i] = a_grad_multigrid(iv, 3 * c_h12_0 + i);
        d1_h[0][2][i] = a_grad_multigrid(iv, 3 * c_h13_0 + i);
        d1_h[1][2][i] = a_grad_multigrid(iv, 3 * c_h23_0 + i);
        d1_h[1][0][i] = d1_h[0][1][i];
        d1_h[2][0][i] = d1_h[0][2][i];
        d1_h[2][1][i] = d1_h[1][2][i];
    }

    Real chris[3][3][3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                for (int l = 0; l < SpaceDim; l++)
                {
                    chris[i][j][k] +=
                        0.5 * h_UU[k][l] *
                        (d1_h[i][l][j] + d1_h[j][l][i] - d1_h[i][j][l]);
                }
            }
        }
    }

    Real Gamma[3] = {0};
    for (int i = 0; i < SpaceDim; i++)
    {
        for (int j = 0; j < SpaceDim; j++)
        {
            for (int k = 0; k < SpaceDim; k++)
            {
                Gamma[k] += h_UU[i][j] * chris[i][j][k];
            }
        }
    }

    multigrid_vars_box(iv, c_Gamma1_0) = Gamma[0];
    multigrid_vars_box(iv, c_Gamma2_0) = Gamma[1];
    multigrid_vars_box(iv, c_Gamma3_0) = Gamma[2];
}

Real calculate_metric_components(LevelData<FArrayBox> &a_multigrid_vars,
                                 const RealVect &a_dx,
                                 const PoissonParameters &a_params)
{

    DataIterator dit = a_multigrid_vars.dataIterator();
    const DisjointBoxLayout &grids =
        a_multigrid_vars.disjointBoxLayout(); // no ghosts

    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        Box b = multigrid_vars_box.box();
        Box b_no_ghosts = grids[dit()];

        FArrayBox grad_multigrid(b_no_ghosts, 3 * NUM_MULTIGRID_VARS);
        get_grad(b_no_ghosts, multigrid_vars_box, Interval(c_psi_0, c_h33_0),
                 a_dx, grad_multigrid, a_params);

        FArrayBox grad2_multigrid(b_no_ghosts, 3 * NUM_MULTIGRID_VARS);
        get_grad2(b_no_ghosts, multigrid_vars_box, Interval(c_h11_0, c_h33_0),
                  a_dx, grad2_multigrid, a_params);

        FArrayBox mixed_grad2_multigrid(b_no_ghosts, 3 * NUM_MULTIGRID_VARS);
        get_mixed_grad2(b_no_ghosts, multigrid_vars_box,
                        Interval(c_h11_0, c_h33_0), a_dx, mixed_grad2_multigrid,
                        a_params);

        FArrayBox h_UU(b, 6);
        get_inverse(b, multigrid_vars_box, Interval(c_h11_0, c_h33_0), a_dx,
                    h_UU, a_params);

        FArrayBox grad_h_UU(b_no_ghosts, 3 * 6);
        get_grad(b_no_ghosts, h_UU, Interval(0, 5), a_dx, grad_h_UU, a_params);

        BoxIterator bit_no_ghosts(b_no_ghosts);
        for (bit_no_ghosts.begin(); bit_no_ghosts.ok(); ++bit_no_ghosts)
        {

            // work out location on the grid
            IntVect iv = bit_no_ghosts();

            // set the phi value - need the distance from centre
            RealVect loc(iv + 0.5 * RealVect::Unit);
            loc *= a_dx;
            loc -= a_params.domainLength / 2.0;

            // JCAurre: set Ricci
            set_ricci(multigrid_vars_box, iv, loc, a_dx, a_params,
                      grad_multigrid, grad2_multigrid, mixed_grad2_multigrid,
                      grad_h_UU);

            set_christoffel(multigrid_vars_box, iv, loc, a_dx, a_params,
                            grad_multigrid);
        }
    }
}
#endif /* SETMETRICBH_HPP */
