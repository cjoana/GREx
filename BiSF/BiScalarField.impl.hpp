/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(BISCALARFIELD_HPP_)
#error "This file should only be included through BiScalarField.hpp"
#endif

#ifndef BISCALARFIELD_IMPL_HPP_
#define BISCALARFIELD_IMPL_HPP_
#include "DimensionDefinitions.hpp"

// Calculate the stress energy tensor elements
template <class potential_t>
template <class data_t, template <typename> class vars_t>
emtensor_t<data_t> ScalarField<potential_t>::compute_emtensor(
    const vars_t<data_t> &vars, const vars_t<Tensor<1, data_t>> &d1,
    const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL) const
{
    emtensor_t<data_t> out;

    // Copy the field vars into SFObject
    SFObject<data_t> vars_sf;
    vars_sf.phi = vars.phi;
    vars_sf.Pi = vars.Pi;

    vars_sf.phi2 = vars.phi2;
    vars_sf.Pi2 = vars.Pi2;

    // call the function which computes the em tensor excluding the potential
    emtensor_excl_potential(out, vars, vars_sf, d1.phi, d1.phi2, h_UU, chris_ULL);

    // set the potential values
    data_t V_of_phi = 0.0;
    data_t dVdphi = 0.0;

    data_t V_of_phi2 = 0.0;
    data_t dVdphi2 = 0.0;


    // compute potential and add constributions to EM Tensor
    my_potential.compute_potential(V_of_phi, dVdphi, V_of_phi2, dVdphi2, vars);

    out.rho += V_of_phi;
    out.S += -3.0 * V_of_phi;
    FOR2(i, j) { out.Sij[i][j] += -vars.h[i][j] * V_of_phi / vars.chi;}

    return out;
}

// Calculate the stress energy tensor elements
template <class potential_t>
template <class data_t, template <typename> class vars_t>
void ScalarField<potential_t>::emtensor_excl_potential(
    emtensor_t<data_t> &out, const vars_t<data_t> &vars,
    const SFObject<data_t> &vars_sf, const Tensor<1, data_t> &d1_phi,
    const Tensor<1, data_t> &d1_phi2,
    const Tensor<2, data_t> &h_UU, const Tensor<3, data_t> &chris_ULL)
{
    // Useful quantity Vt
    data_t Vt = -vars_sf.Pi * vars_sf.Pi;
    FOR2(i, j) { Vt += vars.chi * h_UU[i][j] * d1_phi[i] * d1_phi[j]; }

    data_t Vt2 = -vars_sf.Pi2 * vars_sf.Pi2;
    FOR2(i, j) { Vt2 += vars.chi * h_UU[i][j] * d1_phi2[i] * d1_phi2[j]; }

    // Calculate components of EM Tensor
    // S_ij = T_ij
    FOR2(i, j)
    {
        out.Sij[i][j] =
            -0.5 * vars.h[i][j] * Vt / vars.chi + d1_phi[i] * d1_phi[j]
            -0.5 * vars.h[i][j] * Vt2 / vars.chi + d1_phi2[i] * d1_phi2[j];
    }

    // S = Tr_S_ij
    out.S = vars.chi * TensorAlgebra::compute_trace(out.Sij, h_UU);

    // S_i (note lower index) = - n^a T_ai
    FOR1(i) { out.Si[i] = -d1_phi[i] * vars_sf.Pi - d1_phi2[i] * vars_sf.Pi2; }

    // rho = n^a n^b T_ab
    out.rho = vars_sf.Pi * vars_sf.Pi + 0.5 * Vt +
              vars_sf.Pi2 * vars_sf.Pi2 + 0.5 * Vt2;
}

// Adds in the RHS for the matter vars
template <class potential_t>
template <class data_t, template <typename> class vars_t,
          template <typename> class diff2_vars_t,
          template <typename> class rhs_vars_t>
void ScalarField<potential_t>::add_matter_rhs(
    rhs_vars_t<data_t> &total_rhs, const vars_t<data_t> &vars,
    const vars_t<Tensor<1, data_t>> &d1,
    const diff2_vars_t<Tensor<2, data_t>> &d2,
    const vars_t<data_t> &advec) const
{
    // first get the non potential part of the rhs
    // this may seem a bit long winded, but it makes the function
    // work for more multiple fields

    SFObject<data_t> rhs_sf;
    // advection terms
    SFObject<data_t> advec_sf;
    advec_sf.phi = advec.phi;
    advec_sf.Pi = advec.Pi;
    advec_sf.phi2 = advec.phi2;
    advec_sf.Pi2 = advec.Pi2;
    // the vars
    SFObject<data_t> vars_sf;
    vars_sf.phi = vars.phi;
    vars_sf.Pi = vars.Pi;
    vars_sf.phi2 = vars.phi2;
    vars_sf.Pi2 = vars.Pi2;

    // call the function for the rhs excluding the potential
    matter_rhs_excl_potential(rhs_sf, vars, vars_sf, d1, d1.phi, d2.phi,
                              d1.phi2, d2.phi2,
                              advec_sf);

    // set the potential values
    data_t V_of_phi = 0.0;
    data_t dVdphi = 0.0;

    data_t V_of_phi2 = 0.0;
    data_t dVdphi2 = 0.0;

    // compute potential
    my_potential.compute_potential(V_of_phi, dVdphi, V_of_phi2, dVdphi2, vars);

    // adjust RHS for the potential term
    total_rhs.phi = rhs_sf.phi;
    total_rhs.Pi = rhs_sf.Pi - vars.lapse * dVdphi;

    total_rhs.phi2 = rhs_sf.phi2;
    total_rhs.Pi2 = rhs_sf.Pi2 - vars.lapse * dVdphi2;
}

// the RHS excluding the potential terms
template <class potential_t>
template <class data_t, template <typename> class vars_t>
void ScalarField<potential_t>::matter_rhs_excl_potential(
    SFObject<data_t> &rhs_sf, const vars_t<data_t> &vars,
    const SFObject<data_t> &vars_sf, const vars_t<Tensor<1, data_t>> &d1,
    const Tensor<1, data_t> &d1_phi, const Tensor<2, data_t> &d2_phi,
    const Tensor<1, data_t> &d1_phi2, const Tensor<2, data_t> &d2_phi2,
    const SFObject<data_t> &advec_sf)
{
    using namespace TensorAlgebra;

    const auto h_UU = compute_inverse_sym(vars.h);
    const auto chris = compute_christoffel(d1.h, h_UU);

    // evolution equations for scalar field and (minus) its conjugate momentum
    rhs_sf.phi = vars.lapse * vars_sf.Pi + advec_sf.phi;
    rhs_sf.Pi = vars.lapse * vars.K * vars_sf.Pi + advec_sf.Pi;

    rhs_sf.phi2 = vars.lapse * vars_sf.Pi2 + advec_sf.phi2;
    rhs_sf.Pi2 = vars.lapse * vars.K * vars_sf.Pi2 + advec_sf.Pi2;

    FOR2(i, j)
    {
        // includes non conformal parts of chris not included in chris_ULL
        rhs_sf.Pi += h_UU[i][j] * (-0.5 * d1.chi[j] * vars.lapse * d1_phi[i] +
                                   vars.chi * vars.lapse * d2_phi[i][j] +
                                   vars.chi * d1.lapse[i] * d1_phi[j]);

        rhs_sf.Pi2 += h_UU[i][j] * (-0.5 * d1.chi[j] * vars.lapse * d1_phi2[i] +
                                  vars.chi * vars.lapse * d2_phi2[i][j] +
                                  vars.chi * d1.lapse[i] * d1_phi2[j]);

        FOR1(k)
        {
            rhs_sf.Pi += -vars.chi * vars.lapse * h_UU[i][j] *
                         chris.ULL[k][i][j] * d1_phi[k];

            rhs_sf.Pi2 += -vars.chi * vars.lapse * h_UU[i][j] *
                         chris.ULL[k][i][j] * d1_phi2[k];
        }
    }
}

#endif /* BISCALARFIELD_IMPL_HPP_ */
