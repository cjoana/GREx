/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(FLUIDINITDATA_HPP_)
#error "This file should only be included through FluidInitData.hpp"
#endif

#ifndef FLUIDINITDATA_IMPL_HPP_
#define FLUIDINITDATA_IMPL_HPP_

template <class matter_t>
FluidInitData<matter_t>::FluidInitData(matter_t a_matter, double dx,
                                       double omega, double G_Newton)
    : my_matter(a_matter), m_omega(omega), m_G_Newton(G_Newton),
      m_deriv(dx)
{
}

template <class matter_t>
template <class data_t>
void FluidInitData<matter_t>::compute(Cell<data_t> current_cell) const
{

    // copy data from chombo gridpoint into local variable and calculate derivs
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);
    const auto advec =
        m_deriv.template advection<Vars>(current_cell, vars.shift);

    auto up_vars = current_cell.template load_vars<Vars>();

    rhs_equation(up_vars, vars, d1, d2, advec);

    current_cell.store_vars(up_vars.E, c_E);
}

template <class matter_t>
template <class data_t>
void FluidInitData<matter_t>::rhs_equation(
    Vars<data_t> &up_vars, const Vars<data_t> &vars,
    const Vars<Tensor<1, data_t>> &d1, const Diff2Vars<Tensor<2, data_t>> &d2,
    const Vars<data_t> &advec) const
{

    using namespace TensorAlgebra;

    const auto h_UU = compute_inverse_sym(vars.h);
    const auto chris = compute_christoffel(d1.h, h_UU);

    // Calculate elements of the decomposed stress energy tensor and ricci
    // tensor
    const auto emtensor = my_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);
    const auto ricci = CCZ4Geometry::compute_ricci(vars, d1, d2, h_UU, chris);
    const auto A_UU = raise_all(vars.A, h_UU);
    const data_t tr_AA = compute_trace(vars.A, A_UU);

    data_t rho = ( (GR_SPACEDIM - 1.)*vars.K*vars.K/GR_SPACEDIM
            - tr_AA  + ricci.scalar ) / (16.0 * M_PI * m_G_Newton);

    rho +=  - vars.D;

    up_vars.E = rho;

}

#endif /* FLUIDINITDATA_IMPL_HPP_ */
