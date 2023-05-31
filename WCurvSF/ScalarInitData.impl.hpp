/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#if !defined(SCALARINITDATA_HPP_)
#error "This file should only be included through ScalarInitData.hpp"
#endif

#ifndef SCALARINITDATA_IMPL_HPP_
#define SCALARINITDATA_IMPL_HPP_

template <class matter_t>
ScalarInitData<matter_t>::ScalarInitData(matter_t a_matter, double dx,
                                       double K_mean, double rho_mean, double G_Newton)
    : my_matter(a_matter), m_K_mean(K_mean), m_rho_mean(rho_mean), m_G_Newton(G_Newton),
      m_deriv(dx)
{
}

template <class matter_t>
template <class data_t>
void ScalarInitData<matter_t>::compute(Cell<data_t> current_cell) const
{

    // copy data from chombo gridpoint into local variable and calculate derivs
    const auto vars = current_cell.template load_vars<Vars>();
    const auto d1 = m_deriv.template diff1<Vars>(current_cell);
    const auto d2 = m_deriv.template diff2<Diff2Vars>(current_cell);
    const auto advec =
        m_deriv.template advection<Vars>(current_cell, vars.shift);

    auto up_vars = current_cell.template load_vars<Vars>();

    up_vars.K = m_K_mean;

    rhs_equation(up_vars, vars, d1, d2, advec);

    current_cell.store_vars(up_vars.Pi, c_Pi);
    current_cell.store_vars(up_vars.K, c_K);

    //current_cell.store_vars(5.0, c_phi);
}

template <class matter_t>
template <class data_t>
void ScalarInitData<matter_t>::rhs_equation(
    Vars<data_t> &up_vars, const Vars<data_t> &vars,
    const Vars<Tensor<1, data_t>> &d1, const Diff2Vars<Tensor<2, data_t>> &d2,
    const Vars<data_t> &advec) const
{

    using namespace TensorAlgebra;

    const auto h_UU = compute_inverse_sym(vars.h);
    const auto chris = compute_christoffel(d1.h, h_UU);

    // Calculate elements of the decomposed stress energy tensor and ricci
    // tensor
    //const auto emtensor = my_matter.compute_emtensor(vars, d1, h_UU, chris.ULL);
    const auto ricci = CCZ4Geometry::compute_ricci(vars, d1, d2, h_UU, chris);
    // const auto A_UU = raise_all(vars.A, h_UU);
    // const data_t tr_AA = compute_trace(vars.A, A_UU);

    data_t Pi2 = ( (GR_SPACEDIM - 1.)*up_vars.K*up_vars.K/GR_SPACEDIM  
            // - tr_AA
            + ricci.scalar ) / (16.0 * M_PI * m_G_Newton) - m_rho_mean;  //  m_rho_mean gives currently the Potential, only. 


    data_t Pi = - sqrt(2*Pi2)  ;   

    // std::cout << "Pi2 =  " <<  Pi2  << "  Pi "<<  Pi << endl;

    up_vars.Pi = Pi;

}

#endif /* SCALARINITDATA_IMPL_HPP_ */
