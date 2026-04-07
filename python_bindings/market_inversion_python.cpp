#include "market_inversion.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(python_pcm_inversion, m) {
    m.doc() = "pybind11 pcm_inversion plugin";

    py::enum_<share_inversion::InversionMethod>(m, "InversionMethod")
        .value("NLOPT_SLSQP",       share_inversion::InversionMethod::NLOPT_SLSQP)
        .value("CONE_AWARE_NEWTON", share_inversion::InversionMethod::CONE_AWARE_NEWTON)
        .export_values();

    m.def(
        "invert_shares",
        [](Eigen::MatrixXd& x, Eigen::ArrayXd& p, double sigma_p,
           Eigen::ArrayXd& sigma_x, Eigen::ArrayXXd& grid,
           Eigen::ArrayXd& weights, Eigen::ArrayXd& data_shares,
           double share_equality_tolerances,
           double delta_step_tolerance,
           unsigned max_number_of_function_calls,
           share_inversion::InversionMethod method)
        {
            share_inversion::pcm_parameters param(
                x, p, sigma_p, sigma_x, grid, weights, data_shares,
                std::vector<double>(),
                nlopt::LD_SLSQP,
                share_equality_tolerances,
                delta_step_tolerance,
                max_number_of_function_calls);
            return share_inversion::invert_shares(param, method);
        },
        "Inverts PCM market shares to recover delta (vertical qualities). "
        "Returns delta such that unc_share(delta, ...) == data_shares.",
        py::arg("x"),
        py::arg("p"),
        py::arg("sigma_p"),
        py::arg("sigma_x"),
        py::arg("grid"),
        py::arg("weights"),
        py::arg("data_shares"),
        py::arg("share_equality_tolerances")    = 1e-8,
        py::arg("delta_step_tolerance")         = 1e-5,
        py::arg("max_number_of_function_calls") = 1000,
        py::arg("method")                       = share_inversion::InversionMethod::NLOPT_SLSQP
    );
}
