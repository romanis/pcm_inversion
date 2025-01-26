#include "market_inversion.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(python_pcm_inversion, m) {
    m.doc() = "pybind11 pcm_inversion plugin"; // optional module docstring

    m.def(
        "invert_shares"
        , [](Eigen::MatrixXd& x, Eigen::ArrayXd &p, double sigma_p, Eigen::ArrayXd &sigma_x, Eigen::ArrayXXd& grid, 
                       Eigen::ArrayXd& weights, Eigen::ArrayXd& data_shares, 
                       double share_equality_tolerances = 1e-8, double delta_step_tolerance = 1e-5, unsigned max_number_of_function_calls = 1000){
            share_inversion::pcm_parameters param(x, p, sigma_p, sigma_x, grid, weights, data_shares, std::vector<double>(), nlopt::LD_SLSQP, share_equality_tolerances, delta_step_tolerance, max_number_of_function_calls);
            return share_inversion::invert_shares(param);
        }
        , "A function that computes the delta that minimizes the difference between the shares computed from the data and the shares computed from the delta. Returns an array of delta that minimizes the difference between the shares computed from the data and the shares computed from the delta"
        , py::arg("x"), py::arg("p"), py::arg("sigma_p"), py::arg("sigma_x"), py::arg("grid"), py::arg("weights")
        , py::arg("data_shares")
        , py::arg("share_equality_tolerances") = 1e-8, py::arg("delta_step_tolerance") = 1e-5, py::arg("max_number_of_function_calls") = 1000
    );
}