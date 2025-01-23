#include "pcm_market_share.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;



PYBIND11_MODULE(python_market_share, m) {
    m.doc() = "pybind11 pcm_market_share plugin"; // optional module docstring

    // m.def("unc_share", &pcm_share::unc_share, "A function that computes unconditional market share of each product");
    // m.def("unc_share", &pcm_share::unc_share, "A function that computes unconditional market share of each product without computing jacobian");
    // m.def("cond_share", &pcm_share::cond_share, "A function that computes market share of vertical model conditional on draw of heterogeneity");
    // m.def("cond_share", &pcm_share::cond_share, "A function that computes market share of vertical model conditional on draw of heterogeneity without jacobian");
    m.def(
        "initial_guess"
        , &pcm_share::initial_guess
        , "A function that computes initial guess for delta based on the data",
        py::arg("shares_data"), py::arg("p"), py::arg("sigma_p")
    );
    m.def(
        "conditional_share_no_jacobian"
        , [](const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, bool check_positive_shares){
            return pcm_share::cond_share(delta, p, sigma_p, check_positive_shares);
        }
        , "A function that computes market share of vertical model conditional on draw of heterogeneity without jacobian. Returns an array of market shares conditional on delta, price and sigma_p",
        py::arg("delta"), py::arg("p"), py::arg("sigma_p"), py::arg("check_positive_shares") = true
    );
    m.def(
        "conditional_share_with_jacobian"
        , [](const Eigen::ArrayXd& delta, const Eigen::ArrayXd& p, double sigma_p, bool check_positive_shares){
            Eigen::MatrixXd jacobian;
            auto shares = pcm_share::cond_share(delta, p, sigma_p, jacobian, check_positive_shares);
            return std::make_pair(shares, jacobian);
        }
        , "A function that computes market share of vertical model conditional on draw of heterogeneity. Returns pair of values: the first is an array of market shares, the second is the jacobian of the vector of market shares derivatives with respect to each vertical quality",
        py::arg("delta"), py::arg("p"), py::arg("sigma_p"), py::arg("check_positive_shares") = true
    );
    m.def(
        "unconditional_share_no_jacobian"
        , [](const Eigen::ArrayXd& delta_bar, const Eigen::MatrixXd& x, const Eigen::ArrayXd& p, double sigma_p, const Eigen::ArrayXd& sigma_x, const Eigen::ArrayXXd& grid, const Eigen::ArrayXd & weights){
            return pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights);
        }
        , "A function that computes unconditional market share of each product without computing jacobian. Returns an array of market shares conditional on delta_bar, x, p, sigma_p, sigma_x, grid and weights",
        py::arg("delta_bar"), py::arg("x"), py::arg("p"), py::arg("sigma_p"), py::arg("sigma_x"), py::arg("grid"), py::arg("weights")
    );
    m.def(
        "unconditional_share_with_jacobian"
        , [](const Eigen::ArrayXd& delta_bar, const Eigen::MatrixXd& x, const Eigen::ArrayXd& p, double sigma_p, const Eigen::ArrayXd& sigma_x, const Eigen::ArrayXXd& grid, const Eigen::ArrayXd & weights){
            Eigen::MatrixXd jacobian;
            auto shares = pcm_share::unc_share(delta_bar, x, p, sigma_p, sigma_x, grid, weights, jacobian);
            return std::make_pair(shares, jacobian);
        }
        , "A function that computes unconditional market share of each product. Returns pair of values: the first is an array of market shares, the second is the jacobian of the vector of unconditional market shares derivatives with respect to each average vertical quality",
        py::arg("delta_bar"), py::arg("x"), py::arg("p"), py::arg("sigma_p"), py::arg("sigma_x"), py::arg("grid"), py::arg("weights")
    );
}