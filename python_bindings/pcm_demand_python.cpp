#include "pcm_demand.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {
pcm_demand::Market make_market(const Eigen::MatrixXd & x, const Eigen::ArrayXd & p, const Eigen::ArrayXd & w,
                               const Eigen::MatrixXd & taste_basis, const std::vector<int> & taste_col,
                               const Eigen::MatrixXd & scale_basis) {
    return pcm_demand::Market{x, p, w, taste_basis, taste_col, scale_basis};
}
}

PYBIND11_MODULE(python_pcm_demand, m) {
    m.doc() = "Pure characteristics demand with demographics: shares, derivatives, inversion";

    py::class_<pcm_demand::Market>(m, "Market")
        .def(py::init(&make_market), py::arg("x"), py::arg("p"), py::arg("weights"),
             py::arg("taste_basis"), py::arg("taste_col"), py::arg("scale_basis"))
        .def_readonly("x", &pcm_demand::Market::x)
        .def_readonly("p", &pcm_demand::Market::p);

    m.def("shares",
          [](const Eigen::ArrayXd & delta, const pcm_demand::Market & mk, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, bool with_derivatives) {
              pcm_demand::Params prm{theta, phi, sigma_p};
              pcm_demand::Shares sh;
              {
                  py::gil_scoped_release release;
                  sh = pcm_demand::shares(delta, mk, prm, with_derivatives);
              }
              py::dict d;
              d["s"] = sh.s;
              if (with_derivatives) {
                  d["d_delta"] = sh.d_delta;
                  d["d_theta"] = sh.d_theta;
                  d["d_price"] = sh.d_price;
              }
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("with_derivatives") = true,
          "Shares and derivatives. d_theta columns are [theta..., phi..., sigma_p].");

    m.def("invert",
          [](const Eigen::ArrayXd & s_data, const pcm_demand::Market & mk, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const Eigen::ArrayXd & delta0, double tol,
             unsigned max_evals) {
              pcm_demand::Params prm{theta, phi, sigma_p};
              pcm_demand::InversionResult r;
              {
                  py::gil_scoped_release release;
                  r = pcm_demand::invert(s_data, mk, prm, delta0, tol, max_evals);
              }
              py::dict d;
              d["delta"] = r.delta;
              d["max_abs_residual"] = r.max_abs_residual;
              d["iterations"] = r.iterations;
              d["share_evaluations"] = r.share_evaluations;
              d["converged"] = r.converged;
              return d;
          },
          py::arg("s_data"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("delta0") = Eigen::ArrayXd(), py::arg("tol") = 1e-12, py::arg("max_evals") = 5000);
}
