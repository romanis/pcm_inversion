#include "pcm_second_choice.hpp"
#include "bind_params.hpp"
#include <omp.h>
#include <memory>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_second_choice(py::module_ & m) {
    m.def("second_choice",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const py::object & adjoint, int n_threads, double alpha_lo,
             double a_lo) {
              if (!py::isinstance<pcm_aggregate::Market>(market))
                  throw py::type_error("second_choice: market must be a pcm_kernel.Market");
              const pcm_aggregate::Market mk = market.cast<pcm_aggregate::Market>();
              const pcm_aggregate::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "second_choice");
              if (n_threads < 1) throw py::value_error("second_choice: n_threads must be at least 1");
              std::unique_ptr<pcm_kernel::SecondChoiceAdjoint> adj;
              if (!adjoint.is_none()) {
                  adj = std::make_unique<pcm_kernel::SecondChoiceAdjoint>();
                  adj->A = adjoint["A"].cast<std::vector<Eigen::MatrixXd>>();
                  adj->B = adjoint["B"].cast<Eigen::MatrixXd>();
              }
              pcm_kernel::SecondChoice r;
              std::string error;
              {
                  py::gil_scoped_release release;
                  const int saved = omp_get_max_threads();
                  omp_set_num_threads(n_threads);
                  try {
                      r = pcm_kernel::second_choice(delta, mk, prm, adj.get());
                  } catch (const std::exception & e) {
                      error = e.what();
                  }
                  omp_set_num_threads(saved);
              }
              if (!error.empty()) throw py::value_error(error);
              py::dict d;
              d["C"] = r.C;
              d["s"] = r.s;
              d["N"] = r.N;
              d["out_delta"] = r.out_delta;
              d["out_theta"] = r.out_theta;
              d["draws_with_no_inside_second"] = r.draws_with_no_inside_second;
              d["seconds"] = r.seconds;
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("adjoint") = py::none(), py::kw_only(), py::arg("n_threads") = 1, py::arg("alpha_lo") = 0.0,
          py::arg("a_lo") = 0.0,
          "second_choice(delta, market, theta, phi, sigma_p, adjoint=None, *, n_threads=1, alpha_lo=0.0, a_lo=0.0) "
          "-> dict\n\n"
          "Exact second choices at a fixed delta (GMY second-choice moments): C (N x N) with C[j, k] = P(second = k | "
          "first = j, second inside) (rows of products nobody buys zero), s (N) model shares, N (N x N) the joint "
          "masses sum_i w_i s_ij q_ijk, draws_with_no_inside_second, seconds. adjoint is None or a mapping with A (K "
          "arrays N x N, or an array K x N x N) and B (K x N); then out_delta (K x N) and out_theta (K x P, columns "
          "[theta, phi, sigma_p]) are the derivatives of sum_jk A[c][j, k] N[j, k] - sum_j B[c, j] s_j. Releases the "
          "GIL; bad input raises ValueError." BOUND_DOC);
}
