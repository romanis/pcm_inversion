#include "pcm_groups.hpp"
#include "bind_params.hpp"
#include <omp.h>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_groups(py::module_ & m) {
    m.def("group_sums",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const Eigen::ArrayXi & group_of_draw, int n_groups,
             const Eigen::MatrixXd & U, bool derivatives, int n_threads, double alpha_lo, double a_lo) {
              if (!py::isinstance<pcm_aggregate::Market>(market))
                  throw py::type_error("group_sums: market must be a pcm_kernel.Market");
              const pcm_aggregate::Market mk = market.cast<pcm_aggregate::Market>();
              const pcm_aggregate::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "group_sums");
              if (n_threads < 1) throw py::value_error("group_sums: n_threads must be at least 1");
              pcm_kernel::GroupRequest gr;
              gr.group_of_draw = group_of_draw;
              gr.n_groups = n_groups;
              gr.U = U;
              pcm_kernel::GroupSums r;
              std::string error;
              {
                  py::gil_scoped_release release;
                  const int saved = omp_get_max_threads();
                  omp_set_num_threads(n_threads);
                  try {
                      r = pcm_kernel::group_sums(delta, mk, prm, gr, derivatives);
                  } catch (const std::exception & e) {
                      error = e.what();
                  }
                  omp_set_num_threads(saved);
              }
              if (!error.empty()) throw py::value_error(error);
              py::dict d;
              d["S"] = r.S;
              d["w"] = r.w;
              d["d_delta"] = r.d_delta;
              d["d_theta"] = r.d_theta;
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("group_of_draw"), py::arg("n_groups"), py::arg("U"), py::kw_only(), py::arg("derivatives") = true,
          py::arg("n_threads") = 1, py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0,
          "group_sums(delta, market, theta, phi, sigma_p, group_of_draw, n_groups, U, *, derivatives=True, n_threads=1, "
          "alpha_lo=0.0, a_lo=0.0) -> dict\n\n"
          "Per-household purchase-weighted sums at a fixed delta (GMY micro moments): S (G x (C+1)), column 0 the "
          "group's purchase probability sum_i w_i sum_j s_ij, columns 1..C sum_i w_i sum_j U_jc s_ij; w (G) the groups' "
          "weights; d_delta and d_theta lists of G arrays ((C+1) x N and (C+1) x P, columns [theta, phi, sigma_p]), "
          "empty without derivatives. group_of_draw (R ints) must be non-decreasing in 0..n_groups-1. Each group equals "
          "evaluate_market on its draws with micro H = 1, U' = [1, U] bit for bit. Releases the GIL; bad input raises "
          "ValueError." BOUND_DOC);
}
