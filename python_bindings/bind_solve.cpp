#include "pcm_kernel.hpp"
#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
[[noreturn]] void not_implemented(const std::string & what) {
    PyErr_SetString(PyExc_NotImplementedError, what.c_str());
    throw py::error_already_set();
}
}  // namespace

// WP0 stub: registers the names with their intended signatures; WP1 replaces the bodies.
void bind_solve(py::module_ & m) {
    m.def("solve_market",
          [](py::args, py::kwargs) { not_implemented("pcm_kernel.solve_market is not implemented yet (WP1)"); },
          "solve_market(s_data, market, theta, phi, sigma_p, delta_warm=None, theta_warm=None, micro=None, "
          "**options) -> dict\n\n"
          "Frontier lift, inversion and derivatives for one market in one call (DESIGN 3.2). Returns every "
          "SolveResult field: status, delta, s, ds_ddelta, ds_dtheta, ddelta_dtheta, ds_dp, micro, "
          "micro_d_delta, micro_d_theta, max_abs_residual, rcond, seconds, share_evaluations, stages.");
}
