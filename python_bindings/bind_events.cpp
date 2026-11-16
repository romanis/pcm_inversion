#include "pcm_events.hpp"
#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
[[noreturn]] void not_implemented(const std::string & what) {
    PyErr_SetString(PyExc_NotImplementedError, what.c_str());
    throw py::error_already_set();
}
}  // namespace

// WP0 stub: registers the names with their intended signatures; WP2 replaces the bodies.
void bind_events(py::module_ & m) {
    m.def("max_step",
          [](py::args, py::kwargs) { not_implemented("pcm_kernel.max_step is not implemented yet (WP2)"); },
          "max_step(delta, delta_dot, market, theta, phi, sigma_p, dtheta, tau_max, n_threads) -> dict\n\n"
          "First frontier events along (delta + tau delta_dot, theta + tau dtheta) (DESIGN 3.3). Returns "
          "tau_first, tau_vanish_lb, tau_draw, kind_draw, product_draw, mass_profile.");
    m.def("frontier_signature",
          [](py::args, py::kwargs) { not_implemented("pcm_kernel.frontier_signature is not implemented yet (WP2)"); },
          "frontier_signature(delta, market, theta, phi, sigma_p, n_threads) -> numpy.ndarray[uint64]\n\n"
          "Hash of each draw's frontier F_i (DESIGN 3.3).");
}
