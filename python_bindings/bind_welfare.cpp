#include "pcm_welfare.hpp"
#include <string>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {
[[noreturn]] void not_implemented(const std::string & what) {
    PyErr_SetString(PyExc_NotImplementedError, what.c_str());
    throw py::error_already_set();
}
}  // namespace

// WP0 stub: registers the names with their intended signatures; WP3 replaces the bodies.
void bind_welfare(py::module_ & m) {
    m.def("consumer_surplus",
          [](py::args, py::kwargs) { not_implemented("pcm_kernel.consumer_surplus is not implemented yet (WP3)"); },
          "consumer_surplus(delta, market, theta, phi, sigma_p, n_threads) -> dict\n\n"
          "Consumer surplus by the frontier formula, in price units per household (DESIGN 3.4). Returns "
          "total and per_draw.");
}
