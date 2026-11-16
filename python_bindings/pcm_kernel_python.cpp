#include <pybind11/pybind11.h>

namespace py = pybind11;

// Defined in bind_solve.cpp (WP1), bind_events.cpp (WP2) and bind_welfare.cpp (WP3).
void bind_solve(py::module_ & m);
void bind_events(py::module_ & m);
void bind_welfare(py::module_ & m);

PYBIND11_MODULE(pcm_kernel, m) {
    m.doc() = "Exact pure characteristics model kernel for nfp_harness: solve_market (inversion, "
              "derivatives, micro parts), max_step and frontier_signature (frontier events), "
              "consumer_surplus. Every call releases the GIL. "
              "See docs/estimation_harness/DESIGN.md section 3.";
    bind_solve(m);
    bind_events(m);
    bind_welfare(m);
}
