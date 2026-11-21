#ifndef __BIND_PARAMS_HPP__
#define __BIND_PARAMS_HPP__

#include <string>
#include <pybind11/pybind11.h>
#include "pcm_aggregate.hpp"

// Params from the keyword pair every kernel binding takes: alpha_lo bounds alpha_i = c_i a_i below
// (BoundKind::ALPHA, LEARNINGS 29), a_lo bounds a_i below (BoundKind::A, LEARNINGS 32). At most one is non-zero;
// both zero is the untruncated model.
inline pcm_aggregate::Params bound_params(const Eigen::ArrayXd & theta, const Eigen::ArrayXd & phi, double sigma_p,
                                       double alpha_lo, double a_lo, const char * fn) {
    if (alpha_lo != 0.0 && a_lo != 0.0)
        throw pybind11::value_error(std::string(fn) + ": pass alpha_lo (a bound on alpha_i) or a_lo (a bound on a_i), "
                                    "not both");
    pcm_aggregate::Params prm{theta, phi, sigma_p};
    if (a_lo != 0.0) {
        prm.lo = a_lo;
        prm.bound_kind = pcm_aggregate::BoundKind::A;
    } else {
        prm.lo = alpha_lo;
    }
    return prm;
}

#define BOUND_DOC " alpha_lo bounds alpha_i = c_i a_i below (a_i truncated at alpha_lo / c_i, LEARNINGS 29); a_lo bounds " \
    "a_i below (a_i truncated at a_lo for every draw, so alpha_i >= a_lo c_i, LEARNINGS 32); at most one is non-zero, " \
    "both 0 is the untruncated model."

#endif
