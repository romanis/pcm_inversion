#include "pcm_events.hpp"
#include "bind_params.hpp"

#include <stdexcept>
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// A market is either a registered pcm_aggregate::Market (if some module registered the type) or any
// mapping / object with x, p, weights, taste_basis, taste_col and optionally scale_basis.
pcm_aggregate::Market to_market(const py::object & obj) {
    if (py::isinstance<pcm_aggregate::Market>(obj)) return obj.cast<pcm_aggregate::Market>();
    const bool is_map = py::isinstance<py::dict>(obj);
    auto has = [&](const char * k) { return is_map ? obj.cast<py::dict>().contains(k) : py::hasattr(obj, k); };
    auto get = [&](const char * k) -> py::object {
        if (!has(k)) throw py::value_error(std::string("market is missing '") + k + "'");
        return is_map ? py::object(obj.cast<py::dict>()[k]) : obj.attr(k);
    };
    pcm_aggregate::Market mk;
    mk.x = get("x").cast<Eigen::MatrixXd>();
    mk.p = get("p").cast<Eigen::ArrayXd>();
    mk.weights = get("weights").cast<Eigen::ArrayXd>();
    mk.taste_basis = get("taste_basis").cast<Eigen::MatrixXd>();
    mk.taste_col = get("taste_col").cast<std::vector<int>>();
    if (has("scale_basis") && !get("scale_basis").is_none()) mk.scale_basis = get("scale_basis").cast<Eigen::MatrixXd>();
    else mk.scale_basis = Eigen::MatrixXd::Zero(mk.weights.size(), 0);
    return mk;
}

}  // namespace

void bind_events(py::module_ & m) {
    m.def("max_step",
          [](const Eigen::ArrayXd & delta, const Eigen::ArrayXd & delta_dot, const py::object & market,
             const Eigen::ArrayXd & theta, const Eigen::ArrayXd & phi, double sigma_p, const Eigen::ArrayXd & dtheta,
             double tau_max, int n_threads, double alpha_lo, double a_lo) {
              const pcm_aggregate::Market mk = to_market(market);
              const pcm_aggregate::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "max_step");
              pcm_events::StepEvents ev;
              {
                  py::gil_scoped_release release;
                  ev = pcm_events::max_step(delta, delta_dot, mk, prm, dtheta, tau_max, n_threads);
              }
              py::dict d;
              d["tau_first"] = ev.tau_first;
              d["tau_vanish_lb"] = ev.tau_vanish_lb;
              d["tau_draw"] = ev.tau_draw;
              d["kind_draw"] = ev.kind_draw;
              d["product_draw"] = ev.product_draw;
              d["mass_profile"] = ev.mass_profile;
              d["buyer_mass"] = ev.buyer_mass;
              return d;
          },
          py::arg("delta"), py::arg("delta_dot"), py::arg("market"), py::arg("theta"), py::arg("phi"),
          py::arg("sigma_p"), py::arg("dtheta"), py::arg("tau_max") = 1.0, py::arg("n_threads") = 1, py::kw_only(),
          py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0,
          "max_step(delta, delta_dot, market, theta, phi, sigma_p, dtheta, tau_max=1.0, n_threads=1, *, alpha_lo=0.0, a_lo=0.0) "
          "-> dict\n\n"
          "First frontier events along (delta + tau delta_dot, theta + tau dtheta), tau in [0, tau_max) "
          "(DESIGN 3.3). dtheta has M + Q + 1 entries in [taste, scale, sigma_p] order; only the taste part "
          "moves frontiers (and, under an a_lo bound, the scale part through the shear). market: a mapping or object with x (N x K), p (N, strictly increasing), "
          "weights (R), taste_basis (R x M), taste_col (M), scale_basis (R x Q, optional). Returns "
          "tau_first, tau_vanish_lb, tau_draw (R), kind_draw (R: 0 none, 1 leaves, 2 enters), product_draw "
          "(R, -1 if none), mass_profile (64 x 2: tau, max_j affected buyer mass of j), buyer_mass (N). "
          "Releases the GIL; bit-identical for every n_threads (n_threads <= 0: OpenMP default). With a bound "
          "the frontier is that of y - lo_y,i p, lo_y,i = alpha_lo or a_lo c_i: the products with a window of positive "
          "mass." BOUND_DOC);
    m.def("frontier_signature",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, int n_threads, double alpha_lo, double a_lo) {
              const pcm_aggregate::Market mk = to_market(market);
              const pcm_aggregate::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "frontier_signature");
              std::vector<uint64_t> sig;
              {
                  py::gil_scoped_release release;
                  sig = pcm_events::frontier_signature(delta, mk, prm, n_threads);
              }
              py::array_t<uint64_t> out(static_cast<py::ssize_t>(sig.size()));
              std::copy(sig.begin(), sig.end(), out.mutable_data());
              return out;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("n_threads") = 1, py::kw_only(), py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0,
          "frontier_signature(delta, market, theta, phi, sigma_p, n_threads=1, *, alpha_lo=0.0, a_lo=0.0) -> "
          "numpy.ndarray[uint64]\n\n"
          "Hash of each draw's frontier F_i (DESIGN 3.3). Two points share a draw's hash exactly when that "
          "draw buys the same products (up to 64-bit collisions).");
    m.def("affected_mass",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const py::array_t<bool, py::array::c_style | py::array::forcecast> & changed,
             int n_threads, double alpha_lo, double a_lo) {
              const pcm_aggregate::Market mk = to_market(market);
              const pcm_aggregate::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "affected_mass");
              if (changed.ndim() != 1) throw py::value_error("changed must be one-dimensional");
              std::vector<uint8_t> ch(changed.size());
              for (py::ssize_t i = 0; i < changed.size(); ++i) ch[i] = changed.data()[i] ? 1 : 0;
              Eigen::ArrayXd A;
              {
                  py::gil_scoped_release release;
                  A = pcm_events::affected_mass(delta, mk, prm, ch, n_threads);
              }
              return A;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("changed"), py::arg("n_threads") = 1, py::kw_only(), py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0,
          "affected_mass(delta, market, theta, phi, sigma_p, changed, n_threads=1, *, alpha_lo=0.0, a_lo=0.0) -> "
          "numpy.ndarray (N)\n\n"
          "Realised affected buyer mass for the guard of DESIGN 4.4: A_j = sum over draws i with j in F_i and "
          "changed[i] of w_i, divided by B_j = sum over draws with j in F_i of w_i, with F_i taken at the "
          "given point (0 where B_j = 0). Pass changed = (signature_here != signature_at_trial).");
}
