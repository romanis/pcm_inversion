#include "pcm_kernel.hpp"
#include "bind_params.hpp"
#include <omp.h>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>
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

// A pcm_kernel.Market, or any mapping or object carrying x, p, weights, taste_basis, taste_col,
// scale_basis (the convention bind_welfare.cpp uses as well).
pcm_demand::Market market_from(const py::handle & obj, const char * fn) {
    if (py::isinstance<pcm_demand::Market>(obj)) return obj.cast<pcm_demand::Market>();
    const bool mapping = PyMapping_Check(obj.ptr()) && !py::isinstance<py::str>(obj);
    auto field = [&](const char * name) -> py::object {
        if (mapping && PyMapping_HasKeyString(obj.ptr(), name)) return obj[name];
        if (py::hasattr(obj, name)) return obj.attr(name);
        throw py::type_error(std::string(fn) + ": market has no field '" + name +
                             "' (needs x, p, weights, taste_basis, taste_col, scale_basis)");
    };
    return make_market(field("x").cast<Eigen::MatrixXd>(), field("p").cast<Eigen::ArrayXd>(),
                       field("weights").cast<Eigen::ArrayXd>(), field("taste_basis").cast<Eigen::MatrixXd>(),
                       field("taste_col").cast<std::vector<int>>(), field("scale_basis").cast<Eigen::MatrixXd>());
}

// None, or a mapping / object with H (R x L), U (N x Pm) and h_of_part (Pm ints).
std::unique_ptr<pcm_kernel::MicroRequest> micro_from(const py::object & obj, const char * fn) {
    if (obj.is_none()) return nullptr;
    const bool mapping = PyMapping_Check(obj.ptr()) && !py::isinstance<py::str>(obj);
    auto field = [&](const char * name) -> py::object {
        if (mapping && PyMapping_HasKeyString(obj.ptr(), name)) return obj[name];
        if (py::hasattr(obj, name)) return obj.attr(name);
        throw py::type_error(std::string(fn) + ": micro has no field '" + name + "' (needs H, U, h_of_part)");
    };
    auto mr = std::make_unique<pcm_kernel::MicroRequest>();
    mr->H = field("H").cast<Eigen::MatrixXd>();
    mr->U = field("U").cast<Eigen::MatrixXd>();
    mr->h_of_part = field("h_of_part").cast<std::vector<int>>();
    return mr;
}

// theta_warm is the stacked (P,) vector [theta, phi, sigma_p], in the column order of ds_dtheta.
// lo_warm: the bound delta_warm was solved under (NaN: the bound of theta).
std::unique_ptr<pcm_demand::Params> params_from_stacked(const py::object & obj, int M, int Q,
                                                        const pcm_demand::Params & bound, double lo_warm) {
    if (obj.is_none()) return nullptr;
    const Eigen::ArrayXd v = obj.cast<Eigen::ArrayXd>();
    if (v.size() != M + Q + 1)
        throw py::value_error("solve_market: theta_warm must be the stacked vector [theta, phi, sigma_p] of length " +
                              std::to_string(M + Q + 1) + ", got " + std::to_string(v.size()));
    return std::make_unique<pcm_demand::Params>(
        pcm_demand::Params{v.head(M), v.segment(M, Q), v[M + Q], std::isnan(lo_warm) ? bound.lo : lo_warm,
                           bound.bound_kind});
}

// pcm_demand::ThreadBudget is a std::atomic<int>, which cannot be moved: Python holds it inside this box.
struct ThreadBudgetBox {
    pcm_demand::ThreadBudget n{1};
};

}  // namespace

void bind_solve(py::module_ & m) {
    py::class_<ThreadBudgetBox>(m, "ThreadBudget", py::module_local(),
        "ThreadBudget(n): a team size another Python thread may change while solve_market runs with "
        "thread_budget=it (read before every share evaluation). Results do not depend on it.")
        .def(py::init([](int n) {
                 if (n < 1) throw py::value_error("ThreadBudget: n must be at least 1");
                 auto b = std::make_unique<ThreadBudgetBox>();
                 b->n = n;
                 return b;
             }), py::arg("n"))
        .def_property("value", [](const ThreadBudgetBox & b) { return b.n.load(); },
                      [](ThreadBudgetBox & b, int n) {
                          if (n < 1) throw py::value_error("ThreadBudget: value must be at least 1");
                          b.n.store(n);
                      });

    // Registered here, before bind_events and bind_welfare, so all three accept it. module_local
    // keeps it apart from python_pcm_demand.Market when both modules are loaded.
    py::class_<pcm_demand::Market>(m, "Market", py::module_local(),
        "One market for the kernel: x (N x K), p (N, strictly increasing), weights (R), taste_basis (R x M), "
        "taste_col (M ints), scale_basis (R x Q). Same fields as python_pcm_demand.Market.")
        .def(py::init(&make_market), py::arg("x"), py::arg("p"), py::arg("weights"),
             py::arg("taste_basis"), py::arg("taste_col"), py::arg("scale_basis"))
        .def_readonly("x", &pcm_demand::Market::x)
        .def_readonly("p", &pcm_demand::Market::p)
        .def_readonly("weights", &pcm_demand::Market::weights)
        .def_readonly("taste_basis", &pcm_demand::Market::taste_basis)
        .def_readonly("taste_col", &pcm_demand::Market::taste_col)
        .def_readonly("scale_basis", &pcm_demand::Market::scale_basis);

    m.def("solve_market",
          [](const Eigen::ArrayXd & s_data, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const py::object & delta_warm,
             const py::object & theta_warm, const py::object & micro, double tol, unsigned warm_evals,
             unsigned cold_evals, unsigned path_evals, double path_seconds, int n_threads, bool derivatives,
             bool price_derivatives, double rcond_min, double alpha_lo, double a_lo, unsigned cold_draws,
             int cold_draw_factor, const py::object & thread_budget, double lo_warm) {
              const pcm_demand::Market mk = market_from(market, "solve_market");
              const pcm_demand::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "solve_market");
              const Eigen::ArrayXd dw = delta_warm.is_none() ? Eigen::ArrayXd() : delta_warm.cast<Eigen::ArrayXd>();
              const auto pw = params_from_stacked(theta_warm, static_cast<int>(theta.size()), static_cast<int>(phi.size()),
                                                  prm, lo_warm);
              const auto mr = micro_from(micro, "solve_market");
              pcm_kernel::SolveOptions opt;
              opt.tol = tol;
              opt.warm_evals = warm_evals;
              opt.cold_evals = cold_evals;
              opt.path_evals = path_evals;
              opt.path_seconds = path_seconds;
              opt.n_threads = n_threads;
              opt.derivatives = derivatives;
              opt.price_derivatives = price_derivatives;
              opt.rcond_min = rcond_min;
              opt.cold_draws = cold_draws;
              opt.cold_draw_factor = cold_draw_factor;
              if (!thread_budget.is_none()) opt.thread_budget = &thread_budget.cast<ThreadBudgetBox &>().n;
              pcm_kernel::SolveResult r;
              {
                  py::gil_scoped_release release;
                  r = pcm_kernel::solve_market(s_data, mk, prm, dw, pw.get(), mr.get(), opt);
              }
              py::dict d;
              d["status"] = pcm_kernel::status_name(r.status);
              d["delta"] = r.delta;
              d["s"] = r.s;
              d["ds_ddelta"] = r.ds_ddelta;
              d["ds_dtheta"] = r.ds_dtheta;
              d["ddelta_dtheta"] = r.ddelta_dtheta;
              d["ds_dp"] = r.ds_dp;
              d["micro"] = r.micro;
              d["micro_d_delta"] = r.micro_d_delta;
              d["micro_d_theta"] = r.micro_d_theta;
              d["max_abs_residual"] = r.max_abs_residual;
              d["rcond"] = r.rcond;
              d["seconds"] = r.seconds;
              d["share_evaluations"] = r.share_evaluations;
              d["stages"] = r.stages;
              return d;
          },
          py::arg("s_data"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("delta_warm") = py::none(), py::arg("theta_warm") = py::none(), py::arg("micro") = py::none(),
          py::kw_only(), py::arg("tol") = 1e-9, py::arg("warm_evals") = 2500u, py::arg("cold_evals") = 40000u,
          py::arg("path_evals") = 1500u, py::arg("path_seconds") = 90.0, py::arg("n_threads") = 1,
          py::arg("derivatives") = true, py::arg("price_derivatives") = false, py::arg("rcond_min") = 1e-13,
          py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0, py::arg("cold_draws") = 0u, py::arg("cold_draw_factor") = 8,
          py::arg("thread_budget") = py::none(), py::arg("lo_warm") = std::numeric_limits<double>::quiet_NaN(),
          "solve_market(s_data, market, theta, phi, sigma_p, delta_warm=None, theta_warm=None, micro=None, *, "
          "tol=1e-9, warm_evals=2500, cold_evals=40000, path_evals=1500, path_seconds=90.0, n_threads=1, "
          "derivatives=True, price_derivatives=False, rcond_min=1e-13, alpha_lo=0.0, a_lo=0.0, cold_draws=0, "
          "cold_draw_factor=8, thread_budget=None, lo_warm=nan) -> dict\n\n"
          "lo_warm: the bound (of the kind alpha_lo / a_lo names) delta_warm was solved under, NaN for the bound of "
          "theta; when it differs, the theta path moves the bound from lo_warm to its value as well. "
          "thread_budget: a ThreadBudget whose value replaces n_threads from the next share evaluation on. "
          "cold_draws > 0 turns on the draw homotopy of the cold stage (SolveOptions::cold_draws): invert first on "
          "every k-th draw (at least cold_draws of them), then warm-start each finer level. "
          "Frontier lift, inversion and derivatives for one market in one call (DESIGN 3.2). market is a "
          "pcm_kernel.Market or any mapping or object with x, p, weights, taste_basis, taste_col, scale_basis. "
          "theta_warm is the stacked vector [theta, phi, sigma_p] at which delta_warm was solved. micro is "
          "None or a mapping with H (R x L), U (N x Pm), h_of_part (Pm ints). Returns every SolveResult field: "
          "status ('ok', 'not_converged', 'infeasible_start', 'singular_ds_ddelta', 'invalid_input'), delta, s, "
          "ds_ddelta, ds_dtheta, ddelta_dtheta, ds_dp, micro, micro_d_delta, micro_d_theta, max_abs_residual, "
          "rcond, seconds, share_evaluations, stages. Derivative arrays have shape (0, 0) unless the inversion "
          "converged and they were requested; ddelta_dtheta is NaN when status is 'singular_ds_ddelta'. "
          "Columns of the theta derivatives are [theta, phi, sigma_p]. Bad input is status 'invalid_input' with "
          "the reason in stages. Releases the GIL; n_threads is this call's OpenMP team size." BOUND_DOC " theta_warm is taken at the same bound.");

    m.def("evaluate_market",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, const py::object & micro, bool derivatives,
             bool price_derivatives, int n_threads, double alpha_lo, double a_lo) {
              const pcm_demand::Market mk = market_from(market, "evaluate_market");
              const pcm_demand::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "evaluate_market");
              const auto mr = micro_from(micro, "evaluate_market");
              if (n_threads < 1) throw py::value_error("evaluate_market: n_threads must be at least 1");
              pcm_kernel::MarketEvaluation r;
              std::string error;
              {
                  py::gil_scoped_release release;
                  const int saved = omp_get_max_threads();
                  omp_set_num_threads(n_threads);
                  try {
                      r = pcm_kernel::evaluate_market(delta, mk, prm, mr.get(), derivatives, price_derivatives);
                  } catch (const std::exception & e) {
                      error = e.what();
                  }
                  omp_set_num_threads(saved);
              }
              if (!error.empty()) throw py::value_error("evaluate_market: " + error);
              py::dict d;
              d["s"] = r.s;
              d["ds_ddelta"] = r.ds_ddelta;
              d["ds_dtheta"] = r.ds_dtheta;
              d["ds_dp"] = r.ds_dp;
              d["micro"] = r.micro;
              d["micro_d_delta"] = r.micro_d_delta;
              d["micro_d_theta"] = r.micro_d_theta;
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("micro") = py::none(), py::kw_only(), py::arg("derivatives") = true,
          py::arg("price_derivatives") = false, py::arg("n_threads") = 1, py::arg("alpha_lo") = 0.0,
          py::arg("a_lo") = 0.0,
          "evaluate_market(delta, market, theta, phi, sigma_p, micro=None, *, derivatives=True, "
          "price_derivatives=False, n_threads=1, alpha_lo=0.0, a_lo=0.0) -> dict\n\n"
          "Shares, partial derivatives and micro parts at a fixed delta, without inversion: the draw loop "
          "solve_market runs at its solution. Returns s, ds_ddelta, ds_dtheta, ds_dp, micro, micro_d_delta, "
          "micro_d_theta (unrequested arrays have shape (0, 0) or (0,)). Releases the GIL; bad input raises "
          "ValueError." BOUND_DOC);

    m.def("frontier_anatomy",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, int n_threads, double alpha_lo, double a_lo) {
              const pcm_demand::Market mk = market_from(market, "frontier_anatomy");
              const pcm_demand::Params prm = bound_params(theta, phi, sigma_p, alpha_lo, a_lo, "frontier_anatomy");
              if (n_threads < 1) throw py::value_error("frontier_anatomy: n_threads must be at least 1");
              pcm_kernel::FrontierAnatomy r;
              std::string error;
              {
                  py::gil_scoped_release release;
                  const int saved = omp_get_max_threads();
                  omp_set_num_threads(n_threads);
                  try {
                      r = pcm_kernel::frontier_anatomy(delta, mk, prm);
                  } catch (const std::exception & e) {
                      error = e.what();
                  }
                  omp_set_num_threads(saved);
              }
              if (!error.empty()) throw py::value_error("frontier_anatomy: " + error);
              py::dict d;
              d["frontier_size"] = r.frontier_size;
              d["mass"] = r.mass;
              d["log_range"] = r.log_range;
              d["mass_log_width"] = r.mass_log_width;
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("n_threads") = 1, py::kw_only(), py::arg("alpha_lo") = 0.0, py::arg("a_lo") = 0.0,
          "frontier_anatomy(delta, market, theta, phi, sigma_p, n_threads=1, *, alpha_lo=0.0, a_lo=0.0) -> dict\n\n"
          "Per-draw window anatomy at a fixed delta, (R,) arrays: frontier_size (products with a window of positive "
          "mass), mass (the draw's inside share), log_range (log(e_0 / a_lo,i), e_0 the cutoff of the cheapest "
          "frontier product; 0 for a draw that buys nothing, inf without a bound) and mass_log_width (sum over "
          "positive-mass windows of the window's mass x its width in log alpha). Releases the GIL; bad input raises "
          "ValueError." BOUND_DOC);
}
