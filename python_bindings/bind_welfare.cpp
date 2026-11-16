#include "pcm_welfare.hpp"
#include <string>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

// The market argument is a pcm_demand::Market instance when some loaded module registers that type
// (bind_solve.cpp or python_pcm_demand), or any mapping or object carrying the fields
// x, p, weights, taste_basis, taste_col, scale_basis.
pcm_demand::Market market_from(const py::handle & obj) {
    if (py::detail::get_type_info(typeid(pcm_demand::Market))) {
        try {
            return obj.cast<pcm_demand::Market>();
        } catch (const py::cast_error &) {
        }
    }
    const bool mapping = PyMapping_Check(obj.ptr()) && !py::isinstance<py::str>(obj);
    auto field = [&](const char * name) -> py::object {
        if (mapping && PyMapping_HasKeyString(obj.ptr(), name)) return obj[name];
        if (py::hasattr(obj, name)) return obj.attr(name);
        throw py::type_error(std::string("consumer_surplus: market has no field '") + name +
                             "' (needs x, p, weights, taste_basis, taste_col, scale_basis)");
    };
    pcm_demand::Market mk;
    mk.x = field("x").cast<Eigen::MatrixXd>();
    mk.p = field("p").cast<Eigen::ArrayXd>();
    mk.weights = field("weights").cast<Eigen::ArrayXd>();
    mk.taste_basis = field("taste_basis").cast<Eigen::MatrixXd>();
    mk.taste_col = field("taste_col").cast<std::vector<int>>();
    mk.scale_basis = field("scale_basis").cast<Eigen::MatrixXd>();
    return mk;
}

}  // namespace

void bind_welfare(py::module_ & m) {
    m.def("consumer_surplus",
          [](const Eigen::ArrayXd & delta, const py::object & market, const Eigen::ArrayXd & theta,
             const Eigen::ArrayXd & phi, double sigma_p, int n_threads) {
              const pcm_demand::Market mk = market_from(market);
              const pcm_demand::Params prm{theta, phi, sigma_p};
              pcm_welfare::Surplus r;
              {
                  py::gil_scoped_release release;
                  r = pcm_welfare::consumer_surplus(delta, mk, prm, n_threads);
              }
              py::dict d;
              d["total"] = r.total;
              d["per_draw"] = r.per_draw;
              return d;
          },
          py::arg("delta"), py::arg("market"), py::arg("theta"), py::arg("phi"), py::arg("sigma_p"),
          py::arg("n_threads") = 1,
          "consumer_surplus(delta, market, theta, phi, sigma_p, n_threads=1) -> dict\n\n"
          "Consumer surplus by the frontier formula, in price units per household (DESIGN 3.4). Returns "
          "total (sum_i w_i CS_i) and per_draw (CS_i, unweighted). market is a pcm_demand Market or any "
          "mapping or object with x, p, weights, taste_basis, taste_col, scale_basis. Releases the GIL; "
          "invalid input raises ValueError.");
}
