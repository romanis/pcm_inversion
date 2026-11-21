#include "pcm_welfare.hpp"
#include "pcm_aggregate.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::ArrayXd;
using Eigen::VectorXd;

namespace pcm_welfare {

namespace {

constexpr double SQRT1_2 = 0.70710678118654752440;

inline double Phi(double z) { return 0.5 * std::erfc(-z * SQRT1_2); }

void check_input(const ArrayXd & delta, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm) {
    const auto N = mk.p.size();
    const auto R = mk.weights.size();
    const auto M = prm.theta.size();
    const auto Q = prm.phi.size();
    auto fail = [](const std::string & what) { throw std::invalid_argument("pcm_welfare::consumer_surplus: " + what); };
    if (delta.size() != N) fail("delta has " + std::to_string(delta.size()) + " entries, prices " + std::to_string(N));
    if (mk.x.rows() != N) fail("x has " + std::to_string(mk.x.rows()) + " rows, prices " + std::to_string(N));
    if (!(prm.sigma_p > 0.0) || !std::isfinite(prm.sigma_p)) fail("sigma_p must be positive and finite");
    if (!(prm.lo >= 0.0) || !std::isfinite(prm.lo)) fail("the bound lo must be finite and non-negative");
    if (N > 0 && !(mk.p[0] > 0.0)) fail("prices must be positive");
    for (Eigen::Index k = 1; k < N; ++k)
        if (!(mk.p[k] > mk.p[k - 1])) fail("prices must be strictly increasing");
    if (static_cast<Eigen::Index>(mk.taste_col.size()) != M || mk.taste_basis.cols() != M)
        fail("theta, taste_col and taste_basis columns must have the same length");
    if (M > 0 && mk.taste_basis.rows() != R) fail("taste_basis must have one row per draw");
    for (int col : mk.taste_col)
        if (col < 0 || col >= mk.x.cols()) fail("taste_col entry out of range");
    if (mk.scale_basis.cols() != Q) fail("phi and scale_basis columns must have the same length");
    if (Q > 0 && mk.scale_basis.rows() != R) fail("scale_basis must have one row per draw");
}

}  // namespace

Surplus consumer_surplus(const ArrayXd & delta, const pcm_aggregate::Market & mk,
                         const pcm_aggregate::Params & prm, int n_threads) {
    check_input(delta, mk, prm);
    const int N = static_cast<int>(mk.p.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const double s = prm.sigma_p, s2 = s * s;
    const double inv_mass_all = std::exp(0.5 * s2);                 // E(1/a)

    Surplus out;
    out.per_draw = ArrayXd::Zero(R);
    const int nt = std::max(1, n_threads);
    // per-draw outputs written by index: no reduction
    pcm_aggregate::for_each_draw_block<pcm_aggregate::NoAcc>(R, [&](pcm_aggregate::NoAcc &, int i0, int i1) {
        VectorXd t(K);
        ArrayXd q(N);
        std::vector<int> hull;
        hull.reserve(N + 1);
        for (int i = i0; i < i1; ++i) {
            const double c = pcm_aggregate::conditional_quality(delta, mk, prm, i, t, q);
            // Truncation below at a_lo = draw_a_lo(prm, c), lo / c or lo (LEARNINGS 29, 32): F and E(1/a; a < e) become
            // F_T(e) = (Phi(z) - Phi(z_lo)) / Z and E_T(e) = exp(s^2/2) [Phi((l + s^2)/s) - Phi((l_lo + s^2)/s)] / Z
            // for e > a_lo and 0 below, so the last window ends at a_lo and CS_i <= max_j q_j / a_lo - p_j.
            const bool bound = prm.lo > 0.0;
            double l_lo = 0.0, F0 = 0.0, E0 = 0.0, Z = 1.0;
            if (bound) {
                l_lo = std::log(pcm_aggregate::draw_a_lo(prm, c));
                F0 = Phi(l_lo / s);
                Z = 0.5 * std::erfc(l_lo / s * SQRT1_2);
                E0 = inv_mass_all * Phi((l_lo + s2) / s);
            }
            // upper hull by price (prices are sorted); index -1 is the origin (0, 0)
            auto px = [&](int k) { return k < 0 ? 0.0 : mk.p[k]; };
            auto qy = [&](int k) { return k < 0 ? 0.0 : q[k]; };
            hull.clear();
            hull.push_back(-1);
            for (int k = 0; k < N; ++k) {
                while (hull.size() >= 2) {
                    const int a = hull[hull.size() - 2], b = hull.back();
                    const double cross = (px(b) - px(a)) * (qy(k) - qy(a)) - (qy(b) - qy(a)) * (px(k) - px(a));
                    if (cross >= 0) hull.pop_back(); else break;
                }
                hull.push_back(k);
            }
            // Increasing part of the hull: vertex hull[h] is reached by an edge of slope e_h > 0 and
            // the slopes decrease along it. A consumer with a < e_h prefers hull[h] to hull[h-1], so
            // hull[h] is bought for a in (e_{h+1}, e_h), with e_{h+1} = 0 after the last vertex.
            double cs = 0.0;
            double F_hi = 0.0, E_hi = 0.0;              // F and E(1/a; a < .) at the slope into the vertex
            int b = -1;                                 // current vertex, -1 before the first
            for (size_t h = 1; h <= hull.size(); ++h) {
                double F_lo = 0.0, E_lo = 0.0;
                bool next = false;
                if (h < hull.size()) {
                    const int a = hull[h - 1], v = hull[h];
                    if (qy(v) > qy(a)) {
                        const double l = std::log((qy(v) - qy(a)) / (px(v) - px(a)));
                        if (!bound) {
                            F_lo = Phi(l / s);
                            E_lo = inv_mass_all * Phi((l + s2) / s);
                        } else if (l > l_lo && Z > 0.0) {
                            F_lo = (Phi(l / s) - F0) / Z;
                            E_lo = (inv_mass_all * Phi((l + s2) / s) - E0) / Z;
                        }   // else the window lies below the bound: F = E = 0, as after the last vertex
                        next = true;
                    }
                }
                if (b >= 0) cs += qy(b) * (E_hi - E_lo) - px(b) * (F_hi - F_lo);
                if (!next) break;
                b = hull[h];
                F_hi = F_lo;
                E_hi = E_lo;
            }
            out.per_draw[i] = cs;
        }
    }, [](const pcm_aggregate::NoAcc &) {}, pcm_aggregate::DrawLoop{0, nt});
    // weighted sum in draw order, so the total does not depend on the thread count
    double total = 0.0;
    for (int i = 0; i < R; ++i) total += mk.weights[i] * out.per_draw[i];
    out.total = total;
    return out;
}

}  // namespace pcm_welfare
