#include "pcm_events.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::MatrixXd;

namespace {

constexpr int BLOCK = 64;

int team_size(int n_threads) { return n_threads > 0 ? n_threads : omp_get_max_threads(); }

int n_blocks_of(int R) { return std::max(1, (R + BLOCK - 1) / BLOCK); }

void check_market(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                  const char * who)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    auto fail = [&](const std::string & what) { throw std::invalid_argument(std::string(who) + ": " + what); };
    if (mk.x.rows() != N || mk.p.size() != N) fail("x rows, p and delta must have the same size");
    if (mk.taste_basis.rows() != R || mk.taste_basis.cols() != M || static_cast<int>(mk.taste_col.size()) != M)
        fail("taste_basis must be R x M and taste_col of size M (M = size of theta)");
    for (int c : mk.taste_col)
        if (c < 0 || c >= K) fail("taste_col out of range");
    for (int j = 0; j < N; ++j)
        if (!(j == 0 ? mk.p[0] > 0 : mk.p[j] > mk.p[j - 1])) fail("prices must be positive and strictly increasing");
    if (!delta.allFinite() || !prm.theta.allFinite() || !mk.taste_basis.allFinite() || !mk.weights.allFinite())
        fail("delta, theta, taste_basis and weights must be finite");
    if (!(prm.alpha_lo >= 0.0) || !std::isfinite(prm.alpha_lo)) fail("alpha_lo must be finite and non-negative");
}

// The utility the frontier is built on: y = delta + x . t, sheared to y - alpha_lo p under a bound. Draw i
// buys a vertex with entry slope e (in y / c units) only if e > a_lo = alpha_lo / c, that is if its slope in
// y units exceeds alpha_lo, so the increasing part of the sheared hull is the set of products with a window
// of positive mass, whatever c_i and sigma_p (F5 still holds).
void shear(const pcm_demand::Params & prm, const pcm_demand::Market & mk, ArrayXd & y)
{
    if (prm.alpha_lo > 0.0) y -= prm.alpha_lo * mk.p;
}

// y = base + sum_m coef_m * taste_basis(i, m) * x.col(taste_col[m])
void affine_utility(const ArrayXd & base, const pcm_demand::Market & mk, const ArrayXd & coef, int i, ArrayXd & y)
{
    y = base;
    for (int m = 0; m < coef.size(); ++m) {
        const double a = coef[m] * mk.taste_basis(i, m);
        if (a != 0.0) y += a * mk.x.col(mk.taste_col[m]).array();
    }
}

// Vertices of the increasing part of the upper hull of {(0,0)} U {(p_k, y_k)}, origin excluded, in
// price order. Collinear and duplicate points are not vertices (cross >= 0 pops), as in
// index_positive_shares: a product is a vertex exactly when it has an open interval of buyers.
void frontier(const ArrayXd & y, const ArrayXd & p, std::vector<int> & hull)
{
    const int N = static_cast<int>(y.size());
    auto px = [&](int k) { return k < 0 ? 0.0 : p[k]; };
    auto py = [&](int k) { return k < 0 ? 0.0 : y[k]; };
    hull.clear();
    hull.push_back(-1);
    for (int k = 0; k < N; ++k) {
        while (hull.size() >= 2) {
            const int a = hull[hull.size() - 2], b = hull.back();
            const double cross = (px(b) - px(a)) * (y[k] - py(a)) - (py(b) - py(a)) * (p[k] - px(a));
            if (cross >= 0) hull.pop_back(); else break;
        }
        hull.push_back(k);
    }
    // Upper-hull slopes decrease, so the increasing part is the prefix with positive slopes.
    size_t end = 1;
    while (end < hull.size() && py(hull[end]) > py(hull[end - 1])) ++end;
    hull.resize(end);
    hull.erase(hull.begin());
}

struct FirstEvent {
    double tau;
    int kind;
    int product;
};

// Certificate f(tau) = f0 + tau f1 with f0 >= 0 at tau = 0 (rounding below 0 is read as 0). It
// fails at f0 / (-f1) when f1 < 0. Strict '<' keeps the earliest certificate on ties.
inline void consider(double f0, double f1, int kind, int product, FirstEvent & ev)
{
    if (!(f1 < 0)) return;
    const double t = std::max(f0, 0.0) / (-f1);
    if (t < ev.tau) ev = {t, kind, product};
}

// First event of one draw along y + tau v, given its frontier H (vertex indices in price order).
FirstEvent first_event(const ArrayXd & y, const ArrayXd & v, const ArrayXd & p, const std::vector<int> & H,
                       double tau_max)
{
    FirstEvent ev{tau_max, 0, -1};
    const int m = static_cast<int>(H.size());
    const int N = static_cast<int>(y.size());
    auto P = [&](int r) { return r == 0 ? 0.0 : p[H[r - 1]]; };
    auto Y = [&](int r) { return r == 0 ? 0.0 : y[H[r - 1]]; };
    auto V = [&](int r) { return r == 0 ? 0.0 : v[H[r - 1]]; };

    // C1: strict convexity at interior vertices r = 1..m-1 (vertex h_r leaves)
    for (int r = 1; r < m; ++r) {
        const double gl = P(r) - P(r - 1), gr = P(r + 1) - P(r);
        const double f0 = (Y(r) - Y(r - 1)) * gr - (Y(r + 1) - Y(r)) * gl;
        const double f1 = (V(r) - V(r - 1)) * gr - (V(r + 1) - V(r)) * gl;
        consider(f0, f1, 1, H[r - 1], ev);
    }
    // C2: last slope positive (top vertex leaves)
    if (m >= 1) consider(Y(m) - Y(m - 1), V(m) - V(m - 1), 1, H[m - 1], ev);

    // C3 / C4: every product off the frontier stays strictly below it (product enters)
    int r = 0;          // edge (h_r, h_{r+1}) whose price range holds the current product
    int next = 0;       // index into H of the next vertex
    for (int k = 0; k < N; ++k) {
        if (next < m && H[next] == k) { r = next + 1; ++next; continue; }
        if (r < m) {    // P(r) < p_k < P(r+1)
            const double gap = P(r + 1) - P(r), dp = p[k] - P(r);
            const double f0 = (Y(r) - y[k]) * gap + dp * (Y(r + 1) - Y(r));
            const double f1 = (V(r) - v[k]) * gap + dp * (V(r + 1) - V(r));
            consider(f0, f1, 2, k, ev);
        } else {        // priced above the top vertex (or m = 0): stays below Y_m
            consider(Y(m) - y[k], V(m) - v[k], 2, k, ev);
        }
    }
    return ev;
}

inline uint64_t mix64(uint64_t z)
{
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

uint64_t hash_frontier(const std::vector<int> & H)
{
    uint64_t h = mix64(static_cast<uint64_t>(H.size()));
    for (int k : H) h = mix64(h ^ static_cast<uint64_t>(static_cast<uint32_t>(k)));
    return h;
}

// Frontier of every draw at (delta, theta), flattened: draw i's vertices are flat[off[i]..off[i+1]).
void all_frontiers(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                   int n_threads, std::vector<int> & flat, std::vector<long> & off)
{
    const int R = static_cast<int>(mk.weights.size());
    const int nb = n_blocks_of(R);
    std::vector<std::vector<int>> part(nb);
    std::vector<std::vector<int>> sizes(nb);
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size(n_threads)) if(nb > 1)
    for (int b = 0; b < nb; ++b) {
        ArrayXd y;
        std::vector<int> H;
        H.reserve(delta.size() + 1);
        for (int i = b * BLOCK; i < std::min(R, (b + 1) * BLOCK); ++i) {
            affine_utility(delta, mk, prm.theta, i, y);
            shear(prm, mk, y);
            frontier(y, mk.p, H);
            part[b].insert(part[b].end(), H.begin(), H.end());
            sizes[b].push_back(static_cast<int>(H.size()));
        }
    }
    flat.clear();
    off.assign(1, 0);
    for (int b = 0; b < nb; ++b) {
        flat.insert(flat.end(), part[b].begin(), part[b].end());
        for (int s : sizes[b]) off.push_back(off.back() + s);
    }
}

}  // namespace

namespace pcm_events {

StepEvents max_step(const ArrayXd & delta, const ArrayXd & delta_dot, const pcm_demand::Market & mk,
                    const pcm_demand::Params & prm, const ArrayXd & dtheta, double tau_max, int n_threads)
{
    check_market(delta, mk, prm, "pcm_events::max_step");
    const int N = static_cast<int>(delta.size());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    if (delta_dot.size() != N) throw std::invalid_argument("pcm_events::max_step: delta_dot must have size N");
    if (dtheta.size() != M + Q + 1)
        throw std::invalid_argument("pcm_events::max_step: dtheta must have size M + Q + 1 (taste, scale, sigma_p)");
    if (!delta_dot.allFinite() || !dtheta.allFinite())
        throw std::invalid_argument("pcm_events::max_step: delta_dot and dtheta must be finite");
    if (!(tau_max > 0) || !std::isfinite(tau_max))
        throw std::invalid_argument("pcm_events::max_step: tau_max must be positive and finite");

    const ArrayXd dtaste = dtheta.head(M);   // scale and sigma_p never move a frontier (F5)

    StepEvents out;
    out.tau_draw.resize(R);
    out.kind_draw.resize(R);
    out.product_draw.resize(R);

    const int nb = n_blocks_of(R);
    std::vector<std::vector<int>> part(nb), sizes(nb);
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size(n_threads)) if(nb > 1)
    for (int b = 0; b < nb; ++b) {
        ArrayXd y, v;
        std::vector<int> H;
        H.reserve(N + 1);
        for (int i = b * BLOCK; i < std::min(R, (b + 1) * BLOCK); ++i) {
            affine_utility(delta, mk, prm.theta, i, y);
            shear(prm, mk, y);
            affine_utility(delta_dot, mk, dtaste, i, v);
            frontier(y, mk.p, H);
            const FirstEvent ev = first_event(y, v, mk.p, H, tau_max);
            out.tau_draw[i] = ev.tau;
            out.kind_draw[i] = ev.kind;
            out.product_draw[i] = ev.product;
            part[b].insert(part[b].end(), H.begin(), H.end());
            sizes[b].push_back(static_cast<int>(H.size()));
        }
    }

    // Serial reduction in draw order: deterministic and O(sum_i |F_i|).
    const int G = MASS_GRID;
    auto grid_tau = [&](int g) { return tau_max * static_cast<double>(g) / (G - 1); };
    ArrayXd B = ArrayXd::Zero(N);
    ArrayXd max_tau = ArrayXd::Constant(N, -std::numeric_limits<double>::infinity());
    MatrixXd hist = MatrixXd::Zero(N, G);   // hist(j, g): event mass of j first counted at grid point g
    int i = 0;
    for (int b = 0; b < nb; ++b) {
        size_t pos = 0;
        for (int s : sizes[b]) {
            const double w = mk.weights[i], t = out.tau_draw[i];
            int g = -1;
            if (out.kind_draw[i] != 0) {
                g = std::min(G - 1, static_cast<int>(std::ceil(t / tau_max * (G - 1))));
                while (g > 0 && grid_tau(g - 1) >= t) --g;
                while (g < G - 1 && grid_tau(g) < t) ++g;
            }
            for (int q = 0; q < s; ++q) {
                const int j = part[b][pos + q];
                B[j] += w;
                max_tau[j] = std::max(max_tau[j], t);
                if (g >= 0) hist(j, g) += w;
            }
            pos += s;
            ++i;
        }
    }
    out.buyer_mass = B;
    out.tau_first = R > 0 ? out.tau_draw.minCoeff() : tau_max;
    out.tau_vanish_lb = tau_max;
    for (int j = 0; j < N; ++j)
        if (B[j] > 0) out.tau_vanish_lb = std::min(out.tau_vanish_lb, max_tau[j]);

    out.mass_profile.resize(G, 2);
    ArrayXd cum = ArrayXd::Zero(N);
    for (int g = 0; g < G; ++g) {
        cum += hist.col(g).array();
        double mx = 0.0;
        for (int j = 0; j < N; ++j)
            if (B[j] > 0) mx = std::max(mx, cum[j] / B[j]);
        out.mass_profile(g, 0) = grid_tau(g);
        out.mass_profile(g, 1) = mx;
    }
    return out;
}

std::vector<uint64_t> frontier_signature(const ArrayXd & delta, const pcm_demand::Market & mk,
                                         const pcm_demand::Params & prm, int n_threads)
{
    check_market(delta, mk, prm, "pcm_events::frontier_signature");
    const int R = static_cast<int>(mk.weights.size());
    std::vector<uint64_t> sig(R);
    const int nb = n_blocks_of(R);
#pragma omp parallel for schedule(dynamic, 1) num_threads(team_size(n_threads)) if(nb > 1)
    for (int b = 0; b < nb; ++b) {
        ArrayXd y;
        std::vector<int> H;
        H.reserve(delta.size() + 1);
        for (int i = b * BLOCK; i < std::min(R, (b + 1) * BLOCK); ++i) {
            affine_utility(delta, mk, prm.theta, i, y);
            shear(prm, mk, y);
            frontier(y, mk.p, H);
            sig[i] = hash_frontier(H);
        }
    }
    return sig;
}

ArrayXd affected_mass(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                      const std::vector<uint8_t> & changed, int n_threads)
{
    check_market(delta, mk, prm, "pcm_events::affected_mass");
    const int N = static_cast<int>(delta.size());
    const int R = static_cast<int>(mk.weights.size());
    if (static_cast<int>(changed.size()) != R)
        throw std::invalid_argument("pcm_events::affected_mass: changed must have one entry per draw");
    std::vector<int> flat;
    std::vector<long> off;
    all_frontiers(delta, mk, prm, n_threads, flat, off);
    ArrayXd B = ArrayXd::Zero(N), A = ArrayXd::Zero(N);
    for (int i = 0; i < R; ++i) {
        const double w = mk.weights[i];
        for (long q = off[i]; q < off[i + 1]; ++q) {
            B[flat[q]] += w;
            if (changed[i]) A[flat[q]] += w;
        }
    }
    for (int j = 0; j < N; ++j) A[j] = B[j] > 0 ? A[j] / B[j] : 0.0;
    return A;
}

}  // namespace pcm_events
