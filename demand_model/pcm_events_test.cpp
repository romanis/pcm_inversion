#include <gtest/gtest.h>
#include "pcm_events.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <iterator>
#include <set>

using Eigen::ArrayXd;
using Eigen::MatrixXd;

// defined in pcm_market_share.cpp: products with positive conditional share in the vertical model
std::vector<int> index_positive_shares(const ArrayXd & delta, const ArrayXd & p);

namespace {

using pcm_demand::Market;
using pcm_demand::Params;

ArrayXd vec(std::initializer_list<double> v) {
    ArrayXd a(v.size());
    int i = 0;
    for (double d : v) a[i++] = d;
    return a;
}

// One draw, no taste or scale parameters: y = delta.
Market vertical_market(const ArrayXd & p, int R = 1) {
    Market mk;
    mk.x = MatrixXd::Zero(p.size(), 1);
    mk.p = p;
    mk.weights = ArrayXd::Constant(R, 1.0 / R);
    mk.taste_basis = MatrixXd::Zero(R, 0);
    mk.scale_basis = MatrixXd::Zero(R, 0);
    return mk;
}

Params no_params() { return Params{ArrayXd(0), ArrayXd(0), 1.0}; }

// F_i of a one-draw market, read through affected_mass with changed = {1}.
std::set<int> frontier_set(const ArrayXd & delta, const Market & mk1, const Params & prm) {
    const ArrayXd A = pcm_events::affected_mass(delta, mk1, prm, std::vector<uint8_t>{1}, 1);
    std::set<int> s;
    for (int j = 0; j < A.size(); ++j) if (A[j] > 0) s.insert(j);
    return s;
}

Market one_draw(const Market & mk, int i) {
    Market m1 = mk;
    m1.weights = ArrayXd::Ones(1);
    m1.taste_basis = mk.taste_basis.row(i);
    m1.scale_basis = mk.scale_basis.rows() > 0 ? MatrixXd(mk.scale_basis.row(i)) : MatrixXd::Zero(1, 0);
    return m1;
}

// Random market: N products, R draws, K characteristics, M tastes on random columns, Q scale terms.
Market random_market(std::mt19937_64 & rng, int N, int R, int K, int M, int Q) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(5.0, 50.0);
    Market mk;
    std::vector<double> p(N);
    for (double & v : p) v = ud(rng);
    std::sort(p.begin(), p.end());
    mk.p = Eigen::Map<ArrayXd>(p.data(), N);
    for (int j = 1; j < N; ++j) if (!(mk.p[j] > mk.p[j - 1])) mk.p[j] = mk.p[j - 1] + 1e-3;
    mk.x = MatrixXd(N, K);
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < K; ++k) mk.x(j, k) = k == 0 ? (nd(rng) > 0 ? 1.0 : 0.0) : nd(rng);
    mk.weights = ArrayXd::Constant(R, 1.0 / R);
    mk.taste_basis = MatrixXd(R, M);
    for (int i = 0; i < R; ++i) for (int m = 0; m < M; ++m) mk.taste_basis(i, m) = nd(rng);
    for (int m = 0; m < M; ++m) mk.taste_col.push_back(m % K);
    mk.scale_basis = MatrixXd(R, Q);
    for (int i = 0; i < R; ++i) for (int q = 0; q < Q; ++q) mk.scale_basis(i, q) = nd(rng);
    return mk;
}

// Quality rising and concave in price, so several products sit on each frontier.
ArrayXd concave_delta(std::mt19937_64 & rng, const ArrayXd & p, double noise) {
    std::normal_distribution<double> nd(0.0, noise);
    ArrayXd d(p.size());
    for (int j = 0; j < p.size(); ++j) d[j] = 6.0 * std::sqrt(p[j]) - 10.0 + nd(rng);
    return d;
}

}  // namespace

// 1. B below the A-C line enters at tau = 0.5.
TEST(PcmEvents, Test1ProductEnters) {
    const Market mk = vertical_market(vec({10, 12, 14}));
    const auto ev = pcm_events::max_step(vec({5, 5.5, 7}), vec({0, 1, 0}), mk, no_params(), vec({0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 0.5, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 2);
    EXPECT_EQ(ev.product_draw[0], 1);
    EXPECT_NEAR(ev.tau_first, 0.5, 1e-12);
}

// 2. B on the hull leaves at tau = 0.5.
TEST(PcmEvents, Test2ProductLeaves) {
    const Market mk = vertical_market(vec({10, 12, 14}));
    const auto ev = pcm_events::max_step(vec({5, 6.5, 7}), vec({0, -1, 0}), mk, no_params(), vec({0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 0.5, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 1);
    EXPECT_EQ(ev.product_draw[0], 1);
}

// 3. Last slope reaches zero (C2) at tau = 3.
TEST(PcmEvents, Test3LastSlope) {
    const Market mk = vertical_market(vec({10, 20}));
    const auto ev = pcm_events::max_step(vec({5, 8}), vec({0, -1}), mk, no_params(), vec({0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 3.0, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 1);
    EXPECT_EQ(ev.product_draw[0], 1);
}

// 4. A product priced above the top vertex enters as the new top (C4) at tau = 1.
TEST(PcmEvents, Test4NewTop) {
    const Market mk = vertical_market(vec({10, 20}));
    const auto ev = pcm_events::max_step(vec({5, 4}), vec({0, 1}), mk, no_params(), vec({0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 1.0, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 2);
    EXPECT_EQ(ev.product_draw[0], 1);
}

// 5. Empty frontier: the product enters through the origin edge at tau = 1.
TEST(PcmEvents, Test5EnterThroughOrigin) {
    const Market mk = vertical_market(vec({10}));
    const auto ev = pcm_events::max_step(vec({-1}), vec({1}), mk, no_params(), vec({0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 1.0, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 2);
    EXPECT_EQ(ev.product_draw[0], 0);
    EXPECT_EQ(ev.buyer_mass[0], 0.0);
    EXPECT_EQ(ev.tau_vanish_lb, 10.0);   // no product has a buyer
}

// 6. The same event as test 1, reached through a taste direction with delta fixed.
TEST(PcmEvents, Test6TasteDirection) {
    Market mk = vertical_market(vec({10, 12, 14}));
    mk.x = MatrixXd(3, 1);
    mk.x << 0, 1, 0;
    mk.taste_basis = MatrixXd::Ones(1, 1);
    mk.taste_col = {0};
    const Params prm{vec({0}), ArrayXd(0), 1.0};
    const auto ev = pcm_events::max_step(vec({5, 5.5, 7}), ArrayXd::Zero(3), mk, prm, vec({1, 0}), 10.0, 1);
    EXPECT_NEAR(ev.tau_draw[0], 0.5, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 2);
    EXPECT_EQ(ev.product_draw[0], 1);
}

// 7. Scale and sigma_p directions never move a frontier (F5).
TEST(PcmEvents, Test7ScaleAndSigmaDirections) {
    std::mt19937_64 rng(7);
    const Market mk = random_market(rng, 20, 200, 3, 2, 2);
    const Params prm{vec({0.8, -0.5}), vec({0.3, -0.2}), 1.2};
    const ArrayXd delta = concave_delta(rng, mk.p, 0.5);
    const auto ev = pcm_events::max_step(delta, ArrayXd::Zero(20), mk, prm, vec({0, 0, 1.0, -2.0, 3.0}), 5.0, 4);
    EXPECT_TRUE((ev.tau_draw == 5.0).all());
    EXPECT_TRUE((ev.kind_draw == 0).all());
    EXPECT_TRUE((ev.product_draw == -1).all());
    EXPECT_EQ(ev.tau_first, 5.0);
    EXPECT_EQ(ev.mass_profile.col(1).maxCoeff(), 0.0);
}

// 8. Two draws, weights 0.3 and 0.7, events at tau = 5 and 2.5 on the one product they both buy.
TEST(PcmEvents, Test8MassProfileAndVanishBound) {
    Market mk = vertical_market(vec({10, 30}), 2);
    mk.weights = vec({0.3, 0.7});
    mk.x = MatrixXd(2, 1);
    mk.x << 1, 0;
    mk.taste_basis = MatrixXd(2, 1);
    mk.taste_basis << 1, 2;
    mk.taste_col = {0};
    const Params prm{vec({0}), ArrayXd(0), 1.0};
    // y_i0 = 5 - tau b_i (C2 fails at 5 / b_i); product 1 (y = -100) is never bought
    const double tau_max = 6.4;
    const auto ev = pcm_events::max_step(vec({5, -100}), ArrayXd::Zero(2), mk, prm, vec({-1, 0}), tau_max, 1);
    EXPECT_NEAR(ev.tau_draw[0], 5.0, 1e-12);
    EXPECT_NEAR(ev.tau_draw[1], 2.5, 1e-12);
    EXPECT_EQ(ev.kind_draw[0], 1);
    EXPECT_EQ(ev.kind_draw[1], 1);
    EXPECT_NEAR(ev.tau_first, 2.5, 1e-12);
    EXPECT_NEAR(ev.tau_vanish_lb, 5.0, 1e-12);
    EXPECT_NEAR(ev.buyer_mass[0], 1.0, 1e-15);
    EXPECT_EQ(ev.buyer_mass[1], 0.0);
    ASSERT_EQ(ev.mass_profile.rows(), pcm_events::MASS_GRID);
    for (int g = 0; g < ev.mass_profile.rows(); ++g) {
        const double t = ev.mass_profile(g, 0);
        EXPECT_NEAR(t, tau_max * g / (pcm_events::MASS_GRID - 1), 1e-15);
        const double expect = t < 2.5 ? 0.0 : (t < 5.0 ? 0.7 : 1.0);
        EXPECT_NEAR(ev.mass_profile(g, 1), expect, 1e-15) << "tau " << t;
    }
}

// 9. Brute force: the first signature change on a 10^5-point tau grid matches tau_draw within one
// grid step, and the frontier changes by exactly product_draw (added for kind 2, removed for kind 1).
// The hull itself is checked against index_positive_shares along the path.
TEST(PcmEvents, Test9BruteForce) {
    const int N = 20, R = 50, K = 3, M = 3, Q = 1, G = 100001;
    int n_events = 0, n_none = 0, n_enter = 0, n_leave = 0;
    for (int seed = 1; seed <= 3; ++seed) {
        std::mt19937_64 rng(1000 + seed);
        std::normal_distribution<double> nd(0.0, 1.0);
        const Market mk = random_market(rng, N, R, K, M, Q);
        const Params prm{vec({1.0, 0.7, -0.4}), vec({0.25}), 0.9};
        const ArrayXd delta = concave_delta(rng, mk.p, 1.0);
        ArrayXd delta_dot(N);
        for (int j = 0; j < N; ++j) delta_dot[j] = nd(rng);
        const ArrayXd dtheta = vec({0.6, -0.8, 0.5, 2.0, -1.0});
        const double tau_max = seed == 1 ? 0.05 : (seed == 2 ? 0.3 : 2.0);   // mix of draws with and without events
        const auto ev = pcm_events::max_step(delta, delta_dot, mk, prm, dtheta, tau_max, 4);

        auto point = [&](double t, ArrayXd & d, Params & pr) {
            d = delta + t * delta_dot;
            pr = Params{prm.theta + t * dtheta.head(M), prm.phi + t * dtheta.segment(M, Q), prm.sigma_p + t * dtheta[M + Q]};
        };
        const std::vector<uint64_t> sig0 = pcm_events::frontier_signature(delta, mk, prm, 1);
        std::vector<int> first(R, G);   // first grid index whose signature differs from tau = 0
#pragma omp parallel
        {
            std::vector<int> mine(R, G);
#pragma omp for schedule(static)
            for (int g = 1; g < G; ++g) {
                ArrayXd d; Params pr;
                point(tau_max * g / (G - 1), d, pr);
                const std::vector<uint64_t> s = pcm_events::frontier_signature(d, mk, pr, 1);
                for (int i = 0; i < R; ++i) if (s[i] != sig0[i] && g < mine[i]) mine[i] = g;
            }
#pragma omp critical
            for (int i = 0; i < R; ++i) first[i] = std::min(first[i], mine[i]);
        }
        const double step = tau_max / (G - 1);
        for (int i = 0; i < R; ++i) {
            if (first[i] == G) {
                ++n_none;
                EXPECT_EQ(ev.kind_draw[i], 0) << "seed " << seed << " draw " << i;
                EXPECT_EQ(ev.tau_draw[i], tau_max);
                continue;
            }
            ++n_events;
            const double lo = step * (first[i] - 1), hi = step * first[i];
            EXPECT_GE(ev.tau_draw[i], lo - 1e-12) << "seed " << seed << " draw " << i;
            EXPECT_LE(ev.tau_draw[i], hi + 1e-12) << "seed " << seed << " draw " << i;
            ASSERT_NE(ev.kind_draw[i], 0);
            const Market m1 = one_draw(mk, i);
            ArrayXd d; Params pr;
            const std::set<int> before = frontier_set(delta, m1, prm);
            point(hi, d, pr);
            const std::set<int> after = frontier_set(d, m1, pr);
            std::set<int> diff;
            std::set_symmetric_difference(before.begin(), before.end(), after.begin(), after.end(),
                                          std::inserter(diff, diff.begin()));
            EXPECT_EQ(diff, std::set<int>{ev.product_draw[i]}) << "seed " << seed << " draw " << i;
            if (ev.kind_draw[i] == 2) { ++n_enter; EXPECT_TRUE(after.count(ev.product_draw[i])); }
            else                      { ++n_leave; EXPECT_TRUE(before.count(ev.product_draw[i])); }
        }
        // hull agrees with the vertical model's positive-share set (y / c_i, as pcm_demand::shares uses)
        for (int g = 0; g <= 20; ++g) {
            ArrayXd d; Params pr;
            point(tau_max * g / 20.0, d, pr);
            for (int i = 0; i < R; ++i) {
                const Market m1 = one_draw(mk, i);
                Eigen::VectorXd t = Eigen::VectorXd::Zero(K);
                for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += pr.theta[m] * mk.taste_basis(i, m);
                const double c = std::exp((mk.scale_basis.row(i).array() * pr.phi.transpose()).sum());
                const ArrayXd dc = (d + (mk.x * t).array()) / c;
                const std::vector<int> ind = index_positive_shares(dc, mk.p);
                EXPECT_EQ(frontier_set(d, m1, pr), std::set<int>(ind.begin(), ind.end()));
            }
        }
    }
    std::printf("[ brute    ] draws with an event %d (enter %d, leave %d), without %d\n", n_events, n_enter, n_leave, n_none);
    EXPECT_GT(n_enter, 10);
    EXPECT_GT(n_leave, 10);
    EXPECT_GT(n_none, 5);
}

// Results do not depend on the thread count.
TEST(PcmEvents, ThreadCountInvariance) {
    std::mt19937_64 rng(11);
    const Market mk = random_market(rng, 40, 3000, 4, 4, 2);
    const Params prm{vec({0.5, 1.0, -0.3, 0.2}), vec({0.1, -0.1}), 1.0};
    const ArrayXd delta = concave_delta(rng, mk.p, 1.0);
    std::normal_distribution<double> nd(0.0, 1.0);
    ArrayXd dd(40);
    for (int j = 0; j < 40; ++j) dd[j] = nd(rng);
    const ArrayXd dth = vec({0.3, -0.2, 0.1, 0.4, 0, 0, 0});
    const auto a = pcm_events::max_step(delta, dd, mk, prm, dth, 1.5, 1);
    const auto b = pcm_events::max_step(delta, dd, mk, prm, dth, 1.5, 8);
    EXPECT_TRUE((a.tau_draw == b.tau_draw).all());
    EXPECT_TRUE((a.kind_draw == b.kind_draw).all());
    EXPECT_TRUE((a.product_draw == b.product_draw).all());
    EXPECT_TRUE(a.mass_profile == b.mass_profile);
    EXPECT_TRUE((a.buyer_mass == b.buyer_mass).all());
    EXPECT_EQ(a.tau_vanish_lb, b.tau_vanish_lb);
    EXPECT_EQ(pcm_events::frontier_signature(delta, mk, prm, 1), pcm_events::frontier_signature(delta, mk, prm, 8));
    EXPECT_GT(a.tau_vanish_lb, a.tau_first);
}

TEST(PcmEvents, InvalidInput) {
    const Market mk = vertical_market(vec({10, 12}));
    EXPECT_THROW(pcm_events::max_step(vec({1, 2}), vec({0}), mk, no_params(), vec({0}), 1.0, 1), std::invalid_argument);
    EXPECT_THROW(pcm_events::max_step(vec({1, 2}), vec({0, 0}), mk, no_params(), vec({0, 0}), 1.0, 1), std::invalid_argument);
    EXPECT_THROW(pcm_events::max_step(vec({1, 2}), vec({0, 0}), mk, no_params(), vec({0}), 0.0, 1), std::invalid_argument);
    const Market bad = vertical_market(vec({12, 10}));
    EXPECT_THROW(pcm_events::frontier_signature(vec({1, 2}), bad, no_params(), 1), std::invalid_argument);
}

// Performance acceptance: max_step on 210 products x 16,000 draws costs under 2x shares(..., false).
TEST(PcmEvents, PerformanceVersusShares) {
    const int N = 210, R = 16000, K = 6, M = 8, Q = 2;
    std::mt19937_64 rng(2026);
    const Market mk = random_market(rng, N, R, K, M, Q);
    ArrayXd theta(M);
    for (int m = 0; m < M; ++m) theta[m] = 0.5 + 0.1 * m;
    const Params prm{theta, vec({0.3, -0.2}), 1.0};
    const ArrayXd delta = concave_delta(rng, mk.p, 0.5);
    std::normal_distribution<double> nd(0.0, 1.0);
    ArrayXd dd(N);
    for (int j = 0; j < N; ++j) dd[j] = nd(rng);
    ArrayXd dth = ArrayXd::Zero(M + Q + 1);
    for (int m = 0; m < M; ++m) dth[m] = nd(rng);
    using clk = std::chrono::steady_clock;
    const int saved = omp_get_max_threads();
    for (int T : {1, 16}) {
        omp_set_num_threads(T);   // team size for pcm_demand::shares
        double t_sh = 1e300, t_ms = 1e300;
        double frontier_sum = 0.0;
        for (int rep = 0; rep < 5; ++rep) {
            auto t0 = clk::now();
            const auto sh = pcm_demand::shares(delta, mk, prm, false);
            auto t1 = clk::now();
            const auto ev = pcm_events::max_step(delta, dd, mk, prm, dth, 1.0, T);
            auto t2 = clk::now();
            t_sh = std::min(t_sh, std::chrono::duration<double>(t1 - t0).count());
            t_ms = std::min(t_ms, std::chrono::duration<double>(t2 - t1).count());
            frontier_sum = ev.buyer_mass.sum();
            ASSERT_GT(sh.s.sum(), 0.0);
        }
        std::printf("[ perf     ] threads %2d: shares(false) %.4f s, max_step %.4f s, ratio %.3f, mean |F_i| %.2f\n",
                    T, t_sh, t_ms, t_ms / t_sh, frontier_sum);
        EXPECT_LT(t_ms, 2.0 * t_sh) << "threads " << T;
    }
    omp_set_num_threads(saved);
}
