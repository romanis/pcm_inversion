// Exact second choices (GMY spec WP-G6): hand cases, row sums, an independent remove-and-resolve check, the adjoint
// against central differences, thread-count invariance, input checks.
#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include "pcm_market_share.hpp"
#include "pcm_second_choice.hpp"

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

double F(double a, double sigma) { return 0.5 * std::erfc(-std::log(a) / (sigma * std::sqrt(2.0))); }

// one draw, no heterogeneity: conditional qualities = delta
pcm_demand::Market one_draw(const ArrayXd & p) {
    pcm_demand::Market mk;
    mk.x = MatrixXd::Ones(p.size(), 1);
    mk.p = p;
    mk.weights = ArrayXd::Ones(1);
    mk.taste_basis = MatrixXd(1, 0);
    mk.scale_basis = MatrixXd(1, 0);
    return mk;
}

pcm_demand::Params params(double sigma, double lo = 0.0, pcm_demand::BoundKind kind = pcm_demand::BoundKind::A) {
    pcm_demand::Params prm;
    prm.theta = ArrayXd(0);
    prm.phi = ArrayXd(0);
    prm.sigma_p = sigma;
    prm.lo = lo;
    prm.bound_kind = kind;
    return prm;
}

struct Fixture {
    pcm_demand::Market mk;
    pcm_demand::Params prm;
    ArrayXd delta;
};

// N products, x = [1, z, binary], 2 taste parameters, 1 scale parameter, R draws
Fixture random_market(unsigned seed, pcm_demand::BoundKind kind, double lo, int N = 20, int R = 50) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    Fixture f;
    f.mk.x = MatrixXd(N, 3);
    f.mk.p = ArrayXd(N);
    f.delta = ArrayXd(N);
    for (int j = 0; j < N; ++j) {
        f.mk.x(j, 0) = 1.0;
        f.mk.x(j, 1) = nd(gen);
        f.mk.x(j, 2) = ud(gen) < 0.5 ? 1.0 : 0.0;
        f.mk.p[j] = 1.0 + 0.2 * j + 0.05 * ud(gen);
        f.delta[j] = 1.6 * f.mk.p[j] - 0.08 * f.mk.p[j] * f.mk.p[j] - 0.3 + 0.05 * nd(gen);
    }
    f.mk.weights = ArrayXd(R);
    f.mk.taste_basis = MatrixXd(R, 2);
    f.mk.taste_col = {1, 2};
    f.mk.scale_basis = MatrixXd(R, 1);
    for (int i = 0; i < R; ++i) {
        f.mk.weights[i] = (0.5 + ud(gen)) / R;
        f.mk.taste_basis(i, 0) = nd(gen);
        f.mk.taste_basis(i, 1) = nd(gen);
        f.mk.scale_basis(i, 0) = 0.5 * nd(gen);
    }
    f.prm.theta = ArrayXd(2);
    f.prm.theta << 0.3, 0.2;
    f.prm.phi = ArrayXd(1);
    f.prm.phi << 0.4;
    f.prm.sigma_p = 1.4;
    f.prm.lo = lo;
    f.prm.bound_kind = kind;
    return f;
}

}  // namespace

TEST(SecondChoice, MiddleProductRemoved) {
    ArrayXd p(3), d(3);
    p << 1.0, 2.0, 3.0;
    d << 2.0, 3.5, 4.2;
    for (double sigma : {0.5, 2.5}) {
        const auto sc = pcm_kernel::second_choice(d, one_draw(p), params(sigma), nullptr);
        const double den = F(1.5, sigma) - F(0.7, sigma);
        EXPECT_NEAR(sc.s[1], den, 1e-14);
        EXPECT_NEAR(sc.C(1, 0), (F(1.5, sigma) - F(1.1, sigma)) / den, 1e-12);
        EXPECT_NEAR(sc.C(1, 2), (F(1.1, sigma) - F(0.7, sigma)) / den, 1e-12);
        EXPECT_EQ(sc.C(1, 1), 0.0);
        EXPECT_NEAR(sc.C(0, 1), 1.0, 1e-14);   // cheapest: a in (1.75, 2) buys nothing, (1.5, 1.75) buys product 2
        EXPECT_NEAR(sc.C(2, 1), 1.0, 1e-14);   // dearest: only product 2 is left above the outside good
        EXPECT_EQ(sc.draws_with_no_inside_second, 0u);
    }
}

TEST(SecondChoice, CheapestProductRemoved) {
    ArrayXd p(2), d(2);
    p << 1.0, 2.0;
    d << 1.5, 2.2;
    const double sigma = 1.0;
    const auto sc = pcm_kernel::second_choice(d, one_draw(p), params(sigma), nullptr);
    EXPECT_NEAR(sc.C(0, 1), 1.0, 1e-14);
    EXPECT_NEAR(sc.N(0, 1), sc.s[0], 1e-15);   // per-draw conditioning: the buyers of a in (1.1, 1.5) are dropped
    EXPECT_NEAR(sc.s[0], F(1.5, sigma) - F(0.7, sigma), 1e-14);
}

TEST(SecondChoice, BoundCutsTheWindows) {
    ArrayXd p(3), d(3);
    p << 1.0, 2.0, 3.0;
    d << 2.0, 3.5, 4.2;
    auto sc = pcm_kernel::second_choice(d, one_draw(p), params(1.5, 1.2), nullptr);
    EXPECT_GT(sc.s[1], 0.0);
    EXPECT_NEAR(sc.C(1, 0), 1.0, 1e-14);
    EXPECT_EQ(sc.C(1, 2), 0.0);
    sc = pcm_kernel::second_choice(d, one_draw(p), params(1.5, 1.6), nullptr);
    EXPECT_EQ(sc.s[1], 0.0);
    EXPECT_EQ(sc.C.row(1).norm(), 0.0);
    // one product: its buyers have no inside second choice
    ArrayXd p1(1), d1(1);
    p1 << 1.0;
    d1 << 1.0;
    sc = pcm_kernel::second_choice(d1, one_draw(p1), params(1.0), nullptr);
    EXPECT_GT(sc.s[0], 0.0);
    EXPECT_EQ(sc.C(0, 0), 0.0);
    EXPECT_EQ(sc.draws_with_no_inside_second, 1u);
}

TEST(SecondChoice, RowSumsAndRemoveAndResolve) {
    for (auto kind : {pcm_demand::BoundKind::A, pcm_demand::BoundKind::ALPHA}) {
        const Fixture f = random_market(11, kind, 0.2);
        const auto sc = pcm_kernel::second_choice(f.delta, f.mk, f.prm, nullptr);
        const int N = static_cast<int>(f.delta.size());
        // independent check: per draw, the frontier without j by index_positive_shares on the reduced product set
        MatrixXd Nref = MatrixXd::Zero(N, N);
        ArrayXd sref = ArrayXd::Zero(N);
        for (int i = 0; i < f.mk.weights.size(); ++i) {
            VectorXd t = VectorXd::Zero(3);
            for (int m = 0; m < 2; ++m) t[f.mk.taste_col[m]] += f.prm.theta[m] * f.mk.taste_basis(i, m);
            const double c = std::exp(f.prm.phi[0] * f.mk.scale_basis(i, 0));
            const ArrayXd dc = (f.delta + (f.mk.x * t).array()) / c;
            const double a_lo = pcm_demand::draw_a_lo(f.prm, c);
            const auto tr = pcm_share::detail::make_trunc(a_lo, f.prm.sigma_p);
            auto FT = [&](double e) { return pcm_share::detail::ln_terms(e, f.prm.sigma_p, tr).F; };
            ArrayXd s = ArrayXd::Zero(N);
            pcm_share::cond_share(dc, f.mk.p, f.prm.sigma_p, a_lo, s, nullptr, nullptr, nullptr, nullptr);
            sref += f.mk.weights[i] * s;
            const std::vector<int> full = index_positive_shares(dc, f.mk.p);
            for (size_t r = 0; r < full.size(); ++r) {
                const int j = full[r];
                if (!(s[j] > 0)) continue;
                const double hi = r == 0 ? dc[j] / f.mk.p[j]
                                         : (dc[j] - dc[full[r - 1]]) / (f.mk.p[j] - f.mk.p[full[r - 1]]);
                const double lo = r + 1 < full.size()
                    ? (dc[full[r + 1]] - dc[j]) / (f.mk.p[full[r + 1]] - f.mk.p[j]) : 0.0;
                std::vector<int> keep;
                for (int k = 0; k < N; ++k) if (k != j) keep.push_back(k);
                const ArrayXd d2 = dc(keep), p2 = f.mk.p(keep);
                const std::vector<int> red = index_positive_shares(d2, p2);
                std::vector<double> m(N, 0.0);
                double M = 0.0;
                for (size_t u = 0; u < red.size(); ++u) {
                    const double top = u == 0 ? d2[red[0]] / p2[red[0]]
                                              : (d2[red[u]] - d2[red[u - 1]]) / (p2[red[u]] - p2[red[u - 1]]);
                    const double bot = u + 1 < red.size()
                        ? (d2[red[u + 1]] - d2[red[u]]) / (p2[red[u + 1]] - p2[red[u]]) : 0.0;
                    const double mu = FT(std::min(hi, top)) - FT(std::max(lo, bot));
                    if (mu > 0) { m[keep[red[u]]] = mu; M += mu; }
                }
                if (M > 0)
                    for (int k = 0; k < N; ++k) Nref(j, k) += f.mk.weights[i] * s[j] * m[k] / M;
            }
        }
        for (int j = 0; j < N; ++j) {
            EXPECT_NEAR(sc.s[j], sref[j], 1e-15);
            for (int k = 0; k < N; ++k) EXPECT_NEAR(sc.N(j, k), Nref(j, k), 1e-13) << j << " " << k;
            if (sc.s[j] > 0 && sc.draws_with_no_inside_second == 0) {
                EXPECT_NEAR(sc.C.row(j).sum(), 1.0, 1e-12);
            }
            EXPECT_LE(sc.C.row(j).sum(), 1.0 + 1e-12);
        }
    }
}

TEST(SecondChoice, AdjointMatchesCentralDifferences) {
    // the third case has a high bound, so some buyers of a product have no inside second choice: they add to s_j
    // but not to N, and their -B ds_j term must stay in the adjoint
    struct Case { pcm_demand::BoundKind kind; double lo; bool none; };
    double lo_none = 0.0;   // the smallest of these bounds with such draws on the fixture market
    for (double lo : {0.4, 0.6, 0.8, 1.0, 1.2, 1.4}) {
        const Fixture g = random_market(12, pcm_demand::BoundKind::A, lo);
        if (pcm_kernel::second_choice(g.delta, g.mk, g.prm, nullptr).draws_with_no_inside_second > 0) {
            lo_none = lo;
            break;
        }
    }
    ASSERT_GT(lo_none, 0.0) << "no bound gives a draw without an inside second choice";
    for (const Case cs : {Case{pcm_demand::BoundKind::A, 0.2, false}, Case{pcm_demand::BoundKind::ALPHA, 0.2, false},
                          Case{pcm_demand::BoundKind::A, lo_none, true}}) {
        const Fixture f = random_market(12, cs.kind, cs.lo);
        const int N = static_cast<int>(f.delta.size());
        const int K = 3;
        if (cs.none) {
            EXPECT_GT(pcm_kernel::second_choice(f.delta, f.mk, f.prm, nullptr).draws_with_no_inside_second, 0u);
        }
        std::mt19937 gen(5);
        std::normal_distribution<double> nd(0.0, 1.0);
        pcm_kernel::SecondChoiceAdjoint adj;
        for (int c = 0; c < K; ++c) {
            MatrixXd A(N, N);
            for (int j = 0; j < N; ++j) for (int k = 0; k < N; ++k) A(j, k) = nd(gen);
            adj.A.push_back(A);
        }
        adj.B = MatrixXd(K, N);
        for (int c = 0; c < K; ++c) for (int j = 0; j < N; ++j) adj.B(c, j) = nd(gen);
        auto value = [&](const ArrayXd & d, const pcm_demand::Params & prm) {
            const auto sc = pcm_kernel::second_choice(d, f.mk, prm, nullptr);
            VectorXd v(K);
            for (int c = 0; c < K; ++c) v[c] = (adj.A[c].array() * sc.N.array()).sum() - adj.B.row(c).dot(sc.s.matrix());
            return v;
        };
        const auto sc = pcm_kernel::second_choice(f.delta, f.mk, f.prm, &adj);
        const double h = 1e-7;
        for (int l = 0; l < N; ++l) {
            ArrayXd a = f.delta, b = f.delta;
            a[l] += h;
            b[l] -= h;
            const VectorXd fd = (value(a, f.prm) - value(b, f.prm)) / (2 * h);
            for (int c = 0; c < K; ++c)
                EXPECT_NEAR(sc.out_delta(c, l), fd[c], 1e-6 * (1.0 + std::abs(fd[c]))) << c << " " << l;
        }
        for (int k = 0; k < 4; ++k) {
            pcm_demand::Params a = f.prm, b = f.prm;
            auto bump = [&](pcm_demand::Params & q, double s) {
                if (k < 2) q.theta[k] += s; else if (k < 3) q.phi[0] += s; else q.sigma_p += s;
            };
            bump(a, h);
            bump(b, -h);
            const VectorXd fd = (value(f.delta, a) - value(f.delta, b)) / (2 * h);
            for (int c = 0; c < K; ++c)
                EXPECT_NEAR(sc.out_theta(c, k), fd[c], 1e-6 * (1.0 + std::abs(fd[c]))) << c << " " << k;
        }
    }
}

TEST(SecondChoice, SameForEveryThreadCountAndBadInput) {
    const Fixture f = random_market(13, pcm_demand::BoundKind::A, 0.2, 20, 700);   // 11 blocks
    pcm_kernel::SecondChoiceAdjoint adj;
    adj.A.assign(2, MatrixXd::Ones(20, 20));
    adj.B = MatrixXd::Ones(2, 20);
    omp_set_num_threads(1);
    const auto a = pcm_kernel::second_choice(f.delta, f.mk, f.prm, &adj);
    omp_set_num_threads(8);
    const auto b = pcm_kernel::second_choice(f.delta, f.mk, f.prm, &adj);
    EXPECT_TRUE(a.C == b.C && a.s.matrix() == b.s.matrix() && a.out_delta == b.out_delta && a.out_theta == b.out_theta);
    pcm_kernel::SecondChoiceAdjoint bad;
    bad.A.assign(2, MatrixXd::Ones(3, 3));
    bad.B = MatrixXd::Ones(2, 20);
    EXPECT_THROW(pcm_kernel::second_choice(f.delta, f.mk, f.prm, &bad), std::invalid_argument);
    ArrayXd nan_delta = f.delta;
    nan_delta[0] = std::nan("");
    EXPECT_THROW(pcm_kernel::second_choice(nan_delta, f.mk, f.prm, nullptr), std::invalid_argument);
}
