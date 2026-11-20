// Per-household sums (GMY spec WP-G5): equality with evaluate_market on each household's draws, sums over
// households, derivatives against central differences, thread-count invariance, input checks.
#include <gtest/gtest.h>
#include <omp.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include "pcm_groups.hpp"
#include "pcm_kernel.hpp"

using Eigen::ArrayXd;
using Eigen::ArrayXi;
using Eigen::MatrixXd;

namespace {

struct Fixture {
    pcm_demand::Market mk;
    pcm_demand::Params prm;
    ArrayXd delta;
    pcm_kernel::GroupRequest gr;
};

// 10 products, x = [1, z, binary], 2 taste parameters, 1 scale parameter, 5 households of unequal draw counts.
Fixture make(unsigned seed, pcm_demand::BoundKind kind, double lo) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const int N = 10;
    const std::vector<int> sizes = {7, 12, 3, 70, 9};   // 70 > 64: two blocks for that household
    int R = 0;
    for (int s : sizes) R += s;
    Fixture f;
    f.mk.x = MatrixXd(N, 3);
    f.mk.p = ArrayXd(N);
    f.delta = ArrayXd(N);
    for (int j = 0; j < N; ++j) {
        f.mk.x(j, 0) = 1.0;
        f.mk.x(j, 1) = nd(gen);
        f.mk.x(j, 2) = ud(gen) < 0.5 ? 1.0 : 0.0;
        f.mk.p[j] = 1.0 + 0.4 * j + 0.05 * ud(gen);
        f.delta[j] = 1.8 * f.mk.p[j] - 0.15 * f.mk.p[j] * f.mk.p[j] - 0.3 + 0.002 * nd(gen);
    }
    f.mk.weights = ArrayXd(R);
    f.mk.taste_basis = MatrixXd(R, 2);
    f.mk.taste_col = {1, 2};
    f.mk.scale_basis = MatrixXd(R, 1);
    f.gr.group_of_draw = ArrayXi(R);
    int i = 0;
    for (int g = 0; g < static_cast<int>(sizes.size()); ++g) {
        const double demo = nd(gen);
        for (int k = 0; k < sizes[g]; ++k, ++i) {
            f.gr.group_of_draw[i] = g;
            f.mk.weights[i] = (0.5 + ud(gen)) / R;
            f.mk.taste_basis(i, 0) = nd(gen);
            f.mk.taste_basis(i, 1) = nd(gen);
            f.mk.scale_basis(i, 0) = demo;
        }
    }
    f.gr.n_groups = static_cast<int>(sizes.size());
    f.gr.U = MatrixXd(N, 3);
    f.gr.U.col(0) = f.mk.p.matrix();
    f.gr.U.col(1) = f.mk.x.col(1);
    f.gr.U.col(2) = f.mk.x.col(2);
    f.prm.theta = ArrayXd(2);
    f.prm.theta << 0.15, 0.1;
    f.prm.phi = ArrayXd(1);
    f.prm.phi << 0.3;
    f.prm.sigma_p = 1.3;
    f.prm.lo = lo;
    f.prm.bound_kind = kind;
    return f;
}

// the draws of group g as a market of their own
pcm_demand::Market sub_market(const Fixture & f, int g) {
    std::vector<int> idx;
    for (int i = 0; i < f.gr.group_of_draw.size(); ++i)
        if (f.gr.group_of_draw[i] == g) idx.push_back(i);
    pcm_demand::Market m;
    m.x = f.mk.x;
    m.p = f.mk.p;
    m.weights = f.mk.weights(idx);
    m.taste_basis = f.mk.taste_basis(idx, Eigen::all);
    m.taste_col = f.mk.taste_col;
    m.scale_basis = f.mk.scale_basis(idx, Eigen::all);
    return m;
}

pcm_kernel::MicroRequest micro_ones(const Fixture & f, int R) {
    pcm_kernel::MicroRequest mr;
    const int N = static_cast<int>(f.mk.p.size());
    mr.H = MatrixXd::Ones(R, 1);
    mr.U = MatrixXd(N, f.gr.U.cols() + 1);
    mr.U.col(0).setOnes();
    mr.U.rightCols(f.gr.U.cols()) = f.gr.U;
    mr.h_of_part.assign(mr.U.cols(), 0);
    return mr;
}

struct Case { pcm_demand::BoundKind kind; double lo; };

class GroupSums : public ::testing::TestWithParam<Case> {};

}  // namespace

TEST_P(GroupSums, EqualsEvaluateMarketPerHousehold) {
    const Fixture f = make(3, GetParam().kind, GetParam().lo);
    const auto gs = pcm_kernel::group_sums(f.delta, f.mk, f.prm, f.gr, true);
    for (int g = 0; g < f.gr.n_groups; ++g) {
        const pcm_demand::Market m = sub_market(f, g);
        const auto mr = micro_ones(f, static_cast<int>(m.weights.size()));
        const auto ev = pcm_kernel::evaluate_market(f.delta, m, f.prm, &mr, true, false);
        for (int c = 0; c < gs.S.cols(); ++c) EXPECT_EQ(gs.S(g, c), ev.micro[c]) << "group " << g << " col " << c;
        EXPECT_TRUE(gs.d_delta[g] == ev.micro_d_delta) << "group " << g;
        EXPECT_TRUE(gs.d_theta[g] == ev.micro_d_theta) << "group " << g;
        EXPECT_DOUBLE_EQ(gs.w[g], m.weights.sum());   // a plain sum; Eigen sums in another order
    }
}

TEST_P(GroupSums, SumOverHouseholdsIsTheMarket) {
    const Fixture f = make(4, GetParam().kind, GetParam().lo);
    const auto gs = pcm_kernel::group_sums(f.delta, f.mk, f.prm, f.gr, false);
    const auto ev = pcm_kernel::evaluate_market(f.delta, f.mk, f.prm, nullptr, false, false);
    EXPECT_NEAR(gs.S.col(0).sum(), ev.s.sum(), 1e-14);
    for (int c = 0; c < f.gr.U.cols(); ++c)
        EXPECT_NEAR(gs.S.col(1 + c).sum(), f.gr.U.col(c).dot(ev.s.matrix()), 1e-13);
    EXPECT_TRUE(gs.d_delta.empty() && gs.d_theta.empty());
}

TEST_P(GroupSums, DerivativesMatchCentralDifferences) {
    Fixture f = make(5, GetParam().kind, GetParam().lo);
    const auto gs = pcm_kernel::group_sums(f.delta, f.mk, f.prm, f.gr, true);
    const double h = 1e-6;
    const int N = static_cast<int>(f.delta.size());
    for (int l = 0; l < N; ++l) {
        Fixture a = f, b = f;
        a.delta[l] += h;
        b.delta[l] -= h;
        const MatrixXd fd = (pcm_kernel::group_sums(a.delta, a.mk, a.prm, a.gr, false).S
                             - pcm_kernel::group_sums(b.delta, b.mk, b.prm, b.gr, false).S) / (2 * h);
        for (int g = 0; g < f.gr.n_groups; ++g)
            for (int c = 0; c < fd.cols(); ++c)
                EXPECT_NEAR(gs.d_delta[g](c, l), fd(g, c), 1e-6 * (1.0 + std::abs(fd(g, c)))) << g << " " << c << " " << l;
    }
    const int P = static_cast<int>(f.prm.theta.size() + f.prm.phi.size() + 1);
    for (int k = 0; k < P; ++k) {
        Fixture a = f, b = f;
        auto bump = [&](pcm_demand::Params & q, double s) {
            if (k < 2) q.theta[k] += s; else if (k < 3) q.phi[k - 2] += s; else q.sigma_p += s;
        };
        bump(a.prm, h);
        bump(b.prm, -h);
        const MatrixXd fd = (pcm_kernel::group_sums(a.delta, a.mk, a.prm, a.gr, false).S
                             - pcm_kernel::group_sums(b.delta, b.mk, b.prm, b.gr, false).S) / (2 * h);
        for (int g = 0; g < f.gr.n_groups; ++g)
            for (int c = 0; c < fd.cols(); ++c)
                EXPECT_NEAR(gs.d_theta[g](c, k), fd(g, c), 1e-6 * (1.0 + std::abs(fd(g, c)))) << g << " " << c << " " << k;
    }
}

TEST_P(GroupSums, SameForEveryThreadCount) {
    const Fixture f = make(6, GetParam().kind, GetParam().lo);
    omp_set_num_threads(1);
    const auto a = pcm_kernel::group_sums(f.delta, f.mk, f.prm, f.gr, true);
    omp_set_num_threads(8);
    const auto b = pcm_kernel::group_sums(f.delta, f.mk, f.prm, f.gr, true);
    EXPECT_TRUE(a.S == b.S);
    for (int g = 0; g < f.gr.n_groups; ++g) {
        EXPECT_TRUE(a.d_delta[g] == b.d_delta[g]);
        EXPECT_TRUE(a.d_theta[g] == b.d_theta[g]);
    }
}

INSTANTIATE_TEST_SUITE_P(Bounds, GroupSums,
                         ::testing::Values(Case{pcm_demand::BoundKind::A, 0.0}, Case{pcm_demand::BoundKind::A, 0.2},
                                           Case{pcm_demand::BoundKind::ALPHA, 0.2}));

TEST(GroupSumsInput, RejectsBadGroups) {
    Fixture f = make(7, pcm_demand::BoundKind::A, 0.2);
    Fixture dec = f;
    dec.gr.group_of_draw[5] = 3;   // 3 then back to 0: decreasing
    EXPECT_THROW(pcm_kernel::group_sums(dec.delta, dec.mk, dec.prm, dec.gr, false), std::invalid_argument);
    Fixture out = f;
    out.gr.group_of_draw[out.gr.group_of_draw.size() - 1] = 9;
    EXPECT_THROW(pcm_kernel::group_sums(out.delta, out.mk, out.prm, out.gr, false), std::invalid_argument);
    Fixture rows = f;
    rows.gr.U = MatrixXd::Ones(3, 2);
    EXPECT_THROW(pcm_kernel::group_sums(rows.delta, rows.mk, rows.prm, rows.gr, false), std::invalid_argument);
}
