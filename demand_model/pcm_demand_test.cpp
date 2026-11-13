#include <gtest/gtest.h>
#include <random>
#include "pcm_demand.hpp"
#include "pcm_market_share.hpp"

using Eigen::ArrayXd;
using Eigen::MatrixXd;

namespace {

// 12 products, characteristics [1, z1, z2], 400 draws of (nu x 3, two demographics)
struct Fixture {
    pcm_demand::Market mk;
    pcm_demand::Params prm;
    ArrayXd delta;
};

Fixture make_fixture(unsigned seed, int N = 12, int R = 400) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    Fixture f;
    f.mk.x = MatrixXd(N, 3);
    f.mk.p = ArrayXd(N);
    f.delta = ArrayXd(N);
    for (int j = 0; j < N; ++j) {
        f.mk.x(j, 0) = 1.0;
        f.mk.x(j, 1) = ud(gen) * 10.0;
        f.mk.x(j, 2) = ud(gen) < 0.5 ? 1.0 : 0.0;
        f.mk.p[j] = 0.05 + 0.01 * j + 0.003 * ud(gen);
        // concave in price so every product is on the vertical frontier
        f.delta[j] = 2.0 * f.mk.p[j] - 5.0 * f.mk.p[j] * f.mk.p[j] + 0.0005 * nd(gen);
    }
    f.mk.weights = ArrayXd::Constant(R, 1.0 / R);
    // params: sigma on 3 columns (basis nu), pi on column 0 and 1 for demo 0 (basis D0)
    f.mk.taste_basis = MatrixXd(R, 5);
    f.mk.taste_col = {0, 1, 2, 0, 1};
    f.mk.scale_basis = MatrixXd(R, 2);
    for (int i = 0; i < R; ++i) {
        const double d0 = nd(gen), d1 = ud(gen) < 0.3 ? 1.0 : 0.0;
        for (int m = 0; m < 3; ++m) f.mk.taste_basis(i, m) = nd(gen);
        f.mk.taste_basis(i, 3) = d0;
        f.mk.taste_basis(i, 4) = d0;
        f.mk.scale_basis(i, 0) = d0;
        f.mk.scale_basis(i, 1) = d1;
    }
    f.prm.theta = ArrayXd(5);
    f.prm.theta << 0.02, 0.002, 0.01, 0.01, -0.001;
    f.prm.phi = ArrayXd(2);
    f.prm.phi << 0.3, -0.2;
    f.prm.sigma_p = 0.6;
    return f;
}

double rel_err(const MatrixXd & a, const MatrixXd & b) {
    return (a - b).cwiseAbs().maxCoeff() / std::max(1e-12, b.cwiseAbs().maxCoeff());
}

}  // namespace

TEST(PCMDemand, SharesMatchExistingLibraryWithoutDemographics) {
    // no scale heterogeneity, taste = sigma_x * grid: must equal pcm_share::unc_share
    Fixture f = make_fixture(1);
    const int R = f.mk.weights.size();
    ArrayXd sigma_x(3); sigma_x << 0.02, 0.002, 0.01;
    Eigen::ArrayXXd grid = f.mk.taste_basis.leftCols(3).array();
    pcm_demand::Market mk = f.mk;
    mk.taste_basis = f.mk.taste_basis.leftCols(3);
    mk.taste_col = {0, 1, 2};
    mk.scale_basis = MatrixXd(R, 0);
    pcm_demand::Params prm{sigma_x, ArrayXd(0), f.prm.sigma_p};
    MatrixXd J_old;
    ArrayXd s_old = pcm_share::unc_share(f.delta, f.mk.x, f.mk.p, f.prm.sigma_p, ArrayXd::Ones(3),
                                         (grid.rowwise() * sigma_x.transpose()).eval(), f.mk.weights, J_old);
    auto sh = pcm_demand::shares(f.delta, mk, prm, true);
    EXPECT_LT((sh.s - s_old).abs().maxCoeff(), 1e-13);
    EXPECT_LT(rel_err(sh.d_delta, J_old), 1e-10);
}

TEST(PCMDemand, DerivativesMatchFiniteDifferences) {
    Fixture f = make_fixture(2);
    auto sh = pcm_demand::shares(f.delta, f.mk, f.prm, true);
    ASSERT_TRUE((sh.s > 0).all()) << "fixture should put most products on the frontier";
    const int N = f.delta.size();
    const double h = 1e-7;

    MatrixXd fd_delta(N, N), fd_price(N, N), fd_theta(N, sh.d_theta.cols());
    for (int j = 0; j < N; ++j) {
        ArrayXd dp = f.delta, dm = f.delta; dp[j] += h; dm[j] -= h;
        fd_delta.col(j) = (pcm_demand::shares(dp, f.mk, f.prm, false).s - pcm_demand::shares(dm, f.mk, f.prm, false).s).matrix() / (2 * h);
        pcm_demand::Market up = f.mk, dn = f.mk; up.p[j] += h; dn.p[j] -= h;
        fd_price.col(j) = (pcm_demand::shares(f.delta, up, f.prm, false).s - pcm_demand::shares(f.delta, dn, f.prm, false).s).matrix() / (2 * h);
    }
    const int M = f.prm.theta.size(), Q = f.prm.phi.size();
    for (int k = 0; k < M + Q + 1; ++k) {
        pcm_demand::Params up = f.prm, dn = f.prm;
        if (k < M) { up.theta[k] += h; dn.theta[k] -= h; }
        else if (k < M + Q) { up.phi[k - M] += h; dn.phi[k - M] -= h; }
        else { up.sigma_p += h; dn.sigma_p -= h; }
        fd_theta.col(k) = (pcm_demand::shares(f.delta, f.mk, up, false).s - pcm_demand::shares(f.delta, f.mk, dn, false).s).matrix() / (2 * h);
    }
    EXPECT_LT(rel_err(sh.d_delta, fd_delta), 1e-4);
    EXPECT_LT(rel_err(sh.d_price, fd_price), 1e-4);
    for (int k = 0; k < M + Q + 1; ++k)
        EXPECT_LT(rel_err(sh.d_theta.col(k), fd_theta.col(k)), 1e-4) << "parameter column " << k;
}

TEST(PCMDemand, InversionRecoversDelta) {
    Fixture f = make_fixture(3);
    ArrayXd s = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    ASSERT_TRUE((s > 0).all());
    auto res = pcm_demand::invert(s, f.mk, f.prm, ArrayXd());
    EXPECT_TRUE(res.converged);
    EXPECT_LT(res.max_abs_residual, 1e-12);
    EXPECT_LT((res.delta - f.delta).abs().maxCoeff(), 1e-8);
}

TEST(PCMDemand, InversionWarmStartTakesFewEvaluations) {
    Fixture f = make_fixture(4);
    ArrayXd s = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    auto cold = pcm_demand::invert(s, f.mk, f.prm, ArrayXd());
    auto warm = pcm_demand::invert(s, f.mk, f.prm, (f.delta + 1e-4).eval());
    EXPECT_TRUE(warm.converged);
    EXPECT_LE(warm.share_evaluations, cold.share_evaluations);
}

TEST(PCMDemand, RejectsUnsortedPrices) {
    Fixture f = make_fixture(5);
    std::swap(f.mk.p[2], f.mk.p[3]);
    EXPECT_THROW(pcm_demand::shares(f.delta, f.mk, f.prm, false), std::runtime_error);
}
