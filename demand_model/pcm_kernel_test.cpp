#include <gtest/gtest.h>
#include "pcm_geometry.hpp"
#include <cmath>
#include <random>
#include "pcm_kernel.hpp"
#include "pcm_invert.hpp"
#include "pcm_market_share.hpp"

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using pcm_kernel::Status;

using pcm_geometry::index_positive_shares;   // indices of products on the vertical frontier

namespace {

// The pcm_aggregate_test fixture: 12 products, characteristics [1, z1, z2], R draws of (nu x 3, two
// demographics); 5 taste parameters (sigma on 3 columns, pi on columns 0 and 1), 2 scale parameters.
struct Fixture {
    pcm_aggregate::Market mk;
    pcm_aggregate::Params prm;
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

// Agent weights [1, a demographic in (0, 1)] and product weights [1, e_0, e_5, random].
pcm_kernel::MicroRequest make_micro(const Fixture & f, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    const int R = f.mk.weights.size(), N = f.delta.size();
    pcm_kernel::MicroRequest mr;
    mr.H = MatrixXd(R, 2);
    for (int i = 0; i < R; ++i) { mr.H(i, 0) = 1.0; mr.H(i, 1) = ud(gen); }
    mr.U = MatrixXd::Zero(N, 5);
    mr.U.col(0).setOnes();
    mr.U(0, 1) = 1.0;
    mr.U(5, 2) = 1.0;
    for (int j = 0; j < N; ++j) mr.U(j, 3) = mr.U(j, 4) = ud(gen) - 0.3;
    mr.h_of_part = {0, 0, 0, 0, 1};
    return mr;
}

pcm_aggregate::Params bump(const pcm_aggregate::Params & prm, int k, double h) {
    pcm_aggregate::Params out = prm;
    const int M = prm.theta.size(), Q = prm.phi.size();
    if (k < M) out.theta[k] += h;
    else if (k < M + Q) out.phi[k - M] += h;
    else out.sigma_p += h;
    return out;
}

// Every draw's frontier (products with positive conditional share), as in pcm_aggregate::shares.
std::vector<std::vector<int>> frontiers(const ArrayXd & delta, const pcm_aggregate::Market & mk,
                                        const pcm_aggregate::Params & prm) {
    const int R = mk.weights.size(), M = prm.theta.size(), Q = prm.phi.size();
    std::vector<std::vector<int>> out(R);
    for (int i = 0; i < R; ++i) {
        VectorXd t = VectorXd::Zero(mk.x.cols());
        for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
        const double c = std::exp(Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0);
        out[i] = index_positive_shares((delta + (mk.x * t).array()) / c, mk.p);
    }
    return out;
}

double rel_err(const MatrixXd & a, const MatrixXd & b) {
    return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff());
}

// Exact equality including NaN positions and shapes.
bool same(const MatrixXd & a, const MatrixXd & b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return false;
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        const double x = a.data()[i], y = b.data()[i];
        if (!(x == y || (std::isnan(x) && std::isnan(y)))) return false;
    }
    return true;
}

}  // namespace

// 3.6 kernel test 1: without heterogeneity the vertical-model guess is the exact inverse.
TEST(PcmKernel, NoHeterogeneityReturnsInitialGuess) {
    Fixture f = make_fixture(11);
    pcm_aggregate::Market mk = f.mk;
    const int R = 1;
    mk.weights = ArrayXd::Ones(R);
    mk.taste_basis = MatrixXd(R, 0);
    mk.taste_col = {};
    mk.scale_basis = MatrixXd(R, 0);
    pcm_aggregate::Params prm{ArrayXd(0), ArrayXd(0), 0.6};
    const ArrayXd delta = 2.0 * mk.p - 5.0 * mk.p.square();    // concave: every product on the frontier
    const ArrayXd s_data = pcm_aggregate::shares(delta, mk, prm, false).s;
    ASSERT_TRUE((s_data > 0).all() && s_data.sum() < 1.0);
    const ArrayXd guess = pcm_share::initial_guess(s_data, mk.p, prm.sigma_p);

    pcm_kernel::SolveOptions opt;
    opt.tol = 1e-12;
    auto r = pcm_kernel::solve_market(s_data, mk, prm, ArrayXd(), nullptr, nullptr, opt);
    EXPECT_EQ(r.status, Status::OK) << r.stages;
    EXPECT_LT((r.delta - guess).abs().maxCoeff(), 1e-12);
    EXPECT_LT(r.max_abs_residual, 1e-12);
    EXPECT_EQ(r.stages.rfind("cold ok", 0), 0u) << r.stages;            // first stage converged
    EXPECT_EQ(r.stages.find("warm"), std::string::npos) << r.stages;
    EXPECT_EQ(r.stages.find("path"), std::string::npos) << r.stages;
    EXPECT_EQ(r.ds_dtheta.cols(), 1);                                    // sigma_p only
    EXPECT_TRUE(r.ddelta_dtheta.allFinite());

    // as a warm start the guess is accepted by the warm stage without any Newton step
    auto w = pcm_kernel::solve_market(s_data, mk, prm, guess, nullptr, nullptr, opt);
    EXPECT_EQ(w.status, Status::OK);
    EXPECT_EQ(w.stages.rfind("warm ok", 0), 0u) << w.stages;
    EXPECT_LT((w.delta - guess).abs().maxCoeff(), 1e-12);
}

// 3.6 kernel test 2: ddelta/dtheta against central finite differences of the inversion.
TEST(PcmKernel, DdeltaDthetaMatchesFiniteDifferences) {
    Fixture f = make_fixture(2);
    const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    ASSERT_TRUE((s_data > 0).all());
    pcm_kernel::SolveOptions opt;
    opt.tol = 1e-13;
    auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, ArrayXd(), nullptr, nullptr, opt);
    ASSERT_EQ(r.status, Status::OK) << r.stages;
    EXPECT_LT((r.delta - f.delta).abs().maxCoeff(), 1e-9);

    // same derivatives as pcm_aggregate::shares at the solution
    auto sh = pcm_aggregate::shares(r.delta, f.mk, f.prm, true);
    EXPECT_EQ((r.s - sh.s).abs().maxCoeff(), 0.0);
    EXPECT_LT(rel_err(r.ds_ddelta, sh.d_delta), 1e-13);
    EXPECT_LT(rel_err(r.ds_dtheta, sh.d_theta), 1e-13);
    EXPECT_TRUE(r.ds_dp.size() == 0);

    const int P = f.prm.theta.size() + f.prm.phi.size() + 1;
    ASSERT_EQ(r.ddelta_dtheta.rows(), f.delta.size());
    ASSERT_EQ(r.ddelta_dtheta.cols(), P);
    // Central differences need both inversions on the smooth piece of delta(theta) that holds the
    // solution. Start at h = 1e-5; while some draw's frontier differs between theta - h, theta and
    // theta + h (an exact PCM kink inside the interval), shrink h by 10.
    const auto F0 = frontiers(r.delta, f.mk, f.prm);
    MatrixXd fd(f.delta.size(), P);
    double err = 0.0;
    for (int k = 0; k < P; ++k) {
        double h = 1e-5;
        for (;; h *= 0.1) {
            auto up = pcm_invert::invert(s_data, f.mk, bump(f.prm, k, h), r.delta, 1e-13, 5000);
            auto dn = pcm_invert::invert(s_data, f.mk, bump(f.prm, k, -h), r.delta, 1e-13, 5000);
            ASSERT_TRUE(up.converged && dn.converged) << "parameter " << k;
            fd.col(k) = (up.delta - dn.delta).matrix() / (2 * h);
            const bool smooth = frontiers(up.delta, f.mk, bump(f.prm, k, h)) == F0 &&
                                frontiers(dn.delta, f.mk, bump(f.prm, k, -h)) == F0;
            if (smooth) break;
            ASSERT_GT(h, 1e-8) << "parameter " << k << ": a frontier changes within 1e-8";
        }
        const double e = rel_err(r.ddelta_dtheta.col(k), fd.col(k));
        err = std::max(err, e);
        EXPECT_LT(e, 1e-5) << "parameter " << k << ", h " << h;
        std::printf("  parameter %d: h %.0e, relative error %.3e\n", k, h, e);
    }
    EXPECT_LT(rel_err(r.ddelta_dtheta, fd), 1e-5);
    std::printf("  ddelta_dtheta max relative error vs FD: %.3e\n", err);
}

// 3.6 kernel test 3: micro parts and their partial derivatives.
TEST(PcmKernel, MicroPartsMatchSharesBruteForceAndFiniteDifferences) {
    Fixture f = make_fixture(5);
    const pcm_kernel::MicroRequest mr = make_micro(f, 17);
    const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    pcm_kernel::SolveOptions opt;
    opt.tol = 1e-13;
    auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, f.delta, nullptr, &mr, opt);
    ASSERT_EQ(r.status, Status::OK) << r.stages;
    ASSERT_EQ(r.micro.size(), 5);

    // h = 1, u = 1: inside share sum; h = 1, u = e_j: s_j
    EXPECT_NEAR(r.micro[0], r.s.sum(), 1e-14);
    EXPECT_NEAR(r.micro[1], r.s[0], 1e-15);
    EXPECT_NEAR(r.micro[2], r.s[5], 1e-15);
    EXPECT_NEAR(r.micro[3], (mr.U.col(3).array() * r.s).sum(), 1e-14);
    EXPECT_LT((r.micro_d_delta.row(0).transpose() - r.ds_ddelta.colwise().sum().transpose()).cwiseAbs().maxCoeff(),
              1e-12 * r.ds_ddelta.cwiseAbs().maxCoeff());
    EXPECT_LT((r.micro_d_theta.row(0).transpose() - r.ds_dtheta.colwise().sum().transpose()).cwiseAbs().maxCoeff(),
              1e-12 * r.ds_dtheta.cwiseAbs().maxCoeff());

    // h = demographic: brute force, one single-draw market per draw
    const int R = f.mk.weights.size();
    double brute = 0.0;
    for (int i = 0; i < R; ++i) {
        pcm_aggregate::Market mi = f.mk;
        mi.weights = ArrayXd::Ones(1);
        mi.taste_basis = f.mk.taste_basis.row(i);
        mi.scale_basis = f.mk.scale_basis.row(i);
        const ArrayXd si = pcm_aggregate::shares(r.delta, mi, f.prm, false).s;
        brute += f.mk.weights[i] * mr.H(i, 1) * (mr.U.col(4).array() * si).sum();
    }
    EXPECT_NEAR(r.micro[4], brute, 1e-14);

    // partial derivatives at fixed delta against central differences of evaluate_market
    const int N = f.delta.size(), P = r.micro_d_theta.cols();
    const double h = 1e-7;
    MatrixXd fd_delta(5, N), fd_theta(5, P);
    for (int j = 0; j < N; ++j) {
        ArrayXd dp = r.delta, dm = r.delta;
        dp[j] += h; dm[j] -= h;
        fd_delta.col(j) = (pcm_kernel::evaluate_market(dp, f.mk, f.prm, &mr, false, false).micro -
                           pcm_kernel::evaluate_market(dm, f.mk, f.prm, &mr, false, false).micro) / (2 * h);
    }
    for (int k = 0; k < P; ++k)
        fd_theta.col(k) = (pcm_kernel::evaluate_market(r.delta, f.mk, bump(f.prm, k, h), &mr, false, false).micro -
                           pcm_kernel::evaluate_market(r.delta, f.mk, bump(f.prm, k, -h), &mr, false, false).micro) / (2 * h);
    const double e_delta = rel_err(r.micro_d_delta, fd_delta);
    EXPECT_LT(e_delta, 1e-6);
    double e_theta = 0.0;
    for (int k = 0; k < P; ++k) {
        const double e = rel_err(r.micro_d_theta.col(k), fd_theta.col(k));
        e_theta = std::max(e_theta, e);
        EXPECT_LT(e, 1e-6) << "parameter column " << k;
    }
    std::printf("  micro_d_delta rel error %.3e, micro_d_theta rel error (worst column) %.3e\n", e_delta, e_theta);
}

// 3.6 kernel test 4: bit-identical results with 1 and 8 threads.
TEST(PcmKernel, BitIdenticalAcrossThreadCounts) {
    Fixture f = make_fixture(7, 12, 3000);
    const pcm_kernel::MicroRequest mr = make_micro(f, 3);
    const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    const ArrayXd warm = f.delta + 2e-4;
    pcm_kernel::SolveOptions opt;
    opt.price_derivatives = true;
    opt.n_threads = 1;
    auto a = pcm_kernel::solve_market(s_data, f.mk, f.prm, warm, nullptr, &mr, opt);
    opt.n_threads = 8;
    auto b = pcm_kernel::solve_market(s_data, f.mk, f.prm, warm, nullptr, &mr, opt);
    ASSERT_EQ(a.status, Status::OK) << a.stages;
    EXPECT_EQ(a.status, b.status);
    EXPECT_TRUE(same(a.delta.matrix(), b.delta.matrix()));
    EXPECT_TRUE(same(a.s.matrix(), b.s.matrix()));
    EXPECT_TRUE(same(a.ds_ddelta, b.ds_ddelta));
    EXPECT_TRUE(same(a.ds_dtheta, b.ds_dtheta));
    EXPECT_TRUE(same(a.ddelta_dtheta, b.ddelta_dtheta));
    EXPECT_TRUE(same(a.ds_dp, b.ds_dp));
    EXPECT_TRUE(same(a.micro, b.micro));
    EXPECT_TRUE(same(a.micro_d_delta, b.micro_d_delta));
    EXPECT_TRUE(same(a.micro_d_theta, b.micro_d_theta));
    EXPECT_EQ(a.max_abs_residual, b.max_abs_residual);
    EXPECT_EQ(a.rcond, b.rcond);
    EXPECT_EQ(a.share_evaluations, b.share_evaluations);
    EXPECT_EQ(a.ds_dp.rows(), f.delta.size());
}

// 3.6 kernel test 5: a product below every frontier in the warm start is lifted by the warm stage.
TEST(PcmKernel, WarmStageLiftsProductBelowEveryFrontier) {
    Fixture f = make_fixture(3);
    const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    ASSERT_TRUE((s_data > 0).all());
    ArrayXd warm = f.delta;
    warm[6] -= 1.0;
    ASSERT_EQ(pcm_aggregate::shares(warm, f.mk, f.prm, false).s[6], 0.0) << "product 6 must be off every frontier";
    pcm_kernel::SolveOptions opt;
    opt.tol = 1e-12;
    auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, warm, nullptr, nullptr, opt);
    EXPECT_EQ(r.status, Status::OK) << r.stages;
    EXPECT_EQ(r.stages.rfind("warm ok", 0), 0u) << r.stages;
    EXPECT_EQ(r.stages.find("cold"), std::string::npos) << r.stages;
    EXPECT_LT(r.max_abs_residual, 1e-12);
    EXPECT_LT((r.delta - f.delta).abs().maxCoeff(), 1e-8);
    EXPECT_TRUE(r.ddelta_dtheta.allFinite());
}

// 3.6 kernel test 6: duplicate characteristics and a tiny price gap: OK with rcond reported, or
// SINGULAR_DS_DDELTA, never a silent NaN.
TEST(PcmKernel, NearDuplicateProductsNeverGiveSilentNaN) {
    for (double gap : {1e-3, 1e-6, 1e-9}) {
        Fixture f = make_fixture(9);
        const int a = 6, b = 7;
        f.mk.x.row(b) = f.mk.x.row(a);
        f.mk.p[b] = f.mk.p[a] + gap;
        const double s56 = (f.delta[a] - f.delta[a - 1]) / (f.mk.p[a] - f.mk.p[a - 1]);
        const double s68 = (f.delta[b + 1] - f.delta[a]) / (f.mk.p[b + 1] - f.mk.p[a]);
        f.delta[b] = f.delta[a] + gap * 0.5 * (s56 + s68);
        const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
        ASSERT_TRUE((s_data > 0).all()) << "gap " << gap;

        pcm_kernel::SolveOptions opt;
        auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, f.delta, nullptr, nullptr, opt);
        ASSERT_TRUE(r.status == Status::OK || r.status == Status::SINGULAR_DS_DDELTA)
            << "gap " << gap << ": " << pcm_kernel::status_name(r.status) << " " << r.stages;
        EXPECT_TRUE(std::isfinite(r.rcond)) << "gap " << gap;
        EXPECT_TRUE(r.ds_ddelta.allFinite() && r.ds_dtheta.allFinite());
        if (r.status == Status::OK) {
            EXPECT_GE(r.rcond, opt.rcond_min);
            EXPECT_TRUE(r.ddelta_dtheta.allFinite()) << "gap " << gap;
        } else {
            EXPECT_LT(r.rcond, opt.rcond_min);
            EXPECT_TRUE(r.ddelta_dtheta.array().isNaN().all()) << "gap " << gap;
        }
        std::printf("  gap %.0e: %s, rcond %.3e\n", gap, pcm_kernel::status_name(r.status), r.rcond);

        // the singular path itself: a threshold no Jacobian meets
        opt.rcond_min = 1.0;
        auto s = pcm_kernel::solve_market(s_data, f.mk, f.prm, f.delta, nullptr, nullptr, opt);
        EXPECT_EQ(s.status, Status::SINGULAR_DS_DDELTA);
        EXPECT_EQ(s.rcond, r.rcond);
        EXPECT_TRUE(s.ddelta_dtheta.array().isNaN().all());
        EXPECT_TRUE(s.ds_ddelta.allFinite());
    }
}

// Theta-path stage (DESIGN 3.2 step 3): with the warm stage starved, the path from theta_warm solves it.
TEST(PcmKernel, ThetaPathStageSolvesWhenWarmStageFails) {
    Fixture f = make_fixture(4);
    const ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    pcm_aggregate::Params prm_warm = f.prm;
    prm_warm.theta *= 0.5;
    prm_warm.sigma_p = 0.5;
    auto w = pcm_invert::invert(s_data, f.mk, prm_warm, f.delta, 1e-12, 5000);
    ASSERT_TRUE(w.converged);
    pcm_kernel::SolveOptions opt;
    opt.warm_evals = 1;
    opt.cold_evals = 1;
    auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, w.delta, &prm_warm, nullptr, opt);
    EXPECT_EQ(r.status, Status::OK) << r.stages;
    EXPECT_NE(r.stages.find("path ok"), std::string::npos) << r.stages;
    EXPECT_EQ(r.stages.find("cold"), std::string::npos) << r.stages;
    EXPECT_LT(r.max_abs_residual, opt.tol);
}

TEST(PcmKernel, InvalidInputIsAStatusNotAnException) {
    Fixture f = make_fixture(1);
    ArrayXd s_data = pcm_aggregate::shares(f.delta, f.mk, f.prm, false).s;
    s_data[3] = 0.0;
    pcm_kernel::SolveOptions opt;
    auto r = pcm_kernel::solve_market(s_data, f.mk, f.prm, ArrayXd(), nullptr, nullptr, opt);
    EXPECT_EQ(r.status, Status::INVALID_INPUT);
    EXPECT_NE(r.stages.find("invalid input"), std::string::npos);
    s_data[3] = 0.01;
    pcm_kernel::MicroRequest mr;
    mr.H = MatrixXd::Ones(3, 1);          // wrong number of rows
    mr.U = MatrixXd::Ones(f.delta.size(), 1);
    mr.h_of_part = {0};
    r = pcm_kernel::solve_market(s_data, f.mk, f.prm, ArrayXd(), nullptr, &mr, opt);
    EXPECT_EQ(r.status, Status::INVALID_INPUT);
    EXPECT_EQ(std::string(pcm_kernel::status_name(r.status)), "invalid_input");
}
