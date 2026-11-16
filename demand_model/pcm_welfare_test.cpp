#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <stdexcept>
#include "pcm_demand.hpp"
#include "pcm_welfare.hpp"

using Eigen::ArrayXd;
using Eigen::MatrixXd;

namespace {

double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

struct Fixture {
    pcm_demand::Market mk;
    pcm_demand::Params prm;
    ArrayXd delta;
};

// N products with characteristics [1, z, dummy], prices on [5, 50] with gaps above 0.5, R draws with
// two taste parameters (normal on z and on the dummy) and one price-scale demographic. delta is noisy
// so that frontiers differ across draws and some products are on few of them.
Fixture make_fixture(unsigned seed, int N, int R) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    Fixture f;
    f.mk.x = MatrixXd(N, 3);
    f.mk.p = ArrayXd(N);
    f.delta = ArrayXd(N);
    for (int j = 0; j < N; ++j) {
        f.mk.x(j, 0) = 1.0;
        f.mk.x(j, 1) = ud(gen) * 2.0;
        f.mk.x(j, 2) = ud(gen) < 0.4 ? 1.0 : 0.0;
        f.mk.p[j] = 5.0 + 45.0 * j / std::max(1, N - 1) + 0.3 * ud(gen);
        f.delta[j] = 1.5 * f.mk.p[j] - 0.01 * f.mk.p[j] * f.mk.p[j] - 4.0 + 2.0 * nd(gen);
    }
    f.mk.weights = ArrayXd(R);
    for (int i = 0; i < R; ++i) f.mk.weights[i] = 0.5 + ud(gen);
    f.mk.weights /= f.mk.weights.sum();
    f.mk.taste_basis = MatrixXd(R, 2);
    f.mk.taste_col = {1, 2};
    f.mk.scale_basis = MatrixXd(R, 1);
    for (int i = 0; i < R; ++i) {
        f.mk.taste_basis(i, 0) = nd(gen);
        f.mk.taste_basis(i, 1) = nd(gen);
        f.mk.scale_basis(i, 0) = nd(gen);
    }
    f.prm.theta = ArrayXd(2);
    f.prm.theta << 1.5, 3.0;
    f.prm.phi = ArrayXd(1);
    f.prm.phi << 0.4;
    f.prm.sigma_p = 0.8;
    return f;
}

pcm_demand::Market single_draw_market(const ArrayXd & p) {
    pcm_demand::Market mk;
    mk.x = MatrixXd::Zero(p.size(), 1);
    mk.p = p;
    mk.weights = ArrayXd::Ones(1);
    mk.taste_basis = MatrixXd(1, 0);
    mk.scale_basis = MatrixXd(1, 0);
    return mk;
}

pcm_demand::Params vertical_params(double sigma_p) {
    pcm_demand::Params prm;
    prm.theta = ArrayXd(0);
    prm.phi = ArrayXd(0);
    prm.sigma_p = sigma_p;
    return prm;
}

// Drops product j from a market and its delta.
Fixture drop_product(const Fixture & f, int j) {
    const int N = static_cast<int>(f.mk.p.size());
    Fixture g = f;
    g.mk.x = MatrixXd(N - 1, f.mk.x.cols());
    g.mk.p = ArrayXd(N - 1);
    g.delta = ArrayXd(N - 1);
    for (int k = 0, r = 0; k < N; ++k) {
        if (k == j) continue;
        g.mk.x.row(r) = f.mk.x.row(k);
        g.mk.p[r] = f.mk.p[k];
        g.delta[r] = f.delta[k];
        ++r;
    }
    return g;
}

}  // namespace

// DESIGN 3.6 welfare 1: one draw, one product with q = 2, p = 1, sigma_p = 0.5.
TEST(PcmWelfare, OneProductClosedForm) {
    const double q = 2.0, p = 1.0, s = 0.5;
    const auto mk = single_draw_market(ArrayXd::Constant(1, p));
    const auto r = pcm_welfare::consumer_surplus(ArrayXd::Constant(1, q), mk, vertical_params(s), 1);
    // q E(1/a; a < q/p) - p F(q/p)
    const double e = q / p;
    const double closed = q * std::exp(0.5 * s * s) * Phi((std::log(e) + s * s) / s) - p * Phi(std::log(e) / s);
    EXPECT_NEAR(r.total, closed, 1e-14);
    ASSERT_EQ(r.per_draw.size(), 1);
    EXPECT_NEAR(r.per_draw[0], closed, 1e-14);

    // Independent check of the closed form: integral of max(q/a - p, 0) over a ~ LogNormal(0, s),
    // with a = exp(s z), Simpson on z in [-14, ln(q/p)/s].
    const double lo = -14.0, hi = std::log(e) / s;
    const int n = 200000;
    const double h = (hi - lo) / n;
    auto f = [&](double z) { return (q * std::exp(-s * z) - p) * std::exp(-0.5 * z * z) / std::sqrt(2.0 * M_PI); };
    double acc = f(lo) + f(hi);
    for (int k = 1; k < n; ++k) acc += (k % 2 ? 4.0 : 2.0) * f(lo + k * h);
    EXPECT_NEAR(r.total, acc * h / 3.0, 1e-11);
}

// DESIGN 3.6 welfare 2: (delta, theta, c) -> kappa (delta, theta, c) leaves CS unchanged (LEARNINGS 11).
TEST(PcmWelfare, ScaleInvariance) {
    Fixture f = make_fixture(7, 30, 500);
    const auto r0 = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 1);
    ASSERT_GT(r0.total, 0.0);
    for (double kappa : {0.37, 3.7, 25.0}) {
        Fixture g = f;
        g.delta = kappa * f.delta;
        g.prm.theta = kappa * f.prm.theta;
        // c_i -> kappa c_i: a second scale column of ones with coefficient log(kappa)
        g.mk.scale_basis = MatrixXd(f.mk.scale_basis.rows(), 2);
        g.mk.scale_basis.col(0) = f.mk.scale_basis.col(0);
        g.mk.scale_basis.col(1).setOnes();
        g.prm.phi = ArrayXd(2);
        g.prm.phi << f.prm.phi[0], std::log(kappa);
        const auto r1 = pcm_welfare::consumer_surplus(g.delta, g.mk, g.prm, 1);
        EXPECT_NEAR(r1.total, r0.total, 1e-12 * std::max(1.0, std::abs(r0.total))) << "kappa " << kappa;
        const double tol = 1e-12 * std::max(1.0, r0.per_draw.abs().maxCoeff());
        EXPECT_LE((r1.per_draw - r0.per_draw).abs().maxCoeff(), tol) << "kappa " << kappa;
    }
}

// DESIGN 3.6 welfare 3a: removing a product that is on no draw's frontier changes nothing.
TEST(PcmWelfare, DominatedProductRemoved) {
    Fixture f = make_fixture(11, 30, 500);
    const int j = 13;
    f.delta[j] = -1e3;                      // far below the origin edge for every draw
    const auto r0 = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 1);
    const ArrayXd s = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    ASSERT_EQ(s[j], 0.0);
    const Fixture g = drop_product(f, j);
    const auto r1 = pcm_welfare::consumer_surplus(g.delta, g.mk, g.prm, 1);
    EXPECT_NEAR(r1.total, r0.total, 1e-12 * std::max(1.0, std::abs(r0.total)));
    EXPECT_LE((r1.per_draw - r0.per_draw).abs().maxCoeff(), 1e-12 * std::max(1.0, r0.per_draw.abs().maxCoeff()));

    // removing a product that is bought lowers CS
    const ArrayXd s1 = pcm_demand::shares(g.delta, g.mk, g.prm, false).s;
    int k = 0;
    s1.maxCoeff(&k);
    const auto r2 = pcm_welfare::consumer_surplus(drop_product(g, k).delta, drop_product(g, k).mk, g.prm, 1);
    EXPECT_LT(r2.total, r1.total);
}

// DESIGN 3.6 welfare 3b: removing the only product gives CS = 0; so does a product nobody buys.
TEST(PcmWelfare, NoProductGivesZero) {
    const auto mk1 = single_draw_market(ArrayXd::Constant(1, 1.0));
    const auto prm = vertical_params(0.5);
    const auto r1 = pcm_welfare::consumer_surplus(ArrayXd::Constant(1, 2.0), mk1, prm, 1);
    EXPECT_GT(r1.total, 0.0);
    const auto mk0 = single_draw_market(ArrayXd(0));
    const auto r0 = pcm_welfare::consumer_surplus(ArrayXd(0), mk0, prm, 1);
    EXPECT_EQ(r0.total, 0.0);
    ASSERT_EQ(r0.per_draw.size(), 1);
    EXPECT_EQ(r0.per_draw[0], 0.0);
    const auto rn = pcm_welfare::consumer_surplus(ArrayXd::Constant(1, -2.0), mk1, prm, 1);
    EXPECT_EQ(rn.total, 0.0);
}

// Roy's identity in price units: dCS/dp_j = -s_j, against the existing share code.
TEST(PcmWelfare, PriceDerivativeEqualsMinusShare) {
    Fixture f = make_fixture(3, 30, 500);
    const ArrayXd s = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    const double h = 1e-6;
    int n_bought = 0;
    for (int j = 0; j < f.mk.p.size(); ++j) {
        Fixture up = f, dn = f;
        up.mk.p[j] += h;
        dn.mk.p[j] -= h;
        const double d = (pcm_welfare::consumer_surplus(up.delta, up.mk, up.prm, 1).total -
                          pcm_welfare::consumer_surplus(dn.delta, dn.mk, dn.prm, 1).total) / (2.0 * h);
        EXPECT_NEAR(d, -s[j], 1e-7) << "product " << j;
        n_bought += s[j] > 0.0;
    }
    EXPECT_GE(n_bought, 10);
    EXPECT_LT(n_bought, 30);                // the fixture keeps some products off every frontier
}

TEST(PcmWelfare, ThreadCountDoesNotChangeResult) {
    Fixture f = make_fixture(5, 60, 4000);
    const auto r1 = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 1);
    const auto r8 = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 8);
    EXPECT_EQ(r1.total, r8.total);
    EXPECT_TRUE((r1.per_draw == r8.per_draw).all());
    EXPECT_NEAR(r1.total, (f.mk.weights * r1.per_draw).sum(), 1e-12);
}

TEST(PcmWelfare, InvalidInputThrows) {
    Fixture f = make_fixture(1, 5, 10);
    auto bad = f;
    bad.prm.sigma_p = 0.0;
    EXPECT_THROW(pcm_welfare::consumer_surplus(bad.delta, bad.mk, bad.prm, 1), std::invalid_argument);
    bad = f;
    bad.mk.p[2] = bad.mk.p[1];
    EXPECT_THROW(pcm_welfare::consumer_surplus(bad.delta, bad.mk, bad.prm, 1), std::invalid_argument);
    bad = f;
    EXPECT_THROW(pcm_welfare::consumer_surplus(ArrayXd::Zero(4), bad.mk, bad.prm, 1), std::invalid_argument);
    bad = f;
    bad.mk.taste_col = {1, 7};
    EXPECT_THROW(pcm_welfare::consumer_surplus(bad.delta, bad.mk, bad.prm, 1), std::invalid_argument);
    bad = f;
    bad.prm.phi = ArrayXd::Zero(2);
    EXPECT_THROW(pcm_welfare::consumer_surplus(bad.delta, bad.mk, bad.prm, 1), std::invalid_argument);
}
