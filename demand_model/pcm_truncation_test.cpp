// Price sensitivity bounded below (LEARNINGS 29, SPEC_ANALYSIS v4 sections 2 and 5 item 6): a_i is
// LogNormal(0, sigma_p) truncated below at a_lo,i = alpha_lo / c_i. Tests (i) to (vi) of change 6, plus the
// events kernel on the sheared frontier. The bound on a_i itself (BoundKind::A, a_lo,i = a_lo for every draw,
// LEARNINGS 32) gets the same derivative and quadrature checks, and equals the ALPHA form when c_i = 1.
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include "pcm_demand.hpp"
#include "pcm_events.hpp"
#include "pcm_kernel.hpp"
#include "pcm_welfare.hpp"

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

double Phi(double z) { return 0.5 * std::erfc(-z / std::sqrt(2.0)); }

struct Fixture {
    pcm_demand::Market mk;
    pcm_demand::Params prm;
    ArrayXd delta;
};

// N products [1, z, dummy], prices on [5, 50], R draws with tastes on z and the dummy and two price-scale
// demographics; frontier slopes in conditional units are about 0.3 to 1.5, and alpha_lo = 0.5 cuts the
// windows of many draws, with c_i = exp(phi . b_i) spreading a_lo,i over a factor of about 10.
Fixture make_fixture(unsigned seed, int N = 14, int R = 300, double alpha_lo = 0.5,
                     pcm_demand::BoundKind kind = pcm_demand::BoundKind::ALPHA) {
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
        f.delta[j] = 1.5 * f.mk.p[j] - 0.012 * f.mk.p[j] * f.mk.p[j] - 2.0 + 0.8 * nd(gen);
    }
    f.mk.weights = ArrayXd(R);
    for (int i = 0; i < R; ++i) f.mk.weights[i] = 0.5 + ud(gen);
    f.mk.weights /= f.mk.weights.sum();
    f.mk.taste_basis = MatrixXd(R, 2);
    f.mk.taste_col = {1, 2};
    f.mk.scale_basis = MatrixXd(R, 2);
    for (int i = 0; i < R; ++i) {
        f.mk.taste_basis(i, 0) = nd(gen);
        f.mk.taste_basis(i, 1) = nd(gen);
        f.mk.scale_basis(i, 0) = nd(gen);
        f.mk.scale_basis(i, 1) = ud(gen) < 0.3 ? 1.0 : 0.0;
    }
    f.prm.theta = ArrayXd(2);
    f.prm.theta << 0.8, 1.5;
    f.prm.phi = ArrayXd(2);
    f.prm.phi << 0.5, -0.6;
    f.prm.sigma_p = 0.9;
    f.prm.lo = alpha_lo;
    f.prm.bound_kind = kind;
    return f;
}

pcm_demand::Params bump(const pcm_demand::Params & prm, int k, double h) {
    pcm_demand::Params out = prm;
    const int M = prm.theta.size(), Q = prm.phi.size();
    if (k < M) out.theta[k] += h;
    else if (k < M + Q) out.phi[k - M] += h;
    else out.sigma_p += h;
    return out;
}

// Conditional qualities and price scale of draw i.
void draw_q(const pcm_demand::Market & mk, const pcm_demand::Params & prm, const ArrayXd & delta, int i,
            ArrayXd & q, double & c) {
    VectorXd t = VectorXd::Zero(mk.x.cols());
    for (int m = 0; m < prm.theta.size(); ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
    const int Q = prm.phi.size();
    c = std::exp(Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0);
    q = (delta + (mk.x * t).array()) / c;
}

// The fixture restricted to the products with share above 1e-6 at its own delta.
Fixture positive_only(const Fixture & f) {
    const ArrayXd s = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    std::vector<int> keep;
    for (int j = 0; j < s.size(); ++j) if (s[j] > 1e-6) keep.push_back(j);
    Fixture g = f;
    g.mk.x = f.mk.x(keep, Eigen::all);
    g.mk.p = f.mk.p(keep);
    g.delta = f.delta(keep);
    return g;
}

double rel_err(const MatrixXd & a, const MatrixXd & b) {
    return (a - b).cwiseAbs().maxCoeff() / std::max(1e-300, b.cwiseAbs().maxCoeff());
}

pcm_kernel::MicroRequest make_micro(const Fixture & f) {
    const int R = f.mk.weights.size(), N = f.delta.size();
    pcm_kernel::MicroRequest mr;
    mr.H = MatrixXd(R, 2);
    for (int i = 0; i < R; ++i) { mr.H(i, 0) = 1.0; mr.H(i, 1) = f.mk.scale_basis(i, 1); }
    mr.U = MatrixXd::Zero(N, 3);
    mr.U.col(0).setOnes();
    for (int j = 0; j < N; ++j) mr.U(j, 1) = f.mk.x(j, 2), mr.U(j, 2) = f.mk.x(j, 1) - 1.0;
    mr.h_of_part = {0, 1, 1};
    return mr;
}

// Truncation statistics of a fixture: draws in which the bound removes a frontier vertex, and draws in
// which a positive-share product's window straddles a_lo (e_k > a_lo > e_{k+1} with e_{k+1} > 0).
struct Cut { int removed = 0, straddle = 0; };

Cut truncation_cuts(const Fixture & f) {
    Cut cut;
    const int R = f.mk.weights.size(), N = f.delta.size();
    for (int i = 0; i < R; ++i) {
        ArrayXd q; double c;
        draw_q(f.mk, f.prm, f.delta, i, q, c);
        ArrayXd s0 = ArrayXd::Zero(N), s1 = ArrayXd::Zero(N);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, 0.0, s0, nullptr, nullptr, nullptr, nullptr);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, pcm_demand::draw_a_lo(f.prm, c), s1, nullptr,
                                          nullptr, nullptr, nullptr);
        const int n0 = (s0 > 0).count(), n1 = (s1 > 0).count();
        if (n1 < n0) {
            ++cut.removed;
            if (n1 > 0) ++cut.straddle;
        }
    }
    return cut;
}

}  // namespace

// (i) the truncated formulas reduce to the untruncated ones: alpha_lo = 1e-300 against alpha_lo = 0 (which
// runs the code that existed before the bound) for shares, every derivative, micro parts and CS.
TEST(PcmTruncation, TinyBoundReproducesUntruncatedModel) {
    Fixture f = make_fixture(1);
    pcm_demand::Params p0 = f.prm, p1 = f.prm;
    p0.lo = 0.0;
    p1.lo = 1e-300;
    const auto a = pcm_demand::shares(f.delta, f.mk, p0, true), b = pcm_demand::shares(f.delta, f.mk, p1, true);
    EXPECT_LT((a.s - b.s).abs().maxCoeff(), 1e-12);
    EXPECT_LT((a.d_delta - b.d_delta).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.d_theta - b.d_theta).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.d_price - b.d_price).cwiseAbs().maxCoeff(), 1e-12);
    const auto mr = make_micro(f);
    const auto ea = pcm_kernel::evaluate_market(f.delta, f.mk, p0, &mr, true, true);
    const auto eb = pcm_kernel::evaluate_market(f.delta, f.mk, p1, &mr, true, true);
    EXPECT_LT((ea.micro - eb.micro).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((ea.micro_d_theta - eb.micro_d_theta).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((ea.ds_dtheta - eb.ds_dtheta).cwiseAbs().maxCoeff(), 1e-12);
    const auto ca = pcm_welfare::consumer_surplus(f.delta, f.mk, p0, 1), cb = pcm_welfare::consumer_surplus(f.delta, f.mk, p1, 1);
    EXPECT_LT((ca.per_draw - cb.per_draw).abs().maxCoeff(), 1e-12 * std::max(1.0, ca.per_draw.abs().maxCoeff()));
}

// (ii) analytic derivatives against central differences with a_lo cutting through windows, including
// products whose window straddles a_lo; the kernel's draw pass and micro parts agree with pcm_demand::shares.
void check_derivatives(const Fixture & f) {
    const Cut cut = truncation_cuts(f);
    std::printf("  draws %d: bound removes a vertex in %d, a positive window straddles a_lo in %d\n",
                static_cast<int>(f.mk.weights.size()), cut.removed, cut.straddle);
    ASSERT_GT(cut.straddle, 30) << "fixture must cut through windows";
    const auto sh = pcm_demand::shares(f.delta, f.mk, f.prm, true);
    ASSERT_TRUE((sh.s > 0).all());
    const int N = f.delta.size(), P = sh.d_theta.cols();
    auto S = [&](const ArrayXd & d, const pcm_demand::Market & mk, const pcm_demand::Params & pr) {
        return pcm_demand::shares(d, mk, pr, false).s;
    };
    // central difference with the smaller of two steps when they disagree (a kink inside the larger one)
    auto fd = [&](auto && eval, double h) {
        const ArrayXd a = (eval(h) - eval(-h)) / (2 * h);
        const ArrayXd b = (eval(h / 30) - eval(-h / 30)) / (2 * h / 30);
        return std::make_pair(a, b);
    };
    double worst_d = 0, worst_p = 0, worst_t = 0;
    auto pick = [](const std::pair<ArrayXd, ArrayXd> & ab, const VectorXd & an) {
        const double ea = (ab.first.matrix() - an).cwiseAbs().maxCoeff();
        const double eb = (ab.second.matrix() - an).cwiseAbs().maxCoeff();
        return std::min(ea, eb) / std::max(1e-300, an.cwiseAbs().maxCoeff());
    };
    for (int j = 0; j < N; ++j) {
        auto ed = [&](double h) { ArrayXd d = f.delta; d[j] += h; return S(d, f.mk, f.prm); };
        worst_d = std::max(worst_d, pick(fd(ed, 1e-6), sh.d_delta.col(j)));
        auto ep = [&](double h) { pcm_demand::Market mk = f.mk; mk.p[j] += h; return S(f.delta, mk, f.prm); };
        worst_p = std::max(worst_p, pick(fd(ep, 1e-6), sh.d_price.col(j)));
    }
    for (int k = 0; k < P; ++k) {
        auto et = [&](double h) { return S(f.delta, f.mk, bump(f.prm, k, h)); };
        const double e = pick(fd(et, 1e-6), sh.d_theta.col(k));
        std::printf("  d_theta column %d (%s): relative error %.2e\n", k,
                    k < 2 ? "taste" : (k < 4 ? "phi" : "sigma_p"), e);
        worst_t = std::max(worst_t, e);
    }
    std::printf("  worst relative error: d_delta %.2e, d_price %.2e, d_theta %.2e\n", worst_d, worst_p, worst_t);
    EXPECT_LT(worst_d, 1e-6);
    EXPECT_LT(worst_p, 1e-6);
    EXPECT_LT(worst_t, 1e-6);

    // the kernel draw pass equals pcm_demand::shares, and its micro parts match central differences
    const auto mr = make_micro(f);
    const auto ev = pcm_kernel::evaluate_market(f.delta, f.mk, f.prm, &mr, true, true);
    EXPECT_LT((ev.s - sh.s).abs().maxCoeff(), 1e-14);
    EXPECT_LT(rel_err(ev.ds_dtheta, sh.d_theta), 1e-12);
    EXPECT_LT(rel_err(ev.ds_ddelta, sh.d_delta), 1e-12);
    EXPECT_LT(rel_err(ev.ds_dp, sh.d_price), 1e-12);
    double worst_m = 0;
    for (int k = 0; k < P; ++k) {
        auto em = [&](double h) {
            return pcm_kernel::evaluate_market(f.delta, f.mk, bump(f.prm, k, h), &mr, false, false).micro.array().eval();
        };
        worst_m = std::max(worst_m, pick(fd(em, 1e-6), ev.micro_d_theta.col(k)));
    }
    std::printf("  micro_d_theta worst relative error %.2e\n", worst_m);
    EXPECT_LT(worst_m, 1e-6);
}

TEST(PcmTruncation, DerivativesMatchCentralDifferences) {
    check_derivatives(positive_only(make_fixture(2)));
}

// The same checks with the bound on a_i (no d/dlog c term through the bound): ds/dphi and the micro d_theta
// match central differences.
TEST(PcmTruncation, ABoundDerivativesMatchCentralDifferences) {
    check_derivatives(positive_only(make_fixture(2, 14, 300, 0.5, pcm_demand::BoundKind::A)));
}

// (iii) shares and CS of small vertical markets against quadrature in log a over the truncated density,
// with the integrand's breakpoints found by brute force (every pairwise indifference point), not by a hull.
void check_quadrature(pcm_demand::BoundKind kind) {
    // two draws with different price scales, so a_lo,i = alpha_lo / c_i differs; three products of which the
    // dearest straddles a_lo for draw 0 and lies wholly below it for draw 1
    pcm_demand::Market mk;
    mk.x = MatrixXd::Zero(3, 1);
    mk.p = ArrayXd(3);
    mk.p << 4.0, 9.0, 15.0;
    mk.weights = ArrayXd(2);
    mk.weights << 0.6, 0.4;
    mk.taste_basis = MatrixXd(2, 0);
    mk.scale_basis = MatrixXd(2, 1);
    mk.scale_basis << 0.0, 1.0;
    pcm_demand::Params prm;
    prm.theta = ArrayXd(0);
    prm.phi = ArrayXd(1);
    prm.phi << 0.7;
    prm.sigma_p = 1.3;
    prm.lo = 0.35;
    prm.bound_kind = kind;
    ArrayXd delta(3);
    delta << 6.0, 10.0, 12.6;           // slopes 1.5, 0.8, 0.433 for c = 1; divided by c = 2.01 for draw 1
    const auto sh = pcm_demand::shares(delta, mk, prm, false);
    const auto cs = pcm_welfare::consumer_surplus(delta, mk, prm, 1);
    const int n = 1000000;
    ArrayXd s_quad = ArrayXd::Zero(3);
    double cs_quad_total = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double c = std::exp(prm.phi[0] * mk.scale_basis(i, 0));
        const ArrayXd q = delta / c;
        const double s = prm.sigma_p, l_lo = std::log(pcm_demand::draw_a_lo(prm, c)), l_hi = l_lo + 14.0 * s;
        const double Z = 1.0 - Phi(l_lo / s);
        std::vector<double> bp = {l_lo, l_hi};
        for (int j = 0; j < 3; ++j) {
            bp.push_back(std::log(q[j] / mk.p[j]));
            for (int k = j + 1; k < 3; ++k)
                if ((q[k] - q[j]) / (mk.p[k] - mk.p[j]) > 0) bp.push_back(std::log((q[k] - q[j]) / (mk.p[k] - mk.p[j])));
        }
        std::sort(bp.begin(), bp.end());
        ArrayXd si = ArrayXd::Zero(3);
        double csi = 0.0;
        for (size_t b = 0; b + 1 < bp.size(); ++b) {
            const double lo = std::max(bp[b], l_lo), hi = std::min(bp[b + 1], l_hi);
            if (!(hi > lo)) continue;
            const int m = std::max(8, static_cast<int>(n * (hi - lo) / (l_hi - l_lo))) & ~1;   // even, Simpson
            const double h = (hi - lo) / m;
            // the choice is constant inside a segment: take it at the midpoint, so the endpoints (where two
            // options tie) are not given to the wrong product
            const double am = std::exp(0.5 * (lo + hi));
            int best = -1;
            double um = 0.0;
            for (int j = 0; j < 3; ++j)
                if (q[j] - am * mk.p[j] > um) { um = q[j] - am * mk.p[j]; best = j; }
            for (int k = 0; k <= m && best >= 0; ++k) {
                const double l = lo + k * h, a = std::exp(l);
                const double wgt = (k == 0 || k == m) ? 1.0 : (k % 2 ? 4.0 : 2.0);
                const double dens = std::exp(-0.5 * (l / s) * (l / s)) / (s * std::sqrt(2 * M_PI) * Z);
                const double u = q[best] - a * mk.p[best];
                {
                    si[best] += wgt * h / 3.0 * dens;
                    csi += wgt * h / 3.0 * dens * u / a;
                }
            }
        }
        std::printf("  draw %d: a_lo %.4f, quadrature shares %.12f %.12f %.12f, CS %.12f (kernel %.12f)\n", i,
                    pcm_demand::draw_a_lo(prm, c), si[0], si[1], si[2], csi, cs.per_draw[i]);
        EXPECT_NEAR(csi, cs.per_draw[i], 1e-10);
        s_quad += mk.weights[i] * si;
        cs_quad_total += mk.weights[i] * csi;
    }
    std::printf("  kernel shares %.12f %.12f %.12f quadrature %.12f %.12f %.12f\n", sh.s[0], sh.s[1], sh.s[2], s_quad[0], s_quad[1], s_quad[2]);
    EXPECT_LT((s_quad - sh.s).abs().maxCoeff(), 1e-10);
    EXPECT_NEAR(cs_quad_total, cs.total, 1e-10);
}

TEST(PcmTruncation, VerticalMarketMatchesQuadrature) { check_quadrature(pcm_demand::BoundKind::ALPHA); }

// Welfare with per-draw c under the bound on a_i: the truncation point is a_lo for both draws, the quadrature
// integrates the draw's own conditional qualities q = delta / c_i.
TEST(PcmTruncation, ABoundVerticalMarketMatchesQuadrature) { check_quadrature(pcm_demand::BoundKind::A); }

// (iv) per-draw CS is bounded by the most a consumer at the bound could gain: max_j q_j / a_lo,i - p_j.
TEST(PcmTruncation, PerDrawSurplusIsBounded) {
    for (double alpha_lo : {0.1, 0.5, 1.0}) {
        Fixture f = make_fixture(4, 14, 300, alpha_lo);
        const auto cs = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 4);
        int binding = 0;
        for (int i = 0; i < f.mk.weights.size(); ++i) {
            ArrayXd q; double c;
            draw_q(f.mk, f.prm, f.delta, i, q, c);
            const double bound = std::max(0.0, (q / (alpha_lo / c) - f.mk.p).maxCoeff());
            EXPECT_LE(cs.per_draw[i], bound * (1 + 1e-12)) << "draw " << i << " alpha_lo " << alpha_lo;
            EXPECT_GE(cs.per_draw[i], 0.0);
            binding += cs.per_draw[i] > 0.5 * bound;
        }
        EXPECT_GE(binding, 0);
    }
    // and CS falls as the bound rises (the bound removes the least price-sensitive consumers)
    double prev = 1e300;
    for (double alpha_lo : {0.0, 0.05, 0.2, 0.5, 1.0}) {
        Fixture f = make_fixture(4, 14, 300, alpha_lo);
        const double t = pcm_welfare::consumer_surplus(f.delta, f.mk, f.prm, 1).total;
        EXPECT_LT(t, prev) << "alpha_lo " << alpha_lo;
        prev = t;
    }
}

// (v) a product on every draw's hull whose window lies below a_lo everywhere has zero share; one round of
// frontier_lift (hull of dc - a_lo p) puts it on some consumer's frontier with positive mass, and the kernel
// inverts shares that need such a lift from the vertical guess.
TEST(PcmTruncation, ZeroShareProductIsLiftedInOneRound) {
    Fixture f = make_fixture(5, 8, 200, 0.5);
    f.prm.theta.setZero();                 // no taste heterogeneity: the hull is the same in every draw
    f.prm.phi << 0.1, -0.1;
    const int N = f.delta.size();
    // last product: 0.1 conditional units above its cheaper neighbour, slope well below every a_lo,i
    f.delta[N - 1] = f.delta[N - 2] + 0.1 * (f.mk.p[N - 1] - f.mk.p[N - 2]) * 0.1;
    const auto sh0 = pcm_demand::shares(f.delta, f.mk, f.prm, false);
    ArrayXd s_untr = pcm_demand::shares(f.delta, f.mk, pcm_demand::Params{f.prm.theta, f.prm.phi, f.prm.sigma_p}, false).s;
    ASSERT_GT(s_untr[N - 1], 0.0) << "the product is on the untruncated frontier";
    EXPECT_EQ(sh0.s[N - 1], 0.0);
    const ArrayXd lift = pcm_demand::detail::frontier_lift(f.delta, f.mk, f.prm);
    ASSERT_TRUE(std::isfinite(lift[N - 1]));
    EXPECT_GT(lift[N - 1], 0.0);
    ArrayXd d = f.delta;
    d[N - 1] += lift[N - 1] * (1.0 + 1e-6) + 1e-9;
    const auto sh1 = pcm_demand::shares(d, f.mk, f.prm, false);
    EXPECT_GT(sh1.s[N - 1], 0.0);
    // inversion of positive shares from the vertical guess
    Fixture g = positive_only(make_fixture(6, 14, 300, 0.5));
    const ArrayXd s_data = pcm_demand::shares(g.delta, g.mk, g.prm, false).s;
    ASSERT_TRUE((s_data > 0).all());
    pcm_kernel::SolveOptions opt;
    opt.tol = 1e-12;
    const auto r = pcm_kernel::solve_market(s_data, g.mk, g.prm, ArrayXd(), nullptr, nullptr, opt);
    EXPECT_EQ(r.status, pcm_kernel::Status::OK) << r.stages;
    EXPECT_LT(r.max_abs_residual, 1e-12);
    EXPECT_LT((r.delta - g.delta).abs().maxCoeff(), 1e-7);
}

// (vi) shares are continuous where a frontier slope crosses a_lo (the density jumps there, the cdf does not).
TEST(PcmTruncation, SharesContinuousAtTheBound) {
    pcm_demand::Market mk;
    mk.x = MatrixXd::Zero(2, 1);
    mk.p = ArrayXd(2);
    mk.p << 5.0, 12.0;
    mk.weights = ArrayXd::Ones(1);
    mk.taste_basis = MatrixXd(1, 0);
    mk.scale_basis = MatrixXd(1, 0);
    pcm_demand::Params prm;
    prm.theta = ArrayXd(0);
    prm.phi = ArrayXd(0);
    prm.sigma_p = 1.1;
    prm.lo = 0.4;
    for (double eps : {-1e-9, 1e-9}) {
        ArrayXd d(2);
        d << 8.0, 8.0 + (0.4 + eps) * 7.0;           // slope into product 2 = a_lo + eps
        const auto s = pcm_demand::shares(d, mk, prm, false).s;
        ArrayXd d0(2);
        d0 << 8.0, 8.0 + 0.4 * 7.0;
        const auto s0 = pcm_demand::shares(d0, mk, prm, false).s;
        EXPECT_LT((s - s0).abs().maxCoeff(), 1e-8) << "eps " << eps;
        if (eps < 0) EXPECT_EQ(s[1], 0.0);
        else EXPECT_GT(s[1], 0.0);
    }
    // also with the bound itself moving through a fixed slope
    ArrayXd d(2);
    d << 8.0, 8.0 + 0.4 * 7.0;
    pcm_demand::Params lo = prm, hi = prm;
    lo.lo = 0.4 - 1e-9;
    hi.lo = 0.4 + 1e-9;
    EXPECT_LT((pcm_demand::shares(d, mk, lo, false).s - pcm_demand::shares(d, mk, hi, false).s).abs().maxCoeff(), 1e-8);
}

// The events kernel works on the frontier of products with positive mass: B_j of max_step is the weight of the
// draws in which j has positive conditional share under the bound, and frontier signatures change exactly
// when that set does.
TEST(PcmTruncation, EventsUseThePositiveMassFrontier) {
    Fixture f = make_fixture(7);
    const int N = f.delta.size(), R = f.mk.weights.size(), P = f.prm.theta.size() + f.prm.phi.size() + 1;
    ArrayXd B = ArrayXd::Zero(N);
    for (int i = 0; i < R; ++i) {
        ArrayXd q; double c;
        draw_q(f.mk, f.prm, f.delta, i, q, c);
        ArrayXd s = ArrayXd::Zero(N);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, f.prm.lo / c, s, nullptr, nullptr, nullptr,
                                          nullptr);
        for (int j = 0; j < N; ++j) if (s[j] > 0) B[j] += f.mk.weights[i];
    }
    const auto ev = pcm_events::max_step(f.delta, ArrayXd::Zero(N), f.mk, f.prm, ArrayXd::Zero(P), 1.0, 2);
    EXPECT_LT((ev.buyer_mass - B).abs().maxCoeff(), 1e-14);
    pcm_demand::Params p0 = f.prm;
    p0.lo = 0.0;
    const auto ev0 = pcm_events::max_step(f.delta, ArrayXd::Zero(N), f.mk, p0, ArrayXd::Zero(P), 1.0, 2);
    EXPECT_GT((ev0.buyer_mass - ev.buyer_mass).maxCoeff(), 0.01) << "the bound must remove some buyers";
    // signatures: moving the bound changes the hashes of the draws whose positive-mass set changes, and no others
    pcm_demand::Params p1 = f.prm;
    p1.lo = 0.55;
    const auto s_a = pcm_events::frontier_signature(f.delta, f.mk, f.prm, 2);
    const auto s_b = pcm_events::frontier_signature(f.delta, f.mk, p1, 2);
    int mismatch = 0, changed = 0;
    for (int i = 0; i < R; ++i) {
        ArrayXd q; double c;
        draw_q(f.mk, f.prm, f.delta, i, q, c);
        ArrayXd sa = ArrayXd::Zero(N), sb = ArrayXd::Zero(N);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, f.prm.lo / c, sa, nullptr, nullptr, nullptr,
                                          nullptr);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, p1.lo / c, sb, nullptr, nullptr, nullptr,
                                          nullptr);
        const bool set_changed = ((sa > 0) != (sb > 0)).any();
        changed += set_changed;
        mismatch += set_changed != (s_a[i] != s_b[i]);
    }
    EXPECT_GT(changed, 5);
    EXPECT_EQ(mismatch, 0);
}

// Regression: with no price-scale parameters (c_i = 1) the bound on a_i is the bound on alpha_i, in every kernel
// output, to 1e-12.
TEST(PcmTruncation, ABoundWithoutScaleEqualsAlphaBound) {
    Fixture f = make_fixture(8);
    f.mk.scale_basis = MatrixXd(f.mk.weights.size(), 0);
    f.prm.phi = ArrayXd(0);
    pcm_demand::Params pa = f.prm, pb = f.prm;
    pa.bound_kind = pcm_demand::BoundKind::ALPHA;
    pb.bound_kind = pcm_demand::BoundKind::A;
    const int N = f.delta.size(), P = f.prm.theta.size() + 1;
    pcm_kernel::MicroRequest mr;
    mr.H = MatrixXd::Ones(f.mk.weights.size(), 1);
    mr.U = MatrixXd(N, 2);
    mr.U.col(0).setOnes();
    mr.U.col(1) = f.mk.x.col(2);
    mr.h_of_part = {0, 0};
    const auto a = pcm_kernel::evaluate_market(f.delta, f.mk, pa, &mr, true, true);
    const auto b = pcm_kernel::evaluate_market(f.delta, f.mk, pb, &mr, true, true);
    EXPECT_LT((a.s - b.s).abs().maxCoeff(), 1e-12);
    EXPECT_LT((a.ds_ddelta - b.ds_ddelta).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.ds_dtheta - b.ds_dtheta).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.ds_dp - b.ds_dp).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.micro - b.micro).cwiseAbs().maxCoeff(), 1e-12);
    EXPECT_LT((a.micro_d_theta - b.micro_d_theta).cwiseAbs().maxCoeff(), 1e-12);
    const auto sa = pcm_demand::shares(f.delta, f.mk, pa, true), sb = pcm_demand::shares(f.delta, f.mk, pb, true);
    EXPECT_LT((sa.d_theta - sb.d_theta).cwiseAbs().maxCoeff(), 1e-12);
    const auto ca = pcm_welfare::consumer_surplus(f.delta, f.mk, pa, 2), cb = pcm_welfare::consumer_surplus(f.delta, f.mk, pb, 2);
    EXPECT_LT((ca.per_draw - cb.per_draw).abs().maxCoeff(), 1e-12 * std::max(1.0, ca.per_draw.abs().maxCoeff()));
    const ArrayXd la = pcm_demand::detail::frontier_lift(f.delta, f.mk, pa), lb = pcm_demand::detail::frontier_lift(f.delta, f.mk, pb);
    for (int j = 0; j < N; ++j) EXPECT_EQ(la[j], lb[j]);
    EXPECT_EQ(pcm_events::frontier_signature(f.delta, f.mk, pa, 2), pcm_events::frontier_signature(f.delta, f.mk, pb, 2));
    ArrayXd dth = ArrayXd::Zero(P);
    dth[0] = 0.3;
    const auto ea = pcm_events::max_step(f.delta, ArrayXd::Constant(N, 0.1), f.mk, pa, dth, 1.0, 2);
    const auto eb = pcm_events::max_step(f.delta, ArrayXd::Constant(N, 0.1), f.mk, pb, dth, 1.0, 2);
    EXPECT_LT((ea.tau_draw - eb.tau_draw).abs().maxCoeff(), 1e-12);
    EXPECT_LT((ea.buyer_mass - eb.buyer_mass).abs().maxCoeff(), 1e-12);
    const auto fa = pcm_kernel::frontier_anatomy(f.delta, f.mk, pa), fb = pcm_kernel::frontier_anatomy(f.delta, f.mk, pb);
    EXPECT_TRUE((fa.frontier_size == fb.frontier_size).all());
    EXPECT_LT((fa.mass_log_width - fb.mass_log_width).abs().maxCoeff(), 1e-12);
    // inversion from the vertical guess lands on the same delta
    const Fixture g = positive_only(f);
    pcm_demand::Params ga = pa, gb = pb;
    const ArrayXd s_data = pcm_demand::shares(g.delta, g.mk, ga, false).s;
    ASSERT_TRUE((s_data > 0).all());
    {
        pcm_kernel::SolveOptions opt;
        opt.tol = 1e-12;
        const auto ra = pcm_kernel::solve_market(s_data, g.mk, ga, ArrayXd(), nullptr, nullptr, opt);
        const auto rb = pcm_kernel::solve_market(s_data, g.mk, gb, ArrayXd(), nullptr, nullptr, opt);
        EXPECT_EQ(ra.status, pcm_kernel::Status::OK) << ra.stages;
        EXPECT_LT((ra.delta - rb.delta).abs().maxCoeff(), 1e-12);
    }
}

// Under the bound on a_i the truncation point is the same for every draw, so draws with a larger c_i face a
// larger bound on alpha_i: the events kernel and frontier_lift use the frontier of positive-mass products,
// B_j of max_step is the weight of the draws in which j has positive conditional share, and the two bound forms
// differ when c varies.
TEST(PcmTruncation, ABoundEventsUseThePositiveMassFrontier) {
    Fixture f = make_fixture(7, 14, 300, 0.5, pcm_demand::BoundKind::A);
    const int N = f.delta.size(), R = f.mk.weights.size(), P = f.prm.theta.size() + f.prm.phi.size() + 1;
    ArrayXd B = ArrayXd::Zero(N);
    for (int i = 0; i < R; ++i) {
        ArrayXd q; double c;
        draw_q(f.mk, f.prm, f.delta, i, q, c);
        ArrayXd s = ArrayXd::Zero(N);
        pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, f.prm.lo, s, nullptr, nullptr, nullptr, nullptr);
        for (int j = 0; j < N; ++j) if (s[j] > 0) B[j] += f.mk.weights[i];
    }
    const auto ev = pcm_events::max_step(f.delta, ArrayXd::Zero(N), f.mk, f.prm, ArrayXd::Zero(P), 1.0, 2);
    EXPECT_LT((ev.buyer_mass - B).abs().maxCoeff(), 1e-14);
    pcm_demand::Params pa = f.prm;
    pa.bound_kind = pcm_demand::BoundKind::ALPHA;
    const auto s_a = pcm_demand::shares(f.delta, f.mk, f.prm, false).s, s_alpha = pcm_demand::shares(f.delta, f.mk, pa, false).s;
    EXPECT_GT((s_a - s_alpha).abs().maxCoeff(), 1e-4) << "with c varying the two bounds differ";
    // a pure scale direction moves the A frontier: the linearised first event along dphi matches the brute-force
    // first change of the signature on a fine tau grid, to the grid step plus the linearisation error
    ArrayXd dth = ArrayXd::Zero(P);
    dth[f.prm.theta.size()] = 0.4;
    const auto es = pcm_events::max_step(f.delta, ArrayXd::Zero(N), f.mk, f.prm, dth, 0.5, 2);
    EXPECT_LT(es.tau_first, 0.5) << "a scale step moves some draw's A frontier";
    const auto sig0 = pcm_events::frontier_signature(f.delta, f.mk, f.prm, 2);
    double tau_bf = 0.5;
    for (int g = 1; g <= 2000 && tau_bf == 0.5; ++g) {
        pcm_demand::Params pt = f.prm;
        pt.phi[0] += 0.4 * 0.5 * g / 2000.0;
        if (pcm_events::frontier_signature(f.delta, f.mk, pt, 2) != sig0) tau_bf = 0.5 * g / 2000.0;
    }
    EXPECT_NEAR(es.tau_first, tau_bf, 0.02 * std::max(tau_bf, 1e-3) + 0.5 / 2000.0);
    // frontier_lift lifts a zero-share product onto a positive-mass window
    const auto sh = pcm_demand::shares(f.delta, f.mk, f.prm, false).s;
    const ArrayXd lift = pcm_demand::detail::frontier_lift(f.delta, f.mk, f.prm);
    for (int j = 0; j < N; ++j) {
        if (sh[j] > 0 || !std::isfinite(lift[j])) continue;
        ArrayXd d = f.delta;
        d[j] += lift[j] * (1.0 + 1e-6) + 1e-9;
        EXPECT_GT(pcm_demand::shares(d, f.mk, f.prm, false).s[j], 0.0) << "product " << j;
    }
}

// frontier_anatomy: per-draw mass is the draw's inside share, the mass-weighted window width telescopes to at
// most the log range, and the sizes count the positive-mass windows.
TEST(PcmTruncation, FrontierAnatomyMatchesTheDrawPass) {
    for (auto kind : {pcm_demand::BoundKind::ALPHA, pcm_demand::BoundKind::A}) {
        Fixture f = make_fixture(9, 14, 300, 0.5, kind);
        const int N = f.delta.size(), R = f.mk.weights.size();
        const auto an = pcm_kernel::frontier_anatomy(f.delta, f.mk, f.prm);
        double tot = 0.0;
        for (int i = 0; i < R; ++i) {
            ArrayXd q; double c;
            draw_q(f.mk, f.prm, f.delta, i, q, c);
            ArrayXd s = ArrayXd::Zero(N);
            const double a_lo = pcm_demand::draw_a_lo(f.prm, c);
            pcm_demand::detail::cond_vertical(q, f.mk.p, f.prm.sigma_p, a_lo, s, nullptr, nullptr, nullptr, nullptr);
            EXPECT_NEAR(an.mass[i], s.sum(), 1e-14);
            EXPECT_EQ(an.frontier_size[i], (s > 0).count());
            if (an.mass[i] > 0) {
                EXPECT_LE(an.mass_log_width[i], an.mass[i] * an.log_range[i] * (1 + 1e-12));
                EXPECT_GT(an.log_range[i], 0.0);
                if (an.frontier_size[i] == 1) EXPECT_NEAR(an.mass_log_width[i], an.mass[i] * an.log_range[i], 1e-12);
            }
            tot += f.mk.weights[i] * an.mass[i];
        }
        EXPECT_NEAR(tot, pcm_demand::shares(f.delta, f.mk, f.prm, false).s.sum(), 1e-12);
    }
}
