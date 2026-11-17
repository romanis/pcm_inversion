#include "pcm_demand.hpp"
#include "pcm_market_share.hpp"

#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
#include <limits>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// defined in pcm_market_share.cpp: indices of products on the vertical frontier
std::vector<int> index_positive_shares(const ArrayXd & delta, const ArrayXd & p);

namespace {

constexpr double INV_SQRT_2PI = 0.39894228040143267794;

// LogNormal(0, sigma) cdf F, pdf f and dF/dsigma g at e > 0. All are 0 at e <= 0. h = dF/dlog c is
// zero without truncation.
struct LnTerms { double F, f, g, h; };

// Lower truncation of a ~ LogNormal(0, sigma) at a_lo for one draw: z_lo = log(a_lo) / sigma, the mass
// kept Z = 1 - Phi(z_lo) and phi(z_lo). Off when a_lo <= 0 (the untruncated model, exactly the old code).
struct Trunc {
    bool on = false;
    double a_lo = 0.0, z_lo = 0.0, Phi_lo = 0.0, Z = 1.0, phi_lo = 0.0;
};

inline Trunc make_trunc(double a_lo, double sigma) {
    Trunc tr;
    if (!(a_lo > 0.0)) return tr;
    tr.on = true;
    tr.a_lo = a_lo;
    tr.z_lo = std::log(a_lo) / sigma;
    tr.Phi_lo = 0.5 * std::erfc(-tr.z_lo / std::sqrt(2.0));
    tr.Z = 0.5 * std::erfc(tr.z_lo / std::sqrt(2.0));
    tr.phi_lo = INV_SQRT_2PI * std::exp(-0.5 * tr.z_lo * tr.z_lo);
    return tr;
}

inline LnTerms ln_terms(double e, double sigma, const Trunc & tr) {
    if (!tr.on) {
        if (e <= 0.0) return {0.0, 0.0, 0.0, 0.0};
        const double z = std::log(e) / sigma;
        const double phi = INV_SQRT_2PI * std::exp(-0.5 * z * z);
        return {0.5 * std::erfc(-z / std::sqrt(2.0)), phi / (e * sigma), -phi * z / sigma, 0.0};
    }
    // F_T(e) = (Phi(z) - Phi(z_lo)) / Z for e > a_lo, else 0 (SPEC_ANALYSIS v4 section 2). Z underflowing
    // to 0 means no mass above the bound at all: every window is empty.
    if (e <= tr.a_lo || !(tr.Z > 0.0)) return {0.0, 0.0, 0.0, 0.0};
    const double z = std::log(e) / sigma;
    const double phi = INV_SQRT_2PI * std::exp(-0.5 * z * z);
    const double Qz = 0.5 * std::erfc(z / std::sqrt(2.0));          // 1 - Phi(z)
    // the difference of the two cdfs taken in the tail where it is accurate
    const double num = tr.z_lo < 0.0 ? 0.5 * std::erfc(-z / std::sqrt(2.0)) - tr.Phi_lo : tr.Z - Qz;
    const double one_minus_F = Qz / tr.Z;
    const double sZ = sigma * tr.Z;
    return {num / tr.Z, phi / (e * sZ), (-phi * z + tr.phi_lo * tr.z_lo * one_minus_F) / sZ,
            tr.phi_lo * one_minus_F / sZ};
}

bool feasible(const ArrayXd & s) { return (s > 0).all() && s.sum() < 1.0; }

}  // namespace

namespace pcm_demand::detail {

// Vertical model for one draw, written into global-index outputs (which must be zeroed).
// Products off the frontier get zero share and zero derivatives.
void cond_vertical(const ArrayXd & d, const ArrayXd & p, double sigma, double a_lo,
                   ArrayXd & s, MatrixXd * Jd, ArrayXd * dsig, MatrixXd * Jp, ArrayXd * dlogc)
{
    const std::vector<int> ind = index_positive_shares(d, p);
    const int m = static_cast<int>(ind.size());
    if (m == 0) return;

    // endpoints e_0..e_m (e_m = 0) and the price gaps they divide by
    std::vector<double> e(m + 1, 0.0), gap(m, 0.0);
    gap[0] = p[ind[0]];
    e[0] = d[ind[0]] / gap[0];
    for (int k = 1; k < m; ++k) {
        gap[k] = p[ind[k]] - p[ind[k - 1]];
        e[k] = (d[ind[k]] - d[ind[k - 1]]) / gap[k];
    }
    const Trunc tr = make_trunc(a_lo, sigma);
    std::vector<LnTerms> t(m + 1);
    for (int k = 0; k <= m; ++k) t[k] = ln_terms(e[k], sigma, tr);

    for (int k = 0; k < m; ++k) s[ind[k]] = t[k].F - t[k + 1].F;
    if (!Jd) return;

    // s_k = F(e_k) - F(e_{k+1}); e_k depends on d_k (+1/gap_k) and d_{k-1} (-1/gap_k)
    for (int k = 0; k < m; ++k) {
        const int gk = ind[k];
        (*dsig)[gk] = t[k].g - t[k + 1].g;
        if (dlogc) (*dlogc)[gk] = t[k].h - t[k + 1].h;
        // through e_k
        (*Jd)(gk, gk) += t[k].f / gap[k];
        (*Jp)(gk, gk) += t[k].f * (-e[k] / gap[k]);
        if (k > 0) {
            (*Jd)(gk, ind[k - 1]) += -t[k].f / gap[k];
            (*Jp)(gk, ind[k - 1]) += t[k].f * (e[k] / gap[k]);
        }
        // through e_{k+1} (absent for the last product: e_m = 0 is fixed)
        if (k + 1 < m) {
            const double fk1 = t[k + 1].f, g1 = gap[k + 1];
            (*Jd)(gk, ind[k + 1]) -= fk1 / g1;
            (*Jd)(gk, gk) -= -fk1 / g1;
            (*Jp)(gk, ind[k + 1]) -= fk1 * (-e[k + 1] / g1);
            (*Jp)(gk, gk) -= fk1 * (e[k + 1] / g1);
        }
    }
}

// For every product, the smallest increase in delta_j that puts it on at least one consumer's
// frontier. For draw i the frontier is the increasing part of the upper convex hull of the points
// (p_k, dc_ik) together with the origin; a product below the hull by g (in conditional-quality units)
// needs delta_j raised by c_i * g for that consumer. The lift is the minimum over draws. Products
// already on some frontier get zero. Cost: one hull per draw, about one share evaluation.
ArrayXd frontier_lift(const ArrayXd & delta, const Market & mk, const Params & prm)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    ArrayXd lift = ArrayXd::Constant(N, std::numeric_limits<double>::infinity());
    const int n_blocks = std::max(1, (R + 63) / 64);
    std::vector<ArrayXd> part(n_blocks, lift);
#pragma omp parallel for schedule(dynamic) if(n_blocks > 1)
    for (int b = 0; b < n_blocks; ++b) {
        ArrayXd & acc = part[b];
        VectorXd t(K);
        std::vector<int> hull;
        hull.reserve(N + 1);
        for (int i = (R * b) / n_blocks; i < (R * (b + 1)) / n_blocks; ++i) {
            t.setZero();
            for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
            const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
            const double c = std::exp(log_c);
            ArrayXd dc = (delta + (mk.x * t).array()) / c;
            // With a bound, the hull of the sheared qualities dc - a_lo p: its increasing part holds exactly
            // the vertices with entry slope above a_lo, the products with a window of positive mass. The
            // vertical gap, and so the lift, is the same in both coordinates.
            if (prm.alpha_lo > 0.0) dc -= (prm.alpha_lo / c) * mk.p;
            // upper hull by price (prices are sorted); index -1 stands for the origin (0, 0)
            auto px = [&](int k) { return k < 0 ? 0.0 : mk.p[k]; };
            auto qy = [&](int k) { return k < 0 ? 0.0 : dc[k]; };
            hull.clear();
            hull.push_back(-1);
            for (int k = 0; k < N; ++k) {
                while (hull.size() >= 2) {
                    const int a = hull[hull.size() - 2], bq = hull.back();
                    const double cross = (px(bq) - px(a)) * (qy(k) - qy(a)) - (qy(bq) - qy(a)) * (px(k) - px(a));
                    if (cross >= 0) hull.pop_back(); else break;
                }
                hull.push_back(k);
            }
            // keep the increasing part: drop vertices after the maximum quality
            double top = 0.0;
            size_t end = 1;
            for (size_t h = 1; h < hull.size(); ++h) {
                if (qy(hull[h]) > top) { top = qy(hull[h]); end = h + 1; }
            }
            // evaluate the hull at each product's price
            size_t seg = 0;
            for (int k = 0; k < N; ++k) {
                while (seg + 1 < end && px(hull[seg + 1]) < mk.p[k]) ++seg;
                double h;
                if (seg + 1 >= end) h = top;
                else {
                    const int a = hull[seg], bq = hull[seg + 1];
                    const double w = (mk.p[k] - px(a)) / (px(bq) - px(a));
                    h = qy(a) + w * (qy(bq) - qy(a));
                }
                const double gap = std::max(0.0, h - dc[k]);
                acc[k] = std::min(acc[k], c * gap);
            }
        }
    }
    for (const ArrayXd & acc : part) lift = lift.min(acc);
    return lift;
}

ArrayXd initial_guess(const ArrayXd & s_data, const ArrayXd & p, const Params & prm)
{
    if (!(prm.alpha_lo > 0.0)) return pcm_share::initial_guess(s_data, p, prm.sigma_p);
    // F_T(e_k) = cum_k means Phi(z(e_k)) = Phi_lo + Z cum_k: the untruncated quantile at Phi_lo + Z cum_k.
    // pcm_share::initial_guess takes quantiles at the shares cumulated from each product up, so scale the
    // shares by Z and append one dearer dummy product of share Phi_lo; the real products' guesses do not
    // depend on the dummy's price.
    const Trunc tr = make_trunc(prm.alpha_lo, prm.sigma_p);
    const int N = static_cast<int>(s_data.size());
    if (N == 0) return ArrayXd();
    ArrayXd s2(N + 1), p2(N + 1);
    s2.head(N) = tr.Z * s_data;
    s2[N] = tr.Phi_lo;
    p2.head(N) = p;
    p2[N] = p[N - 1] + std::max(1.0, std::abs(p[N - 1]));
    return pcm_share::initial_guess(s2, p2, prm.sigma_p).head(N);
}

}  // namespace pcm_demand::detail

namespace pcm_demand {

Shares shares(const ArrayXd & delta, const Market & mk, const Params & prm, bool with_derivatives)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    if (mk.x.rows() != N || mk.p.size() != N)
        throw std::runtime_error("pcm_demand::shares: x/p/delta sizes differ");
    if (mk.taste_basis.rows() != R || mk.taste_basis.cols() != M || (int)mk.taste_col.size() != M)
        throw std::runtime_error("pcm_demand::shares: taste_basis must be R x M and taste_col size M");
    if (Q > 0 && (mk.scale_basis.rows() != R || mk.scale_basis.cols() != Q))
        throw std::runtime_error("pcm_demand::shares: scale_basis must be R x Q");
    for (int c : mk.taste_col)
        if (c < 0 || c >= K) throw std::runtime_error("pcm_demand::shares: taste_col out of range");
    for (int j = 1; j < N; ++j)
        if (!(mk.p[j] > mk.p[j - 1])) throw std::runtime_error("pcm_demand::shares: prices must be strictly increasing");
    if (N > 0 && !(mk.p[0] > 0)) throw std::runtime_error("pcm_demand::shares: prices must be positive");
    if (!(prm.sigma_p > 0)) throw std::runtime_error("pcm_demand::shares: sigma_p must be positive");
    if (!(prm.alpha_lo >= 0) || !std::isfinite(prm.alpha_lo))
        throw std::runtime_error("pcm_demand::shares: alpha_lo must be finite and non-negative");

    Shares out;
    out.s = ArrayXd::Zero(N);
    if (with_derivatives) {
        out.d_delta = MatrixXd::Zero(N, N);
        out.d_theta = MatrixXd::Zero(N, M + Q + 1);
        out.d_price = MatrixXd::Zero(N, N);
    }

    // Draws are independent: fixed blocks of 64 draws each accumulate partial sums in parallel,
    // and the blocks are added in index order, so results are bit-identical for any thread count.
    const int block = 64;
    const int n_blocks = std::max(1, (R + block - 1) / block);
    std::vector<Shares> part(n_blocks);

#pragma omp parallel for schedule(dynamic) if(n_blocks > 1)
    for (int b = 0; b < n_blocks; ++b) {
        Shares & acc = part[b];
        acc.s = ArrayXd::Zero(N);
        if (with_derivatives) {
            acc.d_delta = MatrixXd::Zero(N, N);
            acc.d_theta = MatrixXd::Zero(N, M + Q + 1);
            acc.d_price = MatrixXd::Zero(N, N);
        }
        ArrayXd s_c(N), dsig(N), dlogc(N);
        MatrixXd Jd(N, N), Jp(N, N);
        VectorXd t(K);
        const int i0 = static_cast<int>((static_cast<long>(R) * b) / n_blocks);
        const int i1 = static_cast<int>((static_cast<long>(R) * (b + 1)) / n_blocks);

        for (int i = i0; i < i1; ++i) {
            t.setZero();
            for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
            const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
            const double c = std::exp(log_c);
            const ArrayXd dc = (delta + (mk.x * t).array()) / c;

            s_c.setZero();
            if (with_derivatives) { Jd.setZero(); Jp.setZero(); dsig.setZero(); dlogc.setZero(); }
            detail::cond_vertical(dc, mk.p, prm.sigma_p, prm.alpha_lo / c, s_c,
                                  with_derivatives ? &Jd : nullptr, &dsig, &Jp, &dlogc);

            const double w = mk.weights[i];
            acc.s += w * s_c;
            if (!with_derivatives) continue;

            acc.d_delta.noalias() += (w / c) * Jd;
            acc.d_price.noalias() += w * Jp;
            for (int m = 0; m < M; ++m)
                acc.d_theta.col(m).noalias() += (w * mk.taste_basis(i, m) / c) * (Jd * mk.x.col(mk.taste_col[m]));
            if (Q > 0) {
                // log c moves s through dc = y / c (-Jd dc) and, with a bound, through a_lo = alpha_lo / c
                VectorXd Jdc = Jd * dc.matrix();
                if (prm.alpha_lo > 0.0) Jdc -= dlogc.matrix();
                for (int q = 0; q < Q; ++q)
                    acc.d_theta.col(M + q).noalias() += (-w * mk.scale_basis(i, q)) * Jdc;
            }
            acc.d_theta.col(M + Q) += w * dsig.matrix();
        }
    }
    for (const Shares & acc : part) {
        out.s += acc.s;
        if (with_derivatives) {
            out.d_delta += acc.d_delta;
            out.d_theta += acc.d_theta;
            out.d_price += acc.d_price;
        }
    }
    return out;
}

InversionResult invert_from(const ArrayXd & s_data, const Market & mk, const Params & prm,
                            const ArrayXd & delta0, double tol, unsigned max_evals, unsigned & evals);

InversionResult invert(const ArrayXd & s_data, const Market & mk, const Params & prm,
                       const ArrayXd & delta0, double tol, unsigned max_evals)
{
    if ((s_data <= 0).any() || s_data.sum() >= 1.0)
        throw std::runtime_error("pcm_demand::invert: data shares must be positive and sum below one");
    unsigned evals = 0;
    InversionResult res = invert_from(s_data, mk, prm, delta0, tol, max_evals, evals);
    if (res.converged) { res.share_evaluations = evals; return res; }

    // Continuation in the heterogeneity: at lambda = 0 (no taste or scale heterogeneity) the
    // vertical-model guess is exact; raise lambda to 1, warm-starting each step from the last.
    {
        ArrayXd delta = detail::initial_guess(s_data, mk.p, prm);
        double lambda = 0.0, step = 0.25;
        unsigned iters = 0;
        while (lambda < 1.0 && evals < max_evals && step > 1e-5) {
            const double target = std::min(1.0, lambda + step);
            Params pl{prm.theta * target, prm.phi * target, prm.sigma_p, prm.alpha_lo};
            InversionResult r = invert_from(s_data, mk, pl, delta, tol, max_evals, evals);
            iters += r.iterations;
            if (r.converged) { delta = r.delta; lambda = target; step = std::min(0.5, step * 1.5); }
            else             { step *= 0.5; }
        }
        if (lambda >= 1.0) {
            res = invert_from(s_data, mk, prm, delta, tol, max_evals, evals);
            res.iterations += iters;
            if (res.converged) { res.share_evaluations = evals; return res; }
        }
    }

    // Path following in shares: from a feasible delta with known shares s0, solve for the targets
    // s(tau) = (1 - tau) s0 + tau s_data, tau 0 -> 1, each solve warm-started from the last.
    // Newton directions go wrong across kinks (brands entering a consumer's frontier); short
    // steps keep every solve inside one smooth piece most of the time.
    {
        ArrayXd d = res.delta.size() == s_data.size() ? res.delta : ArrayXd();
        ArrayXd s0;
        if (d.size()) { ++evals; s0 = shares(d, mk, prm, false).s; }
        if (d.size() && feasible(s0)) {
            double tau = 0.0, step = 0.1;
            unsigned iters = 0;
            while (tau < 1.0 && evals < max_evals && step > 1e-7) {
                const double target = std::min(1.0, tau + step);
                const ArrayXd st = (1.0 - target) * s0 + target * s_data;
                InversionResult r = invert_from(st, mk, prm, d, tol, max_evals, evals);
                iters += r.iterations;
                if (r.converged) { d = r.delta; tau = target; step = std::min(0.5, step * 2.0); }
                else             { step *= 0.5; }
            }
            if (tau >= 1.0) {
                res = invert_from(s_data, mk, prm, d, tol, max_evals, evals);
                res.iterations += iters;
            }
        }
    }
    res.share_evaluations = evals;
    return res;
}

InversionResult invert_from(const ArrayXd & s_data, const Market & mk, const Params & prm,
                            const ArrayXd & delta0, double tol, unsigned max_evals, unsigned & evals)
{
    const int N = static_cast<int>(s_data.size());
    InversionResult res{ArrayXd(), 0.0, 0, 0, false};
    auto S = [&](const ArrayXd & d) { ++evals; return shares(d, mk, prm, false).s; };
    const ArrayXd log_sd = s_data.log();
    // Residual in log shares: a 0.1% brand weighs as much as a 10% brand.
    auto logres = [&](const ArrayXd & s) { return (s.log() - log_sd).eval(); };

    // starting point: caller's warm start, else the no-heterogeneity vertical guess; either is
    // repaired if some products have zero share
    ArrayXd delta = delta0.size() == N ? delta0 : detail::initial_guess(s_data, mk.p, prm);
    ArrayXd s = S(delta);
    {
        // repair: lift every zero-share product onto some consumer's frontier plus a margin that
        // grows with each round; if the outside good has vanished lower everything
        double margin = 1e-5;
        for (int k = 0; k < 60 && !feasible(s) && evals < max_evals; ++k) {
            const double span = std::max(delta.maxCoeff() - delta.minCoeff(), 1e-3 * delta.abs().maxCoeff() + 1e-8);
            if (s.sum() >= 1.0) delta -= 0.05 * span;
            const ArrayXd lift = detail::frontier_lift(delta, mk, prm);
            ++evals;
            for (int j = 0; j < N; ++j)
                if (s[j] <= 0 && std::isfinite(lift[j])) delta[j] += lift[j] + margin * span;
            s = S(delta);
            if (k % 10 == 9) margin *= 3.0;
        }
        if (!feasible(s)) { res.delta = delta; res.max_abs_residual = INFINITY; return res; }
    }

    double mu = 0.0;
    for (unsigned iter = 0; iter < 150 && evals < max_evals; ++iter) {
        res.iterations = iter;
        Shares sh = shares(delta, mk, prm, true);
        ++evals;
        res.max_abs_residual = (sh.s - s_data).abs().maxCoeff();
        if (res.max_abs_residual < tol) { res.converged = true; break; }
        const ArrayXd r = logres(sh.s);
        const double merit = 0.5 * r.matrix().squaredNorm();
        // Jacobian of log shares: rows of ds/d delta divided by s
        const MatrixXd J = sh.s.inverse().matrix().asDiagonal() * sh.d_delta;

        bool accepted = false;
        for (int attempt = 0; attempt < 6 && !accepted && evals < max_evals; ++attempt) {
            VectorXd step;
            if (mu == 0.0) {
                step = J.partialPivLu().solve(-r.matrix());
                if (!step.allFinite()) { mu = 1e-8; continue; }
            } else {
                MatrixXd A = J.transpose() * J;
                A.diagonal().array() += mu * std::max(A.diagonal().maxCoeff(), 1e-300);
                step = A.ldlt().solve(-(J.transpose() * r.matrix()));
            }
            const double slope = r.matrix().dot(J * step);
            double lambda = 1.0;
            for (int k = 0; k < 14 && evals < max_evals; ++k, lambda *= 0.5) {
                const ArrayXd cand = delta + lambda * step.array();
                const ArrayXd sc = S(cand);
                if (!feasible(sc)) continue;
                const double mc = 0.5 * logres(sc).matrix().squaredNorm();
                if (mc < (slope < 0 ? merit + 1e-4 * lambda * slope : merit)) {
                    delta = cand; accepted = true; break;
                }
            }
            if (accepted) { mu = mu / 10.0; if (mu < 1e-10) mu = 0.0; }
            else          { mu = (mu == 0.0) ? 1e-8 : mu * 100.0; }
        }
        if (!accepted) break;
    }
    if (!res.converged) {
        const ArrayXd r = S(delta) - s_data;
        res.max_abs_residual = r.abs().maxCoeff();
        res.converged = res.max_abs_residual < tol;
    }
    res.delta = delta;
    return res;
}

}  // namespace pcm_demand
