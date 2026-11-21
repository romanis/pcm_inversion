#include "pcm_invert.hpp"
#include "pcm_geometry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pcm_invert {

using pcm_aggregate::Market;
using pcm_aggregate::Params;

namespace {

bool feasible(const ArrayXd & s) { return (s > 0).all() && s.sum() < 1.0; }

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::string fmt_stage(const char * name, bool ok, double secs, unsigned evals) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%s %s %.3gs %uev", name, ok ? "ok" : "failed", secs, evals);
    return buf;
}

void append(std::string & stages, const std::string & s) {
    if (!stages.empty()) stages += "; ";
    stages += s;
}

// Every k-th draw of mk (draws 0, k, 2k, ...), weights rescaled so they sum to the market's draw mass: a coarser
// quadrature of the same market for the draw homotopy of the cold stage.
pcm_aggregate::Market thin_draws(const pcm_aggregate::Market & mk, int k) {
    const int R = static_cast<int>(mk.weights.size());
    const int n = (R + k - 1) / k;
    pcm_aggregate::Market out{mk.x, mk.p, ArrayXd(n), MatrixXd(n, mk.taste_basis.cols()), mk.taste_col,
                           MatrixXd(n, mk.scale_basis.cols())};
    for (int r = 0; r < n; ++r) {
        out.weights[r] = mk.weights[r * k];
        out.taste_basis.row(r) = mk.taste_basis.row(r * k);
        if (mk.scale_basis.cols() > 0) out.scale_basis.row(r) = mk.scale_basis.row(r * k);
    }
    const double sub = out.weights.sum();
    if (sub > 0) out.weights *= mk.weights.sum() / sub;
    return out;
}

pcm_aggregate::Params interpolate(const pcm_aggregate::Params & a, const pcm_aggregate::Params & b, double tau) {
    // the bound moves along the path only when the warm start was solved under another one (continuation in a_lo)
    const double lo = a.lo == b.lo ? b.lo : (1.0 - tau) * a.lo + tau * b.lo;
    return pcm_aggregate::Params{(1.0 - tau) * a.theta + tau * b.theta, (1.0 - tau) * a.phi + tau * b.phi,
                              (1.0 - tau) * a.sigma_p + tau * b.sigma_p, lo, b.bound_kind};
}

}  // namespace

// robust_restart: a second attempt with the robust repair when a repaired start does not converge (the caller's warm
// start only; the continuation steps below keep the original repair)
InversionResult invert_from(const ArrayXd & s_data, const Market & mk, const Params & prm,
                            const ArrayXd & delta0, double tol, unsigned max_evals, unsigned & evals,
                            bool robust_restart = false);

InversionResult invert(const ArrayXd & s_data, const Market & mk, const Params & prm,
                       const ArrayXd & delta0, double tol, unsigned max_evals)
{
    if ((s_data <= 0).any() || s_data.sum() >= 1.0)
        throw std::runtime_error("pcm_invert::invert: data shares must be positive and sum below one");
    unsigned evals = 0;
    InversionResult res = invert_from(s_data, mk, prm, delta0, tol, max_evals, evals, delta0.size() == s_data.size());
    if (res.converged) { res.share_evaluations = evals; return res; }

    // Continuation in the heterogeneity: at lambda = 0 (no taste or scale heterogeneity) the
    // vertical-model guess is exact; raise lambda to 1, warm-starting each step from the last.
    {
        ArrayXd delta = pcm_geometry::initial_guess(s_data, mk.p, prm.sigma_p, prm.lo);
        double lambda = 0.0, step = 0.25;
        unsigned iters = 0;
        while (lambda < 1.0 && evals < max_evals && step > 1e-5) {
            const double target = std::min(1.0, lambda + step);
            Params pl = prm;
            pl.theta = prm.theta * target;
            pl.phi = prm.phi * target;
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
        if (d.size()) { ++evals; s0 = pcm_aggregate::shares(d, mk, prm, false).s; }
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
                            const ArrayXd & delta0, double tol, unsigned max_evals, unsigned & evals,
                            bool robust_restart)
{
    const int N = static_cast<int>(s_data.size());
    auto S = [&](const ArrayXd & d) { ++evals; return pcm_aggregate::shares(d, mk, prm, false).s; };
    const ArrayXd log_sd = s_data.log();
    // Residual in log shares: a 0.1% brand weighs as much as a 10% brand.
    auto logres = [&](const ArrayXd & s) { return (s.log() - log_sd).eval(); };

    // repair: lift every zero-share product onto some consumer's frontier plus a margin that grows with each round; if
    // the outside good has vanished lower everything. ``robust`` adds the two rules of the second attempt (below).
    auto repair = [&](ArrayXd & delta, ArrayXd & s, bool robust) {
        double margin = 1e-5;
        std::vector<char> lifted(N, 0);
        bool repaired = false;
        for (int k = 0; k < 60 && !feasible(s) && evals < max_evals; ++k) {
            const double span = std::max(delta.maxCoeff() - delta.minCoeff(), 1e-3 * delta.abs().maxCoeff() + 1e-8);
            if (s.sum() >= 1.0) delta -= 0.05 * span;
            const ArrayXd lift = pcm_aggregate::frontier_lift(delta, mk, prm);
            ++evals;
            bool cycling = false;
            if (robust)
                for (int j = 0; j < N; ++j) cycling |= s[j] <= 0 && lifted[j];
            for (int j = 0; j < N; ++j) {
                if (s[j] <= 0 && std::isfinite(lift[j])) {
                    delta[j] += lift[j] + margin * span;
                    lifted[j] = 1;
                } else if (cycling && lifted[j]) {
                    delta[j] += margin * span;
                }
            }
            repaired = true;
            s = S(delta);
            if (k % 10 == 9) margin *= 3.0;
        }
        if (!robust || !repaired || !feasible(s)) return;
        // A product lifted onto a hull by the margin alone holds a sliver of one draw's window (shares of 1e-19 against
        // targets of 1e-6): its row of the log-share Jacobian is 1e17 times the others, so the Newton step is no use
        // and every damped step is negligible. Raise each product whose share is below 1% of its target by an
        // increment that doubles while it stays below (halved for all when a round would make some share zero).
        const double span = std::max(delta.maxCoeff() - delta.minCoeff(), 1e-3 * delta.abs().maxCoeff() + 1e-8);
        ArrayXd inc = ArrayXd::Constant(N, margin * span);
        for (int k = 0; k < 60 && evals < max_evals; ++k) {
            const Eigen::Array<bool, Eigen::Dynamic, 1> low = s < 1e-2 * s_data;
            if (!low.any()) break;
            const ArrayXd cand = delta + low.cast<double>() * inc;
            const ArrayXd sc = S(cand);
            if (!feasible(sc)) { inc *= 0.5; continue; }
            delta = cand;
            s = sc;
            inc = low.select(2.0 * inc, inc);
        }
    };

    // damped Newton on the log shares from a feasible delta
    auto newton = [&](ArrayXd delta) {
        InversionResult res{ArrayXd(), 0.0, 0, 0, false};
        double mu = 0.0;
        for (unsigned iter = 0; iter < 150 && evals < max_evals; ++iter) {
            res.iterations = iter;
            pcm_aggregate::Shares sh = pcm_aggregate::shares(delta, mk, prm, true);
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
    };

    // starting point: caller's warm start, else the no-heterogeneity vertical guess; either is
    // repaired if some products have zero share
    const ArrayXd start = delta0.size() == N ? delta0 : pcm_geometry::initial_guess(s_data, mk.p, prm.sigma_p, prm.lo);
    ArrayXd delta = start;
    ArrayXd s = S(delta);
    const bool needed_repair = !feasible(s);
    repair(delta, s, false);
    InversionResult res{delta, INFINITY, 0, 0, false};
    if (feasible(s)) res = newton(delta);
    if (res.converged || !needed_repair || !robust_restart || evals >= max_evals) return res;

    // Second attempt, only from a caller's warm start that needed a repair and did not converge (a cold start keeps the
    // continuation of ``invert``, so every inversion that converged before is unchanged). Products of shares near 1e-7 are bought by one draw each; two of them on the same draw
    // displace each other, so lifting one onto that draw's hull drops the other below the new chord, round after
    // round (GMY 2015 at a 2e-5 step: 60 rounds and no feasible start). Once a product lifted before is at zero again,
    // every product lifted so far is raised by the margin too, and the pair moves up together onto the hull; then the
    // products the lift left with a sliver of a window are raised toward their targets before Newton starts.
    ArrayXd delta2 = start;
    ArrayXd s2 = S(delta2);
    repair(delta2, s2, true);
    if (!feasible(s2)) return res;
    InversionResult res2 = newton(delta2);
    if (res2.converged || !std::isfinite(res.max_abs_residual) || res2.max_abs_residual < res.max_abs_residual)
        return res2;
    return res;
}

void solve(const ArrayXd & s_data, const Market & mk, const Params & prm, const ArrayXd & delta_warm,
           const Params * prm_warm, const Options & opt, Solution & out)
{
    const int N = static_cast<int>(s_data.size());
    ArrayXd & best_delta = out.delta;
    double & best_res = out.residual;   // finite once some stage reached a feasible delta
    bool & converged = out.converged;
    unsigned & evals = out.evals;

    // Keeps the delta with the smallest finite residual over the stages (a converged one wins).
    auto record = [&](const InversionResult & r) {
        evals += r.share_evaluations;
        if (r.delta.size() != N) return;
        const bool feasible = std::isfinite(r.max_abs_residual);
        if (r.converged || (feasible && r.max_abs_residual < best_res) || best_delta.size() == 0) {
            best_delta = r.delta;
            if (feasible) best_res = r.max_abs_residual;
        }
        if (r.converged) converged = true;
    };

    {
        // 2. warm stage
        if (delta_warm.size() == N) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            record(invert(s_data, mk, prm, delta_warm, opt.tol, opt.warm_evals));
            append(out.stages, fmt_stage("warm", converged, seconds_since(ts), evals - e0));
        }
        // 3. theta-path continuation from the theta the warm start was solved at
        if (!converged && prm_warm && delta_warm.size() == N) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            ArrayXd d = delta_warm;
            double tau = 0.0, step = 0.5;
            while (tau < 1.0 && step >= 0.01 && seconds_since(ts) < opt.path_seconds) {
                const double target = std::min(1.0, tau + step);
                InversionResult r =
                    invert(s_data, mk, interpolate(*prm_warm, prm, target), d, opt.tol, opt.path_evals);
                evals += r.share_evaluations;
                if (r.converged) { d = r.delta; tau = target; step *= 2.0; }
                else             { step *= 0.5; }
            }
            if (tau >= 1.0) {
                // d passed invert's residual test at tau = 1, that is at prm itself
                best_delta = d;
                best_res = 0.0;
                converged = true;
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), " tau=%.3g", tau);
            append(out.stages, fmt_stage("path", converged, seconds_since(ts), evals - e0) + buf);
        }
        // 4a. draw homotopy: cold on a thinned market, then each finer level warm-started from the coarser one
        if (!converged && opt.cold_draws > 0) {
            const long R = mk.weights.size();
            long k = 1;
            while (R / (k * opt.cold_draw_factor) >= static_cast<long>(opt.cold_draws)) k *= opt.cold_draw_factor;
            ArrayXd d;
            bool ok = k > 1;
            for (; ok && k > 1; k /= opt.cold_draw_factor) {
                const auto ts = Clock::now();
                const pcm_aggregate::Market sub = thin_draws(mk, static_cast<int>(k));
                const InversionResult r =
                    invert(s_data, sub, prm, d, opt.tol, d.size() ? opt.warm_evals : opt.cold_evals);
                evals += r.share_evaluations;
                ok = r.converged;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "draws R=%ld %s %.3gs %uev", static_cast<long>(sub.weights.size()),
                              ok ? "ok" : "failed", seconds_since(ts), r.share_evaluations);
                append(out.stages, buf);
                if (ok) d = r.delta;
            }
            if (ok) {
                const auto ts = Clock::now();
                const unsigned e0 = evals;
                record(invert(s_data, mk, prm, d, opt.tol, opt.warm_evals));
                append(out.stages, fmt_stage("draws warm", converged, seconds_since(ts), evals - e0));
            }
        }
        // 4. cold stage
        if (!converged) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            record(invert(s_data, mk, prm, ArrayXd(), opt.tol, opt.cold_evals));
            append(out.stages, fmt_stage("cold", converged, seconds_since(ts), evals - e0));
        }
    }
}

}  // namespace pcm_invert
