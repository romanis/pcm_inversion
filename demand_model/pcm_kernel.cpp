#include "pcm_kernel.hpp"
#include "pcm_market_share.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pcm_kernel {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Sets the calling thread's OpenMP team size (a per-thread ICV) and restores it on scope exit.
class ThreadScope {
public:
    explicit ThreadScope(int n) : saved_(omp_get_max_threads()) { omp_set_num_threads(n); }
    ~ThreadScope() { omp_set_num_threads(saved_); }
    ThreadScope(const ThreadScope &) = delete;
    ThreadScope & operator=(const ThreadScope &) = delete;
private:
    int saved_;
};

void require(bool ok, const std::string & what) {
    if (!ok) throw std::invalid_argument(what);
}

void validate_market(const pcm_demand::Market & mk, const pcm_demand::Params & prm, int N) {
    const int R = static_cast<int>(mk.weights.size());
    const int K = static_cast<int>(mk.x.cols());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    require(mk.x.rows() == N && mk.p.size() == N, "market x and p must have one row per product");
    require(mk.taste_basis.rows() == R && mk.taste_basis.cols() == M && static_cast<int>(mk.taste_col.size()) == M,
            "taste_basis must be R x M and taste_col of size M (M = len(theta))");
    require(Q == 0 || (mk.scale_basis.rows() == R && mk.scale_basis.cols() == Q),
            "scale_basis must be R x Q (Q = len(phi))");
    for (int c : mk.taste_col) require(c >= 0 && c < K, "taste_col out of range");
    for (int j = 0; j < N; ++j) {
        require(std::isfinite(mk.p[j]), "prices must be finite");
        if (j > 0) require(mk.p[j] > mk.p[j - 1], "prices must be strictly increasing");
    }
    require(N == 0 || mk.p[0] > 0, "prices must be positive");
    require(R > 0, "market needs at least one draw");
    require(mk.weights.allFinite() && (mk.weights >= 0).all(), "draw weights must be finite and non-negative");
    require(mk.x.allFinite() && mk.taste_basis.allFinite(), "x and taste_basis must be finite");
    require(Q == 0 || mk.scale_basis.allFinite(), "scale_basis must be finite");
    require(prm.theta.allFinite() && prm.phi.allFinite(), "theta and phi must be finite");
    require(std::isfinite(prm.sigma_p) && prm.sigma_p > 0, "sigma_p must be positive and finite");
    require(std::isfinite(prm.lo) && prm.lo >= 0, "the bound (alpha_lo or a_lo) must be finite and non-negative");
}

void validate_micro(const MicroRequest & mr, int N, int R) {
    require(mr.H.rows() == R, "micro H must have one row per draw");
    require(mr.U.rows() == N, "micro U must have one row per product");
    require(static_cast<int>(mr.h_of_part.size()) == mr.U.cols(), "micro h_of_part must have one entry per column of U");
    for (int h : mr.h_of_part) require(h >= 0 && h < mr.H.cols(), "micro h_of_part out of range");
    require(mr.H.allFinite() && mr.U.allFinite(), "micro H and U must be finite");
}

// Partial sums of one block of draws.
struct Block {
    ArrayXd s;
    MatrixXd ds_ddelta, ds_dtheta, ds_dp;
    VectorXd micro;
    MatrixXd micro_d_delta, micro_d_theta;
};

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
pcm_demand::Market thin_draws(const pcm_demand::Market & mk, int k) {
    const int R = static_cast<int>(mk.weights.size());
    const int n = (R + k - 1) / k;
    pcm_demand::Market out{mk.x, mk.p, ArrayXd(n), MatrixXd(n, mk.taste_basis.cols()), mk.taste_col,
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

pcm_demand::Params interpolate(const pcm_demand::Params & a, const pcm_demand::Params & b, double tau) {
    // the bound moves along the path only when the warm start was solved under another one (continuation in a_lo)
    const double lo = a.lo == b.lo ? b.lo : (1.0 - tau) * a.lo + tau * b.lo;
    return pcm_demand::Params{(1.0 - tau) * a.theta + tau * b.theta, (1.0 - tau) * a.phi + tau * b.phi,
                              (1.0 - tau) * a.sigma_p + tau * b.sigma_p, lo, b.bound_kind};
}

}  // namespace

const char * status_name(Status s) {
    switch (s) {
        case Status::OK: return "ok";
        case Status::NOT_CONVERGED: return "not_converged";
        case Status::INFEASIBLE_START: return "infeasible_start";
        case Status::SINGULAR_DS_DDELTA: return "singular_ds_ddelta";
        case Status::INVALID_INPUT: return "invalid_input";
    }
    return "invalid_input";
}

MarketEvaluation evaluate_market(const ArrayXd & delta, const pcm_demand::Market & mk,
                                 const pcm_demand::Params & prm, const MicroRequest * micro,
                                 bool derivatives, bool price_derivatives)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    const int P = M + Q + 1;
    validate_market(mk, prm, N);
    require(delta.allFinite(), "delta must be finite");
    if (micro) validate_micro(*micro, N, R);
    const int Pm = micro ? static_cast<int>(micro->U.cols()) : 0;
    const bool jac = derivatives || price_derivatives;

    // Same block partition as pcm_demand::shares: fixed blocks of 64 draws, summed in block order.
    const int block = 64;
    const int n_blocks = std::max(1, (R + block - 1) / block);
    std::vector<Block> part(n_blocks);
    pcm_demand::apply_thread_budget();

#pragma omp parallel for schedule(dynamic) if(n_blocks > 1)
    for (int b = 0; b < n_blocks; ++b) {
        Block & acc = part[b];
        acc.s = ArrayXd::Zero(N);
        if (derivatives) {
            acc.ds_ddelta = MatrixXd::Zero(N, N);
            acc.ds_dtheta = MatrixXd::Zero(N, P);
        }
        if (price_derivatives) acc.ds_dp = MatrixXd::Zero(N, N);
        if (micro) {
            acc.micro = VectorXd::Zero(Pm);
            if (derivatives) {
                acc.micro_d_delta = MatrixXd::Zero(Pm, N);
                acc.micro_d_theta = MatrixXd::Zero(Pm, P);
            }
        }
        ArrayXd s_c = ArrayXd::Zero(N), dsig = ArrayXd::Zero(N), dlogc = ArrayXd::Zero(N);
        const bool bound = pcm_demand::bound_moves_with_scale(prm);   // log c acts through a_lo = lo / c
        MatrixXd Jd, Jp;
        if (jac) { Jd = MatrixXd::Zero(N, N); Jp = MatrixXd::Zero(N, N); }
        VectorXd t(K);
        std::vector<int> S;
        S.reserve(N);
        MatrixXd Jc, uJ;
        VectorXd us, ud, ul, xs;
        const int i0 = static_cast<int>((static_cast<long>(R) * b) / n_blocks);
        const int i1 = static_cast<int>((static_cast<long>(R) * (b + 1)) / n_blocks);

        for (int i = i0; i < i1; ++i) {
            t.setZero();
            for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
            const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
            const double c = std::exp(log_c);
            const ArrayXd dc = (delta + (mk.x * t).array()) / c;

            s_c.setZero();
            if (jac) { dsig.setZero(); if (bound) dlogc.setZero(); }
            pcm_share::cond_share(dc, mk.p, prm.sigma_p, pcm_demand::draw_a_lo(prm, c), s_c, jac ? &Jd : nullptr,
                                              &dsig, &Jp, bound ? &dlogc : nullptr);

            const double w = mk.weights[i];
            acc.s += w * s_c;
            if (micro) us.noalias() = micro->U.transpose() * s_c.matrix();
            if (!jac) {
                for (int p = 0; p < Pm; ++p) acc.micro[p] += w * micro->H(i, micro->h_of_part[p]) * us[p];
                continue;
            }

            // Support of the conditional Jacobian: Jd is nonzero only on S x S, where S holds the
            // frontier products whose diagonal entry is nonzero (each diagonal entry is a sum of the
            // nonnegative terms f / gap that make up its row and column). Working on S x S instead
            // of N x N gives the same entries at a fraction of the cost.
            S.clear();
            for (int j = 0; j < N; ++j)
                if (Jd(j, j) != 0.0 || s_c[j] != 0.0 || dsig[j] != 0.0 || (bound && dlogc[j] != 0.0)) S.push_back(j);
            const int ns = static_cast<int>(S.size());
            Jc = Jd(S, S);
            if (derivatives) {
                acc.ds_ddelta(S, S) += (w / c) * Jc;
                for (int m = 0; m < M; ++m) {
                    xs = mk.x.col(mk.taste_col[m])(S);
                    acc.ds_dtheta(S, m) += (w * mk.taste_basis(i, m) / c) * (Jc * xs);
                }
                if (Q > 0) {
                    // log c acts through dc = y / c (-Jd dc) and, with an ALPHA bound, through a_lo = lo / c
                    VectorXd Jdc = Jc * dc(S).matrix();
                    if (bound) Jdc -= dlogc(S).matrix();
                    for (int q = 0; q < Q; ++q) acc.ds_dtheta(S, M + q) += (-w * mk.scale_basis(i, q)) * Jdc;
                }
                acc.ds_dtheta.col(M + Q) += w * dsig.matrix();
            }
            if (price_derivatives) acc.ds_dp(S, S) += w * Jp(S, S);

            if (micro) {
                if (derivatives) {
                    uJ.noalias() = micro->U(S, Eigen::all).transpose() * Jc;   // Pm x ns
                    ud.noalias() = micro->U.transpose() * dsig.matrix();
                    if (bound && Q > 0) ul.noalias() = micro->U.transpose() * dlogc.matrix();
                }
                for (int p = 0; p < Pm; ++p) {
                    const double wh = w * micro->H(i, micro->h_of_part[p]);
                    acc.micro[p] += wh * us[p];
                    if (!derivatives || wh == 0.0) continue;
                    for (int k = 0; k < ns; ++k) acc.micro_d_delta(p, S[k]) += (wh / c) * uJ(p, k);
                    for (int m = 0; m < M; ++m) {
                        double v = 0.0;
                        const auto xm = mk.x.col(mk.taste_col[m]);
                        for (int k = 0; k < ns; ++k) v += uJ(p, k) * xm[S[k]];
                        acc.micro_d_theta(p, m) += (wh * mk.taste_basis(i, m) / c) * v;
                    }
                    if (Q > 0) {
                        double v = 0.0;
                        for (int k = 0; k < ns; ++k) v += uJ(p, k) * dc[S[k]];
                        if (bound) v -= ul[p];
                        for (int q = 0; q < Q; ++q) acc.micro_d_theta(p, M + q) += -wh * mk.scale_basis(i, q) * v;
                    }
                    acc.micro_d_theta(p, M + Q) += wh * ud[p];
                }
            }
            // restore the zero state cond_share requires
            Jd(S, S).setZero();
            Jp(S, S).setZero();
        }
    }

    MarketEvaluation out;
    out.s = ArrayXd::Zero(N);
    if (derivatives) { out.ds_ddelta = MatrixXd::Zero(N, N); out.ds_dtheta = MatrixXd::Zero(N, P); }
    if (price_derivatives) out.ds_dp = MatrixXd::Zero(N, N);
    if (micro) {
        out.micro = VectorXd::Zero(Pm);
        if (derivatives) { out.micro_d_delta = MatrixXd::Zero(Pm, N); out.micro_d_theta = MatrixXd::Zero(Pm, P); }
    }
    for (const Block & acc : part) {
        out.s += acc.s;
        if (derivatives) { out.ds_ddelta += acc.ds_ddelta; out.ds_dtheta += acc.ds_dtheta; }
        if (price_derivatives) out.ds_dp += acc.ds_dp;
        if (micro) {
            out.micro += acc.micro;
            if (derivatives) { out.micro_d_delta += acc.micro_d_delta; out.micro_d_theta += acc.micro_d_theta; }
        }
    }
    return out;
}

SolveResult solve_market(const ArrayXd & s_data, const pcm_demand::Market & mk,
                         const pcm_demand::Params & prm, const ArrayXd & delta_warm,
                         const pcm_demand::Params * prm_warm, const MicroRequest * micro,
                         const SolveOptions & opt)
{
    const auto t0 = Clock::now();
    SolveResult res;
    const int N = static_cast<int>(s_data.size());
    try {
        validate_market(mk, prm, N);
        require(N > 0, "market has no products");
        require(s_data.allFinite() && (s_data > 0).all() && s_data.sum() < 1.0,
                "data shares must be positive and sum below one");
        require(delta_warm.size() == 0 || (delta_warm.size() == N && delta_warm.allFinite()),
                "delta_warm must be empty or a finite vector with one entry per product");
        if (prm_warm) {
            require(prm_warm->theta.size() == prm.theta.size() && prm_warm->phi.size() == prm.phi.size(),
                    "theta_warm must have the layout of theta");
            require(prm_warm->theta.allFinite() && prm_warm->phi.allFinite() && std::isfinite(prm_warm->sigma_p) &&
                    prm_warm->sigma_p > 0, "theta_warm must be finite with positive sigma_p");
            require(prm_warm->bound_kind == prm.bound_kind && std::isfinite(prm_warm->lo) && prm_warm->lo >= 0,
                    "theta_warm must have the bound kind of theta and a finite, non-negative bound");
        }
        if (micro) validate_micro(*micro, N, static_cast<int>(mk.weights.size()));
        require(opt.tol > 0 && opt.n_threads >= 1 && opt.rcond_min >= 0 && opt.path_seconds >= 0,
                "options need tol > 0, n_threads >= 1, rcond_min >= 0, path_seconds >= 0");
        require(opt.cold_draw_factor >= 2, "options need cold_draw_factor >= 2");
    } catch (const std::exception & e) {
        res.status = Status::INVALID_INPUT;
        res.stages = std::string("invalid input: ") + e.what();
        res.max_abs_residual = std::numeric_limits<double>::infinity();
        res.seconds = seconds_since(t0);
        return res;
    }

    ThreadScope threads(opt.n_threads);
    pcm_demand::ThreadBudgetScope budget(opt.thread_budget);
    const double inf = std::numeric_limits<double>::infinity();
    ArrayXd best_delta;
    double best_res = inf;           // finite once some stage reached a feasible delta
    bool converged = false;
    unsigned evals = 0;

    // Keeps the delta with the smallest finite residual over the stages (a converged one wins).
    auto record = [&](const pcm_demand::InversionResult & r) {
        evals += r.share_evaluations;
        if (r.delta.size() != N) return;
        const bool feasible = std::isfinite(r.max_abs_residual);
        if (r.converged || (feasible && r.max_abs_residual < best_res) || best_delta.size() == 0) {
            best_delta = r.delta;
            if (feasible) best_res = r.max_abs_residual;
        }
        if (r.converged) converged = true;
    };

    try {
        // 2. warm stage
        if (delta_warm.size() == N) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            record(pcm_demand::invert(s_data, mk, prm, delta_warm, opt.tol, opt.warm_evals));
            append(res.stages, fmt_stage("warm", converged, seconds_since(ts), evals - e0));
        }
        // 3. theta-path continuation from the theta the warm start was solved at
        if (!converged && prm_warm && delta_warm.size() == N) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            ArrayXd d = delta_warm;
            double tau = 0.0, step = 0.5;
            while (tau < 1.0 && step >= 0.01 && seconds_since(ts) < opt.path_seconds) {
                const double target = std::min(1.0, tau + step);
                pcm_demand::InversionResult r =
                    pcm_demand::invert(s_data, mk, interpolate(*prm_warm, prm, target), d, opt.tol, opt.path_evals);
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
            append(res.stages, fmt_stage("path", converged, seconds_since(ts), evals - e0) + buf);
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
                const pcm_demand::Market sub = thin_draws(mk, static_cast<int>(k));
                const pcm_demand::InversionResult r =
                    pcm_demand::invert(s_data, sub, prm, d, opt.tol, d.size() ? opt.warm_evals : opt.cold_evals);
                evals += r.share_evaluations;
                ok = r.converged;
                char buf[160];
                std::snprintf(buf, sizeof(buf), "draws R=%ld %s %.3gs %uev", static_cast<long>(sub.weights.size()),
                              ok ? "ok" : "failed", seconds_since(ts), r.share_evaluations);
                append(res.stages, buf);
                if (ok) d = r.delta;
            }
            if (ok) {
                const auto ts = Clock::now();
                const unsigned e0 = evals;
                record(pcm_demand::invert(s_data, mk, prm, d, opt.tol, opt.warm_evals));
                append(res.stages, fmt_stage("draws warm", converged, seconds_since(ts), evals - e0));
            }
        }
        // 4. cold stage
        if (!converged) {
            const auto ts = Clock::now();
            const unsigned e0 = evals;
            record(pcm_demand::invert(s_data, mk, prm, ArrayXd(), opt.tol, opt.cold_evals));
            append(res.stages, fmt_stage("cold", converged, seconds_since(ts), evals - e0));
        }
    } catch (const std::exception & e) {
        res.status = Status::INVALID_INPUT;
        append(res.stages, std::string("invalid input: ") + e.what());
        res.max_abs_residual = inf;
        res.share_evaluations = evals;
        res.seconds = seconds_since(t0);
        return res;
    }

    // 5. status
    if (best_delta.size() != N) best_delta = pcm_demand::detail::initial_guess(s_data, mk.p, prm);
    res.delta = best_delta;
    const bool feasible_seen = std::isfinite(best_res);

    // 6-7. one pass at the returned delta: shares, derivatives, micro parts
    const auto td = Clock::now();
    const bool want_d = converged && opt.derivatives;
    const bool want_p = converged && opt.price_derivatives;
    MarketEvaluation ev = evaluate_market(res.delta, mk, prm, micro, want_d, want_p);
    ++evals;
    res.s = ev.s;
    res.max_abs_residual = (ev.s - s_data).abs().maxCoeff();
    res.micro = ev.micro;
    res.ds_ddelta = ev.ds_ddelta;
    res.ds_dtheta = ev.ds_dtheta;
    res.ds_dp = ev.ds_dp;
    res.micro_d_delta = ev.micro_d_delta;
    res.micro_d_theta = ev.micro_d_theta;
    res.status = converged ? Status::OK : (feasible_seen ? Status::NOT_CONVERGED : Status::INFEASIBLE_START);
    res.rcond = std::numeric_limits<double>::quiet_NaN();

    if (want_d) {
        const MatrixXd & A = res.ds_ddelta;
        double rc = 0.0;
        Eigen::FullPivLU<MatrixXd> lu;
        if (A.allFinite()) {
            lu.compute(A);
            const ArrayXd piv = lu.matrixLU().diagonal().array().abs();
            const double pmax = piv.maxCoeff();
            rc = pmax > 0 ? piv.minCoeff() / pmax : 0.0;     // pivot ratio of U
            if (rc > 0) {
                const double est = lu.rcond();               // 1-norm estimate
                rc = std::isfinite(est) ? std::min(rc, est) : 0.0;
            }
        }
        res.rcond = rc;
        if (!(rc >= opt.rcond_min) || rc == 0.0) {
            res.status = Status::SINGULAR_DS_DDELTA;
            res.ddelta_dtheta = MatrixXd::Constant(N, res.ds_dtheta.cols(), std::numeric_limits<double>::quiet_NaN());
        } else {
            res.ddelta_dtheta = -lu.solve(res.ds_dtheta);
            if (!res.ddelta_dtheta.allFinite()) {
                res.status = Status::SINGULAR_DS_DDELTA;
                res.ddelta_dtheta.setConstant(std::numeric_limits<double>::quiet_NaN());
            }
        }
    }
    {
        char buf[96];
        if (want_d) std::snprintf(buf, sizeof(buf), "derivatives %.3gs rcond=%.3g", seconds_since(td), res.rcond);
        else        std::snprintf(buf, sizeof(buf), "evaluate %.3gs", seconds_since(td));
        append(res.stages, buf);
    }
    res.share_evaluations = evals;
    res.seconds = seconds_since(t0);
    return res;
}

FrontierAnatomy frontier_anatomy(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    validate_market(mk, prm, N);
    require(delta.allFinite(), "delta must be finite");
    FrontierAnatomy out;
    out.frontier_size = Eigen::ArrayXi::Zero(R);
    out.mass = ArrayXd::Zero(R);
    out.log_range = ArrayXd::Zero(R);
    out.mass_log_width = ArrayXd::Zero(R);
#pragma omp parallel for schedule(dynamic, 64) if(R > 64)
    for (int i = 0; i < R; ++i) {
        VectorXd t = VectorXd::Zero(K);
        for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
        const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
        const double c = std::exp(log_c);
        const ArrayXd dc = (delta + (mk.x * t).array()) / c;
        const pcm_demand::detail::DrawAnatomy a =
            pcm_demand::detail::draw_anatomy(dc, mk.p, prm.sigma_p, pcm_demand::draw_a_lo(prm, c));
        out.frontier_size[i] = a.frontier_size;
        out.mass[i] = a.mass;
        out.log_range[i] = a.log_range;
        out.mass_log_width[i] = a.mass_log_width;
    }
    return out;
}

}  // namespace pcm_kernel
