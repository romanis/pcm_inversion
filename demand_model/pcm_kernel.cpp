#include "pcm_kernel.hpp"
#include "pcm_aggregate.hpp"
#include "pcm_invert.hpp"
#include "pcm_geometry.hpp"

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

void validate_market(const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm, int N) {
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

void append(std::string & stages, const std::string & s) {
    if (!stages.empty()) stages += "; ";
    stages += s;
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

MarketEvaluation evaluate_market(const ArrayXd & delta, const pcm_aggregate::Market & mk,
                                 const pcm_aggregate::Params & prm, const MicroRequest * micro,
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

    MarketEvaluation out;
    out.s = ArrayXd::Zero(N);
    if (derivatives) { out.ds_ddelta = MatrixXd::Zero(N, N); out.ds_dtheta = MatrixXd::Zero(N, P); }
    if (price_derivatives) out.ds_dp = MatrixXd::Zero(N, N);
    if (micro) {
        out.micro = VectorXd::Zero(Pm);
        if (derivatives) { out.micro_d_delta = MatrixXd::Zero(Pm, N); out.micro_d_theta = MatrixXd::Zero(Pm, P); }
    }

    // The draw loop of pcm_aggregate::shares: fixed blocks of 64 draws, summed in block order.
    pcm_aggregate::for_each_draw_block<Block>(R, [&](Block & acc, int i0, int i1) {
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
        const bool bound = pcm_aggregate::bound_moves_with_scale(prm);   // log c acts through a_lo = lo / c
        MatrixXd Jd, Jp;
        if (jac) { Jd = MatrixXd::Zero(N, N); Jp = MatrixXd::Zero(N, N); }
        VectorXd t(K);
        std::vector<int> S;
        S.reserve(N);
        MatrixXd Jc, uJ;
        VectorXd us, ud, ul, xs;
        ArrayXd dc(N);

        for (int i = i0; i < i1; ++i) {
            const double c = pcm_aggregate::conditional_quality(delta, mk, prm, i, t, dc);

            s_c.setZero();
            if (jac) { dsig.setZero(); if (bound) dlogc.setZero(); }
            pcm_geometry::cond_share(dc, mk.p, prm.sigma_p, pcm_aggregate::draw_a_lo(prm, c), s_c, jac ? &Jd : nullptr,
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
    }, [&](const Block & acc) {
        out.s += acc.s;
        if (derivatives) { out.ds_ddelta += acc.ds_ddelta; out.ds_dtheta += acc.ds_dtheta; }
        if (price_derivatives) out.ds_dp += acc.ds_dp;
        if (micro) {
            out.micro += acc.micro;
            if (derivatives) { out.micro_d_delta += acc.micro_d_delta; out.micro_d_theta += acc.micro_d_theta; }
        }
    });
    return out;
}

SolveResult solve_market(const ArrayXd & s_data, const pcm_aggregate::Market & mk,
                         const pcm_aggregate::Params & prm, const ArrayXd & delta_warm,
                         const pcm_aggregate::Params * prm_warm, const MicroRequest * micro,
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
    pcm_aggregate::ThreadBudgetScope budget(opt.thread_budget);
    const double inf = std::numeric_limits<double>::infinity();
    pcm_invert::Solution sol;
    const pcm_invert::Options iopt{opt.tol, opt.warm_evals, opt.cold_evals, opt.path_evals, opt.path_seconds,
                                   opt.cold_draws, opt.cold_draw_factor};
    try {
        pcm_invert::solve(s_data, mk, prm, delta_warm, prm_warm, iopt, sol);
    } catch (const std::exception & e) {
        res.status = Status::INVALID_INPUT;
        res.stages = sol.stages;
        append(res.stages, std::string("invalid input: ") + e.what());
        res.max_abs_residual = inf;
        res.share_evaluations = sol.evals;
        res.seconds = seconds_since(t0);
        return res;
    }

    res.stages = sol.stages;
    ArrayXd & best_delta = sol.delta;
    const double best_res = sol.residual;
    const bool converged = sol.converged;
    unsigned evals = sol.evals;

    // 5. status
    if (best_delta.size() != N) best_delta = pcm_geometry::initial_guess(s_data, mk.p, prm.sigma_p, prm.lo);
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

FrontierAnatomy frontier_anatomy(const ArrayXd & delta, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm)
{
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    validate_market(mk, prm, N);
    require(delta.allFinite(), "delta must be finite");
    FrontierAnatomy out;
    out.frontier_size = Eigen::ArrayXi::Zero(R);
    out.mass = ArrayXd::Zero(R);
    out.log_range = ArrayXd::Zero(R);
    out.mass_log_width = ArrayXd::Zero(R);
    // per-draw outputs: no reduction; the calling thread's team, without a thread budget
    pcm_aggregate::for_each_draw_block<pcm_aggregate::NoAcc>(R, [&](pcm_aggregate::NoAcc &, int i0, int i1) {
        VectorXd t(K);
        ArrayXd dc(N);
        for (int i = i0; i < i1; ++i) {
            const double c = pcm_aggregate::conditional_quality(delta, mk, prm, i, t, dc);
            const pcm_geometry::DrawAnatomy a =
                pcm_geometry::draw_anatomy(dc, mk.p, prm.sigma_p, pcm_aggregate::draw_a_lo(prm, c));
            out.frontier_size[i] = a.frontier_size;
            out.mass[i] = a.mass;
            out.log_range[i] = a.log_range;
            out.mass_log_width[i] = a.mass_log_width;
        }
    }, [](const pcm_aggregate::NoAcc &) {}, pcm_aggregate::DrawLoop{0, omp_get_max_threads()});
    return out;
}

}  // namespace pcm_kernel
