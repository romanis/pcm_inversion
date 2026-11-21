#include "pcm_groups.hpp"
#include "pcm_aggregate.hpp"
#include "pcm_geometry.hpp"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace pcm_kernel {

namespace {

void require(bool ok, const std::string & what) {
    if (!ok) throw std::invalid_argument("group_sums: " + what);
}

void validate(const ArrayXd & delta, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm,
              const GroupRequest & gr) {
    const int N = static_cast<int>(delta.size());
    const int R = static_cast<int>(mk.weights.size());
    const int K = static_cast<int>(mk.x.cols());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    require(mk.x.rows() == N && mk.p.size() == N, "market x and p must have one row per product");
    require(mk.taste_basis.rows() == R && mk.taste_basis.cols() == M && static_cast<int>(mk.taste_col.size()) == M,
            "taste_basis must be R x M and taste_col of size M (M = len(theta))");
    require(Q == 0 || (mk.scale_basis.rows() == R && mk.scale_basis.cols() == Q), "scale_basis must be R x Q");
    for (int c : mk.taste_col) require(c >= 0 && c < K, "taste_col out of range");
    for (int j = 0; j < N; ++j) {
        require(std::isfinite(mk.p[j]), "prices must be finite");
        if (j > 0) require(mk.p[j] > mk.p[j - 1], "prices must be strictly increasing");
    }
    require(N == 0 || mk.p[0] > 0, "prices must be positive");
    require(R > 0, "market needs at least one draw");
    require(mk.weights.allFinite() && (mk.weights >= 0).all(), "draw weights must be finite and non-negative");
    require(mk.x.allFinite() && mk.taste_basis.allFinite() && (Q == 0 || mk.scale_basis.allFinite()),
            "x, taste_basis and scale_basis must be finite");
    require(prm.theta.allFinite() && prm.phi.allFinite(), "theta and phi must be finite");
    require(std::isfinite(prm.sigma_p) && prm.sigma_p > 0, "sigma_p must be positive and finite");
    require(std::isfinite(prm.lo) && prm.lo >= 0, "the bound (alpha_lo or a_lo) must be finite and non-negative");
    require(delta.allFinite(), "delta must be finite");
    require(gr.n_groups >= 1, "n_groups must be at least 1");
    require(gr.group_of_draw.size() == R, "group_of_draw must have one entry per draw");
    for (int i = 0; i < R; ++i) {
        require(gr.group_of_draw[i] >= 0 && gr.group_of_draw[i] < gr.n_groups, "group_of_draw out of range");
        if (i > 0) require(gr.group_of_draw[i] >= gr.group_of_draw[i - 1], "group_of_draw must be non-decreasing");
    }
    require(gr.U.rows() == N, "U must have one row per product");
    require(gr.U.allFinite(), "U must be finite");
}

// Partial sums of one block of a group's draws.
struct GroupBlock {
    VectorXd acc;
    MatrixXd acc_d, acc_t;
};

}  // namespace

GroupSums group_sums(const ArrayXd & delta, const pcm_aggregate::Market & mk, const pcm_aggregate::Params & prm,
                     const GroupRequest & gr, bool derivatives)
{
    validate(delta, mk, prm, gr);
    const int N = static_cast<int>(delta.size());
    const int K = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    const int P = M + Q + 1;
    const int G = gr.n_groups;
    const int C1 = static_cast<int>(gr.U.cols()) + 1;
    MatrixXd Up(N, C1);
    Up.col(0).setOnes();
    if (C1 > 1) Up.rightCols(C1 - 1) = gr.U;

    // first draw of each group (empty groups have first == end)
    std::vector<int> first(G + 1, R);
    for (int i = R - 1; i >= 0; --i) first[gr.group_of_draw[i]] = i;
    for (int g = G - 1; g >= 0; --g) if (first[g] > first[g + 1]) first[g] = first[g + 1];

    GroupSums out;
    out.S = MatrixXd::Zero(G, C1);
    out.w = ArrayXd::Zero(G);
    if (derivatives) {
        out.d_delta.assign(G, MatrixXd::Zero(C1, N));
        out.d_theta.assign(G, MatrixXd::Zero(C1, P));
    }
    const bool bound = pcm_aggregate::bound_moves_with_scale(prm);   // log c acts through a_lo = lo / c

#pragma omp parallel if(G > 1)
    {
        ArrayXd s_c = ArrayXd::Zero(N), dsig = ArrayXd::Zero(N), dlogc = ArrayXd::Zero(N);
        MatrixXd Jd;
        if (derivatives) Jd = MatrixXd::Zero(N, N);
        VectorXd t(K), us, ud, ul, xs;
        ArrayXd dc(N);
        MatrixXd Jc, uJ;
        std::vector<int> S;
        S.reserve(N);
#pragma omp for schedule(dynamic, 8)
        for (int g = 0; g < G; ++g) {
            const int g0 = first[g], g1 = first[g + 1];
            const int Rg = g1 - g0;
            if (Rg <= 0) continue;
            VectorXd tot = VectorXd::Zero(C1);
            double wsum = 0.0;
            // the block partition of evaluate_market on a market of this group's draws, run serially by the
            // thread that owns the group, one block held at a time
            pcm_aggregate::for_each_draw_block<GroupBlock>(Rg, [&](GroupBlock & blk, int i0, int i1) {
                VectorXd & acc = blk.acc;
                MatrixXd & acc_d = blk.acc_d;
                MatrixXd & acc_t = blk.acc_t;
                acc = VectorXd::Zero(C1);
                if (derivatives) { acc_d = MatrixXd::Zero(C1, N); acc_t = MatrixXd::Zero(C1, P); }
                for (int i = i0; i < i1; ++i) {
                    const double c = pcm_aggregate::conditional_quality(delta, mk, prm, i, t, dc);
                    s_c.setZero();
                    if (derivatives) { dsig.setZero(); if (bound) dlogc.setZero(); }
                    pcm_geometry::cond_share(dc, mk.p, prm.sigma_p, pcm_aggregate::draw_a_lo(prm, c), s_c,
                                          derivatives ? &Jd : nullptr, &dsig, nullptr, bound ? &dlogc : nullptr);
                    const double w = mk.weights[i];
                    us.noalias() = Up.transpose() * s_c.matrix();
                    const double wh = w * 1.0;
                    if (!derivatives) {
                        for (int p = 0; p < C1; ++p) acc[p] += wh * us[p];
                        continue;
                    }
                    S.clear();
                    for (int j = 0; j < N; ++j)
                        if (Jd(j, j) != 0.0 || s_c[j] != 0.0 || dsig[j] != 0.0 || (bound && dlogc[j] != 0.0))
                            S.push_back(j);
                    const int ns = static_cast<int>(S.size());
                    Jc = Jd(S, S);
                    uJ.noalias() = Up(S, Eigen::all).transpose() * Jc;   // C1 x ns
                    ud.noalias() = Up.transpose() * dsig.matrix();
                    if (bound && Q > 0) ul.noalias() = Up.transpose() * dlogc.matrix();
                    for (int p = 0; p < C1; ++p) {
                        acc[p] += wh * us[p];
                        if (wh == 0.0) continue;
                        for (int k = 0; k < ns; ++k) acc_d(p, S[k]) += (wh / c) * uJ(p, k);
                        for (int m = 0; m < M; ++m) {
                            double v = 0.0;
                            const auto xm = mk.x.col(mk.taste_col[m]);
                            for (int k = 0; k < ns; ++k) v += uJ(p, k) * xm[S[k]];
                            acc_t(p, m) += (wh * mk.taste_basis(i, m) / c) * v;
                        }
                        if (Q > 0) {
                            double v = 0.0;
                            for (int k = 0; k < ns; ++k) v += uJ(p, k) * dc[S[k]];
                            if (bound) v -= ul[p];
                            for (int q = 0; q < Q; ++q) acc_t(p, M + q) += -wh * mk.scale_basis(i, q) * v;
                        }
                        acc_t(p, M + Q) += wh * ud[p];
                    }
                    Jd(S, S).setZero();   // restore the zero state cond_share requires
                }
            }, [&](const GroupBlock & blk) {
                tot += blk.acc;
                if (derivatives) { out.d_delta[g] += blk.acc_d; out.d_theta[g] += blk.acc_t; }
            }, pcm_aggregate::DrawLoop{g0, -1, 1});
            for (int i = g0; i < g1; ++i) wsum += mk.weights[i];
            out.S.row(g) = tot.transpose();
            out.w[g] = wsum;
        }
    }
    return out;
}

}  // namespace pcm_kernel
