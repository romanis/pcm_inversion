#include "pcm_second_choice.hpp"
#include "pcm_market_share.hpp"

#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using pcm_share::detail::LnTerms;
using pcm_share::detail::Trunc;
using pcm_share::detail::ln_terms;
using pcm_share::detail::make_trunc;

namespace pcm_kernel {

namespace {

constexpr int ORIGIN = -1;   // the outside good at (p, d) = (0, 0)
constexpr int FIXED = -2;    // a cutoff that does not move (0 or +-inf)

void require(bool ok, const std::string & what) {
    if (!ok) throw std::invalid_argument("second_choice: " + what);
}

void validate(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
              const SecondChoiceAdjoint * adj) {
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
    if (adj) {
        const int Ka = static_cast<int>(adj->A.size());
        require(Ka >= 1, "the adjoint needs at least one A matrix");
        for (const MatrixXd & A : adj->A) require(A.rows() == N && A.cols() == N && A.allFinite(),
                                                  "every adjoint A must be a finite N x N matrix");
        require(adj->B.rows() == Ka && adj->B.cols() == N && adj->B.allFinite(), "adjoint B must be finite K x N");
    }
}

// a cutoff e = (d_a - d_b) / (p_a - p_b) (b = ORIGIN: d_a / p_a), or a fixed value (a = FIXED)
struct Cut {
    double e;
    int a, b;
};

struct Block {
    ArrayXd s;
    MatrixXd N, od, ot;
    unsigned none = 0;
};

}  // namespace

SecondChoice second_choice(const ArrayXd & delta, const pcm_demand::Market & mk, const pcm_demand::Params & prm,
                           const SecondChoiceAdjoint * adj)
{
    const auto t0 = std::chrono::steady_clock::now();
    validate(delta, mk, prm, adj);
    const int N = static_cast<int>(delta.size());
    const int Kx = static_cast<int>(mk.x.cols());
    const int R = static_cast<int>(mk.weights.size());
    const int M = static_cast<int>(prm.theta.size());
    const int Q = static_cast<int>(prm.phi.size());
    const int P = M + Q + 1;
    const int K = adj ? static_cast<int>(adj->A.size()) : 0;
    const bool scale_bound = pcm_demand::bound_moves_with_scale(prm);   // log c acts through a_lo = lo / c
    const double inf = std::numeric_limits<double>::infinity();

    // A'_c[j, k] for every c contiguous: Af[(j N + k) K + c]
    std::vector<double> Af;
    if (adj) {
        Af.resize(static_cast<size_t>(N) * N * K);
        for (int c = 0; c < K; ++c)
            for (int k = 0; k < N; ++k)
                for (int j = 0; j < N; ++j) Af[(static_cast<size_t>(j) * N + k) * K + c] = adj->A[c](j, k);
    }

    const int block = 64;
    const int n_blocks = std::max(1, (R + block - 1) / block);
    const int wave = std::min(n_blocks, 64);   // blocks held at once: memory wave x N^2
    SecondChoice out;
    out.s = ArrayXd::Zero(N);
    out.N = MatrixXd::Zero(N, N);
    if (adj) { out.out_delta = MatrixXd::Zero(K, N); out.out_theta = MatrixXd::Zero(K, P); }
    std::vector<Block> part(wave);

    for (int w0 = 0; w0 < n_blocks; w0 += wave) {
        const int w1 = std::min(n_blocks, w0 + wave);
#pragma omp parallel if(w1 - w0 > 1)
        {
            VectorXd t(Kx);
            ArrayXd s_c = ArrayXd::Zero(N), dsig = ArrayXd::Zero(N), dlogc = ArrayXd::Zero(N);
            MatrixXd Jd;
            if (adj) Jd = MatrixXd::Zero(N, N);
            MatrixXd adc;                                    // K x N adjoint on the conditional qualities
            if (adj) adc = MatrixXd::Zero(K, N);
            std::vector<char> touched(N, 0);
            std::vector<int> tlist;
            VectorXd a_sig(K), a_lc(K), U(K), cs(K), cm(K), tmp(K);
            std::vector<double> e;
            std::vector<int> hull, V;
            std::vector<double> ec, mass;
            std::vector<Cut> c_hi, c_lo;
#pragma omp for schedule(dynamic)
            for (int b = w0; b < w1; ++b) {
                Block & acc = part[b - w0];
                acc.s = ArrayXd::Zero(N);
                acc.N = MatrixXd::Zero(N, N);
                acc.none = 0;
                if (adj) { acc.od = MatrixXd::Zero(K, N); acc.ot = MatrixXd::Zero(K, P); }
                const int i0 = static_cast<int>((static_cast<long>(R) * b) / n_blocks);
                const int i1 = static_cast<int>((static_cast<long>(R) * (b + 1)) / n_blocks);
                for (int i = i0; i < i1; ++i) {
                    t.setZero();
                    for (int m = 0; m < M; ++m) t[mk.taste_col[m]] += prm.theta[m] * mk.taste_basis(i, m);
                    const double log_c = Q > 0 ? (mk.scale_basis.row(i).array() * prm.phi.transpose()).sum() : 0.0;
                    const double c = std::exp(log_c);
                    const ArrayXd dc = (delta + (mk.x * t).array()) / c;
                    const double a_lo = pcm_demand::draw_a_lo(prm, c);
                    const std::vector<int> ind = index_positive_shares(dc, mk.p);
                    const int m = static_cast<int>(ind.size());
                    if (m == 0) continue;
                    s_c.setZero();
                    if (adj) { dsig.setZero(); if (scale_bound) dlogc.setZero(); }
                    pcm_share::detail::vertical_share(ind, dc, mk.p, prm.sigma_p, a_lo, s_c, adj ? &Jd : nullptr,
                                                      &dsig, nullptr, scale_bound ? &dlogc : nullptr, false);
                    const double w = mk.weights[i];
                    acc.s += w * s_c;
                    // the frontier cutoffs e_0..e_m, as vertical_share forms them
                    e.assign(m + 1, 0.0);
                    e[0] = dc[ind[0]] / mk.p[ind[0]];
                    for (int k = 1; k < m; ++k)
                        e[k] = (dc[ind[k]] - dc[ind[k - 1]]) / (mk.p[ind[k]] - mk.p[ind[k - 1]]);
                    const Trunc tr = make_trunc(a_lo, prm.sigma_p);
                    auto px = [&](int k) { return k < 0 ? 0.0 : mk.p[k]; };
                    auto qy = [&](int k) { return k < 0 ? 0.0 : dc[k]; };
                    if (adj) { a_sig.setZero(); a_lc.setZero(); }
                    // the adjoint of F(cut) with coefficient kappa
                    auto add_cut = [&](const Cut & cut, const VectorXd & kappa) {
                        if (cut.a == FIXED) return;
                        const LnTerms lt = ln_terms(cut.e, prm.sigma_p, tr);
                        if (lt.f != 0.0) {
                            const double gap = px(cut.a) - px(cut.b);
                            const double fg = lt.f / gap;
                            adc.col(cut.a) += fg * kappa;
                            if (!touched[cut.a]) { touched[cut.a] = 1; tlist.push_back(cut.a); }
                            if (cut.b >= 0) {
                                adc.col(cut.b) -= fg * kappa;
                                if (!touched[cut.b]) { touched[cut.b] = 1; tlist.push_back(cut.b); }
                            }
                        }
                        a_sig += lt.g * kappa;
                        if (scale_bound) a_lc += lt.h * kappa;
                    };
                    for (int r = 0; r < m; ++r) {
                        const int j = ind[r];
                        const double s = s_c[j];
                        if (!(s > 0.0)) continue;
                        const int L = r > 0 ? ind[r - 1] : ORIGIN;
                        const bool hasR = r + 1 < m;
                        const int Rn = hasR ? ind[r + 1] : ORIGIN;
                        const Cut hi{e[r], j, L};
                        const Cut lo = hasR ? Cut{e[r + 1], Rn, j} : Cut{0.0, FIXED, FIXED};
                        // reduced chain: upper hull of L, the products priced between L and R (j removed), R
                        hull.clear();
                        hull.push_back(L);
                        const int k_end = hasR ? Rn : N;
                        for (int k = L + 1; k <= k_end && k < N; ++k) {
                            if (k == j) continue;
                            while (hull.size() >= 2) {
                                const int a = hull[hull.size() - 2], bq = hull.back();
                                const double cross = (px(bq) - px(a)) * (qy(k) - qy(a)) - (qy(bq) - qy(a)) * (px(k) - px(a));
                                if (cross >= 0) hull.pop_back(); else break;
                            }
                            hull.push_back(k);
                        }
                        if (!hasR) {   // without R: the increasing part only
                            double top = qy(L);
                            size_t end = 1;
                            for (size_t h = 1; h < hull.size(); ++h)
                                if (qy(hull[h]) > top) { top = qy(hull[h]); end = h + 1; }
                            hull.resize(end);
                        }
                        V = hull;
                        const int nv = static_cast<int>(V.size());
                        ec.assign(std::max(nv - 1, 0), 0.0);
                        for (int u = 0; u + 1 < nv; ++u)
                            ec[u] = (qy(V[u + 1]) - qy(V[u])) / (px(V[u + 1]) - px(V[u]));
                        mass.assign(nv, 0.0);
                        c_hi.assign(nv, Cut{0.0, FIXED, FIXED});
                        c_lo.assign(nv, Cut{0.0, FIXED, FIXED});
                        double Mij = 0.0;
                        for (int u = 0; u < nv; ++u) {
                            if (V[u] == ORIGIN) continue;   // buys nothing: not an inside second choice
                            const Cut top = u == 0 ? Cut{inf, FIXED, FIXED} : Cut{ec[u - 1], V[u], V[u - 1]};
                            const Cut bot = u + 1 < nv ? Cut{ec[u], V[u + 1], V[u]} : Cut{-inf, FIXED, FIXED};
                            const Cut & ch = top.e < hi.e ? top : hi;
                            const Cut & cl = bot.e > lo.e ? bot : lo;
                            const double mu = ln_terms(ch.e, prm.sigma_p, tr).F - ln_terms(cl.e, prm.sigma_p, tr).F;
                            if (!(mu > 0.0)) continue;
                            mass[u] = mu;
                            c_hi[u] = ch;
                            c_lo[u] = cl;
                            Mij += mu;
                        }
                        const bool inside = Mij > 0.0;
                        if (!inside) ++acc.none;
                        for (int u = 0; inside && u < nv; ++u)
                            if (mass[u] > 0.0) acc.N(j, V[u]) += w * s * mass[u] / Mij;
                        if (!adj) continue;
                        // T = (s / M) sum_u A'(j, V_u) m_u - B(j) s. A draw whose buyers of j have no inside second
                        // choice adds nothing to N but still to s_j, so its -B(j) ds_j term stays.
                        U.setZero();
                        for (int u = 0; inside && u < nv; ++u)
                            if (mass[u] > 0.0)
                                U += mass[u] * Eigen::Map<const VectorXd>(&Af[(static_cast<size_t>(j) * N + V[u]) * K], K);
                        for (int u = 0; inside && u < nv; ++u) {
                            if (!(mass[u] > 0.0)) continue;
                            cm = (s / Mij) * (Eigen::Map<const VectorXd>(&Af[(static_cast<size_t>(j) * N + V[u]) * K], K)
                                              - U / Mij);
                            add_cut(c_hi[u], cm);
                            tmp = -cm;
                            add_cut(c_lo[u], tmp);
                        }
                        cs = -adj->B.col(j);
                        if (inside) cs += U / Mij;
                        // ds_j = sum_k Jd(j, k) d dc_k + dsig_j d sigma + dlogc_j d log c
                        const int nb[3] = {L, j, hasR ? Rn : ORIGIN};
                        for (int k : nb) {
                            if (k < 0 || Jd(j, k) == 0.0) continue;
                            adc.col(k) += Jd(j, k) * cs;
                            if (!touched[k]) { touched[k] = 1; tlist.push_back(k); }
                        }
                        a_sig += dsig[j] * cs;
                        if (scale_bound) a_lc += dlogc[j] * cs;
                    }
                    if (adj) {
                        // dc_a = (delta_a + x_a t) / c: delta, taste and scale columns
                        VectorXd sum_dc = VectorXd::Zero(K);
                        for (int a : tlist) {
                            acc.od.col(a) += (w / c) * adc.col(a);
                            sum_dc += dc[a] * adc.col(a);
                        }
                        for (int mm = 0; mm < M; ++mm) {
                            VectorXd v = VectorXd::Zero(K);
                            const auto xm = mk.x.col(mk.taste_col[mm]);
                            for (int a : tlist) v += xm[a] * adc.col(a);
                            acc.ot.col(mm) += (w * mk.taste_basis(i, mm) / c) * v;
                        }
                        for (int q = 0; q < Q; ++q) acc.ot.col(M + q) += (w * mk.scale_basis(i, q)) * (a_lc - sum_dc);
                        acc.ot.col(M + Q) += w * a_sig;
                        for (int a : tlist) { adc.col(a).setZero(); touched[a] = 0; }
                        tlist.clear();
                        Jd(ind, ind).setZero();   // the zero state vertical_share requires
                    }
                }
            }
        }
        for (int b = w0; b < w1; ++b) {   // block order: results do not depend on the thread count
            const Block & acc = part[b - w0];
            out.s += acc.s;
            out.N += acc.N;
            out.draws_with_no_inside_second += acc.none;
            if (adj) { out.out_delta += acc.od; out.out_theta += acc.ot; }
        }
    }
    out.C = MatrixXd::Zero(N, N);
    for (int j = 0; j < N; ++j)
        if (out.s[j] > 0.0) out.C.row(j) = out.N.row(j) / out.s[j];
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace pcm_kernel
