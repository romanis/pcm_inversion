# Learnings from estimating the pure characteristics model

Working notes for the write-up. Each entry records what we found, the evidence, and the fix.
Dates are when the finding was established.

## 1. The PCM utility scale is free, and a fixed-weight GMM objective exploits it (2026-11-15)

**Finding.** In the PCM, utility is u_ij = delta_j + x_j . t_i - alpha_i p_j with no idiosyncratic
error. Choices are invariant to multiplying (delta, t, alpha) by a common constant, so one
normalisation is needed. Normalising the median alpha to 1 pins delta to "dollars as valued by the
median consumer". When most consumers buy the outside good (12% buy a car in Petrin's data), the
median consumer never buys, and the scale of every inside delta is set instead by the marginal
buyer: in the vertical limit delta_0 = e_0 p_0 with F(e_0) = inside share, so
e_0 = exp(sigma_p Phi^-1(0.12)) = exp(-1.17 sigma_p). Raising sigma_p shrinks the whole delta vector,
and with it xi = delta - X beta, exponentially.

A GMM objective with a theta-independent weight matrix, q = n g' W g with g = Z' xi / n, is
quadratic in the scale of xi. It is therefore minimised by making xi small, which the estimator does
by pushing sigma_p up (buyers become an extreme low-alpha tail) and every taste spread sigma_x down
(a wide taste spread lets ordinary-alpha consumers buy, which forces delta up toward p). The result
looks like a near-vertical model with own-price elasticities of -50 to -500. Two-step GMM does not
help: W = S^-1 is computed at the step-0 point where xi is already tiny.

In BLP this cannot happen. The logit error has fixed variance (pi^2/6), which pins the scale of
utility for every consumer, buyer or not. In the PCM nothing pins it, and the median-alpha
normalisation pins it to the wrong consumer in a way that sigma_p can undo.

**Evidence.** `petrin_replication/profile_sigma_ray.py`, results in
`petrin_replication/results/profile_sigma_ray_nu4.csv`. Along a ray sigma = lambda * ($2k taste
spreads), at lambda = 0.02: sd(xi) falls from $480 to $44 as sigma_p goes 1 -> 3.5 (factor 11) and
q_fixed falls from 18.2 to 0.096 (factor 190, about 11^2). Along lambda at fixed sigma_p, sd(xi)
rises 5x and q_fixed rises 10x to 1700x. The continuously-updated objective
q_CUE = n g' S(theta)^-1 g, which is invariant to xi -> kappa xi, falls along the same ray
(53.9 -> 32.2 at sigma_p = 1, 46.6 -> 28.6 at sigma_p = 2) and barely moves in sigma_p. Same
pattern seen earlier on Nevo's cereal data (sigma_const -> 0, sigma_p -> bound), misread there as
an economic finding about horizontal differentiation.

**Fix.** Estimate by continuously-updated GMM (`PCM_CUE=1` in `pcm_petrin.py`), which depends only
on the correlation structure between xi and the instruments. Analytic gradient:
dq/dtheta_k = n [2 dg_k' a - (2/n) (dV_k a)' (V a)] with a = S^-1 g, V the cluster sums of z_j xi_j,
dV_k the cluster sums of z_j dxi_jk. Checked against finite differences in
`petrin_replication/check_cue_gradient.py`. beta stays concentrated out with the 2SLS weighting.

**For the write-up.** Any PCM estimated by GMM on delta-based moments needs either a scale-free
objective (CUE) or a normalisation that does not move with theta. This is a general point about
the estimator, not about cars or cereal, and we have not seen it stated in the PCM literature we
checked (Berry and Pakes 2007 report Monte Carlos only).

## 2. Own-price elasticities in the PCM scale as p / (price gap to the nearest substitute) (2026-11-14)

In the vertical limit, product k is bought by consumers with alpha in (e_{k+1}, e_k), where
e_k = (delta_k - delta_{k-1}) / (p_k - p_{k-1}). A price rise dp_k moves both endpoints by
e / Delta p, so d log s_k / d log p_k ~ -(p_k / Delta p_k) * (f(e) e / s_k). With 150-210 cars in
$5k-$50k the median p / Delta p is in the hundreds, and a small-share product has f(e) e / s_k
large. Moderate elasticities require horizontal taste dispersion comparable to the price level
(thousands of dollars), not to the price gap. This is why the vertical corner of the parameter
space gives -500 and BLP-scale heterogeneity ($2k-$6k) gives -6 to -8.

## 3. Products with the same observables and a tiny price gap make the share Jacobian singular (2026-11-14)

Petrin's prices are list prices deflated by the CPI: 95 exact ties and 1% of neighbouring gaps
under $1. The conditional share derivative between two products is f(e) / (price gap). Spreading
prices to a $10 minimum gap (on $5k-$50k) made all 13 markets invertible with no economic change.

## 4. A body type with a large seller needs its own taste dimension (2026-11-14)

Petrin's BLP has random coefficients on the constant, hpwt, space, air, mpd, fwd, and demographic
interactions on the body dummies, but no random coefficient on the body dummies themselves: the
logit error gives a minivan its own following. In the PCM, when sigma_const fell, the 1993 Dodge
Caravan (0.5% share, the largest seller) could reach at most 0.27%: raising its delta sweeps up a
dense cluster of $9.5k-$10.5k sedans first. Random coefficients on mi, sw, su, pv (`PCM_BODY_SIGMA=1`)
fix it.

## 5. Zero-share products are repaired by lifting onto the nearest consumer frontier (2026-11-14)

A product with zero predicted share is off every consumer's upper convex hull in (price, quality)
space. The minimum delta increase that puts it on at least one hull is computable in one pass
(`frontier_lift` in `demand_model/pcm_demand.cpp`). Blind bumps did not converge on 200-product
markets; the exact lift does.

## 6. Sparse grids are the wrong quadrature for PCM shares (2026-11-14)

Tasmanian Gauss-Hermite sparse grids at 10 dimensions carry negative weight mass 90x to 2800x
the integral. PCM shares are step functions of the taste draw, so the sign cancellation fails:
share errors of 6 to 33 and dozens of negative shares, worsening with more points. Scrambled Sobol
with equal weights: max error 0.0065 at 1024 points, 0.0007 at 16384. Use QMC.

## 7. Nevo's fake cereal data were generated from a logit, so they favour BLP by construction (2026-11-14)

The 24 brands differ only in sugar and mushy; everything else that separates them in the
simulation is 24-dimensional epsilon noise. Forcing that into vertical xi is what the PCM had to do
there. It is not a test of PCM against BLP on real demand.

## 8. PCM inversion time varies 10x across markets at the same number of products (2026-11-15)

**Finding.** The serial inversion time (1 OMP thread) at nu=4 ranges from 186s (market 1982, 148
products) to 1745s (market 1988, 205 products) — a 9x spread for a 38% difference in J. Product
count is a poor proxy for inversion difficulty. The slowest market is not the largest (1992 has
210 products but takes only 906s; 1989 has 199 products but takes 699s). The likely driver is
the structure of the price-quality convex hull: a market with many near-ties on the hull needs
more Newton steps because the Jacobian is ill-conditioned near its boundary points.

**Evidence.** `petrin_replication/results/market_times_nu4.csv` (serial, 1 thread per market).

**Implication for practitioners.** When parallelising a PCM estimator with one Python thread per
market, wall-clock time per evaluation equals the slowest market's inversion time regardless of
how many markets run in parallel. This is easy to miss when benchmarking on small problems.

## 9. OpenMP parallelism within the PCM inversion reduces bottleneck market time by 5x (2026-11-15)

**Finding.** The C++ share-computation kernel uses OpenMP over consumers
(`#pragma omp parallel for schedule(dynamic, 10)` in `market_share/pcm_market_share.cpp`).
Allocating threads proportionally to empirical market timing and setting `omp_set_num_threads()`
via ctypes before each market call reduced the nu=4 evaluation from 426s to 78s (5.4x speedup).
This is a resource-redistribution gain: total CPU-seconds is unchanged, but the slowest market
goes from 426s (1 thread) to ~78s (~52 threads).

**How.** `PCMProblem.__init__` loads `libgomp.so.1` via ctypes, reads `market_times_nu4.csv` for
the allocation weights, and stores per-market thread budgets. `_solve_market` calls
`gomp.omp_set_num_threads(budget)` at entry. Controlled by `PCM_OMP_TOTAL` (default: `os.cpu_count()`).

**Caveat.** The nu=4 serial timing is not a reliable proxy for nu=16 difficulty: the per-market
rank order changes substantially. The proper approach is to run a brief serial profile at the
target nu (or at a small consumers-per-market value) before the full estimation, then use those
weights. Alternatively, weight by J^2 (Newton work scales as J^2 for the linear solve in the
Jacobian step), which is a better prior than J^1 or historical timing from a different nu.

## 10. MPEC + CUE is valid but the main benefit over nested CUE is robustness, not convergence rate (2026-11-16)

**Finding.** MPEC (Dube, Fox, Su 2012) lifts delta to an explicit decision variable alongside theta
and enforces sigma(delta, theta) = s as equality constraints for an NLP solver (IPOPT/KNITRO).

For fixed-W GMM: q = delta' M'ZWZ'M delta / n is quadratic in delta, so d^2q/ddelta^2 = Z'M'WMZ/n
is a constant PD matrix. IPOPT exploits this directly — each NLP iteration takes a full Newton step
in the (theta, delta) space without re-inverting — which is both faster and more robust than the
nested approach.

For CUE: q_CUE = g' S(delta, theta)^-1 g is a ratio of quadratics in delta. d^2q/ddelta^2 is not
constant and must be recomputed at each NLP iteration. The convergence rate advantage of MPEC over
nested CUE is lost. The remaining benefit is that IPOPT handles infeasible trial points via its
interior-point penalty rather than cold Newton restarts (which cost 237s per market in the nu=16
log for market 1991 on eval 2).

**Practical conclusion for Petrin data.** With the OMP parallelism fix applied, all 13 markets
converge on the first evaluation with the warm start and no cold restart is triggered. The
bottleneck is arithmetic, not inversion failures. MPEC overhead (NLP setup, constraint Jacobian
assembly for ~2400 constraints x ~2418 variables) would likely exceed any robustness savings.
Revisit if estimation at nu>=32 shows frequent cold restarts across many evaluations.

**IPOPT availability note.** Brew-installed IPOPT on this machine (glibc 2.29 requirement) is
incompatible with the system glibc. cyipopt therefore cannot be imported. scipy's `trust-constr`
and `SLSQP` support equality constraints but are not competitive with IPOPT at this problem size.

**For the write-up.** The MPEC benefit for CUE is conceptually different from MPEC for fixed-W GMM.
Papers that show MPEC speedups (Dube-Fox-Su 2012, Conlon-Gortmaker 2020) use fixed-W. For CUE the
argument rests on inversion robustness only, and the case depends on how often cold restarts occur.

## 11. The normalisation must be fixed on primitives, and counterfactual objects do not depend on it (2026-11-16)

**Finding.** The PCM needs one scale normalisation because (delta, t, alpha) -> kappa (delta, t, alpha)
leaves every choice unchanged. Writing log alpha_i = mu + sigma_p nu_i, the scaling is exactly a shift
of mu, so the one free quantity is mu. Fixing median alpha = 1 (mu = 0) is a normalisation on a
primitive of the model and is the right kind of normalisation. The pathology in entry 1 was never
the normalisation; it was the fixed-weight objective, which rewarded a small xi. Under CUE the same
median-alpha normalisation is harmless.

Two tempting alternatives are wrong, and both were proposed in this project before being retracted.
(a) Normalising the alpha of the average buyer or of the marginal buyer. Both are functions of the
equilibrium (who buys depends on delta, prices and sigma_p), not of primitives. Taste dispersion
moves the buy/no-buy margin: raising sigma_p puts more mass in the low-alpha tail that can afford a
car, so the inside share changes with sigma_p. A normalisation anchored to such an object has to be
re-anchored after every parameter change and after every counterfactual, which is to say it is not a
normalisation but a moving target. (b) The claim that "the buyer population does not change with
sigma_p" is false for the same reason.

**Why counterfactuals do not need the anchor to survive.** Every object we report is invariant to
kappa. Shares and the ranking of products are invariant by construction. Elasticities are ratios
of share changes to price changes and carry no utility units. Marginal costs recovered from Nash
first-order conditions are in dollars. Consumer surplus in dollars is computed per consumer as
max_j u_ij / alpha_i; multiplying numerator and denominator by kappa leaves it unchanged. So a
counterfactual that removes the minivan, moves the marginal buyer and changes the inside share
produces the same dollar welfare change under any kappa, as long as theta (including mu and sigma_p)
is held at its estimated value.

Worked example. Anna has alpha = 0.5, Boris alpha = 2, one product with delta = 10 at p = $4.
Anna's utility is 10 - 0.5 * 4 = 8, which is 8 / 0.5 = $16 in her dollars. Rescale by kappa = 2:
delta = 20, alpha_A = 1, alpha_B = 4. Anna's utility is 20 - 4 = 16, which is 16 / 1 = $16. Boris
has 10 - 8 = 2 before and 20 - 16 = 4 after, both $1 in his dollars, and if the product is removed
both lose exactly what they had. The normalisation changes the labels on the utility axis, never a
dollar figure.

**What the normalisation does affect.** Only the interpretation of utility-denominated parameters:
delta, xi and the taste spreads sigma_x are "dollars as valued by the consumer with alpha = 1".
Under median alpha = 1 that consumer does not buy a car, so a sigma_x of $2,000 means $2,000 to a
non-buyer and roughly $2,000 / alpha_i to buyer i. For reporting, convert spreads to buyer dollars
post-estimation (divide by the mean alpha among predicted buyers at the estimate). That is a
reporting convention applied once, not a normalisation to be carried into counterfactuals.

**For the write-up.** State the normalisation as mu = 0, justify it as the unique free scale, and
show the invariance of the reported objects. Report taste spreads in both median-consumer and
buyer dollars. Do not describe any normalisation on buyers.

## 12. Song's PCM normalisations: price-coefficient location and the outside option over time (2026-11-16)

**Source.** Minjae Song, "The Quality Adjusted Price Index in the Pure Characteristics Demand
Model", Simon School WP FR 09-04 (2008), CPU market 1993-2000, 321 product-quarters. The
estimation method is in Song (2007, RAND 38:429-446) and Song (2008, "Estimating the Pure
Characteristics Demand Model: A Computational Note"). The utility is
u_ijt = x_jt beta_i - alpha_i p_jt + xi_t + dxi_jt, u_i0t = delta_0t.

**Normalisation 1: log alpha ~ N(0, theta), location fixed at zero.** Song fixes the location of
the lognormal price coefficient at 0 and estimates only its spread (theta = 0.89 and 0.54 in the
two specifications; 0.84 and 0.50 with a second random coefficient). This is the same median
alpha = 1 normalisation as entry 11 and is a published precedent for it. The estimated constant in
delta is negative (-1.08, -0.73), i.e. the alpha = 1 consumer values the base CPU below the outside
good, exactly the pattern we see in Petrin: the normalising consumer is not a buyer, and that is
fine. His welfare objects are CV_it = (u_it - u_it-1) / alpha_i, which is our formula and is scale
invariant for the reason given in entry 11. He reports "the average consumer" as the one with
alpha_i equal to the mean; this is a reporting convention, not a normalisation.

**Normalisation 2: the outside-option level over time is not identified from shares.** Time
dummies in delta pick up the sum of two things, the mean of unobserved quality xi_t and the utility
of the outside option delta_0t, and shares cannot separate them. Song runs three identifying
assumptions and shows they matter enormously:

| assumption | time dummies identify | CPU index 96Q2 -> 00Q3 |
|---|---|---|
| delta_0t = 0 for all t | xi_t | rises to 905 (nonsense: more people bought better chips at lower prices) |
| xi_t = 0 for all t | delta_0t | falls to 1.4 |
| PBL (1993): mean dxi_jt of continuing products unchanged | delta_0t from continuing products | falls to 2.1 |

The third is the one he adopts. The change in delta_0t is read off the mean change of
(delta_jt - delta_0t) across products present in both t-1 and t, on the assumption that their
unobserved quality did not move on average. This matters more in the PCM than in BLP because the
buy / no-buy margin is a hard cutoff in alpha: whoever is assigned to the outside good is pinned by
delta_jt - delta_0t for the cheapest product, so a misassigned delta_0t relabels buyers as
non-buyers across years rather than shifting a smooth probability.

**Normalisation 3: period-mean unobservables xi_t can contaminate group welfare.** When products
enter or exit, xi_t moves even if every continuing product's unobserved quality is fixed, and that
shifts the surplus of consumers whose choice set did not change. Song recomputes group indices
with xi_t subtracted. He also runs a "dxi_jt fixed at the first-period value" variant that treats
residual movement as measurement error.

**What applies to Petrin.** (a) Keep mu = 0; cite Song as precedent alongside Berry and Pakes.
(b) Our X1 holds Petrin's pgnp, trend, trend2 with u_i0 = 0 in every year. That is Song's first
row, the assumption he rejects for CPUs. It is harmless for the 1984 no-minivan counterfactual,
which compares two choice sets in one year with one delta_0, so delta_0,1984 cancels. It is not
harmless for anything compared across years (welfare paths, a quality-adjusted car price index,
or the cross-year stability of sigma_x in buyer dollars), and for those the PBL continuing-product
assumption should replace the trend. (c) The group-by-alpha-quantile welfare table (Song's Table 6)
is the natural PCM deliverable for the minivan: who gained, by price-sensitivity quintile and
family size, which BLP cannot produce cleanly because the logit draw reassigns consumers each
period. (d) Song's one-RC model gave cache WTP of $80-110 and the two-RC model $200-400, matching
observed price gaps. The one-RC model forced a horizontal attribute to be vertical and understated
it. Same mechanism as entries 4 and 7.

## 13. CUE has the mirror-image artifact: it rewards a large, lumpy xi, reached through infinite taste spread on body-type dummies (2026-11-16)

**What happened.** Both CUE runs (4,000 and 16,000 consumers per market, warm start theta at
q = 39.6, sigma around $2-4k, sigma_p = 1.9) walked for 1,900-2,300 evaluations in the same
direction. The nu=4 run stopped with the L-BFGS convergence message at q = 0.803 after 8,550 s;
|grad| was still 0.38, so it is a flat region, not a stationary point. The nu=16 run was at
q = 1.146 and still descending when the process died at evaluation 2,288 (no error in the log;
the session that launched it was recycled). End points, in $1000:

| | nu=4 (stopped) | nu=16 (eval 2288) | start |
|---|---|---|---|
| sigma_sw | 117 | 126 | 3.8 |
| sigma_su | 107 | 132 | 2.1 |
| sigma_const | 27.3 | 0.36 | 3.8 |
| sigma_hpwt | 8.1 | 27.9 | 4.2 |
| sigma_air | 17.0 | 16.8 | 2.1 |
| sigma_pv | 19.5 | 32.0 | 2.1 |
| sigma_p (log alpha sd) | 0.20 | 0.14 | 1.89 |
| sd(xi) | 10.1 | 11.3 | 1.38 |
| q_CUE | 0.80 | 1.15 | 39.6 |

Where the spread landed differs between the two runs (sigma_const in one, sigma_hpwt in the
other), so the point is not stable in the number of draws either. Clustered SEs are NaN or 10^5.

**What the point means economically.** A station-wagon taste sd of $117k on a dummy, against
cars priced $5-50k, means 67% of consumers have a wagon taste larger in absolute value than
any price. Each consumer is either wagon-only or never-wagon by their draw; the same for SUVs
(64%). The market is cut into separate body-type pools, and beta_sw = -$251k, beta_su = -$220k
set the pool sizes to the observed body-type shares. Within a pool sigma_p = 0.2 gives alpha a
5th-95th range of 0.72-1.39, which is the near-vertical model again. At the nu=4 point the median
own-price elasticity is -219 (fixed-W gave -56), the median product has nonzero cross-elasticity
with 7 of about 185 rivals, and under 0.3% of cross-elasticity mass crosses a body type. The
ownership-weighted share Jacobian in 1984 is singular, so the Nash first-order condition and the
minivan counterfactual cannot be computed.

**Mechanism.** CUE is q = g' S(theta)^-1 g with g = Z'xi/n and S the clustered covariance of
z_j xi_j at the same theta. Entry 1 showed q is invariant to xi -> kappa xi. It is not invariant
to the shape of xi. g averages xi across products, S squares it product by product. Make a few
products' xi very large and S grows faster than g'g, so q falls, with no improvement in how xi
correlates with the instruments. This is the known weakness of CUE (Hansen, Heaton and Yaron
1996; Newey and Smith 2004): the objective can be lowered by making the moments noisy. The
optimizer found the route: sd(xi) rose from $1.4k to $10k while q fell from 39.6 to 0.8.

**Why the route is open.** Nothing in aggregate shares with characteristics-of-rivals instruments
pins how much substitution crosses body types. Fixed-W GMM takes one degenerate exit (shrink xi,
vertical model, entry 1) and CUE takes the other (inflate xi, segmented near-vertical model).
BLP never faces this because the logit error forces positive cross-body substitution for every
consumer. Petrin faced it and solved it with micro moments: E[minivan | family size], E[SUV |
age] and so on from the CEX, which are exactly moments on body-type choice by demographic.
Those moments are in the pyblp Petrin data and we have not used them.

**Implication for the estimation plan.** (a) Neither fixed-W nor CUE on aggregate moments alone
is a usable estimator for this specification. (b) Add Petrin's micro moments to the moment
vector; they are the economically right identification for the body-type spreads and make the
PCM spec match BLP-with-micro-moments, which is the comparison the paper wants. (c) With micro
moments in place, use two-step GMM with W from a bounded, interior point (sigma on dummies capped
at the price range, say $50k, as a normalisation of an unidentified direction, not a prior), and
report that cap. (d) Treat a falling q together with a rising sd(xi) as the diagnostic for this
artifact, as a falling q with a falling sd(xi) was for entry 1.

## 14. MPEC does not apply to the exact PCM, with or without micro moments, and it is not the cure for the degenerate estimates (2026-11-16)

**Setting.** Entry 13 left the plan at nested two-step GMM with aggregate moments plus Petrin's
CEX micro moments, from an interior start, with sigma on the body-type dummies capped at the
price range. The question was whether an MPEC formulation (Dube, Fox and Su 2012: treat delta
as decision variables, impose sigma(delta, theta) = s as equality constraints, hand the problem
to an interior-point NLP solver) would help there. Entry 10 answered for CUE in general terms;
this entry is the sharper answer for the exact PCM.

**The results are an identification problem, not an optimisation problem.** Both the fixed-W
run (entry 1) and the CUE runs (entry 13) converged to genuine minima or plateaus of the
objective they were given. The fixed-W descent toward sigma_p -> infinity is a real descent
direction of that objective, and the CUE point at q = 0.80 is a real flat region of that
objective. An algorithm that reaches the same points faster changes nothing. The fix is to
change what the moments reward, which is what the micro moments do.

**Three properties of BLP that MPEC relies on and the exact PCM lacks.**

1. *Smooth constraints.* In BLP every consumer buys every product with positive probability, so
   shares are C-infinity in delta. In the PCM, for one taste draw, a product B priced between A
   and C has share zero while it lies below the A-C line in (price, delta) space and share
   F(e_AB) - F(e_BC) once above it. Example: p = 10, 12, 14 for A, B, C with delta_A = 5 and
   delta_C = 7; B is on the hull only when delta_B > 6. At delta_B = 5.99 the share and its
   derivative are both zero; at 6.01 the derivative is about f(e)(1/2 + 1/2) = 0.4 per $1,000.
   The second derivative is a spike at 6. With 16,000 draws the constraint Hessian is a sum of
   16,000 spikes per product. IPOPT's Newton step on the Lagrangian uses a Hessian that jumps
   between iterates, so the second-order convergence that MPEC is bought for does not arrive.

2. *LICQ.* A product below the frontier for every draw has d sigma_j / d delta = 0 as a whole
   row while its constraint sigma_j = s_j > 0 is violated. The gradient gives the solver no
   direction in which to raise delta_j. This is the singular Jacobian that stalled the nu=16
   run on markets 1991-1993. The nested code recovers with the frontier lift, a combinatorial
   step (compute the exact delta_j that puts j on some consumer's hull). An NLP solver has no
   such step, and because MPEC iterates are infeasible by design, this state is visited at most
   iterations rather than only at bad trial points.

3. *A fixed quadratic objective in delta.* With fixed W and aggregate moments only,
   q = delta' M' Z W Z' M delta / n has a constant Hessian. Micro moments are nonlinear in delta
   (predicted E[family size | minivan buyer] is a ratio of share sums) and pool across years, so
   they couple delta across markets. The objective Hessian stays low rank (22 aggregate plus
   about 10 micro moments), so this is a mild loss, but it removes the one piece of MPEC
   structure that would otherwise have survived. Adding micro moments moves the MPEC verdict in
   the wrong direction, not the right one.

**Ceiling on what MPEC could save.** Per evaluation, the nested code pays one share-and-Jacobian
pass per Newton iteration, 3 to 6 iterations on a warm start. MPEC pays one pass per NLP
iteration and needs roughly as many outer iterations as L-BFGS. The ceiling is therefore a 3-6x
saving on smooth stretches, with a loss on kinked ones. The actual wall-clock cost is the
16k-draw share pass on markets 1985, 1988 and 1990, which MPEC pays identically; entry 9 already
addressed that cost with OMP threads.

**What the micro moments do and do not identify.** P(buy | income quartile) pins the alpha
distribution against income: at sigma_p -> infinity the unobserved spread swamps the income
gradient and purchase probability goes flat across quartiles, which closes the fixed-W escape
route of entry 1. E[family size | body type] and the age moments identify pi_body x fs and the
ratio pi / sigma_body: with sigma_sw = $117k the wagon pool is assigned by the random draw, so
wagon buyers would have the population mean family size, contrary to the data. They do not pin
the absolute level of sigma_body: a corner with pi and sigma both scaled up keeps the pools
sorted by family size and matches these moments. Second-choice data would pin it; Petrin has
none. What rules that corner out is the fixed-W penalty on the lumpy xi it produces (sd(xi)
rose 1.4 -> 10 at the CUE corner). So fixed W with micro moments is the right pairing, CUE is
not, and the price-range cap on sigma_body is the stated normalisation for the direction the
data leave quiet.

**Check before the full run.** Rerun the sigma_p ray profile (profile_sigma_ray.py) with the
micro moments included. q must stop being monotone in sigma_p. If it is still monotone, the
micro moments are not doing what this entry claims and the plan needs rethinking before any
long estimation.

**Smoothing is not an escape route.** See entry 15: the smoothed PCM is rejected for this
project, so MPEC and sparse grids stay off the table.

## 15. The smoothed PCM (logit error with scale lambda -> 0) is rejected, permanently (2026-11-16)

**Decision.** This project estimates the exact PCM. Adding a logit error with scale lambda and
sending lambda toward zero (Berry and Pakes 2007) is not used for estimation, inversion or
counterfactuals. The reasons are mathematical, computational and economic, and each one holds
by itself.

**Mathematical: the kinks become stiffness, not smoothness.** With the error, the share of a
product that sits a utility gap g below a consumer's frontier is about exp(-g / lambda). Its
derivative is about exp(-g / lambda) / lambda. Example: g = 0.5, lambda = 0.01 gives
exp(-50) = 2e-22, below double-precision relevance next to shares of 1e-3. So the "full-rank
Jacobian" of the smoothed model is numerically the same zero row as the exact PCM, and LICQ
fails in floating point exactly where it fails in exact arithmetic. Near a hull crossing the
second derivative is of order 1/lambda, so the Hessian is bounded only in theory. Smoothing
moves the kink into the condition number.

**Computational: the contraction dies as lambda -> 0.** The BLP contraction modulus tends to 1
as the error scale shrinks (Berry and Pakes 2007 state this), and Newton steps scale with
1/lambda. Small lambda means slow or unstable inversion; large lambda means a different model.
There is no lambda that is both close to the PCM and well conditioned.

**Economic: any lambda > 0 is the BLP love-of-variety term we are trying to measure without.**
The logit error adds lambda * log(sum exp(.)) to expected surplus, which grows with the number
of products regardless of characteristics. Entry 13 / the Petrin comparison is precisely about
how much of the $367M minivan gain is that term. A smoothed PCM puts it back, with a size set
by an unidentified tuning constant. If lambda is estimated, the model drifts back to BLP; if it
is fixed, every welfare number depends on an arbitrary choice.

**What is still allowed.** Using exact PCM structure inside our own algorithms (frontier lift,
breakpoint-aware line search, see the custom optimizer plan) is the approach. Nothing that
changes the model's shares is.

## 16. Every Petrin PCM estimate so far was under-identified by construction (2026-11-16)

**Finding.** `petrin_replication/estimate_pcm.py` used Z = demand_instruments0..21 (22 columns)
as the only instruments, and concentrated out the 13 linear coefficients of X1 (constant,
characteristics, body dummies, pgnp, trend, trend2) by IV on that Z. pyblp builds
ZD = [demand_instruments, exogenous X1] (35 columns). On the 2,407 Petrin products,
rank(Z) = 22 and rank([Z, X1]) = 35, so X1 is not in the span of Z. The moment count was 22 against
13 + 14 = 27 unknowns (base spec) or 13 + 18 = 31 (body-sigma spec). Five to nine directions of
theta were unidentified by the moment conditions themselves.

**What this changes.** The flat directions, the drift between 4,000 and 16,000 draws, and the
near-singular SEs (entries 1 and 13) all have this as a direct cause. The scale mechanism of
entry 1 is still correct mathematics (q is quadratic in the scale of xi under fixed W and the PCM
has a free scale), and the CUE shape mechanism of entry 13 is still correct, but neither run can
be read as evidence about what the data identify. The fixed-W and CUE endpoints are results of an
estimator with too few moments, not of the PCM on Petrin's data.

**Lesson for the paper.** Use the full instrument matrix [exogenous X1, excluded instruments],
check rank(Z) and rank(Z'X1) before any estimation, and count moments against unknowns
(concentrated linear coefficients included). The estimation harness runs this check before every
estimation and refuses to start when moments < unknowns. Found by the harness design review
(docs/estimation_harness/DESIGN.md, F1).

**Nevo.** The Nevo PCM absorbs brand fixed effects by within transformation, so there is no
concentrated X1 apart from the absorbed effects, and its 20 instruments are not affected in the
same way. The harness check will confirm this when Nevo is ported.
