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

## 17. The earlier PCM consumer-surplus numbers came from a buggy function and are void (2026-11-16)

**Bug.** `consumer_surplus` in the old local `petrin_replication/pcm_postestimation.py` walked
the upper hull of (0,0), (p_k, q_k) and gave each frontier product the alpha interval
(e_lo, previous e), starting with previous e = infinity. For the first segment, from the outside
good to the cheapest frontier product, e_lo = q/p is the cutoff below which a consumer buys. The
code therefore assigned that product the consumers with alpha > q/p, who buy nothing, and every
later product inherited an interval shifted by one segment.

**Example.** One product, q = 2, p = 1, log alpha ~ N(0, 0.5^2). Buyers are alpha < 2, and
CS = 2 E[1/alpha; alpha < 2] - P(alpha < 2) = 1.282. Quadrature of max(q/alpha - p, 0) gives the
same number. The old function returned -0.016, the "surplus" of the non-buyers, which is negative.
On a random 30-product, 500-draw market it gave 3.90 against the correct 42.72, and produced
negative per-draw surplus.

**Consequence.** The PCM minivan CS gain of about -$1M (entry 13 era, file
pcm_vs_blp_body_nu16.csv) is void on two counts: this bug, and the under-identified estimator of
entry 16. The economic reading built on it ("the PCM says the minivan added almost nothing, so
Petrin's $367M is mostly logit variety") is withdrawn until the harness re-estimates and recomputes
CS with the tested kernel. Elasticities and margins came from other code and are affected only by
entry 16.

**Fix.** `demand_model/pcm_welfare.cpp` (harness WP3, commit 1a3b7bc) computes CS by the frontier
formula with the correct intervals, tested against a closed form, independent quadrature,
scale invariance, dominated-product invariance and Roy's identity (dCS/dp_j = -s_j).

## 18. The trust-radius collapse came from exponentiating price-over-income, not from the price tie break (2026-11-17)

**Symptom.** In the harness prefit (Petrin, 4,096 draws per market) trial points 1.5e-6 from the
accepted point failed to invert, mostly in market 1993, and the trust radius fell from 0.2 to
1.5e-6. The failing products were the most expensive cars (positions 188-198 of 202, neighbour
price gaps $277 to $2,931), not the tie-broken pairs ($10 gaps).

**Cause.** The PCM spec carried Petrin's BLP price interactions low/income, mid/income,
high/income into the exponent: log c_i = phi_low low_i/income_i + .... In BLP these enter
linearly, alpha_i = pi_g g_i/income_i, so the lowest CEX income in the data (0.0036, against a
low-group median of 5.8 and an overall maximum of 359) only multiplies alpha by 280.
In the PCM the same ratio is exponentiated: low/income reaches 280, and phi_low = -0.091 gives
c_i = exp(-25.6) = 7e-12. A handful of draws (0.1% have c_i < 1e-6) become price-insensitive by
eleven orders of magnitude. Such a consumer buys the car with the highest delta + x.t at any
price, and switches between luxury cars when delta changes by about c_i times a price gap, around
1e-11. Each switch moves a share of about 1e-4 by one draw weight (2.4e-4). The share map is then
discontinuous at the inversion tolerance of 1e-9.

**Evidence (market 1993, prefit theta).**

| | c_i min | cond(ds/ddelta) | cold solve | warm solve at +1e-6 |
|---|---|---|---|---|
| prefit phi | 7.3e-12 | 1.7e11 at the accepted delta | infeasible_start, 636 s | not_converged, residual 1.2e-9 |
| phi_low = 0 | 0.26 | 2.5e4 | ok in 18 s | ok in 5 evaluations, residual 4e-14 |

The prefit also has the wrong sign economically: phi_low < 0 makes the poorest households the
least price-sensitive, while Petrin's alpha_i proportional to 1/income makes them the most.

**Price ties.** Not the cause here. The kernel requires strictly increasing prices because the
alpha cutoff between adjacent frontier products is (difference in conditional quality)/(price
gap). Exact ties can be handled exactly: at equal prices, for each taste draw only the product
with the higher delta + x.t can be on the frontier. With horizontal tastes the products still
compete, through the taste draws. The $10 spread is a convenience, not a requirement.

**Lesson.** Moving a BLP demographic interaction into a PCM exponent changes its scale from
linear to exponential. Check the range of every regressor that enters log alpha, and use
log(income) rather than 1/income, so that the price-sensitivity range is bounded by the income
range.

## 19. Specification review by an independent economist agent (claude-fable-5.1), and the resulting spec (2026-11-17)

An economist agent on a different model read entries 1-18, the specs and the failed gate, ran short
data checks, and proposed the specification below. The points that change the plan:

**Substitution in the exact PCM is local, and vertical within clusters of identical observables.**
Median p / (gap to the next-priced product) is 173 in 1984. 43 of 163 products in 1984 share the
profile (fwd only), 39 have no features, and there are 92 exact price ties, 17 of them with
identical x. Two products with the same x are a vertical pair for every consumer, with own
elasticity of order p / gap. Moderate elasticities need per-product quality dispersion across
consumers of thousands of dollars, which BLP gets from the logit error (sd 1.28 utils at about 0.6
utils per $1,000, about $2,100) and the PCM must get from the taste spreads.

**The gate as run could not pass on substance.** It fixed phi at the prefit values while varying
sigma_p, so the income purchase-rate moments drifted and q_micro rose mechanically (112 to 188 along
lambda = 1) while q_aggregate still fell (83 to 68). Only a profile that re-solves phi at each point
tests entry 14.

**The data require price sensitivity that is flat across income groups.** Pooled CEX purchase
rates: low about 0.102, mid 0.079, high 0.158. Petrin's BLP fits this with pi_g / median income_g of
0.67, 0.67, 0.58, i.e. alpha flat across groups and proportional to 1/income within them. Within the
low group 1/income spans a factor of 3,000 (1% of agents report incomes under $262), so any
exponentiated function of income inherits that tail (entry 18).

**Weak identification of the spreads.** At the last prefit, Cragg-Donald is 12.9 against a critical
value of 49.8. BLP sum-of-characteristics instruments are built for beta, not sigma. Gandhi-Houde
local differentiation instruments target the spreads.

**Resulting spec.** Taste spreads on const, hpwt, space, air, mpd, fwd and one common "family hauler"
factor fam = max(mi, sw, su, pv), capped at $25k so under 1% of consumers value it beyond the price
range (the old $66.1k cap left 32% beyond it, inside the segmented corner of entry 13). Body-specific
spreads fixed at 0 in the main run, since only second-choice data would pin them (entry 14). Petrin's
pi_b log(fs) fv on the four body dummies. log alpha_i = phi_mid mid_i + phi_high high_i + sigma_p
nu_ip with the low group as reference: c_i is bounded by construction. Robustness: phi_y fixed at 1
on log(ybar / income) with income winsorised to [1, 150]. Merge the 17 same-firm identical-x exact
twins; keep the $10 spread for the other ties. Starts mapped from pyblp's BLP-micro estimate divided
by alpha of about 0.65 utils per $1,000: pi = (0.75, 0.28, 0.20, 0.41); all spreads about $2,000 per
sd. Gate: concentrated sigma_p profile with (phi_mid, phi_high) re-solved at each point. Step 1
fixed W1, step 2 with S^-1 and Petrin's micro covariances.

**Fallback if the concentrated profile is still monotone in sigma_p.** Replace the aggregate block by
its Sargan form, q_S = n xi' P_Z xi / xi' xi. It is invariant to the scale of xi, which closes the
entry 1 route, and it divides by the global variance rather than product by product, so a single
huge xi_j drives q_S toward n h_jj, not toward zero, which keeps the entry 13 route closed. If the
profile is flat rather than monotone, calibrate sigma_p to BLP-micro's inside-share semi-elasticity
and report it as a calibration.

**Plausibility targets.** Median own elasticity in [-8, -3]; per-product quality dispersion across
consumers $1,500-$6,000; minivan diversion to wagons, SUVs and vans at least 0.15 of inside
diversion; inside-share semi-elasticity to a uniform 1% price rise in [-2, -0.5]; median Nash margin
0.15-0.40; each micro moment within 2 Petrin sds. Note from the agent's read of the BLP outputs: the
demand-only BLP fit with the PCM's moments gives own elasticity -1.97, margin 0.57 and a 1984 gain
of $1,046M at fixed prices, so demand-only BLP is itself fragile on these moments.

## 20. With phi matched to the income purchase rates, the aggregate objective still prefers ever larger sigma_p, and elasticities stay implausible (2026-11-17)

**Run.** Spec of entry 19 at 4,096 draws per market, concentrated sigma_p ray profile: lambda in
{0.5, 1, 2} times the reference spreads, sigma_p in {0.5, ..., 2.5}, (phi_mid, phi_high) re-solved
at every point with the income rows up-weighted so both purchase rates are matched. All 18 points
inverted (entry 18's failures are gone).

**Result.** q_fixed falls monotonically in sigma_p in every slice. At lambda = 1: q_fixed 1,607,
645, 360, 221, 192, 183 for sigma_p 0.5 to 2.5. Its aggregate block falls 1,498 to 76 while sd(xi)
falls 1.53 to 0.59, so it tracks sd(xi)^2 (the entry 1 mechanism). The non-income micro block barely
moves (109 to 107 at lambda = 1; at lambda = 0.5 it bottoms at sigma_p = 1.5 and rises 171 to 179).
The scale-free CUE aggregate block is almost flat in sigma_p (191 to 186). So the aggregate moments
carry almost no information on sigma_p beyond the size of xi, and the micro moments that exist do not
pin it.

**Economics along the profile.** Median own elasticity -88 at sigma_p = 0.5 and -12 at 2.5
(lambda = 1); nonzero cross-elasticities per product 14 to 8; diversion leaving the body type
essentially zero (0.004 to 0); inside-share semi-elasticity to a 1% price rise -2.0 to -0.4. Larger
sigma_p buys lower own elasticities only by spreading alpha so widely that substitution becomes
vertical within body types and nothing crosses them. Larger lambda (more horizontal spread) lowers
own elasticities too (-14 at lambda = 2, sigma_p = 2) but raises q_fixed.

**Sargan form.** The same profile with the aggregate block in Sargan form (entry 19 fallback, run
petrin_pcm_e1559c0b2b) is also monotone decreasing: at lambda = 1, q_fixed 753, 639, 515, 381, 337,
322. So the preference for large sigma_p is not only the scale of xi: R^2 of xi on Z also falls as
sigma_p rises.

## 21. Economist review round 2: in the exact PCM, own elasticity is about -35 to -45 / sigma_p on this product set, and sigma_p needs the supply side (2026-11-17)

Second review by the economist agent (claude-fable-5.1), with its own evaluations at 1,024 draws and
a per-draw frontier decomposition of 1984.

**Why each objective prefers larger sigma_p.** Fixed W tracks sd(xi)^2 (entry 1). The Sargan form is
scale-free but falls through shape: larger sigma_p flattens delta in price, which is the direction
in which rival-characteristic instruments have leverage, so R^2 of xi on Z falls from 0.27 to 0.09.
The CUE aggregate block moves 191 to 186 across sigma_p 0.5 to 2.5, which is noise against 22
overidentifying restrictions. The demand data carry no usable information on sigma_p: the income
purchase rates pin phi_mid and phi_high, E[fs | body] pins pi_b, and nothing pins within-group
dispersion of alpha. Petrin's BLP has no unobserved alpha dispersion (alpha = pi_g / income, pooled
log-alpha sd 0.50) and gets its smoothing from the logit error.

**The own-elasticity law.** For a draw with product j on its frontier, j is bought on an alpha
window of mass m_ij; a price rise shifts both window edges by dp / (hull price gap), so own
elasticity = -(p_j / hull gap) x (edge densities) / (sigma_p x E[m_ij | on frontier]). Hull gaps are
already about $2,300 because each draw's frontier is sparse. The binding term is the window mass,
0.05 to 0.11: a product is on the frontier of 9 to 20 times as many consumers as buy it. The
inversion fixes the z-windows from the shares, so elasticity scales as 1 / sigma_p. Measured
elasticity x sigma_p is 31 to 54 at every point evaluated. Ten times larger horizontal spreads
($2,000 per sd) move it only from 43 to 35; sigma_const of 8 instead of 3 changes nothing. An own
elasticity of -8 would need sigma_p near 4.4. The ratio of own to market elasticity is about 35 to
45 in the exact PCM on these 160 to 210 cars (median p / price gap 174), against 1 to 1.5 in BLP.
No choice of horizontal spreads delivers BLP-like own elasticities.

**Correction to entry 19.** Its natural-unit starts were $150 to $1,000 per sd, not the intended
$2,000. The reference direction is now defined per sd of each characteristic. This does not change
the conclusion above.

**Substitution diagnostics.** The family-hauler factor works: minivan diversion to wagons, SUVs and
vans is 0.18 to 0.31 of total. "Diversion leaving body, median" was zero because the median product
is a car; the right statistic is car-to-hauler diversion relative to the hauler inside share.
Outside diversion near zero is structural in the PCM: only a consumer's cheapest frontier product
borders the outside good.

**What pins sigma_p: the supply side.** Rival characteristics move prices only through markups. If
markups were 3% (own elasticity -48), prices would equal costs and the rival sums would have no
first-stage power. Check (a): regressing log price on Petrin's cost basis, the 22 demand instruments
add F = 14.5 (R^2 0.835 to 0.851) and the 16 supply instruments F = 15.3, so markups are identified
from prices. Petrin estimated with the supply side; the PCM runs so far dropped it, which is why
sigma_p floats. Main path: add log mc = w gamma + omega with Nash markups from the PCM's ds/dp,
w = Petrin's cost basis, Z_s = [w, supply_instruments]. Expected sigma_p 2 to 3, own elasticity -10
to -15, margins 7 to 10%, reported as the exact PCM's numbers beside BLP's -4.5 and 24%.

**Fallback.** If the supply profile is still monotone or flat, treat sigma_p as set-identified:
estimate the rest at sigma_p in {1, 1.5, 2, 2.5, 3}, report elasticities, margins and the minivan
counterfactual as a band, headline sigma_p = 2.0 labelled a calibration. BLP-based calibrations of
sigma_p are rejected: Petrin's income-implied dispersion (0.50) gives elasticities near -90, and
matching BLP's market semi-elasticity gives sigma_p near 0.3.

**Plausibility targets reset for the exact PCM.** Median own elasticity (-15, -5), margins (0.07,
0.25), inside-share elasticity (-1.5, -0.4), minivan diversion to haulers at least 0.15 of inside.

## 22. Supply block: Nash markups above price at large sigma_p; FOC costs clipped at a floor (2026-11-17)

**Finding (supply check, nu = 4, lambda = 1, phi concentrated on demand moments).** The supply
block could not be evaluated with a hard mc > 0 guard for sigma_p >= 1. Products with |own
elasticity| < 1 get a Nash markup larger than their price: 1 product at sigma_p = 1 (1989, firm 19,
$4.8k, own elasticity -0.68), 2 at sigma_p = 2, 94 at 3 and 336 at 4, mostly cars.

| sigma_p | q demand (agg + micro) | supply q, mc <= 0 dropped | mc <= 0 | median own elasticity | elasticity x sigma_p | median margin | market semi-elasticity |
|---|---|---|---|---|---|---|---|
| 0.5 | 1,901 | 9.4 | 0 | -51.1 | 25.5 | 0.035 | -1.25 |
| 1 | 548 | 8.7 | 1 | -27.9 | 27.9 | 0.064 | -0.80 |
| 2 | 337 | 11.2 | 2 | -12.3 | 24.7 | 0.158 | -0.42 |
| 3 | 326 | 26.4 | 94 | -7.3 | 21.8 | 0.273 | -0.27 |
| 4 | 335 | 40.5 | 336 | -4.8 | 19.2 | 0.405 | -0.20 |

With per-sd spreads (entry 21) and phi re-solved on demand moments, the demand objective has an
interior minimum near sigma_p = 3. The supply diagnostic is lowest near sigma_p = 1 and rises
steeply from 3, as markups grow and many products lose a positive cost. Minivan diversion to
haulers is 0.25 to 0.30 of inside diversion. Elasticity x sigma_p is 19 to 28 here, below entry 21's
band because phi_high moves from -0.36 to -2.30 along the profile.

**Treatment.** FOC costs are clipped at mc = $500, as pyblp's costs_bounds does, with zero derivative
for clipped products, and the clipped share reported. A clipped product's log-cost residual (log 0.5
against about 2.5) is the supply block's penalty on the large-sigma_p region. A headline estimate
with more than 1% of products clipped is flagged.

## 23. With the supply side and a cost floor, the joint objective has an interior minimum in sigma_p (2026-11-17)

Gate at 4,096 draws per market, all three blocks (aggregate, micro, supply), phi re-solved at each
point, all 21 points inverted, no core.

| lambda | argmin sigma_p of q_fixed | q_fixed at min | own elasticity at min | margin at min | minivan diversion to haulers |
|---|---|---|---|---|---|
| 0.5 | 2.5 | 212.7 | -10.4 | 0.18 | 0.22 |
| 1 | 2.0 | 348.2 | -12.3 | 0.16 | 0.29 |
| 2 | 2.0 | 1,057.8 | -10.0 | 0.18 | 0.34 |

The two sides bound sigma_p from opposite directions. The aggregate block falls in sigma_p (the
entry 1 and 21 mechanisms) and the supply block rises steeply past sigma_p of about 2 (lambda = 1:
8.7 at 1.5, 49.6 at 3, 140.3 at 4), because large markups push costs to the floor. The supply block
alone is flat on the left (9.4 at 0.5, 8.7 at 1.5), so it fails its own both-ends rule; the total
passes. Elasticity x sigma_p is 17 to 38, consistent with entry 21's law. Smaller horizontal spreads
(lambda = 0.5) fit better at this point, so the estimator should be free to move lambda. No product
was clipped at the minima (share_mc_nonpositive 0 at every point).

The main estimation (extensive, step 1, step 2, post, stability) was launched at 16,384 draws per
market from the prefit point.

## 24. Step 1 at 16,384 draws slid back into the scale artifact through the price-sensitivity distribution (2026-11-17)

**Run.** Main estimation (extensive, then step 1 with fixed W1 on aggregate + micro + supply, 73
moments, 14 nonlinear and 25 linear parameters) from the prefit point, nu = 16 (run
petrin_pcm_8cdfbfe173).

**Result.** Step 1 ended STALLED at q = 23.85 (aggregate 5.6, micro 11.6, supply 6.7) from 332.7,
with projected gradient 437 and 10 consecutive rejections at a trust radius of 7e-9. Step 2 was
refused. The diagnostics named it: SCALE_ARTIFACT (q_fixed 333 to 23.9 while q_cue rose 287 to 310;
sd(xi) 1.21 to 0.15) and WEAK_IDENTIFICATION (Cragg-Donald-style 4.1 against 79.1).

**Mechanism.** At the stall sigma_p = 2.30 and phi_high = -0.76 (from -0.42), so buyers sit far in
the low-alpha tail: mean alpha among buyers is 0.053 against median 1 in the low group. Every
utility-scale parameter shrank in median-consumer dollars (sigma_const 0.48, sigma_hpwt 0.68 per
unit, beta_hpwt -0.45), so delta and xi shrank with them and the fixed-W aggregate block fell
quadratically. This is entry 1's route, reached here through phi_high and sigma_p jointly. In buyer
dollars the spreads stayed sensible ($2k to $13k per sd), which is why the scale artifact looks like
a normalisation question. The supply block did not stop it: its 6.7 at the stall is lower than the
gate's 8 to 10.

**Economics at the stall (not interpreted, did not converge).** Median own elasticity -25.6, 58% of
products below -20, margin 7.1%, minivan diversion to haulers 0.51 of inside, car-to-hauler ratio
0.010.

**Next.** The fixed-W aggregate block must be scale-free when every nonlinear parameter is free:
the Sargan form (entry 19 fallback) with the micro and supply blocks, gate first at nu = 4.

## 25. Economist review round 3: the markup level is not pinned by the data; calibrate it, estimate the rest scale-free (2026-11-17)

Third review by the economist agent (claude-fable-5.1), with checks at 4,096 draws at the
extensive point and the step-1 stall of entry 24.

**What the slide was.** Not a pure rescaling: elasticities and margins changed (-10.2 to -25.6,
18.4% to 7.1%). The taste spreads shrank 2.5x (const) to 50x (fwd) in median-consumer units, so who
buys was decided by alpha alone, buyers became the far tail of a lognormal with sd 2.3 (mean alpha of
buyers 0.24 to 0.053), and the inversion set delta near tail-alpha x p. sd(xi) fell 8x: 2.7x from
sd(delta), 3x from fit. The fixed-W block fell 47x against 64x for sd(xi)^2 while CUE rose: entry 1,
reached through sigma_x.

**Sargan is necessary, not sufficient.** At nu = 4 the Sargan block charges the slide 74 points (175
at the extensive point, 248 at the stall; CUE 78) against a supply gain of 7. The supply block is
invariant to rescaling but not neutral: PCM markups are lumpy (own elasticity 5th percentile -166,
95th -4.4), lumpy markups are log-cost noise Z_s cannot explain, so the block falls as elasticities
grow. It bounds markups only from above through the cost floor; the passing gate of entry 23 was the
fixed-W artifact balanced against the floor, not an estimate. Scale-free blocks are nearly flat in
sigma_p, so a free Sargan step 1 will sit on a weak ridge (Cragg-Donald 4.1 against 79).

**No normalisation fixes a fixed-W objective.** Median alpha in another group only rescales xi;
mean alpha makes it worse; buyer anchors are equilibrium objects (entry 11); fixing phi or beta_hpwt
closes too little or is weakly pinned. Two-step S(theta_gate)^-1 is not scale-free even to first
order. CUE on the aggregate block with fixed micro and supply blocks is coherent; Sargan is preferred
for the lumpy-xi route, CUE as a cross-check.

**The stall was not a Jacobian error.** The micro block is stiff along pi_b (curvature about 1.7e6
per natural unit squared from Petrin's covariance weights; a 1e-3 probe raised it 0.54 as predicted).
The supply block has step-size-independent jumps of about 0.01 where a draw's frontier changes ds/dp,
and its central difference at h = 1e-6 returns the slope of a piece 1e-6 wide. A projected-gradient
tolerance of 0.005 sits below that noise floor. Fixes: supply secant step 1e-3 in natural units, a
CONVERGED_NOISE class, Levenberg-Marquardt damping on rejection, pi_b concentrated on the body micro
moments at the start of step 1.

**Main path.** Sargan aggregate block by default. A binding markup-level moment, median (p - mc)/p =
m*, reported as a margin calibration. A margin_band stage over m* in {0.10, 0.15, 0.20, 0.24, 0.30}
with all other nonlinear parameters free; headline m* = 0.24, the median margin of Petrin's BLP with
micro and supply on this product set, so the comparison with BLP holds the markup level fixed and
isolates substitution. The Sargan J across m* decides whether m* is estimated (interior minimum
rising 3.84 at both ends) or calibrated (the band is the result). Fallback: fixed W1 with the margin
moment binding, sigma_p fixed at 2.0, lambda free, band over m*.

## 26. Sargan aggregate block with micro and supply: the gate prefers larger horizontal spreads (2026-11-17)

Same gate as entry 23 (nu = 4, phi concentrated, 21 points, all inverted) with the aggregate block in
Sargan form. q_fixed has an interior minimum in every slice: lambda 0.5 at sigma_p 2.5 (336.8),
lambda 1 at 2.0 (252.1), lambda 2 at 1.5 (213.2). Under fixed W the ranking was the reverse (lambda
0.5 best at 212.7, lambda 2 worst at 1,057.8), because the fixed-W block rewarded the small xi of
narrow spreads. With a scale-free aggregate block the objective prefers the widest spreads tried,
which is the direction BLP-like substitution needs. Elasticities at the minima: -10.4, -12.3 and
-13.6; margins 0.18, 0.16, 0.14. The supply-only both-ends rule still fails (flat left), as entry 25
explains: supply bounds markups only from above. These are the starting points for the margin band.

## 27. A binding margin row in a Gauss-Newton objective makes every direction collinear; calibrate through sigma_p instead (2026-11-17)

**Run.** margin_band at nu = 16 (run petrin_pcm_8cdfbfe173), Sargan + micro + supply + one margin row
weighted 1e4 / (n 0.01^2), from the Sargan gate minimum (lambda 1, sigma_p 2, margin 0.16).

**Result.** All five points stopped within 2 to 4 minutes: m* = 0.10, 0.24, 0.30 STALLED at q of 4e5,
2.4e5, 8.6e5 with projected gradients of 6e6 to 9e6; m* = 0.15 CONVERGED_NOISE at 4.3e4; m* = 0.20
STALLED at q 254.7 (aggregate 173.8, micro 9.8, supply 11.8, margin 59.3), sigma_p 2.07. The report
flagged every taste spread as collinear with sigma_p in G (correlations 0.990 to 0.999).

**Mechanism.** The margin row's weight is about 1e8 / n, so G'WG is rank one along the margin
gradient. Every nonlinear parameter moves the margin, so the Gauss-Newton model cannot separate
them, and the trimmed-mean margin is itself a kinked function of theta (markups jump with ds/dp), so
trial steps along the dominant direction overshoot. A penalty that binds by weight is the wrong tool
for a nonsmooth calibration target.

**Change.** Calibrate the markup level through sigma_p, the parameter that sets it (entry 21: own
elasticity about -25 / sigma_p): the sigma_p_band stage fixes sigma_p at each of 1.5, 2.0, 2.5, 2.75,
3.0 and estimates everything else on Sargan + micro + supply with no margin row. At the gate, the
median margin is 0.16 at sigma_p 2, 0.21 at 2.5 and 0.27 at 3, so 2.75 is close to Petrin's BLP
median margin of 0.24, and the band reports the margin each sigma_p implies.

## 28. sigma_p band at 16,384 draws: elasticities and margins reach BLP's level at sigma_p 2.5, but consumer surplus explodes with sigma_p (2026-11-17)

Run petrin_pcm_8cdfbfe173, stage sigma_p_band, Sargan + micro + supply, sigma_p fixed, all other
nonlinear parameters free, step 1 only (no point converged, so step 2 never ran).

| sigma_p | class | q | own elasticity | margin | minivan diversion to haulers | 1984 minivan CS gain, fixed / Nash ($M) |
|---|---|---|---|---|---|---|
| 1.5 | FLAT_PROGRESS | 108.3 | -16.6 | 0.11 | 0.50 | 1,309 / 984 |
| 2.0 | FLAT_PROGRESS | 115.5 | -14.3 | 0.14 | 0.56 | 2,312 / 4,133 |
| 2.5 | FLAT_PROGRESS | 138.5 | -7.7 | 0.24 | 0.45 | 9,854 / 8,895 |
| 2.75 | FLAT_PROGRESS | 154.2 | -6.5 | 0.29 | 0.45 | 17,166 / 25,903 |
| 3.0 | FLAT_PROGRESS | 148.6 | -6.5 | 0.29 | 0.50 | 31,748 / 30,458 |
| BLP micro | converged | | -4.5 | 0.24 | 0.14 | 381 / 426 |

The Nash re-solve reported a singular Jacobian at every point, so the Nash column is unreliable.

**Reading.** Substitution is the PCM's: about half of minivan diversion goes to other haulers, against
14% in BLP. At sigma_p 2.5 the median margin equals BLP's and the own elasticity is -7.7. But the CS
gain grows with sigma_p by a factor of 24 from 1.5 to 3.0. CS in dollars is utility over alpha, and
under a lognormal, E[1/alpha] = exp(sigma_p^2 / 2) grows from 3.1 to 90 over this range. The surplus is
then carried by a thin tail of nearly price-insensitive buyers whose willingness to pay is unbounded.
The fit (q) is best at sigma_p 1.5, where the margin is 0.11, below BLP's.

**Consequence.** The unbounded lognormal tail of 1/alpha makes the welfare counterfactual
uninformative at the sigma_p the margins need. The price-sensitivity distribution needs bounded
support (for example a truncated lognormal or a finite mixture), which the PCM kernel would have to
integrate in closed form per alpha segment.

## 29. Economist review round 4: bound price sensitivity from below; fixed-price CS is the headline, Nash by bounded best responses (2026-11-17)

Fourth review by the economist agent (claude-fable-5.1), with an independent Python integration of
the 1984 counterfactual.

**The CS explosion is the 1/alpha tail and nothing else.** The kernel's 1984 gain ($113.5 per
household at sigma_p 2.5) is reproduced to four digits by an independent integration, and the band
gains track E[1/a] = exp(sigma_p^2 / 2). At sigma_p 2.5, 6% of minivan buyers (alpha below 0.003)
carry 79% of the gain, gain per buyer scales as 1/alpha over five decades, and the mean is $748k per
buyer. Truncating a quantile does not fix it: cutting the lowest 10% of a still leaves $1.07B.

**Bounded distribution.** alpha_i = c_i a_i with a_i lognormal truncated below at a_lo,i = alpha_lo /
c_i; the median normalisation stays on the untruncated parent (a primitive, entry 11); the upper
tail stays unbounded (those consumers never buy). Point masses are rejected: they make shares step
functions of delta. Closed forms: F_T(e) = (Phi(z(e)) - Phi(z_lo)) / Z for e > a_lo, with Z = 1 -
Phi(z_lo); f_T, dF_T / d sigma_p and a new phi term through a_lo; truncated E[1/a] for CS; the
frontier lift uses the hull of (p, dc - a_lo p).

**Choosing alpha_lo.** Under Petrin's alpha = pi / income on the 1984 CEX (income winsorised to
[1, 150]), alpha relative to the low-group median has minimum 0.235, 1st percentile 0.415, median
0.95. Main value alpha_lo = 0.2, robustness 0.1 and 0.4. Price-range check: at alpha_lo = 0.2 the most
quality-loving consumer values the 1984 delta span ($21k in median-consumer dollars) at $105k, twice
the price range; at alpha 1e-4 it would be $210M. Expected 1984 gain after re-inversion about $0.3B
(BLP $381M).

**Elasticity law under the bound.** Own elasticity becomes about -25 / (sigma_p Z) with Z the retained
mass (0.47 to 0.80 across groups), so the margin-0.24 point moves to sigma_p of about 3 to 3.5 with no
welfare cost. The bound also closes the phi_high route of entry 24's slide.

**Nash re-solve.** The observed and fixed-price systems are well conditioned (rcond 4.6e-5). Newton
trial steps of several $1,000 (markup-map slopes up to 7 per $1) price products off every frontier,
giving exactly zero shares and a zero row. A damped zeta map with a 2% cap fails the same way; the
best-response map has slopes above one, so no fixed-point scheme converges, and with finite draws and
$10 gaps a pure-strategy equilibrium near observed prices may not exist. Headline: fixed-price CS.
Beside it, a "bounded Nash" from Gauss-Seidel best responses, each firm searching its own prices in a
+-10% box on the continuous profit function, with a deviation check and the CS range if it cycles.

**FLAT_PROGRESS was noise-converged.** Each band point descended fast for 10 to 17 iterations, then
rejections at radii 0.03 to 0.1 collapsed the trust region and 25 to 30 accepted steps each lowered
q by about the noise floor (0.0028), moving theta by under 0.02. The projected gradient and the
Gauss-Newton predicted decrease are not convergence statistics on this ridge. Fix: More scaling of
the trust region, no shrink on sub-noise rejections, CONVERGED_NOISE when the predicted decrease is
below 10x the noise floor, and a terminal probe.

## 30. Truncated alpha (alpha_lo = 0.2): the CS tail is gone, and markups fall at fixed theta (2026-11-17)

Kernel with alpha_i = c_i a_i, a_i lognormal truncated below at alpha_lo / c_i (exact shares,
derivatives and CS; derivatives match central differences to 8e-9, per-draw CS matches Simpson
quadrature to 1e-12, alpha_lo = 0 reproduces the previous kernel).

Check at the sigma_p 2.5 band point's theta, 1984 only, re-inverted with the bound:

| | alpha_lo = 0 | alpha_lo = 0.2 |
|---|---|---|
| own elasticity, median | -8.84 | -33.9 |
| Nash margin, median | 0.270 | 0.060 |
| fixed-price 1984 minivan CS gain | $9,853.7M | $161.6M |

The bound removes the low-alpha buyers that half of 1984 minivan sales relied on, so delta has to
fall to refit the shares (largest change 14.3) and own elasticity rises 3.8x, more than the 1/Z factor
alone. The gain is $960 per minivan buyer against BLP's $2.3k. Bounded Nash (Gauss-Seidel best
responses in a +-10% box, 20 sweeps, 64,341 evaluations) did not settle: gain $196M, ranging $196M to
$277M over the last 5 sweeps; no price hit the box edge; the largest profitable 2% deviation raised a
firm's profit by 0.26%.

These numbers are at a theta estimated without the bound; the band re-estimates everything else at
each sigma_p. Because the margin at sigma_p 2.5 fell to 0.06, the band grid is extended to
{2.5, 3, 3.5, 4, 4.5, 5}.

## 31. MPEC on the exact PCM: sound for demand-only problems, not for the supply-side estimator; entry 14 amended (2026-11-17)

**What entry 14 got right and wrong.** The share constraints of the exact PCM with alpha integrated in
closed form are C0 and piecewise C-infinity in delta (Jacobian jumps of w f(e)/gap when a product enters a
draw's hull), not C1 as in BLP. That is a semismooth system, and Newton-type methods on it converge
superlinearly when they use the Jacobian of the active piece, which is what the nested inversion already does
and what an NLP solver with exact first derivatives does. On Petrin 1984 at 16,384 draws the aggregate
Jacobian's rate of change along the Newton direction is 0.19 relative per unit step whether 0 or 136 draws
change frontier inside the step, and the Taylor remainder is quadratic with an event-independent constant. So objection 1 of entry 14 (no second-order convergence because the
Hessian is a train of spikes) is withdrawn for the share constraints; it stands for Hessians built by finite
differences (a kink inside the difference step gives spurious curvature w f/(gap h)) and for the supply block
(below). Objection 2 (LICQ fails at zero-share products) stands and is sharper: with log-share constraints a
zero share is log 0, IPOPT refuses to start there and treats it as an evaluation error in the line search, so
an MPEC needs a lifted feasible start and lift-and-restart restoration; the level form s/s_data - 1 lets theta
run away. Objection 3 (micro moments break the quadratic objective) is mild: the ratio micro moments and the
Sargan block are cheap smooth functions of (delta, theta) with analytic gradients from the kernel's existing
outputs.

**Prototype.** On the harness's synthetic data (3 markets, 76 products, 1,024 draws, aggregate + micro, fixed
W; the `test_recovers_true_theta` problem), IPOPT 3.14.20 (conda-forge cyipopt; the brew build is
glibc-incompatible) with log constraints, exact Jacobians and a Gauss-Newton Lagrangian Hessian converged
quadratically in 10 iterations and 25 share passes to theta_true within 5e-11, against 29 nested Gauss-Newton
evaluations (about 260 share passes) to 3e-7; from a far start it recovered theta_true where nested GN ended
FLAT_PROGRESS. With micro values shifted so the optimum has a non-zero residual, the Gauss-Newton Hessian lost
its edge (150 iterations without convergence; L-BFGS reached the nested point to 6e-3 but also did not
converge), and a finite-difference Lagrangian Hessian got worse the larger the step (35, 85 iterations, failure
at h = 1e-6, 1e-4, 1e-3): the kink effect, where it exists.

**Why it does not replace the nested outer on Petrin.** The supply block is discontinuous in (delta, theta)
at finite draws: Nash markups depend on ds/dp, a staircase with jumps of 2e-3 to 4e-2 in log mc per frontier
event on 1984, which exceed the smooth first-order change at 3% of a Newton step. Its gradient needs second
derivatives of shares that the kernel does not compute, and the active-piece slope is the wrong slope (entry
25). The nested code handles this with secants at 1e-3 and the CONVERGED_NOISE class; IPOPT has neither and
would stall at the same floor. MPEC fixes none of the current failure modes: failed trial inversions become
evaluation errors and restarts, the noise-floor stall is the supply block's, and the kernel cost (1.3 s per
derivative pass over 13 markets at 16 threads; 0.24 s for a warm inversion of 1984) is not what the band runs
spend their time on. A demand-only MPEC is kept as an optional cross-check (`--outer mpec`, refused with a
supply block); the default stays nested. For a small demand-only PCM (20-30 products, one horizontal dimension,
few quadrature nodes) MPEC is a sound and fast estimator, with the same lift-at-start requirement.

**Rule for the write-up.** Smoothness of the exact PCM in delta equals the number of continuous taste
dimensions integrated analytically: 0 gives step functions, 1 (alpha) gives C0 with a Lipschitz-like aggregate
Jacobian at large R, 2 gives C1, the population model is C1. Objects that are functions of the share Jacobian
(markups, elasticities) are one order rougher than shares and are the real nonsmoothness of the estimator.

## 32. Economist review round 5: under a lower bound the own elasticity is set by the buyers' log-alpha range, not by sigma_p; phi_high at -3 was the pooled bound; bound the unobserved factor under Petrin's income law (2026-11-17)

Fifth review by the economist agent (claude-fable-5.1), on market 1984 at the 16,384-draw band points
(sigma_p 2.5 and 3.0, alpha_lo = 0.2), with a per-draw frontier decomposition that reproduces the kernel's
shares and own-price derivatives to 1e-15, and re-inversions of 1984 under other bounds.

**No mass point.** The kernel truncates and renormalises, so no draw has mass at the bound. The high-income
group (phi_high = -3, parent median alpha 0.05) has 71% of its parent mass cut and a density at the bound 2.7x
the low group's, but buyers are not there: median buyer alpha is 2.5-4.5 x alpha_lo in every group, own
elasticities by group are -25 / -24 / -23, dropping the high group at fixed delta moves the median by 0.3, and
7% of the own-price derivative comes from window edges within 1.5 alpha_lo.

**The bounded elasticity law.** For a buying draw the frontier products share the log-alpha range
log(e_0 / a_lo,i) between the cheapest cutoff and the bound; a product's own elasticity contribution is
(p / hull gap) / (its window width in log alpha). 1984 at the band-2.5 theta: 4.2 products per buying draw,
range 2.6, sales-weighted window width 1.03, own elasticity x width = 24; re-inverted at alpha_lo = 0.05:
width 1.67, -12.3, product 20.5; at alpha_lo = 0: -6.6, margin 0.35. The frontier structure does not move with
the bound; the width does. sigma_p no longer stretches the range, so it no longer lowers elasticities: between
sigma_p 2.5 and 3.0 the estimator shrank sigma_const (3.4 -> 0.1) and sigma_fam along the scale direction,
frontiers grew from 4.2 to 4.7 products, and the median own elasticity went from -23 to -26. Entry 29's
prediction (margin 0.24 near sigma_p 3-3.5) assumed the entry 21 law, which the bound repeals.

**phi_high at -3.** Under a bound on alpha_i a group shift acts through the truncation: lowering phi_high
raises the high-income purchase rate only by piling the kept tail against alpha_lo, 0.011 per unit of phi_high
near -2 at fixed delta, so matching 0.158 needs about -2.2 and the joint fit goes to the layout bound. All three
blocks push it there (phi_high -3 -> -2.4: aggregate +2.2, micro +3.9, supply +5.3), the supply block most,
because compressed windows mean larger ds/dp, smaller markups and smoother implied costs. With the bound on
a_i the same moment is matched at phi_high = -0.63.

**Fix.** Petrin's income law with the bound on the unobserved factor: log alpha_i = log(ybar / clip(y_i, 1,
150)) + phi_mid mid_i + phi_high high_i + log a_i, a_i >= a_lo, so alpha_lo,i = a_lo c_i is proportional to
1 / income and the WTP cap is proportional to income. Petrin's pi ratios imply phi_mid 1.14, phi_high 1.80;
re-solved on the 1984 rates at the band tastes they are 1.86 and 1.13. The alpha log(y - p) form is rejected:
36% of CEX households have income below the cheapest car while the low group buys at 10%, and it adds
curvature without touching the range mechanism. At the band tastes the fix gives -20.7 / margin 0.098 / 1984
gain $566M against -23.3 / 0.083 / $398M; the pure income law without phi gives -12.7 / 0.18 / $1.4B but a
mid-income purchase rate of 0.130 against 0.079, so the two income moments pull the groups back together.

**a_lo is a calibration of the welfare tail.** Same spec, phi re-solved: a_lo 0.4 -> -29.8 / 0.068 / $266M;
0.2 -> -20.7 / 0.098 / $566M; 0.1 -> -18.2 / 0.108 / $743M; 0.05 -> -14.3 / 0.157 / $1,219M. Elasticity and
welfare move together because both are the depth of the price-sensitivity tail below the marginal buyer, and
no moment in hand pins it. Report an a_lo band (headline 0.2, welfare at BLP's size) in place of the sigma_p
band; estimate sigma_p. A median own elasticity inside (-15, -5) together with a gain near $400M is not
available from the bound. Products per frontier is the other lever (elasticity about 5.8 x products per
buying draw / range): product-specific horizontal dispersion, e.g. the capped body sigmas, is the only route
to -12 at a_lo = 0.2.

**The step-2 crawl.** After a terminal probe wins (gains 0.5-1.1 against a noise floor of 0.004, at steps of
0.01-0.1 units along coordinates or the Gauss-Newton direction), the restart kept the collapsed trust radius
(1.2e-5), so each following iteration re-probed at 84 evaluations: four restarts, 42 minutes, q 240.2 ->
238.1. Below the supply secant step (1e-3) the Gauss-Newton model predicts the slope of a piece narrower than
the kink spacing and every trial is rejected whatever the direction. Fixes: reset Delta to the winning probe
step on a restart, floor Delta at the secant step, make the probe a compass-search fallback, recompute the
More scales after a restart.

## 33. A cold PCM inversion costs thousands of share evaluations; run it on fewer draws, then warm-start (2026-11-19)

**Finding.** At the GMY round-5 step-1 theta (sigma on z log number of trims 6.7, sigma_p 3.56, a_lo 0.2) a cold
inversion needs 1,600 to 35,000 share evaluations per market, whatever the draw count, while a warm start needs 20
to 60. The three 39-year a_lo runs at nu 32 x 2,000 households (65,536 draws per market) had 33 (a_lo 0.2) or 39
(0.1, 0.3) markets cold and spent 5h50m on their first evaluation without finishing.

**Worked example (1980, 161 products, nu 4, 8,192 draws).** The vertical-model guess gives 112 of the 161 products
a share of zero at full heterogeneity, and an inside share of 0.461 against 0.315 in the data. The frontier-lift
repair runs 60 rounds (121 evaluations): each round lifts the zero-share products onto some draw's hull, which pushes
others off (31, 33, 27, ... 15 still at zero), so there is no feasible start. The heterogeneity continuation then
tries lambda 0.25 and 0.125 (121 evaluations each, the same failure) and gets a feasible start at lambda 0.0625 after
56 repair rounds, but there product 57 (data share 1.1e-3) has a model share of 9.7e-19, a log residual of -34.7,
and another product is 2,286 times its target. Newton on the log shares takes 85 damped iterations (967 evaluations)
to converge there; lambda 0.16, 0.30, 0.51, 0.82 and 1 take 214, 82, 75, 118 and 191 more: 2,013 evaluations in all.
At nu 32 the count is similar (1,638) but every evaluation costs eight times as much (244 s on 16 threads).

**Fix.** `SolveOptions::cold_draws`: run the cold path on every k-th draw, then warm-start each finer level (a nu 4
solution warm-starts nu 32 in 20 to 60 evaluations). One full 39-year first evaluation with `--cold-draws 8192` on 96
threads: 31 minutes, the slowest market (2017, 291 products) 15,711 evaluations on 8,192 draws. A delta solved at
a_lo 0.2 warm-starts a_lo 0.1 and 0.3 in 18 to 38 evaluations, so one `run --stage warm` serves all three a_lo runs.
The remaining cost is the cold path itself; a better cold start would need a different initial guess, not more draws.

## 34. GMY on all 39 years: substitution is pinned, the markup level is a_lo; post-2008 margins reach BLP only at a_lo 0.1 (2026-11-20)

(Entry 35 on the branch feat/gmy-pcm-estimation, where the Petrin a_lo band entry holds 33; numbered after master's 33 here.)

**Setup.** GMY moments (CEX and MRI micro, second-choice correlations, a Sargan aggregate block on xi with GMY's
excluded instruments, no supply block), 39 years, nu 32 x 2,000 CPS households, antithetic tastes, a_lo fixed at 0.1,
0.2 and 0.3, then step 2 and a stability rerun at 64 draws. Runs gmy_pcm_1dc37b821f, gmy_pcm_3d9429594e and
gmy_pcm_284c469079, each about 21 hours on 60 threads and up to 335 GB.

**Finding 1: step 2 never moves.** All three made 0 accepted steps in step 2 (111-113 evaluations each), so every
estimate is the step-1 point under the efficient aggregate weight. Under that weight the aggregate block is 18.9,
19.3 and 19.8 against about 217-220 under the fixed weight.

**Finding 2: substitution does not depend on a_lo.** Across the band:

| | a_lo 0.1 | a_lo 0.2 | a_lo 0.3 |
|---|---|---|---|
| q (step 2) | 1,003.8 | 1,009.5 | 1,038.3 |
| micro / aggregate / second choice | 774.8 / 18.9 / 210.1 | 796.5 / 19.3 / 193.8 | 813.4 / 19.8 / 205.1 |
| second-choice RMSE | 0.075 | 0.072 | 0.075 |
| electric second choice (data 0.383; GMY Table 4 0.132) | 0.379 | 0.383 | 0.382 |
| purchase rate Q3/Q1 (data 3.64; Table 4 2.63) | 3.15 | 3.18 | 3.18 |
| own-type diversion car / van / SUV / truck | 0.88 / 0.62 / 0.72 / 0.81 | 0.88 / 0.62 / 0.72 / 0.81 | 0.88 / 0.63 / 0.72 / 0.81 |
| median own elasticity | -9.5 | -13.6 | -14.5 |
| median Lerner, all years | 0.145 | 0.101 | 0.095 |

The aggregate block moves by 0.9 across the band, so GMY's instruments say nothing about a_lo (entry 32's window law
at work: changing a_lo re-sorts the same quantiles, the inversion absorbs the shift in delta). The 34.5 gap between
a_lo 0.1 and 0.3 is almost all micro (38.6), which prefers the lower bound.

**Finding 3: the Lerner gap to BLP is a pre-2008 gap.** Median Lerner, PCM a_lo 0.1 against GMY's Table 4 parameters
evaluated with the same code: 1980 0.175 vs 0.317, 1999 0.112 vs 0.190, 2009 0.198 vs 0.186, 2015 0.161 vs 0.179,
2018 0.179 vs 0.183. From 2008 on the two agree within 0.02. At a_lo 0.2 and 0.3 every year is about a third lower
(a_lo 0.2: 1980 0.125, 2009 0.134, 2018 0.129), so the convergence is a property of a_lo 0.1 only.

**Finding 4: 32 draws are enough for the economics, not for every spread.** Doubling to 64 draws and re-optimising
moved the median elasticity by 4.1%, 6.1% and 4.1% and the median margin by 3.5%, 4.2% and 3.7% (a_lo 0.1, 0.2, 0.3);
sigma_p by 0.03-0.05 and phi_inc by at most 0.003. sigma_truck rose by 0.22-0.36 in all three runs (about 1.6 se), so
truck tastes are under-integrated at 32 draws; phi_age (+0.1) and pi_lux_age (+0.3) move inside one se. q at the
estimate rose by 14.5, 18.0 and 23.8, so 32 draws understate q; the ranking across a_lo is unchanged. The harness's
UNSTABLE flag fires on 17-22 parameters because its threshold is 0.1 natural unit (0.01 for a spread); headline
changes are under its 10% threshold.

**Flags common to all three.** SD_XI_BLOWUP (xi kurtosis 10.2, the top 92 products carry 26% of sum xi^2),
WEAK_IDENTIFICATION (screen statistic 3e-4 to 5e-4), and sigma on z log number of trims at or near its cap of 10.
Consumer surplus (about $111k per household in 1980 at a_lo 0.1) is not a headline.

**Consequence.** Report GMY as a band in a_lo with substitution as the result. The markup level needs moments that
see it: the rival-characteristic supply instruments (the 6-year slices show an interior minimum near a_lo 0.05), to
be re-estimated with price-sensitivity parameters free at each a_lo.

## 35. GMY 39 years, extended moments: the cost shifters belong in the cost equation, a_lo is identified only by supply and income-make moments, and the literature has never estimated the tail (2026-11-22)

**Setup.** GMY moments extended with an aggregate block on xi, a markup-shifter supply block (supply_ms), within-model
demand differences (demand_fd) and within-model supply differences (supply_fd), joint CUE, ownership block dropped.
39 years, 2,000 CPS households, PCM at nu 64 with a_lo free. Comparators: a mixed logit with the PCM's utility on the
same moments and weight (pcm_matched), and a mixed logit with GMY's utility. Second choices carry GMY's weight of 30.

**Finding 1: the old supply blocks excluded true cost shifters.** supply_ms and supply_fd treated the lagged exchange
rate, steel weight and the NAFTA and Korea trade terms as excluded instruments, i.e. required implied marginal cost to
be uncorrelated with them. GMY do the opposite: the exchange rate is the excluded instrument for price on the demand
side (codes/linearDemand.do line 120) and a regressor in their post-estimation cost equation (codes/counterfactuals.m
lines 101-108, Table 12, "RXR"). Steel weight and the trade terms appear nowhere in their estimation. The exclusion was
ours. It rewards specifications whose markups are large and variable enough to absorb shifter-driven price changes:
under it the GMY-utility logit paid 174.7 on supply_fd at a median Lerner of 0.054. Fix (4e71ced, --cost-shifters
in_cost): the six shifters enter the cost equation, only markup shifters are excluded, supply_fd differences markup
shifters, demand unchanged. GMY-utility logit refit: median Lerner 0.054 -> 0.093, median own elasticity -21.7 ->
-12.5, supply_ms 147.8 -> 99.8, supply_fd 174.7 -> 161.9. The default stays the old spec so earlier runs reproduce.
The 8-year low-a_lo result leaned on supply_fd and must be re-run under in_cost before it is quoted.

**Finding 2: the pcm_matched logit was stuck, not converged.** GMY's micro moments average E[x | purchase] per
household as S_h / P_h. With the PCM's lognormal price sensitivity (sigma_p 1.75) some price coefficients reach
-8,747, every inside utility is below -745, and P_h underflows: 462 households had P_h = 0 exactly and were dropped as
never-buyers, 27 more were denormal (P_h 3.5e-323 gave 9.0 for an exact 9.0502). q jumped in steps, the
finite-difference Jacobian was wrong, least squares rejected 18 steps and stopped at q 897.1 with gradient 1.3e5. Fix
(27b7e58): the conditional mean in log space; PCM goldens bit-identical, GMY-utility logit within 2e-9 in q. Refits
fall to 806.5 (old supply spec) and 779.4 (in_cost) and are still creeping. The earlier median Lerner 0.31 and own
elasticity -4.0 for pcm_matched are void.

**Finding 3: q under one moment set is not comparable with q under another.** in_cost removes 6 restrictions from
supply_ms, swaps 6 for 8 in supply_fd and adds 6 cost coefficients (overidentifying df 85 -> 87). Across moment sets
only each J against its own df can be compared. GMY's classic fit was estimated on 1,000 households, so rescored on
2,000 it is not the minimum of the 2,000-household classic objective; it must be refit on the same draws before a
cross-fit table.

**Finding 4: the PCM's gap to the matched logit is structural.** At the old-spec CUE points (exact re-evaluation, q
matched to 1e-12) the PCM was 1,286.7 against the logit's 897.1. The gap sits in micro +206 (the PCM overshoots CEX
income-price gradients, z +10.8 on Q5 - Q1, and undershoots MRI, z -10.2: one price sensitivity sets both whether and
how much a household pays), second choices +127 (too similar to first choices on footprint, horsepower, luxury, SUV;
1999 footprint 0.816 vs 0.680) and supply_ms +90 (partly the $500 cost floor, which clips 132 logit products against
12 PCM products). The PCM is better on the aggregate block (about -29) and on age x price (142 vs 205). Compass rounds
gain 1-5 in q per 3-4 hours, so descent does not close it.

**Finding 5: GMY's weight of 30 under-weights second choices.** 500 draws of secondChoice_bootstrap.mat through the
same moment code (after the small-share drop) give a median SE of 0.0071 (0.0031-0.019) per correlation against
0.033 implied by 30, a median 22x more weight. Re-scored blocks: pcm_matched 75 -> 2,224, GMY-utility 77 -> 2,345.
Second choices are where the PCM loses, so the 30 helps it. Keep 30 as GMY's headline; report bootstrap weights as a
robustness check that favours the logit.

**Finding 6: which moments identify a_lo.** Local tests at the old-spec PCM point (a_lo 0.057): sensitivity t per
0.01 of a_lo, information gain on a_lo's SE given the existing blocks, and each block's own profile over 0.025-0.1.

| Candidate | sum t^2 | Gain on a_lo SE (efficient W) | Own profile |
|---|---|---|---|
| C4 supply: rival distances in characteristic space (in_cost) | 0.61 | +35% (0.0027 -> 0.0023) | falls with a_lo, opposite to existing supply |
| C6c: premium-make purchase share by income quintile (CEX, MRI, 10 rows) | 7.6 | +10% (+47% diagonal W) | interior minimum near 0.05 |
| C6b/C6d/C6e: luxury, European brand, log hp by income | 2.4-5.0 | +3-10% each | monotone, badly fit |
| C6a: price paid at Q5 | 6.1 | +1% (redundant) | |
| C3: exchange rate x price rank, luxury, EU | 0.04 | 0% | flat |
| C4 demand form | 0.01 | 0% | flat |
| C5: top-quintile income share x price rank | 0.02 | 0% | flat |

Demand-side instruments carry nothing: a change in a_lo is absorbed into delta by the inversion, so xi barely moves
along any exogenous instrument (entry 34 again). The existing moments pin a_lo locally through the income-price micro
rows and supply_fd; with other parameters fixed q is +105 at 0.025 and +119 at 0.1, but on 8 years the re-optimised
profile was nearly flat, so this is local only. Price-band competition measures are endogenous (rival prices are set
in the same equilibrium); only characteristic-space distances are defensible, under the usual assumption that
characteristics are set before prices. The gas-guzzler tax is out (mpg responds to it). The luxury tax is plausible
but list prices near the threshold respond to it. Data limits: CPS incomes are winsorised at $225k, two CEX College /
NoCollege level rows contradict their difference row, there is no household purchase data (true price-paid quantiles
need BLS CEX microdata; MRI is proprietary), and income reaches luxury and brand only through price sensitivity.

**Finding 7: the literature assumes the tail; nobody estimates it.** Review of the applied PCM papers from Song
(2007, CPUs) and Berry and Pakes (2007), working-paper versions where journals blocked access:
- Why shares cannot identify it: shares pin alpha only at the cutoffs between neighbours. Markups depend on the
  density at those cutoffs (Berry, Linton and Pakes 2004, eq. 19); surplus of the top product's buyers depends on
  E[1/alpha] below the lowest cutoff, which shares say nothing about (Berry and Haile 2014, section 5.2, Theorem 2).
  Only a parametric family links the two, which is why a_lo moves markups and welfare together here.
- Who names it: Song (2007): E(1/alpha | top buyers) "increases exponentially" as alpha -> 0; he reports only a
  conservative lower bound (top buyers indifferent to the next product) and checks that welfare trends survive; his
  logit welfare is 10-13x his PCM welfare. Berry and Pakes (2007): reservation prices above observed prices cannot
  be estimated nonparametrically. Bajari and Benkard (2005): drop products whose surplus bound is infinite. Tebaldi,
  Torgovitsky and Yang (2023): removing a product has "no logical upper bound" while small price changes are tight.
- Fixes in use, all assumed: lognormal with normalised location (Song; Berry and Pakes; Lucarelli et al.; the
  location drops out of dollar welfare, only the shape matters); bounded support (Bresnahan's uniform; Petrin's
  truncated chi-square on tastes, not price); a functional-form floor (Thomassen: alpha1 + alpha2 exp(-alpha3 income),
  alpha1 > 0, the closest analogue to a_lo, fixed by assumption); income-scaled sensitivity (Petrin ln(y - p); GMY
  linear in demographics with no unobserved part); a WTP cap as a multiple of price (Thomassen 2014); reporting
  devices (lower bounds, price indices by alpha group, local price changes only).
- What has pinned the level: cost-side instruments (pin the dispersion, fragile: Song's dispersion moves from 0.84
  to 1.55 with the market-size definition); outside calibration (MicroBLP's -11 from GM's market elasticity); income
  micro moments (the income gradient, not the tail within a cell); second choices (substitution, not level).
- Not covered: Nosko on CPUs, Crawford-Shcherbakov-Shum, Lee-Sovinsky, Igami et al., Miller-Yeo.

**Our contribution, ranked.**
1. Estimate the lower support point of price sensitivity (a_lo) from data and profile surplus, markups and the GMM
   objective jointly over it. This turns Song's one-sided bound into an interval and shows how markups and welfare
   co-move with a tail parameter, which no paper reports. The free-a_lo run and its profile stage deliver it.
2. Tail-targeted moments: premium-make purchase shares by income quintile (C6c, the only candidate with an interior
   minimum) and, with CEX microdata, shares of top-income buyers above price thresholds.
3. Identify the markup level from characteristic-space rival distances on the supply side (C4, -35% on a_lo's SE),
   with cost shifters correctly in the cost equation, not from demand-side instruments, which carry no information
   on a_lo in the PCM.
4. Tie the bound to a primitive: an income-based floor or a WTP cap, not a functional form.
5. Outside margin data for luxury segments to pin the density at the lowest cutoffs.
6. A welfare split: the part identified from observed cutoffs against the part extrapolated below the lowest one.
7. A stability check: the a_lo profile across market-size definitions (Song's dispersion fragility).

**Consequence.** Final moment set for the three-way comparison: in_cost supply, C4 rival distances in supply_ms, and
C6c with GMY's 33 micro rows under the bootstrap covariance. Refit the PCM (two a_lo starts) and the matched logit on
it, refit GMY's classic logit on the same 2,000 households, and score all three on both moment sets, comparing q only
within a moment set.

## 36. GMY 39 years, three-way comparison: the PCM's second choices have too little support, and low-dimensional horizontal tastes do not fix it (2026-11-27)

**Setup.** 39 years, 2,000 CPS households, joint CUE. Two moment sets: classic (GMY's 33 micro rows and second
choices, 74 moments) and C4+C6c (entry 35's final set: aggregate, supply_ms with rival distances, within-model
differences, in_cost supply, GMY's micro rows plus 10 premium-make rows, 256 moments). Estimates: the PCM on a_lo
{0.02, 0.03, 0.05, 0.08} fitted on C4+C6c; the mixed logit with the PCM's utility (matched BLP) fitted on C4+C6c;
GMY's paper BLP (their utility) fitted on classic. Results in the alo-moments worktree
(`gmy_replication/results/crossfit/`, `.kiro_local/comparison_tables.md`, `.kiro_local/spectrum_*.json`).

**Finding 1: a_lo profile on C4+C6c.** q 1,859.97 at 0.02, 1,853.71 at 0.03, 1,909.39 at 0.05, 1,995.95 at 0.08; the
minimum is 0.03. Median Lerner in 2018 falls from 0.393 to 0.213 across the grid and median own elasticity rises from
-3.22 to -5.69, so the bound still moves markups and elasticities together (entry 33).

**Finding 2: GMY's paper BLP is at its minimum.** Refit on classic at 2,000 households with ftol and xtol 1e-12,
central differences and restart rounds (a63ce17), from the old estimate and from two starts perturbed by
exp(N(0, 0.4^2)) (start q 1,297.7 and 4,090.1): all three end at q 904.987726, optimality 0.0007-0.0036. The earlier
suspicion of an early stop is withdrawn.

**Finding 3: cross-fit matrix (scored, not refit; `score`, 39d7b8e).** Only q in the same column compare.

| Estimate | Classic (df) | C4+C6c (df) |
|---|---|---|
| PCM a_lo 0.03 | 1,040.0 (47) | 1,853.7 (104) |
| Matched BLP | 773.4 (46) | 1,447.2 (103) |
| Paper BLP | 905.0 (52) | 1,895.9 (108) |

Every J p-value is below 1e-130. On C4+C6c the matched BLP beats the PCM on micro (1,227.8 vs 1,479.5) and second
choices (90.6 vs 198.0); the PCM is better on the aggregate (22.8 vs 35.3) and within-model demand blocks (5.2 vs
7.1). 267 of the PCM's micro q comes from the MRI price-by-income rows (50 for the logit). The matched BLP's 773 vs
the paper BLP's 905 on classic is a model difference (lognormal price sensitivity, 6 more parameters), not an early
stop.

**Finding 4: which comparator the PCM beats.** On 8 years the PCM (q about 1,050-1,058) beat a logit with GMY's utility
refit on the same moments (1,168.4). On 39 years it still scores about level with the paper BLP on C4+C6c (1,853.7 vs
an unrefit 1,895.9). It loses by 406 to the logit with its own utility, which could not be fit before the micro
underflow fix (27b7e58) and the PR #9 second-choice rounding fix. So the gain over GMY comes from the utility
(lognormal price sensitivity tied to income), which the PCM and the matched logit share; the loss to the matched
logit is in substitution.

**Finding 5: counterfactuals.** Paper BLP: all scenarios converge (Big 3 collusion 2018: prices +6.9%). PCM: no
scenario or baseline converges at any a_lo; the +/-10% bound on each product's best response binds (170 of 295
products in 2018 collusion at 0.03), so the reported changes (collusion 2018 +5.3% to +8.1% on affected prices) are
where the bound stops prices, not equilibria. Matched BLP: 2018 collusion converges at prices +202% with the inside
share rising 0.34 -> 0.63, 1980 collusion +1,685%; households with near-zero lognormal price sensitivity probably
leave no Nash equilibrium, but a solver bug is not ruled out. None of the PCM or matched-BLP counterfactuals is usable.

**Finding 6: make-group tastes and latent factors stall.** PR #10, from the a_lo 0.03 point, new spreads started at
0.5 (q +225 over the baseline, because a zero spread is a kink). A: random tastes on Japanese and premium makes, best
accepted q 1,976.0 after 41 hours. B: two latent factors on 7 make groups, 1,972.5. Both are still above the 1,853.7
they nest. In A, sigma_lux went 0.022 -> 0.000 while sigma_premium_make stayed near 0.46: premium_make is nested in
lux and took over its role. The runs mostly show an optimizer problem from the start and the kink; they do not show a
gain.

**Finding 7: the excess second choices are not low-rank.** GMY's product-level respondent counts (secondChoiceOut.mat,
1991, 1999, 2005, 2015). Null: Poisson log-linear with first- and second-choice product effects, |x_j - x_k| over
price and the 11 second-choice characteristics, optionally same make and same parent; spectrum of the symmetrised
Pearson residuals against a parametric bootstrap.
- Against characteristic distances alone, the top 8 eigenvalues are all 2-4x the null's 99th percentile (1991 with
  brand: 83 down to 41 against about 16-22), the spectrum declines slowly, and the top two carry 13-17% of the excess.
  The leading axes are segment loyalty (1999: sporty coupes vs minivans; family and entry-luxury sedans vs roadsters)
  and are only partly stable across years (|corr| 0.55-0.74 between adjacent years, about 0.3 for 1991 vs 2015).
- Against the PCM's own predicted second choices at a_lo 0.03 (log(C_jk + eps) added to the null, eps 1e-5 and 1e-3),
  the PCM's coefficient is 0.02-0.10, the deviance falls 1-2% and the excess 5-8%; the spectrum barely moves (1991:
  84, 73, 63, 59 -> 81, 71, 61, 57). The first eigenvalue is inside its band in 1999 and some 2015 variants; ranks 2-8
  stay 2-3x the null in every year.

**Finding 8: the PCM's second choices have too little support.** At a_lo 0.03 the PCM puts exactly zero probability
on 83-89% of first-second product pairs, and 54-61% of respondents' second choices fall on those pairs (1991 61%, 1999
58%, 2005 54%, 2015 57%). The median product has 28 possible second choices in the model against 31-45 observed. This
is the structure of the model: given one taste draw, products are ordered on one price-quality frontier, and removing j
sends that household to a product next to j on its frontier. Effective second-choice products are 4.6-5.8 in the PCM
against 11.4-16.5 in the data.

**Finding 9: segment loyalty pushes the wrong way.** Seven segments from GMY's vehicle types with their lux and sport
flags (car_lux, car_mass, car_sport, pickup, suv_lux, suv_mass, van; no Ward's segment field in the package). The
share of second choices in the first choice's segment is 0.71-0.73 in the data, 0.75-0.78 in the PCM and 0.73-0.75 in
the matched BLP. A home-segment bonus at the CUE point (exact over types, delta re-inverted) raises the second-choice
block for every segment: common lambda 198.0 -> 259.0 at 0.3; car_lux 325.5 and car_mass 290.4 at 1.0; the best case
is van at 0.3, -0.1. By the envelope theorem the extended model's optimum stays at lambda = 0. Only the second-choice
block was scored.

**Finding 10: a PCM with a free logit-error scale is the matched BLP.** It is a random-coefficients logit with the PCM's
utility under another scale normalisation, so it nests the PCM at sigma_eps = 0 and repeats the cross-fit comparison;
its small-sigma_eps end is the smoothed PCM rejected in entry 15. It is not a separate test.

**Consequence.** The fit gap between the PCM and the matched logit is in substitution, and the structured ways to add
a few horizontal dimensions (make groups, low-rank latent coordinates, home segments) do not close it, because the
missing second choices are spread across many directions and fall outside the support the PCM's frontier allows. That
is evidence that this market needs many horizontal taste dimensions, which the logit error supplies; it does not show
that the iid form is required. Open before the Grieco-Murry email: a wider Nash bound for the PCM counterfactuals and
an equilibrium-existence check for the matched BLP.

## 37. GMY 39 years: the survey measures consideration, brand exits cannot rank the models, and the PCM's identified new-product welfare is a fraction of the logits' (2026-11-27)

Analysis scripts and reports are untracked under `.kiro_local/exits/` and `.kiro_local/welfare/` in the
pcm_inversion-alo-moments worktree. Every fit was rebuilt from its stored options and reproduced its stored q to a
relative 2.7e-12 or better (PCM a_lo 0.03 1853.713865, matched BLP 1447.248151, paper BLP 904.987726 on GMY's
classic moments; PCM a_lo 0.02, 0.05, 0.08 too). All counterfactuals hold prices and the remaining products' delta
fixed.

**Finding 1: the second-choice data are stated consideration, not counterfactual choice.** GMY section 2.4 and
footnote 3 (March 2023 version): MaritzCX asks recent buyers whether they considered other models and which one they
"most seriously considered"; GMY use the first listed. Buyers who considered nothing are not in the sample, and there
is no "would not have bought" answer. The 54-61% of reported second choices that fall on PCM-zero pairs (entry 36,
finding 8) are consistent with a list of models shopped rather than only with a failure of substitution. Conlon,
Mortimer and Sarkis (draft, Nov 22 2026) fit GMY's 2015 matrix with a type model and need rank 45-80, which agrees
with our flat spectrum.

**Finding 2: brand exits do not rank the models.** Plymouth (1999-2001), Oldsmobile (2003-05), Pontiac, Saturn and
Hummer (2009-10) and Mercury (2010-11) are phase-outs, measured from the last full year. On make-by-segment lines the
three models' predicted diversion vectors differ by 0.24-0.60 of the exiting volume; ordinary year-to-year churn moves
the same lines by 2.4-18 times that volume. Pooled across events no model's slope on the revealed changes is
distinguishable from zero, and where an event slope clears its placebo range a fake exit of a comparable surviving
brand fits as well. Same-parent capture is 1.12 of the exiting volume for GM (placebo sd 0.26) and 1.90 for Ford
(sd 0.94) against 0.23-0.44 predicted and 0.37-0.49 in the survey, mostly redesigns launched in the exit year (GM 0.47
of the raw 0.62; Ford 0.90 of 1.60, the Explorer 0.71 alone), which no fixed-product model can predict. The stated
second choices explain at most 11% of revealed changes (inside the placebo band) and sit closer to the models; the
paper BLP matches them best, in sample.

**Finding 3: the outside-good margin is where the models differ most.** Exiting-brand buyers who leave the market:
PCM 0.4-4.4%, matched BLP 1.7-7.7%, paper BLP 21-44%. For 1981-2018 new models: 2.0%, 5.2% and 24.8% of buyers,
carrying 0.2%, 2.3% and 22.1% of the loss. A PCM leaver sits at its buy/no-buy cutoff and loses little; a logit leaver
loses the iid error of the product it gave up. Market growth makes the margin untestable with exits.

**Finding 4: love of variety, measured.** An exact clone gains under $0.0001 per household in the PCM at every a_lo; a
logit clone of the 2018 F-Series gains $29-80 per household. A second identical copy of each year's new models is worth
0% of their value in the PCM, 61.5% in the matched BLP truncated at a = 0.03, 64.1% as fitted and 89.3% in the paper
BLP. Removing every second model within a line costs the logits relatively more than removing a whole segment
(density/segment ratio 0.62-0.69 paper BLP, 0.33-0.49 PCM between cutoffs).

**Finding 5: PCM new-product welfare is a bound, and only its lower end is identified.** Introductions 1981-2018,
2015 dollars:

| Model | Lower bound, $B | Upper bound (fitted total), $B |
|---|---|---|
| PCM a_lo 0.02 | 37.5 | 700.7 |
| PCM a_lo 0.03 | 36.1 | 467.7 |
| PCM a_lo 0.05 | 34.3 | 276.3 |
| PCM a_lo 0.08 | 32.3 | 158.7 |
| Matched BLP, as fitted | 431.6 | 431.6 |
| Matched BLP, a truncated at 0.03 (rescored, not refit; q 1,499.1) | 171.5 | 171.5 |
| Paper BLP (classic moments) | 79.4 | 79.4 |

The lower bound is the part between observed cutoffs ($29.2-30.2B) plus a conservative tail: below a draw's lowest
cutoff c* it buys its top product K, and with b = 1/alpha the loss l = y_K b - p_K - max_{j != K}(y_j b - p_j) never
rises with alpha, so E[l] >= E[l; alpha >= c*] + P(alpha < c*) l(c*). The bound holds for any distribution of a below
c* with the same mass, which the share inversion pins; when K's frontier neighbour stays, l(c*) = 0 (Song 2007), which
is why the Model S lower bound is $0. The lower bound moves 14% across a_lo 0.02-0.08, the total 4.4 times; 93.6% of
the a_lo 0.03 total is below the lowest cutoffs. Without a floor on a the upper bound is infinite. Entry 29's CEX floor
does not map to a_lo here: GMY's c_i already uses each household's income and age, so a is residual heterogeneity
that nothing in the data bounds. The matched BLP has the same problem in its own form: its untruncated lognormal
(sigma_p 2.05) puts 58-72% of level CS on the 4.4% of households with a below 0.03. Landmark examples (entry year,
$M total, PCM lower-upper at a_lo 0.03 / matched truncated / paper BLP): minivans 1984 273-11,193 / 6,490 / 1,028;
Explorer 1990 404-1,416 / 1,131 / 803; Model S 2012 0-668 / 25 / 13.

**Finding 6: why the PCM Nash re-solve fails (entry 29, restated).** For one draw a product sells on an interval of
alpha between its frontier neighbours' cutoffs, and a price rise past the point where those cutoffs meet removes it
from that draw's frontier and makes its neighbours direct rivals. Summed over draws shares stay continuous, but ds/dp
jumps at every such point, so profits are kinked and can have several local maxima. Newton steps price products off
every frontier, the damped zeta map fails the same way, best responses have slopes above one and cycle, and the
+-10% best-response box binds for most products in every scenario at every a_lo. With finite draws a pure-strategy
equilibrium near observed prices may not exist.

**Consequence.** On welfare the PCM's identified answer for 1981-2018 introductions is $32-38B, against $79-432B for
the logits, most of whose difference is love of variety and the outside-good error. Every logit value falls inside the
PCM interval at a_lo 0.03, so the PCM does not contradict them without a floor on a; the defensible headline is the
lower bound. Open: the no-second-choice refits of the PCM and the matched BLP (branch feat/drop-second-choice-block),
a consideration-set PCM for the survey moment, and a refit of the matched BLP with the a >= 0.03 truncation.
