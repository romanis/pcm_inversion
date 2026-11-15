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
