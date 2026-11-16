# Estimation harness: requirements (owner: Roman Istomin, 2026-11-16)

This file is the contract for every agent working on the harness. Read it in full, plus
`docs/LEARNINGS.md` (entries 1, 4, 7, 11, 13, 14, 15 are essential), before writing anything.

## Goal

A general, reusable estimation harness for nested-fixed-point GMM demand models, with the exact
Pure Characteristics Model (PCM) as the first model and Petrin (2002) minivans as the proof of
concept. It must not be tailored to one paper: a new paper is a short spec (data, parameter
layout, moments, normalisations, caps), not a new estimation script.

## Hard constraints

- Exact PCM only. The smoothed PCM (logit error with scale lambda -> 0) is rejected permanently
  (LEARNINGS 15). No smoothing of shares anywhere, including as a numerical device that changes
  shares.
- Nested fixed point (inner inversion delta(theta)), not MPEC (LEARNINGS 14).
- Heavy per-consumer / per-product work stays in C++ (`market_share/`, `inversion_algorithm/`,
  `demand_model/`), exposed through pybind11 (`python_bindings/`), releasing the GIL.
- Outer optimizer, moments, weighting and diagnostics in Python (numpy/scipy), model-agnostic.
- Use the machine's 192 cores: market-level parallelism plus per-market OpenMP thread allocation
  (LEARNINGS 8-9). No OMP_NUM_THREADS=1 for production runs.
- Long runs must be checkpointed and resumable from disk after every accepted outer step.
  Processes started from an agent shell die when the session is recycled (this killed the nu=16
  run at eval 2,288). A killed run must resume, not restart.
- Python: use the anaconda interpreter the existing Petrin scripts use, not system python2.
- Scratch files under $KIROCREW_SCRATCH; results under `petrin_replication/results/harness/`.
- Commit on branch `feat/estimation-harness`. No em dashes in docs. Push to that branch only.

## Layers (agreed design, refine but do not abandon)

1. Kernel (C++): shares, d s / d delta, d s / d theta for u = delta + x t - alpha p with arbitrary
   random-coefficient draws; frontier lift; inversion as one call; NEW `max_step(delta, theta,
   direction)` returning the first step length at which a product enters or leaves some
   consumer's frontier (simplex-style ratio test).
2. Optimizer (Python, model-agnostic): minimise g(theta)' W g(theta). Problem hooks:
   `evaluate(theta) -> g, G, status`; optional `max_step(theta, dtheta)`; optional
   `restore(theta)`. Levenberg-Marquardt / trust-region Gauss-Newton, automatic scaling from G,
   weighting options: fixed W, two-step, CUE.
3. Moments + spec (Python): composable moment objects returning value plus Jacobians in delta
   and theta; framework applies the implicit-function chain rule through delta(theta). Aggregate
   IV moments and Petrin micro moments (pyblp's Petrin data carries them) are moment objects.

## Diagnostics: the output must be informative (core requirement)

The harness does not choose the econometric model, but it must detect and NAME problems. Every
run ends with a structured report (JSON + readable text) that classifies the outcome. At minimum:

- Convergence class: converged (projected gradient and step criteria, stated), stalled/flat,
  hit iteration/time limit, failed inversion. Never report "converged" on a flat stop with a
  large gradient (the CUE run "converged" at |grad| = 0.38).
- Flat objective: SVD of W^{1/2} G; small singular values reported as directions in named
  parameters.
- Units vs identification: re-test after Jacobi / natural-unit rescaling. If the small singular
  value disappears it is a units problem; if it persists it is an identification problem.
- Known degeneracies recognised by name: utility scale invariance (LEARNINGS 1, 11), parameters
  running to bounds or to infinity, taste spreads far beyond the price range (segmented corner,
  LEARNINGS 13), vertical collapse (sigma_x -> 0), sd(xi) blowing up while q falls.
- Collinearity: rank and condition of X1, of Z, of Z'X; columns of G nearly collinear; weak
  instruments (first-stage-style rank statistic on G, e.g. Cragg-Donald / Kleibergen-Paap style).
- Overidentification: Hansen J with dof.
- Economic sanity at the estimate: own-price elasticity distribution, number of nonzero
  cross-elasticities per product, diversion across groups (e.g. body types), margins, inside
  share. Flag values outside plausible ranges declared in the spec.
- Integration stability: re-evaluate at 2x draws; report change in q and in estimates.
- Local vs global: optional multi-start; report spread of endpoints.
- Standard errors only when the Jacobian is well conditioned; otherwise say why they are omitted.

## Proof of concept: Petrin

Rewrite the Petrin PCM as a spec on the harness, with aggregate moments plus Petrin's micro
moments, fixed-W two-step from an interior point, sigma on body dummies capped at the price
range as a stated normalisation. First check: sigma_p ray profile with micro moments must no
longer be monotone (LEARNINGS 14). Then the full estimation, then post-estimation (elasticities,
margins, 1984 no-minivan counterfactual) against the BLP columns already in
`petrin_replication/results/blp_*_summary.json`.

## Analysis

Only when an objective genuinely converged (per the diagnostic report) is the result analysed
economically and econometrically, and alternative specifications proposed and run, aiming at a
specification as close to BLP as possible with stable normalisations.

## Repository hygiene (owner instruction, 2026-11-16)

Each replication folder tracks exactly two scripts: `estimate_blp.py` and `estimate_pcm.py`.
Each produces all meaningful analysis for its model (estimation, diagnostics report,
postestimation, counterfactuals) from the harness. Logs, results, probe and check scripts stay
local and untracked; `.gitignore` enforces this. Reusable code lives in the harness package,
not in replication folders.
