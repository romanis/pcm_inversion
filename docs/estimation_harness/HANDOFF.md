# Wave A handoff notes (verbatim from package reports, 2026-11-16)

Integration check: build-rel2 builds clean; ctest 82/82; pytest 87 passed, 3 skipped (WP7, WP8, WP9a placeholders).

## WP1

Others are active. Let me check how WP3 handles the `Market` type in its binding.WP1 is finished and committed as `4357d95` on `feat/estimation-harness`. The push went through on the first try (`3295272..4357d95`). All WP1 acceptance tests pass. There are two deviations from DESIGN: one extra test file outside WP1's ownership list, and a variable finite-difference step in kernel test 2.

**Files changed**
- `/local/home/istomin/pcm_inversion/demand_model/pcm_kernel.hpp`
- `/local/home/istomin/pcm_inversion/demand_model/pcm_kernel.cpp`
- `/local/home/istomin/pcm_inversion/demand_model/pcm_kernel_test.cpp`
- `/local/home/istomin/pcm_inversion/python_bindings/bind_solve.cpp`
- `/local/home/istomin/pcm_inversion/demand_model/pcm_demand.hpp` and `/local/home/istomin/pcm_inversion/demand_model/pcm_demand.cpp`: only the move of `frontier_lift` and `cond_vertical` into `pcm_demand::detail`, with no behaviour change.
- `/local/home/istomin/pcm_inversion/nfp_harness/tests/test_kernel_solve.py`: new file (deviation 1).

I built in `$KIROCREW_SCRATCH/wp1-build` with googletest from `build-rel2/_deps`. The `pcm_kernel_py` target writes its `.so` into the shared, gitignored `nfp_harness/_native/`, as its CMake target is set up to do.

**Acceptance outputs**
```
[ RUN      ] PcmKernel.NoHeterogeneityReturnsInitialGuess            [ OK ]
[ RUN      ] PcmKernel.DdeltaDthetaMatchesFiniteDifferences
  parameter 0: h 1e-05, relative error 1.152e-08
  parameter 1: h 1e-07, relative error 5.239e-10
  parameter 2: h 1e-06, relative error 1.206e-10
  parameter 3: h 1e-05, relative error 1.537e-08
  parameter 4: h 1e-07, relative error 3.850e-09
  parameter 5: h 1e-05, relative error 7.555e-11
  parameter 6: h 1e-05, relative error 1.860e-11
  parameter 7: h 1e-05, relative error 1.846e-11
  ddelta_dtheta max relative error vs FD: 1.537e-08                 [ OK ]
[ RUN      ] PcmKernel.MicroPartsMatchSharesBruteForceAndFiniteDifferences
  micro_d_delta rel error 4.588e-11, micro_d_theta rel error (worst column) 1.192e-08   [ OK ]
[ RUN      ] PcmKernel.BitIdenticalAcrossThreadCounts                [ OK ]
[ RUN      ] PcmKernel.WarmStageLiftsProductBelowEveryFrontier       [ OK ]
[ RUN      ] PcmKernel.NearDuplicateProductsNeverGiveSilentNaN
  gap 1e-03: ok, rcond 2.486e-03
  gap 1e-06: ok, rcond 3.402e-06
  gap 1e-09: ok, rcond 3.403e-09                                     [ OK ]
[ RUN      ] PcmKernel.ThetaPathStageSolvesWhenWarmStageFails        [ OK ]
[ RUN      ] PcmKernel.InvalidInputIsAStatusNotAnException           [ OK ]
[  PASSED  ] 8 tests.

ctest (whole tree, including pcm_demand_test, events, welfare): 100% tests passed, 0 tests failed out of 82

pytest nfp_harness/tests/test_kernel_solve.py:
serial 0.731s, two threads 0.419s, ratio 0.57
2 passed
```
The GIL ratio is 0.57 rather than 0.5 because the two test markets take 0.42 s and 0.30 s alone, so the threaded run lasts as long as the slower one. I also timed a 200-product, 16,384-draw market at 16 threads. The kernel's draw pass took 0.068 s against 0.137 s for `pcm_demand::shares`, with identical `ds_ddelta` and a `ds_dtheta` difference of 2e-16.

**Deviations from DESIGN**
1. **Extra test file.** The GIL thread test lives in the new `nfp_harness/tests/test_kernel_solve.py`. WP1 owns no Python test file, and no other package owns this one, so it cannot collide with anyone's work.
2. **Kernel test 2 uses a variable step.** A fixed h = 1e-5 gives errors of 2-4% on parameters 1 and 4 (the taste parameters on the 0-10 characteristic). This is an exact PCM kink: some draw's frontier changes inside the ±h interval. The test now starts at h = 1e-5 and shrinks it by 10 until no draw's frontier differs between theta - h, theta and theta + h. That brought every column under 1e-8.
3. **Extra pieces.**
   - Two kernel tests beyond 1-6: the theta-path stage and invalid input.
   - A public `evaluate_market`, which is the draw pass at a fixed delta with no inversion, in C++ and Python. The micro finite-difference test uses it, and WP7 can too.
   - `status_name()`, which turns a status into its lower-case string.
4. **Faster draw pass.** The pass works only on each draw's frontier products, not the full N x N matrix. The results equal `pcm_demand::shares` (ds/ddelta bit-identical, ds/dtheta to 2e-16).

**What WP7 (PCM adapter) needs to know**
- **Call signature:** `solve_market(s_data, market, theta, phi, sigma_p, delta_warm=None, theta_warm=None, micro=None, *, tol, warm_evals, cold_evals, path_evals, path_seconds, n_threads, derivatives, price_derivatives, rcond_min)`.
- **`theta_warm`** is the stacked `(P,)` vector `[theta, phi, sigma_p]`, the same column order as `ds_dtheta`.
- **`market`** is a `pcm_kernel.Market(x, p, weights, taste_basis, taste_col, scale_basis)`, or any dict or object with those fields. `bind_solve` registers `Market` (module-local), and the WP2 and WP3 bindings rely on that registration.
- **`micro`** is `None` or a dict `{"H": (R, L), "U": (N, Pm), "h_of_part": (Pm,)}`. U already holds v - v0. The parts are raw sums, `sum_i w_i h(i) u' s_i`. The outside value v0 H_t and the division by Den are left to Python.
- **Status strings:** `"ok"`, `"not_converged"`, `"infeasible_start"`, `"singular_ds_ddelta"`, `"invalid_input"`. The adapter must map them to `EvalStatus`. Bad input never raises; it returns `invalid_input` with the reason in `stages`.
- **Derivative fields** have shape `(0, 0)` unless the inversion converged and they were requested. With `singular_ds_ddelta`, `ddelta_dtheta` is NaN but the other derivatives are filled. `s` and `micro` are always computed at the returned delta, except for `invalid_input`.
- **Threads:** `n_threads` sets the OpenMP team size for that call's thread only and is restored afterwards. Tests run with the default 192-thread team are much slower (1.8 s against 0.1 s), so production needs `OMP_WAIT_POLICY=PASSIVE` and explicit `n_threads` (DESIGN 1.3).
- **`rcond`** is the smaller of the LU pivot ratio and Eigen's 1-norm estimate.

The `chorus` MCP server failed to start in this session; nothing here needed it.

## WP2

**Deviations from DESIGN:**
- **Two extra fields on `StepEvents`:** `buyer_mass` (B_j per product), and `product_draw = -1` when a draw has no event.
- **New function `affected_mass`.** The 4.4 guard needs the realised max_j A_j, and per-draw hashes alone cannot give it. Signature: `affected_mass(delta, market, theta, phi, sigma_p, changed, n_threads)`, with `changed = sig_accepted != sig_trial` and F_i taken at the point you pass. It returns A_j for each product.
- **Products with no buyer are skipped in `tau_vanish_lb`.** If no product has a buyer, it is tau_max. Only draws with a real event (kind != 0) count in `mass_profile`.
- **Ties within one draw:** a leaving vertex (C1, C2) is reported before an entering product (C3, C4). Within each group the cheaper product comes first. DESIGN tests 1 and 2 are degenerate cases: in test 1, A is collinear with the origin and C, so A is not on the frontier; in test 2, four points become collinear at tau = 0.5. The expected answers still come out because of this rule.
- **`n_threads` uses a `num_threads` clause, so the caller's OpenMP setting is unchanged.** `n_threads <= 0` means the OpenMP default.
- **Binding defaults:** `tau_max=1.0` and `n_threads=1`.

**For WP7 (PCM adapter):**
- **Passing the market:** pass `market` as a dict or any object with `x`, `p`, `weights`, `taste_basis`, `taste_col`, and optionally `scale_basis`. If some module registers `pcm_demand::Market` as a pybind type, an instance of it is also accepted. I did not register the type, so it cannot clash with WP1.
- **`dtheta`** must have M+Q+1 entries in [taste, scale, sigma_p] order. Only the taste part matters, and `sigma_p` is not used.
- **Prices** must be positive and strictly increasing, the same requirement as `shares`. Invalid input raises `ValueError`.
- **Expect a large share of draws to have early events.** On a random 210 x 16,000 market with N(0,1) directions, 36% of draws had an event by tau = 0.05, and max_j A_j was already 1.0. The cap will usually come from `q_mass` (DESIGN 8, risk 3).
- **Predictions are exact when delta is held fixed.** In that case the predicted share of draws with an event matched the realised signature changes exactly (0.362875 for both).

**For WP5 (optimizer):** `mass_profile` has 64 rows (`pcm_events::MASS_GRID`), with tau_g = tau_max g / 63, and it is non-decreasing. Interpolate it to find the tau at which it reaches `q_mass`. `tau_first` is close to 0 at realistic draw counts and should not be used as a cap.

**MCP:** the `chorus` server failed to start. It was not needed.




## WP3

WP3 is done and all its acceptance tests pass, with one deviation. The kernel does **not** match `consumer_surplus` in `pcm_postestimation.py` to 1e-10, because that reference function is wrong. The kernel matches the reference with its bug corrected to 1.4e-14.

**Commit:** `1a3b7bc` on `feat/estimation-harness`. The push succeeded (`cb26a76..1a3b7bc`).

**Files changed (committed, only these four were staged):**
- `/local/home/istomin/pcm_inversion/demand_model/pcm_welfare.hpp`
- `/local/home/istomin/pcm_inversion/demand_model/pcm_welfare.cpp`
- `/local/home/istomin/pcm_inversion/demand_model/pcm_welfare_test.cpp`
- `/local/home/istomin/pcm_inversion/python_bindings/bind_welfare.cpp`

**The bug in the reference.** The function in `/local/home/istomin/pcm_inversion/petrin_replication/pcm_postestimation.py` (lines 45-90) gives each frontier product the wrong range of price sensitivity a. Its docstring has the right formula, `(e_{k+1}, e_k)`. A consumer buys the cheapest product rather than nothing when a < q/p. The code instead sets the cheapest product's range to (q/p, ∞), which is the consumers who prefer nothing.

A worked example from DESIGN test 1: one product with q = 2, p = 1, sigma_p = 0.5.
- The correct CS is 2·E(1/a; a < 2) − 1·P(a < 2) = 1.28198. Simpson quadrature of max(q/a − p, 0) gives the same value.
- The reference returns −0.01568. That is the surplus of the consumers who do not buy, and it is negative.

**Acceptance outputs.** The C++ gtest suite passes (`ctest -R pcm_welfare`: 100% of 7 tests passed). It covers:
- DESIGN 3.6 welfare 1, the closed form above, also checked against quadrature to 1e-11.
- Welfare 2, scale invariance for kappa in {0.37, 3.7, 25}, to 1e-12.
- Welfare 3, in two parts: dropping a dominated product leaves CS unchanged to 1e-12, and with no products CS is 0.
- Three extra tests:
  - Roy's identity, dCS/dp_j = −s_j against `pcm_demand::shares`, to 1e-7.
  - Results bit-identical with 1 and 8 threads.
  - Bad input throws.

The Python check was run on a random 30-product, 500-draw market:

```
kernel total                    42.717565055738156
pcm_postestimation (as is)      3.897513804340491   |diff| 3.882e+01
pcm_postestimation (fixed)      42.717565055738170   |diff| 1.421e-14
per-draw max |kernel - fixed|   4.263e-14
per-draw max |kernel - quad|    4.527e-07   (independent quadrature)
per-draw max |as-is - quad|     1.729e+02
min per-draw CS: kernel 3.1679, as-is reference -12.3577
python_pcm_demand.Market input: |diff| 0.0 per-draw equal: True
Roy: max |dCS/dp + s| = 1.720e-08 (shares from python_pcm_demand)
ValueError on unsorted prices: pcm_welfare::consumer_surplus: prices must be strictly increasing
ALL WP3 PYTHON ACCEPTANCE CHECKS PASSED
```

**Other deviations from DESIGN:**
- **Bit-identical results.** Instead of 64-draw block sums, each draw's CS is computed in parallel and the weighted total is summed in draw order. This gives the same guarantee.
- **Python module tested in scratch.** I did not build the `pcm_kernel_py` target, because it writes into the shared `nfp_harness/_native`. The other agents' `bind_solve` was still being edited, and building would race their builds. I compiled `bind_welfare.cpp` and `pcm_welfare.cpp` into a standalone module under `$KIROCREW_SCRATCH/wp3`. In my own cmake tree, `bind_welfare.cpp.o` for the real target compiles with no warnings.
- **Build setup.** My build tree is `$KIROCREW_SCRATCH/wp3-build`. It reuses googletest from `build-rel2/_deps` and needed `-DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPYTHON_EXECUTABLE=...` to configure.

**What WP8 (postestimation) must know:**
- **Old results are invalid.** Every old PCM CS number comes from the buggy function and is wrong, including `petrin_replication/results/pcm_vs_blp_body_nu16.csv`, which reports a minivan gain of −$1.25M. Do not use them as a numeric fixture for CS. Elasticities and margins come from other code and are not affected.
- **Call signature.** It is `consumer_surplus(delta, market, theta, phi, sigma_p, n_threads=1)` and returns `{"total", "per_draw"}`.
  - `total` is the weighted sum of w_i·CS_i, in $1000 per household.
  - `per_draw` is each draw's CS_i, not multiplied by its weight, ready for tables by alpha quintile and family size.
- **The `market` argument.** It can be any registered `pcm_demand::Market` (from `python_pcm_demand` or from WP1's binding if it registers one). It can also be a dict or object with `x, p, weights, taste_basis, taste_col, scale_basis`.
- **Input rules.** The GIL is released during the call. Prices must be positive and strictly increasing, so re-sort counterfactual prices before calling. Invalid input raises `ValueError`.

**Unavailable MCP server:** `chorus` failed to start. It was not needed.




## WP4

**Deviations from DESIGN and why:**
- `LinearIV` also refuses a rank-deficient Z. Without that check, the default weight (Z'Z/n)^{-1} would be a near-singular inverse that returns garbage instead of an error.
- The moment count uses rank(Z) + micro rows, not the column count, so a duplicated instrument cannot pass the check.
- `NFPGMMProblem` takes an optional `W`. Without one it uses the aggregate block of the linear fit and the identity on micro rows; `initial_weighting(theta0)` gives pyblp's W1.
- `set_weighting` refuses a W that couples aggregate and micro rows, because concentrating beta with the aggregate block alone is only exact for block-diagonal W (DESIGN 2.4).
- The problem requires exactly one aggregate IV block with the same Z as `LinearIV`, and one micro moment object per dataset, since covariances across moment objects are taken as zero.
- `max_step` exists on the problem instance only when the model has it, so the optimizer's `hasattr` check reflects what the model can do.
- Extra methods beyond DESIGN: `full_jacobian(ev)` gives the (beta, theta) Jacobian for standard errors, and `cue_fixed_covariance(ev)`, `micro_covariance(ev)` and `objective(g)` are helpers. `CUE(hold_micro_block=False)` raises `NotImplementedError` (DESIGN 8.9).
- `sobol_normal_draws` refuses an n that is not a power of two unless `require_power_of_two=False` (owner decision 6). `cross_agents(..., nu_per_agent=None)` splits R draws into consecutive per-agent blocks of 4 or 5 draws (4,096 draws over 1,000 agents), each with weight w_a / count_a, so every agent keeps its total weight.
- The micro moment sign follows DESIGN, g = model minus data. pyblp uses data minus model; this makes no difference to q or S.

**What WP7 (PCM adapter) must know:**
1. **Micro output contract** (top of `moments.py`): `sol.outputs["micro_parts"][part.name]` is a list with one dict per market, in `model.markets` order. Each dict has:
   - `sum`: sum over draws of w_i h(i) times the sum over products of u_j s_ij, with u = v - v0;
   - `d_delta`: (J_t,), the partial in delta;
   - `d_theta`: (P,), the partial in theta;
   - `h_sum`: the sum of w_i h(i);
   - `w_sum`: the sum of w_i.
   h is the product of `agent_columns` and v the product of `product_columns`. Product parts arrive by name, `"a*b"`, with the columns concatenated and v0 = v1_0 v2_0. The kernel just multiplies the columns, so age times age equals age for the binary age column.
2. **D:** `sol.D` is a per-market (J_t, P) list in `model.markets` order. The problem maps it to rows through `MarketIndex.rows`.
3. **Covariance re-solve:** `problem.covariance(ev)` re-solves the model once (no Jacobian) with the product parts if the evaluation did not request them. That solve overwrites the trial state at the same theta, so the next `evaluate` must not depend on that trial state surviving.
4. **Evaluation aux:** `ev.aux["solution"]` holds the InnerSolution, which `max_step` and covariance use. `ev.aux` also carries `delta`, `xi`, `beta`, `D` (n x P), `sd_xi`, `q` and `q_blocks`.
5. **Optional hooks:** `state_dict`, `load_state_dict`, `commit` and `restore` on the model are called if present.


## WP5

**Deviations from DESIGN and why:**
- **Ridge only when ill-conditioned.** `mu = 1e-10 max(d)^2` is applied only when the Jacobi-scaled A has `s_min/s_max < 1e-2`. Applied on every step, it biased each step by about 1e-10 × cond², which made test 1's 1e-10 tolerance fail.
- **CUE model.** The Gauss-Newton residual is shifted inside range(A), so the step model has the exact CUE gradient while the Hessian stays `2n G'S^{-1}G`.
- **`max_step` cap.** `tau_cap` also takes the min with the model's own `StepLimit.tau_cap`, so WP7 should return 1 there unless it has a hard cap. "Mass reaches q_mass" means the first grid point where the mass is at least q_mass. q_mass has a floor of 1e-3. Realised mass is read from `ev.aux["realised_mass"]`.
- **Additions to my own files.** `OptimizerOptions` gained `stage`, `theta0`, `record_cue`, `q_mass_min`, `two_step_gate` and `two_step_require_converged`. Step 2 of a two-step run starts only if step 1 converged. `classify` gained a `Delta` keyword and returns None when no stopping rule fires.
- **Checkpoint files.** The checkpoint is a token-matched pair: `checkpoint.npz` and `checkpoint.json`, plus `checkpoint.prev.npz` so a crash between the two file replacements still loads. Finished stages are stored in `stages/<stage>.npz` and `.json`. `--new-run` renames the old directory to `<dir>.old-<UTC timestamp>` and never deletes it.
- **`detach`.** It uses `start_new_session=True`, the `setsid(2)` system call, rather than the `setsid` binary. The binary forks when its caller is already a session leader, and then the pid it reports is wrong.

**Contract WP4/WP7 must provide** (`problem.py` is WP4's; WP7 supplies what it delegates to the model; all duck-typed):
- **Units:** `problem.units` as an array if present, else `problem.layout.units(problem.data)`, with None passed when `data` is absent. Start, bounds, `cap` and `free` come from `layout.specs`.
- **n:** `problem.n`, else `problem.n_products`, else `ev.aux["n"]`, else 1.
- **Weighting:** a W array, the string `"cue"`, or an object with `mode` in {fixed, two_step, cue} or a class named `Fixed`/`TwoStep`/`CUE`. Fixed needs `.W`, TwoStep needs `.W1`, and CUE may have `.W1` (used only for the q_fixed column).
- **CUE terms:** `cue_terms` must return `(V: C×k cluster sums, dV: C×k×P)` with `S = V'V/n`.
- **Optional hooks:** a hook that returns None or raises `NotImplementedError` is treated as absent. `ev.aux["xi"]` is optional and feeds sd(xi).

**CLI and resume contract for WP9:**
```python
rd = RunDirectory(base, spec_name, spec_hash, hash_options(result_changing_cli_opts),
                  new_run=args.new_run, options=opts)   # raises SpecMismatchError on hash change
if args.detach:
    runtime.detach(rd.path / "log.txt")   # parent prints pid and exits 0; no-op in the child
runtime.setup_logging(rd.path / "log.txt"); runtime.check_environment()
res = minimize(problem, weighting, OptimizerOptions(stage="step1"), checkpoint=rd)
res2 = minimize(problem, Fixed(W2), OptimizerOptions(stage="step2", theta0=res.theta), checkpoint=rd)
# `status` subcommand: runtime.status(rd.path)
```
- **Resuming:** relaunch the identical command. `minimize` resumes the named stage from the checkpoint, or returns a finished stage's stored result after one evaluation.
- **Stage names:** use a different name for each stage. A TwoStep weighting runs `<stage>.step1` and `<stage>.step2` itself.
- **Problem state:** the problem must implement `state_dict()` and `load_state_dict()` with nested dicts, lists, scalars and numpy arrays only (no pickle). This is what makes the warm deltas survive a restart.


## WP6

## Deviations from DESIGN and why
1. **Order condition added.** The Z'X1 rank check alone does not catch the real Petrin case. There, Z'X1 is 22 x 13 with full rank 13, but 22 moments face 31 unknowns. I added `order_condition_block`, which fails with UNDER_IDENTIFIED and the counts, and a flag, X1_NOT_IN_Z, for X1 columns outside span(Z). Both appear in the report.
2. **Weak-identification column scale.** DESIGN says "sd_k the column scale" without defining it. `run_diagnostics` scales by natural units, so the statistic is the precision of the worst one-natural-unit move. Without units it falls back to the column root mean square.
3. **UNITS verdict for in-between ratios.** A direction with a natural-units ratio between 1e-6 and 1e-4 that is fine in Jacobi units is also labelled UNITS. DESIGN only defines UNITS below 1e-6.
4. **Fewer than 10% gain in RUN_TO_INFINITY.** It is measured against (1 + q) at the start of the window, the scale of the DESIGN 4.6 stopping rules. Otherwise a q near 0 that keeps halving (DESIGN 4.8 test 4) would never count as little gain. "Above 5 times its start" is read as |theta| > 5·max(|start|, unit).
5. **Standardisation and correlations.** Columns are scaled to unit norm without centring, so the constant in X1 survives. Canonical correlations and Jacobian-column correlations are uncentred.
6. **Integration status key.** The integration block keeps DESIGN's `"status": "STABLE"/"UNSTABLE"` and stores the block status as `block_status`.
7. **Cap shown as NOTE.** NORMALISATION_CAP_ACTIVE is rendered as `NOTE:`, not `FLAG:`, because DESIGN calls it "not a failure".
8. **Non-finite JSON values.** NaN is written as null and infinity as the strings "inf"/"-inf", so the files are strict JSON.
9. **Extra keyword arguments.** Some stub signatures gained optional keyword arguments (`units`, `binary_columns`, `x1_names`, `converged`, `names`, `headlines`, `thresholds`). Calls with the original positional arguments still work.
10. **`DiagnosticInputs` and `build_report`.** I added these to `diagnostics/__init__.py`. `run_diagnostics` fills the inputs from options, then spec attributes, then `evaluation.aux`.

Things the later packages need to know:
- **Missing WP5/WP8 code:** if `convergence_block` (WP5) or `economics_block` (WP8) raises NotImplementedError, the report falls back to the DESIGN 5.1 message and an economics block marked not run.
- **Gradient for RUN_TO_BOUND:** it defaults to 2n·G'Wg, which is not the true CUE gradient. WP9 should pass the exact gradient through options.
- **Linear-coefficient SEs:** these need `G_full` in `aux` or options. Without it, SEs cover theta only, and the report says so.


## WP7

**The edit outside my files.** The optimizer reads `ev.aux["realised_mass"]`, but `NFPGMMProblem.evaluate` never copied anything from the model's outputs into `aux`. The model has no other way to deliver the realised mass. After `aux = {...}` I added: if `"realised_mass"` is in `sol.outputs`, copy it into `aux`.

**Acceptance outputs**

The synthetic data are generated by the model. Products sit at random angles on the unit circle in (x1, x2), xi is made orthogonal to Z = [X1, p, p², rival sum of x1, x1·x2, x1·p] in the sample, and shares are the kernel's shares at the true theta. Micro moment values are the model's predictions at the true theta, so g(true theta) = 0.

My first designs (one taste dimension on a characteristic tied to price) left many products with zero share. In the PCM a product that is on no consumer's frontier gets nothing. The generator therefore drops products with share below 1e-4 and re-orthogonalises xi until none is left. For the larger case this gives 3 markets of 23, 25 and 28 products (from 44 candidates each), with 1,024 draws per market.

- **DESIGN case (2 markets, 8 products, 256 draws, micro moments).** G matches central finite differences of g. The worst relative error is 8.7e-09 against the 1e-5 target. Each finite-difference step is shrunk until no draw's frontier differs between theta - h, theta and theta + h. g, G and delta are bit-identical with 1 and 16 threads.
- **(a) 3 markets of about 25 products, 1,024 draws.** The worst relative error between G and finite differences is 7.2e-08 (sigma_x1).
- **(b) Recovery.** The run used the weighting from `initial_weighting`, the `max_step` hook was active, and the class is **CONVERGED**.

| parameter | true | start | estimate |
|---|---|---|---|
| sigma_x1 | 6.0 | 7.8 | 5.99999971 |
| sigma_x2 | 5.0 | 3.5 | 4.99999968 |
| pi_x1_demo | 0.5 | 0.7 | 0.49999999 |
| phi_inc | 0.8 | 0.6 | 0.80000002 |
| sigma_p | 1.0 | 1.3 | 1.00000005 |

  q = 1.1e-14 after 28 accepted steps and 29 evaluations; 23 of the steps were capped by `max_step`. The projected gradient is 1.6e-06. The linear coefficients are recovered to 1e-5.
- **Petrin at nu_per_agent = 1 (1,024 draws per market), DESIGN spec 6.2 start point.** All 13 markets converge. The largest share residual is 8.3e-10 against a tolerance of 1e-9. Deltas match `python_pcm_demand.invert` exactly (difference 0.0). A cold solve takes 30 s; the slowest market, 1988, takes 30 s on 18 threads. A warm solve after a 1% move takes 1.3 s.
- **(c) Full suite.** `/home/istomin/anaconda3/bin/python -m pytest -q nfp_harness/tests` gives 99 passed, 2 skipped (the WP8 and WP9a placeholders) in 96 s. The Petrin test takes about 80 s of that. `ruff check` is clean on the three files, and mypy is clean on `pcm.py`.

**What WP8 and WP9 need to know about the adapter**
- **Constructor.** `PCMModel(spec, data, layout, nu_per_agent, seed, total_threads=None)`.
  - `spec` is a `PCMSpec(characteristics, ...)` or anything carrying its fields. It holds the column names (`market_ids`, `prices`, `shares`, agent `weights`), the solve settings (`tol`, `warm_evals`, `cold_evals`, `path_evals`, `path_seconds`, `rcond_min`), `draws_per_market`, `track_realised_mass` and `max_step_tau`.
  - `data` is a mapping or object with `products` and an optional `agents`, as DataFrames or dicts of arrays, with positional rows.
  - A characteristic named `"const"` that the product table lacks is a column of ones.
  - The constructor resolves the layout's automatic natural units. The optimizer relies on this, because `NFPGMMProblem` has no `data` attribute.
- **Parameters are read from the layout**, and the layout order is free.
  - A `"taste"` parameter needs a `column` among the characteristics. Its `basis` is an agent column for a demographic interaction, any other name for a standard normal draw dimension (two parameters with the same name share that draw), or None for 1.
  - A `"scale"` parameter's `basis` must be an agent column.
  - There must be exactly one `"sigma_p"`.
  - Derived demographics such as `logfs_fv` and `low_inc` must already be columns of the agent table.
  - D and the micro `d_theta` come back in layout order. `kernel_params(theta)` and `to_kernel_order` / `from_kernel_order` convert to and from the kernel order.
- **Draws.** R_t = 2^ceil(log2(A_t · nu)): 4, 16 and 32 per agent give 4,096, 16,384 and 32,768 draws for Petrin's 1,000 agents. Sobol seed = seed + market position (in sorted id order), crossed with agents in blocks so each agent keeps its weight. `market_draws(t)` returns `agent_of_draw` and `agent_rows`, for CS tables by alpha quintile and family size.
- **Prices must be strictly increasing within a market after sorting**, or the constructor raises. Call `data.break_price_ties` first. `markets[t].rows` lists product rows in price order, and every per-market array follows it. `kernel_market(t)` returns the `pcm_kernel.Market` that WP8's `consumer_surplus` and the counterfactuals need.
- **`sol.outputs`** holds `micro_parts`, `theta`, `shares` (n,) and `realised_mass`. It also holds `ds_dp`, one (J_t, J_t) matrix per market in price order, when `KernelRequests(price_derivatives=True)` is passed.
- **Statuses.** Kernel `not_converged` and `infeasible_start` become INVERSION_FAILED, `singular_ds_ddelta` becomes SINGULAR_INNER_JACOBIAN, and `invalid_input` becomes OUT_OF_DOMAIN. Non-finite D or delta is NUMERICAL. `per_market` entries carry `kernel_status`, `stages`, `rcond`, `threads` and `seconds`.
- **Warm starts.**
  - `solve` warm-starts only from the accepted state. Before the first `commit` every solve is cold, which takes about 30 s for Petrin even at nu = 1.
  - `commit(theta)` promotes only trial states solved at exactly that theta.
  - `state_dict` and `load_state_dict` carry per-market delta and theta plus the thread weights; loading refuses a state from other markets.
  - `accepted_delta()` returns the accepted delta.
- **`max_step`** uses `tau_max = 2` and returns `tau_cap = 1`. With WP5's `0.5 · tau_vanish_lb` rule, `tau_max = 1` would have capped every step at half its length.
- **Threads.** Each market gets `allocate_threads` of `total_threads`: weights start at J²R and then follow the measured CPU time. Markets run in a thread pool. Call `model.close()` when done, or use the model as a context manager.

The `chorus` MCP server failed to start in this session; nothing needed it. Scratch tuning scripts are in `$KIROCREW_SCRATCH/wp7` and are untracked.


## WP9a

- **Concentrated sigma_p profile on the 2-market data:** q_fixed is 0.051, 1.5e-11, 0.035 at sigma_p 0.8, 1.0, 1.25, so INTERIOR_MINIMUM at the truth. The point at 1.25 ended STALLED rather than CONVERGED. The class is shown in the table.
- **Planted fixed-W scale-artifact toy** (delta = exp(-sigma_p) times a base vector): q_fixed falls by a factor of e at each 0.5 step, giving MONOTONE_DECREASING with every step falling. q_cue is FLAT. The gate fails with "Profile check failed: q is still monotone in sigma_p."
- **Killed run resumes:** a child process (0.3 s per evaluation) was killed with SIGKILL after at least 3 points were saved. The rerun printed `RESUMED n SOLVES 8-n`, so only the remaining points were computed, and its result equals the uninterrupted run bit for bit.
- **Petrin at nu_per_agent = 1 (DESIGN WP9a):** a 2 x 2 grid (lambda in {0.5, 1}, sigma_p in {1.0, 1.5}) with Z = [X1, demand_instruments], no micro moments, ran on 4 workers with 48 threads each in 96 s and wrote `profile.csv`.
- **Full suite:**
  - In the shared tree it gives 2 failed, 130 passed. Both failures are in `test_pcm_post.py`, which holds WP8's uncommitted, in-progress work, not mine.
  - On a clean `git archive HEAD` copy plus my two files, `/home/istomin/anaconda3/bin/python -m pytest -q nfp_harness/tests` gives **118 passed, 1 skipped** (the WP8 placeholder).
  - ruff and mypy are clean on `profile.py`.

**One Petrin result for the owner.** In that 2 x 2 grid, with the full instrument set but no micro moments, q_fixed fell from 168.8 to 52.3 (lambda 0.5) and from 287.0 to 123.6 (lambda 1) as sigma_p went from 1.0 to 1.5. Over the same moves q_cue went from 212.9 to 212.8 and from 196.7 to 185.3. Two points per slice is not a verdict. It is the LEARNINGS 1 pattern, now with Z = [X1, Z_excl].

**API WP9 calls for the sigma_p ray gate (DESIGN 6.4), once with and once without micro moments:**

```python
from nfp_harness.profile import ray_profile, NotMonotoneDecreasing

SIG_REF = {"sigma_const": 2.0, "sigma_hpwt": 2.0, "sigma_space": 2.0, "sigma_air": 1.0, "sigma_mpd": 1.0,
           "sigma_fwd": 2.0, "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0}
SPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
gate = NotMonotoneDecreasing("sigma_p", by=("lambda",), column="q_fixed", rise=3.84,
    fail_message="Profile check failed: q is still monotone in sigma_p with micro moments. LEARNINGS 14 "
                 "predicted otherwise; the plan needs the owner before any long estimation.")
for tag, problem, factory in (("micro", prob_micro, factory_micro), ("no_micro", prob_agg, factory_agg)):
    W1 = problem.initial_weighting(theta_prefit)       # blockdiag((Z'Z/n)^-1, (n V_micro)^-1); aggregate block only without micro
    res = ray_profile(problem, theta_prefit, SIG_REF, [0.5, 1.0, 2.0], {"sigma_p": SPS}, origin=0.0,
                      weighting=W1, run_dir=rd.path / f"profile_sigma_ray_{tag}", tag=tag,
                      workers=4, problem_factory=factory,   # factory(worker, threads) -> NFPGMMProblem
                      economics=econ,                       # optional: (problem, ev) -> {"median_own_el": ...}
                      pass_rule=gate)
    passed, message = res.verdict                           # also written to profile.txt; table: res.table()
    verdicts = res.classify("sigma_p", "q_fixed", by=("lambda",))   # {(lambda,): ShapeVerdict}; also "q_cue", "q_fixed:micro"
```

- **sigma_p bound:** the profile problem's layout must allow sigma_p up to 3.5. The estimation bound of 3.0 would make the 3.5 point OUT_OF_DOMAIN. Use `layout.replace(sigma_p={"upper": 3.5})` when building that problem.
- **Parallel points:** `PCMModel` has no `spare_capacity()` hook, so points run one at a time with the model's own market threads unless `workers` and `problem_factory` are passed. Each worker gets `os.cpu_count() // workers` threads; 4 workers on this host means 48 threads each.
- **Concentrated profiles:** add `concentrate=True` (with `hold=` for parameters to keep fixed and `cold=False` to warm-start along the grid).
- **Resume:** rerunning the identical call resumes from the saved points. A changed configuration raises `ProfileMismatchError` unless `new_run=True`.

The `chorus` MCP server failed to start; nothing here needed it.
## WP8

One limitation matters for WP9: when products are removed and prices are re-solved to a new Nash equilibrium, the price solver often fails to converge on PCM markets. It reports `not_converged` instead of hiding it, but WP9 should expect it on the 1984 counterfactual.

**Files** (committed with `git add` on only these three):
- `/local/home/istomin/pcm_inversion/nfp_harness/models/pcm_post.py`
- `/local/home/istomin/pcm_inversion/nfp_harness/diagnostics/economics.py`
- `/local/home/istomin/pcm_inversion/nfp_harness/tests/test_pcm_post.py`

`docs/estimation_harness/HANDOFF.md` has uncommitted edits that are not mine. I left them alone.

**Acceptance outputs** (synthetic PCM: 2 markets, 256 draws each, 3 firms, one body-type-like dummy):
```
elasticity vs finite differences, worst relative error 2.93e-10
Nash FOC residual at computed costs 2.78e-17
market 1984: CS change from dropping a dominated product 0.00e+00
market 1985: CS change from dropping a dominated product 0.00e+00
kappa 0.37 and 3.7: CS 2.377662734484 -> 2.377662734484 (relative change 0.0e+00);
  CS change from removal: fixed prices -0.5275495478, Nash -1.0794665052, both identical under kappa
Nash re-solve: converged after 13 iterations, FOC residual 4.9e-16
old nu=16 fixture: own elasticity max relative error 2.8e-16, margin 1.4e-15, nonzero cross counts equal
FLAG: nonzero_cross_median = 4 is outside the declared plausible range [100, inf].   (planted)
test_pcm_post.py: 14 passed
full suite: /home/istomin/anaconda3/bin/python -m pytest -q nfp_harness/tests -> 132 passed in 195 s
```
- **DESIGN WP8 fixture test:** I rebuilt the old body-sigma nu=16 markets with the new kernel. They match the old `pcm_product_level_body_nu16.csv` (from the old postestimation script).
- **Singular Nash Jacobian:** it comes back as `NASH_JACOBIAN_SINGULAR`, both from `nash_markups` and when a counterfactual hits a zero-share product. Nothing raises.
- **Checks:** ruff is clean, mypy is clean on my two modules, and there are no em dashes.

**Deviations**
- **Nash solver.** DESIGN's damped fixed point `p <- p + 0.5 r(p)`, with `r(p) = c + markup(p) - p`, cannot converge when d markup/dp has a real eigenvalue above 1. In one synthetic market that eigenvalue was 12.1, so `1 + 0.5 (12.1 - 1) = 6.5` and every iteration amplifies the error.
  - The default is now `method="newton"`. It takes Newton steps with a parallel finite-difference Jacobian. When no backtracked step lowers max |r|, it takes one damped fixed-point step instead. This is needed because ds/dp jumps wherever a draw's frontier changes.
  - DESIGN's iteration stays available as `method="fixed_point"`.
  - On 12 synthetic cases the Newton method converged in 6 and the fixed point in 4.
  - At a finite number of draws an exact price equilibrium may not exist, so `not_converged` can be the correct answer.
- **Alpha quintiles.** CS by alpha quintile groups draws by c_i, each draw's median alpha. The lognormal part a_i is integrated analytically inside each draw's CS, so it cannot be split across quintiles.

**API for WP9**
```python
from nfp_harness.models import pcm_post as pp
from nfp_harness.diagnostics.economics import economics_block, economics_values
# model = PCMModel(...); theta = result.theta; delta = result.evaluation.aux["delta"]   (n,) row order

post = pp.postestimation(model, theta, delta, firm_ids="firm_ids",
                         groups={"mi": "mi", "sw": "sw", "su": "su", "pv": "pv"}, group_other="car")
# post["products"]: (n,) arrays for product_level.csv (own_elasticity, nonzero_cross(_ratio), outside_diversion,
#   diversion_within_group / _leaving_group / _to_<g>, markup, cost, margin, shares, prices, group, market_ids)
# post["markets"], post["group_matrix"], post["nash_singular"]

cf = pp.drop_products_counterfactual(model, theta, delta, 1984, products["mi"].to_numpy() == 1,
                                     resolve_prices=True, tol=1e-7, max_iterations=500, tabulate=("fs",))
# cf["observed"|"fixed_prices"|"nash"]["cs"] in $1000 per household; cf["nash"]["status"] in
#   converged / not_converged / NASH_JACOBIAN_SINGULAR / nonpositive_price
# gain_musd = households_1984 * (cf["observed"]["cs"] - cf["nash"]["cs"]) / 1000
# cf["tables"]["alpha_quintile"], cf["tables"]["fs"]: cs_by_group rows (cs_<scenario>, cs_total_<scenario>, mass)

pp.compare_blp(pcm_summary, {"BLP micro": json_micro, "BLP no micro": json_nomicro}, note="...")
```
- **Spec fields `economics_block(problem, result, spec)` reads:** `firm_column`, `groups`, `group_other`, `price_range` and `plausible`.
- **Keys for `plausible`:** `own_elasticity_median`, `share_own_elasticity_below_-50`, `nonzero_cross_median`, `diversion_leaving_group_median`, `margin_median`, `inside_share_elasticity_median`, `sd_xi_over_sd_delta`, and `spread_buyer_dollars` (one range for all spreads, or a mapping by parameter name). A bound may be the string `"price_range"`.
- **Inputs WP9 passes on:** pass `cross_group_diversion = block.values["cross_group_diversion"]` to `run_diagnostics` options, so the segmented-corner detector sees it.

The `chorus` MCP server failed to start; nothing needed it. Scratch scripts are in `$KIROCREW_SCRATCH/wp8` and are untracked.

