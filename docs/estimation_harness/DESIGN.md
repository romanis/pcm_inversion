# Estimation harness: design (2026-11-16)

This is the design the harness was built from. `docs/LEARNINGS.md` records what estimation on real
data showed afterwards, and it wins where the two disagree. In particular the Petrin specification
of section 6 was later changed: the supply block, the Sargan-form aggregate moments, and the lower
bound on price sensitivity placed on the unobserved factor (LEARNINGS 19-32).

The harness estimates nested-fixed-point GMM demand models. The first model is the exact pure
characteristics model (PCM) and the first application is Petrin (2002). A paper is a spec: data,
parameter layout, moments, normalisations, caps and plausible ranges. The optimizer, moments,
weighting, diagnostics and checkpointing do not know which paper or model they run.

Contents: 0 findings from reading the code, 1 package layout and build, 2 problem and moment
interfaces, 3 C++ kernel, 4 optimizer, 5 diagnostics, 6 Petrin proof of concept and a Nevo
sketch, 7 work packages, 8 risks and open questions.

## 0. Findings from reading the current code

These change the plan and the coders must know them.

**F1. The current Petrin PCM uses only the 22 excluded instruments, so it is under-identified.**
`petrin_replication/estimate_pcm.py` sets `Z = demand_instruments0..21` (22 columns) and
concentrates out 13 linear coefficients (X1) by IV with that Z. pyblp builds
`ZD = [demand_instruments, exogenous X1 columns]` (`nevo_replication/.vendor/pyblp/primitives.py`
lines 135-144), which is 35 columns. A rank check on the Petrin data gives rank(Z) = 22 and
rank([Z, X1]) = 35, so X1 is not in the span of Z. With 22 moments, 13 linear coefficients and 18
nonlinear parameters (body-sigma spec) the old estimator had 31 unknowns for 22 moments. Nine
directions of theta were unidentified by construction. This is a direct cause of flat directions
and drift in LEARNINGS 1 and 13, on top of the mechanisms described there. The harness uses
`Z = [X1, demand_instruments]` (35 columns) exactly as pyblp does. With Petrin's 10 micro moments
the count is 45 moments, 13 + 18 = 31 unknowns, 14 overidentifying restrictions.

**F2. The zero-row regularisation path in `_solve_market` cannot run.** It reads
`m.cpp.shape[0]`, and `python_pcm_demand.Market` has no `shape`. Any zero row in ds/d delta would
raise `AttributeError`. At a converged inversion every data share is positive, so every product is
on some consumer's frontier and no row is zero. The harness removes the regularisation and reports
a singular ds/d delta as a named status (section 3.2).

**F3. Warm starts are overwritten by rejected trial points.** `PCMProblem.delta` stores the delta
of every converged market, including trial points that the line search then rejects. The harness
separates the accepted state from trial state (`commit` / `restore`, section 2.1).

**F4. The interpreter is `python`** (Python 3.12, numpy 1.26.4 with
OpenBLAS, scipy 1.13.1). pyblp 1.3.0 is vendored at `nevo_replication/.vendor` (gitignored).
pybind11 comes from the user site-packages. The existing
configured build tree `build-rel2` already holds googletest in `_deps`, so it does not need network
access to reconfigure. `estimate_pcm.py` and `pcm_petrin.py` in `petrin_replication/` are
byte-identical, and so are `nevo_replication/estimate_pcm.py` and `pcm_nevo.py`.

**F5. Frontier membership depends only on y_ij = delta_j + x_j . t_i.** In
`pcm_demand::shares` the conditional quality is `(delta + x t_i) / c_i` and the vertical model is
solved in that quality. Dividing every point (p_j, y_ij) and the origin by c_i > 0 does not change
which points lie on the upper hull, and sigma_p only changes the lognormal masses between hull
breakpoints. So phi (price-scale parameters) and sigma_p never move a product on or off a
frontier by themselves; only delta and the taste parameters do. Section 3.3 uses this.

## 1. Package layout and build

### 1.1 Directories

```
pcm_inversion/
  nfp_harness/                     new Python package (tracked)
    __init__.py                    version, public names
    types.py                       frozen dataclasses shared by every module (WP0)
    params.py                      ParameterSpec, ParameterLayout, unit rules, bounds
    linear.py                      concentrated linear parameters (IV, absorbed fixed effects)
    moments.py                     Moment protocol, IVMoments, MicroDataset, RatioMicroMoment
    problem.py                     NFPGMMProblem: model + moments + linear -> evaluate()
    weighting.py                   fixed, two-step, CUE; clustered S; micro covariance
    optimizer.py                   box trust-region Gauss-Newton, acceptance, stopping
    checkpoint.py                  atomic checkpoint write/read, run directory, resume
    runtime.py                     thread environment, detach (setsid), heartbeat, logging
    profile.py                     ray and grid profiles of the objective
    diagnostics/
      __init__.py                  run_diagnostics(): orchestrates the blocks below
      convergence.py               convergence class
      identification.py            SVD in natural and Jacobi units, named directions, weak ID
      collinearity.py              X1, Z, Z'X1, columns of G
      degeneracies.py              named degeneracy detectors
      overid.py                    Hansen J
      economics.py                 economic sanity block against spec ranges
      stability.py                 draws doubling, multi-start summary
      inference.py                 SE gating and sandwich SEs
      report.py                    JSON schema, validation, text rendering
    models/
      __init__.py
      pcm.py                       PCM adapter: markets, draws, kernel calls, threads, warm starts
      pcm_post.py                  elasticities, diversion, Nash costs, CS, counterfactuals
    data.py                        tie breaking, QMC draws, agent crossing (model-agnostic helpers)
    _native/
      __init__.py                  imports pcm_kernel, explains how to build if it is missing
      pcm_kernel.cpython-312-x86_64-linux-gnu.so   build output (gitignored)
    tests/                         pytest tests, one file per module
  demand_model/
    pcm_kernel.hpp / .cpp          solve_market (inversion + derivatives + micro parts)
    pcm_events.hpp / .cpp          max_step and frontier signatures
    pcm_welfare.hpp / .cpp         consumer surplus by frontier formula
    pcm_kernel_test.cpp, pcm_events_test.cpp, pcm_welfare_test.cpp
  python_bindings/
    pcm_kernel_python.cpp          PYBIND11_MODULE(pcm_kernel, m) calling the three bind_* functions
    bind_solve.cpp, bind_events.cpp, bind_welfare.cpp
  petrin_replication/estimate_pcm.py   rewritten as a Petrin spec on the harness
  petrin_replication/estimate_blp.py   unchanged
  nevo_replication/estimate_pcm.py     rewritten as a Nevo spec (last work package)
```

Every replication folder keeps exactly `estimate_blp.py` and `estimate_pcm.py` tracked. All
reusable code goes in `nfp_harness`. The untracked helpers (`pcm_petrin.py`,
`pcm_postestimation.py`, `profile_sigma_ray.py`, the `check_*` and `profile_market_times*` scripts)
stay local. No work package deletes them; their logic moves into `profile.py`, `models/pcm.py` and
`models/pcm_post.py`.

Results go to `petrin_replication/results/harness/<run_id>/`, which `.gitignore` already excludes
(`petrin_replication/*`).

### 1.2 CMake

Existing targets (`pcm_market_share`, `market_inversion`, `pcm_demand`, `python_pcm_demand`, ...)
are not changed. Additions:

`demand_model/CMakeLists.txt`:

```cmake
add_library(pcm_kernel STATIC
    pcm_kernel.cpp pcm_events.cpp pcm_welfare.cpp pcm_demand.cpp
    ${ROOT}/market_share/pcm_market_share.cpp)
set_target_properties(pcm_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pcm_kernel PUBLIC ${ROOT}/market_share ${ROOT}/demand_model ${Boost_INCLUDE_DIR})
target_link_libraries(pcm_kernel PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX)
foreach(t pcm_kernel_test pcm_events_test pcm_welfare_test)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} gtest_main pcm_kernel)
  gtest_discover_tests(${t})
endforeach()
```

The static library recompiles `pcm_demand.cpp` and `pcm_market_share.cpp` so the Python module is
self-contained and does not depend on a shared library found through the build tree's RPATH.

`python_bindings/CMakeLists.txt`, inside `if(pybind11_FOUND)`:

```cmake
pybind11_add_module(pcm_kernel_py
    ${ROOT}/python_bindings/pcm_kernel_python.cpp ${ROOT}/python_bindings/bind_solve.cpp
    ${ROOT}/python_bindings/bind_events.cpp ${ROOT}/python_bindings/bind_welfare.cpp)
target_link_libraries(pcm_kernel_py PRIVATE pcm_kernel)
set_target_properties(pcm_kernel_py PROPERTIES OUTPUT_NAME pcm_kernel
    LIBRARY_OUTPUT_DIRECTORY ${ROOT}/nfp_harness/_native)
```

Build command (reuses the configured tree from F4):

```
cmake -S . -B build-rel2 \
      -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=python \
      -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build-rel2 -j 64 --target pcm_kernel_py pcm_kernel_test pcm_events_test pcm_welfare_test
ctest --test-dir build-rel2 -R "pcm_(kernel|events|welfare)" --output-on-failure
```

`.gitignore` gains `nfp_harness/_native/*.so`.

### 1.3 Imports and runtime

Replication scripts start with:

```python
import os, sys, pathlib
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")   # idle OpenMP teams must not spin on 192 cores
os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")    # numpy work is small; keep cores for the kernel
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "nevo_replication" / ".vendor")]
```

The environment lines must run before numpy is imported, so they live in the script, not in the
package. `nfp_harness.runtime.check_environment()` warns if they were not set. Python tests run with
`python -m pytest nfp_harness/tests`.

## 2. Problem, moments and parameters

All shared dataclasses live in `nfp_harness/types.py`. WP0 writes that file in full and it is then
frozen; later packages extend behaviour, not field lists, unless the owner of `types.py` agrees.

### 2.1 Problem interface (what the optimizer sees)

```python
class EvalStatus(enum.Enum):
    OK = "ok"
    INVERSION_FAILED = "inversion_failed"        # some market did not reach the share tolerance
    SINGULAR_INNER_JACOBIAN = "singular_ds_ddelta"
    OUT_OF_DOMAIN = "out_of_domain"              # theta outside hard bounds (e.g. sigma_p <= 0)
    NUMERICAL = "numerical"                      # NaN/inf in g or G

@dataclass(frozen=True)
class Evaluation:
    theta: np.ndarray            # (P,) natural parameterisation
    g: np.ndarray                # (k,) stacked moments
    G: np.ndarray | None         # (k, P) total derivative dg/dtheta (chain rule through delta)
    status: EvalStatus
    blocks: dict[str, slice]     # moment block name -> rows of g, e.g. {"aggregate": 0:35, "micro": 35:45}
    aux: dict                    # model outputs: delta, xi, beta, per-market residuals, timings, notes
    seconds: float

@dataclass(frozen=True)
class StepLimit:
    tau_cap: float               # largest step fraction the model accepts along dtheta (1 = full step)
    tau_first: float             # first frontier event on any draw (exact for the linearised path)
    tau_vanish_lb: float         # lower bound on the step at which some product loses all buyers
    mass_profile: np.ndarray     # (n, 2) pairs (tau, max over products of affected buyer mass)
    note: str

class Problem(Protocol):
    layout: ParameterLayout
    def evaluate(self, theta: np.ndarray, need_jacobian: bool = True) -> Evaluation: ...
    def covariance(self, ev: Evaluation) -> np.ndarray: ...          # S(theta), (k, k)
    def set_weighting(self, W: np.ndarray) -> None: ...              # re-concentrates linear params
    # optional hooks; the optimizer checks with hasattr
    def max_step(self, ev: Evaluation, dtheta: np.ndarray) -> StepLimit: ...
    def commit(self, theta: np.ndarray) -> None: ...                 # theta accepted: promote trial state
    def restore(self, theta: np.ndarray) -> None: ...                # trial rejected: back to accepted state
    def cue_terms(self, ev: Evaluation) -> tuple[np.ndarray, np.ndarray]: ...  # (V: C x k, dV: C x k x P)
    def state_dict(self) -> dict: ...                                # warm starts etc. for checkpoints
    def load_state_dict(self, d: dict) -> None: ...
```

`evaluate` never raises for model failures; it returns a non-OK status. The optimizer treats any
non-OK trial as a rejected step (section 4.3).

`commit` and `restore` define warm-start semantics. The model keeps an accepted state (per market:
delta, the theta it was solved at) and a trial state. `evaluate` always warm-starts from the
accepted state and writes the trial state. `commit(theta)` copies the trial state to the accepted
state. `restore(theta)` discards it. This fixes F3.

### 2.2 The generic NFP-GMM problem

`nfp_harness.problem.NFPGMMProblem(model, moments, linear, layout)` implements `Problem` for any
model that has an inner inversion. The model interface:

```python
class InnerModel(Protocol):
    markets: list[MarketIndex]      # MarketIndex(market_id, rows: np.ndarray of product rows)
    n_products: int
    def solve(self, theta, requests: KernelRequests, need_jacobian: bool) -> InnerSolution: ...
    def max_step(self, sol: InnerSolution, dtheta) -> StepLimit: ...     # optional
    def commit(self, theta) -> None: ...
    def restore(self, theta) -> None: ...

@dataclass
class InnerSolution:
    delta: np.ndarray                      # (n,)
    D: list[np.ndarray] | None             # per market (J_t, P): d delta_t / d theta
    status: EvalStatus
    per_market: list[dict]                 # converged, residual, evaluations, seconds, threads, stages
    outputs: dict[str, object]             # kernel products requested by moments (micro parts ...)
```

`KernelRequests` is how a moment asks the model for extra kernel outputs (for the PCM: micro parts,
section 3.2). `NFPGMMProblem.__init__` collects `moment.requests()` from every moment once.

### 2.3 Moment interface and the chain rule

```python
@dataclass
class MomentBlock:
    name: str
    value: np.ndarray                      # (k_b,)
    d_delta: np.ndarray | None             # (k_b, n) partial derivative in delta, theta held fixed
    d_theta: np.ndarray | None             # (k_b, P) partial derivative in theta, delta held fixed
    data: np.ndarray | None                # (k_b,) data values for display (micro), else None

class Moment(Protocol):
    name: str
    def requests(self) -> KernelRequests: ...
    def evaluate(self, sol: InnerSolution, linear: LinearFit, need_jacobian: bool) -> MomentBlock: ...
    def covariance_block(self, sol: InnerSolution, linear: LinearFit) -> np.ndarray: ...
```

The framework applies the implicit-function chain rule once, for all moments:

```
D_t = d delta_t / d theta = -(d s_t / d delta_t)^{-1} (d s_t / d theta)      (from the kernel, per market)
G_b = d_theta_b + sum_t d_delta_b[:, rows_t] @ D_t
```

Moments never see D. A moment that depends on delta only through xi gives `d_delta`; a micro
moment gives both. `d_delta` is dense (k_b x n); for Petrin that is 45 x 2407, which is small.

**Aggregate IV moments** (`IVMoments(Z, linear)`): xi = delta - X1 beta(delta) with beta concentrated
out (section 2.4), so xi = M delta and g = Z' M delta / n, `d_delta = Z' M / n`, `d_theta = 0`.
Covariance block: clustered S = (1/n) sum_c (sum_{j in c} z_j xi_j)(sum_{j in c} z_j xi_j)'.

**Micro moments** pooled across markets (`MicroDataset`, `MicroPart`, `RatioMicroMoment`). This
mirrors pyblp 1.3.0 (`economies/problem.py` lines 1215-1260). A part is a pair (agent weight h,
product weight vector v with outside value v0). Its value is

```
part = (1 / Den) sum_t sum_i w_it h(i) [ v0 (1 - sum_j s_ijt) + sum_j v_j s_ijt ]
     = (1 / Den) sum_t [ v0 H_t + sum_i w_it h(i) sum_j (v_j - v0) s_ijt ],      H_t = sum_i w_it h(i)
Den  = sum_t sum_i w_it   (13 for Petrin: agent weights sum to one in each market)
```

s_ijt is draw i's choice probability, already integrated over the lognormal price sensitivity in
closed form. The kernel returns, per market, `sum_i w_i h(i) sum_j u_j s_ij` and its partial
derivatives in delta_t and theta for each requested (h, u = v - v0) pair (section 3.2). The micro
moment is a function f of parts (for Petrin, a ratio), with g = f(parts) - data value. Its partials
are `d_delta = grad_f . d parts / d delta` and `d_theta = grad_f . d parts / d theta`. Pooling across
markets enters only through the sum over t, so the chain rule above applies unchanged.

Micro covariance (needed only when W is updated): exactly as pyblp, part covariances are
`E[v1 v2] - E[v1] E[v2]` with `E[v1 v2]` computed as a part with h = h1 h2 and v = v1 * v2
(elementwise, v0 = v1_0 v2_0), divided by the dataset's `observations` (29,125 for Petrin's CEX).
The micro covariance is `grad_f Sigma_parts grad_f'`. The stacked S is
`blockdiag(S_aggregate, n * V_micro)` with n the number of products (pyblp
`results/problem_results.py` line 756). The product parts are requested only when
`covariance()` is called.

### 2.4 Concentrated linear parameters

`linear.LinearIV(X1, Z_block_of_W)` concentrates beta out with the aggregate block of W:
`beta = (X1' Z W_aa Z' X1)^{-1} X1' Z W_aa Z' delta`, `M = I - X1 (...)`. This is exact when W is
block diagonal between aggregate and micro moments (true for all three weighting modes here, since
S is block diagonal) and micro moments do not depend on beta (true: they depend on delta only).
`set_weighting(W)` recomputes M. `linear.Absorb(groups)` replaces X1 with within-group demeaning for
fixed effects (Nevo brand dummies).

For inference the full Jacobian over (beta, theta) is used, not the concentrated one: the aggregate
rows get `[-Z' X1 / n, Z' D / n]` and the micro rows get `[0, G_micro]` (section 5.12).

### 2.5 Parameter layout

```python
@dataclass(frozen=True)
class ParameterSpec:
    name: str                  # e.g. "sigma_sw", "pi_mi_logfs_fv", "phi_low_inc", "sigma_p"
    role: str                  # model-specific: "taste", "scale", "sigma_p"
    column: str | None         # characteristic the taste loads on ("sw"); None for scale/sigma_p
    basis: str | None          # draw column: "nu3", "logfs_fv", "low_inc"
    start: float
    lower: float               # hard lower bound (domain or sign normalisation)
    upper: float               # hard upper bound
    cap: bool = False          # True if 'upper' is a stated normalisation, not a domain guard
    unit: float | None = None  # natural unit; None means the automatic rule below
    free: bool = True          # False fixes it at start (used by pre-fits and profiles)
    note: str = ""             # printed in the report next to the parameter

class ParameterLayout:
    specs: list[ParameterSpec]
    names: list[str]; P: int
    def units(self, data) -> np.ndarray          # automatic rule applied where unit is None
    def bounds(self) -> tuple[np.ndarray, np.ndarray]
    def free_mask(self) -> np.ndarray
    def to_scaled(self, theta) / from_scaled(u)  # u = (theta - start) / unit
```

The optimizer works in identity coordinates divided by the natural unit (no log transforms), so a
sigma that collapses to zero reaches its lower bound 0 and is visible as such, instead of running
off to minus infinity in log space.

Automatic natural units (overridable per parameter in the spec): prices and delta are in $1000.
- taste spread on column k: `unit = 1 / sd(x_k)` ($1000 of utility per standard deviation of the
  characteristic across products; 1 for a dummy with sd near 0.5 gives about 2, so dummies use
  `unit = 1` explicitly in the spec);
- demographic interaction on column k with basis b: `unit = 1 / (sd(x_k) sd(b))`;
- scale parameter phi_q on basis b: `unit = 1 / sd(b)` (one sd of the basis moves log alpha by 1);
- sigma_p: `unit = 0.1`.

Normalisations are declared in the spec and printed in every report:
- `median_alpha`: log alpha_i = log c_i + log a_i with a_i ~ LogNormal(0, sigma_p) and
  log c_i = phi . scale_basis_i with no constant, so mu = 0 (LEARNINGS 11). Fixed, not a parameter.
- `cap`: a parameter with `cap=True` has its upper bound reported as a normalisation of a direction
  the data leave quiet. Ending at it is `NORMALISATION_CAP_ACTIVE`, not a failure.
- `sign`: taste spreads have lower bound 0 (draws are symmetric, so sigma and -sigma are the same
  model).

## 3. C++ kernel additions

Model, per draw i in a market with N products sorted by strictly increasing price:
`u_ij = delta_j + x_j . t_i - c_i a_i p_j`, `u_i0 = 0`, `t_i = sum_m theta_m taste_basis(i, m) e_{col(m)}`,
`log c_i = phi . scale_basis_i`, `a_i ~ LogNormal(0, sigma_p)`. Write `y_ij = delta_j + x_j . t_i`.
Draw i's frontier F_i is the set of vertices of the increasing part of the upper convex hull of
`{(0, 0)} U {(p_j, y_ij)}`, excluding the origin. Product j in F_i is bought by draw i for
`a c_i` in `(e_{next}, e_j)`, where e are the hull slopes (`cond_vertical` in `pcm_demand.cpp`).

### 3.1 Market input

`pcm_demand::Market` is reused unchanged (x, p, weights, taste_basis, taste_col, scale_basis). A new
struct holds agent-level micro weights:

```cpp
namespace pcm_kernel {
struct MicroRequest {
    Eigen::MatrixXd H;            // R x L agent weights h_l(i) (rows aligned with draws)
    Eigen::MatrixXd U;            // N x Pm product weights u_p = v_p - v0_p
    std::vector<int> h_of_part;   // Pm, column of H used by part p
};
```

### 3.2 `solve_market`: frontier lift, inversion and derivatives in one call

```cpp
enum class Status { OK, NOT_CONVERGED, INFEASIBLE_START, SINGULAR_DS_DDELTA, INVALID_INPUT };

struct SolveOptions {
    double tol = 1e-9;                 // max |s - s_data|
    unsigned warm_evals = 2500, cold_evals = 40000, path_evals = 1500;
    double path_seconds = 90.0;        // budget for the theta-path continuation
    int n_threads = 1;                 // OpenMP threads for this call
    bool derivatives = true;           // ds/ddelta, ds/dtheta, ddelta/dtheta
    bool price_derivatives = false;    // ds/dp (postestimation)
    double rcond_min = 1e-13;          // below this ds/ddelta is reported singular
};

struct SolveResult {
    Status status;
    Eigen::ArrayXd delta, s;
    Eigen::MatrixXd ds_ddelta, ds_dtheta, ddelta_dtheta, ds_dp;   // N x N, N x P, N x P, N x N
    Eigen::VectorXd micro;              // Pm part sums  sum_i w_i h(i) sum_j u_j s_ij
    Eigen::MatrixXd micro_d_delta;      // Pm x N
    Eigen::MatrixXd micro_d_theta;      // Pm x P
    double max_abs_residual, rcond, seconds;
    unsigned share_evaluations;
    std::string stages;                 // "warm 3s; path tau=1 12s" etc.
};

SolveResult solve_market(const Eigen::ArrayXd& s_data, const pcm_demand::Market& mk,
                         const pcm_demand::Params& prm,
                         const Eigen::ArrayXd& delta_warm,            // empty: no warm start
                         const pcm_demand::Params* prm_warm,          // theta at which delta_warm solved
                         const MicroRequest* micro,                   // nullptr: none
                         const SolveOptions& opt);
}
```

P = M + Q + 1 in the column order of `pcm_demand::Shares::d_theta`: taste, scale, sigma_p.

Algorithm (moves `_solve_market` from Python into C++, all with the GIL released by the binding):
1. Call `omp_set_num_threads(opt.n_threads)` at entry. The OpenMP thread-count ICV is per calling
   thread, so each Python worker thread sets its own team size. This replaces the ctypes call into
   libgomp.
2. Warm stage: `pcm_demand::invert(s_data, mk, prm, delta_warm, tol, warm_evals)` when a warm start
   exists, else skip to step 4. `invert` already repairs zero-share products with `frontier_lift`
   and falls back to heterogeneity continuation and share-path continuation.
3. Theta-path stage, if step 2 failed and `prm_warm` is given: solve along
   `theta(tau) = theta_warm + tau (theta - theta_warm)` with tau stepping from 0 to 1 (start step 0.5,
   double on success, halve on failure, stop below 0.01 or when `path_seconds` is spent), each solve
   warm-started from the last converged tau.
4. Cold stage: `invert(s_data, mk, prm, empty, tol, cold_evals)`.
5. Status: `OK` if the residual is below tol, `INFEASIBLE_START` if no feasible delta was ever
   found, else `NOT_CONVERGED`.
6. Derivatives at the converged delta: one `pcm_demand::shares(..., true)` pass (with ds/dp only if
   requested). Factor ds/ddelta with `Eigen::FullPivLU`, estimate the reciprocal condition number
   from the LU (ratio of smallest to largest absolute pivot of U, then confirmed by a 1-norm
   estimate). If rcond < `rcond_min`, status `SINGULAR_DS_DDELTA` and `ddelta_dtheta` is NaN. Else
   `ddelta_dtheta = -LU.solve(ds_dtheta)`. No regularisation of any row (F2).
7. Micro parts in the same draw loop as step 6, so they cost no extra hull. For draw i, with its
   conditional share vector s_i (nonzero only on F_i) and its conditional Jacobian Jd_i (nonzero
   only on F_i x F_i, tridiagonal in frontier order), and for each part p with
   `w_h = w_i H(i, h_of_part[p])`:
   ```
   micro[p]          += w_h * u_p' s_i
   micro_d_delta[p]  += (w_h / c_i) * u_p' Jd_i
   micro_d_theta[p,m] += (w_h * taste_basis(i,m) / c_i) * u_p' Jd_i x_{col(m)}      (taste)
   micro_d_theta[p,M+q] += -w_h * scale_basis(i,q) * u_p' Jd_i dc_i                 (scale)
   micro_d_theta[p,M+Q] += w_h * u_p' dsig_i                                        (sigma_p)
   ```
   These are the same expressions `pcm_demand::shares` uses for aggregate shares, with the weight
   w_i replaced by w_i h(i) and a contraction with u_p. Accumulate in the same fixed blocks of 64
   draws, summed in block order, so results are bit-identical for every thread count.

`pcm_demand.cpp` changes: `frontier_lift` and `cond_vertical` move out of the anonymous namespace
into `namespace pcm_demand::detail` and are declared in `pcm_demand.hpp`, so `pcm_kernel.cpp` can
call `cond_vertical` per draw. No behaviour changes.

### 3.3 `max_step`: first frontier events along a direction

```cpp
struct StepEvents {
    double tau_first;                    // min over draws of tau_i
    double tau_vanish_lb;                // min_j max_{i: j in F_i} tau_i
    Eigen::ArrayXd tau_draw;             // R, first event per draw (tau_max if none)
    Eigen::ArrayXi kind_draw;            // R, 0 none, 1 product leaves, 2 product enters
    Eigen::ArrayXi product_draw;         // R, product involved in the first event
    Eigen::MatrixXd mass_profile;        // n_grid x 2: (tau, max_j affected mass of j)
};
StepEvents max_step(const Eigen::ArrayXd& delta, const Eigen::ArrayXd& delta_dot,
                    const pcm_demand::Market& mk, const pcm_demand::Params& prm,
                    const Eigen::ArrayXd& dtheta, double tau_max, int n_threads);
std::vector<uint64_t> frontier_signature(const Eigen::ArrayXd& delta, const pcm_demand::Market& mk,
                                         const pcm_demand::Params& prm, int n_threads);   // hash of F_i per draw
```

`delta_dot = D dtheta` comes from the current evaluation (the linearisation of delta(theta)).

**Path.** Along the step, `delta(tau) = delta + tau delta_dot` and `theta(tau) = theta + tau dtheta`.
t_i is linear in the taste parameters, so `y_ij(tau) = y_ij + tau v_ij` with
`v_ij = delta_dot_j + x_j . dt_i`, `dt_i = sum_m dtheta_m taste_basis(i,m) e_{col(m)}`. By F5 the
scale and sigma_p components of dtheta do not enter: membership depends on y only. So every point
of the hull problem moves on a straight line in tau, and every hull certificate below is an affine
function `f(tau) = f0 + tau f1` with `f0 > 0` at tau = 0. A certificate fails at `tau = f0 / (-f1)`
when `f1 < 0` and never otherwise.

**Certificates for draw i.** Let the hull vertices be `h_0 = origin, h_1, ..., h_m` (the frontier
in price order), with `P_r = p_{h_r}`, `Y_r = y_{i h_r}`, `P_0 = Y_0 = 0`.
- (C1) Strict convexity at each interior vertex r = 1..m-1:
  `f = (Y_r - Y_{r-1})(P_{r+1} - P_r) - (Y_{r+1} - Y_r)(P_r - P_{r-1}) > 0`. Failure: h_r leaves.
- (C2) Last slope positive: `f = Y_m - Y_{m-1} > 0`. Failure: h_m leaves (only a <= 0 would buy it).
- (C3) Each product k not on the frontier, priced between `P_r` and `P_{r+1}` (r = 0..m-1):
  `f = (Y_r - y_k)(P_{r+1} - P_r) + (p_k - P_r)(Y_{r+1} - Y_r) > 0` (k strictly below the edge).
  Failure: k enters.
- (C4) Each product k priced above `P_m`: `f = Y_m - y_k > 0`. Failure: k enters as the new top.
  If m = 0 (the draw buys nothing), C4 is `f = -y_k > 0` for every k.

Until some certificate of draw i fails, its hull does not change, so its first event is
`tau_i = min` over its certificates (at most 2N of them). Building the hull is the monotone chain
already in `frontier_lift`, so the cost per call is about one share evaluation. A certificate with
`f0 = 0` at tau = 0 (exact collinearity; measure zero with continuous draws) counts as an event at
tau = 0 if `f1 < 0` and is ignored otherwise.

**Aggregates.** Only one draw's first event is exact under the linearisation; after it, that draw's
hull changes and its later events are not computed. The returned quantities use only first events:
- `tau_first = min_i tau_i`. With 16,000 draws per market this is close to zero and is reported, not
  used as a cap.
- `mass_profile`: for each product j with buyer mass `B_j = sum_{i: j in F_i} w_i`, the affected
  mass `A_j(tau) = sum_{i: j in F_i, tau_i <= tau} w_i / B_j`, and the profile is `max_j A_j(tau)` on a
  grid of 64 tau values from 0 to tau_max. It bounds the fraction of j's buyers whose frontier has
  changed, which bounds how far ds/ddelta and ds/dtheta can have moved from the values in G.
- `tau_vanish_lb = min_j max_{i: j in F_i} tau_i`. Product j can lose its last buyer only after every
  draw that buys it has had at least one event, so no product can reach zero share for
  `tau < tau_vanish_lb`. This is a provable lower bound for the linearised path.

**When it is exact.** The event times are exact for the affine path `(delta + tau delta_dot,
theta + tau dtheta)`. In the share map with delta held fixed (`delta_dot = 0`, as in counterfactual
price paths and unit tests) that path is the true path, so the events are exact. In the outer
problem delta(theta) is not affine; the true path differs by `O(tau^2 |d^2 delta / d theta^2|)` and
the events are a first-order prediction. The optimizer guards against this error in three ways
(section 4.4): it caps steps by affected mass rather than by `tau_first`, it measures the realised
change with `frontier_signature` at every trial point and adapts the cap, and it accepts steps only
on the actual objective (gain ratio), with inversion failures treated as rejections.

### 3.4 Consumer surplus (`pcm_welfare`)

```cpp
struct Surplus { double total; Eigen::ArrayXd per_draw; };   // in price units per household
Surplus consumer_surplus(const Eigen::ArrayXd& delta, const pcm_demand::Market& mk,
                         const pcm_demand::Params& prm, int n_threads);
```

The formula is the one in `petrin_replication/pcm_postestimation.py` moved to C++: with
`q_k = y_ik / c_i` on the frontier and slopes `e_k`,
`CS_i = sum_k [ q_k E(1/a; e_{k+1} < a < e_k) - p_k P(e_{k+1} < a < e_k) ]`,
`E(1/a; lo < a < hi) = exp(s^2/2) [Phi((ln hi + s^2)/s) - Phi((ln lo + s^2)/s)]` for LogNormal(0, s).
The Python loop over 16,000 draws per market is replaced by the OpenMP draw loop with the same
64-draw blocks. `per_draw` lets postestimation tabulate CS by alpha quantile and family size
(LEARNINGS 12c).

### 3.5 Per-market thread allocation

`models/pcm.py` runs markets in a `ThreadPoolExecutor` with one Python thread per market (13 for
Petrin). Each `solve_market` call gets `n_threads`. Allocation with total T = `os.cpu_count()` (192):
- first evaluation: weight `J_t^2 R_t` (LEARNINGS 9);
- afterwards: weight = exponential moving average (factor 0.5) of measured CPU work
  `seconds_t * n_threads_t` from previous evaluations;
- `n_t = max(1, floor(T w_t / sum w))`, leftover threads to the markets with the largest weight.
The allocation is written to the run log every evaluation. `OMP_WAIT_POLICY=PASSIVE` (section 1.3)
stops idle teams from spinning.

### 3.6 Kernel unit tests (tiny markets with hand-computed answers)

`pcm_events_test.cpp` (single draw, no taste parameters unless stated, `delta_dot` given, so the
events are exact):
1. p = (10, 12, 14), y = (5, 5.5, 7). B is below the A-C line (6 at p = 12). `delta_dot = (0, 1, 0)`:
   B enters at tau = 0.5, kind 2, product 1.
2. Same prices, y = (5, 6.5, 7), `delta_dot = (0, -1, 0)`: B leaves at tau = 0.5.
3. p = (10, 20), y = (5, 8), `delta_dot = (0, -1)`: C2 fails when y_2 = y_1, tau = 3.
4. p = (10, 20), y = (5, 4), `delta_dot = (0, 1)`: C4 fails at tau = 1.
5. One product, p = 10, y = -1, `delta_dot = 1`: enters through the origin edge at tau = 1.
6. Taste direction: x = (0, 1, 0) column "dummy", one taste parameter with basis 1 for the draw,
   delta fixed, `dtheta = 1` on it: same answer as test 1.
7. Scale and sigma_p directions only: `tau_draw = tau_max` for all draws (F5).
8. Two draws with weights 0.3 and 0.7 and known tau_i: check `mass_profile` steps at the known taus
   and `tau_vanish_lb`.
9. Brute force: random 20-product, 50-draw markets; scan tau on a grid of 10^5 points with
   `frontier_signature`; the first signature change per draw matches `tau_draw` within the grid step.

`pcm_kernel_test.cpp`:
1. No heterogeneity (no taste or scale parameters): `pcm_share::initial_guess` is the exact inverse;
   `solve_market` returns it within 1e-12 on the first stage.
2. `ddelta_dtheta` against central finite differences of the inversion (h = 1e-5 in each
   parameter, inversion tol 1e-13): max relative error below 1e-5 on the `pcm_demand_test` fixture.
3. Micro parts: h = 1, u = 1 equals the inside share sum; h = 1, u = e_j equals s_j; derivatives
   against finite differences below 1e-6.
4. Same results bit-for-bit with n_threads = 1 and 8.
5. A product forced below every frontier (delta very low) with positive data share: the warm stage
   lifts it and converges; status OK.
6. Duplicate characteristics and a $0.001 price gap: status `SINGULAR_DS_DDELTA` or OK with rcond
   reported, never silent NaN in G.

`pcm_welfare_test.cpp`:
1. One draw, one product with q = 2, p = 1, sigma_p = 0.5: CS equals the closed form
   `q E(1/a; a < q/p) - p F(q/p)`.
2. Scale invariance (LEARNINGS 11, Anna and Boris): multiplying delta and taste parameters by kappa
   and c_i by kappa leaves CS unchanged to 1e-12.
3. Removing a dominated product changes nothing; removing the only product gives CS = 0.

## 4. Optimizer

`optimizer.minimize(problem, weighting, options, checkpoint) -> OptimizeResult`. It minimises
`q(theta) = n g' W g` over the free parameters within the box `[lower, upper]`.

### 4.1 Scaling

Coordinates: `u = (theta - start) / unit` (natural units, section 2.5). Inside each step the linear
system is column-scaled by Jacobi factors `d_k = max(d_k_previous, ||(W^{1/2} G)_{:,k}|| unit_k)`
(More 1978: non-decreasing, so scaling does not chase noise), floored at `1e-8 max_k d_k`. The trust
region is a box in natural units, which keeps steps interpretable ("at most 0.3 of a natural unit in
each parameter") and matches the radius that prevented inversion stalls in the old code.

### 4.2 Step

At the current point with residual `r = W^{1/2} g` and Jacobian `A = W^{1/2} G diag(unit)`:

```
minimise  ||r + A d||^2 + mu ||diag(d_k) d||^2
subject to  max(-Delta, lower_u - u) <= d <= min(Delta, upper_u - u)     (free parameters only)
```

solved with `scipy.optimize.lsq_linear` (method "bvls" for P <= 30, else "trf") on the stacked
system `[A; sqrt(mu) diag(d)]`. mu is a small ridge for rank deficiency, `mu = 1e-10 max(d)^2`,
not the step control; the box radius Delta is the step control. Fixed parameters are removed.

If the problem has `max_step`, compute `limit = problem.max_step(ev, d * unit)` and truncate:
`tau_cap = min(1, tau at which mass_profile reaches q_mass, 0.5 tau_vanish_lb)`, `d <- tau_cap d`.
`q_mass` starts at 0.2 and adapts (section 4.4).

### 4.3 Acceptance and radius

Predicted value `m(d) = n ||r + A d||^2`, actual `q_trial` from `evaluate`.
`rho = (q - q_trial) / (q - m(d))`.
- Trial status not OK: reject, `restore`, `Delta <- 0.25 ||d||_inf`, record the status in history.
- `rho >= 1e-4`: accept, `commit`, write checkpoint.
- `rho < 0.25`: `Delta <- 0.25 ||d||_inf`. `rho > 0.75` and the step reached the box
  (`||d||_inf >= 0.9 Delta`): `Delta <- min(2 Delta, Delta_max)`. `Delta_0 = 0.3`,
  `Delta_max = 5`, `Delta_min = 1e-8`.
- A step truncated by `max_step` with `rho > 0.75`: `q_mass <- min(0.8, 1.5 q_mass)`.

### 4.4 Guards for the linearisation in `max_step`

After every trial evaluation, the PCM model compares `frontier_signature` at the trial point with
the accepted point and computes the realised affected mass `max_j A_j`. If the realised mass exceeds
twice the predicted mass at `tau_cap`, `q_mass <- q_mass / 2`. Both numbers are logged per step and
summarised in the report (`kink_limited_steps`, `predicted_vs_realised_mass`).

### 4.5 Weighting modes (`weighting.py`)

- `Fixed(W)`: W given by the spec. Default for step 1:
  `W1 = blockdiag((Z'Z/n)^{-1}, (n V_micro(theta_0))^{-1})`, which is pyblp's initial weighting with
  micro moments.
- `TwoStep(W1)`: run to convergence with W1, then `W2 = S(theta_hat_1)^{-1}` with
  `S = blockdiag(S_aggregate_clustered, n V_micro)`, re-concentrate beta (`set_weighting`), restart
  the optimizer at theta_hat_1 with a fresh history segment. Stage boundaries are checkpointed.
  Step 2 starts only if the step-1 report has no `SCALE_ARTIFACT` or `SD_XI_BLOWUP` (section 5.4);
  otherwise the run stops and says so, because W2 would inherit the degenerate point. When the aggregate
  block is in Sargan form (`SarganIVMoments`, scale-free) `SD_XI_BLOWUP` is a warning and step 2 runs: that
  form cannot be lowered by shrinking xi, and a concentrated xi drives it toward n h_jj, not 0 (LEARNINGS 19).
- `CUE`: `q = n g' S(theta)^{-1} g`. Gradient exact from `cue_terms` (LEARNINGS 1 formula),
  Hessian approximated by `2 n G' S^{-1} G` in the same trust-region loop, acceptance on the actual q.
  The micro block of S is held at its value at the CUE start point and the report states it. CUE is
  supported for generality; the Petrin plan does not use it (LEARNINGS 13, 14).

Every accepted step records both `q_fixed` (with W1) and `q_cue` (cheap: S from xi) together with
`sd(xi)`, because the degeneracy detectors need both.

### 4.6 Stopping rules and classes

Let `pg` be the projected gradient in natural units: `grad_k = 2 n (G' W g)_k unit_k`, set to zero
for parameters at a bound whose gradient points outward. Let `dq_gn` be the predicted decrease of
the full Gauss-Newton step (the step of 4.2 with `Delta = inf`, bounds kept).

| class | rule |
|---|---|
| `CONVERGED` | status OK and `max_k |pg_k| <= 1e-3 (1 + q)^{1/2}` and `dq_gn <= 1e-6 (1 + q)` |
| `CONVERGED_AT_CAP` | as CONVERGED, with one or more `cap=True` parameters at their cap |
| `STALLED` | `Delta < Delta_min` or 10 consecutive rejections, while the CONVERGED test fails |
| `FLAT_PROGRESS` | last 25 accepted steps reduce q by less than `1e-3 (1 + q)` in total, while the CONVERGED test fails |
| `ITERATION_LIMIT` / `TIME_LIMIT` | limits from options; the CONVERGED test fails |
| `INVERSION_FAILED` | the start point, or the accepted point after `restore`, cannot be evaluated |

The gradient tolerance is in chi-square units per natural unit: at CONVERGED, moving one natural
unit in any free parameter changes q by at most `1e-3 (1+q)^{1/2}` to first order. The CUE run of
LEARNINGS 13 (q = 0.80, |grad| = 0.38) fails this test and would be `FLAT_PROGRESS` or `STALLED`.
The report always prints the measured values next to the thresholds.

### 4.7 Checkpoint and resume

`checkpoint.RunDirectory(base, spec_hash, run_key)`:
`run_key` is a hash of the CLI options that change results (nu, seed, stage list, weighting). The
run directory is `<base>/<spec_name>_<run_key[:10]>/`. Files:
- `checkpoint.npz` + `checkpoint.json`: stage, iteration, theta, Delta, q_mass, Jacobi factors,
  W and its provenance, the stage's W1, best point, problem `state_dict()` (per-market accepted
  delta and theta), thread allocation weights, history arrays, spec hash, code commit;
- `history.csv`: one row per trial (accepted flag, q, q_fixed, q_cue, sd_xi, rho, Delta, tau_cap,
  status, seconds, theta);
- `log.txt`, `heartbeat` (timestamp, pid, iteration, rewritten each evaluation);
- `report.json`, `report.txt` at the end of each stage.

Writes are atomic: write `checkpoint.tmp.npz`, `fsync`, `os.replace`, then the JSON the same way.
A checkpoint is written after every accepted step and at every stage boundary.

Resume: running the same command finds the run directory, checks `spec_hash` (refuses with a clear
message on mismatch unless `--new-run`), loads the checkpoint, calls `problem.load_state_dict`,
re-evaluates the accepted point once (fast: warm deltas are in the checkpoint) and continues. An
evaluation in progress when the process died is lost; nothing else is.

Process survival: `runtime.detach(log_path)` re-executes the same command under `setsid` with
stdin closed and stdout/stderr to `log.txt`, so recycling the agent shell does not kill the run.
Scripts expose it as `--detach`. `runtime.status(run_dir)` reads the heartbeat and reports whether
the pid is alive.

### 4.8 Optimizer tests (WP5, toy problems in pure numpy)

1. Linear least squares: converges in one accepted step, class CONVERGED, theta exact to 1e-10.
2. Rosenbrock residuals `[10 (y - x^2), 1 - x]` from (-1.2, 1): reaches (1, 1), CONVERGED.
3. Optimum outside the box: ends at the bound, CONVERGED (with bound flag); with `cap=True`,
   CONVERGED_AT_CAP.
4. A residual whose q falls slowly forever (`r = [exp(-theta)]`) under an iteration limit: never
   CONVERGED; class FLAT_PROGRESS or ITERATION_LIMIT and the degeneracy test of 5.4 flags
   `RUN_TO_INFINITY`.
5. Classifier unit test with a synthetic history: q change below 1e-6 per step and
   `max |pg| = 0.38`: not CONVERGED.
6. Piecewise-linear residual with a kink and a `max_step` oracle: every accepted step stays within
   one piece or crosses the kink once; converges.
7. Failing evaluations in a region: steps into it are rejected, `restore` is called, radius shrinks,
   run still converges outside it.
8. Kill and resume: a subprocess with 0.2 s per evaluation is killed with SIGKILL after 5 accepted
   steps; rerunning the same command resumes at step 5 and the final theta equals the uninterrupted
   run's bit for bit.
9. Two-step: on a linear IV toy, the two-step estimate equals the closed-form efficient GMM estimate.
10. CUE: gradient equals finite differences to 1e-6 on a toy with clustered S.

## 5. Diagnostics

`diagnostics.run_diagnostics(problem, result, spec, options) -> Report` runs at the end of every
stage. Each block returns a dict with `status` in `{"ok", "flag", "fail", "not_run"}`, the numbers,
the thresholds and a one-sentence `message` from the templates below. Thresholds come from
`spec.diagnostics` with the defaults shown.

### 5.1 Convergence class

Section 4.6. Message template: `"Convergence: {class}. Projected gradient {pg:.2e} (tolerance
{tol:.2e}); predicted Gauss-Newton decrease {dq:.2e} (tolerance {tolq:.2e}); {n_acc} accepted
steps, {n_eval} evaluations, {hours:.1f} h."` Every other block that depends on a stationary point
(SEs, J test, economic interpretation) prints `"not interpreted: objective did not converge"` when
the class is not CONVERGED or CONVERGED_AT_CAP.

### 5.2 Flat directions and units versus identification

Two matrices at the final point: natural `A_nat = W^{1/2} G diag(unit)` and Jacobi
`A_jac = A_nat diag(1 / ||A_nat[:, k]||)`. SVD of each: singular values `s_1 >= ... >= s_P` and right
vectors v. Relative condition `rc = s_P / s_1`.

For every singular direction with `s_l / s_1 < 1e-4` in either matrix:
- `dq_per_unit = n s_l^2` from `A_nat` (change in q for a unit move along the direction in natural
  units). A direction with `dq_per_unit < 1` is `FLAT_IN_NATURAL_UNITS`: a full natural unit moves q by
  less than one chi-square unit.
- loadings: the components of v (mapped back to natural units and normalised to length one) with
  absolute value above 0.2, printed as `"+0.71 sigma_p, -0.52 sigma_hpwt, -0.40 sigma_const"`.
- verdict: `UNITS` if `s_l/s_1 < 1e-6` in `A_nat` and the matching direction has `s/s_1 >= 1e-4` in
  `A_jac` (rescaling removes it; suggested new unit = the column norm ratio); `IDENTIFICATION` if
  `s/s_1 < 1e-6` in `A_jac`; `WEAK` if between `1e-6` and `1e-4` in `A_jac`.
- named directions: the spec may declare known directions as vectors in natural units (for Petrin:
  `scale_direction` = +sigma_p, -all sigma_x in proportion (LEARNINGS 1); `body_level_direction` =
  +sigma_body and +pi_body in proportion (LEARNINGS 14)). A flat direction with |cosine| above 0.8 to
  a declared one is reported under that name.

Message: `"Flat direction {l}: {verdict} (s/s1 = {r:.1e} Jacobi, {rn:.1e} natural; 1 unit moves q by
{dq:.2f}); loads {loadings}; matches {name}."`

### 5.3 Collinearity and weak instruments

- X1, Z: rank and condition number after standardising columns (SVD, rank tolerance 1e-10 of
  `s_1`). Flag `cond > 1e8` or rank deficiency.
- Z'X1: rank must equal the number of linear parameters; report the smallest canonical correlation
  between Z and X1. Fail on rank deficiency (this is the check that catches F1).
- Columns of `A_jac`: list pairs with |correlation| above 0.99.
- Weak identification: Cragg-Donald-style minimum eigenvalue statistic on the moment Jacobian,
  `CD = n lambda_min(Gs' S^{-1} Gs)` with `Gs = G diag(1 / sd_k)` and `sd_k` the column scale, compared
  with the chi-square(k - P + 1) 95% critical value (Wright 2003 rank test, here without the sampling
  variance of G, which the report states). Below the critical value: `WEAK_IDENTIFICATION`.

### 5.4 Named degeneracies

Each detector reads the final point and the accepted-step history (columns of `history.csv`).

| name | rule (defaults) | message |
|---|---|---|
| `SCALE_ARTIFACT` (LEARNINGS 1) | over the run `q_fixed` falls by a factor above 10 while `q_cue` changes by a factor below 2, and `sd(xi)` falls by a factor above 2 | "q fell with sd(xi)^2 while the scale-free objective did not: the fixed-W objective is rewarding a small xi through the utility scale." |
| `SD_XI_BLOWUP` (LEARNINGS 13) | q falls by more than 50% while sd(xi) rises by a factor above 2; or kurtosis(xi) > 10; or the top 1% of products carry more than 25% of sum xi^2 | "q fell while xi grew and became lumpy: the objective is being lowered by noisy moments, not by fit." |
| `RUN_TO_BOUND` | a non-cap parameter within 1e-3 units of a bound with outward gradient | "{name} ended at its bound {b}; the objective wants to go further." |
| `NORMALISATION_CAP_ACTIVE` | a cap parameter at its cap | "{name} is at its cap {b}, a stated normalisation of a direction the data leave quiet." |
| `RUN_TO_INFINITY` | over the last 10 accepted steps |theta_k| rose monotonically and is above 5 times its start in natural units, while q fell by less than 10% | "{name} is running away ({v0} to {v1}) with little gain in q." |
| `SEGMENTED_CORNER` (LEARNINGS 13) | for a taste spread on a binary column, the share of consumers with |taste| above the price range, `2 (1 - Phi(range / sigma))`, exceeds 0.2; or cross-group diversion below 1% | "{name} = {v} splits consumers into separate {column} pools ({pct}% have taste beyond the price range)." |
| `VERTICAL_COLLAPSE` | all non-cap taste spreads below 0.05 natural units, or median nonzero cross-elasticities per product <= 2, or median own elasticity below -50 | "Taste spreads collapsed: the model is close to vertical (median own elasticity {el})." |

### 5.5 Overidentification

Hansen `J = n g' S^{-1} g` at the final point with S from the same point,
`dof = k - (P_free + K_linear)`; for Petrin `45 - (18 + 13) = 14`, minus any parameter at a cap. The
p-value is reported. J is computed only for two-step and CUE runs; for a fixed-W step 1 the report
says why it is not a J statistic.

### 5.6 Economic sanity (`economics.py`, uses `models/pcm_post.py`)

At the final point, per product and summarised by median, 5th and 95th percentile:
- own-price elasticity `e_jj = (ds_j/dp_j) p_j / s_j`;
- number of nonzero cross-elasticities per product (`|e_jk| > 1e-10`) and the ratio to `J_t - 1`;
- diversion `D_jk = -(ds_k/dp_j) / (ds_j/dp_j)`, outside diversion `1 - sum_k D_jk`, and the share of
  inside diversion that stays within the product's group (spec-declared groups; Petrin: body type
  `mi, sw, su, pv, car`);
- margins `(p - c)/p` from the multi-product Nash first-order condition with `firm_ids`; flagged if
  any market's ownership-weighted ds/dp has rcond below 1e-12 (`NASH_JACOBIAN_SINGULAR`, which stops
  the counterfactual), or if any margin is negative or above 1;
- inside share semi-elasticity to a common 1% price rise;
- taste spreads converted to buyer dollars (divide by the mean alpha among predicted buyers,
  LEARNINGS 11), compared with the price range.

Each statistic has a plausible range in the spec (`spec.plausible`). Out-of-range values are flags
with the template `"{stat} = {v} is outside the declared plausible range [{lo}, {hi}]."` Ranges for
Petrin are in section 6.6.

### 5.7 Integration stability

At the final point: re-evaluate with draws doubled (`nu_per_agent x 2`, new seeds). Report the change
in q and run at most 20 optimizer iterations from the estimate with the doubled draws. Report each
parameter change in SE units when SEs are available, else in natural units. `STABLE` if every change
is below 0.5 SE (or 0.1 natural units), else `UNSTABLE` with the parameters listed. Same for the
median own elasticity and the counterfactual headline.

### 5.8 Multi-start (optional)

`--multistart K`: K starts from a scrambled Sobol design in the box start +- 50% of each parameter's
natural unit times 3, run as K independent runs (separate run directories), each with
`192 / min(K, 4)` threads, at most 4 at a time. Report a table of endpoints, q, class; endpoints
closer than 0.1 natural units in every parameter count as the same; report the number of distinct
endpoints and the q spread.

### 5.9 Standard errors

Computed only if all hold: class CONVERGED or CONVERGED_AT_CAP; `rc(A_jac) >= 1e-6`; no `RUN_TO_BOUND`
or `RUN_TO_INFINITY`. Cap parameters are treated as fixed and get no SE (stated). Sandwich over
(beta, theta) with the full Jacobian of 2.4:
`V = (G'WG)^{-1} G'W S W G (G'WG)^{-1} / n`, which reduces to `(G'S^{-1}G)^{-1}/n` at the two-step
estimate. Clustered by `clustering_ids`. If any condition fails: `"Standard errors omitted:
{reason}."`

### 5.10 Report format

JSON (`report.json`, validated against a schema in `report.py` with a version number):

```json
{
  "schema_version": 1, "run_id": "...", "spec": {"name": "petrin_pcm", "hash": "...", "normalisations": ["..."]},
  "stage": "step2", "created": "2026-11-16T22:00:00Z", "code_commit": "...",
  "convergence": {"class": "CONVERGED", "projected_gradient": 1e-4, "pg_tol": 5.7e-3,
                   "gn_decrease": 1e-8, "gn_tol": 3.2e-5, "accepted_steps": 41, "evaluations": 57,
                   "wall_seconds": 3600, "kink_limited_steps": 6, "message": "..."},
  "objective": {"q": 31.2, "parts": {"aggregate": 20.1, "micro": 11.1}, "weighting": "two_step",
                "n": 2407, "k": 45, "p_free": 18, "k_linear": 13},
  "parameters": [{"name": "sigma_sw", "value": 4.1, "unit": 1.0, "lower": 0, "upper": 66.1,
                  "cap": true, "at_bound": false, "se": 0.9, "note": "..."}],
  "linear": [{"name": "const", "value": -8.9, "se": 1.4}],
  "identification": {"rc_natural": 1e-5, "rc_jacobi": 1e-3, "directions": [{"index": 18,
                     "verdict": "WEAK", "dq_per_unit": 0.4, "loadings": {"sigma_p": 0.7},
                     "named": "scale_direction", "message": "..."}]},
  "collinearity": {"X1": {}, "Z": {}, "ZX1": {}, "G_pairs": []},
  "weak_identification": {"statistic": 0.0, "dof": 0, "critical_95": 0.0, "verdict": "ok"},
  "hansen_j": {"J": 0.0, "dof": 14, "p_value": 0.0},
  "degeneracies": [{"name": "SCALE_ARTIFACT", "detected": false, "evidence": {}, "message": ""}],
  "economics": {"own_elasticity": {"median": -4.1, "p05": -9, "p95": -2}, "flags": []},
  "integration": {"status": "STABLE", "dq": 0.3, "changes": {}},
  "multistart": null,
  "standard_errors": {"computed": true, "reason": null},
  "files": {"history": "history.csv", "checkpoint": "checkpoint.npz"}
}
```

Text rendering (`report.txt`) prints the same content in fixed order: a headline line
(`RESULT: CONVERGED at q = 31.2 (two-step, 45 moments, 31 parameters). 2 flags.`), then one section
per block with its message lines first and the numbers after, flags marked `FLAG:` and failures
`FAIL:`. Nothing in the text is absent from the JSON.

## 6. Petrin proof of concept

### 6.1 Data

pyblp 1.3.0 Petrin data (`pyblp.data.PETRIN_PRODUCTS_LOCATION`, `PETRIN_AGENTS_LOCATION`,
`PETRIN_VALUES_LOCATION`): 2,407 products in 13 markets (1981-1993, 148 to 210 products), 898
clusters, 27 firms; 1,000 CEX agents per market with weight 0.001, demographics `income, low, mid,
high, fs, age, fv`. Prices in $1000 range from 3.39 to 69.49. Prices are spread to a $10 minimum gap
within each market (`data.break_price_ties`, LEARNINGS 3).

Consumers: each agent is crossed with `nu_per_agent` scrambled Sobol standard normal draws over the
taste dimensions (seed = base seed + market id), weight `0.001 / nu_per_agent`. Development runs use
`nu_per_agent = 4`, final runs 16, and the stability check doubles it. pyblp's `nodes0..5` columns are
not used (section 8, question 4).

### 6.2 Specification

```
u_ijt = delta_jt + x_jt . t_i - c_i a_i p_jt,   u_i0t = 0
x  = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv]
t_i = sigma_k nu_ik                on all 10 columns (body sigmas on mi, sw, su, pv; LEARNINGS 4)
    + pi_b log(fs_i) fv_i          on mi, sw, su, pv
log c_i = phi_low low_i/income_i + phi_mid mid_i/income_i + phi_high high_i/income_i
a_i ~ LogNormal(0, sigma_p)
delta_jt = X1_jt beta + xi_jt,  X1 = [1, hpwt, space, air, mpd, fwd, mi, sw, su, pv, pgnp, trend, trend2]
Z = [X1, demand_instruments0..21]  (35 columns, F1)
```

| parameter | start | lower | upper | unit | note |
|---|---|---|---|---|---|
| sigma_const | 3.0 | 0 | 661 | 1 | upper = 10 x price range, domain guard |
| sigma_hpwt, sigma_space, sigma_air, sigma_mpd, sigma_fwd | 2.0, 1.0, 0.5, 1.0, 1.0 | 0 | 661 / sd(x) | auto | domain guard |
| sigma_mi, sigma_sw, sigma_su, sigma_pv | 2.0 each | 0 | 66.1 | 1 | cap = price range, normalisation |
| pi_mi, pi_sw, pi_su, pi_pv (x log(fs) fv) | 0.3 each | -inf | inf | auto | |
| phi_low, phi_mid, phi_high | from pre-fit, else 0 | -inf | inf | auto | |
| sigma_p | 1.0 | 0.05 | 3.0 | 0.1 | |

Normalisations stated in the report: median alpha = 1 at `c_i = 1` (mu = 0; LEARNINGS 11, Song 2008
precedent in LEARNINGS 12); body sigmas capped at the price range `max p - min p = 66.1` ($1000);
taste spreads non-negative; `u_i0 = 0` in every year with pgnp and trends in X1 (harmless for the
1984 counterfactual, LEARNINGS 12b).

Start point: the table above, then a pre-fit of (phi_low, phi_mid, phi_high) alone, all else fixed
(`free=False`), on the two income micro moments plus the aggregate block with W1, at most 30
iterations. This puts the price-income gradient near the CEX purchase rates before the full run. The
pre-fit is a stage of its own and is checkpointed.

### 6.3 Moments

Aggregate: `g_agg = Z' xi / n`, 35 rows.

Micro: Petrin's 10 CEX moments, as defined in `petrin_replication/estimate_blp.py`
`build_micro_moments()`: one `MicroDataset` "CEX" with `observations = 29125`, all markets, weights
one for every product and the outside good. In pyblp's notation the parts are
`compute_values = outer(demographics[:, d], r_[outside, X2[:, x]])`; X2 columns 7..10 are
`mi, sw, su, pv`; demographic columns are 5 `age`, 6 `fs`, 7 `mid`, 8 `high` (`age` is a 0/1
indicator in the agent data). Values from `PETRIN_VALUES_LOCATION`:

| moment | parts (numerator / denominator) | value |
|---|---|---|
| E[age_i \| mi_j] | E[age_i mi_j] / E[mi_j] | 0.7830 |
| E[age_i \| sw_j] | E[age_i sw_j] / E[sw_j] | 0.7300 |
| E[age_i \| su_j] | E[age_i su_j] / E[su_j] | 0.7400 |
| E[age_i \| pv_j] | E[age_i pv_j] / E[pv_j] | 0.6520 |
| E[fs_i \| mi_j] | E[fs_i mi_j] / E[mi_j] | 3.8600 |
| E[fs_i \| sw_j] | E[fs_i sw_j] / E[sw_j] | 3.1700 |
| E[fs_i \| su_j] | E[fs_i su_j] / E[su_j] | 2.9700 |
| E[fs_i \| pv_j] | E[fs_i pv_j] / E[pv_j] | 3.4700 |
| E[1{j>0} \| mid_i] | E[1{j>0} mid_i] / E[mid_i] (outside value 1 in the denominator) | 0.0794 |
| E[1{j>0} \| high_i] | E[1{j>0} high_i] / E[high_i] | 0.1581 |

In the kernel's terms (section 2.3): h in {1, age, fs, mid, high}, u in {mi, sw, su, pv, 1}, outside
value 0 for all numerators and the body denominators, 1 for `E[mid_i]` and `E[high_i]`, which are
therefore constants `sum_t H_t / Den`. Every market enters with equal weight because agent weights
sum to one per market, exactly as in pyblp. The micro block of S uses the part products of section
2.3 (h in {age, fs, mid, high, age fs, ...} as needed; age^2 = age since it is binary).

Moment count 45, unknowns 31, overidentifying restrictions 14.

### 6.4 Stages (`petrin_replication/estimate_pcm.py run --stage ...`)

1. `profile`: sigma_p ray profile with micro moments (the first check, LEARNINGS 14). Weighting W1 at
   the start point. Grid: rays `sigma = lambda x SIG_REF` with SIG_REF = (const 2, hpwt 2, space 2,
   air 1, mpd 1, fwd 2, body 2 each) in $1000, lambda in {0.5, 1, 2}; sigma_p in
   {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}; pis and phis at the pre-fit values. Each point is inverted cold
   (as in `profile_sigma_ray.py`) and records `q`, `q_agg`, `q_micro`, `q_cue`, sd(xi), median own
   elasticity, inside share by income group. Grid points run in parallel across 4 workers with 48
   threads each. Output `profile_sigma_ray.csv` and a verdict.
   **Pass:** for every lambda, q is not monotone decreasing in sigma_p over the grid: either the
   minimum over the grid is interior, or `q(3.5) - min q >= 3.84`. The report also shows whether
   `q_micro` rises with sigma_p (the mechanism LEARNINGS 14 predicts: purchase rates flatten across
   income). **Fail:** the run stops with `"Profile check failed: q is still monotone in sigma_p with
   micro moments. LEARNINGS 14 predicted otherwise; the plan needs the owner before any long
   estimation."` Steps 2-5 do not start.
2. `prefit`: section 6.2.
3. `step1`: fixed W1, from the pre-fit point.
4. `step2`: W2 = S(theta_hat_1)^{-1} (gated as in 4.5), from theta_hat_1.
5. `post`: diagnostics at the final point, postestimation (6.5), comparison with BLP, stability check.
`--stage all` runs 1 to 5 in order; each completed stage is skipped on resume.

### 6.5 Postestimation and the 1984 counterfactual (`models/pcm_post.py`)

At the final point, with `nu_per_agent` as estimated:
1. Elasticities, diversion, margins (5.6), per product, written to `product_level.csv`.
2. Costs in 1984: `c = p - markup`, `markup = -(Omega * (ds/dp)')^{-1} s` with Omega the same-firm
   indicator.
3. Counterfactual: drop the products with `mi = 1` in 1984; delta and xi of the remaining products
   unchanged; prices solved by the damped fixed point `p <- p + 0.5 (c + markup(p) - p)` with prices
   re-sorted and ties spread each iteration; stop when the step is below 1e-7, at most 500 iterations;
   non-convergence is reported, not hidden.
4. Consumer surplus with `pcm_welfare::consumer_surplus` at the observed and counterfactual
   equilibria; gain = `households_1984 x (CS_0 - CS_1)` in $ million (CS is in $1000 per household).
   Also by alpha quintile and family size (LEARNINGS 12c).
5. Comparison table against `petrin_replication/results/blp_micro_summary.json` and
   `blp_nomicro_summary.json` (minivan CS gain 1984: BLP micro $425.9M, BLP no micro $365.0M; median own
   elasticity -4.48 and -3.78; median margin 0.243 and 0.300). The BLP fits estimate supply jointly;
   the PCM uses demand-side moments and FOC costs only. The table says so (section 8, question 5).

### 6.6 Plausible ranges declared in the Petrin spec

| statistic | range |
|---|---|
| median own-price elasticity | [-12, -2] |
| share of products with own elasticity below -50 | <= 0.01 |
| median nonzero cross-elasticities per product | >= 10 |
| median share of inside diversion leaving the body type | >= 0.05 |
| median margin | [0.05, 0.60] |
| body sigmas in buyer dollars | <= price range |
| sd(xi) / sd(delta) | <= 1 |

These are the owner's to change; they only trigger flags.

### 6.7 Nevo sketch (generality)

```python
spec = Spec(
    name="nevo_pcm", data=load_pyblp("NEVO"),
    characteristics=["const", "sugar", "mushy"],
    taste=[Taste("sigma_const", "const", "nu0"), Taste("sigma_sugar", "sugar", "nu1"),
           Taste("sigma_mushy", "mushy", "nu2"),
           Taste("pi_const_income", "const", "income"), Taste("pi_sugar_age", "sugar", "age"), ...],
    scale=[Scale("phi_income", "income_dm"), Scale("phi_income2", "income2_dm"), Scale("phi_child", "child_dm")],
    linear=Absorb("product_ids"),                          # brand fixed effects, X1 empty
    moments=[IVMoments(Z=demean(demand_instruments0..19, by="product_ids"))],
    draws=AgentsTimesGaussHermite(per_market=20, nodes=5),
    normalisations=["median alpha = 1 at mean demographics (scale bases demeaned)"],
    plausible={...},
)
```

No new estimation code: the same model adapter, optimizer, diagnostics and report.

## 7. Work packages

Each package is sized for one coder agent. "Owns" lists every file the package may create or edit.
No two packages that run in parallel own the same file. Every package must read
`LEARNINGS.md` and this file first, and run its acceptance tests before
committing on `feat/estimation-harness`.

**Wave 0 (serial).**

**WP0 Scaffold.** Owns: root `CMakeLists.txt` (install line only if needed),
`demand_model/CMakeLists.txt`, `python_bindings/CMakeLists.txt`, `.gitignore`,
`python_bindings/pcm_kernel_python.cpp`, stub versions of every new file in section 1.1, and
`nfp_harness/types.py` in full (sections 2.1, 2.2, 2.3, 2.5 dataclasses, `KernelRequests`,
`OptimizeResult`, `HistoryRecord`, `Report` skeleton). Stubs raise `NotImplementedError` with the
signature and docstring from this document. Acceptance: the build command of 1.2 succeeds;
`python -c "import nfp_harness._native as n; print(n.pcm_kernel.__doc__)"`
works from the repo root; `pytest nfp_harness/tests` collects with all tests skipped; existing
gtests still pass. After WP0, ownership of each stub passes to the package below and `types.py` is
frozen.

**Wave A (parallel, after WP0).**

**WP1 Kernel solve.** Owns: `demand_model/pcm_kernel.{hpp,cpp}`, `demand_model/pcm_kernel_test.cpp`,
`python_bindings/bind_solve.cpp`, `demand_model/pcm_demand.{hpp,cpp}` (only the move of
`frontier_lift` and `cond_vertical` to `pcm_demand::detail`). Implements 3.2 and 3.5's `n_threads`.
Acceptance: tests 3.6 kernel 1-6; `pcm_demand_test` unchanged and passing; binding returns a dict
with every `SolveResult` field and releases the GIL (a Python test runs two markets in threads and
sees wall time below 0.7 of serial).

**WP2 Kernel events.** Owns: `demand_model/pcm_events.{hpp,cpp}`, `demand_model/pcm_events_test.cpp`,
`python_bindings/bind_events.cpp`. Implements 3.3 with its own hull code (does not edit
`pcm_demand.cpp`). Acceptance: tests 3.6 events 1-9; cost of one `max_step` call on a 210-product,
16,000-draw market below 2 times one `shares(..., false)` call.

**WP3 Kernel welfare.** Owns: `demand_model/pcm_welfare.{hpp,cpp}`,
`demand_model/pcm_welfare_test.cpp`, `python_bindings/bind_welfare.cpp`. Implements 3.4. Acceptance:
tests 3.6 welfare 1-3; matches `consumer_surplus` in the local `pcm_postestimation.py` to 1e-10 on a
random 30-product, 500-draw market.

**WP4 Python core.** Owns: `nfp_harness/params.py`, `linear.py`, `moments.py`, `problem.py`,
`weighting.py`, `data.py`, `tests/test_params.py`, `tests/test_moments.py`, `tests/test_problem.py`,
`tests/test_weighting.py`. Implements sections 2.2-2.5 and the weighting algebra of 4.5 (S,
micro covariance, W1, W2, CUE terms). Tests use a pure-numpy fake `InnerModel` with known
delta(theta) and D. Acceptance: chain-rule G matches central finite differences to 1e-7; IV block
with `Z = [X1, Z_excl]` reproduces 2SLS beta; ratio micro moment gradient matches finite differences;
micro covariance matches a hand calculation on a 3-agent, 2-product toy and pyblp's formula
(observations scaling, `n V` block).

**WP5 Optimizer and checkpoint.** Owns: `nfp_harness/optimizer.py`, `checkpoint.py`, `runtime.py`,
`diagnostics/convergence.py`, `tests/test_optimizer.py`, `tests/test_checkpoint.py`. Implements
section 4 and the class rules of 4.6. Uses only `types.py` and a toy `Problem`. Acceptance: tests
4.8 1-10.

**WP6 Diagnostics.** Owns: `nfp_harness/diagnostics/__init__.py`, `identification.py`,
`collinearity.py`, `degeneracies.py`, `overid.py`, `stability.py`, `inference.py`, `report.py`,
`tests/test_diagnostics.py`. (Not `convergence.py`, which WP5 owns, and not `economics.py`.)
Implements 5.2-5.5, 5.7-5.10 on synthetic `Evaluation`s and histories. Acceptance: planted cases are
detected and named: a G with a column scaled by 1e-7 gives verdict UNITS; a G with two identical
columns gives IDENTIFICATION with the right loadings; Z without X1 fails the Z'X1 rank check; a
history with q falling as sd(xi)^2 gives SCALE_ARTIFACT; one with sd(xi) rising 5x gives
SD_XI_BLOWUP; JSON validates against the schema; text rendering contains every flag.

**Wave B (after WP1, WP2, WP4).**

**WP7 PCM model adapter.** Owns: `nfp_harness/models/__init__.py`, `models/pcm.py`,
`tests/test_pcm_model.py`. Builds markets and draws from a spec, implements `InnerModel` (solve via
`solve_market`, micro requests, thread allocation of 3.5, commit/restore, `state_dict`, `max_step`
via WP2 and `frontier_signature` guard of 4.4). Acceptance: on a synthetic 2-market, 8-product,
256-draw problem with micro moments, total G matches central finite differences of `evaluate` to
1e-5 relative; results identical with 1 and 16 threads; one Petrin evaluation at `nu_per_agent = 1`
converges in all 13 markets and its aggregate shares and deltas match `python_pcm_demand.invert` at
the same theta to the inversion tolerance.

**Wave C (after WP3 and WP7; WP8 and WP9a in parallel).**

**WP8 PCM postestimation and economics.** Owns: `nfp_harness/models/pcm_post.py`,
`diagnostics/economics.py`, `tests/test_pcm_post.py`. Implements 5.6 and 6.5. Acceptance: on the
saved local estimate `petrin_replication/results/pcm_estimates_body_nu16.npz` (run with the old
spec, so only as a numeric fixture) the elasticities and margins match the local
`pcm_postestimation.py` output to 1e-8; the counterfactual reports the singular Nash Jacobian as
`NASH_JACOBIAN_SINGULAR` if it occurs, rather than failing.

**WP9a Profiles.** Owns: `nfp_harness/profile.py`, `tests/test_profile.py`. Generic ray and grid
profiles over named parameters with parallel workers, cold or warm inversion, CSV output and the pass
rule as a callable. Acceptance: on a toy problem the profile matches direct evaluations; on Petrin at
`nu_per_agent = 1` a 2 x 2 grid runs and writes the CSV.

**Wave D (after WP5, WP6, WP8, WP9a).**

**WP9 Petrin spec and run.** Owns: `petrin_replication/estimate_pcm.py`. Writes the spec of 6.1-6.6
and the CLI (`run --stage profile|prefit|step1|step2|post|all`, `--nu`, `--detach`, `--new-run`,
`--multistart K`, `status`). Acceptance, in order: (1) `--stage profile --nu 4` completes and its
verdict is in `report.txt`; if it fails, WP9 stops and reports to the owner; (2) `--stage all --nu 4`
runs detached, survives a manual `kill -9` and resumes; (3) the step-2 report exists with every
block present; (4) the comparison table against both BLP summaries is printed. Whether the estimate
is economically acceptable is the owner's call, not an acceptance criterion.

**WP10 Nevo spec (after WP9, optional).** Owns: `nevo_replication/estimate_pcm.py`. Implements 6.7.
Acceptance: one-step estimate runs end to end and produces a report.

Parallelism summary: WP0 alone; then WP1, WP2, WP3, WP4, WP5, WP6 in parallel; then WP7; then WP8
and WP9a in parallel; then WP9; then WP10.

## 8. Risks and open questions

1. **F1 changes the reading of earlier results.** The fixed-W and CUE PCM runs were under-identified
   by nine directions. The mechanisms in LEARNINGS 1 and 13 may still hold, but their size was
   measured on an estimator with 22 moments for 31 unknowns. The owner may want a LEARNINGS entry and
   a rerun of the old profile with `Z = [X1, Z_excl]` before reading WP9's profile.
2. **Cap value.** The requirements said "capped at the price range"; LEARNINGS 13 says "say $50k". The
   pooled range is $66.1k; per-year ranges are smaller. This design uses the pooled 66.1. Owner to
   confirm.
3. **max_step at 16,000 draws.** `tau_first` will be near zero, so the cap rests on the affected-mass
   rule with an adaptive threshold. If realised mass routinely exceeds the prediction, the cap
   degrades to the box radius and `max_step` adds little. The logged predicted-versus-realised mass
   will show this in the first runs.
4. **Draws differ from BLP.** pyblp's agent data carries `nodes0..5` (mean 2.6, range 0 to 7.8, not
   standard normal) that the BLP fit uses; the PCM draws its own Sobol normals. For a like-for-like
   comparison the owner may prefer the pyblp nodes (transformed or as is). Open.
5. **Supply side.** The BLP fits estimate a log-cost supply side jointly (16 supply instruments). The
   PCM uses demand moments only and recovers costs from the FOC. A supply block can be added as a
   moment object later; whether to do it before comparing headline numbers is the owner's decision.
6. **Two-step from a degenerate step 1.** If step 1 lands at a scale or segmented corner, W2 encodes
   it. The gate in 4.5 stops the run instead; the owner decides what to change.
7. **sigma_p domain.** The old code bounded sigma_p at 2 because inversions failed above it; the
   profile in LEARNINGS 1 inverted cold at 3.5. This design uses 3.0 for estimation and 3.5 in the
   profile. Inversion failures above 2 will appear as `INVERSION_FAILED` trials.
8. **Weak-identification statistic.** The Cragg-Donald-style statistic ignores the sampling variance
   of G, so it is a screen, not a test with exact size. A Kleibergen-Paap-type version needs the
   covariance of vec(G), which requires per-cluster derivatives of every moment. Deferred.
9. **CUE with micro moments.** The micro block of S is held fixed in CUE mode. Exact CUE with micro
   moments would need the derivative of the model-implied part covariances. Not needed for Petrin.
10. **Sobol sizes.** 1,000 agents x 4 draws gives 4,000 points per market, not a power of two, so
    scipy warns about balance. Crossing fixed agent rows with a power-of-two sequence per agent is an
    alternative; it changes results and is left to the owner.
11. **Oversubscription.** 13 OpenMP teams plus OpenBLAS threads on 192 cores. The environment lines
    in 1.3 address it; WP7 measures CPU utilisation in its Petrin smoke test.
12. **Unavailable tool in this session.** The `chorus` MCP server failed to start; it was not needed.

## 9. Owner decisions on the open questions (2026-11-16)

1. F1 is confirmed independently (rank 22 vs 35) and recorded as LEARNINGS 16. WP9 also runs the
   sigma_p ray profile without micro moments but with Z = [X1, Z_excl], so the effect of the full
   instrument set and of the micro moments are reported separately.
2. Body-sigma cap: the pooled price range, $66.1k. One primitive number, fixed across years and
   counterfactuals, reported as a normalisation.
3. PCM keeps its own scrambled Sobol standard normals for tastes, crossed with pyblp's agent
   demographics. pyblp's nodes are not used for the PCM. The difference is stated in the write-up.
4. No PCM supply block. `petrin_replication/estimate_blp.py` adds a demand-only BLP fit (same
   moments as the PCM: aggregate + micro, no supply), reported beside the two existing BLP columns,
   so the headline comparison is like for like.
5. sigma_p upper bound 3.0. Hitting it is reported by the bound detector, not hidden.
6. Draws per market are powers of two: 4,096 for development, 16,384 for the reported run, 32,768
   for the draws-doubling check.
7. The plausible ranges in 6.6 are accepted.
8. CUE may hold the micro covariance block fixed; CUE is not used for the Petrin headline.
