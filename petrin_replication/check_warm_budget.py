"""How many share evaluations does a warm-started inversion need across one trust-region step?"""
import os, sys, re, time, numpy as np, pandas as pd, pyblp, warnings; warnings.filterwarnings('ignore')
os.environ["PCM_BODY_SIGMA"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcm_petrin as pp, python_pcm_demand as c
P = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION); A = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
L = open(os.path.join(pp.OUT, 'pcm_estimation_cue_nu4.log')).read()
T = [np.array([float(v) for v in s.split()]) for s in re.findall(r'theta=\[([^\]]*)\]', L)]
t0, t1 = T[0], T[1]
print("step:", np.round(t1 - t0, 3))
ms = pp.build_markets(P, A, 4)
for m in ms:
    if m.market_id not in (1984, 1990, 1992, 1993):
        continue
    d0 = c.invert(m.shares, m.cpp, *pp.split(t0), np.empty(0), 1e-10, 40000)['delta']
    for tol in (1e-10, 1e-8):
        t = time.time()
        r = c.invert(m.shares, m.cpp, *pp.split(t1), d0, tol, 60000)
        print(f"{m.market_id} tol={tol:g}: conv {r['converged']} resid {r['max_abs_residual']:.1e} "
              f"evals {r['share_evaluations']} iters {r['iterations']} {time.time()-t:.0f}s", flush=True)
