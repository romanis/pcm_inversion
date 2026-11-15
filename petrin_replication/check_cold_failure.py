"""Cold inversion at large taste heterogeneity via continuation in sigma_x: scale the taste
parameters by lam from small to 1, warm-starting each solve from the previous one."""
import os, sys, re, time, numpy as np, pandas as pd, pyblp, warnings; warnings.filterwarnings('ignore')
os.environ["PCM_BODY_SIGMA"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcm_petrin as pp, python_pcm_demand as c
P = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION); A = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
L = open(os.path.join(pp.OUT, 'pcm_estimation_cue_nu4.log')).read()
T = [np.array([float(v) for v in s.split()]) for s in re.findall(r'theta=\[([^\]]*)\]', L)]
t1 = T[1]
nu = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ms = pp.build_markets(P, A, nu)
th1, ph1, sp1 = pp.split(t1)
for m in ms:
    if m.market_id not in (1984, 1989, 1990, 1991, 1992, 1993):
        continue
    t0 = time.time(); d = np.empty(0); ok = True; ev = 0; path = []
    for lam in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0):
        r = c.invert(m.shares, m.cpp, th1 * lam, ph1, sp1, d, 1e-10, 6000)
        ev += r['share_evaluations']; path.append(f"{lam}:{'ok' if r['converged'] else 'X'}({r['share_evaluations']})")
        if not r['converged']:
            ok = False; break
        d = r['delta']
    print(f"{m.market_id}: {'converged' if ok else 'FAILED'} total evals {ev} {time.time()-t0:.0f}s  {' '.join(path)}", flush=True)
