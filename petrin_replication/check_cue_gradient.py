"""Finite-difference check of the CUE objective gradient, and a start file for the CUE run."""
import os, sys, numpy as np, pandas as pd, pyblp, warnings; warnings.filterwarnings('ignore')
os.environ["PCM_BODY_SIGMA"] = "1"; os.environ["PCM_CUE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcm_petrin as pp
P = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION); A = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
prob = pp.PCMProblem(P, A, nu_per_agent=4, path_seconds=45)
base = pd.read_csv(os.path.join(pp.OUT, 'pcm_estimates_body_nu16.csv'), index_col=0).estimate
th = base.to_numpy().copy()
for n, v in {"sigma_const": 4.0, "sigma_hpwt": 4.0, "sigma_space": 4.0, "sigma_air": 2.0, "sigma_mpd": 2.0, "sigma_fwd": 4.0,
             "sigma_mi": 4.0, "sigma_sw": 4.0, "sigma_su": 4.0, "sigma_pv": 4.0, "sigma_p": 2.0}.items():
    th[pp.NAMES.index(n)] = v
pd.DataFrame({"estimate": th}, index=pp.NAMES).to_csv(os.path.join(pp.OUT, "pcm_start_cue.csv"))
q, g = prob.objective(th); q, g = prob.objective(th)
for k in [0, 6, 10, 14, 17]:
    h = 1e-4 * max(1, abs(th[k])); e = np.zeros_like(th); e[k] = h
    qp, _ = prob.objective(th + e); qm, _ = prob.objective(th - e)
    print(f"{pp.NAMES[k]:18s} analytic {g[k]:10.4f}  fd {(qp - qm) / (2 * h):10.4f}", flush=True)
