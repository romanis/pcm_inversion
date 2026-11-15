"""Profile of the GMM objective along rays in sigma-space, under two weightings.

  q_fixed : n g' (Z'Z/n)^-1 g            (step-0 weighting; quadratic in the scale of xi)
  q_cue   : n g' S(theta)^-1 g           (continuously updated, clustered S; scale-free)

Ray: sigma = lam * SIG_REF, SIG_REF in $1000 (const 2, hpwt 2, space 2, air 1, mpd 1, fwd 2, body
types 2), demographic pis and phis at the nu16 estimate, repeated for several sigma_p.
Also records sd(xi) (the scale) and the median own-price elasticity.
"""
import os, sys, time, numpy as np, pandas as pd, pyblp, warnings; warnings.filterwarnings('ignore')
os.environ["PCM_BODY_SIGMA"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pcm_petrin as pp, python_pcm_demand as c

P = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION); A = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)
NU = int(os.environ.get("NU", "4"))
prob = pp.PCMProblem(P, A, nu_per_agent=NU, path_seconds=45)
base = pd.read_csv(os.path.join(pp.OUT, 'pcm_estimates_body_nu16.csv'), index_col=0).estimate
cluster = pd.factorize(P.clustering_ids.to_numpy())[0]
SIG_REF = {"sigma_const": 2.0, "sigma_hpwt": 2.0, "sigma_space": 2.0, "sigma_air": 1.0, "sigma_mpd": 1.0, "sigma_fwd": 2.0,
           "sigma_mi": 2.0, "sigma_sw": 2.0, "sigma_su": 2.0, "sigma_pv": 2.0}
LAMS = [float(v) for v in os.environ.get("LAMS", "0.02,0.1,0.25,0.5,1.0,2.0,3.0").split(",")]
SPS = [float(v) for v in os.environ.get("SPS", "1.0,2.0,3.5").split(",")]


def clustered_S(xi):
    zxi = prob.Z * xi[:, None]
    S = np.zeros((prob.Z.shape[1],) * 2)
    for k in np.unique(cluster):
        v = zxi[cluster == k].sum(0)
        S += np.outer(v, v)
    return S / prob.n


rows = []
for sp in SPS:
    for m in prob.markets:
        m.delta = None; m.theta_ok = None
    for lam in LAMS:
        th = base.to_numpy().copy()
        for n, v in SIG_REF.items():
            th[pp.NAMES.index(n)] = lam * v
        th[pp.NAMES.index("sigma_p")] = sp
        t0 = time.time()
        q, g = prob.objective(th)
        ok = prob.last["ok"]
        rec = dict(sigma_p=sp, lam=lam, ok=ok, q_fixed=q if ok else np.nan, q_cue=np.nan, sd_xi=np.nan, med_own_el=np.nan, sec=np.nan)
        if ok:
            xi = prob.last["xi"]
            gm = prob.Z.T @ xi / prob.n
            S = clustered_S(xi)
            rec["q_cue"] = float(prob.n * gm @ np.linalg.solve(S, gm))
            rec["sd_xi"] = float(xi.std())
            own = []
            for m in prob.markets:
                d = c.shares(prob.last["delta"][m.rows], m.cpp, *pp.split(th), True)
                own.append(np.diag(d["d_price"]) * m.cpp.p / d["s"])
            rec["med_own_el"] = float(np.median(np.concatenate(own)))
        rec["sec"] = time.time() - t0
        rows.append(rec)
        print(f"sigma_p={sp} lam={lam}: q_fixed={rec['q_fixed']:.4f} q_cue={rec['q_cue']:.3f} sd_xi={rec['sd_xi']:.4f} "
              f"median own el={rec['med_own_el']:.1f} ok={ok} {rec['sec']:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(pp.OUT, f"profile_sigma_ray_nu{NU}.csv"), index=False)
