"""Profile per-market inversion times to identify slow markets.

Run from repo root:
  PYTHONPATH=nevo_replication/.vendor:build-rel2/python_bindings \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/istomin/anaconda3/bin/python petrin_replication/profile_market_times.py
"""
import pathlib
import time
import numpy as np
import pandas as pd
import pyblp
from pcm_petrin import PCMProblem

OUT = pathlib.Path(__file__).resolve().parent / "results"

theta0 = pd.read_csv(OUT / "pcm_start_cue.csv", index_col=0)["estimate"].to_numpy(float)

products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
agents   = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)

for nu in [4, 16]:
    print("=== nu={} consumers/agent ({} total/market) ===".format(nu, 1000 * nu))
    prob = PCMProblem(products, agents, nu_per_agent=nu, workers=1)  # serial so times are clean

    rows = []
    for m in prob.markets:
        t0 = time.time()
        prob._solve_market(m, theta0, grad=False)
        rows.append({
            "market_id": m.market_id,
            "n_products": len(m.shares),
            "time_s": time.time() - t0,
        })

    df = pd.DataFrame(rows).sort_values("time_s", ascending=False).reset_index(drop=True)
    print("Rank  market  n_prod  time_s")
    for i, r in df.head(15).iterrows():
        print("  {:2d}  {:>8}  {:3d}  {:.2f}s".format(
              i + 1, r["market_id"], int(r["n_products"]), r["time_s"]))
    total  = df.time_s.sum()
    top10  = df.head(10).time_s.sum()
    print("Total {:.1f}s | top-10 = {:.1f}s ({:.0f}%) | max {:.2f}s | median {:.3f}s".format(
          total, top10, 100.0 * top10 / total, df.time_s.max(), df.time_s.median()))
    df.to_csv(OUT / "market_times_nu{}.csv".format(nu), index=False)
    print("")
