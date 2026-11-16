"""Profile per-market inversion time at nu=16 using 14 OMP threads each.
This gives the correct allocation weights for the full CUE estimation.
Run: PCM_OMP_TOTAL=14 .../python profile_market_times_nu16.py
"""
import sys, time, pathlib
import pandas as pd
import numpy as np
import pyblp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pcm_petrin import PCMProblem

OUT = pathlib.Path(__file__).resolve().parent / "results"
theta0 = pd.read_csv(OUT / "pcm_start_cue.csv", index_col=0)["estimate"].to_numpy(float)
products = pd.read_csv(pyblp.data.PETRIN_PRODUCTS_LOCATION)
agents   = pd.read_csv(pyblp.data.PETRIN_AGENTS_LOCATION)

# serial: workers=1, 14 OMP threads per market
prob = PCMProblem(products, agents, nu_per_agent=16, workers=1)
rows = []
for m in prob.markets:
    t0 = time.time()
    prob._solve_market(m, theta0, grad=False)
    t = round(time.time() - t0, 2)
    rows.append({"market_id": m.market_id, "n_products": len(m.shares), "time_s": t})
    print(f"  mkt {m.market_id}: {len(m.shares)} prod  {t:.1f}s", flush=True)

df = pd.DataFrame(rows).sort_values("time_s", ascending=False)
df.to_csv(OUT / "market_times_nu16.csv", index=False)
print("Saved market_times_nu16.csv")
print(df.to_string(index=False))
