"""Probe: invert PCM shares market by market on Nevo's fake cereal data.

Utility of consumer i for brand j in market t (PCM, no logit error):
    u_ijt = delta_jt + sum_k x_jk * (sigma_k * nu_ik) - alpha_i * p_jt,
    log alpha_i ~ N(0, sigma_p),   outside good u_i0 = 0.
x = [1, sugar, mushy]. Taste draws are a Gauss-Hermite product grid.
This only checks that the inversion runs on real-shaped data; parameters are
placeholders, not estimates.

Run: PYTHONPATH=nevo_replication/.vendor:build-dbg/python_bindings \
     python nevo_replication/probe_inversion.py [cone|nlopt] [n_nodes]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import pyblp
import python_pcm_inversion as inv
import python_market_share as ms


def gh_grid(dim: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    z, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    mesh = np.stack(np.meshgrid(*[z] * dim, indexing="ij"), -1).reshape(-1, dim)
    wts = np.prod(np.stack(np.meshgrid(*[w] * dim, indexing="ij"), -1).reshape(-1, dim), axis=1)
    return mesh, wts


def main() -> None:
    method = {"cone": inv.CONE_AWARE_NEWTON, "nlopt": inv.NLOPT_SLSQP}[sys.argv[1] if len(sys.argv) > 1 else "cone"]
    n_nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    products = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
    sigma_x = np.array([0.05, 0.003, 0.03])
    sigma_p = 0.5
    grid, weights = gh_grid(3, n_nodes)
    fails = 0
    t0 = time.time()
    for mkt, df in products.groupby("market_ids", sort=True):
        df = df.sort_values("prices")
        x = np.column_stack([np.ones(len(df)), df.sugar, df.mushy]).astype(float)
        p = df.prices.to_numpy(float)
        s = df.shares.to_numpy(float)
        try:
            d = inv.invert_shares(x, p, sigma_p, sigma_x, grid, weights, s, method=method)
            s_hat = ms.unconditional_share_no_jacobian(d, x, p, sigma_p, sigma_x, grid, weights)
            err = float(np.max(np.abs(np.asarray(s_hat) - s)))
            if not np.isfinite(err) or err > 1e-6:
                fails += 1
            print(f"{mkt}: max|s_hat-s|={err:.2e}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"{mkt}: FAILED {type(e).__name__}: {e}")
    print(f"done {time.time() - t0:.1f}s, failures {fails}/94, grid {len(weights)} nodes")


if __name__ == "__main__":
    main()
