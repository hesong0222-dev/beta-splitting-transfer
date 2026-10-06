#!/usr/bin/env python3
"""Check of Corollary 8.1(i'): for beta <= -1, (E S_n - E C_n)/(n log n) -> I_beta(min)/mu_beta,
I_beta(min) = int_0^1 x^beta (1-x)^beta min(x,1-x) dx.  Prints the ratio at several n and the limit."""
import sys, math
sys.argv = ['x']
import numpy as np, mpmath as mp
import apps_check as A
for beta, N in [(-1.5, 30000), (-1.0, 100000), (-1.2, 30000)]:
    names, M = A.solve(beta, N)
    I = 2 * float(mp.quad(lambda x: x ** (beta + 1) * (1 - x) ** beta, [0, 0.25, 0.5]))
    lim = I / A.mu(beta)
    pts = [n for n in (1000, 3000, 10000, 30000, 100000) if n <= N]
    r = ' '.join(f"{n}:{(M[0, n] - M[1, n]) / (n * math.log(n)):.4f}" for n in pts)
    ns = np.array(pts, float); y = np.array([(M[0, n] - M[1, n]) / (n * math.log(n)) for n in pts])
    X = np.vstack([np.ones_like(ns), 1 / np.log(ns)]).T
    Af = np.linalg.lstsq(X, y, rcond=None)[0][0]
    print(f"beta={beta}: I(min)={I:.6f} limit I/mu={lim:.6f}   ratios {r}   fit A+B/log n: A={Af:.5f} (A/limit={Af/lim:.4f})")
