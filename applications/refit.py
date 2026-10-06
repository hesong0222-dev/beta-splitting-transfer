#!/usr/bin/env python3
"""Refit of the two cells of the main verification table whose original fit basis (../fits.py) omitted the
true second-order term: beta=-1.2, b_n = n log n (missing K' n^{1.2}) and beta=-0.5, b_n = n log n
(missing the n^{-(beta+1)} relative correction).  Exact recurrence up to NMAX, least squares on a
geometric grid in [NMAX/15, NMAX]."""
import sys, math
import numpy as np
from scipy.special import gammaln, gamma as G
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30000

def solve(beta, toll, N):
    n_all = np.arange(N + 1, dtype=float)
    lg = gammaln(beta + n_all + 1) - gammaln(n_all + 1)
    a = np.zeros(N + 1)
    for n in range(2, N + 1):
        w = lg[1:n] + lg[n - 1:0:-1]; w = np.exp(w - w.max()); p = w / w.sum()
        a[n] = 2 * (a[1:n] @ p) + toll(n)
    return a

def lsq(cols, y):
    X = np.array(cols).T / y[:, None]; sc = np.abs(X).max(axis=0)
    c, *_ = np.linalg.lstsq(X / sc, np.ones_like(y), rcond=None)
    return c / sc

for beta, pred, mk in [(-1.2, 12.9616, lambda n, L: [n**1.2 * L, n**1.2, n * L * L, n * L, n]),
                       (-0.5, 1.29435, lambda n, L: [n * L**2, n * L, n, n**0.5 * L, n**0.5])]:
    a = solve(beta, lambda n: n * math.log(n), NMAX)
    grid = np.unique(np.geomspace(NMAX // 15, NMAX, 40).astype(int)).astype(float)
    c = lsq(mk(grid, np.log(grid)), a[grid.astype(int)])
    print(f"beta={beta}  b_n=n log n  fitted leading={c[0]:.5f}  pred={pred}  fit/pred={c[0]/pred:.4f}")
