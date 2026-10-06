#!/usr/bin/env python3
"""
Monte Carlo sanity check of the exact moment recurrences (moments.py): simulate beta-splitting trees
with n leaves and compare sample mean/var/skewness of Sackin, Colless, total cophenetic with the
recurrence values at the same n.   Usage: python3 mc.py BETA N NSAMPLES [seed]
"""
import sys, math, time
import numpy as np
from moments import weights, central_moments

beta = float(sys.argv[1]); n0 = int(sys.argv[2]); R = int(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rng = np.random.default_rng(seed)
g = weights(beta, n0)
cdf_cache = {}
def split(n):
    c = cdf_cache.get(n)
    if c is None:
        w = g[1:n] * g[n - 1:0:-1]; c = np.cumsum(w); c /= c[-1]
        if n <= 4096: cdf_cache[n] = c
    return 1 + int(np.searchsorted(c, rng.random()))

def tree_stats(n):
    S = C = P = 0.0
    stack = [n]
    while stack:
        m = stack.pop()
        if m < 2: continue
        j = split(m)
        S += m; C += abs(m - 2 * j); P += j * (j - 1) / 2 + (m - j) * (m - j - 1) / 2
        stack.append(j); stack.append(m - j)
    return S, C, P

t0 = time.time()
X = np.array([tree_stats(n0) for _ in range(R)])
t1 = time.time()
m, c2, c3, _ = {}, {}, {}, {}
for i, stat in enumerate(['sackin', 'colless', 'cophenetic']):
    mm, v2, v3, v4 = central_moments(beta, n0, stat, kmax=3, g=g)
    x = X[:, i]; mu = x.mean(); var = x.var(ddof=1); sk = ((x - mu) ** 3).mean() / var ** 1.5
    se_mean = math.sqrt(var / R); se_var = var * math.sqrt(2 / (R - 1))
    print(f"beta={beta} n={n0} R={R} {stat:10s}: MC mean={mu:.6g} (+-{se_mean:.2g})  exact={mm[n0]:.6g}  z={(mu-mm[n0])/se_mean:+.2f} | "
          f"MC var={var:.5g} (+-{se_var:.2g}) exact={v2[n0]:.5g} z={(var-v2[n0])/se_var:+.2f} | MC skew={sk:.3f} exact={v3[n0]/v2[n0]**1.5:.3f}")
print(f"simulation time {t1-t0:.1f}s")
