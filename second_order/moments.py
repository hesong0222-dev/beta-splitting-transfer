#!/usr/bin/env python3
"""
Exact mean and central moments (orders 2,3,4) of additive shape parameters under the
beta-splitting model, by the moment recurrences.

Statistic X_n = X_J + X'_{n-J} + t_n(J), J ~ p_{n,j} (split of the root), X_1 = 0.
  mean   m_n = 2 sum_j p_{n,j} m_j + sum_j p_{n,j} t_n(j)
  Delta_n(j) = m_j + m_{n-j} + t_n(j) - m_n              (all tolls below are symmetric in j <-> n-j)
  c2_n = 2 sum p c2_j + sum p Delta^2
  c3_n = 2 sum p c3_j + sum p (6 c2_j Delta + Delta^3)
  c4_n = 2 sum p c4_j + sum p (8 c3_j Delta + 6 c2_j c2_{n-j} + 12 c2_j Delta^2 + Delta^4)
Tolls: sackin t=n, colless t=|n-2j|, cophenetic t=C(j,2)+C(n-j,2)  (total cophenetic index).

Usage: python3 moments.py BETA NMAX [stats]   -> moments_<beta>_<NMAX>.npz and a printed table.
"""
import sys, json, math
import numpy as np
from scipy.special import gammaln

def weights(beta, nmax):
    j = np.arange(0, nmax + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = gammaln(beta + j + 1) - gammaln(j + 1)
    g = np.exp(lg - lg[1]); g[0] = 0.0
    return g

TOLLS = {
    'sackin':     lambda n, j: np.full_like(j, float(n)),
    'colless':    lambda n, j: np.abs(n - 2 * j),
    'cophenetic': lambda n, j: j * (j - 1) / 2 + (n - j) * (n - j - 1) / 2,
}

def central_moments(beta, nmax, stat, kmax=4, g=None):
    if g is None: g = weights(beta, nmax)
    toll = TOLLS[stat]
    m = np.zeros(nmax + 1); c2 = np.zeros(nmax + 1); c3 = np.zeros(nmax + 1); c4 = np.zeros(nmax + 1)
    for n in range(2, nmax + 1):
        j = np.arange(1, n, dtype=float)
        p = g[1:n] * g[n - 1:0:-1]; p /= p.sum()
        mj = m[1:n]; mr = mj[::-1]
        t = toll(n, j)
        m[n] = p @ (mj + mr + t)
        D = mj + mr + t - m[n]
        c2j = c2[1:n]; c2r = c2j[::-1]
        D2 = D * D
        c2[n] = 2 * (p @ c2j) + p @ D2
        if kmax >= 3:
            c3j = c3[1:n]
            c3[n] = 2 * (p @ c3j) + p @ (6 * c2j * D + D2 * D)
        if kmax >= 4:
            c4[n] = 2 * (p @ c4[1:n]) + p @ (8 * c3j * D + 6 * c2j * c2r + 12 * c2j * D2 + D2 * D2)
    return m, c2, c3, c4

def mean_only(beta, nmax, bfun, g=None):
    """mean recurrence a_n = 2 sum p a_j + b_n for a deterministic toll b_n (array-valued function of n)."""
    if g is None: g = weights(beta, nmax)
    a = np.zeros(nmax + 1)
    b = bfun(np.arange(0, nmax + 1, dtype=float)); b[:2] = 0
    for n in range(2, nmax + 1):
        p = g[1:n] * g[n - 1:0:-1]
        a[n] = 2 * (p @ a[1:n]) / p.sum() + b[n]
    return a

if __name__ == '__main__':
    beta = float(sys.argv[1]); nmax = int(sys.argv[2])
    stats = sys.argv[3].split(',') if len(sys.argv) > 3 else ['sackin', 'colless', 'cophenetic']
    g = weights(beta, nmax)
    out = {}
    checks = [n for n in (100, 300, 1000, 3000, 10000, 30000, 100000) if n <= nmax]
    for stat in stats:
        m, c2, c3, c4 = central_moments(beta, nmax, stat, g=g)
        out[stat] = dict(m=m, c2=c2, c3=c3, c4=c4)
        print(f"beta={beta} {stat}")
        print("   n        mean          var       sd/mean     skew      kurt")
        for n in checks:
            sd = math.sqrt(c2[n])
            print(f"{n:7d} {m[n]:13.6g} {c2[n]:13.6g} {sd/m[n]:9.5f} {c3[n]/sd**3:9.5f} {c4[n]/c2[n]**2:9.5f}")
    np.savez_compressed(f"moments_{beta}_{nmax}.npz", **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
