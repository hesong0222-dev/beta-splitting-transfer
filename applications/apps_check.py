#!/usr/bin/env python3
"""Numerical check of the application constants (Section "Applications" of main.tex).

Solves exactly (float64) the beta-splitting recurrence
    a_n = 2 sum_{j<n} p_{n,j} a_j + b_n,  a_1 = 0,
for the tolls of
    Sackin       b_n = n
    Colless      b_n = E|n - 2 J_n|
    cophenetic   b_n = C(n,2)          (Phi_n = a_n - C(n,2))
    QColless     b_n = E (n - 2 J_n)^2 (quadratic Colless index)
    cherries     b_n = 1{n=2}
and compares with the predicted first-order constants.  The leading coefficient is
extracted by least squares on a geometric grid with the basis of the predicted
expansion (leading term + lower-order terms); 'ratio' = a_n / predicted leading term
at n = NMAX, 'fit' = fitted leading coefficient / predicted.

Usage: python3 apps_check.py NMAX beta1,beta2,...
"""
import sys, math
import numpy as np
from scipy.special import gammaln, gamma as G, digamma
import mpmath as mp

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BETAS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [-1.5, -1.0, 0.0]


def Bfun(a, b):
    ab = a + b
    if abs(ab - round(ab)) < 1e-12 and round(ab) <= 0:
        return 0.0
    return G(a) * G(b) / G(ab)


def _quad(beta, h, hr):
    """int_0^1 h(s) s^{beta+1} (1-s)^beta ds, where hr(u) = h(1-u) is supplied separately so that it
    can be evaluated without cancellation for small u (h(s) = O(1-s) as s -> 1).
    Both endpoints can be singular for beta < -1; on [0,1/2] substitute s = t^m and on [1/2,1]
    substitute 1-s = t^m, m = 1/(beta+2), which makes both integrands bounded."""
    m = 1 / (beta + 2) if beta < 0 else 1.0
    T = mp.mpf(0.5) ** (1 / m)
    left = lambda t: m * t ** (m - 1) * h(t ** m) * (t ** m) ** (beta + 1) * (1 - t ** m) ** beta
    right = lambda t: m * t ** (m * (beta + 1) - 1) * hr(t ** m) * (1 - t ** m) ** (beta + 1)
    return float(mp.quad(left, [0, T]) + mp.quad(right, [0, T]))


def phi(beta, z):
    """phi_beta(z) = int_0^1 (1-s^z) s^{beta+1}(1-s)^beta ds."""
    return _quad(beta, lambda s: 1 - s ** z, lambda u: -mp.expm1(z * mp.log1p(-u)))


def mu(beta):
    """mu_beta = int_0^1 log(1/s) s^{beta+1}(1-s)^beta ds."""
    return _quad(beta, lambda s: -mp.log(s), lambda u: -mp.log1p(-u))


def predictions(beta):
    """Return dict toll -> (leading coefficient, exponent s, log power m, fit basis)."""
    P = {}
    m = mu(beta)
    if beta > -1:
        pinf = Bfun(beta + 2, beta + 1)
        cb = 2.0 ** (-2 * beta - 1) / ((beta + 1) * Bfun(beta + 1, beta + 1))
        P['Sackin'] = (pinf / m, 1, 1)
        P['Colless'] = (cb * pinf / m, 1, 1)
        P['cophenetic'] = ((beta + 2) / (2 * (beta + 1)), 2, 0)
        P['QColless'] = (1 / (beta + 1), 2, 0)
    elif beta == -1:
        P['Sackin'] = (3 / math.pi ** 2, 1, 2)
        P['Colless'] = (3 / math.pi ** 2, 1, 2)
        P['cophenetic'] = (0.5, 2, 1)
        P['QColless'] = (1.0, 2, 1)
    else:
        g1 = abs(G(beta + 1))
        P['Sackin'] = (g1 / phi(beta, -beta - 1), -beta, 0)
        P['Colless'] = (g1 / phi(beta, -beta - 1), -beta, 0)
        P['cophenetic'] = (g1 / (2 * phi(beta, -beta)), 1 - beta, 0)
        P['QColless'] = (g1 / phi(beta, -beta), 1 - beta, 0)
    P['cherries'] = (phi(beta, 1) / (2 * m), 1, 0)
    return P


def solve(beta, N):
    n_all = np.arange(N + 1, dtype=float)
    if beta == -1:
        H = np.zeros(N + 1); H[1:] = np.cumsum(1 / n_all[1:])
    else:
        lg = gammaln(beta + n_all + 1) - gammaln(n_all + 1)
    names = ['Sackin', 'Colless', 'cophenetic', 'QColless', 'cherries']
    A = np.zeros((len(names), N + 1))
    for n in range(2, N + 1):
        j = np.arange(1, n, dtype=float)
        if beta == -1:
            p = n / (2 * H[n - 1] * j * (n - j))
        else:
            w = lg[1:n] + lg[n - 1:0:-1]
            w = np.exp(w - w.max()); p = w / w.sum()
        d = n - 2 * j
        tolls = np.array([n, p @ np.abs(d), n * (n - 1) / 2, p @ (d * d), 1.0 if n == 2 else 0.0])
        A[:, n] = 2 * (A[:, 1:n] @ p) + tolls
    A[2] -= n_all * (n_all - 1) / 2          # cophenetic index excludes the root
    return names, A


def fit_leading(n, y, s, m, beta):
    """Least squares for y_n = K n^s log^m n + lower terms; returns K."""
    L = np.log(n)
    gam = max(-beta - 1, 0)
    if beta < -1:      # K n^s + K1 n^{s-gam} + K2 n log n + K3 n
        cols = [n ** s, n ** (s - gam), n * L, n]
    elif m >= 1:       # y/n^s = A L^m + B L^{m-1} + ... + const + (C/L  or  C n^{-(beta+1)} if -1<beta<0)
        last = n ** s / L if beta == -1 or beta >= 0 else n ** (s - beta - 1)
        cols = [n ** s * L ** m] + [n ** s * L ** i for i in range(m - 1, -1, -1)] + [last]
    elif -1 < beta < 0:  # K n^s + K1 n^{s-(beta+1)} + K2 n log n + K3 n
        cols = [n ** s, n ** (s - beta - 1), n * L, n]
    else:              # power, beta >= 0: K n^s + K1 n log n + K2 n
        cols = [n ** s, n * L, n]
    X = np.array(cols).T
    sc = np.abs(X).max(axis=0)
    coef, *_ = np.linalg.lstsq(X / sc / y[:, None] * 1.0, np.ones_like(y), rcond=None)
    return coef[0] / sc[0]


def main():
    print(f"NMAX={NMAX}")
    for beta in BETAS:
        P = predictions(beta)
        names, A = solve(beta, NMAX)
        grid = np.unique(np.geomspace(NMAX // 15, NMAX, 40).astype(int))
        print(f"\n=== beta={beta}")
        for i, nm in enumerate(names):
            K, s, m = P[nm]
            nN = float(NMAX)
            lead = K * nN ** s * math.log(nN) ** m
            ratio = A[i, NMAX] / lead
            if nm == 'cherries':
                fitr = ratio
            else:
                Kf = fit_leading(grid.astype(float), A[i, grid], s, m, beta)
                fitr = Kf / K
            print(f"  {nm:11s} pred {K:.6g} n^{s:g} log^{m} n   a_N/pred={ratio:.4f}   fit/pred={fitr:.4f}")


if __name__ == '__main__':
    main()
