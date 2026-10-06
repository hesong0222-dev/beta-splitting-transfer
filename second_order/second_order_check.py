#!/usr/bin/env python3
"""
Numerical check of the second-order transfer theorems (second_order.tex, Thm C (beta=-1) and Thm D (-2<beta<-1)).
Exact recurrence a_n = 2 sum p a_j + b_n, b_n = n^alpha, a_1 = 0, n <= NMAX; residual after subtracting the
two-term expansion, divided by the claimed error scale, must stay bounded (and typically converge).
Usage: python3 second_order_check.py NMAX
"""
import sys, math
import numpy as np
import mpmath as mp
from scipy.special import digamma, polygamma, gamma as Gamma
sys.path.insert(0, '..')
import numerics as N          # phi, mu, gam from the first-order work
from moments import mean_only, weights

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
CHK = [n for n in (1000, 3000, 10000, 30000, 100000) if n <= NMAX]
z2, z3, gE = math.pi**2/6, 1.2020569031595942, np.euler_gamma

def kappa(beta):      # B(beta+2, beta+1) by analytic continuation (= phi_beta(infty) for beta>-1)
    return float(mp.beta(beta + 2, beta + 1))

print("=== beta = -1 (Theorem C) ===")
g = weights(-1.0, NMAX)
for al in (1.0, 1.5, 2.0, 3.0):
    a = mean_only(-1.0, NMAX, lambda n: n**al, g=g)
    n = np.arange(NMAX + 1, dtype=float); L = np.log(np.maximum(n, 1))
    if al == 1.0:
        pred = n * ((3/math.pi**2) * L**2 + (gE*z2 + z3)/z2**2 * L)
        scale = n; sname = "n"
    else:
        H = digamma(al) + gE; B = gE/H - polygamma(1, al)/H**2
        pred = n**al * (L/H + B)
        if al - 1 > 1:   scale = n**(al-1) * L; sname = "n^{a-1} log n"
        elif al - 1 == 1: scale = n * L**2;     sname = "n log^2 n"
        else:             scale = n;            sname = "n"
    r = a - pred
    print(f" toll n^{al}: (a_n - two-term)/({sname}) at n=" + ", ".join(f"{m}: {r[m]/scale[m]:+.5f}" for m in CHK)
          + (f"   [B_alpha={B:.5f}]" if al != 1.0 else "   [AP25 coefficients 3/pi^2, (gE z2+z3)/z2^2]"))

print("\n=== -2 < beta < -1 (Theorem D) ===")
for beta in (-1.2, -1.5, -1.8):
    g = weights(beta, NMAX); gam = -beta - 1; G = abs(Gamma(beta + 1)); kap = kappa(beta); mu = N.mu(beta)
    print(f" beta={beta}: gamma={gam}, |Gamma(beta+1)|={G:.6f}, kappa=B(beta+2,beta+1)={kap:.6f}, mu={mu:.6f}, alpha_c={1-gam:.2f}")
    for al in sorted({0.7, 0.9, 1.0, 1.5, 2.0, 3.0}):
        if al <= 1 - gam + 1e-9: continue
        a = mean_only(beta, NMAX, lambda n: n**al, g=g)
        n = np.arange(NMAX + 1, dtype=float); L = np.log(np.maximum(n, 1))
        s = al + gam; K = G / N.phi(beta, s - 1)
        eta = min(1.0, al, 2*gam)
        if al > 1 + 1e-9:
            pred = K * n**s + kap / N.phi(beta, al - 1) * n**al
            second = f"kappa/phi(a-1) n^a = {kap/N.phi(beta, al-1):+.5f} n^{al}"
        elif abs(al - 1) < 1e-9:
            pred = K * n**s + kap / mu * n * L
            second = f"(kappa/mu) n log n = {kap/mu:+.5f} n log n"
        else:
            pred = K * n**s
            second = "none (next term O(n))"
        scale = n**(s - eta) * L + n
        r = a - pred
        print(f"   toll n^{al}: K={K:.5f}; 2nd term {second}; eta={eta:.2f}; (a_n - expansion)/(n^{{s-eta}} log n + n) at n="
              + ", ".join(f"{m}: {r[m]/scale[m]:+.4f}" for m in CHK))
