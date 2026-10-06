#!/usr/bin/env python3
"""Logged checks quoted in Section 9 of main.tex:
(a) quadrature values of phi_beta(z), mu_beta (apps_check.py) against the closed forms of Lemma 2.2;
(b) the identity (1/mu_beta) sum_{k>=2} phi_beta(k-1)/(k(k-1)) = 1 (Remark 7.4(1)), evaluated with
    ../numerics.py's fringe_constant (partial sum to K = 2e6 plus asymptotic tail)."""
import sys, os, math
sys.argv = ['x']
import mpmath as mp
import apps_check as A
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numerics as N
for b in [-1.9, -1.5, -1.2, -0.5, 0.0, 1.0, 5.0]:
    err = 0.0
    for z in [0.25, 0.5, 1.0, 1.5, 2.0, -b if b < -1 else 3.0]:
        cf = A.Bfun(b + 2, b + 1) - A.Bfun(z + b + 2, b + 1)
        err = max(err, abs(A.phi(b, z) - cf) / cf)
    Bz = lambda z: mp.gamma(z + b + 2) * mp.gamma(b + 1) * mp.rgamma(z + 2 * b + 3)
    mu_cf = float(-mp.diff(Bz, 0))
    print(f"beta={b:5}: max rel err phi quad vs closed form = {err:.1e};  mu quad={A.mu(b):.10g} closed={mu_cf:.10g}"
          f";  fringe identity sum = {N.check_fringe_identity(b):.10f}")
mu1 = A.mu(-1.0)
print(f"beta=-1: mu quad = {mu1:.12f}, pi^2/6 = {math.pi**2/6:.12f}; fringe identity = {N.check_fringe_identity(-1.0):.10f}")
