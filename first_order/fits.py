#!/usr/bin/env python3
"""
Least-squares fits of the asymptotic form of a_n (exact recurrence values) on a geometric grid
n in [NMIN, NMAX], to extract the LEADING coefficient and compare with the prediction of theorem.tex.
Usage: python3 fits.py NMAX [betas]
"""
import sys, math, json
import numpy as np
from scipy.special import digamma, polygamma, gamma as Gamma
import numerics as N

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
BETAS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [-1.9, -1.5, -1.2, -1.0, -0.5, 0.0, 1.0, 5.0]
NMIN = max(2000, NMAX//30)

def grid(nmin, nmax, m=60):
    g = np.unique(np.round(np.exp(np.linspace(math.log(nmin), math.log(nmax), m))).astype(int))
    return g

def lsq(cols, y):
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef

def main():
    res = {}
    for beta in BETAS:
        sols = N.solve(beta, NMAX)
        g = N.gam(beta); mu = N.mu(beta)
        ns = grid(NMIN, NMAX).astype(float)
        L = np.log(ns)
        print(f"\n=== beta={beta}  fit window n in [{NMIN},{NMAX}]  (gamma={g:.3g})")
        res[str(beta)] = {}
        for toll, a in sols.items():
            al, k = N.TOLL_EXP[toll]
            y = a[ns.astype(int)]
            ac = 1 - g
            if al > ac + 1e-12:          # super-critical
                if beta > -1:
                    pred = N.phi_inf(beta)/N.phi(beta, al-1)
                    # a_n = K n^al log^k n + K1 n^{al-beta-1} log^k n + K2 n log n + K3 n
                    cols = [ns**al*L**k, ns**(al-beta-1)*L**k, ns*L, ns]
                    c = lsq(cols, y); K = c[0]; form = "K n^a log^k n + K1 n^{a-b-1}log^k n + K2 n log n + K3 n"
                    extra = ""
                elif abs(beta+1) < 1e-12:
                    H = digamma(al) + np.euler_gamma
                    pred = 1.0/H
                    # a_n / n^al = A log^{k+1} n + B log^k n + C log^{k-1} n ...
                    yy = y/ns**al
                    if k == 0:
                        cols = [L, np.ones_like(L), 1/L]
                        c = lsq(cols, yy); K = c[0]
                        Bpred = np.euler_gamma/H - polygamma(1, al)/H**2
                        extra = f"  2nd coeff B fit={c[1]:.4f} heuristic={Bpred:.4f}"
                        form = "a_n/n^a = A log n + B + C/log n"
                    else:
                        cols = [L**(k+1), L**k, L**(k-1)]
                        c = lsq(cols, yy); K = c[0]; extra = ""
                        form = "a_n/n^a = A log^{k+1} n + B log^k n + C log^{k-1} n"
                else:
                    pred = abs(Gamma(beta+1))/N.phi(beta, al+g-1)
                    cols = [ns**(al+g)*L**k, ns**al*L**k, ns*L, ns]
                    if k == 1: cols = [ns**(al+g)*L, ns**(al+g), ns**al*L, ns**al, ns]
                    c = lsq(cols, y); K = c[0]; extra = ""
                    form = "K n^{a+g} log^k n + K1 n^a log^k n + K2 n log n + K3 n"
            elif abs(al-ac) < 1e-12:     # critical
                if beta > -1:
                    pred = N.phi_inf(beta)/((k+1)*mu)
                    yy = y/ns
                    cols = [L**(k+1), L**k, np.ones_like(L)] if k >= 1 else [L, np.ones_like(L), ns**(-beta-1)]
                    c = lsq(cols, yy); K = c[0]; extra = ""
                    form = "a_n/n = A log^{k+1} n + B log^k n + C"
                elif abs(beta+1) < 1e-12:
                    pred = 1.0/((k+2)*mu)
                    yy = y/ns
                    cols = [L**(k+2), L**(k+1), L**k, L**(k-1)] if k >= 1 else [L**2, L, np.ones_like(L), 1/L]
                    c = lsq(cols, yy); K = c[0]
                    extra = f"  2nd coeff fit={c[1]:.4f}" + ("  (Aldous-Pittel: 0.7952)" if k == 0 else "")
                    form = "a_n/n = A log^{k+2} n + B log^{k+1} n + ..."
                else:
                    pred = abs(Gamma(beta+1))/((k+1)*mu)
                    yy = y/ns
                    cols = [L**(k+1), L**k, ns**(-g)*L**k, np.ones_like(L)]
                    c = lsq(cols, yy); K = c[0]; extra = ""
                    form = "a_n/n = A log^{k+1} n + B log^k n + C n^{-g} log^k n + D"
            else:                         # linear (fringe) regime
                pred = N.fringe_constant(beta, toll)
                yy = y/ns
                # partial-sum prediction: (1/mu) sum_{k<=n} b_k phi(k-1)/(k(k-1))
                kk = np.arange(2, int(ns[-1])+1, dtype=float)
                terms = N.TOLLS[toll](kk)*N.phi_km1(beta, kk)/(kk*(kk-1))/mu
                cs = np.cumsum(terms)
                partial = cs[ns.astype(int)-2]
                K = yy[-1]
                extra = f"  a_n/n at n={int(ns[-1])}: {yy[-1]:.4f}; partial fringe sum to n: {partial[-1]:.4f}"
                form = "a_n/n (no fit)"
            print(f"  b_n={toll:8s} leading coeff: fit={K:.5f} pred={pred:.5f} ratio={K/pred:.4f}   [{form}]{extra}")
            res[str(beta)][toll] = {'fit': float(K), 'pred': float(pred), 'ratio': float(K/pred), 'form': form, 'extra': extra}
    with open(f'fits_out_{NMAX}.json', 'w') as f:
        json.dump(res, f, indent=1)

if __name__ == '__main__':
    main()
