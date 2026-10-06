#!/usr/bin/env python3
"""
Summary table: limit constants predicted by the moment recursions (limits.py) versus values extrapolated
from the exact recurrences (moments_*.npz).  Extrapolation = least squares on a geometric grid of n with
the expected correction terms (n^{-g}, n^{-2g}, 1/n, log n/n for beta<-1; 1/log n, 1/log^2 n for beta=-1).
"""
import math, glob
import numpy as np
import mpmath as mp
import limits as LM

def fit0(ns, y, cols):
    A = np.column_stack(cols); return np.linalg.lstsq(A, y, rcond=None)[0][0]

rows = []
for beta in (-1.8, -1.5, -1.2):
    g = -beta - 1
    d = np.load(f'moments_{beta}_30000.npz')
    ns = np.unique(np.round(np.exp(np.linspace(math.log(1000), math.log(30000), 50))).astype(int)); n = ns.astype(float); L = np.log(n)
    cols = [np.ones_like(n), n**-g, n**(-2*g), n**-g*L, 1/n, L/n]
    mu = LM.mu_beta_lt(beta, 4); rho = LM.rho_beta_lt(beta, 4)
    for stat, mom, sc in (('sackin', mu, 1+g), ('colless', mu, 1+g), ('cophenetic', rho, 2+g)):
        m, c2, c3, c4 = (d[f'{stat}_{k}'] for k in ('m', 'c2', 'c3', 'c4'))
        m1, v, sk, ku = LM.central(mom)
        rows.append((f"beta={beta} {stat:10s} X_n/n^{sc:.1f}",
                     ("mean", float(m1), fit0(ns, m[ns]/n**sc, cols)),
                     ("var", float(v), fit0(ns, c2[ns]/n**(2*sc), cols)),
                     ("skew", float(sk), fit0(ns, c3[ns]/c2[ns]**1.5, cols)),
                     ("kurt", float(ku), fit0(ns, c4[ns]/c2[ns]**2, cols))))
# beta=-1
d = np.load('moments_-1.0_100000.npz'); d30 = np.load('moments_-1.0_30000.npz')
nu = LM.nu_crit_sackin(4); rho = LM.rho_crit_coph(4)
ns = np.unique(np.round(np.exp(np.linspace(math.log(3000), math.log(100000), 50))).astype(int)); n = ns.astype(float); L = np.log(n)
cols = [np.ones_like(n), 1/L, 1/L**2]; colsLL = [np.ones_like(n), np.log(L)/L, 1/L, 1/L**2]
m, c2, c3, c4 = (d[f'sackin_{k}'] for k in ('m', 'c2', 'c3', 'c4'))
rows.append(("beta=-1 sackin (S_n-ES_n)/(n log n)",
             ("var", float(nu[2]), fit0(ns, c2[ns]/(n*L)**2, cols)),
             ("nu3", float(nu[3]), fit0(ns, c3[ns]/(n*L)**3, colsLL)),
             ("skew", float(nu[3]/nu[2]**1.5), fit0(ns, c3[ns]/c2[ns]**1.5, colsLL)),
             ("kurt", float(nu[4]/nu[2]**2), fit0(ns, c4[ns]/c2[ns]**2, colsLL))))
ns3 = np.unique(np.round(np.exp(np.linspace(math.log(1000), math.log(30000), 50))).astype(int)); n3 = ns3.astype(float); L3 = np.log(n3)
cols3 = [np.ones_like(n3), 1/L3, 1/L3**2]; cols3LL = [np.ones_like(n3), np.log(L3)/L3, 1/L3, 1/L3**2]
m, c2, c3, c4 = (d30[f'colless_{k}'] for k in ('m', 'c2', 'c3', 'c4'))
rows.append(("beta=-1 colless (C_n-EC_n)/(n log n) [n<=3e4]",
             ("var", float(nu[2]), fit0(ns3, c2[ns3]/(n3*L3)**2, cols3)),
             ("skew", float(nu[3]/nu[2]**1.5), fit0(ns3, c3[ns3]/c2[ns3]**1.5, cols3LL)),
             ("kurt", float(nu[4]/nu[2]**2), fit0(ns3, c4[ns3]/c2[ns3]**2, cols3LL))))
m, c2, c3, c4 = (d[f'cophenetic_{k}'] for k in ('m', 'c2', 'c3', 'c4'))
m1, v, sk, ku = LM.central(rho)
rows.append(("beta=-1 cophenetic Phi_n/(n^2 log n)",
             ("mean", float(m1), fit0(ns, m[ns]/(n*n*L), cols)),
             ("var", float(v), fit0(ns, c2[ns]/(n**4*L**2), cols)),
             ("skew", float(sk), fit0(ns, c3[ns]/c2[ns]**1.5, cols)),
             ("kurt", float(ku), fit0(ns, c4[ns]/c2[ns]**2, cols))))
# mean Sackin - Colless at beta=-1 vs 12 log2/pi^2
mS = d30['sackin_m']; mC = d30['colless_m']
y = (mS[ns3] - mC[ns3])/(n3*L3)
rows.append(("beta=-1 (ES_n-EC_n)/(n log n)", ("limit", 12*math.log(2)/math.pi**2, fit0(ns3, y, cols3)), ("at n=3e4", 12*math.log(2)/math.pi**2, float(y[-1]))))

print(f"{'case':48s} {'quantity':8s} {'predicted':>12s} {'extrapolated':>13s} {'rel.diff':>9s}")
for r in rows:
    for q, p, f in r[1:]:
        print(f"{r[0]:48s} {q:8s} {p:12.5f} {f:13.5f} {abs(f-p)/abs(p):9.2e}")
