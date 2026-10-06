#!/usr/bin/env python3
"""
Exact solution of the beta-splitting moment recurrence
    a_n = sum_{j=1}^{n-1} p_{n,j} (a_j + a_{n-j}) + b_n ,  a_1 = 0,
    p_{n,j} = g_j g_{n-j} / sum_i g_i g_{n-i},   g_j = Gamma(beta+j+1)/j!,
and comparison with the predicted asymptotics of theorem.tex.

Usage:  python3 numerics.py NMAX [beta1,beta2,...]
Writes  numerics_out_<NMAX>.json  and prints a table.

Columns: ratio a_n/pred at the check points; "ext" = crude extrapolation of the
ratio assuming a relative correction proportional to 1/log n (log-type regimes)
or to n^{-gamma} (beta<-1 power regimes), using the last two check points.
"""
import sys, json, math
import numpy as np
from scipy.special import gammaln, digamma, gamma as Gamma
from scipy.integrate import quad
import mpmath as mp

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BETAS = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [-1.9, -1.5, -1.2, -1.0, -0.5, 0.0, 1.0, 5.0]
CHECK = sorted(set([n for n in (1000, 3000, 10000, 30000) if n < NMAX] + [NMAX]))

TOLLS = {
    '1':      lambda n: np.ones_like(n, dtype=float),
    'log n':  lambda n: np.log(n),
    'n^1/2':  lambda n: np.sqrt(n),
    'n':      lambda n: n.astype(float),
    'n log n':lambda n: n*np.log(n),
    'n^3/2':  lambda n: n**1.5,
    'n^2':    lambda n: n**2.0,
}
TOLL_EXP = {'1':(0,0), 'log n':(0,1), 'n^1/2':(0.5,0), 'n':(1,0), 'n log n':(1,1), 'n^3/2':(1.5,0), 'n^2':(2,0)}
TOLL_LIST = list(TOLLS)

# ---------- Laplace exponent phi_beta and friends ----------
def Bfun(a, b):
    """Beta function B(a,b)=Gamma(a)Gamma(b)/Gamma(a+b) by analytic continuation; 0 if a+b is a nonpositive integer."""
    ab = a + b
    if abs(ab - round(ab)) < 1e-12 and round(ab) <= 0:
        return 0.0
    return Gamma(a)*Gamma(b)/Gamma(ab)

def phi(beta, z):
    """phi_beta(z) = int_0^1 (1-s^z) s^{beta+1} (1-s)^beta ds."""
    if abs(beta+1) < 1e-12:
        return digamma(z+1) - digamma(1.0)
    return Bfun(beta+2, beta+1) - Bfun(z+beta+2, beta+1)

def mu(beta):
    """mu_beta = phi_beta'(0) = int_0^1 (-log s) s^{beta+1}(1-s)^beta ds  (numerical quadrature)."""
    if abs(beta+1) < 1e-12:
        return math.pi**2/6
    # mu = -d/dz B(z+beta+2, beta+1) at z=0, with B(a,b)=Gamma(a)Gamma(b)/Gamma(a+b) (rgamma handles the pole)
    Bz = lambda z: mp.gamma(z+beta+2)*mp.gamma(beta+1)*mp.rgamma(z+2*beta+3)
    return float(-mp.diff(Bz, 0))

def phi_inf(beta):
    return Bfun(beta+2, beta+1)   # beta > -1

def gam(beta):
    return max(-beta-1.0, 0.0)

# ---------- fringe constant for small tolls ----------
def phi_km1(beta, k):
    """phi_beta(k-1) for an array k>=2."""
    if abs(beta+1) < 1e-12:
        return digamma(k) + np.euler_gamma           # h_{k-1}
    sgn = np.sign(Gamma(beta+1))
    return Bfun(beta+2, beta+1) - sgn*np.exp(gammaln(k+beta+1)+gammaln(beta+1)-gammaln(k+2*beta+2))

def fringe_constant(beta, toll, K=2_000_000):
    """(1/mu) sum_{k>=2} b_k phi_beta(k-1)/(k(k-1)), with an integral tail beyond K using
    phi_beta(k-1) ~ |Gamma(beta+1)| k^gamma + B(beta+2,beta+1)  (beta<-1),  h_{k-1} ~ log k + EulerGamma (beta=-1),
    phi_inf - B(k+beta+1,beta+1) ~ phi_inf  (beta>-1)."""
    k = np.arange(2, K+1, dtype=float)
    s = float(np.sum(TOLLS[toll](k)*phi_km1(beta, k)/(k*(k-1))))
    al, kk = TOLL_EXP[toll]
    # tail  sum_{k>K} b_k phi(k-1)/k^2  ~  int_X^inf x^{al-2} log^kk(x) * phi_asym(x) dx,  X=K+1/2,
    # with phi_asym(x) = c1 x^g + c0  (beta<-1),  log x + EulerGamma (beta=-1),  phi_inf (beta>-1).
    X = K + 0.5
    def J(p, k):   # int_X^inf x^{p-1} log^k x dx  for p<0, k in {0,1,2}
        assert p < 0
        t = [1.0, 1.0/(-p), 2.0/p**2]          # int_0^inf t^i e^{pt} dt = i!/(-p)^{i+1} -> times 1/(-p)
        L = math.log(X)
        return X**p*sum(math.comb(k, i)*L**(k-i)*math.factorial(i)/(-p)**(i+1) for i in range(k+1))
    if beta < -1:
        g = -beta-1
        tail = abs(Gamma(beta+1))*J(al+g-1, kk) + Bfun(beta+2, beta+1)*J(al-1, kk)
    elif abs(beta+1) < 1e-12:
        tail = J(al-1, kk+1) + np.euler_gamma*J(al-1, kk)
    else:
        tail = phi_inf(beta)*J(al-1, kk)
    return (s + tail)/mu(beta)

def check_fringe_identity(beta, K=2_000_000):
    """sanity: (1/mu) sum_k phi(k-1)/(k(k-1)) must equal 1 (toll b=1 gives a_n = n-1)."""
    return fringe_constant(beta, '1', K)

# ---------- predicted leading asymptotics ----------
def prediction(beta, toll, n, fringe_cache={}):
    """Return (name, value, corr_type) of predicted leading term for a_n.  corr_type in {'log','pow','none'}."""
    al, k = TOLL_EXP[toll]
    g = gam(beta); L = math.log(n)
    ac = 1.0 - g            # critical exponent
    if al > ac + 1e-12:     # superlinear regime
        if beta > -1:
            Kc = phi_inf(beta)/phi(beta, al-1)
            return f"{Kc:.6g} n^{al} log^{k} n", Kc*n**al*L**k, ('log' if k > 0 else 'log')
        elif abs(beta+1) < 1e-12:
            Kc = 1.0/phi(beta, al-1)
            return f"{Kc:.6g} n^{al} log^{k+1} n", Kc*n**al*L**(k+1), 'log'
        else:
            Kc = abs(Gamma(beta+1))/phi(beta, al+g-1)
            return f"{Kc:.6g} n^{al+g:.3g} log^{k} n", Kc*n**(al+g)*L**k, ('log' if k > 0 else 'pow')
    elif abs(al-ac) < 1e-12:   # critical regime
        if beta > -1:
            Kc = phi_inf(beta)/((k+1)*mu(beta))
            return f"{Kc:.6g} n log^{k+1} n", Kc*n*L**(k+1), 'log'
        elif abs(beta+1) < 1e-12:
            Kc = 1.0/((k+2)*mu(beta))
            return f"{Kc:.6g} n log^{k+2} n", Kc*n*L**(k+2), 'log'
        else:
            Kc = abs(Gamma(beta+1))/((k+1)*mu(beta))
            return f"{Kc:.6g} n log^{k+1} n", Kc*n*L**(k+1), 'log'
    else:                     # small-toll (linear) regime
        key = (beta, toll)
        if key not in fringe_cache:
            fringe_cache[key] = fringe_constant(beta, toll)
        Kc = fringe_cache[key]
        return f"{Kc:.6g} n (fringe sum)", Kc*n, 'none'

# ---------- exact recurrence (all tolls at once) ----------
def solve(beta, nmax):
    j = np.arange(0, nmax+1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        lg = gammaln(beta+j+1) - gammaln(j+1)
    g = np.exp(lg - lg[1])          # g_j = Gamma(beta+j+1)/j!, normalised; positive for j>=1
    g[0] = 0.0
    T = len(TOLL_LIST)
    n_arr = np.arange(0, nmax+1, dtype=float)
    B = np.zeros((nmax+1, T))
    for t, toll in enumerate(TOLL_LIST):
        bb = TOLLS[toll](np.maximum(n_arr, 1)); bb[0] = 0; bb[1] = 0
        B[:, t] = bb
    A = np.zeros((nmax+1, T))
    for n in range(2, nmax+1):
        w = g[1:n]*g[n-1:0:-1]          # g_j g_{n-j}, j=1..n-1
        Z = w.sum()
        A[n] = 2.0*(w @ A[1:n])/Z + B[n]
    return {toll: A[:, t] for t, toll in enumerate(TOLL_LIST)}

def extrapolate(r1, n1, r2, n2, ctype, beta):
    if ctype == 'log':
        L1, L2 = math.log(n1), math.log(n2)
        return (r2*L2 - r1*L1)/(L2 - L1)
    if ctype == 'pow':
        g = gam(beta); w1, w2 = n1**g, n2**g
        return (r2*w2 - r1*w1)/(w2 - w1)
    return r2

def main():
    out = {}
    print(f"NMAX={NMAX}  check points={CHECK}")
    for beta in BETAS:
        sols = solve(beta, NMAX)
        out[str(beta)] = {}
        print(f"\n=== beta = {beta}   (gamma={gam(beta):.3g}, mu_beta={mu(beta):.6g}, phi_beta(n-1) ~ "
              + (f"{phi_inf(beta):.6g}" if beta > -1 else ("log n" if abs(beta+1) < 1e-12 else f"{abs(Gamma(beta+1)):.6g} n^{gam(beta):.3g}")) + ")")
        for toll, a in sols.items():
            ratios = {}
            for n in CHECK:
                name, pred, ctype = prediction(beta, toll, n)
                ratios[n] = a[n]/pred
            n1, n2 = CHECK[-2], CHECK[-1]
            ext = extrapolate(ratios[n1], n1, ratios[n2], n2, ctype, beta)
            out[str(beta)][toll] = {'pred': name, 'corr': ctype,
                                    'ratios': {str(n): float(v) for n, v in ratios.items()},
                                    'ext': float(ext),
                                    'a_at_check': {str(n): float(a[n]) for n in CHECK}}
            print(f"  b_n={toll:8s} pred ~ {name:30s} a_n/pred: " +
                  "  ".join(f"{n}:{ratios[n]:.4f}" for n in CHECK) + f"   ext({ctype}):{ext:.4f}")
    with open(f'numerics_out_{NMAX}.json', 'w') as f:
        json.dump(out, f, indent=1)

if __name__ == '__main__':
    main()
