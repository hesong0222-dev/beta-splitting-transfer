#!/usr/bin/env python3
"""
Limit moments from the moment-transfer recursions (second_order.tex, Section 5).

(1) -2<beta<-1, Sackin (and Colless):  S_n / n^{1+g} -> law with moments mu_k,  g=-beta-1,
    mu_k = [ k |Gamma(beta+1)| mu_{k-1} + (1/2) sum_{a+b=k, a,b>=1} C(k,a) mu_a mu_b B(a(1+g)-g, b(1+g)-g) ] / phi_beta(k(1+g)-1).
(2) beta=-1, Sackin/Colless centred: (S_n - E S_n)/(n log n) -> W with moments nu_k (nu_0=1, nu_1=0),
    nu_k = I_k / h_{k-1},   I_k = k nu_{k-1} + (1/2) sum' k!/(a!b!c!) nu_a nu_b (-A)^c int_0^1 x^{a-1}(1-x)^{b-1} h(x)^c dx,
    A = 6/pi^2, h = binary entropy (nats), sum' over a+b+c=k except (k,0,0),(0,k,0).
(3) beta=-1, total cophenetic: Phi_n/(n^2 log n) -> law with moments rho_k,
    rho_k = [ (k/2) rho_{k-1} + (1/2) sum_{a+b=k,a,b>=1} C(k,a) rho_a rho_b B(2a,2b) ] / h_{2k-1}.
"""
import math, sys
import mpmath as mp
mp.mp.dps = 30

def phi(beta, z):
    return mp.beta(beta + 2, beta + 1) - mp.beta(z + beta + 2, beta + 1)

def harmonic(m):
    return mp.fsum(mp.mpf(1) / i for i in range(1, m + 1)) if m >= 1 else mp.mpf(0)

def mu_beta_lt(beta, K):
    g = -beta - 1; G = abs(mp.gamma(beta + 1))
    mu = [mp.mpf(1)]
    for k in range(1, K + 1):
        s = k * G * mu[k - 1]
        for a in range(1, k):
            b = k - a
            s += mp.mpf(1) / 2 * mp.binomial(k, a) * mu[a] * mu[b] * mp.beta(a * (1 + g) - g, b * (1 + g) - g)
        mu.append(s / phi(beta, k * (1 + g) - 1))
    return mu

_hcache = {}
def hint(a, b, c):
    """int_0^1 x^{a-1} (1-x)^{b-1} h(x)^c dx  (a,b>=0, c>=0; a=0 or b=0 requires c>=1)."""
    key = (a, b, c)
    if key in _hcache: return _hcache[key]
    h = lambda x: -x * mp.log(x) - (1 - x) * mp.log(1 - x)
    if c == 0:
        v = mp.beta(a, b)
    else:
        v = mp.quad(lambda x: x ** (a - 1) * (1 - x) ** (b - 1) * h(x) ** c, [0, mp.mpf(1) / 2, 1])
    _hcache[key] = v
    return v

def nu_crit_sackin(K):
    A = 6 / mp.pi ** 2
    nu = [mp.mpf(1), mp.mpf(0)]
    for k in range(2, K + 1):
        I = k * nu[k - 1]
        for a in range(0, k + 1):
            for b in range(0, k + 1 - a):
                c = k - a - b
                if (a, b) in ((k, 0), (0, k)): continue
                if nu[a] == 0 or nu[b] == 0: continue
                I += mp.mpf(1) / 2 * mp.factorial(k) / (mp.factorial(a) * mp.factorial(b) * mp.factorial(c)) \
                     * nu[a] * nu[b] * (-A) ** c * hint(a, b, c)
        nu.append(I / harmonic(k - 1))
    return nu

def rho_crit_coph(K):
    rho = [mp.mpf(1)]
    for k in range(1, K + 1):
        s = mp.mpf(k) / 2 * rho[k - 1]
        for a in range(1, k):
            b = k - a
            s += mp.mpf(1) / 2 * mp.binomial(k, a) * rho[a] * rho[b] * mp.beta(2 * a, 2 * b)
        rho.append(s / harmonic(2 * k - 1))
    return rho

def rho_beta_lt(beta, K):
    """-2<beta<-1, total cophenetic: Phi_n/n^{2+g} limit moments."""
    g = -beta - 1; G = abs(mp.gamma(beta + 1))
    rho = [mp.mpf(1)]
    for k in range(1, K + 1):
        s = mp.mpf(k) / 2 * G * rho[k - 1]
        for a in range(1, k):
            b = k - a
            s += mp.mpf(1) / 2 * mp.binomial(k, a) * rho[a] * rho[b] * mp.beta(a * (2 + g) + beta + 1, b * (2 + g) + beta + 1)
        rho.append(s / phi(beta, k * (2 + g) - 1))
    return rho

def central(mom):
    """raw moments [1, m1, m2, m3, m4] -> (mean, var, skew, kurt)"""
    m1, m2, m3, m4 = mom[1], mom[2], mom[3], mom[4]
    v = m2 - m1 ** 2
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    return m1, v, c3 / v ** 1.5, c4 / v ** 2

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print("== beta<-1: Sackin/Colless, S_n/n^{1+g} limit moments ==")
    for beta in (-1.2, -1.5, -1.8):
        mu = mu_beta_lt(beta, K)
        m1, v, sk, ku = central(mu)
        print(f"beta={beta}: mu1={mp.nstr(m1,8)} var={mp.nstr(v,8)} sd/mean={mp.nstr(mp.sqrt(v)/m1,6)} skew={mp.nstr(sk,6)} kurt={mp.nstr(ku,6)}")
        print("   mu_k^(1/k)/k :", [mp.nstr(mu[k] ** (mp.mpf(1) / k) / k, 4) for k in range(1, K + 1)])
    print("   PDA check: var should be 10/3-pi =", mp.nstr(mp.mpf(10) / 3 - mp.pi, 10))
    print("== beta<-1: total cophenetic, Phi_n/n^{2+g} limit moments ==")
    for beta in (-1.2, -1.5, -1.8):
        rho = rho_beta_lt(beta, K)
        m1, v, sk, ku = central(rho)
        print(f"beta={beta}: rho1={mp.nstr(m1,8)} var={mp.nstr(v,8)} sd/mean={mp.nstr(mp.sqrt(v)/m1,6)} skew={mp.nstr(sk,6)} kurt={mp.nstr(ku,6)}")
    print("== beta=-1: Sackin/Colless centred, (S_n-ES_n)/(n log n) limit moments nu_k ==")
    nu = nu_crit_sackin(K)
    print("   nu_k:", [mp.nstr(x, 6) for x in nu])
    print(f"   var={mp.nstr(nu[2],8)} (=(36/pi^4)(2z3-z2)={mp.nstr(36/mp.pi**4*(2*mp.zeta(3)-mp.zeta(2)),8)}) skew={mp.nstr(nu[3]/nu[2]**1.5,6)} kurt={mp.nstr(nu[4]/nu[2]**2,6)}")
    print("   nu_k^(1/k)/k :", [mp.nstr(abs(nu[k]) ** (mp.mpf(1) / k) / k, 4) for k in range(2, K + 1)])
    print("== beta=-1: total cophenetic, Phi_n/(n^2 log n) limit moments rho_k ==")
    rho = rho_crit_coph(K)
    m1, v, sk, ku = central(rho)
    print(f"   rho_k: {[mp.nstr(x,6) for x in rho[:6]]}")
    print(f"   mean={mp.nstr(m1,6)} var={mp.nstr(v,8)} var/mean^2={mp.nstr(v/m1**2,8)} (2/11={mp.nstr(mp.mpf(2)/11,8)}) skew={mp.nstr(sk,6)} kurt={mp.nstr(ku,6)}")
    print("   rho_k^(1/k)/k :", [mp.nstr(rho[k] ** (mp.mpf(1) / k) / k, 4) for k in range(1, K + 1)])
