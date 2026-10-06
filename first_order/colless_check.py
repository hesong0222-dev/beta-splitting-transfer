import numpy as np, math
from scipy.special import gammaln
# beta=-1 : Sackin (toll n) vs Colless (toll E|n-2J|), check (E S_n - E C_n)/(n log n) -> 12 log2/pi^2
NMAX=30000
n_arr=np.arange(NMAX+1,dtype=float)
H=np.zeros(NMAX+1); H[1:]=np.cumsum(1/n_arr[1:])   # H[m]=h_m
S=np.zeros(NMAX+1); C=np.zeros(NMAX+1)
for n in range(2,NMAX+1):
    j=np.arange(1,n,dtype=float)
    p=n/(2*H[n-1]*j*(n-j))
    conv=2*np.dot(p,S[1:n]); S[n]=conv+n
    toll=np.dot(p,np.abs(n-2*j))
    C[n]=2*np.dot(p,C[1:n])+toll
for n in [1000,3000,10000,30000]:
    print(n, S[n]/(n*math.log(n)**2), C[n]/(n*math.log(n)**2), (S[n]-C[n])/(n*math.log(n)), 12*math.log(2)/math.pi**2)
