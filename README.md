# Supplementary code: Transfer theorems and the phase diagram of limit laws for Aldous's beta-splitting trees

Python 3 with numpy, scipy, mpmath (`pip install numpy scipy mpmath`). Run scripts from their own folder.

- first_order/numerics.py, fits.py   exact recurrences a_n = sum_j p_{n,j}(a_j+a_{n-j}) + b_n for beta in {-1.9,...,5} and tolls {1, log n, sqrt n, n, n log n, n^{3/2}, n^2}; least-squares leading coefficients vs Theorems A and B (usage: `python3 numerics.py NMAX`)
- first_order/colless_check.py   Sackin vs Colless at beta = -1
- applications/   constants of the applications section (Sackin, Colless, cophenetic, quadratic Colless, cherries); phi_check.py checks phi_beta closed forms
- second_order/moments.py   exact recurrences for central moments 2-4 (writes moments_*.npz); limits.py limit-moment recursions; second_order_check.py second-order transfer checks; mc.py Monte Carlo; summary_table.py tables of the paper

Paper: H. Song, *Transfer theorems and the phase diagram of limit laws for Aldous's beta-splitting trees* (2026), arXiv link to be added.

License: MIT.
