#! usr/bin/env python3

import numpy as np
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.special import gamma
import time

def chi_sq_cdf(n_A, n_B, r):
	n = n_A + n_B
	dfs = [n_A - 1, n_B - 1, 1]
	coffs = [((n_A + 1)*(n-1) - r*(n+1)*(n_A -1))/(n_A - 1), ((n_B + 1)*(n-1) - r*(n+1)*(n_B -1))/(n_B - 1), - r*(n + 1)]
	if(dfs[0] == dfs[1]):
		dfs = [dfs[0] + dfs[1], 1]
		coffs = [coffs[0], coffs[2]]
		if(coffs[1] - coffs[0] < 0):
			coffs[0], coffs[1] = coffs[1], coffs[0]
			dfs[0], dfs[1] = dfs[1], dfs[0]
		(p1, p2) = (dfs[0]/2) - 1 , (dfs[1]/2) - 1
		(a1, a2) = coffs[0], coffs[1]
		const = gamma((sum(dfs))/2)/(gamma(dfs[0]/2)*gamma(dfs[1]/2))
		f = lambda z: const * ((1-z)**p1)*(z**p2)
		upper = np.maximum(np.minimum(1, a1/(a1 - a2)), 0)
		return quad(f, 0, upper)
	if(coffs[1] - coffs[0] < 0):
		coffs[0], coffs[1] = coffs[1], coffs[0]
		dfs[0], dfs[1] = dfs[1], dfs[0]
	(p1, p2, p3) = (dfs[0]/2) - 1 , (dfs[1]/2) - 1, (dfs[2]/2) - 1
	(a1, a2, a3) = coffs[0], coffs[1], coffs[2]
	const = gamma((sum(dfs))/2)/(gamma(dfs[0]/2)*gamma(dfs[1]/2)*gamma(dfs[2]/2))
	f = lambda y, x: const * ((1-x-y)**p1)*(y**p2)*(x**p3)
	h = lambda x:  (x*(a1 - a3) - a1)/(a2-a1)
	g = lambda x:  1-x
	upper = lambda x:  np.maximum(np.minimum(h(x), g(x)), 0)
	return  dblquad(f, 0, 1, 0, upper)

def chi_sq_cdf_sim(n_A, n_B, r, N):
	n = n_A + n_B
	dfs = [n_A - 1, n_B - 1, 1]
	coffs = [((n_A + 1)*(n-1) - r*(n+1)*(n_A -1))/(n_A - 1), ((n_B + 1)*(n-1) - r*(n+1)*(n_B -1))/(n_B - 1), - r*(n + 1)]
	chi_1 = list(np.random.chisquare(dfs[0],N))
	chi_2 = list(np.random.chisquare(dfs[1],N))
	chi_3 = list(np.random.chisquare(dfs[2],N))
	chi_mix = [1 if coffs[0]*chi_1[i] + coffs[1]*chi_2[i] + coffs[2]*chi_3[i] < 0 else 0 for i in range(N)]
	return(sum(chi_mix)/N)

	
def compute_p_value(n_A, n_B, r):
	if n_A < 2 or n_B < 2:
		raise ValueError('There must be at least two data point in each region.')
	n = n_A + n_B
	dfs = [n_A - 1, n_B - 1, 1]
	coffs = np.array([((n_A + 1)*(n-1) - r*(n+1)*(n_A -1))/(n_A - 1), ((n_B + 1)*(n-1) - r*(n+1)*(n_B -1))/(n_B - 1), - r*(n + 1)])/(n-1)
	(p1, p2, p3) = -dfs[0]/2 , -dfs[1]/2, -dfs[2]/2
	(a1, a2, a3) = coffs[0], coffs[1], coffs[2]
	phi = lambda t: ((1 - 2*a1*1j*t)**p1)*((1 - 2*a2*1j*t)**p2)*((1 - 2*a3*1j*t)**p3)
	integrand = lambda t: np.imag(phi(t))/t
	(integral, error) = quad(integrand, 0, float("inf"))
	p_val = 1/2 - integral/np.pi
	if p_val == 0:
		return 0
	rel_error = abs(error/p_val)
	if rel_error > 1/10:
		if r == 1:
			return 0.1573
		if r > 1:
			return 1
		if r < 1:
			return 0
	return max(p_val, 0)


r = .99
n_A, n_B = (500, 501)
#print(compute_p_value(n_A,n_B, r))

t1 = time.time()
result_computed = chi_sq_cdf(n_A, n_B, r)
t2 = time.time()
result_simulated = chi_sq_cdf_sim(n_A, n_B, r,  1000000)
t3 = time.time()
result_computed_fast = compute_p_value(n_A, n_B, r)
t4 = time.time()

print('Computed value took ' + str(t2 - t1) + ' and is given by:')
print(result_computed)
print('Simulated value took ' + str(t3 - t2) + ' and is given by:')
print(result_simulated)
print('Fast Computed value took ' + str(t4 - t3) + ' and is given by:')
print(result_computed_fast)


