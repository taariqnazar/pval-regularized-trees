#! usr/bin/env python3

import numpy as np
import pdb
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.special import gamma
import math

import time



def A(m, c, r, i):
	l = [m[i-1] + k for k in range(r)]
	return np.prod(l)*((1-c[0]/c[i-1])**r)/np.math.factorial(r)

def chi_sq_dist(x, K_acc, df, coffs):
	c = coffs
	n = df
	m = list(np.array(n)/2)
	s = sum(m)
	print(c)
	print(m)

	b = [(c[0]/c[i])**m[i] if c[0]/c[i] > 0 else -(-c[0]/c[i])**m[i] for i in range(3)]
	print(b)
	a = []
	g = []
	for j in range(K_acc):
		fac1 = [A(m, c, k, 2) for k in range(j+1)]
		fac2 = [A(m, c, j-k, 3) for k in range(j+1)]
		a_j = sum([fac1[k]*fac2[k] for k in range(j+1)])
		a.append(a_j)
		g_j = lambda y : (y**(s+j-1))*np.exp(-y/(2*c[0]))/(gamma(s+j)*(2*c[0])**(s+j))
		g_value = quad(g_j, -math.inf, x)[0]
		g.append(g_value)
	print(a)
	print(g)
	return b[1]*b[2]*sum([a[k]*g[k] for k in range(K_acc)])


def chi_sq_cdf(dfs, coffs):
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

def chi_sq_cdf_test(x,dfs, coffs, N):
	 chi_1 = list(np.random.chisquare(dfs[0],N))
	 chi_2 = list(np.random.chisquare(dfs[1],N))
	 chi_3 = list(np.random.chisquare(dfs[2],N))
	 chi_mix = [1 if coffs[0]*chi_1[i] + coffs[1]*chi_2[i] + coffs[2]*chi_3[i] < x else 0 for i in range(N)]
	 return(sum(chi_mix)/N)

def F(r, n_A, n_B):
	dfs = [n_A - 1, n_B - 1, 1]
	coffs = [((n_A + 1)*(n-1) - r*(n+1)*(n_A -1))/(n_A - 1), ((n_B + 1)*(n-1) - r*(n+1)*(n_B -1))/(n_B - 1), - r*(n + 1)]
	return chi_sq_cdf(dfs, coffs)[0]


# instable if r is not close to 1...
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
	#print(p_val, error)
	#print(error/p_val)
	if p_val == 0:
		return 0
	rel_error = error/p_val
	if rel_error > 1/100:
		if r == 1:
			return 0.1573
		if r > 1:
			return 1
		if r < 1:
			return 0
	return max(p_val, 0)

def compute_p_value_old(n_A, n_B, r, N = 100000):
	if n_A < 3 or n_B < 3:
		raise ValueError('There must be at least three data point in each region.')
	if n_A + n_B > 3000:
		if r == 1:
			return 0.1573
		if r < 1:
			return 0
		if r > 1:
			return 1
	if 320 < n_A + n_B <= 3000:
		return chi_sq_cdf_sim(n_A, n_B, r, N)
	return chi_sq_cdf(n_A, n_B, r)[0]


x = 0
r = .5
n_A, n_B = (500000, 500100)
n = n_A + n_B
dfs = [n_A - 1, n_B - 1, 1]
coffs = [((n_A + 1)*(n-1) - r*(n+1)*(n_A -1))/(n_A - 1), ((n_B + 1)*(n-1) - r*(n+1)*(n_B -1))/(n_B - 1), - r*(n + 1)]


#print(chi_sq_cdf(x, dfs, coffs))

#print('Simulated:')
#print(chi_sq_cdf_test(x, dfs, coffs, 1000000))

axis = np.arange(0.01, 2, 0.01)
Fs = [compute_p_value(n_A, n_B, r) for r in axis]
plt.plot(axis, Fs)
plt.show()



#dfs = [2,3,1]
#coffs = [-5,-4, 1]

#print(dfs)
#print(coffs)

#print(chi_sq_cdf(x, dfs, coffs))

'''
(p1, p2, p3) = (dfs[0]/2) - 1 , (dfs[1]/2) - 1, (dfs[2]/2) - 1
(a1, a2, a3) = coffs[0], coffs[1], coffs[2]
print(p1, p2, p3)
const = gamma((sum(dfs))/2)/(gamma(dfs[0]/2)*gamma(dfs[1]/2)*gamma(dfs[2]/2))

print(const)
f = lambda y, x: const * ((1-x-y)**p1)*(y**p2)*(x**p3)
h = lambda x:  (x*(a1 - a3) - a1)/(a2-a1)
g = lambda x:  1-x
zero = lambda x: 0
#x_lim = a1/(a1 - a3)
x_lim = 7
#print('x_lim: ' + str(x_lim))
#print(a2 - a1)
#print(a1 - a3)
t1 = np.arange(0.0, 1.0, 0.01)
plt.plot(t1, h(t1))
#plt.show()
#print("intersection:")
#print((2*a1 - a2)/(a2 - a3))
#print((a1/(a2 - a1) - 1)/((a1-a3)/(a2-a3)))

if(a2 - a1 > 0):
	# u is upper bound
	u = lambda x:  np.maximum(np.minimum(h(x), g(x)), 0)
	if(a1 - a3 > 0):
		print('Case 1a')
		t1 = np.arange(0.0, 1.0, 0.001)
		plt.plot(t1, u(t1))
		plt.show()
		print(  dblquad(f, 0, 1, 0, u)[0])
		print( dblquad(f, 0, 1, 0, u))
	else:
		print('Case 1b')
		t1 = np.arange(0.0, 1.0, 0.001)
		plt.plot(t1, u(t1))
		plt.show()
		print(  dblquad(f, 0, 1, 0, u)[0])
		print( dblquad(f, 0, 1, 0, u))
else:
	# l2 is lower bound
	l1 = lambda x:  np.maximum(h(x), 0)
	l2 = lambda x: np.minimum(l1(x), g(x))
	if(a1 - a3 > 0):
		print('Case 2a')
		#print('Integrating from 0 to x_lim')
		t1 = np.arange(0.0, 1.0, 0.001)
		plt.plot(t1, l2(t1))
		plt.show()
		print(  dblquad(f, 0, 1, l2, g)[0])
		print( dblquad(f, 0, 1, l2, g))
	else:
		print('Case 2b')
		t1 = np.arange(0.0, 1.0, 0.001)
		plt.plot(t1, l(t1))
		plt.show()
		print(  dblquad(f, 0, 1, l2, g)[0])
		print(dblquad(f, 0, 1, l2, g))

'''


#print(chi_sq_dist(x, K_acc, df, coffs))






