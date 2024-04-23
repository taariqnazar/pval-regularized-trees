#! usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
import pandas
import utils


#### seed for random generation ####

np.random.seed(0)

#### Hyperparameters ####

delta = .05


#### Load Data ####

X, Y, feature_names = utils.load_cal_housing()
d = len(feature_names)

mid = int(np.floor(len(Y)*7/8))

X_train, Y_train = X[mid:], Y[mid:]
print(len(Y_train))
X_test, Y_test = X[:mid], Y[:mid]

print('samplesize for mu: ', len(X_train))


#### CART tree ####

K = 10
min_dp_leaf = 10

cart = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf, random_state=0, ccp_alpha = 0).fit(X_train, Y_train)
pred_test = cart.predict(X_test)
pred_train = cart.predict(X_train)
msep_sr = utils.get_MSE(Y_test, pred_test)
mse_sr = utils.get_MSE(Y_train, pred_train)
print(msep_sr, mse_sr)
print(cart.score(X_test, Y_test))
ccp_alphas = cart.cost_complexity_pruning_path(X_train, Y_train)['ccp_alphas']
#print(ccp_alphas)
print('complexity Tstar', len(utils.get_optimal_leaves(cart, delta, d)))

utils.prune_to_T_star(cart, delta, d)
pred_test_T_star = cart.predict(X_test)
MSEP_T_star = utils.get_MSE(Y_test, pred_test_T_star)
print('MSEP_T_star', MSEP_T_star)



#### Plot ####

#tree.plot_tree(cart, fontsize = 7, impurity = False, label = 'none', feature_names = feature_names)

mseps = []
mses = []
R_sqs = []
is_subtree = []
cum_pvals = []
flipped_ccps = np.flip(ccp_alphas)
min_subtree_index_001 = 0
min_subtree_index_05 = 0
min_subtree_index_10 = 0
max_subtree_cum_pvals = 0
#print(flipped_ccps)
for k in range(len(flipped_ccps)):
	cart1 = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf, random_state=0, ccp_alpha = flipped_ccps[k]).fit(X_train, Y_train)
	pred_test = cart1.predict(X_test)
	pred_train = cart1.predict(X_train)
	mseps.append(utils.get_MSE(Y_test, pred_test))
	mses.append(utils.get_MSE(Y_train, pred_train))
	R_sqs.append(cart1.score(X_test, Y_test))
	cum_pval = utils.get_cum_p_val(cart1, d)
	cum_pvals.append(cum_pval)
	#print('cum p_val: ', utils.get_cum_p_val(cart1, d))
	if (cum_pval <= 0.05):
		max_subtree_cum_pvals = k
	if (utils.is_subtree_of_T_star(cart1, 0.0001, d)):
		min_subtree_index_0001 = k
	if (utils.is_subtree_of_T_star(cart1, 0.001, d)):
		min_subtree_index_001 = k
	if (utils.is_subtree_of_T_star(cart1, 0.05, d)):
		min_subtree_index_05 = k
	if (utils.is_subtree_of_T_star(cart1, 0.1, d)):
		min_subtree_index_10 = k
	if (utils.is_subtree_of_T_star(cart1, 0.3, d)):
		min_subtree_index_30 = k


print('index cum pvalue', max_subtree_cum_pvals)
print(min_subtree_index_05)
print(flipped_ccps[min_subtree_index_05]) 
#0.001148998713409591
#95
#0.0006823916342277119

#tree.plot_tree(cart1, fontsize = 7, impurity = False, label = 'none', feature_names = feature_names)


fig, ax = plt.subplots(1, 1)
#print(mseps, mses, R_sqs)

plt.plot(mseps, color = 'blue')
plt.plot(mses, color = 'red')
#plt.plot(R_sqs, color = 'green')
#plt.plot(is_subtree)

plt.axvline(x = min_subtree_index_30, color = 'lightgreen', label = 'delta = 0.3')
plt.axvline(x = min_subtree_index_05, color = 'green', label = 'delta = 0.05')
plt.axvline(x = max_subtree_cum_pvals, color = 'darkgreen', label = 'delta = 0.05, cum_pval')
plt.axhline(y = MSEP_T_star, color = 'purple')
plt.legend()
#plt.hist(Y, bins = 100)



plt.show()


##### Simulation study ####

# Regression function

mu = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf, random_state=0, ccp_alpha = flipped_ccps[min_subtree_index_05]).fit(X_train, Y_train)

true_n_leaves = mu.get_n_leaves()
print('true number leaves', true_n_leaves)
print('mu depth', mu.get_depth())
pred_test = mu.predict(X_test)
msep_sr = np.sqrt(sum([(p - y)**2 for (p, y) in zip(pred_test, Y_test)])/len(Y_test))
print('sigma_hat', msep_sr)
print('R^2: ', mu.score(X_test, Y_test))

#samplesize for mu:  2460
#17
#0.004316358590290614
#true number leaves 21
#sigma_hat 0.6652169979197043
#R^2:  0.5404595264945545

sigma_sq = 0.66**2

np.random.seed(1)

# number of simulations
M = 1

# sample size per simulation
N = 2640

# Responses

#print('R^2: ', cart1.score(X_test, Y_test))

n_leaves = []

for m in range(M):
	X_m = X_train.sample(n=N, replace = True)
	#X_m = X_train
	mus_m = mu.predict(X_m)
	Y_m = []
	for l in range(N):
		Y_m.append(np.random.normal(mus_m[l], sigma_sq))
	Y_m = np.array(Y_m)
	T = DecisionTreeRegressor(max_depth=K, min_samples_leaf = 10).fit(X_m, Y_m)
	n_optimal = len(utils.get_optimal_leaves(T, delta, d))
	n_leaves.append(n_optimal)




#print(len(Y))


plt.hist(n_leaves, bins = 100)
plt.axvline(x = true_n_leaves, color = 'green')
#plt.legend()


#tree.plot_tree(mu, fontsize = 7, impurity = False, label = 'none', feature_names = feature_names)

plt.show()


plt.plot(cum_pvals[:22])
plt.axhline(y = 0.05, color = 'purple')
plt.axvline(x = max_subtree_cum_pvals, color = 'darkgreen', label = 'delta = 0.05, cum_pval')

plt.show()
