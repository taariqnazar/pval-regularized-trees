
#! usr/bin/env python3

import numpy as np
from scipy.stats import norm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


#### seed for random generation ####

np.random.seed(0)

#### Hyperparameters ####

N = 200
delta = .05



#### Covariate generation ####
'''
# 2-dimensional Gaussian copula
X_mean = [0, 0]
X_cov = [[5, 3], [3, 10]]
X = np.random.multivariate_normal(X_mean, X_cov, N)
X = np.c_[norm.cdf(X[:,0], loc=X_mean[0], scale=X_cov[0][0]), norm.cdf(X[:,1], loc=X_mean[1], scale=X_cov[1][1])
'''

# 10-dimensional iid Gaussian
X_mean = np.zeros(10)
X_cov = np.identity(10)
X = np.random.multivariate_normal(X_mean, X_cov, N)

d = 10


#### Regression functions ####

# Neufeld example
b = 1
a = 1
mu = lambda x: b*(1 if x[0] <= 0 else 0)*(1 + a*(1 if x[1] > 0 else 0) + (1 if x[1]*x[2] > 0 else 0))


#### Responses ####

sigma_sq = 1
mus = [mu(x) for x in X]
Y = np.random.multivariate_normal(mus, sigma_sq * np.identity(N))


#### CART tree ####

K = 4
min_dp_leaf = 2
cart = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf).fit(X, Y)


#### Yao Davis test statistic computation ###

def get_YD_statistics(cart):
	YD_statistics = []
	node_count = cart.tree_.node_count
	children_left = cart.tree_.children_left
	children_right = cart.tree_.children_right
	impurities = cart.tree_.impurity
	sample_size = cart.tree_.n_node_samples
	for k in range(node_count):
		if(children_left[k] == -1):
			YD_statistics.append(-1)
			continue
		n = sample_size[k]
		n_left = sample_size[children_left[k]]
		n_right = sample_size[children_right[k]]
		S = n*impurities[k]
		S_left = n_left*impurities[children_left[k]]
		S_right = n_right*impurities[children_right[k]]
		YD_statistic = (S - S_left - S_right)/(S/n)
		YD_statistics.append(YD_statistic)		
	return YD_statistics


YD_statistics = get_YD_statistics(cart)
print(YD_statistics)


#### p-values ####

def Psi(n, u, d):
	K = np.sqrt(u) - ((2*(np.log(np.log(n))))**(-1/2))*(np.log(np.log(np.log(n))) + np.log(2))
	p = d*(1 - norm.cdf(K)**(2*np.log(n/2)))
	return p

p_values = [Psi(cart.tree_.n_node_samples[k], YD_statistics[k], d) if (YD_statistics[k] != -1) else -1 for k in range(len(YD_statistics))]
print(p_values)

#### Optimal tree ####

def get_optimal_leaves(p_values, cart):
	stack = [0]
	leaves_in_optimal_tree = []
	while stack != []:
		node = stack.pop()
		if p_values[node] > delta or p_values[node] == -1:
			leaves_in_optimal_tree.append(node)
		else:
			stack.append(cart.tree_.children_right[node])
			stack.append(cart.tree_.children_left[node])
	return leaves_in_optimal_tree


optimal_leaves = get_optimal_leaves(p_values, cart)






#Continue here and check if p_values work well!!!!



'''
#print(clf.apply(X))
print(clf.tree_.node_count)
print(clf.tree_.children_left)
print(clf.tree_.children_right)
print(clf.tree_.impurity)
#print(clf.tree_.n_node_samples)
'''      

#Return the index of the leaf that each sample is predicted as.

#print(clf.cost_complexity_pruning_path(X, Y))
#Compute the pruning path during Minimal Cost-Complexity Pruning.
#print(clf.decision_path(X[0].reshape(1, -1)))
#Return the decision path in the tree.


#get_metadata_routing()
#Get metadata routing of this object.

#print(clf.get_params())










#### Plots ####

# Plot covariates

fig, ax = plt.subplots(1, 1)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
#ax.set_xlabel('X_1')
#ax.set_ylabel('X_2')

#ax.scatter(X[:,2], Y, s=2)


# Plot tree

tree.plot_tree(cart, fontsize = 7	)


plt.show()





	



