
#! usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
import utils
#import dtreeviz


#### seed for random generation ####

np.random.seed(20)

#### Hyperparameters ####

N = 10000
delta = .05


# 10-dimensional iid Gaussian
d = 10

X = np.random.normal(0, 1, size = (N, d))


#### Regression functions ####

# Neufeld example
b = 1
a = 2
mu = lambda x: b*(1 if x[0] <= 0 else 0)*(1 + a*(1 if x[1] > 0 else 0) + (1 if x[1]*x[2] > 0 else 0))


#### Responses ####

sigma_sq = 1**2
mus = [mu(x) for x in X]
Y = np.random.normal(mus, sigma_sq, N)


#### CART tree ####

K = 10
min_dp_leaf = 10
cart = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf).fit(X, Y)


#### Validation #####

X_test = np.random.normal(0, sigma_sq, size = (N, d))
mus_test = [mu(x) for x in X_test]
Y_test = np.random.normal(mus_test, sigma_sq, N)
#print('Score: ', cart.score(X_test, Y_test))

pred_test = cart.predict(X_test)

msep = utils.get_MSE(Y_test, pred_test)

print(msep)



#p_values = utils.get_p_values(cart, d)
#print(p_values)


optimal_leaves = utils.get_optimal_leaves(cart, delta, d)

#print(optimal_leaves)


#tree.plot_tree(cart, fontsize = 7, impurity = False, label = 'none')

utils.prune(optimal_leaves, cart)
#tree.plot_tree(cart, fontsize = 7, impurity = False, label = 'none')

pred_test = cart.predict(X_test)

msep = utils.get_MSE(Y_test, pred_test)

print(msep)

###### Simulation #########

M = 500
precision_rates = []

Ns = np.arange(100, 10000, 100)

for N in Ns:

	mseps = []
	number_leaves = []

	precision_rate = 0

	for m in range(M):
		X_m = np.random.normal(0, sigma_sq, size = (N, d))
		mus_m = [mu(x) for x in X_m]
		Y_m = np.random.normal(mus_m, sigma_sq, N)
		tree = DecisionTreeRegressor(max_depth=K, min_samples_leaf = min_dp_leaf).fit(X_m, Y_m)
		n_leaves = len(utils.get_optimal_leaves(tree, delta, d))
		number_leaves.append(n_leaves)
		if n_leaves == 5:
			precision_rate +=1
		utils.prune_to_T_star(tree, delta, d)
		pred_test = tree.predict(X_test)
		mseps.append(utils.get_MSE(Y_test, pred_test))

	precision_rate = precision_rate/M
	precision_rates.append(precision_rate)

	print('Precision rate' , precision_rate)





#print(mseps)

#print(number_leaves)




#### Plots ####

plt.plot(Ns, precision_rates)

#plt.hist(mseps, bins = 100)

#plt.hist(number_leaves, bins = 100)




# Plot tree

#tree.plot_tree(cart, fontsize = 7, impurity = False, label = 'none')
#features_names = ['X1', 'X2', 'X3', 'X4','X5', 'X6','X7', 'X8', 'X9', 'X10']


# dtreeviz 
'''
viz_rmodel = dtreeviz.model(cart, X, Y, feature_names= features_names,target_name='Y')
v = viz_rmodel.view(depth_range_to_display=(0, 2), scale=2)
v.save('img_url.svg')
'''





plt.show()





	



