import numpy as np
from scipy.stats import norm
from sklearn.tree import DecisionTreeRegressor

def P_T(t,n):
    a_n = 1 / (np.sqrt(2*np.log(np.log(n))))
    K_t = t - a_n*(np.log(np.log(np.log(n))) + np.log(2)) / ( np.sqrt(2*np.log(np.log(n))))
    p = (norm.cdf(K_t))**(2*np.log(n/2))
    return p

def P_U(u,n):
	K = np.sqrt(u) - ((2*(np.log(np.log(n))))**(-1/2))*(np.log(np.log(np.log(n))) + np.log(2))
	p = d*(1 - norm.cdf(K)**(2*np.log(n/2)))
	return p

def get_tree_split_pvalue(tr: DecisionTreeRegressor):
    pvalues = get_tree_split_pvalues(tr)
    return np.sum([p for p in pvalues if p > 0])


def get_tree_split_pvalues(tr: DecisionTreeRegressor):
    n_nodes = tr.tree_.node_count
    children_left = tr.tree_.children_left
    children_right = tr.tree_.children_right
    feature = tr.tree_.feature
    threshold = tr.tree_.threshold
    values = tr.tree_.value

    n_samples = tr.tree_.n_node_samples
    impurities = tr.tree_.impurity

    ps = -np.ones(len(values))

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))

            #TODO: Check this
            # Calculate p-vals for this node
            #Impurty = SSE/N
            E = n_samples[node_id]*impurities[node_id] \
                    - (n_samples[children_left[node_id]]*impurities[children_left[node_id]] \
                    + n_samples[children_right[node_id]]*impurities[children_right[node_id]])
            E = np.sqrt(E/impurities[node_id])
            ps[node_id] = 1 - P_T(E, n_samples[node_id]) # P-value for this node
        else:
            is_leaves[node_id] = True
    return ps
