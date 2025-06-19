import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from sklearn.datasets import fetch_california_housing
import pandas

def generate_neufeldt_data(a,b, n=200, p=10, sigma=5.):
    """
    Generate the Neufeldt data, given paramters:
        a,b,n,sigma
    """
    X = np.random.normal(0,1,(n,p))
    mu = 1 + a*(X[:,1]>0) + ((X[:,2]*X[:,1]) > 0)
    mu = (X[:,0] <= 0)*mu
    mu = b*mu
    y = mu + np.random.normal(0, sigma, n)
    return X,y 

def build_ccp_model(X,y, cv_frac=0.5):
    """
    Build a tree using CCP and cross validation
    parameters:
        X, y: dataset 
        cv_frac: fraction of data that should be used for valdiation
    return:
        tree
    """
    if cv_frac>0.0:
        X_train, X_val, y_train, y_val = train_test_split(X,y,
                                                          train_size=cv_frac)

        tree = DecisionTreeRegressor(random_state=0,max_depth=5)
        path = tree.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas

        trees = []
        for ccp in ccp_alphas:
            t_ = DecisionTreeRegressor(random_state=0, ccp_alpha=ccp,
                                       max_depth=4)
            t_.fit(X_train,y_train)
            trees.append(t_)
        scores = [t.score(X_val, y_val) for t in trees]
        max_ind = np.argmax(scores)
        return trees[max_ind]
    else:
        return 0

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

def is_subtree_of_T_star(cart, delta, d):
    YD_statistics = get_YD_statistics(cart)
    p_values = get_p_values(cart, d)
    number_of_acceptances = sum([1 if p > delta else 0 for p in p_values])
    if(number_of_acceptances > 0):
        return False
    else:
        return True

def Psi(n, u, d):
    K = np.sqrt(u) - ((2*(np.log(np.log(n))))**(-1/2))*(np.log(np.log(np.log(n))) + np.log(2))
    p = d*(1 - norm.cdf(K)**(2*np.log(n/2)))
    return p

def get_p_values(cart, d):
    YD_statistics = get_YD_statistics(cart)
    return [Psi(cart.tree_.n_node_samples[k], YD_statistics[k], d) if (YD_statistics[k] != -1) else -1 for k in range(len(YD_statistics))]

def get_optimal_leaves(cart, delta, d):
    YD_statistics = get_YD_statistics(cart)
    p_values = get_p_values(cart, d)
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

def prune(nodes_to_prune, tree):
    while nodes_to_prune != []:
        n = nodes_to_prune.pop()
        if(tree.tree_.children_left[n] != -1):
            nodes_to_prune.append(tree.tree_.children_left[n])
            tree.tree_.children_left[n] = -1
        if(tree.tree_.children_right[n] != -1):
            nodes_to_prune.append(tree.tree_.children_right[n])
            tree.tree_.children_right[n] = -1

def get_MSE(Y, pred):
    return sum([(y - p)**2 for (y,p) in zip(Y, pred)])/len(Y)

def prune_to_T_star(cart, delta, d):
    optimal_leaves = get_optimal_leaves(cart, delta, d)
    prune(optimal_leaves, cart)

def get_leaves(tree):
    children_left = tree.tree_.children_left
    leaves = []
    for k in range(len(children_left)):
        if children_left[k] == -1:
            leaves.append(k)
    return leaves

def get_cum_p_val(tree, d):
    p_vals = get_p_values(tree, d)
    return sum([p for p in p_vals if p != -1])


def load_cal_housing():
    housing = fetch_california_housing()
    feature_names =  housing.feature_names
    df = pandas.DataFrame(housing.data)
    df['MedHouseVal'] = housing.target
    df = df[df['MedHouseVal'] <= 5]  
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.loc[:, df.columns != 'MedHouseVal']
    Y = df['MedHouseVal']
    return X, Y, feature_names


def build_our_model(X,y, significance=0.1):
    d = X.shape[1]
    tree = DecisionTreeRegressor(random_state=0, max_depth=4)
    tree.fit(X,y)

    ntp = [index for index, value in enumerate(get_ps(tree) >=significance/d) if value]

    prune(tree, ntp)
    return tree

def P_T(t,n):
    a_n = 1 / (np.sqrt(2*np.log(np.log(n))))
    K_t = t - a_n*(np.log(np.log(np.log(n))) + np.log(2)) / ( np.sqrt(2*np.log(np.log(n))))
    p = (norm.cdf(K_t))**(2*np.log(n/2))
    return p

def get_ps(tr, metric=False):
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

            # Calculate p-vals for this node
            #Impurty = SSE/N
            E = n_samples[node_id]*impurities[node_id] \
                    - (n_samples[children_left[node_id]]*impurities[children_left[node_id]] \
                    + n_samples[children_right[node_id]]*impurities[children_right[node_id]])
            E = np.sqrt(E/impurities[node_id])
            ps[node_id] = 1 - P_T(E, n_samples[node_id])
        else:
            is_leaves[node_id] = True
    return ps

def prune(tr,nodes_to_prune):
    """
    mutates the original tree!
    """
    for n in nodes_to_prune:
        tr.tree_.children_left[n] = -1
        tr.tree_.children_right[n] = -1

def get_leaf_nodes(tree):
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [0]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id  = stack.pop()

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append(children_left[node_id])
            stack.append(children_right[node_id])
        else:
            is_leaves[node_id] = True


    return is_leaves
