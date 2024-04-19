import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import norm

def generate_neufeldt_data(a,b, n=200, p=10, sigma=5):
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

        tree = DecisionTreeRegressor(random_state=0,max_depth=4)
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

def is_subtree_of_T_star(cart, delta):
    YD_statistics = get_YD_statistics(cart)
    p_values = [Psi(cart.tree_.n_node_samples[k], YD_statistics[k], d) if (YD_statistics[k] != -1) else -1 for k in range(len(YD_statistics))]
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
