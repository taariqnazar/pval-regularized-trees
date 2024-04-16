import numpy as np 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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
