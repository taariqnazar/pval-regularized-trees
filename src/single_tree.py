from sklearn.tree import DecisionTreeRegressor

from stats import get_tree_split_pvalue

def train_regression_tree(X,y, **kwargs):
    tree = DecisionTreeRegressor(**kwargs)
    tree.fit(X,y)
    return tree

def recurive_pval_regularised_tree(X,y, signficanced_level):
    tree = train_regression_tree(X,y)
    return tree

def ccp_pval_regularised_tree(X,y, signficanced_level, **kwargs):
    rt = DecisionTreeRegressor(random_state=1, **kwargs)  

    path = rt.cost_complexity_pruning_path(X, y)
    ccps = path.ccp_alphas

    prev_tree = train_regression_tree(X,y, ccp_alpha=ccps[-1], random_state=1,
                                      **kwargs)
    for ccp in ccps[:-1][::-1]:
        tree = train_regression_tree(X,y, ccp_alpha=ccp, random_state=1, **kwargs)
        p_value = get_tree_split_pvalue(tree)
        if p_value < signficanced_level:
            return prev_tree
        prev_tree = tree

    return tree
