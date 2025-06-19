from sklearn.tree import DecisionTreeRegressor

from utils import get_cum_p_val


def train_regression_tree(X, y, **kwargs):
    tree = DecisionTreeRegressor(**kwargs)
    tree.fit(X, y)
    return tree

def ccp_pval_regularised_tree(X, y, significance_level, **kwargs):
    rt = DecisionTreeRegressor(**kwargs)
    d = X.shape[1]

    path = rt.cost_complexity_pruning_path(X, y)
    try: 
        ccps = path.ccp_alphas[1:]
        prev_tree = train_regression_tree(X, y, ccp_alpha=ccps[-1], **kwargs)
        for ccp in ccps[::-1][1:]:
            tree = train_regression_tree(X, y, ccp_alpha=ccp, **kwargs)
            p_value = get_cum_p_val(tree, d)
            if p_value > significance_level:
                return prev_tree
            prev_tree = tree

        return tree
    except:
        ccp = path.ccp_alphas[-1]
        tree = train_regression_tree(X, y, ccp_alpha=ccp, **kwargs)
        return tree

