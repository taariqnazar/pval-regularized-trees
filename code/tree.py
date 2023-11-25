from sklearn.tree import DecisionTreeRegressor 
import numpy as np

from .chi_sq_distr_func import chi_sq_cdf_sim

class StatisticalRT(DecisionTreeRegressor):
    """
    Same as DecisionTreeRegressor but with extra methods. 
    """

    # Store ccp_alphas when training
    #def fit():
    # self.ccp_alphas = ...
    #    pass

    def get_table(self, significance):
        """
        returns significance and Significance of tree 
        """
        pvals, splitting_sequence, _ = self._get_metrics()
        alphas = np.cumsum(pvals[splitting_sequence])

        # Fix below, analyse what significance corresponds to alpha, right now there is a lenght missmatch
        return {"significance":alphas, "cost-complexity":self.ccp_alphas[::-1]}

    
    def _get_metrics(self):
        """
        Retrieves P-values for all split nodes of the tree 
        """
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        impurities = self.tree_.impurity
        sample_size = self.tree_.n_node_samples
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

        # Populate values
        ratios = -1*np.ones(shape=n_nodes, dtype=np.float64) # Ratio drop of making split per node
        pvals = -1*np.ones(shape=n_nodes, dtype=np.float64) # P-values based on rations per node
        reduction = np.nan*np.ones(shape=n_nodes, dtype=np.float64) # the absolute reduction of total mse if split is made in node
        # The reduction enables us to decide the split sequence
        a
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
                
                mse = impurities[node_id]
                mse_left = impurities[children_left[node_id]]
                mse_right = impurities[children_right[node_id]]
            
                n = sample_size[node_id]
                n_left = sample_size[children_left[node_id]]
                n_right = sample_size[children_right[node_id]]

                red = (n_left*mse_left + n_right*mse_right - n*mse)/10 # DIVISOR WRONG!!! MUST BE LENGHT OF FITTING DATASET, OK to get split sequence 
                reduction[node_id] = red

                # Make the Variance unbiased
                mse *= n/(n-1)
                mse_left *= n_left/(n_left-1)
                mse_right *= n_right/(n_right-1) 
                
                msep1 = (n_left + 1)*mse_left + (n_right + 1)*mse_right
                msep0 = (n + 1)*mse
                r = msep1/msep0
                ratios[node_id] = r
            

                # Calculate P-values here
                l_neg = -r*(n + 1)/(n-1)
                l_left = (n_left + 1)/(n_left - 1) + l_neg
                l_right = (n_right + 1)/(n_right - 1) + l_neg

                #p = chi_sq_cdf(0, [n_left - 1, n_right - 1, 1] , [l_left, l_right, l_neg])
                #pvals[node_id] = p[0]
                
                p = chi_sq_cdf_sim(0, [n_left - 1, n_right - 1, 1] , [l_left, l_right, l_neg], int(1e5))
                pvals[node_id] = p
            else:
                is_leaves[node_id] = True

        splitting_sequence = np.argsort(reduction)[:-np.sum(np.isnan(reduction))] # order and remove the leaf indices, LEAF=NAN
        return pvals, splitting_sequence, ratios

if __name__ == '__main__':
    
    pass
