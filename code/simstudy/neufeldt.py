"""
# The general flow of test
-Import data
-Build model
-Evaluate model 
-Plot Models

Hothorn:
    - Unbiased selection rule( This follows for us from Shih )
    - Suffers from overfitting
    - Prediction accuracy in comparison to something?
"""

import numpy as np 
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from utils import generate_neufeldt_data, build_ccp_model

"""
def contingency_table(tree: DecisionTreeRegressor, X, true_split):
    axis, point = true_split
    n = X.shape[0]

    true_region = np.zeros(n)
    true_region[ X[:, axis] <= point ] = 1

    data_path = tree.decision_path(X).toarray()
    model_region = data_path[:,1] 

    table = np.zeros((2,2))
    for i in range(n):
        r,c = int(true_region[i]), int(model_region[i])
        table[r,c] += 1

    table /= n
    print((tree.tree_.feature[0], tree.tree_.threshold[0]))
    return table
"""

if __name__ == '__main__':
    print("Neufeldt simulation study, comparison to our tree")
    a = [0.5,1,2]
    b = [i+1 for i in range(10)]
    for a_ in a:
        for b_ in b:
            print(f"results for a={a_}, b={b_}")
            node_size=[]
            true_size = 0
            true_struct = 0 
            trials = 500
            for k in range(trials):
                print(f"{k+1}/{trials}", end="\r")
                X,y = generate_neufeldt_data(a_,b_, n=200)
                tree = build_ccp_model(X,y)
                size = tree.tree_.node_count 
                leaves = (size + 1)/2
                node_size.append(leaves)

                # Check if size and or struct is same
                if leaves == 5:
                    true_size +=1

                    children_right = tree.tree_.children_right
                    children_left = tree.tree_.children_left
                    feature = tree.tree_.feature

                    # Check for Neufeldt structure
                    # Maybe find a better way to check if struct is the same
                    node_of_interest = 0
                    if feature[node_of_interest] == 0:
                        node_of_interest=children_left[node_of_interest]

                        if feature[node_of_interest] == 1:

                            l = children_left[node_of_interest]
                            r = children_right[node_of_interest]

                            if feature[l] == 2 and feature[r]==2:
                                    true_struct +=1
                        

            p_size = true_size/trials
            p_struct = true_struct/trials

            break
        break

    print(f"Fraction of correct size: {p_size:2f}")
    print(f"Fraction of true struct: {p_struct:2f}")
    plt.hist(node_size)
    plt.show()
    print("Done")

