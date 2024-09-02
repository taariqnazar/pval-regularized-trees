import numpy as np
import pandas as pd 
import sys
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

from utils import get_p_values

def generate_data(N, alphas, sample_sizes, signals):
    # Get frequency of splits made.
    data = []
    i = 0
    tot_iter = N*len(sample_sizes)*len(signals)*len(alphas)
    for alpha in alphas:
        for size in sample_sizes:
            for signal in signals:
                cntr = 0 
                for _ in range(N):
                    # Generate data, study the number of times splits are made.
                    X = np.random.uniform(-1,1, size=(size,1))
                    y = np.random.normal(size=size) + signal*(X[:,0]>0.5)
                    
                    
                    tree = DecisionTreeRegressor(random_state=0,max_depth=1)
                    tree.fit(X,y)

                    p = get_p_values(tree, X.shape[1])[0]
                    if p <= alpha:
                        cntr+=1

                    print('\r', f'Completed: {i+1}/{tot_iter}', sep='', end='')
                    sys.stdout.flush()
                    i+=1

                freq = cntr/N
                #print(f"Frequency for sample-size: {size} - signal:{signal}: {freq}")
                data.append((alpha, size, signal, freq))

    with open("signal-size.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ =="__main__":
    print("This is the single split test.")


    N = 10000
    alphas = [0.001,0.005, 0.01,0.05,0.1]
    sample_sizes = [10, 100, 1000, 1000, 10000 ,100000]
    signals = [0.01, 0.1, 1, 10, 100]
    #generate_data(N, alphas, sample_sizes, signals)

    with open("signal-size.json", "r") as f:
       data = json.load(f)

    df = pd.DataFrame(data, columns=["alpha", "sample_size", "signal",
                                     "freq"])#.set_index(["alpha", "sample_size",
                                             #            "signal"])

    
    print(df.pivot_table(columns=["signal"], index=["alpha", "sample_size"],
                         aggfunc="first"))

    #print(df.to_latex(float_format="{:0.2f}".format))


    """    for alpha in df.index.get_level_values("alpha").unique():
        print(df.loc[alpha].to_latex(float_format="{:0.2f}".format, index=True))

        input("Press Enter to continue...")
"""

