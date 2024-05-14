import ast
from collections import defaultdict
import pandas as pd 
import matplotlib.pyplot as plt 


# Import data 
results_file = "./results/ourtree_200_resuslts.txt"

if __name__ == "__main__":
    with open(results_file) as file:
        results = file.read()
        results = ast.literal_eval(results)

    data = defaultdict()
    size_data = [] 
    struct_data = []
    for key in results.keys():
        a,b = key
        val = results[key]
        trials = val["trials"]
        p_size = val["same_size"]/trials
        p_struct = val["same_split"]/trials

        size_data.append((a,b,p_size))
        struct_data.append((a,b,p_struct))

    size_ = pd.DataFrame(size_data).pivot(columns=0, index=1, values=2).rename_axis(columns="a", index="b")
    struct_ = pd.DataFrame(struct_data).pivot(columns=0, index=1, values=2).rename_axis(columns="a", index="b")


    cols = struct_.columns

    fig, axs = plt.subplots(2,len(cols))

    for i, column in enumerate(cols):
        struct_[column].plot(ax=axs[0,i])
        axs[0,i].set_title(f"a={column}")
        axs[0,i].set_xlabel("")

    for i, column in enumerate(cols):
        struct_[column].plot(ax=axs[1,i])


    """
    fig, axs = plt.subplots(1,2)
    struct_.plot(ax=axs[0])
    axs[0].set_title("Correct structure")
    #axs[0].set_xlabel("b")
    axs[0].set_ylabel("freq")

    size_.plot(ax=axs[1])
    axs[1].set_title("Correct size")
    #axs[1].set_xlabel("b")
    axs[1].set_ylabel("freq")
    """

    plt.tight_layout()
    plt.show()
