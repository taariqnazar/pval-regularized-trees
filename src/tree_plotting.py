from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

# Sample Data
X = np.array([[i] for i in range(1, 11)])
y = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])

# Train a Regression Tree
reg_tree = DecisionTreeRegressor(max_depth=3)
reg_tree.fit(X, y)


# Export the tree as a DOT file with rpart-like styling
dot_data = export_graphviz(
    reg_tree,
    out_file=None,
    feature_names=["X"],
    filled=True,
    rounded=False,  # rpart uses rectangular nodes
    special_characters=True,
    impurity=False,  # Hide impurity for a cleaner look
    proportion=True,  # Show proportions instead of counts
)

# Render with Graphviz
graph = graphviz.Source(dot_data)
graph.render("rpart_style_tree", format="png", cleanup=True)
graph.view()
