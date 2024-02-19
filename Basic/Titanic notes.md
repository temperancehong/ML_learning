# Random Forest

Supervised learning.

Forest, an ensemble of Decision Trees. Bagging method.

For both **classification** and **regression** problem.

Random Forest brings extra randomness into the model when it is growing the trees. 
Instead of searching for the best feature while splitting a node, it searches for the best feature among a random subset of features,
creating a wide diversity.

Random Forest makes it very easy to measure the relative importance of each feature 
by looking at how much the tree nodes, that use that feature, reduce impurity on average (across all trees in the forest).

