from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np

class RandomTree(DecisionTree):
        
    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpInfoGain)

    def fit(self, X, y):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]
        
        DecisionTree.fit(self, bootstrap_X, bootstrap_y)