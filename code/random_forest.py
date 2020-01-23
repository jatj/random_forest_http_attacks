from random_tree import RandomTree
import numpy as np
import utils
import json

class RandomForest():

    def __init__(self, max_depth, num_trees):
        self.max_depth = max_depth
        self.num_trees = num_trees

    def fit(self, X, y):
        N, D = X.shape
        trees = [None] * self.num_trees
        for i in range(self.num_trees):
            model = RandomTree(max_depth=self.max_depth)
            model.fit(X, y)
            trees[i] = model
        self.trees = trees

    def predict(self, Xtest):
        T, D = Xtest.shape
        predictions = np.zeros([T, self.num_trees])
        for j in range(self.num_trees):
            predictions[:, j] = self.trees[j].predict(Xtest)
        predictions_mode = np.zeros(T)
        for i in range(T):
            predictions_mode[i] = utils.mode(predictions[i,:])
        return predictions_mode

    def export(self):
        trees = []
        for i in range(self.num_trees):
            trees.append(self.trees[i].export())
        
        return json.dumps({
            "model": trees
        })

    def load(self, filename):
        file = open(filename, "r")
        jsonModel = file.read()
        model = json.loads(jsonModel)["model"]
        self.trees = []
        for i in range(len(model)):
            tree = RandomTree(max_depth=self.max_depth)
            tree.load(model[i])
            self.trees.append(tree)
        file.close()