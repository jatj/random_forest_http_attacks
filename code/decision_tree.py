import numpy as np
from decision_stump import DecisionStumpErrorRate

class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class
    

    def fit(self, X, y):
        # Fits a decision tree using greedy recursive splitting
        N, D = X.shape
        
        # Learn a decision stump
        splitModel = self.stump_class()
        splitModel.fit(X, y)

        if self.max_depth <= 1 or splitModel.splitVariable is None:
            # If we have reached the maximum depth or the decision stump does
            # nothing, use the decision stump

            self.splitModel = splitModel
            self.subModel1 = None
            self.subModel0 = None
            return
        
        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel.splitVariable
        value = splitModel.splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:,j] > value
        splitIndex0 = X[:,j] <= value

        # Fit decision tree to each split
        self.splitModel = splitModel
        self.subModel1 = DecisionTree(self.max_depth-1, stump_class=self.stump_class)
        self.subModel1.fit(X[splitIndex1], y[splitIndex1])
        self.subModel0 = DecisionTree(self.max_depth-1, stump_class=self.stump_class)
        self.subModel0.fit(X[splitIndex0], y[splitIndex0])


    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        # GET VALUES FROM MODEL
        splitVariable = self.splitModel.splitVariable
        splitValue = self.splitModel.splitValue
        splitSat = self.splitModel.splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        # the case with depth=1, just a single stump.
        elif self.subModel1 is None:
            return self.splitModel.predict(X)

        else:
            # Recurse on both sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:,j] > value
            splitIndex0 = X[:,j] <= value

            y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self.subModel0.predict(X[splitIndex0])
        return y
    
    def export(self):
        if self.subModel1 is None and self.subModel0 is None:
            # Return only split model
            return self.splitModel.export()
        else: 
            model = self.splitModel.export()
            model['1'] = self.subModel1.export()
            model['0'] = self.subModel0.export()
            return model

    def load(self, model):
        self.splitModel = self.stump_class()
        self.splitModel.load(model)
        if '0' in model:
            self.subModel0 = DecisionTree(self.max_depth)
            self.subModel0.load(model['0'])
        else:
            self.subModel0 = None
            
        if '1' in model:
            self.subModel1 = DecisionTree(self.max_depth)
            self.subModel1.load(model['1'])
        else:
            self.subModel1 = None