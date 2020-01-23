import numpy as np
import utils

class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 
        
        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return
            
        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] > value])
                y_not = utils.mode(y[X[:,d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat

    def export(self):
        model = {}
        if self.splitVariable is not None:
            model['d'] = int(self.splitVariable)
        if self.splitValue is not None:
            model['x'] = int(self.splitValue)
        if self.splitSat is not None:
            model['s'] = int(self.splitSat)
        if self.splitNot is not None:
            model['n'] = int(self.splitNot)
        return model

    def load(self, model):
        if 'd' in model:
            self.splitVariable = np.int64(model['d'])
        else: 
            self.splitVariable = None
        if 'x' in model:
            self.splitValue = np.int64(model['x'])
        else: 
            self.splitValue = None
        if 's' in model:
            self.splitSat = np.int64(model['s'])
        else:
            self.splitSat = None
        if 'n' in model:
            self.splitNot = np.int64(model['n'])
        else:
            self.splitNot = None



class DecisionStumpEquality:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y)    
        
        # Get the index of the largest value in count.  
        # Thus, y_mode is the mode (most popular value) of y
        y_mode = np.argmax(count) 

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        X = np.round(X)

        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]

                # Find most likely class for each split
                y_sat = utils.mode(y[X[:,d] == value])
                y_not = utils.mode(y[X[:,d] != value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] != value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # This is the lowest error, store this value
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not

    def predict(self, X):

        M, D = X.shape
        X = np.round(X)

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] == self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat

    def export(self):
        model = {}
        if self.splitVariable is not None:
            model['d'] = int(self.splitVariable)
        if self.splitValue is not None:
            model['x'] = int(self.splitValue)
        if self.splitSat is not None:
            model['s'] = int(self.splitSat)
        if self.splitNot is not None:
            model['n'] = int(self.splitNot)
        return model

    def load(self, model):
        if 'd' in model:
            self.splitVariable = np.int64(model['d'])
        else: 
            self.splitVariable = None
        if 'x' in model:
            self.splitValue = np.int64(model['x'])
        else: 
            self.splitValue = None
        if 's' in model:
            self.splitSat = np.int64(model['s'])
        else:
            self.splitSat = None
        if 'n' in model:
            self.splitNot = np.int64(model['n'])
        else:
            self.splitNot = None


"""
A helper function that computes the entropy of the 
discrete distribution p (stored in a 1D numpy array).
The elements of p should add up to 1.
This function ensures lim p-->0 of p log(p) = 0
which is mathematically true, but numerically results in NaN
because log(0) returns -Inf.
"""
def entropy(p):
    plogp = 0*p # initialize full of zeros
    plogp[p>0] = p[p>0]*np.log(p[p>0]) # only do the computation when p>0
    return -np.sum(plogp)
    
# This is not required, but one way to simplify the code is 
# to have this class inherit from DecisionStumpErrorRate.
# Which methods (init, fit, predict) do you need to overwrite?
class DecisionStumpInfoGain(DecisionStumpErrorRate):

    def fit(self, X, y, split_features=None):
                
        N, D = X.shape

        # Address the trivial case where we do not split
        count = np.bincount(y)

        # Compute total entropy (needed for information gain)
        p = count/np.sum(count); # Convert counts to probabilities
        entropyTotal = entropy(p)

        maxGain = 0
        infoGain = 0
        self.splitVariable = None
        self.splitValue = None
        self.splitSat = np.argmax(count)
        self.splitNot = None

        # Check if labels are not all equal
        if np.unique(y).size <= 1:
            return

        if split_features is None:
            split_features = range(D)
        
        for d in split_features:
            thresholds = np.unique(X[:,d])
            for value in thresholds[:-1]:
                # Count number of class labels where the feature is greater than threshold
                y_vals = y[X[:,d] > value]
                count1 = np.bincount(y_vals, minlength=len(count))
                # count1 = np.pad(count1, (0,len(count)-len(count1)), \
                #                 mode='constant', constant_values=0)  # pad end with zeros to ensure same length as 'count'
                count0 = count-count1
                                
                # Compute infogain
                p1 = count1/np.sum(count1)
                p0 = count0/np.sum(count0)
                H1 = entropy(p1)
                H0 = entropy(p0)
                prob1 = np.sum(X[:,d] > value)/N
                prob0 = 1-prob1

                infoGain = entropyTotal - prob1*H1 - prob0*H0
                # assert infoGain >= 0
                # Compare to minimum error so far
                if infoGain > maxGain:
                    # This is the highest information gain, store this value
                    maxGain = infoGain
                    splitVariable = d
                    splitValue = value
                    splitSat = np.argmax(count1)
                    splitNot = np.argmax(count0)
    
        if infoGain > 0: # if there's an actual split. rather than everything going to one side. there are other ways of checking this condition...
            self.splitVariable = splitVariable
            self.splitValue = splitValue
            self.splitSat = splitSat
            self.splitNot = splitNot

    def export(self):
        model = {}
        if self.splitVariable is not None:
            model['d'] = int(self.splitVariable)
        if self.splitValue is not None:
            model['x'] = int(self.splitValue)
        if self.splitSat is not None:
            model['s'] = int(self.splitSat)
        if self.splitNot is not None:
            model['n'] = int(self.splitNot)
        return model

    def load(self, model):
        if 'd' in model:
            self.splitVariable = np.int64(model['d'])
        else: 
            self.splitVariable = None
        if 'x' in model:
            self.splitValue = np.int64(model['x'])
        else: 
            self.splitValue = None
        if 's' in model:
            self.splitSat = np.int64(model['s'])
        else:
            self.splitSat = None
        if 'n' in model:
            self.splitNot = np.int64(model['n'])
        else:
            self.splitNot = None