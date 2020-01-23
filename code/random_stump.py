import numpy as np
import utils
from decision_stump import DecisionStumpInfoGain


class RandomStumpInfoGain(DecisionStumpInfoGain):
    
    def fit(self, X, y):
        
        # Randomly select k features.
        # This can be done by randomly permuting
        # the feature indices and taking the first k
        D = X.shape[1]
        k = int(np.floor(np.sqrt(D)))
        
        chosen_features = np.random.choice(D, k, replace=False)
                
        DecisionStumpInfoGain.fit(self, X, y, split_features=chosen_features)
