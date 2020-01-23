# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# our code
import utils

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

def load_csv(filename):
    return pd.read_csv(os.path.join('..','data',filename)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', required=True)

    io_args = parser.parse_args()
    model = io_args.model
    raw_data = load_csv('AppDDos.csv')
    ignore_features = ['srcip','srcport','dstip','dstport','proto','dscp','firstTime','flast','blast','class']

    # Prepare data
    y = raw_data['class']
    X = raw_data.drop(columns=ignore_features)

    if model == 'random_forest_bin':
        # X = X.head(n=5000)
        # y = y.head(n=5000)
        # Change all non normal class to a general class attack
        y_bin = np.array([], dtype=np.dtype(int))

        # Use class index instead of label
        n_total_normal = 0
        n_total_attack = 0
        for i in range(len(y.values)):
            if y[i] == 'normal':
                y_bin = np.append(y_bin, [0])
                n_total_normal += 1
            else:
                y_bin = np.append(y_bin, [1])
                n_total_attack += 1
                
        classes = list(set(y_bin))
        features = X.columns.values

        model = RandomForest(np.inf,50)
        model.fit(X.values,y_bin)
        y_pred = model.predict(X.values)
        n_wrong_normal = 0
        n_wrong_attack = 0
        for i in range(y_bin.size):
            if y_pred[i] != y_bin[i]: 
                if  y_bin[i] == 0:
                    n_wrong_normal += 1
                else:
                    n_wrong_attack += 1

        print("    # total attack: %d" % n_total_attack)
        print("    # wrong attack: %d" % n_wrong_attack)
        print("    # total normal: %d" % n_total_normal)
        print("    # wrong normal: %d" % n_wrong_normal)
        tr_error = np.mean(y_pred != y_bin)
        print("    Training error: %.3f" % tr_error)
        
        print("    Exporting model")
        file = open("../data/random_forest_bin.json", "w")
        file.write(model.export())
        file.close()
        print("    Model exported in file: ../data/random_forest_bin.json")
    elif model == 'load_random_forest_bin':
        X = X.head(n=5000)
        y = y.head(n=5000)
        # Change all non normal class to a general class attack
        y_bin = np.array([], dtype=np.dtype(int))

        # Use class index instead of label
        n_total_normal = 0
        n_total_attack = 0
        for i in range(len(y.values)):
            if y[i] == 'normal':
                y_bin = np.append(y_bin, [0])
                n_total_normal += 1
            else:
                y_bin = np.append(y_bin, [1])
                n_total_attack += 1
                
        classes = list(set(y_bin))
        features = X.columns.values

        model = RandomForest(np.inf,50)
        model.load("../data/random_forest_bin.json")
        y_pred = model.predict(X.values)
        n_wrong_normal = 0
        n_wrong_attack = 0
        for i in range(y_bin.size):
            if y_pred[i] != y_bin[i]: 
                if  y_bin[i] == 0:
                    n_wrong_normal += 1
                else:
                    n_wrong_attack += 1
            # elif y_bin[i] == 1:
            #     print("True attack: ")
            #     print(X.iloc[i, :])
            #     exit

        print("    # total attack: %d" % n_total_attack)
        print("    # wrong attack: %d" % n_wrong_attack)
        print("    # total normal: %d" % n_total_normal)
        print("    # wrong normal: %d" % n_wrong_normal)
        tr_error = np.mean(y_pred != y_bin)
        print("    Training error: %.3f" % tr_error)
        
    else:
        print('Unknown model: %s' % model)
