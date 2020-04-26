# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# our code
import utils
from dataset import FlowPolicy, Dataset, DatasetInfo

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

def analyze(model, dataset):
    y_pred = model.predict(dataset.flows.values)
    n_wrong_normal = 0
    n_wrong_attack = 0
    for i in range(dataset.classes_bin.size):
        if y_pred[i] != dataset.classes_bin[i]: 
            if  dataset.classes_bin[i] == 0:
                n_wrong_normal += 1
            else:
                n_wrong_attack += 1
    error = np.mean(y_pred != dataset.classes_bin)
    return error, y_pred, n_wrong_normal, n_wrong_attack

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--model', required=True)
    parser.add_argument('-v','--version', required=False)
    io_args = parser.parse_args()
    model_type = io_args.model
    version = io_args.version

    raw_data = utils.load_csv('AppDDos.csv')
    # all classes: 'srcip','srcport','dstip','dstport','proto','total_fpackets','total_fvolume','total_bpackets','total_bvolume','min_fpktl','mean_fpktl','max_fpktl','std_fpktl','min_bpktl','mean_bpktl','max_bpktl','std_bpktl','min_fiat','mean_fiat','max_fiat','std_fiat','min_biat','mean_biat','max_biat','std_biat','duration','min_active','mean_active','max_active','std_active','min_idle','mean_idle','max_idle','std_idle','sflow_fpackets','sflow_fbytes','sflow_bpackets','sflow_bbytes','fpsh_cnt','bpsh_cnt','furg_cnt','burg_cnt','total_fhlen','total_bhlen','dscp','firstTime','flast','blast','class'
    ignore_features = [
        'srcip',
        'srcport',
        'dstip',
        'dstport',
        'proto',
        'total_fpackets',
        'total_fvolume',
        'total_bpackets',
        'total_bvolume',
        # 'min_fpktl',
        # 'mean_fpktl',
        # 'max_fpktl',
        # 'std_fpktl',
        # 'min_bpktl',
        # 'mean_bpktl',
        # 'max_bpktl',
        # 'std_bpktl',
        # 'min_fiat',
        # 'mean_fiat',
        # 'max_fiat',
        # 'std_fiat',
        # 'min_biat',
        # 'mean_biat',
        # 'max_biat',
        # 'std_biat',
        # 'duration',
        # 'min_active',
        # 'mean_active',
        # 'max_active',
        'std_active',
        'min_idle',
        'mean_idle',
        'max_idle',
        'std_idle',
        'sflow_fpackets',
        'sflow_fbytes',
        'sflow_bpackets',
        'sflow_bbytes',
        'fpsh_cnt',
        # 'bpsh_cnt',
        'furg_cnt',
        'burg_cnt',
        'total_fhlen',
        'total_bhlen',
        'dscp',
        'firstTime',
        'flast',
        'blast',
        'class'
    ]

    # Prepare data
    y = raw_data['class']
    X = raw_data.drop(columns=ignore_features)
    dataset = Dataset(X,y,'http_ddos')

    print('Running: %s' % model_type)
    if model_type == 'random_forest_bin':
        # dataset.crop(5000)
        dataset.matching(FlowPolicy(['rudy','slowread','slowloris']))
        training_set, validation_set, _ = dataset.generate_sets()
        training_set.binarize()
        validation_set.binarize()

        model = RandomForest(np.inf,50)
        model.fit(training_set.flows.values,training_set.classes_bin)

        training_error, training_y_pred, training_n_wrong_normal, training_n_wrong_attack = analyze(model, training_set)
        training_info = training_set.analyze()
        print('    Training error: %.3f' % training_error)
        print('    Training relative attack error: %.3f' % (training_n_wrong_attack/training_info.attack_n))
        print('    Training relative normal error: %.3f' % (training_n_wrong_normal/training_info.normal_n))

        validation_error, validation_y_pred, validation_n_wrong_normal, validation_n_wrong_attack = analyze(model, validation_set)
        validation_info = validation_set.analyze()
        print('    Validation error: %.3f' % validation_error)
        print('    Validation relative attack error: %.3f' % (validation_n_wrong_attack/validation_info.attack_n))
        print('    Validation relative normal error: %.3f' % (validation_n_wrong_normal/validation_info.normal_n))
        
        print('    Exporting model')
        path = '../data/%s.json' % model_type
        utils.export_model(path, model)
        print('    Model exported in file: %s' % path)
    elif model_type == 'load_random_forest_bin':
        # total attack: 301
        # wrong attack: 276
        # total normal: 4699
        # wrong normal: 32
        # Total training error: 0.062
        # Attack training relative error: 0.917
        # Normal training relative error: 0.007
        # slowhttptest -u http://10.0.0.1 -c 5000 -i 10 -p 5 -r 100
        
        # dataset.crop(5000)
        dataset.matching(FlowPolicy(['rudy','slowread','slowloris']))
        dataset.binarize()
        
        model = RandomForest(np.inf,50)
        if version == None:
            model.load('../data/random_forest_bin.json')
        else:
            model.load('../data/random_forest_bin_%s.json' % version)

        testing_error, testing_y_pred, testing_n_wrong_normal, testing_n_wrong_attack = analyze(model, dataset)
        testing_info = dataset.analyze()
        print('    Testing error: %.3f' % testing_error)
        print('    Testing relative attack error: %.3f' % (testing_n_wrong_attack/testing_info.attack_n))
        print('    Testing relative normal error: %.3f' % (testing_n_wrong_normal/testing_info.normal_n))
    elif model_type == 'train_slowloris':
        pass
    elif model_type == 'analyze_dataset':
        # X = X.head(n=5000)
        # y = y.head(n=5000)
        pass
    else:
        print('Unknown model: %s' % model_type)
