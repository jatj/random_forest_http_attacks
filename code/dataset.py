# basics
import logging
import numpy as np
import math
import pandas as pd

class FlowPolicy(object):
    '''
    Helper class to filter flows according to a policy.
    TODO(abrahamtorres): support filtering by ip and direction (src, dst)
    TODO(abrahamtorres): support filtering by mac and direction (src, dst)
    '''
    # Available attacks in dataset
    supported_attacks = ['slowbody2','slowread','ddossim','goldeneye','slowheaders','rudy','hulk','slowloris']
    # Class to be considered as 'normal' flow
    normal_class = 'normal'
    # Available classes in dataset
    supported_classes = ['slowbody2','slowread','ddossim','goldeneye','slowheaders','rudy','hulk','slowloris','normal']

    def __init__(self, attacks):
        self.logger = logging.getLogger('flow_policy')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.WARNING)
        self.logger.addHandler(self.console_handler)
        self.attacks = []
        for attack in attacks:
            if attack in self.supported_attacks:
                self.attacks.append(attack)
            else:
                self.logger.warn('Ignoring class %s' % attack)

    def isValid(self, flow, class_):
        if(class_ in self.attacks):
            return True
        else:
            self.logger.debug('Invalid flow with class %s', class_)

class DatasetInfo(object):
    '''
    Holds important info of a dataset

    Attributes:
    ----------
        attack_n: total number of attack flows
        attack_pct: percentage of attack flows
        normal_n: total number of normal flows
        normal_pct: percentage of normal flows
        size: total size of dataset
        classes_n: dictionary from class label to number of attack flows with that class
        classes_pct: dictionary from class label to percentage of attack flows with that class
        attacks_r_pct: dictionary from attack label to percentage of attack flows with that class relative to the total number of attacks
    '''

    def __init__(self, flows, classes, name: str):
        self.name = name

        # Initialize count dictionary
        self.classes_n = {}
        for class_ in FlowPolicy.supported_classes:
            self.classes_n[class_] = 0

        # Extract counts
        for i in range(len(classes.keys())):
            self.classes_n[classes[classes.keys()[i]]]+=1

        # Initialize counters
        self.size = len(classes.values)
        self.attack_n = 0;
        for class_ in FlowPolicy.supported_attacks:
            self.attack_n += self.classes_n[class_]
        self.attack_pct = self.attack_n/self.size
        self.normal_n = self.classes_n[FlowPolicy.normal_class];
        self.normal_pct = self.normal_n/self.size

        # Initialize percentage dictionary
        self.classes_pct = {}
        for class_ in FlowPolicy.supported_classes:
            if self.size != 0:
                self.classes_pct[class_] = self.classes_n[class_]/self.size
        self.attacks_r_pct = {}
        for class_ in FlowPolicy.supported_attacks:
            if self.attack_n != 0:
                self.attacks_r_pct[class_] = self.classes_n[class_]/self.attack_n

class Dataset(object):
    '''
    Class to hold and transform the dataset.

    Methods:
        matching: filtering flows with policy
        generate_sets: create separate training, validation and verification sets
        binarize: transforms the dataset classes into binary class 0=normal 1=attack
        analyze: returns statistics of the flows contained in the dataset
    '''

    def __init__(self, flows, classes, name='default_dataset'):
        self.flows = flows
        self.classes = classes
        self.classes_bin = None
        self.name = name
        self.logger = logging.getLogger('flow_policy')
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.WARNING)
        self.logger.addHandler(self.console_handler)

    def matching(self, policy: FlowPolicy):
        ''' 
        Will filter the flows and only keep those who matched the provided policy
        Parameters
        ----------
        policy : FlowPolicy
            The policy to validate against each flow to be considered
        Returns
        ----------
        filtered The number of filtered flows
        '''
        filtered = 0
        n = len(self.classes.values)
        for i in range(n):
            if not policy.isValid(self.flows.values[i], self.classes.values[i]):
                self.flows.drop(i)
                self.classes.drop(i)
                filtered+=1
        return filtered

    def generate_sets(self, training_size=0.7, validation_size=0.3, verification_size=None, keep_proportions=True):
        ''' 
        Will split the current data in to at least two separate datasets (training and validation).
        A third dataset can be created for verification if [verification_size] is set.
        Parameters
        ----------
        training_size : Number
            The size in percentage of the training set
        validation_size : Number
            The size in percentage of the validation set
        verification_size : Number
            The size in percentage of the verification set
        keep_proportions : Boolean
            Whether to keep proportions between normal and attack flows in each set or no, by randomly
            accross all flows in the dataset.
            e.g. if the original dataset contains 80% normal and 20% attack flows. And the parameters are
            training_size=0.7, validation_size=0.3, verification_size=None, keep_proportions=True
            the resulting training set will have 80% x 70% normal flows and 20% x 70% attack flows.
        '''
        # TODO(abrahamtorres): validate sizes add to 1
        
        # Generate sets
        training_set = self.generate_set(training_size, '%s_training_set' % self.name, keep_proportions)
        validation_set = self.generate_set(validation_size, '%s_validation_set' % self.name, keep_proportions)
        verification_set = None
        if verification_size is not None:
            verification_set = self.generate_set(verification_size, '%s_verification_set' % self.name, keep_proportions)

        return training_set, validation_set, verification_set

    def generate_set(self, size, name, keep_proportions=True):
        # TODO(abrahamtorres): document
        if keep_proportions:
            # TODO(abrahamtorres): prevent repetitions between different sets
            available_attack_indexes = []
            available_normal_indexes = []
            for i in range(len(self.classes.values)):
                if self.classes[i] == FlowPolicy.normal_class:
                    available_normal_indexes.append(i)
                else:
                    available_attack_indexes.append(i)

            # Add all attack flows
            attack_flows = self.flows.iloc[available_attack_indexes].sample(frac=size)
            attack_classes = self.classes.iloc[available_attack_indexes].sample(frac=size)

            # Add all normal flows
            normal_flows = self.flows.iloc[available_normal_indexes].sample(frac=size)
            normal_classes = self.classes.iloc[available_normal_indexes].sample(frac=size)

            flows = pd.concat([attack_flows, normal_flows])
            classes = pd.concat([attack_classes, normal_classes])
            return Dataset(flows, classes, name)
        else:
            flows = self.flows.sample(frac=size)
            classes = self.classes.sample(frac=size)
            return Dataset(flows, classes, name)

    def binarize(self, normalValue=0, attackValue=1):
        '''
        Will transform the dataset classes to a binary value, grouping all attacks into same class.
        Parameters
        ___________
            normalValue: the value for the normal class default=0
            attackValue: the value for the attack class default=1
        '''
        self.classes_bin = np.array([], dtype=np.dtype(int))
        for i in range(len(self.classes.keys())):
            if self.classes[self.classes.keys()[i]] == FlowPolicy.normal_class:
                self.classes_bin = np.append(self.classes_bin, [normalValue])
            else:
                self.classes_bin = np.append(self.classes_bin, [attackValue])
        return self.classes_bin

    def analyze(self):
        '''
        Will gather analyze and return the dataset info
        Returns
        ----------
        DatasetInfo containing important metrics of the dataset
        '''
        return DatasetInfo(self.flows, self.classes, self.name)

    def crop(self, n):
        '''
        Reduce the size of the dataset to the first n samples
        Parameters
        ___________
            n: desired size of the dataset
        '''
        self.flows = self.flows.head(n)
        self.classes = self.classes.head(n)