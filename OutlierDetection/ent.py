#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:44:24 2021

@author: alysonweidmann
"""


import random
import numpy as np
from math import log, e
import pandas as pd


class EntropyProblem:
    """
    Calculate the information entropy for discrete 1D data. Works with
    categorical or numerical datasets and takes array or dataframe as 
    input
    
    Parameters:
    ----------
    data : array-like
        Input data to calculate information entropy
        
    size : int
        Number of neighbors to explore
        
    
    Attributes:
    ----------
    name : str, float, or int
        Identifier for each neighbor successor. Set as None when
        instantiating the initial problem. Is set automatically when
        getting successors
        
    initial_state : int
        Random seed to initialize successor selection from neighbors
    
    base : float
        Information entropy of input data
        
    """
    
    def __init__(self, data, size=20, name=None):
        self.data = data
        self.size = size
        self.initial_state= np.random.choice(len(data))
        self.name = name
        self.__base = None
        
        
    @property
    def base(self):
        """Calculate information entropy of dataset in base e and set as 
           attribute"""
        
        self.__base = self.calc_entropy()
        return self.__base
    
    
    def calc_entropy(self):
        """Calculate information entropy of dataset in base e"""
        n_labels = len(self.data)
        
        if n_labels <= 1:
            return 0
        
        value, counts = np.unique(self.data, return_counts=True)
        probs = counts/n_labels
        n_classes = np.count_nonzero(probs)
        
        if n_classes <= 1:
            return 0
        
        ent = 0.
        
        for i in probs:
            ent -= i * log(i, e)
        
        return ent
    
    
    def neighbors(self):
        """Gets range of neighbors surrounding initial state to create
           successors"""
        
        
        if self.size > len(self.data):
            raise Exception('Size is too larget for data set.'
                            'Set a value <= {}'.format(len(self.data)))
        
        # Set initial bounds of successor set
        min_neighbor = self.initial_state - self.size
        max_neighbor = self.initial_state + self.size
        
        # Ensure successor bounds fall within range of the actual data
        if min_neighbor < 0:
            min_value = 0
        else:
            min_value = min_neighbor
            
        if max_neighbor > len(self.data):
            max_value = len(self.data)
        else:
            max_value = max_neighbor
            
        return range(min_value, max_value)
    
    
    def get_successors(self):
        """
        Generates set of successor EntropyProblem instances from the 
        neighborhood of the initial state, each with a unique neighbor 
        designated as an outlier and removed. The new information entropy is 
        recalculated as a property in each successor instance.

        Returns
        -------
        Generator object of EntropyProblem successors

        """
    
        exists = set()
        
        for i in self.neighbors():
            if isinstance(self.data, (pd.DataFrame, pd.Series)):
                new_data = self.data.drop(index=i)
                new_data = new_data.reset_index(drop=True)
                value = self.data.loc[self.data.index==i].values.item()
                
            elif isinstance(self.data, np.ndarray):
                new_data = np.delete(self.data, i)
                value = self.data[i]
                
            if self.size <= len(new_data):
                size = self.size
            else:
                size = len(new_data)
                
            if value not in exists:
                yield EntropyProblem(new_data, size=size, name=value)
                exists.add(value)
                
                
    def select_successor(self):
        """Returns a single successor at random from neighbors"""
        
        all_successors = self.get_successors()
        successor = np.random.choice(list(all_successors))
        
        return successor