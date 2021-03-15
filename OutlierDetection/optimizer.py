#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:33:04 2021

@author: alysonweidmann
"""


import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from matplotlib.ticker import MaxNLocator
from itertools import chain, count

"""Search optimization algorithms to solve the EntropyProblem for
   outlier detection using entropy minimization"""

class BaseOptimizer(metaclass=ABCMeta):
    """Abstract base class for Optimizer. This class is not meant to be
       instaniated directly"""
    
    def __init__(self):
        self.outliers = {}
   
    
    @abstractmethod
    def min_solve(self, problem):
        """Abstract method to minimize the information entropy in a dataset
        
        Parameters
        ----------
        problem : EntropyProblem or related Problem
            Optimization problem instance. Contains input data and calculates
            information entropy as stored in a property called 'base'. Must
            also contain a 'get_successors()' method, which generates all of
            the successors from the neighborhood of the initial/current state

        Returns
        -------
        problem : Problem
            Solution state of optimization problem. Contains resulting
            data set with outliers removed as an attribute
        
        """
        pass
    
    
    @abstractmethod
    def max_solve(self, problem):
        """Abstract method to maxmiize the information entropy in a dataset
        
        Parameters
        ----------
        problem : EntropyProblem or related Problem
            Optimization problem instance. Contains input data and calculates
            information entropy as stored in a property called 'base'. Must
            also contain a 'get_successors()' method, which generates all of
            the successors from the neighborhood of the initial/current state

        Returns
        -------
        problem : Problem
            Solution state of optimization problem. Contains resulting
            data set with outliers removed as an attribute
        """
        pass
    
    
    def _check_is_solved(self):
        """Checks if min_solve or max_solve have been called"""
        
        if any(self.outliers) == False:
            raise Exception('No outliers detected! Has solver been fit?')
            
        return self
    
    
    def plot_iter(self):
        """Generates a plot of data entropy following each iteration"""
        
        self._check_is_solved()
        
        max_iter = len(self.outliers.values())
        
        ax = plt.figure().gca()
        sns.lineplot(x=range(1, max_iter+1),
                     y=list(self.outliers.values()),
                     drawstyle='steps-pre',
                     ax=ax
                    )
        plt.xlabel('Iterations')
        plt.ylabel('Entropy')
        
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.show()
        
        

class HillClimber(BaseOptimizer):
    """
    Uses Greedy Hill Climbing algorithm to find a local entropy
    minimum/maximum of a dataset. 
       
    Parameters:
    ----------
    epochs : int
        Max number of iterations before terminating
        
    Attributes:
    -----------
    outliers : dict
        Dictionary of outliers removed with each iteration. Stores data
        value as the key and the entropy of the resulting dataset as the value
    
    """
    
    def __init__(self, epochs=20):
        self.epochs = epochs
        
        
    def min_solve(self, problem):
        """
        Optimize to minimize information entropy in a Problem        
        """
        
        self.outliers = {}
        
        for _ in range(self.epochs):
            neighbor = min(problem.get_successors(), key=lambda x: x.base)
            
            if neighbor.base > problem.base: break
            problem = neighbor
            
            self.outliers[problem.name] = problem.base
            
        return problem
    
    
    def max_solve(self, problem):
        """
        Optimize to maximize information entropy in a Problem
        """
        
        
        for _ in range(self.epochs):
            neighbor = min(problem.get_successors(), key=lambda x: x.base)
            
            if neighbor.base < problem.base: break
            problem = neighbor
            
            self.outliers[problem.name] = problem.base
            
        return problem
    
    
class LocalBeamSearch(BaseOptimizer):
    """Identical to HillClimber, except begins with k randomly generated
       states and selects k best successors from list of neighbors over
       all k states
       
    Parameters:
    ----------
    
    epochs : int
        Max number of iterations before terminating
        
    beam_width : int
        Number of samples to maintain during search
       
    """
       
    def __init__(self, epochs=20, beam_width=5):
        self.epochs = epochs
        self.beam_width = beam_width
        
    
    def min_solve(self, problem):
        
        self.outliers = {}
        
        beam = [problem]
        for t in range(self.epochs):
            
            # union all neighbors
            neighborhood = chain(*(n.get_successors() for n in beam))
            
            beam = sorted(neighborhood,
                          key=lambda x: x.base,
                          reverse=True)[-self.beam_width:]
            
            if all([node.base > problem.base for node in beam]): break
            
            self.outliers[beam[-1].name] = beam[-1].base
            
        return beam[-1]
    
    
    def max_solve(self, problem):
        
        self.outliers = {}
        
        beam = [problem]
        for t in range(self.epochs):
            
            # union all neighbors
            neighborhood = chain(*(n.get_successors for n in beam))
            
            beam = sorted(neighborhood, key=lambda x: x.base)[-self.beam_width:]
            
            if all([node.base < problem.base for node in beam]): break
            
            self.outliers[beam[-1].name] = beam[-1].base
            
        return beam[-1]