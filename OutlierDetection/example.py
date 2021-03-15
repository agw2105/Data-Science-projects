#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 17:29:22 2021

@author: alysonweidmann
"""


import numpy as np
from OutlierDetection.ent import EntropyProblem
from OutlierDetection.optimizer import HillClimber


data = [1,1,1,1,0,0,0,2,2,1,1,1,0,0,2,2,1]
# convert to numpy array
data = np.array(data)

ep = EntropyProblem(data=data,
                    size=5)

solver = HillClimber(epochs=5)

solved_ep = solver.min_solve(ep)

# print results
print(solver.outliers)
print(solved_ep.data)

# Plot entropy data over iterations
solver.plot_iter()