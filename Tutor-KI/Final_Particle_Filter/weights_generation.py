# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 22:04:50 2020

@author: riema
"""
import numpy as np

for i in range(10):
    weights = np.random.uniform(size=(100, 1))
    weights /= weights.sum()
    np.save('./data/weights/weights_%s.npy' %i, weights)