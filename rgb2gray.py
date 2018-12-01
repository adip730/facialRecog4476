# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:11:17 2018

@author: Luis
"""

import numpy as np

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])