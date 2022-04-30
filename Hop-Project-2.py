# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 15:28:22 2022

@author: tahat
"""

import matplotlib.pyplot as plt
import numpy as np
from neurodynex.hopfield_network import pattern_tools, plot_tools, network

def random_pattern(n,a):
    return pattern_tools.PatternFactory(n).create_random_pattern(a)

def weight_matrix(m,n,a,b,patterns):
    w=np.zeros(n,n)
    cprime=1/a/(1-a)/n
    for i in range(m):
        patterns=random_pattern(n,a)
        w=w+np.matmul(patterns-a,patterns-b)*cprime
    return w

def hamming_distance(x,y):
    return (sum(x)+sum(y)-np.dot(x,y))/np.size(x)

def map_function(theta,w,sigma_t0):
    sigma_t1=np.zeros(np.size(sigma_t0))
    for i in range(np.size(sigma_t0)):
        sigma_t1[i]=0.5*(1+np.sign(sum(np.dot(w[i],sigma_t0)-theta)))
    return sigma_t1

    
pattern_size = 5
nr_neurons=300
# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons= nr_neurons)


    
    
