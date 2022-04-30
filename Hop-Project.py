# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:41:45 2022

@author: tahat
"""
import matplotlib.pyplot as plt
import numpy as np
from neurodynex3.hopfield_network import pattern_tools, plot_tools, network

pattern_size = 300

# create an instance of the class HopfieldNetwork
hopfield_net = network.HopfieldNetwork(nr_neurons= pattern_size)